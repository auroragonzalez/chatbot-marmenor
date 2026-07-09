import os
import shutil
import re
import logging
from langchain_community.document_loaders import PyPDFLoader, PyPDFDirectoryLoader
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_huggingface import HuggingFaceEmbeddings
from langchain_chroma import Chroma
from langchain_ollama import OllamaLLM
from langchain_core.prompts import ChatPromptTemplate
from langchain_core.documents import Document

logging.getLogger("pypdf").setLevel(logging.ERROR)
logging.getLogger("pdfminer").setLevel(logging.ERROR)
BASE_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))

UMBRAL_SIMILITUD = 0.175
MODELO_EMBEDDINGS = "intfloat/multilingual-e5-small"
MODELO_LLM = os.environ.get("MODELO_LLM", "llama3")
CHROMA_PATH = os.path.join(BASE_DIR, "chroma_db")
K_RELEVANTES = 5

class GestorRAG:
    
    def __init__(self, db_path=CHROMA_PATH, modelo_embeddings=MODELO_EMBEDDINGS):
        self.db_path = db_path
        self.modelo_embeddings = modelo_embeddings
        
        print("Inicializando motor...")
        
        self.embedding_function = HuggingFaceEmbeddings(
            model_name=self.modelo_embeddings,
            model_kwargs={'device': 'cpu'},
            encode_kwargs={'normalize_embeddings': True}
        )
        

        self.db = Chroma(
            persist_directory=self.db_path,
            embedding_function=self.embedding_function,
            collection_metadata={"hnsw:space": "cosine"}
        )
        print(f" Base de datos vectorial lista en: {self.db_path}")

    def es_chunk_bibliografia(self, texto):
        """Detecta si un chunk es mayoritariamente bibliografía mediante densidad de patrones."""
        lineas = [l.strip() for l in texto.split('\n') if l.strip()]
        if not lineas:
            return False
            
        patron = re.compile(r'.{10,}\(\d{4}\)|\bDOI\b|https?://', re.IGNORECASE)
        matches = sum(1 for l in lineas if patron.search(l))
        
        #Si más de la mitad de las líneas parecen referencias, descartamos el chunk
        return (matches / len(lineas)) > 0.5

    def cargar_directorio(self, ruta_directorio):
        """Carga los PDFs de una carpeta."""
        if not os.path.exists(ruta_directorio) or not os.path.isdir(ruta_directorio):
            raise FileNotFoundError(f"La carpeta no existe o no es un directorio: {ruta_directorio}")
        
        print(f"Cargando todos los PDFs desde: {ruta_directorio}...")
        loader = PyPDFDirectoryLoader(ruta_directorio)
        documentos = loader.load()
        print(f"Se han cargado {len(documentos)} páginas en total.")
        return documentos

    def dividir_texto(self, documentos):
        """Divide las páginas en fragmentos (chunks) más pequeños."""
        text_splitter = RecursiveCharacterTextSplitter(
            chunk_size=1000,      # Caracteres por fragmento
            chunk_overlap=200,    # Solapamiento para mantener contexto
            length_function=len,
            is_separator_regex=False,
        )
        return text_splitter.split_documents(documentos)

    def calcular_ids_chunks(self, chunks):
        """
        Asigna un ID único: 'ruta/archivo:pagina:indice'.
        Esto permite evitar duplicados si procesamos el mismo archivo varias veces.
        """
        ultimo_id_pagina = None
        indice_chunk_actual = 0

        for chunk in chunks:
            fuente = chunk.metadata.get("source")
            pagina = chunk.metadata.get("page") + 1 # Las páginas empiezan en 0, pero para el usuario es más natural empezar en 1
            id_pagina_actual = f"{fuente}:{pagina}"

            # Si seguimos en la misma página, incrementamos índice
            if id_pagina_actual == ultimo_id_pagina:
                indice_chunk_actual += 1
            else:
                indice_chunk_actual = 0

            ultimo_id_pagina = id_pagina_actual
            
            # Asignamos el ID único al metadata
            chunk_id = f"{id_pagina_actual}:{indice_chunk_actual}"
            chunk.metadata["id"] = chunk_id
            
        return chunks

    def agregar_a_chroma(self, chunks_con_ids):
        """
        Compara los IDs nuevos con los existentes y guarda SOLO los nuevos.
        """
        # Obtener IDs que ya existen en la base de datos
        items_existentes = self.db.get(include=[])  # solo trae IDs
        # Lo pasamos a un conjunto para que las búsquedas sean O(1) en vez de O(n) (listas)
        ids_existentes = set(items_existentes["ids"])
        
        print(f"Documentos existentes en DB: {len(ids_existentes)}")

        # Buscamos los chunks nuevos (aquellos cuyo ID no está en la base de datos)
        chunks_nuevos = []
        for chunk in chunks_con_ids:
            if chunk.metadata["id"] not in ids_existentes:
                if self.es_chunk_bibliografia(chunk.page_content):
                    continue
                if not chunk.page_content.startswith("passage: "):
                    chunk.page_content = "passage: " + chunk.page_content
                chunks_nuevos.append(chunk)

        if len(chunks_nuevos) > 0:
            print(f"Añadiendo {len(chunks_nuevos)} nuevos fragmentos...")
            new_chunk_ids = [chunk.metadata["id"] for chunk in chunks_nuevos]
            self.db.add_documents(chunks_nuevos, ids=new_chunk_ids)
            print("Guardado completado.")
        else:
            print("No hay documentos nuevos.")

    def ingestar(self, ruta_pdf):
        """Método público para procesar un PDF completo."""
        try:
            documentos = self.cargar_directorio(ruta_pdf)
            chunks = self.dividir_texto(documentos)
            chunks_con_ids = self.calcular_ids_chunks(chunks)
            self.agregar_a_chroma(chunks_con_ids)
        except Exception as e:
            print(f"Error crítico en la ingesta: {e}")


    #Método estático que usaremos para extraer el número de página del ID.
    @staticmethod
    def extraer_pagina(id_string):
        try:
            #Sacamos el número que está justo antes del último ':'
            return int(id_string.split(':')[-2])
        except:
            return 0
        
    def consultar(self, query, k_relevantes=K_RELEVANTES, umbral_similitud=UMBRAL_SIMILITUD):

        print(f"\nPensando: '{query}'...")

        query_ = "query: " + query if not query.startswith("query: ") else query
        
        # OVER-FETCHING: Capturamos el doble por si hay duplicados o bibliografía
        k_busqueda = k_relevantes * 2
        resultados = self.db.similarity_search_with_score(query_, k=k_busqueda)

        # Buscamos los que estén repetidos
        textos_vistos = set()
        resultados_unicos = []

        for doc, score in resultados:
            if score < umbral_similitud:
                texto_limpio = doc.page_content.strip()

                if texto_limpio not in textos_vistos:
                    textos_vistos.add(texto_limpio)
                    resultados_unicos.append((doc, score))

                    if len(resultados_unicos) == k_relevantes:
                        break

        if not resultados_unicos:
            return "No se encontró información relevante en la base de datos."

        contexto_texto = "\n\n---\n\n".join([doc.page_content for doc, _ in resultados_unicos])
        
        fuentes = list(set([doc.metadata.get("id", "Desconocido") for doc, _ in resultados_unicos]))
        fuentes_ids_ordenadas = sorted(fuentes, key=GestorRAG.extraer_pagina)

        return {
            "contexto": contexto_texto,
            "fuentes": fuentes_ids_ordenadas
        }

    def limpiar_db(self):
        if os.path.exists(self.db_path):
            shutil.rmtree(self.db_path)
            print("Base de datos eliminada.")

    @staticmethod
    def mensaje_llm(query, contexto_texto):

        template = """
        Eres un asistente experto en investigación científica.
        Usa el siguiente contexto para responder a la pregunta del usuario.
        
        Si la respuesta no está en el contexto, di "No tengo información suficiente".
        Responde siempre en español.

        CONTEXTO:
        {context}

        ---

        PREGUNTA: {question}
        """
        
        prompt_template = ChatPromptTemplate.from_template(template)
        prompt = prompt_template.format(context=contexto_texto, question=query)
        
        # Inferencia

        modelo = OllamaLLM(model=MODELO_LLM, temperature=0.0)
        return modelo.invoke(prompt)

if __name__ == "__main__":
    #Inicializamos la clase RAG
    rag = GestorRAG()
    
    ruta_pdf_prueba = os.path.join(BASE_DIR, "datos", "Docs")
    
    # Se ingesta el PDF 
    rag.ingestar(ruta_pdf_prueba)
    
    while True:
        pregunta = input("\n¿Qué quieres saber del documento? (o 'salir'): ")
        
        if pregunta.lower() in ['salir', 'exit', 'q']:
            print("Cerrando el sistema RAG de prueba...")
            break
        
        if not pregunta.strip(): continue

        try:

            resultado_rag = rag.consultar(pregunta)
            
            print("\n" + "-"*40)
            
            if isinstance(resultado_rag, str):
                print(f" ASISTENTE:\n{resultado_rag}")

            else:

                respuesta_texto = rag.mensaje_llm(pregunta, resultado_rag["contexto"])
                
                print(f" ASISTENTE:\n{respuesta_texto}")
                print("\nFuentes:")
                for fuente in resultado_rag["fuentes"]:
                    print(f"- {fuente}")
                    
            print("-" * 40)
            
        except Exception as e:
            print(f" Ocurrió un error: {e}")