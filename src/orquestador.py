import os
import pandas as pd
from langchain_ollama import OllamaLLM
from generadorSQL import ConsultaSQL
from RAG import GestorRAG
from clasificadorIntencion import ClasificadorIntencion
from reintentos import invocar_con_reintentos, es_error_transitorio

MENSAJE_ERROR_CONEXION = (
    "⚠️ No se pudo contactar con el modelo de lenguaje (servicio en la nube de "
    "Ollama) por un problema de red temporal. Vuelve a intentarlo en unos segundos."
)

BASE_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
MODELO_LLM = os.environ.get("MODELO_LLM", "llama3")

K_RELEVANTES_RAG = 5
UMBRAL_RAG = 0.175
LIMITE_FILAS = 1000
LIMITE_INTERPRETACION = 15
PROMPTS_POR_TIPO = {
    "AGREGACION": (
        "La tabla contiene un único valor agregado (máximo, mínimo, media, conteo...).\n"
        "Responde en una o dos frases directas mencionando el valor, la fecha/hora y SIEMPRE el lugar, estación o boya donde se registró (si aparece en la tabla)."
    ),
    "SERIE_TEMPORAL": (
        "Muestra la tabla en formato markdown.\n"
        "Describe brevemente la evolución temporal que muestran los datos."
    ),
    "TABLA_PEQUEÑA": (
        "Responde de forma directa, natural y profesional.\n"
        "Integra los valores de la tabla en tu respuesta o muéstralos como una pequeña lista si es más claro."
    )
}
class Orquestador:
    def __init__(self, ruta_csv, ruta_directorio_pdfs, modelo=MODELO_LLM):
        
        #DuckDB 
        if not os.path.exists(ruta_csv):
            raise FileNotFoundError(f"No encuentro el CSV: {ruta_csv}")
        self.sql_engine = ConsultaSQL(ruta_csv)
        print("Motor SQL: Listo (DuckDB conectado)")

        # Clasificador de intención
        self.clasificador_intencion = ClasificadorIntencion()

        # RAG
        self.rag_engine = GestorRAG()
        self.rag_engine.ingestar(ruta_directorio_pdfs)
        print("Motor RAG: Listo (Base de datos vectorial cargada)")
        self.llm_generador = OllamaLLM(model=MODELO_LLM, temperature=0.0, num_ctx=8192)

    def clasificar_pregunta(self, pregunta):
        tipo, scores = self.clasificador_intencion.clasificar(pregunta)
        return tipo

    def ejecutar_sql(self, pregunta_usuario):

        print("Consultando base de datos DuckDB...")
        try:

            df_resultado = self.sql_engine.generar_y_ejecutar(pregunta_usuario)
            
            if not df_resultado.empty:
                print("\nDATOS EXTRAÍDOS DE DUCKDB:")
                print(df_resultado.to_markdown(index=False)) 
                print("-" * 40)
                
            return df_resultado
                
        except ValueError as ve:
            return f"ERROR DE SEGURIDAD/VALIDACIÓN: {ve}"
        except RuntimeError as runtime_error:
            return f"ERROR DE EJECUCIÓN SQL (Tras reintentos): {runtime_error}"
        except Exception as e:
            if es_error_transitorio(e):
                return MENSAJE_ERROR_CONEXION
            return f"ERROR DESCONOCIDO EN SQL: {e}"

    def ejecutar_RAG(self, pregunta_usuario, k_relevantes=K_RELEVANTES_RAG, umbral=UMBRAL_RAG):

        resultado = self.rag_engine.consultar(pregunta_usuario, k_relevantes, umbral)
        if isinstance(resultado, str):
            print("  -> No se encontró contexto relevante.")
        else:
            print("  -> Literatura recuperada correctamente.")
        return resultado
    
    def ejecutar_pipeline(self, pregunta_usuario):

        print(f"Procesando solicitud: '{pregunta_usuario}'\n")

        tipo = self.clasificar_pregunta(pregunta_usuario)

        if tipo == "SQL":
            datos = self.ejecutar_sql(pregunta_usuario)
            return self.generar_sql_only(pregunta_usuario, datos)

        elif tipo == "RAG":
            literatura = self.ejecutar_RAG(pregunta_usuario)
            return self.generar_rag_only(pregunta_usuario, literatura)

        elif tipo == "HIBRIDO":
            datos = self.ejecutar_sql(pregunta_usuario)
            literatura = self.ejecutar_RAG(pregunta_usuario)

            # ejecutar_sql devuelve un DataFrame si tuvo éxito y un str si falló
            sql_invalido = isinstance(datos, str)
            sql_vacio = (not isinstance(datos, str)) and self._sin_datos_reales(datos)

            if sql_invalido or sql_vacio:
                aviso = (
                    "No se pudo obtener evidencia empírica válida de la base de datos "
                    "para esta consulta, por lo que la siguiente respuesta se fundamenta "
                    "únicamente en la literatura científica.\n"
                )
                nota = self._nota_disponibilidad() if sql_vacio else ""
                return aviso + nota + "\n\n" + self.generar_rag_only(pregunta_usuario, literatura)

            return self.generar_sintesis(pregunta_usuario, datos, literatura)

    def detectar_tipo_resultado(self, df):
        total_filas = len(df)

        if total_filas == 1:
            return "AGREGACION"

        tiene_fecha = any(
            pd.api.types.is_datetime64_any_dtype(df[col]) or "date" in col.lower() or "time" in col.lower()
            for col in df.columns
        )

        if total_filas <= LIMITE_INTERPRETACION:
            if tiene_fecha:
                return "SERIE_TEMPORAL"
            return "TABLA_PEQUEÑA"

        return "TABLA_GRANDE"

    def _sin_datos_reales(self, df):
        """True si el resultado SQL no contiene ningún valor numérico real:
        tabla vacía, o agregados NaN porque el filtro (p. ej. un año fuera de
        rango) no encontró ninguna fila. Un conteo de 0 NO cuenta como vacío."""
        if not isinstance(df, pd.DataFrame):
            return False
        if df.empty:
            return True
        numericas = df.select_dtypes(include="number")
        return len(numericas.columns) > 0 and bool(numericas.isna().all().all())

    def _nota_disponibilidad(self):
        """Aviso con la última fecha disponible de las tablas base consultadas,
        para que el usuario pueda reformular dentro del rango existente."""
        rangos = self.sql_engine.rango_fechas_disponible(
            self.sql_engine.ultimas_tablas_usadas or []
        )
        if not rangos:
            return ""
        lineas = [
            f"- **{t}**: el dato más reciente es del {mx} (disponible desde {mn})."
            for (t, _col, mn, mx) in rangos
        ]
        return (
            "\n\nNo hay registros para el periodo solicitado. "
            "Puedes reformular la pregunta dentro del rango disponible:\n"
            + "\n".join(lineas)
        )

    def generar_sql_only(self, pregunta, df_resultado):

        if isinstance(df_resultado, str): return df_resultado
        if self._sin_datos_reales(df_resultado):
            return (
                "No se han encontrado datos para el periodo o filtro solicitado."
                + self._nota_disponibilidad()
            )

        total_filas = len(df_resultado)
        tipo = self.detectar_tipo_resultado(df_resultado)

        if tipo == "TABLA_GRANDE":
            print("Filas devueltas por SQL:", total_filas)
            
            if total_filas > LIMITE_FILAS:
                return (
                    f"Se encontraron {total_filas} registros en total.\n\n"
                    f"Se muestran los primeros {LIMITE_FILAS}:\n\n"
                    f"{df_resultado.head(LIMITE_FILAS).to_markdown(index=False)}\n\n"
                    f"*(Tabla truncada. {total_filas - LIMITE_FILAS} registros adicionales no mostrados.)*"
                )
            else:
                return (
                    f"Se encontraron {total_filas} registros en total:\n\n"
                    f"{df_resultado.to_markdown(index=False)}"
                )

        datos = (
            f"Total de registros encontrados: {total_filas}.\n\n"
            f"{df_resultado.to_markdown(index=False)}"
        )

        instruccion_especifica = PROMPTS_POR_TIPO[tipo]

        prompt = (
            "Eres un analista de datos especializado en ciencias marinas. Responde a la pregunta del usuario usando exclusivamente los datos proporcionados.\n\n"
            f"<<< PREGUNTA >>>\n{pregunta}\n<<< FIN_PREGUNTA >>>\n\n"
            f"<<< DATOS >>>\n{datos}\n<<< FIN_DATOS >>>\n\n"
            "INSTRUCCIONES:\n"
            f"{instruccion_especifica}\n\n"
            "REGLAS GENERALES:\n"
            "- NO repitas las etiquetas <<< PREGUNTA >>> ni <<< DATOS >>> en tu respuesta.\n"
            "- NO inventes datos que no estén en la tabla.\n"
            "- Responde en el mismo idioma de la pregunta."
        )

        respuesta_llm = invocar_con_reintentos(self.llm_generador, prompt)
        
        # Recuperamos las tablas que acaba de usar DuckDB
        fuentes_sql = self.sql_engine.ultimas_tablas_usadas
        
        if fuentes_sql:
            respuesta_llm += "\nTablas de datos consultadas:\n"
            for tabla in fuentes_sql:
                respuesta_llm += f"- {tabla}\n"
                
        return respuesta_llm


    def generar_rag_only(self, pregunta, resultado_rag, incluir_fuentes=True):

        if isinstance(resultado_rag, str):
            contexto = resultado_rag  # El mensaje de "No se encontró..."
            fuentes = []              # Lista de fuentes vacía
        else:
            contexto = resultado_rag["contexto"]
            fuentes = resultado_rag["fuentes"]

        prompt = (
            "Eres un investigador científico riguroso. Tu tarea es responder a la pregunta del usuario "
            "basándote ÚNICAMENTE en el contexto de la literatura proporcionada.\n\n"
            f"<<<PREGUNTA>>>\n{pregunta}\n<<<FIN_PREGUNTA>>>\n\n"
            f"<<<CONTEXTO_CIENTIFICO>>>\n{contexto}\n<<<FIN_CONTEXTO>>>\n\n"
            "INSTRUCCIONES CRÍTICAS:\n\n"
            "1. CERO ALUCINACIONES\n"
            "Responde solo con hechos que aparezcan explícitamente en el contexto. No uses tus conocimientos previos.\n\n"
            "2. ANTI-COMPLACENCIA\n"
            "Si la pregunta menciona lugares, herramientas o conceptos que NO aparecen en el contexto, "
            "TIENES PROHIBIDO intentar asimilarlos o forzar una relación.\n\n"
            "3. CONFESIÓN DE IGNORANCIA\n"
            "Si el contexto no responde a la pregunta, di textualmente: "
            "'La literatura científica recuperada no menciona [lo que el usuario ha preguntado]. "
            "El documento trata sobre [de qué trata el contexto].'\n\n"
             "4. IGNORA BIBLIOGRAFÍA\n"
            "Si algún fragmento del contexto es una lista de referencias bibliográficas "
            "(contiene patrones como 'Autor et al., año', DOI o URLs), ignóralo completamente.\n\n"
            "5. Sé directo y académico. Responde en el mismo idioma de la pregunta."
        )

        respuesta = invocar_con_reintentos(self.llm_generador, prompt)

        if incluir_fuentes and fuentes:
            respuesta_final = respuesta + "\nFuentes consultadas:\n"
            for fuente in fuentes:
                respuesta_final += f"- {fuente}\n"
            return respuesta_final

        return respuesta


    def generar_sintesis(self, pregunta, datos, resultado_rag):

        if isinstance(datos, str):
            datos_str = datos
            total_filas = 0
        else:
            total_filas = len(datos)

            if total_filas > LIMITE_FILAS:
                df_reducido = datos.head(LIMITE_FILAS)
                datos_str = (
                    f"Total de registros devueltos: {total_filas}. "
                    f"Se muestran los primeros {LIMITE_FILAS}:\n\n"
                    f"{df_reducido.to_markdown(index=False)}\n\n"
                    f"[NOTA DEL SISTEMA: Tabla truncada. Solo se muestran {LIMITE_FILAS} de {total_filas} registros.]"
                )
            else:
                datos_str = (
                    f"Total de registros devueltos: {total_filas}.\n\n"
                    f"{datos.to_markdown(index=False)}"
                )

        if isinstance(resultado_rag, str):
            contexto = resultado_rag
            fuentes = []
        else:
            contexto = resultado_rag["contexto"]
            fuentes = resultado_rag["fuentes"]

        prompt_sintesis = (
            "Eres un analista de datos estrictamente objetivo. Usa solo la información de los bloques proporcionados.\n\n"
            f"<<<PREGUNTA>>>\n{pregunta}\n<<<FIN_PREGUNTA>>>\n\n"
            f"<<<DATOS_SQL>>>\n{datos_str}\n<<<FIN_DATOS_SQL>>>\n\n"
            f"<<<CONTEXTO_CIENTIFICO>>>\n{contexto}\n<<<FIN_CONTEXTO_CIENTIFICO>>>\n\n"
            "INSTRUCCIONES:\n\n"
            "1. CERO ALUCINACIONES\n"
            "No inventes datos, fechas ni conclusiones que no estén en la tabla.\n\n"
            "2. CONTEXTO CIENTÍFICO\n"
            "Usa ÚNICAMENTE el texto del bloque CONTEXTO_CIENTIFICO para la parte teórica. "
            "Aunque conozcas información relevante sobre el tema, NO la incluyas si no está en el contexto. \n\n"
            "3. IDIOMA Y FORMATO\n"
            "Responde en el mismo idioma de la pregunta. Sé directo y conciso."
        )

        respuesta_limpia = invocar_con_reintentos(self.llm_generador, prompt_sintesis)

        fuentes_sql = self.sql_engine.ultimas_tablas_usadas
        
        hay_datos_sql = (not isinstance(datos, str)) and not datos.empty

        if (fuentes_sql and hay_datos_sql) or fuentes:
            respuesta_final = respuesta_limpia + "\n\n---\n"

            if fuentes_sql and hay_datos_sql:
                respuesta_final += "\nTablas de datos consultadas:\n"
                for tabla in fuentes_sql:
                    respuesta_final += f"- {tabla}\n"

            if fuentes:
                respuesta_final += "Literatura científica consultada:\n"
                for fuente in fuentes:
                    respuesta_final += f"- {fuente}\n"

            return respuesta_final

        return respuesta_limpia

if __name__ == "__main__":

    ruta_csv_real = os.path.join(BASE_DIR, "datos", "datos_procesados")
    ruta_directorio_pdfs = os.path.join(BASE_DIR, "datos", "Docs")
    
    
    try:
        app = Orquestador(ruta_csv_real, ruta_directorio_pdfs)

        while True:
            pregunta = input("\n¿Qué quieres saber de los datos? (o 'salir'): ")
            
            if pregunta.lower() in ["salir", "exit", "q"]:
                break
            if not pregunta.strip(): continue

            respuesta = app.ejecutar_pipeline(pregunta)
            
            print("\nRespuesta final inferida:")
            print("-" * 40)
            print(respuesta)
            print("-" * 40)

    except Exception as e:
        print(f"\n Error fatal iniciando el orquestador: {e}")