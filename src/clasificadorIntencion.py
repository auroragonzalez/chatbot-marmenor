from langchain_huggingface import HuggingFaceEmbeddings
import numpy as np

MARGEN_EMPATE = 0.008
MARGEN_HIBRIDO = 0.015
TOP_K = 4
MODELO_EMBEDDINGS = "intfloat/multilingual-e5-small"

class ClasificadorIntencion:
    def __init__(self):
        self.embedder = HuggingFaceEmbeddings(
            model_name=MODELO_EMBEDDINGS,
            model_kwargs={'device': 'cpu'},
            encode_kwargs={'normalize_embeddings': True} 
        )

        self.ejemplos = {
            "SQL": [
                "¿Cuál fue la temperatura media mensual registrada en la boya de Los Urrutias durante el verano?",
                "Filtra los registros de la estación E3 donde la concentración de clorofila-a superó los 5 mg/m3 y devuélveme la fecha exacta.",
                "Dime el valor máximo de precipitación acumulada en la estación meteorológica de San Javier.",
                "¿Cuántos días el nivel de oxígeno disuelto en la Cubeta Sur estuvo por debajo de 4 mg/L?",
                "¿Cuál es el registro de caudal más alto en la rambla del Albujón según los datos del SAIH?",
                "Muestra los valores de conductividad medidos en la boya de Entreislas cuando la temperatura del agua superó los 28 grados.",
                "¿Qué día se registró el nivel freático más bajo en los piezómetros del Campo de Cartagena?",
                "Busca en la base de datos los registros donde el pH de la boya de Los Nietos haya bajado de 7.5.",
                "Dime a qué hora exacta se midió la temperatura del agua más alta en la boya de Los Narejos.",
                "Extrae todos los registros de la estación E3 donde la salinidad sea mayor a 44 PSU.",
                "¿Cuál fue el valor máximo o mínimo registrado?",
                "¿Cuántos registros superan este umbral en la tabla?",
                "Muéstrame los datos de la estación del mes de julio.",
                "Dame el promedio mensual y la desviación estándar.",
                "¿Qué día y hora exactos se registró el valor más extremo?",
                "Filtra los registros por variable y devuelve la tabla resultante.",
                "¿Cuál es el valor mínimo de oxígeno disuelto registrado en la base de datos?",
                "Calcula la media de turbidez agrupada por mes.",
                "¿Cuántos registros superan el umbral de salinidad de 38 PSU?",
                "Dame la desviación típica de la temperatura del agua para cada año disponible.",
                "Muestra los registros donde la concentración de clorofila-a fue mayor a 10 mg/m3.",
                "¿Cuál fue el valor medio anual de salinidad en 2025?",
                "¿Cuáles han sido los valores medios de temperatura en 2024?",
                "Dame la media de cada variable agrupada por año.",
                "¿Cuál es el promedio de clorofila-a durante 2023?",
                "Media anual de oxígeno disuelto por estación."
            ],
            "RAG": [
                "Según los informes del IEO-CSIC, ¿cuáles son las principales causas del fenómeno de la mancha blanca en el Mar Menor?",
                "¿Qué ventajas tiene el algoritmo C2RCC para estimar clorofila-a con Sentinel-2 en aguas someras según la literatura?",
                "¿Qué modelos de Machine Learning y Deep Learning se proponen en la literatura para monitorizar la calidad del agua?",
                "¿Qué metodología se utilizó para aplicar Landsat-8 y Sentinel-2 como sistema de alerta temprana en lagunas costeras?",
                "¿Cómo se realiza la corrección atmosférica sobre aguas continentales según los estudios de teledetección?",
                "¿Qué relación causal existe entre la anoxia y la mortandad masiva de peces según los papers?",
                "¿Qué es el índice NDCI propuesto por Mishra y Mishra y cómo se calcula?",
                "Detalla el uso de redes neuronales artificiales para modelar la eutrofización en ecosistemas acuáticos.",
                "¿Qué metodología, modelo empírico o algoritmo propone el estudio?",
                "Según el artículo, ¿cómo se define este fenómeno?",
                "¿Qué conclusiones o resultados principales extrae el paper?",
                "Según la literatura científica, ¿qué factores influyen en este proceso?",
                "¿Qué instrumentos o herramientas se recomiendan según el documento?",
                "Explícame la teoría descrita en el documento sobre este proceso.",
                "¿Qué factores influyen en la eutrofización de lagunas costeras según la literatura científica?",
                "¿Qué sensores satelitales se recomiendan para monitorizar la calidad del agua marina?",
                "¿Cómo afecta la turbidez a la penetración de luz en ecosistemas acuáticos según los estudios?",
                "¿Qué índices espectrales se usan para detectar floraciones de algas en aguas costeras?",
                "Según la literatura, ¿cómo afecta la escorrentía agrícola a la salinidad y nutrientes de una laguna?",
                "¿Qué métodos analíticos se emplean para medir parámetros biofísicos en agua de mar?",
                "¿Qué variabilidad espacio-temporal presentan los parámetros de calidad del agua en lagunas costeras?"
            ],
            "HIBRIDO": [
                "Cruza los picos de clorofila-a medidos en la estación E3 con los umbrales de eutrofización definidos en los informes del IEO-CSIC.",
                "¿Explica la teoría de Pérez-Ruzafa la bajada repentina de salinidad que muestran los datos de la boya de Los Urrutias?",
                "Verifica si las temperaturas extremas registradas en la boya de Pedruchillo coinciden con los periodos de crisis ecológica mencionados en los artículos.",
                "Basado en la evolución del pH y materia orgánica de la estación E3 en 2023, ¿qué fase de eutrofización estaría atravesando el Mar Menor según los autores?",
                "Relaciona los niveles altos de sólidos disueltos totales en los datos empíricos con las causas de la mancha blanca descritas por el CSIC.",
                "Analiza los datos de precipitación de San Javier e indica cómo este tipo de eventos DANA afectan a la turbidez según la literatura.",
                "¿Coinciden los datos medidos en nuestra tabla con lo que describe la literatura?",
                "Compara los registros reales de la estación con la teoría del artículo.",
                "¿Es normal el valor registrado hoy si nos basamos en los estudios científicos?",
                "¿Hay anomalías en nuestra base de datos respecto a los umbrales del paper?",
                "Valida los picos de nuestra tabla usando el modelo teórico del documento.",
                "¿Nuestra tabla empírica refleja los episodios descritos por los autores?",
                "¿Podemos usar la teoría del artículo para interpretar los datos de nuestra tabla?",
                "¿Los patrones de nuestra base de datos coinciden con las tendencias descritas en el paper?",
                "Cómo se relacionan los registros de nuestra tabla con los conceptos teóricos del documento?",
                "Extrae el valor medio de oxígeno disuelto de los datos empíricos e indica si, según los papers, estos niveles son críticos para las especies bentónicas.",
                "A partir de la temperatura máxima registrada en los datos, investiga en la bibliografía el efecto del calentamiento en la dinámica de nutrientes.",
                "Compara el caudal máximo registrado en los datos con los impactos de escorrentía descritos en la literatura científica.",
                "Calcula la concentración media de clorofila-a en los datos empíricos y contrástala con los umbrales de eutrofización definidos en los artículos.",
                "Analiza las tendencias de nuestra tabla de datos y explica si concuerdan con los patrones documentados en los papers",
                "¿Justifica la literatura el comportamiento anómalo del pH observado en nuestra base de datos durante los episodios de lluvia intensa?"
            ]
        }

        self.vectores_ejemplos = []
        for tipo, frases in self.ejemplos.items():
            for frase in frases:
                vector = self.embedder.embed_query("query: " + frase)
                self.vectores_ejemplos.append({
                    "tipo": tipo,
                    "texto": frase,
                    "vector": np.array(vector)
                })

    def similitud_coseno(self, a, b):
        """
        OPTIMIZACIÓN MATEMÁTICA: 
        Como usamos normalize_embeddings=True, la longitud de los vectores es 1.
        Por tanto,  np.dot equivale exactamente a la Similitud del Coseno,
        ahorrando ciclos de CPU.
        """
        return np.dot(a, b)

    def clasificar(self, pregunta):

        emb_pregunta = np.array(
            self.embedder.embed_query("query: " + pregunta)
        )

        resultados = []

        for item in self.vectores_ejemplos:
            score = self.similitud_coseno(
                emb_pregunta,
                item["vector"]
            )
            resultados.append({
                "tipo": item["tipo"],
                "score": float(score),
                "texto_match": item["texto"]
            })

        resultados = sorted(
            resultados,
            key=lambda x: x["score"],
            reverse=True
        )

        mejor_match = resultados[0]

        # ----------------------------------
        # calcular promedio por categoría
        # ----------------------------------

        scores_categoria = {"SQL": [], "RAG": [], "HIBRIDO": []}

        for r in resultados:
            scores_categoria[r["tipo"]].append(r["score"])

        scores = {}
        for categoria, lista in scores_categoria.items():
            top_k = sorted(lista, reverse=True)[:TOP_K]
            scores[categoria] = float(np.mean(top_k))

        tipo_detectado = max(scores, key=scores.get)
        mejor_score = scores[tipo_detectado]
        

        # ----------------------------------
        # empate SQL vs RAG (si no se ha detectado híbrido)
        # ----------------------------------
        if tipo_detectado != "HIBRIDO":
            empate_sql_rag = abs(scores["SQL"] - scores["RAG"]) < MARGEN_EMPATE
            # El empate SQL/RAG solo es un HÍBRIDO real si el banco HÍBRIDO
            # también lo respalda: su score debe estar cerca del ganador.
            # Así evitamos que una agregación pura (que comparte vocabulario
            # con RAG, p.ej. "salinidad") se promocione a HÍBRIDO por un
            # empate espurio y se ponga a buscar en los papers sin necesidad.
            hibrido_competitivo = scores["HIBRIDO"] >= mejor_score - MARGEN_HIBRIDO
            if empate_sql_rag and hibrido_competitivo:
                print(
                    f"Empate real detectado "
                    f"(Dif SQL/RAG: {abs(scores['SQL'] - scores['RAG']):.3f}, "
                    f"HIBRIDO: {scores['HIBRIDO']:.3f}). "
                    f"Forzando ruta HÍBRIDA."
                )
                return "HIBRIDO", scores
            
        return tipo_detectado, scores
