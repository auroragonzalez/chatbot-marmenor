"""
Servidor web del chatbot híbrido SQL + RAG.

Envuelve el `Orquestador` del TFG en una API FastAPI y sirve una interfaz de
chat en el navegador. El orquestador es pesado de inicializar (carga DuckDB,
el modelo de embeddings e ingesta los PDFs en ChromaDB), así que se construye
una única vez en un hilo de fondo al arrancar; la UI consulta /status hasta
que está listo.

Configuración por variables de entorno (todas opcionales):
  MODELO_LLM  Modelo de Ollama a usar        (def. "llama3")
  RUTA_CSV    Carpeta con los CSV procesados  (def. <repo>/datos/datos_procesados)
  RUTA_PDFS   Carpeta con los PDF del corpus  (def. <repo>/datos/Docs)
  HOST/PORT   Interfaz y puerto del servidor  (def. 0.0.0.0:8000)
"""
import os
import sys
import threading
import traceback
from contextlib import asynccontextmanager

from fastapi import FastAPI
from fastapi.responses import FileResponse, JSONResponse
from pydantic import BaseModel

BASE_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
SRC_DIR = os.path.join(BASE_DIR, "src")
WEB_DIR = os.path.dirname(os.path.abspath(__file__))

# El orquestador y sus módulos usan imports planos (from generadorSQL import ...),
# así que src/ debe estar en el path.
sys.path.insert(0, SRC_DIR)

RUTA_CSV = os.environ.get("RUTA_CSV", os.path.join(BASE_DIR, "datos", "datos_procesados"))
RUTA_PDFS = os.environ.get("RUTA_PDFS", os.path.join(BASE_DIR, "datos", "Docs"))

# Estado global del motor. Se rellena desde un hilo de fondo.
_estado = {"listo": False, "error": None}
_orquestador = None
# El pipeline no es seguro para concurrencia (DuckDB + estado mutable en el
# motor SQL), así que serializamos las peticiones.
_lock = threading.Lock()


def _inicializar_motor():
    """Construye el Orquestador. Se ejecuta en un hilo aparte al arrancar."""
    global _orquestador
    try:
        from orquestador import Orquestador  # import diferido: es costoso

        modelo = os.environ.get("MODELO_LLM", "llama3")
        print(f"[init] Inicializando orquestador (modelo={modelo})...", flush=True)
        print(f"[init]   CSV : {RUTA_CSV}", flush=True)
        print(f"[init]   PDFs: {RUTA_PDFS}", flush=True)
        _orquestador = Orquestador(RUTA_CSV, RUTA_PDFS)
        _estado["listo"] = True
        print("[init] Orquestador listo.", flush=True)
    except Exception as e:  # noqa: BLE001 - queremos exponer cualquier fallo en la UI
        _estado["error"] = f"{type(e).__name__}: {e}"
        traceback.print_exc()


@asynccontextmanager
async def lifespan(app: FastAPI):
    hilo = threading.Thread(target=_inicializar_motor, daemon=True)
    hilo.start()
    yield


app = FastAPI(title="Chatbot híbrido SQL + RAG", lifespan=lifespan)


class Mensaje(BaseModel):
    message: str


@app.get("/")
def index():
    return FileResponse(os.path.join(WEB_DIR, "index.html"))


@app.get("/status")
def status():
    return {
        "ready": _estado["listo"],
        "error": _estado["error"],
        "model": os.environ.get("MODELO_LLM", "llama3"),
    }


@app.post("/chat")
def chat(msg: Mensaje):
    if _estado["error"]:
        return JSONResponse(status_code=503, content={"error": _estado["error"]})
    if not _estado["listo"] or _orquestador is None:
        return JSONResponse(
            status_code=503,
            content={"error": "El motor todavía se está inicializando. Espera unos segundos."},
        )

    pregunta = (msg.message or "").strip()
    if not pregunta:
        return JSONResponse(status_code=400, content={"error": "Pregunta vacía."})

    # Solo una petición a la vez recorre el pipeline.
    with _lock:
        try:
            respuesta = _orquestador.ejecutar_pipeline(pregunta)
        except Exception as e:  # noqa: BLE001
            traceback.print_exc()
            return JSONResponse(
                status_code=500,
                content={"error": f"Error procesando la consulta: {type(e).__name__}: {e}"},
            )

    return {"reply": respuesta if isinstance(respuesta, str) else str(respuesta)}


if __name__ == "__main__":
    import uvicorn

    host = os.environ.get("HOST", "0.0.0.0")
    port = int(os.environ.get("PORT", "8000"))
    uvicorn.run(app, host=host, port=port)
