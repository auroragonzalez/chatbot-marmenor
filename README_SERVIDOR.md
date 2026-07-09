# Chatbot híbrido SQL + RAG — Mar Menor

Paquete autocontenido del chatbot web (FastAPI + interfaz de chat) para desplegar
en un servidor. Incluye el código del pipeline, los datos procesados, el corpus
de PDFs y la base vectorial ChromaDB ya construida, así que arranca sin reingestar.

## Contenido

```
chatbot-mar-menor/
├── src/                 # Pipeline: orquestador, SQL, RAG, clasificador, reintentos
├── web/                 # app.py (FastAPI) + index.html (UI)
├── datos/
│   ├── datos_procesados/  # 5 CSV de series temporales (DuckDB en memoria)
│   └── Docs/              # 36 PDF del corpus
├── chroma_db/           # Base vectorial persistida (cosine) ya indexada
├── requirements.txt
├── run_chatbot.sh
└── LICENSE
```

## Requisitos

- **Python 3.12+**
- **Ollama** en ejecución y accesible, con el modelo deseado descargado.
  Por defecto `run_chatbot.sh` usa `deepseek-v4-pro:cloud`; para uno local:
  `ollama pull llama3` y exporta `MODELO_LLM=llama3`.

## Puesta en marcha

```bash
cd chatbot-mar-menor
python3.12 -m venv venv && source venv/bin/activate
pip install -r requirements.txt

# Para exponerlo fuera del servidor, escucha en todas las interfaces:
HOST=0.0.0.0 PORT=8000 ./run_chatbot.sh
```

El script sirve la UI en `http://HOST:PORT`. El motor se construye en un hilo de
fondo al arrancar (carga DuckDB + embeddings); la interfaz consulta `/status`
hasta que está listo. Endpoints: `GET /` (UI), `GET /status`, `POST /chat`.

## Variables de entorno

| Variable     | Por defecto                       | Descripción                          |
|--------------|-----------------------------------|--------------------------------------|
| `MODELO_LLM` | `deepseek-v4-pro:cloud`           | Modelo de Ollama a usar              |
| `RUTA_CSV`   | `datos/datos_procesados`          | Carpeta con los CSV                  |
| `RUTA_PDFS`  | `datos/Docs`                      | Carpeta con los PDF                  |
| `HOST`       | `127.0.0.1`                       | Interfaz de escucha (`0.0.0.0` = externa) |
| `PORT`       | `8000`                            | Puerto                               |

## Notas de despliegue

- `run_chatbot.sh` espera el intérprete en `./venv/bin/python`; crea el venv
  dentro de esta carpeta como se indica arriba.
- Para producción conviene poner un reverse proxy (nginx/caddy) delante y, si
  hace falta más de una instancia, recuerda que el pipeline **no es seguro para
  concurrencia** (serializa las peticiones con un lock): escala con procesos
  independientes, no con hilos.
- Si actualizas los PDF de `datos/Docs/`, borra `chroma_db/` para forzar la
  reingesta en el siguiente arranque.
