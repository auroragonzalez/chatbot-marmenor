#!/usr/bin/env bash
# Lanza el chatbot web (backend FastAPI + interfaz de chat).
#
# Uso:
#   ./run_chatbot.sh [RUTA_DATOS]
#
# RUTA_DATOS es la carpeta que contiene 'datos_procesados/' (CSV) y 'Docs/' (PDF).
# Si no se pasa, se usa <repo>/datos.
#
# Variables de entorno reconocidas: MODELO_LLM, RUTA_CSV, RUTA_PDFS, HOST, PORT.
set -eu
cd "$(dirname "$0")"

DATOS="${1:-$PWD/datos}"

export MODELO_LLM="${MODELO_LLM:-deepseek-v4-pro:cloud}"
export RUTA_CSV="${RUTA_CSV:-$DATOS/datos_procesados}"
export RUTA_PDFS="${RUTA_PDFS:-$DATOS/Docs}"
export HOST="${HOST:-127.0.0.1}"
export PORT="${PORT:-8000}"

echo "Modelo : $MODELO_LLM"
echo "CSV    : $RUTA_CSV"
echo "PDFs   : $RUTA_PDFS"
echo "URL    : http://$HOST:$PORT"
echo

exec ./venv/bin/python -m uvicorn web.app:app --host "$HOST" --port "$PORT"
