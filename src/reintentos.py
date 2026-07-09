"""
Utilidades de robustez para las llamadas al LLM.

El sistema puede usar un modelo en la nube de Ollama (p. ej. `deepseek-v4-pro:cloud`),
cuyas invocaciones salen por red hacia ollama.com y, por tanto, pueden fallar de
forma transitoria (timeouts, 502/503, fallos de DNS, conexión reseteada...).
Estas funciones detectan esos errores y reintentan con espera exponencial en
lugar de abortar la consulta a la primera.
"""
import time

# Fragmentos típicos presentes en el mensaje de un error de red transitorio.
_ERRORES_TRANSITORIOS = (
    "502", "503", "504", "bad gateway", "unavailable", "gateway",
    "timeout", "i/o timeout", "deadline", "temporarily",
    "dial tcp", "connection", "connect", "reset", "refused", "broken pipe",
    "eof", "lookup", "no such host", "name resolution", "network",
)


def es_error_transitorio(exc):
    """True si la excepción parece un fallo de red/servicio recuperable."""
    msg = str(exc).lower()
    return any(frag in msg for frag in _ERRORES_TRANSITORIOS)


def invocar_con_reintentos(llm, prompt, intentos=3, espera_inicial=1.5):
    """
    Ejecuta `llm.invoke(prompt)` reintentando ante errores transitorios de red.

    Reintenta hasta `intentos` veces con espera exponencial (1.5s, 3s, ...).
    Los errores NO transitorios (o el último intento) se propagan tal cual.
    """
    espera = espera_inicial
    ultimo = None
    for i in range(intentos):
        try:
            return llm.invoke(prompt)
        except Exception as e:  # noqa: BLE001
            ultimo = e
            if not es_error_transitorio(e) or i == intentos - 1:
                raise
            print(
                f"  [reintento LLM {i + 1}/{intentos - 1}] fallo de red transitorio "
                f"({type(e).__name__}). Reintentando en {espera:.0f}s...",
                flush=True,
            )
            time.sleep(espera)
            espera *= 2
    raise ultimo
