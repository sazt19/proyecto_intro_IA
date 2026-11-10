"""Evaluación del agente con preguntas de prueba fijas.

Para cada pregunta revisa si el agente llamó a la herramienta esperada y si citó
la fuente. Guarda la tabla en docs/evaluacion_agente.md.

Uso:
    python -m src.evaluate
"""
import re

from src.agent import NO_INFO, ask
from src.config import DOCS_DIR

# (pregunta, herramientas esperadas; vacío = debe negarse a responder)
PREGUNTAS = [
    ("¿Qué estaciones hay disponibles?", {"list_stations"}),
    ("¿Cuánto PM2.5 hubo en Bello el 2024-03-15?", {"get_pm25"}),
    ("¿Cómo está el aire en Envigado?", {"get_pm25"}),
    ("¿Cómo estará el aire mañana en Sabaneta?", {"predict_tomorrow"}),
    ("Predice la calidad del aire en Caldas para el día siguiente al 2024-04-10", {"predict_tomorrow"}),
    ("¿Cuál es el límite diario de PM2.5 en Colombia?", {"search_docs"}),
    ("¿Qué recomienda la OMS sobre el PM2.5?", {"search_docs"}),
    ("¿Qué es el POECA?", {"search_docs"}),
    ("¿Cuáles son los niveles de alerta y emergencia por PM2.5?", {"search_docs"}),
    ("¿Qué significa que el aire esté en categoría dañina para grupos sensibles?", {"search_docs"}),
    ("¿Cómo está el aire en Barbosa hoy y cómo estará mañana?", {"get_pm25", "predict_tomorrow"}),
    ("¿Cuánto PM2.5 hubo en Bello el 2020-01-01?", {"get_pm25"}),  # fecha fuera de rango
    ("¿Cuál es la receta de la bandeja paisa?", set()),  # fuera de tema
]

FUENTES = re.compile(r"\[|SIATA|Resoluci[oó]n|OMS|AMVA|Open-Meteo|fuente", re.I)


def main():
    filas = []
    for i, (pregunta, esperadas) in enumerate(PREGUNTAS, 1):
        try:
            r = ask(pregunta)
            usadas = {t["tool"] for t in r["trace"]}
            respuesta = r["answer"]
        except Exception as e:
            usadas, respuesta = set(), f"ERROR: {e}"
        if esperadas:
            ok_tool = esperadas <= usadas
            ok_cita = bool(FUENTES.search(respuesta))
        else:  # fuera de tema: debe negarse sin inventar
            ok_tool = NO_INFO in respuesta
            ok_cita = None
        filas.append((i, pregunta, esperadas, usadas, ok_tool, ok_cita, respuesta))
        print(f"{i:2d}. herramienta {'OK' if ok_tool else 'FALLA'} | {pregunta}")

    si = lambda v: "—" if v is None else ("✅" if v else "❌")
    lineas = [
        "# Evaluación del agente",
        "",
        "Generado con `python -m src.evaluate`.",
        "",
        "| # | Pregunta | Herramienta esperada | Herramientas usadas | ¿Correcta? | ¿Citó fuente? |",
        "|---|---|---|---|---|---|",
    ]
    for i, p, esp, us, ok_t, ok_c, _ in filas:
        esp_txt = ", ".join(sorted(esp)) or "ninguna (debe negarse)"
        lineas.append(f"| {i} | {p} | {esp_txt} | {', '.join(sorted(us)) or '—'} | {si(ok_t)} | {si(ok_c)} |")
    n_t = sum(f[4] for f in filas)
    con_cita = [f for f in filas if f[5] is not None]
    n_c = sum(f[5] for f in con_cita)
    lineas += [
        "",
        f"**Herramienta correcta:** {n_t}/{len(filas)} · **Citó fuente:** {n_c}/{len(con_cita)}",
        "",
        "## Respuestas",
        "",
    ]
    for i, p, *_, resp in filas:
        lineas += [f"**{i}. {p}**", "", resp.replace("\n", " "), ""]

    salida = DOCS_DIR / "evaluacion_agente.md"
    salida.write_text("\n".join(lineas), encoding="utf-8")
    print(f"\nHerramienta correcta {n_t}/{len(filas)} · citó fuente {n_c}/{len(con_cita)} -> {salida}")


if __name__ == "__main__":
    main()
