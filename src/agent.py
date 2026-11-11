"""Agente LLM con loop agéntico manual (NVIDIA NIM, API compatible con OpenAI).

Loop: el modelo decide -> se ejecutan las herramientas pedidas -> los resultados
vuelven como mensajes "tool" -> se repite hasta respuesta final o T_MAX pasos.

Uso rápido:
    python -m src.agent "¿Cómo está el aire en Itagüí?"
"""
import json
import re
import sys
import unicodedata

from openai import OpenAI

from src.config import NVIDIA_BASE_URL, nvidia_api_key, nvidia_model
from src.tools import TOOL_SCHEMAS, TOOLS

T_MAX = 5

NO_INFO = "No puedo responder esa pregunta porque no tengo la información necesaria para hacerlo."
SCOPE_HINT = (
    "Puedo ayudarte con el PM2.5 medido en las estaciones de SIATA, la predicción del día "
    "siguiente y lo que dicen la normativa y las guías sobre calidad del aire."
)

# Saludos, agradecimientos y despedidas: se permiten sin consultar herramientas
_SMALL_TALK = re.compile(
    r"^(hola|holi|buenas|buenos dias|buenas tardes|buenas noches|hey|que mas|que tal|"
    r"como estas|gracias|muchas gracias|ok|listo|chao|adios|hasta luego)\b"
)


def is_small_talk(text: str) -> bool:
    t = unicodedata.normalize("NFKD", text.lower()).encode("ascii", "ignore").decode()
    t = re.sub(r"[^a-z ]", " ", t).strip()
    t = re.sub(r"\s+", " ", t)
    return bool(_SMALL_TALK.match(t)) and len(t.split()) <= 6

SYSTEM_PROMPT = """Eres un asistente de calidad del aire del Valle de Aburrá (Medellín, Colombia).
Ayudas a los ciudadanos a entender el PM2.5 y qué hacer al respecto.

Regla principal (obligatoria):
- Responde ÚNICAMENTE con información que obtengas de las herramientas en esta conversación.
  No uses tu conocimiento general, ni siquiera para temas de calidad del aire.
- Si la pregunta no trata sobre la calidad del aire del Valle de Aburrá, o si las herramientas
  no devuelven la información necesaria (error, sin datos o sin documentos relevantes), responde:
  "No puedo responder esa pregunta porque no tengo la información necesaria para hacerlo."
  Puedes añadir en una frase qué tipo de preguntas sí puedes responder. No intentes adivinar.
- Excepción: a saludos, agradecimientos o despedidas responde de forma amable y breve,
  y presenta en una frase en qué puedes ayudar.

Reglas:
- Responde siempre en español, claro y breve.
- Para datos medidos de un día usa get_pm25; para el peor o mejor día, el promedio o los días
  sobre la norma de una estación usa station_summary; para el pronóstico del día siguiente usa predict_tomorrow;
  para normas, protocolos de episodios críticos o recomendaciones de salud usa search_docs.
  Si no sabes el código de una estación, usa list_stations.
- No inventes cifras: usa solo lo que devuelven las herramientas.
- Cita siempre la fuente al final, por ejemplo [SIATA] o [Resolución 2254 de 2017, p. 5].
- Los datos disponibles son históricos (julio 2023 a julio 2024); dilo cuando sea relevante.
- Si una herramienta devuelve un error, explícalo y sugiere una alternativa.
- No das consejo médico: ante síntomas, recomienda consultar a un profesional de la salud."""


def _client() -> OpenAI:
    key = nvidia_api_key()
    if not key:
        raise RuntimeError("Falta NVIDIA_API_KEY (configúrala en .env o en los Secrets de Streamlit)")
    return OpenAI(base_url=NVIDIA_BASE_URL, api_key=key, timeout=90, max_retries=1)


def run_tool(name: str, arguments: str) -> dict:
    """Ejecuta una herramienta; cualquier error vuelve como observación."""
    if name not in TOOLS:
        return {"error": f"Herramienta desconocida: {name}"}
    try:
        args = json.loads(arguments or "{}")
        return TOOLS[name](**args)
    except Exception as e:  # el error se le muestra al modelo, no rompe la app
        return {"error": f"{type(e).__name__}: {e}"}


def ask(
    question: str,
    history: list[dict] | None = None,
    model: str | None = None,
    context: str | None = None,
) -> dict:
    """Responde una pregunta. Devuelve la respuesta y la traza de herramientas usadas."""
    client = _client()
    model = model or nvidia_model()
    messages = [{"role": "system", "content": SYSTEM_PROMPT}]
    messages += history or []
    # El contexto del dashboard (ej. estación seleccionada) va aparte de la pregunta
    # para que no afecte la detección de saludos
    content = f"{question}\n({context})" if context else question
    messages.append({"role": "user", "content": content})
    trace = []

    for _ in range(T_MAX):
        resp = client.chat.completions.create(
            model=model,
            messages=messages,
            tools=TOOL_SCHEMAS,
            temperature=0.2,
            max_tokens=1024,
        )
        msg = resp.choices[0].message
        if not msg.tool_calls:
            answer = (msg.content or "").strip()
            # Sin ninguna herramienta consultada, la respuesta no vendría del
            # contexto sino del conocimiento general del modelo: se rechaza.
            if not trace and NO_INFO not in answer and not is_small_talk(question):
                answer = f"{NO_INFO} {SCOPE_HINT}"
            return {"answer": answer, "trace": trace}

        messages.append(
            {
                "role": "assistant",
                "content": msg.content or "",
                "tool_calls": [tc.model_dump() for tc in msg.tool_calls],
            }
        )
        for tc in msg.tool_calls:
            result = run_tool(tc.function.name, tc.function.arguments)
            trace.append(
                {"tool": tc.function.name, "args": tc.function.arguments, "result": result}
            )
            messages.append(
                {
                    "role": "tool",
                    "tool_call_id": tc.id,
                    "content": json.dumps(result, ensure_ascii=False, default=str),
                }
            )

    # Se alcanzó T_MAX: un último paso sin herramientas para que responda con
    # lo que ya obtuvo, en vez de quedarse sin respuesta
    messages.append(
        {
            "role": "system",
            "content": "Ya no puedes usar más herramientas. Responde ahora solo con la "
            "información que obtuviste y cita la fuente. Si no alcanza para responder, "
            "dilo con la frase indicada.",
        }
    )
    try:
        resp = client.chat.completions.create(
            model=model,
            messages=messages,
            tools=TOOL_SCHEMAS,
            tool_choice="none",
            temperature=0.2,
            # Más margen: el modelo razona antes de responder y con 1024
            # tokens a veces se queda sin espacio para la respuesta
            max_tokens=4096,
        )
        answer = (resp.choices[0].message.content or "").strip()
    except Exception:
        answer = ""
    return {
        "answer": answer
        or "No pude completar la respuesta en el número máximo de pasos. Intenta reformular la pregunta.",
        "trace": trace,
    }


if __name__ == "__main__":
    q = " ".join(sys.argv[1:]) or "¿Cómo está el aire en Itagüí y cómo estará mañana?"
    out = ask(q)
    for t in out["trace"]:
        print(f"[tool] {t['tool']}({t['args']})")
    print("\n" + out["answer"])
