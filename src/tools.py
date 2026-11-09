"""Herramientas que el agente LLM puede llamar (tool calling).

Cada herramienta devuelve un dict serializable a JSON. Los errores se devuelven
como {"error": ...} para que el agente los reciba como observación.
"""
from functools import lru_cache

import pandas as pd

from src.config import STATES, ica_category
from src.data import load_daily, load_stations
from src.model import FEATURES, SoftmaxRegression, build_features

FUENTE_DATOS = "SIATA (PM2.5) y Open-Meteo (meteorología)"


@lru_cache(maxsize=1)
def _daily() -> pd.DataFrame:
    return load_daily()


@lru_cache(maxsize=1)
def _stations() -> pd.DataFrame:
    return load_stations()


@lru_cache(maxsize=1)
def _features() -> pd.DataFrame:
    return build_features(_daily())


@lru_cache(maxsize=1)
def _model() -> SoftmaxRegression:
    return SoftmaxRegression.load()


def _resolve_station(estacion: str) -> str:
    """Acepta el código (ITA-CJUS) o parte del nombre (Itagüí, Bello...)."""
    st = _stations()
    q = estacion.strip().lower()
    exact = st[st["codigo"].str.lower() == q]
    if len(exact):
        return exact.iloc[0]["codigo"]
    partial = st[st["nombre"].str.lower().str.contains(q, regex=False)]
    if len(partial) == 1:
        return partial.iloc[0]["codigo"]
    if len(partial) > 1:
        opciones = ", ".join(partial["codigo"])
        raise ValueError(f"'{estacion}' es ambiguo; opciones: {opciones}")
    raise ValueError(f"Estación '{estacion}' no encontrada. Usa list_stations().")


def list_stations() -> dict:
    st = _stations()
    return {
        "estaciones": st[["codigo", "nombre"]].to_dict(orient="records"),
        "fuente": FUENTE_DATOS,
    }


def get_pm25(estacion: str, fecha: str | None = None) -> dict:
    codigo = _resolve_station(estacion)
    df = _daily()
    df = df[df["codigo"] == codigo]
    if fecha:
        row = df[df["fecha"] == pd.to_datetime(fecha)]
        if row.empty:
            rango = f"{df['fecha'].min():%Y-%m-%d} a {df['fecha'].max():%Y-%m-%d}"
            return {"error": f"Sin dato válido para {codigo} el {fecha}. Rango disponible: {rango}"}
    else:
        row = df.tail(1)
    r = row.iloc[0]
    categoria, _ = ica_category(r["pm25"])
    return {
        "estacion": codigo,
        "fecha": f"{r['fecha']:%Y-%m-%d}",
        "pm25_promedio_24h_ugm3": round(float(r["pm25"]), 1),
        "categoria_ica": categoria,
        "lluvia_mm": r["lluvia_mm"],
        "viento_max_kmh": r["viento_max_kmh"],
        "nota": "El conjunto de datos histórico termina el 2024-07-30; 'último dato' se refiere a esa fecha.",
        "fuente": FUENTE_DATOS,
    }


def predict_tomorrow(estacion: str, fecha: str | None = None) -> dict:
    codigo = _resolve_station(estacion)
    f = _features()
    f = f[f["codigo"] == codigo].dropna(subset=FEATURES)
    if fecha:
        f = f[f["fecha"] == pd.to_datetime(fecha)]
        if f.empty:
            return {"error": f"No hay datos suficientes para predecir desde {fecha} en {codigo}"}
    row = f.tail(1)
    proba = _model().predict_proba(row[FEATURES].to_numpy())[0]
    base = row.iloc[0]["fecha"]
    return {
        "estacion": codigo,
        "fecha_base": f"{base:%Y-%m-%d}",
        "fecha_predicha": f"{base + pd.Timedelta(days=1):%Y-%m-%d}",
        "categoria_predicha": STATES[int(proba.argmax())],
        "probabilidades": {s: round(float(p), 3) for s, p in zip(STATES, proba)},
        "modelo": "Regresión logística multinomial propia (accuracy 0.77 en prueba temporal)",
        "fuente": FUENTE_DATOS,
    }


def search_docs(query: str, k: int = 4) -> dict:
    from src.rag import search  # import diferido: carga el índice solo si se usa

    return {"resultados": search(query, k=k)}


TOOLS = {
    "list_stations": list_stations,
    "get_pm25": get_pm25,
    "predict_tomorrow": predict_tomorrow,
    "search_docs": search_docs,
}

_ESTACION = {
    "type": "string",
    "description": "Código de estación SIATA (ej. ITA-CJUS) o nombre/municipio (ej. Bello)",
}
_FECHA = {"type": "string", "description": "Fecha AAAA-MM-DD (opcional; por defecto el último dato)"}

TOOL_SCHEMAS = [
    {
        "type": "function",
        "function": {
            "name": "list_stations",
            "description": "Lista las estaciones de calidad del aire de SIATA disponibles.",
            "parameters": {"type": "object", "properties": {}},
        },
    },
    {
        "type": "function",
        "function": {
            "name": "get_pm25",
            "description": "Consulta el PM2.5 promedio diario medido en una estación y su categoría ICA.",
            "parameters": {
                "type": "object",
                "properties": {"estacion": _ESTACION, "fecha": _FECHA},
                "required": ["estacion"],
            },
        },
    },
    {
        "type": "function",
        "function": {
            "name": "predict_tomorrow",
            "description": "Predice la categoría ICA de PM2.5 del día siguiente en una estación con el modelo propio.",
            "parameters": {
                "type": "object",
                "properties": {"estacion": _ESTACION, "fecha": _FECHA},
                "required": ["estacion"],
            },
        },
    },
    {
        "type": "function",
        "function": {
            "name": "search_docs",
            "description": (
                "Busca en la normativa y guías (Res. 2254/2017, POECA del AMVA, OMS) "
                "para responder sobre normas, episodios críticos y recomendaciones de salud."
            ),
            "parameters": {
                "type": "object",
                "properties": {"query": {"type": "string", "description": "Pregunta o tema a buscar"}},
                "required": ["query"],
            },
        },
    },
]
