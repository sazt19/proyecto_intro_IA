"""Configuración central: rutas y credenciales.

La clave de NVIDIA se lee de .env (local) o de st.secrets (Streamlit Cloud).
Nunca se escribe en el código ni se sube al repositorio.
"""
import os
from pathlib import Path

from dotenv import load_dotenv

ROOT = Path(__file__).resolve().parent.parent
DATA_RAW = ROOT / "data" / "raw"
DATA_PROCESSED = ROOT / "data" / "processed"
DOCS_DIR = ROOT / "docs"
INDEX_DIR = ROOT / "data" / "index"

load_dotenv(ROOT / ".env")

NVIDIA_BASE_URL = "https://integrate.api.nvidia.com/v1"
DEFAULT_MODEL = "nvidia/nemotron-3.5-lightning-30b-a3b"


def _secret(name: str, default: str | None = None) -> str | None:
    value = os.getenv(name)
    if value:
        return value
    try:
        import streamlit as st

        return st.secrets.get(name, default)
    except Exception:
        return default


def nvidia_api_key() -> str | None:
    return _secret("NVIDIA_API_KEY")


def nvidia_model() -> str:
    return _secret("NVIDIA_MODEL", DEFAULT_MODEL) or DEFAULT_MODEL


# Categorías ICA para PM2.5 (Resolución 2254 de 2017, promedio 24 h, µg/m³)
ICA_PM25 = [
    (0, 12, "Buena", "#00e400"),
    (12, 37, "Aceptable", "#ffff00"),
    (37, 55, "Dañina para grupos sensibles", "#ff7e00"),
    (55, 150, "Dañina para la salud", "#ff0000"),
    (150, 250, "Muy dañina para la salud", "#8f3f97"),
    (250, 10_000, "Peligrosa", "#7e0023"),
]

# Clases del modelo, agrupando categorías ICA (en el año de datos solo hubo
# 2 días > 55 µg/m³, así que "dañina" agrupa todo lo que supera 37 µg/m³)
STATES = ["buena", "aceptable", "danina"]


def ica_category(pm25: float) -> tuple[str, str]:
    for lo, hi, name, color in ICA_PM25:
        if pm25 <= hi:
            return name, color
    return ICA_PM25[-1][2], ICA_PM25[-1][3]


def state_of(pm25: float) -> int:
    """0 = buena (<=12), 1 = aceptable (12-37], 2 = dañina (>37)."""
    if pm25 <= 12:
        return 0
    if pm25 <= 37:
        return 1
    return 2
