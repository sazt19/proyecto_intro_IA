"""Descarga y limpieza de datos reales.

Fuentes:
- SIATA (Sistema de Alerta Temprana de Medellín y el Valle de Aburrá):
  PM2.5 horario por estación, https://siata.gov.co/EntregaData1/Datos_SIATA_Aire_pm25.json
- Open-Meteo Historical Weather API (reanálisis ERA5): precipitación,
  viento y humedad diarios por coordenadas, https://open-meteo.com

Uso:
    python -m src.data --download   # descarga y regenera data/processed/
    python -m src.data              # solo reprocesa desde data/raw/

Los CSV de data/processed/ se suben al repo (plan B si las APIs fallan).
"""
import argparse
import json

import numpy as np
import pandas as pd
import requests

from src.config import DATA_PROCESSED, DATA_RAW

SIATA_URL = "https://siata.gov.co/EntregaData1/Datos_SIATA_Aire_pm25.json"
METEO_URL = "https://archive-api.open-meteo.com/v1/archive"
MIN_HOURS_PER_DAY = 18  # criterio de representatividad (75 % de datos horarios)

STATIONS_CSV = DATA_PROCESSED / "estaciones.csv"
DAILY_CSV = DATA_PROCESSED / "pm25_diario.csv"
METEO_CSV = DATA_PROCESSED / "meteo_diario.csv"


def download_siata() -> None:
    DATA_RAW.mkdir(parents=True, exist_ok=True)
    resp = requests.get(SIATA_URL, timeout=180)
    resp.raise_for_status()
    (DATA_RAW / "siata_pm25.json").write_bytes(resp.content)


def process_siata() -> tuple[pd.DataFrame, pd.DataFrame]:
    raw = json.loads((DATA_RAW / "siata_pm25.json").read_text(encoding="utf-8"))
    stations, rows = [], []
    for st in raw:
        name = st["nombre"]
        if name.startswith("_OFF"):  # estaciones fuera de operación
            continue
        code = st["nombreCorto"]
        stations.append(
            {"codigo": code, "nombre": name, "lat": st["latitud"], "lon": st["longitud"]}
        )
        for d in st["datos"]:
            rows.append((code, d["fecha"], d["valor"]))

    hourly = pd.DataFrame(rows, columns=["codigo", "fecha", "pm25"])
    hourly["fecha"] = pd.to_datetime(hourly["fecha"])
    hourly["pm25"] = pd.to_numeric(hourly["pm25"], errors="coerce")
    hourly.loc[(hourly["pm25"] < 0) | (hourly["pm25"] > 500), "pm25"] = np.nan
    hourly["dia"] = hourly["fecha"].dt.normalize()

    daily = (
        hourly.groupby(["codigo", "dia"])["pm25"]
        .agg(pm25="mean", horas="count")
        .reset_index()
    )
    daily = daily[daily["horas"] >= MIN_HOURS_PER_DAY].drop(columns="horas")
    daily["pm25"] = daily["pm25"].round(2)

    stations = pd.DataFrame(stations)
    stations = stations[stations["codigo"].isin(daily["codigo"].unique())]
    return stations.reset_index(drop=True), daily.rename(columns={"dia": "fecha"})


def download_meteo(stations: pd.DataFrame, start: str, end: str) -> pd.DataFrame:
    frames = []
    for _, st in stations.iterrows():
        params = {
            "latitude": st["lat"],
            "longitude": st["lon"],
            "start_date": start,
            "end_date": end,
            "daily": "precipitation_sum,wind_speed_10m_max,relative_humidity_2m_mean",
            "timezone": "America/Bogota",
        }
        resp = requests.get(METEO_URL, params=params, timeout=60)
        resp.raise_for_status()
        d = resp.json()["daily"]
        frames.append(
            pd.DataFrame(
                {
                    "codigo": st["codigo"],
                    "fecha": pd.to_datetime(d["time"]),
                    "lluvia_mm": d["precipitation_sum"],
                    "viento_max_kmh": d["wind_speed_10m_max"],
                    "humedad_pct": d["relative_humidity_2m_mean"],
                }
            )
        )
    return pd.concat(frames, ignore_index=True)


def load_stations() -> pd.DataFrame:
    return pd.read_csv(STATIONS_CSV)


def load_daily() -> pd.DataFrame:
    """PM2.5 diario unido con meteorología, ordenado por estación y fecha."""
    pm = pd.read_csv(DAILY_CSV, parse_dates=["fecha"])
    met = pd.read_csv(METEO_CSV, parse_dates=["fecha"])
    df = pm.merge(met, on=["codigo", "fecha"], how="left")
    return df.sort_values(["codigo", "fecha"]).reset_index(drop=True)


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--download", action="store_true", help="descargar de las APIs")
    args = parser.parse_args()

    if args.download:
        print("Descargando SIATA...")
        download_siata()

    stations, daily = process_siata()
    DATA_PROCESSED.mkdir(parents=True, exist_ok=True)
    stations.to_csv(STATIONS_CSV, index=False)
    daily.to_csv(DAILY_CSV, index=False, date_format="%Y-%m-%d")
    print(f"{len(stations)} estaciones, {len(daily)} días-estación válidos")

    if args.download or not METEO_CSV.exists():
        start = daily["fecha"].min().strftime("%Y-%m-%d")
        end = daily["fecha"].max().strftime("%Y-%m-%d")
        print(f"Descargando Open-Meteo {start} a {end}...")
        meteo = download_meteo(stations, start, end)
        meteo.to_csv(METEO_CSV, index=False, date_format="%Y-%m-%d")
    print("Listo:", DATA_PROCESSED)


if __name__ == "__main__":
    main()
