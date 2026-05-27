"""
Open-Meteo Fetcher
==================
Prévisions pré-course et données historiques horaires.
Pas de clé API requise.
"""

import requests
import pandas as pd
from datetime import datetime, timedelta


BASE_ARCHIVE = "https://archive-api.open-meteo.com/v1/archive"
BASE_FORECAST = "https://api.open-meteo.com/v1/forecast"

VARIABLES_HOURLY = [
    "temperature_2m",
    "precipitation",
    "precipitation_probability",
    "wind_speed_10m",
    "wind_direction_10m",
    "relative_humidity_2m",
    "surface_pressure",
    "cloud_cover",
]


def _parse_response(data: dict) -> pd.DataFrame:
    """Parse la réponse Open-Meteo en DataFrame propre."""
    hourly = data.get("hourly", {})
    df = pd.DataFrame(hourly)
    df["time"] = pd.to_datetime(df["time"])
    df = df.rename(columns={
        "temperature_2m": "temp_c",
        "precipitation": "rain_mm",
        "precipitation_probability": "rain_prob_pct",
        "wind_speed_10m": "wind_kmh",
        "wind_direction_10m": "wind_dir_deg",
        "relative_humidity_2m": "humidity_pct",
        "surface_pressure": "pressure_hpa",
        "cloud_cover": "cloud_pct",
    })
    return df


def get_race_day_forecast(lat: float, lon: float, race_date: str) -> pd.DataFrame:
    """
    Prévisions heure par heure pour le jour de la course.
    race_date : "YYYY-MM-DD"
    Retourne un DataFrame avec les colonnes météo.
    """
    params = {
        "latitude": lat,
        "longitude": lon,
        "hourly": ",".join(VARIABLES_HOURLY),
        "start_date": race_date,
        "end_date": race_date,
        "wind_speed_unit": "kmh",
        "timezone": "auto",
        "models": "best_match",  # meilleur modèle disponible (AROME en Europe)
    }
    r = requests.get(BASE_FORECAST, params=params, timeout=10)
    r.raise_for_status()
    df = _parse_response(r.json())
    df["source"] = "open-meteo forecast"
    return df


def get_multi_model_forecast(lat: float, lon: float, race_date: str) -> dict[str, pd.DataFrame]:
    """
    Compare plusieurs modèles NWP pour la même journée.
    Retourne un dict {model_name: DataFrame}
    Idéal pour le contenu "qui avait raison ?"
    """
    models = {
        "GFS (US)": "gfs_seamless",
        "ICON (DE)": "icon_seamless",
        "AROME (FR)": "meteofrance_seamless",
        "IFS (EU)": "ecmwf_ifs025",
    }
    results = {}
    for name, model_id in models.items():
        try:
            params = {
                "latitude": lat,
                "longitude": lon,
                "hourly": "temperature_2m,precipitation,precipitation_probability,wind_speed_10m",
                "start_date": race_date,
                "end_date": race_date,
                "wind_speed_unit": "kmh",
                "timezone": "auto",
                "models": model_id,
            }
            r = requests.get(BASE_FORECAST, params=params, timeout=10)
            if r.status_code == 200:
                df = _parse_response(r.json())
                df["model"] = name
                results[name] = df
        except Exception as e:
            print(f"  Modèle {name} indisponible : {e}")
    return results


def get_historical_race_conditions(lat: float, lon: float,
                                    start_date: str, end_date: str) -> pd.DataFrame:
    """
    Données historiques horaires pour une période donnée.
    Utilise ERA5 en backend pour le passé (>5 jours).
    start_date, end_date : "YYYY-MM-DD"
    """
    params = {
        "latitude": lat,
        "longitude": lon,
        "hourly": ",".join(VARIABLES_HOURLY),
        "start_date": start_date,
        "end_date": end_date,
        "wind_speed_unit": "kmh",
        "timezone": "UTC",
    }
    r = requests.get(BASE_ARCHIVE, params=params, timeout=15)
    r.raise_for_status()
    df = _parse_response(r.json())
    df["source"] = "open-meteo archive (ERA5)"
    return df


def get_circuit_climatology(lat: float, lon: float,
                             month: int, years: range = range(2010, 2025)) -> pd.DataFrame:
    """
    Climatologie d'un circuit sur N années pour un mois donné.
    Permet de construire : "Spa est-il vraiment le circuit le plus pluvieux ?"
    """
    all_dfs = []
    for year in years:
        # Premier et dernier jour du mois
        start = f"{year}-{month:02d}-01"
        # Dernier jour du mois (approx)
        if month == 12:
            end = f"{year}-{month:02d}-31"
        else:
            end = f"{year}-{month+1:02d}-01"
        try:
            df = get_historical_race_conditions(lat, lon, start, end)
            df["year"] = year
            all_dfs.append(df)
        except Exception:
            pass
    if not all_dfs:
        return pd.DataFrame()
    return pd.concat(all_dfs, ignore_index=True)