"""
Caption Generator
=================
Utilise l'API Claude pour rédiger le commentaire social media
à partir des données brutes. Tu valides avant de poster.
"""

import anthropic
import json
from dataclasses import dataclass
from typing import Literal


@dataclass
class SessionStats:
    """Résumé statistique d'une session, passé au LLM."""
    gp_name: str
    session_type: str          # "Race", "Qualifying", etc.
    year: int

    # Météo
    avg_air_temp: float
    avg_track_temp: float
    max_wind_speed: float
    had_rain: bool
    rain_duration_min: float   # 0 si pas de pluie

    # Sport
    winner: str
    winner_team: str
    fastest_lap_driver: str
    fastest_lap_time: str      # "1:23.456"

    # Insights météo calculés
    temp_delta_track_air: float   # TrackTemp - AirTemp
    wind_direction_deg: float
    key_insight: str              # ex: "La piste a séché en 8 tours après l'averse au T35"


def build_stats_from_session(session, weather_df, drivers: dict) -> SessionStats:
    """Construit un SessionStats depuis les objets FastF1."""
    w = weather_df
    results = session.results

    winner_row = results.iloc[0]
    fl_row = results.sort_values("FastestLapTime").iloc[0]

    had_rain = bool(w["Rainfall"].astype(float).max() > 0)
    rain_time = float((w["Rainfall"].astype(float) > 0).sum())  # minutes approx

    # Insight automatique
    if had_rain:
        rain_start = w[w["Rainfall"].astype(float) > 0]["Time"].iloc[0]
        insight = f"Pluie déclenchée à la minute {rain_start:.0f} de session"
    else:
        delta = float(w["TrackTemp"].mean() - w["AirTemp"].mean())
        insight = f"Écart piste/air moyen de {delta:.1f}°C — piste {('très chaude' if delta > 15 else 'tempérée')}"

    fl_time = fl_row.get("FastestLapTime", "N/A")
    if hasattr(fl_time, "total_seconds"):
        s = fl_time.total_seconds()
        fl_str = f"{int(s//60)}:{s%60:06.3f}"
    else:
        fl_str = str(fl_time)

    return SessionStats(
        gp_name=session.event["EventName"],
        session_type=session.name,
        year=session.event.year,
        avg_air_temp=round(float(w["AirTemp"].mean()), 1),
        avg_track_temp=round(float(w["TrackTemp"].mean()), 1),
        max_wind_speed=round(float(w["WindSpeed"].max()), 1),
        had_rain=had_rain,
        rain_duration_min=rain_time,
        winner=winner_row["Abbreviation"],
        winner_team=winner_row["TeamName"],
        fastest_lap_driver=fl_row["Abbreviation"],
        fastest_lap_time=fl_str,
        temp_delta_track_air=round(float(w["TrackTemp"].mean() - w["AirTemp"].mean()), 1),
        wind_direction_deg=round(float(w["WindDirection"].mean()), 0),
        key_insight=insight,
    )


def generate_caption(
    stats: SessionStats,
    platform: Literal["twitter", "substack"] = "twitter",
    lang: Literal["fr", "en"] = "fr",
    tone: str = "analytique et direct, comme un ingénieur qui vulgarise"
) -> dict:
    """
    Génère un brouillon de post via Claude.
    Retourne {"caption": str, "hashtags": list, "hook": str}
    """
    client = anthropic.Anthropic()

    platform_instructions = {
        "twitter": "Thread X/Twitter de 3-4 tweets max. Premier tweet = hook accrocheur. Utilise des emojis avec parcimonie. Sous 280 caractères par tweet.",
        "substack": "Introduction de newsletter (200-300 mots). Ton analytique mais accessible. Commence par un fait surprenant.",
    }

    prompt = f"""Tu es le rédacteur d'un compte data-météo dédié à la F1.
Ton angle : montrer comment la météo influence les courses, avec de vrais chiffres.
Ton style : {tone}

Voici les données brutes de la session :
```json
{json.dumps(stats.__dict__, indent=2)}
```

Génère un post {platform_instructions[platform]}

Réponds UNIQUEMENT en JSON avec cette structure :
{{
  "hook": "première phrase accrocheuse (question ou fact surprising)",
  "caption": "contenu complet du post",
  "hashtags": ["F1", "Météo", ...],
  "key_numbers": ["stat 1", "stat 2", "stat 3"]
}}

Langue : {lang}. Ne jamais inventer des données non présentes dans les stats fournies."""

    message = client.messages.create(
        model="claude-sonnet-4-20250514",
        max_tokens=800,
        messages=[{"role": "user", "content": prompt}]
    )

    raw = message.content[0].text
    # Nettoyage si le LLM ajoute des backticks
    raw = raw.strip().removeprefix("```json").removesuffix("```").strip()

    try:
        return json.loads(raw)
    except json.JSONDecodeError:
        # Fallback : retourner le texte brut
        return {"hook": "", "caption": raw, "hashtags": [], "key_numbers": []}


def format_caption_for_review(caption_data: dict, platform: str) -> str:
    """Formate le brouillon pour relecture dans le terminal."""
    sep = "─" * 60
    lines = [
        sep,
        f"  BROUILLON — {platform.upper()}",
        sep,
        f"  HOOK : {caption_data.get('hook', '')}",
        "",
        caption_data.get("caption", ""),
        "",
        f"  Hashtags : {' '.join('#' + h for h in caption_data.get('hashtags', []))}",
        "",
        f"  Chiffres clés : {' | '.join(caption_data.get('key_numbers', []))}",
        sep,
    ]
    return "\n".join(lines)