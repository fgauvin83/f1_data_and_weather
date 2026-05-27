"""
F1 Météo Pipeline — Orchestrateur principal
============================================

Usage :
  # Mode automatique (détecte la dernière session)
  python pipeline.py

  # Mode manuel sur un GP spécifique
  python pipeline.py --gp "Monaco" --year 2024 --session R

  # Uniquement la prévision pré-course (vendredi)
  python pipeline.py --mode prerace --gp "Silverstone" --date 2025-07-06

  # Climatologie historique d'un circuit
  python pipeline.py --mode clim --gp "Spa" --lat 50.437 --lon 5.971 --month 8
"""

import argparse
import json
import sys
from datetime import datetime
from pathlib import Path

# Ajout du dossier parent au path
sys.path.insert(0, str(Path(__file__).parent))

from data.fastf1_fetcher import (
    get_latest_session, load_session,
    get_weather_df, get_top_drivers_laps
)
from data.openmeteo_fetcher import (
    get_multi_model_forecast,
    get_historical_race_conditions,
    get_circuit_climatology,
)
from plots.generator import (
    plot_prerace_forecast,
    plot_session_debrief,
    plot_track_drying,
    plot_circuit_climatology,
)
from posts.caption_generator import (
    build_stats_from_session,
    generate_caption,
    format_caption_for_review,
)

# ── Circuits F1 — coordonnées GPS ────────────────────────────────────────────

CIRCUITS = {
    "Bahrain":       (26.0325, 50.5106),
    "Saudi Arabia":  (21.6319, 39.1044),
    "Australia":     (-37.8497, 144.9680),
    "Japan":         (34.8431, 136.5407),
    "China":         (31.3389, 121.2197),
    "Miami":         (25.9581, -80.2389),
    "Emilia Romagna":(44.3439, 11.7167),
    "Monaco":        (43.7347, 7.4206),
    "Canada":        (45.5000, -73.5228),
    "Spain":         (41.5700, 2.2611),
    "Austria":       (47.2197, 14.7647),
    "Great Britain": (52.0786, -1.0169),
    "Hungary":       (47.5789, 19.2486),
    "Belgium":       (50.4372, 5.9714),
    "Netherlands":   (52.3888, 4.5409),
    "Italy":         (45.6156, 9.2811),
    "Azerbaijan":    (40.3725, 49.8533),
    "Singapore":     (1.2914, 103.8640),
    "United States": (30.1328, -97.6411),
    "Mexico":        (19.4042, -99.0907),
    "Brazil":        (-23.7036, -46.6997),
    "Las Vegas":     (36.1147, -115.1730),
    "Qatar":         (25.4900, 51.4542),
    "Abu Dhabi":     (24.4672, 54.6031),
}


def get_circuit_coords(gp_name: str, lat: float = None, lon: float = None):
    """Retourne les coords GPS d'un circuit."""
    if lat and lon:
        return lat, lon
    for key, coords in CIRCUITS.items():
        if key.lower() in gp_name.lower() or gp_name.lower() in key.lower():
            return coords
    raise ValueError(f"Circuit '{gp_name}' non trouvé. Spécifie --lat et --lon.")


def run_postrace(gp: str, year: int, session_type: str,
                 output_dir: Path, with_caption: bool = True):
    """Mode post-course : charge la session, génère les graphes + brouillon."""

    print(f"\n{'─'*50}")
    print(f"  POST-COURSE : {gp} {year} [{session_type}]")
    print(f"{'─'*50}")

    print("\n[1/4] Chargement des données FastF1...")
    session = load_session(year, gp, session_type)
    weather_df = get_weather_df(session)
    drivers = get_top_drivers_laps(session, n=3)

    print(f"  → Météo : {len(weather_df)} points | Pilotes : {list(drivers.keys())}")

    print("\n[2/4] Génération des graphes...")
    output_dir.mkdir(parents=True, exist_ok=True)

    p1 = plot_session_debrief(
        weather_df, session.laps, drivers, session_type, gp, output_dir
    )
    p2 = plot_track_drying(
        weather_df, session.laps, drivers, gp, output_dir
    )

    print("\n[3/4] Résumé des stats météo")
    print(f"  Air moy.   : {weather_df['AirTemp'].mean():.1f}°C")
    print(f"  Piste moy. : {weather_df['TrackTemp'].mean():.1f}°C")
    print(f"  Vent max   : {weather_df['WindSpeed'].max():.1f} km/h")
    print(f"  Pluie      : {'OUI' if weather_df['Rainfall'].astype(float).max() > 0 else 'NON'}")

    if with_caption:
        print("\n[4/4] Génération du brouillon (Claude API)...")
        try:
            stats = build_stats_from_session(session, weather_df, drivers)
            caption = generate_caption(stats, platform="twitter", lang="fr")
            print(format_caption_for_review(caption, "twitter"))

            # Sauvegarde du brouillon
            draft_path = output_dir / f"draft_{gp.replace(' ','_').lower()}.json"
            with open(draft_path, "w", encoding="utf-8") as f:
                json.dump(caption, f, ensure_ascii=False, indent=2)
            print(f"  Brouillon sauvegardé : {draft_path.name}")
        except Exception as e:
            print(f"  (Claude API non configurée — skip caption : {e})")

    return [p1, p2]


def run_prerace(gp: str, race_date: str, lat: float, lon: float,
                output_dir: Path, with_caption: bool = True):
    """Mode pré-course (vendredi) : prévisions multi-modèles."""

    print(f"\n{'─'*50}")
    print(f"  PRÉ-COURSE : {gp} | {race_date}")
    print(f"{'─'*50}")

    print("\n[1/3] Téléchargement des prévisions multi-modèles...")
    models = get_multi_model_forecast(lat, lon, race_date)
    print(f"  → {len(models)} modèles disponibles : {list(models.keys())}")

    print("\n[2/3] Génération du graphe...")
    output_dir.mkdir(parents=True, exist_ok=True)
    p = plot_prerace_forecast(models, gp, race_date, output_dir)

    if with_caption:
        print("\n[3/3] Brouillon caption...")
        # Pour la prévision, on génère un prompt simplifié sans FastF1
        first_model = list(models.values())[0]
        peak_rain = first_model["rain_prob_pct"].max() if "rain_prob_pct" in first_model else 0
        avg_temp  = first_model["temp_c"].mean() if "temp_c" in first_model else 20

        caption_info = {
            "hook": f"GP {gp} dimanche — {peak_rain:.0f}% de risque pluie au pic",
            "caption": (
                f"🌦️ GP {gp} — météo du dimanche\n\n"
                f"• Temp. max : {avg_temp:.0f}°C\n"
                f"• Risque pluie (pic) : {peak_rain:.0f}%\n"
                f"• Modèles : {', '.join(models.keys())}\n\n"
                f"[COMPLÉTER avec l'analyse des divergences entre modèles]"
            ),
            "hashtags": ["F1", gp.replace(" ",""), "Météo", "Formula1"],
            "key_numbers": [f"{peak_rain:.0f}% pluie", f"{avg_temp:.0f}°C"]
        }
        print(format_caption_for_review(caption_info, "twitter"))

    return [p]


def run_climatology(gp: str, lat: float, lon: float, month: int,
                    output_dir: Path):
    """Mode climatologie : analyse historique sur 15 ans."""

    import calendar
    month_label = f"{calendar.month_name[month]} (2010-2024)"

    print(f"\n{'─'*50}")
    print(f"  CLIMATOLOGIE : {gp} | {month_label}")
    print(f"{'─'*50}")

    print("\n[1/2] Téléchargement données historiques (patience ~30s)...")
    clim_df = get_circuit_climatology(lat, lon, month, years=range(2010, 2025))

    if clim_df.empty:
        print("  ✗ Aucune donnée disponible.")
        return []

    print(f"  → {len(clim_df)} points horaires chargés")

    print("\n[2/2] Génération du graphe climatologique...")
    output_dir.mkdir(parents=True, exist_ok=True)
    p = plot_circuit_climatology(clim_df, gp, month_label, output_dir)

    return [p]


def run_auto(output_dir: Path):
    """Mode automatique : détecte la dernière session et lance le pipeline."""
    print("\n[AUTO] Recherche de la dernière session disponible...")
    info = get_latest_session()
    if not info:
        print("  Aucune session récente trouvée.")
        return

    print(f"  → {info['gp']} {info['year']} (round {info['round']})")
    lat, lon = get_circuit_coords(info["gp"], info.get("lat"), info.get("lon"))
    run_postrace(info["gp"], info["year"], "R", output_dir)


# ── CLI ───────────────────────────────────────────────────────────────────────

def main():
    parser = argparse.ArgumentParser(description="F1 Météo Pipeline")
    parser.add_argument("--mode",    choices=["auto","postrace","prerace","clim"],
                        default="auto")
    parser.add_argument("--gp",     type=str, help="Nom du GP (ex: 'Monaco')")
    parser.add_argument("--year",   type=int, default=datetime.now().year)
    parser.add_argument("--session",type=str, default="R",
                        help="R, Q, FP1, FP2, FP3")
    parser.add_argument("--date",   type=str, help="YYYY-MM-DD (mode prerace)")
    parser.add_argument("--lat",    type=float)
    parser.add_argument("--lon",    type=float)
    parser.add_argument("--month",  type=int, help="Mois 1-12 (mode clim)")
    parser.add_argument("--no-caption", action="store_true")
    parser.add_argument("--output", type=str, default="output")
    args = parser.parse_args()

    output_dir = Path(args.output)

    if args.mode == "auto":
        run_auto(output_dir)

    elif args.mode == "postrace":
        if not args.gp:
            print("Erreur : --gp requis en mode postrace")
            sys.exit(1)
        run_postrace(args.gp, args.year, args.session, output_dir,
                     with_caption=not args.no_caption)

    elif args.mode == "prerace":
        if not (args.gp and args.date):
            print("Erreur : --gp et --date requis en mode prerace")
            sys.exit(1)
        lat, lon = get_circuit_coords(args.gp, args.lat, args.lon)
        run_prerace(args.gp, args.date, lat, lon, output_dir,
                    with_caption=not args.no_caption)

    elif args.mode == "clim":
        if not (args.gp and args.month):
            print("Erreur : --gp et --month requis en mode clim")
            sys.exit(1)
        lat, lon = get_circuit_coords(args.gp, args.lat, args.lon)
        run_climatology(args.gp, lat, lon, args.month, output_dir)


if __name__ == "__main__":
    main()