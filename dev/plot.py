"""
Plots Generator
===============
Génère tous les graphes du pipeline.
Thème sombre F1, export PNG haute résolution.
"""
 
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import matplotlib.gridspec as gridspec
import matplotlib.ticker as mticker
from matplotlib.collections import LineCollection
from matplotlib.patches import FancyArrowPatch
import numpy as np
import pandas as pd
from pathlib import Path
from datetime import datetime
 
# ── Thème global ──────────────────────────────────────────────────────────────
 
DARK_BG   = "#0f0f0f"
PANEL_BG  = "#1a1a1a"
GRID_COL  = "#2a2a2a"
TEXT_COL  = "#e0e0e0"
SUB_COL   = "#888888"
ACCENT    = "#FF1801"   # Rouge F1
 
plt.rcParams.update({
    "figure.facecolor": DARK_BG,
    "axes.facecolor":   PANEL_BG,
    "axes.edgecolor":   "#333",
    "axes.labelcolor":  SUB_COL,
    "xtick.color":      SUB_COL,
    "ytick.color":      SUB_COL,
    "grid.color":       GRID_COL,
    "grid.linewidth":   0.5,
    "text.color":       TEXT_COL,
    "font.family":      "monospace",
    "font.size":        10,
    "axes.titlesize":   11,
    "axes.titlecolor":  TEXT_COL,
    "axes.spines.top":  False,
    "axes.spines.right":False,
})
 
 
def _save(fig, output_dir: Path, filename: str) -> Path:
    path = output_dir / filename
    fig.savefig(path, dpi=180, bbox_inches="tight", facecolor=DARK_BG)
    plt.close(fig)
    print(f"  ✓ Sauvegardé : {path.name}")
    return path
 
 
# ── 1. Prévision pré-course multi-modèles ────────────────────────────────────
 
def plot_prerace_forecast(models_data: dict, gp_name: str,
                          race_date: str, output_dir: Path) -> Path:
    """
    Graphe vendredi avant la course : comparaison multi-modèles.
    models_data : dict {"GFS": df, "ICON": df, ...}
    """
    colors = ["#1E80FF", "#FF6B35", "#1D9E75", "#A855F7", "#F59E0B"]
    fig, axes = plt.subplots(2, 1, figsize=(12, 7), sharex=True)
    fig.suptitle(
        f"PRÉVISION MÉTÉO — {gp_name.upper()}  |  {race_date}",
        fontsize=13, fontweight="bold", y=0.97, color=ACCENT
    )
 
    ax_rain, ax_temp = axes
 
    for i, (model_name, df) in enumerate(models_data.items()):
        c = colors[i % len(colors)]
        hours = df["time"].dt.hour
 
        # Précipitations
        if "rain_prob_pct" in df.columns:
            ax_rain.plot(df["time"], df["rain_prob_pct"],
                         color=c, lw=1.8, label=model_name, marker="o",
                         markersize=3, markevery=3)
        # Température
        if "temp_c" in df.columns:
            ax_temp.plot(df["time"], df["temp_c"],
                         color=c, lw=1.8, label=model_name)
 
    # Zones dangereuses
    ax_rain.axhspan(50, 100, alpha=0.05, color="red", zorder=0)
    ax_rain.axhline(50, color="red", lw=0.8, ls="--", alpha=0.4, label="Seuil 50%")
 
    ax_rain.set_ylabel("Prob. précipitations (%)")
    ax_rain.set_ylim(0, 105)
    ax_rain.set_title("Probabilité de pluie — comparaison modèles", pad=6)
    ax_rain.legend(loc="upper right", fontsize=9, framealpha=0.2,
                   facecolor=PANEL_BG, edgecolor="#444")
    ax_rain.grid(True)
 
    ax_temp.set_ylabel("Température (°C)")
    ax_temp.set_title("Température de l'air", pad=6)
    ax_temp.grid(True)
    ax_temp.xaxis.set_major_formatter(matplotlib.dates.DateFormatter("%Hh"))
    ax_temp.xaxis.set_major_locator(matplotlib.dates.HourLocator(interval=2))
    plt.xticks(rotation=30, ha="right")
 
    fig.text(0.98, 0.01, f"Sources: Open-Meteo | @F1MeteoData | {datetime.now():%d/%m/%Y}",
             ha="right", fontsize=8, color=SUB_COL)
 
    return _save(fig, output_dir, f"prerace_{gp_name.replace(' ','_').lower()}.png")
 
 
# ── 2. Débrief session — météo + telemetrie croisées ─────────────────────────
 
def plot_session_debrief(weather_df: pd.DataFrame, laps_df: pd.DataFrame,
                         drivers: dict, session_name: str,
                         gp_name: str, output_dir: Path) -> Path:
    """
    Graphe post-session : vitesse de séchage de piste, temp piste/air,
    delta pneus. Le contenu "retour de course".
    """
    fig = plt.figure(figsize=(14, 10))
    fig.suptitle(
        f"DÉBRIEF MÉTÉO — {gp_name.upper()}  [{session_name}]",
        fontsize=13, fontweight="bold", y=0.98, color=ACCENT
    )
    gs = gridspec.GridSpec(3, 2, hspace=0.5, wspace=0.35,
                           left=0.07, right=0.97, top=0.93, bottom=0.07)
 
    ax_temp   = fig.add_subplot(gs[0, :])   # Air + TrackTemp
    ax_rain   = fig.add_subplot(gs[1, 0])   # Pluie
    ax_wind   = fig.add_subplot(gs[1, 1])   # Vent
    ax_laps   = fig.add_subplot(gs[2, :])   # Temps au tour par pilote
 
    t = weather_df["Time"]  # minutes
 
    # ── Températures
    ax_temp.plot(t, weather_df["AirTemp"], color="#1E80FF", lw=1.6,
                 label="Air (°C)")
    ax_temp.plot(t, weather_df["TrackTemp"], color="#FF6B35", lw=1.6,
                 label="Piste (°C)")
    ax_temp.fill_between(t, weather_df["AirTemp"], weather_df["TrackTemp"],
                         alpha=0.08, color="#FF6B35")
    ax_temp.set_title("Température air & piste", pad=5)
    ax_temp.set_ylabel("°C")
    ax_temp.legend(fontsize=9, framealpha=0.2, facecolor=PANEL_BG, edgecolor="#444")
    ax_temp.grid(True)
 
    # ── Pluie
    rain = weather_df["Rainfall"].astype(float)
    ax_rain.fill_between(t, rain, alpha=0.6, color="#1E80FF", step="mid")
    ax_rain.set_title("Précipitations", pad=5)
    ax_rain.set_ylabel("mm")
    ax_rain.set_xlabel("Minutes depuis début session")
    ax_rain.grid(True, axis="x")
 
    # ── Vent
    ax_wind.plot(t, weather_df["WindSpeed"], color="#1D9E75", lw=1.5)
    ax_wind.set_title("Vitesse du vent", pad=5)
    ax_wind.set_ylabel("km/h")
    ax_wind.set_xlabel("Minutes depuis début session")
    ax_wind.grid(True)
 
    # ── Temps au tour pilotes
    if not laps_df.empty and drivers:
        for code, info in drivers.items():
            driver_laps = laps_df.pick_driver(code).pick_quicklaps()
            if driver_laps.empty:
                continue
            lap_nums = driver_laps["LapNumber"]
            lap_times = driver_laps["LapTime"].dt.total_seconds()
            ax_laps.plot(lap_nums, lap_times, color=info["color"],
                         lw=1.5, label=f"{code} (P{info['position']})",
                         marker="o", markersize=2.5)
 
    ax_laps.set_title("Temps au tour (secondes)", pad=5)
    ax_laps.set_ylabel("sec")
    ax_laps.set_xlabel("Tour")
    ax_laps.legend(fontsize=9, framealpha=0.2, facecolor=PANEL_BG, edgecolor="#444")
    ax_laps.grid(True)
 
    fig.text(0.98, 0.01,
             f"Sources: FastF1 | @F1MeteoData | {datetime.now():%d/%m/%Y}",
             ha="right", fontsize=8, color=SUB_COL)
 
    return _save(fig, output_dir, f"debrief_{gp_name.replace(' ','_').lower()}.png")
 
 
# ── 3. Graphe séchage de piste ────────────────────────────────────────────────
 
def plot_track_drying(weather_df: pd.DataFrame, laps_df: pd.DataFrame,
                      drivers: dict, gp_name: str, output_dir: Path) -> Path:
    """
    Focus sur la fenêtre de séchage : pluie + delta de temps au tour.
    LE graphe signature pour les courses sous la pluie.
    """
    fig, (ax1, ax2) = plt.subplots(2, 1, figsize=(12, 7), sharex=False)
    fig.suptitle(
        f"SÉCHAGE DE PISTE — {gp_name.upper()}",
        fontsize=13, fontweight="bold", y=0.97, color=ACCENT
    )
 
    t = weather_df["Time"]
 
    # Fond : zones de pluie
    rain_mask = weather_df["Rainfall"].astype(float) > 0
    if rain_mask.any():
        ax1.fill_between(t, 0, 1, where=rain_mask, transform=ax1.get_xaxis_transform(),
                         alpha=0.15, color="#1E80FF", label="Pluie active")
        ax2.fill_between(t, 0, 1, where=rain_mask, transform=ax2.get_xaxis_transform(),
                         alpha=0.15, color="#1E80FF")
 
    ax1.plot(t, weather_df["TrackTemp"], color="#FF6B35", lw=2, label="Temp. piste")
    ax1.plot(t, weather_df["AirTemp"],   color="#888",    lw=1.2, ls="--", label="Temp. air")
    ax1.set_ylabel("°C")
    ax1.set_title("Température de piste (proxy du grip)", pad=5)
    ax1.legend(fontsize=9, framealpha=0.2, facecolor=PANEL_BG, edgecolor="#444")
    ax1.grid(True)
 
    if not laps_df.empty and drivers:
        for code, info in drivers.items():
            laps = laps_df.pick_driver(code).pick_quicklaps()
            if laps.empty:
                continue
            ax2.scatter(laps["LapNumber"], laps["LapTime"].dt.total_seconds(),
                        color=info["color"], s=20, label=code, zorder=3)
            # Trend lissé
            if len(laps) > 4:
                from scipy.signal import savgol_filter
                smooth = savgol_filter(laps["LapTime"].dt.total_seconds(), 5, 2)
                ax2.plot(laps["LapNumber"], smooth, color=info["color"], lw=1.5, alpha=0.7)
 
    ax2.set_xlabel("Tour")
    ax2.set_ylabel("Temps au tour (sec)")
    ax2.set_title("Progression des temps au tour — corrélation avec le séchage", pad=5)
    ax2.legend(fontsize=9, framealpha=0.2, facecolor=PANEL_BG, edgecolor="#444")
    ax2.grid(True)
 
    fig.text(0.98, 0.01,
             f"Sources: FastF1 | @F1MeteoData | {datetime.now():%d/%m/%Y}",
             ha="right", fontsize=8, color=SUB_COL)
 
    return _save(fig, output_dir, f"drying_{gp_name.replace(' ','_').lower()}.png")
 
 
# ── 4. Carte climatologique (ERA5 / Open-Meteo historique) ───────────────────
 
def plot_circuit_climatology(clim_df: pd.DataFrame, circuit_name: str,
                              month_label: str, output_dir: Path) -> Path:
    """
    Distribution pluviométrique historique sur un circuit.
    Contenu evergreen parfait pour les semaines sans GP.
    """
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(13, 5))
    fig.suptitle(
        f"CLIMATOLOGIE — {circuit_name.upper()} | {month_label}",
        fontsize=13, fontweight="bold", y=1.01, color=ACCENT
    )
 
    # Précipitations journalières par année
    daily = clim_df.groupby([clim_df["time"].dt.year,
                              clim_df["time"].dt.day])["rain_mm"].sum()
    annual_rain = daily.groupby(level=0).mean().reset_index()
    annual_rain.columns = ["year", "avg_daily_rain_mm"]
 
    ax1.bar(annual_rain["year"], annual_rain["avg_daily_rain_mm"],
            color=ACCENT, alpha=0.7, width=0.8)
    z = np.polyfit(annual_rain["year"], annual_rain["avg_daily_rain_mm"], 1)
    p = np.poly1d(z)
    ax1.plot(annual_rain["year"], p(annual_rain["year"]),
             color="#1E80FF", lw=2, ls="--", label=f"Tendance ({z[0]:+.2f} mm/an)")
    ax1.set_xlabel("Année")
    ax1.set_ylabel("Pluie journalière moyenne (mm)")
    ax1.set_title("Évolution des précipitations", pad=5)
    ax1.legend(fontsize=9, framealpha=0.2, facecolor=PANEL_BG, edgecolor="#444")
    ax1.grid(True, axis="y")
 
    # Distribution température
    if "temp_c" in clim_df.columns:
        temps = clim_df["temp_c"].dropna()
        ax2.hist(temps, bins=30, color="#FF6B35", alpha=0.7, edgecolor="#333")
        ax2.axvline(temps.mean(), color="#1E80FF", lw=2,
                    label=f"Moyenne : {temps.mean():.1f}°C")
        ax2.axvline(temps.quantile(0.1), color="#888", lw=1.2, ls="--",
                    label=f"P10 : {temps.quantile(0.1):.1f}°C")
        ax2.axvline(temps.quantile(0.9), color="#888", lw=1.2, ls="--",
                    label=f"P90 : {temps.quantile(0.9):.1f}°C")
        ax2.set_xlabel("Température (°C)")
        ax2.set_ylabel("Fréquence")
        ax2.set_title("Distribution des températures", pad=5)
        ax2.legend(fontsize=9, framealpha=0.2, facecolor=PANEL_BG, edgecolor="#444")
        ax2.grid(True, axis="y")
 
    fig.text(0.98, -0.02,
             f"Sources: Open-Meteo (ERA5) | @F1MeteoData | {datetime.now():%d/%m/%Y}",
             ha="right", fontsize=8, color=SUB_COL)
 
    return _save(fig, output_dir, f"clim_{circuit_name.replace(' ','_').lower()}.png")