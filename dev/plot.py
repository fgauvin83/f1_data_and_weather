"""
Plots Generator — Version 2
============================
Thème sombre F1. Toutes les figures météo × sport.

Figures disponibles :
  CARTE CIRCUIT
  ─────────────
  plot_brake_map          Carte de freinage (circuit coloré par intensité freinage)
  plot_throttle_map       Carte d'accélération (circuit coloré par gaz)
  plot_speed_map          Carte de vitesse (circuit coloré par vitesse)
  plot_driver_diff_map    Carte de différence entre 2 pilotes (vitesse/gaz/frein)

  MÉTÉO × PERFORMANCE
  ────────────────────
  plot_tracktemp_rain_laptimes   TrackTemp + pluie + temps au tour (fenêtre de séchage)
  plot_tracktemp_degradation     TrackTemp × dégradation pneu par stint
  plot_wind_topspeed             Direction/vitesse vent × vitesse de pointe par tour
  plot_pressure_performance      Pression atmo × temps au tour (tous les GPs d'une saison)

  TÉLÉMÉTRIE SEC / HUMIDE
  ────────────────────────
  plot_dry_vs_wet_telemetry      Superposition télémétrie même pilote, sec vs humide

  DÉBRIEF SESSION
  ────────────────
  plot_session_debrief           Vue globale météo + temps au tour (existant)
  plot_prerace_forecast          Prévisions multi-modèles pré-course (existant)
  plot_circuit_climatology       Climatologie historique circuit (existant)
"""

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import matplotlib.gridspec as gridspec
import matplotlib.ticker as mticker
import matplotlib.dates as mdates
from matplotlib.collections import LineCollection
from matplotlib.patches import FancyArrowPatch
import matplotlib.colors as mcolors
import numpy as np
import pandas as pd
from pathlib import Path
from datetime import datetime
from typing import Literal

# ── Thème global ──────────────────────────────────────────────────────────────

DARK_BG  = "#0f0f0f"
PANEL_BG = "#1a1a1a"
GRID_COL = "#2a2a2a"
TEXT_COL = "#e0e0e0"
SUB_COL  = "#888888"
ACCENT   = "#FF1801"   # Rouge F1

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

CMAPS = {
    "brake":    "RdYlGn_r",   # vert=relâché → rouge=freinage fort
    "throttle": "RdYlGn",     # rouge=coupé → vert=plein gaz
    "speed":    "plasma",
    "diff":     "RdBu_r",     # bleu=pilote2 plus rapide, rouge=pilote1
    "rain":     "Blues",
    "temp":     "YlOrRd",
    "pressure": "viridis",
}


# ── Utilitaires ───────────────────────────────────────────────────────────────

def _save(fig, output_dir: Path, filename: str) -> Path:
    output_dir.mkdir(parents=True, exist_ok=True)
    path = output_dir / filename
    fig.savefig(path, dpi=180, bbox_inches="tight", facecolor=DARK_BG)
    plt.close(fig)
    print(f"  ✓ {path.name}")
    return path


def _watermark(fig, source: str):
    fig.text(0.98, 0.01,
             f"{source} | @F1MeteoData | {datetime.now():%d/%m/%Y}",
             ha="right", fontsize=7, color=SUB_COL)


def _circuit_line_collection(x, y, values, cmap: str, vmin=None, vmax=None,
                               linewidth: float = 4) -> LineCollection:
    """Crée un LineCollection coloré par 'values' pour tracer le circuit."""
    points   = np.array([x, y]).T.reshape(-1, 1, 2)
    segments = np.concatenate([points[:-1], points[1:]], axis=1)
    norm     = plt.Normalize(vmin=vmin or values.min(),
                              vmax=vmax or values.max())
    lc = LineCollection(segments, cmap=cmap, norm=norm, linewidth=linewidth,
                        capstyle="round")
    lc.set_array(values[:-1])
    return lc


def _align_pos_to_tel(tel, pos):
    """Interpole la position GPS sur l'index temporel de la télémétrie."""
    tel_t = tel["SessionTime"].dt.total_seconds().values
    pos_t = pos["SessionTime"].dt.total_seconds().values
    x = np.interp(tel_t, pos_t, pos["X"].values)
    y = np.interp(tel_t, pos_t, pos["Y"].values)
    return x, y


def _add_circuit_background(ax, x, y, alpha=0.12):
    """Trace le circuit en fond gris."""
    ax.plot(x, y, color="#444", lw=6, zorder=0, solid_capstyle="round")


def _style_map_ax(ax):
    ax.set_aspect("equal")
    ax.axis("off")
    ax.set_facecolor(PANEL_BG)


def _add_colorbar(fig, ax, lc, label: str):
    cb = fig.colorbar(lc, ax=ax, fraction=0.025, pad=0.02, shrink=0.7)
    cb.set_label(label, fontsize=8, color=SUB_COL)
    cb.ax.yaxis.set_tick_params(color=SUB_COL, labelsize=7)
    plt.setp(cb.ax.yaxis.get_ticklabels(), color=SUB_COL)


# ═══════════════════════════════════════════════════════════════════════════════
#  CARTES DE CIRCUIT
# ═══════════════════════════════════════════════════════════════════════════════

def plot_brake_map(lap, session, driver_name: str,
                   gp_name: str, output_dir: Path) -> Path:
    """
    Circuit coloré par intensité de freinage (0–100%).
    Rouge = freinage maximal, Vert = relâché.
    """
    tel = lap.get_car_data().add_distance()
    pos = lap.get_pos_data()
    x, y = _align_pos_to_tel(tel, pos)
    brake = tel["Brake"].astype(float).values

    fig, ax = plt.subplots(figsize=(10, 8))
    fig.suptitle(f"CARTE DE FREINAGE — {driver_name}  |  {gp_name}",
                 fontsize=12, fontweight="bold", color=ACCENT)

    _add_circuit_background(ax, x, y)
    lc = _circuit_line_collection(x, y, brake, CMAPS["brake"], vmin=0, vmax=1)
    ax.add_collection(lc)
    _add_colorbar(fig, ax, lc, "Freinage (0=relâché · 1=max)")
    _style_map_ax(ax)
    ax.set_xlim(x.min() - 150, x.max() + 150)
    ax.set_ylim(y.min() - 150, y.max() + 150)

    # Annotation départ
    ax.scatter(x[0], y[0], color=ACCENT, s=60, zorder=5, label="Départ")
    ax.legend(fontsize=8, framealpha=0.2, facecolor=PANEL_BG, edgecolor="#444",
              loc="upper right")

    _watermark(fig, "FastF1")
    return _save(fig, output_dir,
                 f"brake_map_{driver_name}_{gp_name.replace(' ','_').lower()}.png")


def plot_throttle_map(lap, session, driver_name: str,
                      gp_name: str, output_dir: Path) -> Path:
    """
    Circuit coloré par ouverture des gaz (0–100%).
    Vert = plein gaz, Rouge = lever de pied.
    """
    tel = lap.get_car_data().add_distance()
    pos = lap.get_pos_data()
    x, y = _align_pos_to_tel(tel, pos)
    throttle = tel["Throttle"].astype(float).values / 100.0

    fig, ax = plt.subplots(figsize=(10, 8))
    fig.suptitle(f"CARTE D'ACCÉLÉRATION — {driver_name}  |  {gp_name}",
                 fontsize=12, fontweight="bold", color=ACCENT)

    _add_circuit_background(ax, x, y)
    lc = _circuit_line_collection(x, y, throttle, CMAPS["throttle"], vmin=0, vmax=1)
    ax.add_collection(lc)
    _add_colorbar(fig, ax, lc, "Gaz (0=coupé · 1=plein gaz)")
    _style_map_ax(ax)
    ax.set_xlim(x.min() - 150, x.max() + 150)
    ax.set_ylim(y.min() - 150, y.max() + 150)

    _watermark(fig, "FastF1")
    return _save(fig, output_dir,
                 f"throttle_map_{driver_name}_{gp_name.replace(' ','_').lower()}.png")


def plot_speed_map(lap, session, driver_name: str,
                   gp_name: str, output_dir: Path) -> Path:
    """Circuit coloré par vitesse instantanée."""
    tel = lap.get_car_data().add_distance()
    pos = lap.get_pos_data()
    x, y = _align_pos_to_tel(tel, pos)
    speed = tel["Speed"].values.astype(float)

    fig, ax = plt.subplots(figsize=(10, 8))
    fig.suptitle(f"CARTE DE VITESSE — {driver_name}  |  {gp_name}",
                 fontsize=12, fontweight="bold", color=ACCENT)

    _add_circuit_background(ax, x, y)
    lc = _circuit_line_collection(x, y, speed, CMAPS["speed"])
    ax.add_collection(lc)
    _add_colorbar(fig, ax, lc, "Vitesse (km/h)")
    _style_map_ax(ax)
    ax.set_xlim(x.min() - 150, x.max() + 150)
    ax.set_ylim(y.min() - 150, y.max() + 150)

    _watermark(fig, "FastF1")
    return _save(fig, output_dir,
                 f"speed_map_{driver_name}_{gp_name.replace(' ','_').lower()}.png")


def plot_driver_diff_map(lap1, lap2, session,
                          driver1: str, driver2: str,
                          metric: Literal["speed", "throttle", "brake"],
                          gp_name: str, output_dir: Path) -> Path:
    """
    Carte de différence entre deux pilotes sur le circuit.
    metric = "speed"    → diff de vitesse (km/h)
    metric = "throttle" → diff d'ouverture des gaz (%)
    metric = "brake"    → diff de freinage (0/1)

    Rouge = driver1 supérieur, Bleu = driver2 supérieur.
    """
    tel1 = lap1.get_car_data().add_distance()
    tel2 = lap2.get_car_data().add_distance()
    pos1 = lap1.get_pos_data()
    x, y = _align_pos_to_tel(tel1, pos1)

    # Rééchantillonnage sur distance commune
    dist1 = tel1["Distance"].values
    dist2 = tel2["Distance"].values
    dist_common = np.linspace(
        max(dist1.min(), dist2.min()),
        min(dist1.max(), dist2.max()),
        min(len(dist1), len(dist2))
    )

    col_map = {"speed": "Speed", "throttle": "Throttle", "brake": "Brake"}
    col = col_map[metric]

    v1 = np.interp(dist_common, dist1, tel1[col].astype(float).values)
    v2 = np.interp(dist_common, dist2, tel2[col].astype(float).values)
    diff = v1 - v2   # positif = driver1 plus fort

    # Rééchantillonner aussi x/y sur dist_common
    x_c = np.interp(dist_common, dist1, x[:len(dist1)])
    y_c = np.interp(dist_common, dist1, y[:len(dist1)])

    labels = {"speed": "Δ Vitesse (km/h)", "throttle": "Δ Gaz (%)",
              "brake": "Δ Freinage"}
    sym    = max(abs(diff.min()), abs(diff.max()))

    fig, ax = plt.subplots(figsize=(10, 8))
    fig.suptitle(
        f"DIFFÉRENCE — {driver1} vs {driver2}  |  {metric.upper()}  |  {gp_name}",
        fontsize=11, fontweight="bold", color=ACCENT
    )

    _add_circuit_background(ax, x_c, y_c)
    lc = _circuit_line_collection(x_c, y_c, diff, CMAPS["diff"],
                                   vmin=-sym, vmax=sym, linewidth=5)
    ax.add_collection(lc)
    _add_colorbar(fig, ax, lc, f"{labels[metric]}\n← {driver2} | {driver1} →")
    _style_map_ax(ax)
    ax.set_xlim(x_c.min() - 150, x_c.max() + 150)
    ax.set_ylim(y_c.min() - 150, y_c.max() + 150)

    # Légende
    from matplotlib.patches import Patch
    legend_elements = [
        Patch(facecolor="#CC2222", label=f"{driver1} supérieur"),
        Patch(facecolor="#2255CC", label=f"{driver2} supérieur"),
    ]
    ax.legend(handles=legend_elements, fontsize=8, framealpha=0.2,
              facecolor=PANEL_BG, edgecolor="#444", loc="upper right")

    _watermark(fig, "FastF1")
    return _save(fig, output_dir,
                 f"diff_{metric}_{driver1}_vs_{driver2}_{gp_name.replace(' ','_').lower()}.png")


# ═══════════════════════════════════════════════════════════════════════════════
#  MÉTÉO × PERFORMANCE
# ═══════════════════════════════════════════════════════════════════════════════

def plot_tracktemp_rain_laptimes(weather_df: pd.DataFrame,
                                  laps_df, drivers: dict,
                                  gp_name: str, output_dir: Path) -> Path:
    """
    3 panneaux superposés :
    - Temp piste + précipitations (axe gauche/droit)
    - Temps au tour par pilote
    - Delta par rapport au meilleur tour sec

    LE graphe signature pour les courses sous la pluie.
    """
    fig = plt.figure(figsize=(14, 10))
    fig.suptitle(f"PISTE · PLUIE · PERFORMANCES — {gp_name}",
                 fontsize=13, fontweight="bold", color=ACCENT)
    gs = gridspec.GridSpec(3, 1, hspace=0.12, left=0.08, right=0.93,
                           top=0.93, bottom=0.07,
                           height_ratios=[1.2, 2, 1])

    ax_meteo = fig.add_subplot(gs[0])
    ax_laps  = fig.add_subplot(gs[1], sharex=ax_meteo)
    ax_delta = fig.add_subplot(gs[2], sharex=ax_meteo)

    t = weather_df["Time"].values  # minutes

    # ── Panneau 1 : météo ────────────────────────────────────────────────────
    ax_meteo.fill_between(t, weather_df["Rainfall"].astype(float).values,
                           alpha=0.55, color="#1E80FF", step="mid",
                           label="Précipitations")
    ax_meteo.set_ylabel("Pluie (mm)", fontsize=9, color="#1E80FF")
    ax_meteo.tick_params(axis="y", colors="#1E80FF", labelsize=8)

    ax_t2 = ax_meteo.twinx()
    ax_t2.plot(t, weather_df["TrackTemp"].values, color="#FF6B35",
               lw=1.8, label="TrackTemp")
    ax_t2.set_ylabel("Temp. piste (°C)", fontsize=9, color="#FF6B35")
    ax_t2.tick_params(axis="y", colors="#FF6B35", labelsize=8)
    ax_t2.spines["right"].set_visible(True)
    ax_t2.spines["right"].set_color("#333")

    # Zones de pluie en fond (propagées sur tous les axes)
    rain_mask = weather_df["Rainfall"].astype(float).values > 0
    for ax in [ax_meteo, ax_laps, ax_delta]:
        ax.fill_between(t, 0, 1, where=rain_mask,
                        transform=ax.get_xaxis_transform(),
                        alpha=0.07, color="#1E80FF", zorder=0)

    ax_meteo.set_title("Conditions météo", pad=4, fontsize=9, color=SUB_COL,
                        loc="left")
    ax_meteo.tick_params(labelbottom=False)
    ax_meteo.grid(True, axis="x", zorder=-1)

    # ── Panneau 2 : temps au tour ────────────────────────────────────────────
    ref_time = None
    for code, info in drivers.items():
        if laps_df is None or (hasattr(laps_df, 'empty') and laps_df.empty):
            break
        dl = laps_df.pick_driver(code).pick_quicklaps()
        if dl.empty:
            continue
        lap_min  = dl["LapNumber"].values
        lap_sec  = dl["LapTime"].dt.total_seconds().values

        ax_laps.scatter(lap_min, lap_sec, color=info["color"], s=18,
                        zorder=3, label=code)
        if len(dl) > 4:
            try:
                from scipy.signal import savgol_filter
                smooth = savgol_filter(lap_sec, min(7, len(lap_sec)//2*2+1), 2)
                ax_laps.plot(lap_min, smooth, color=info["color"],
                             lw=1.5, alpha=0.8)
            except Exception:
                pass

        if ref_time is None:
            dry_laps = dl[dl["LapTime"].dt.total_seconds() < np.percentile(lap_sec, 30)]
            if not dry_laps.empty:
                ref_time = dry_laps["LapTime"].dt.total_seconds().min()

    ax_laps.set_ylabel("Temps au tour (sec)", fontsize=9)
    ax_laps.legend(fontsize=8, framealpha=0.2, facecolor=PANEL_BG,
                   edgecolor="#444", ncol=3, loc="upper right")
    ax_laps.grid(True)
    ax_laps.tick_params(labelbottom=False)

    # ── Panneau 3 : delta vs meilleur tour sec ───────────────────────────────
    if ref_time is not None:
        for code, info in drivers.items():
            if laps_df is None or (hasattr(laps_df, 'empty') and laps_df.empty):
                break
            dl = laps_df.pick_driver(code).pick_quicklaps()
            if dl.empty:
                continue
            delta = dl["LapTime"].dt.total_seconds().values - ref_time
            ax_delta.bar(dl["LapNumber"].values, delta,
                         color=info["color"], alpha=0.6, width=0.6,
                         label=code)

    ax_delta.axhline(0, color=SUB_COL, lw=0.8, ls="--")
    ax_delta.set_ylabel("Δ vs meilleur sec (s)", fontsize=9)
    ax_delta.set_xlabel("Tour", fontsize=9)
    ax_delta.grid(True, axis="y")

    # Retire les ticks x des 2 premiers
    plt.setp(ax_meteo.get_xticklabels(), visible=False)
    plt.setp(ax_laps.get_xticklabels(), visible=False)

    _watermark(fig, "FastF1")
    return _save(fig, output_dir,
                 f"tracktemp_rain_laps_{gp_name.replace(' ','_').lower()}.png")


def plot_tracktemp_degradation(weather_df: pd.DataFrame,
                                laps_df, drivers: dict,
                                gp_name: str, output_dir: Path) -> Path:
    """
    TrackTemp × dégradation pneu par stint.
    Montre la corrélation entre chaleur de piste et perte de performance.
    """
    fig, axes = plt.subplots(1, 2, figsize=(14, 6))
    fig.suptitle(f"TEMPÉRATURE DE PISTE × DÉGRADATION PNEU — {gp_name}",
                 fontsize=12, fontweight="bold", color=ACCENT)

    ax_time  = axes[0]   # Évolution temporelle
    ax_corr  = axes[1]   # Scatter TrackTemp vs dégradation

    t = weather_df["Time"].values
    track_t = weather_df["TrackTemp"].values

    ax_time.plot(t, track_t, color="#FF6B35", lw=2, label="TrackTemp (°C)")
    ax_time.set_ylabel("Température piste (°C)", color="#FF6B35", fontsize=9)
    ax_time.tick_params(axis="y", colors="#FF6B35", labelsize=8)
    ax_time.set_xlabel("Tour / minute de session", fontsize=9)
    ax_time.grid(True)

    colors_used = []
    if laps_df is not None and not (hasattr(laps_df, 'empty') and laps_df.empty):
        ax_lap = ax_time.twinx()
        ax_lap.spines["right"].set_visible(True)
        ax_lap.spines["right"].set_color("#333")

        all_track_t_interp = []
        all_degradation    = []
        all_colors         = []

        for code, info in drivers.items():
            dl = laps_df.pick_driver(code)
            if dl.empty:
                continue

            # Identifier les stints
            stints = dl["Stint"].unique() if "Stint" in dl.columns else [1]
            for stint_id in stints:
                stint_laps = dl[dl["Stint"] == stint_id] if "Stint" in dl.columns else dl
                if len(stint_laps) < 3:
                    continue

                lap_nums = stint_laps["LapNumber"].values
                lap_sec  = stint_laps["LapTime"].dt.total_seconds().values

                # Dégradation = perte de temps par tour dans le stint
                degradation = lap_sec - lap_sec[0]  # delta vs premier tour du stint

                # TrackTemp au moment de chaque tour (interpolation sur tour ≈ minutes)
                track_at_lap = np.interp(lap_nums, t, track_t,
                                          left=track_t[0], right=track_t[-1])

                ax_lap.scatter(lap_nums, degradation, color=info["color"],
                               s=18, alpha=0.7, zorder=3)
                if len(degradation) > 3:
                    try:
                        from scipy.signal import savgol_filter
                        sm = savgol_filter(degradation, min(5, len(degradation)//2*2+1), 2)
                        ax_lap.plot(lap_nums, sm, color=info["color"], lw=1.5, alpha=0.8,
                                    label=f"{code} stint {stint_id}")
                    except Exception:
                        pass

                all_track_t_interp.extend(track_at_lap.tolist())
                all_degradation.extend(degradation.tolist())
                all_colors.extend([info["color"]] * len(degradation))

        ax_lap.set_ylabel("Dégradation (Δsec vs tour 1 stint)", fontsize=9)
        ax_lap.legend(fontsize=7, framealpha=0.2, facecolor=PANEL_BG,
                      edgecolor="#444", loc="upper left")

        # Scatter corrélation
        if all_track_t_interp:
            ax_corr.scatter(all_track_t_interp, all_degradation,
                            c=all_colors, s=20, alpha=0.6)
            # Trend line
            z = np.polyfit(all_track_t_interp, all_degradation, 1)
            p_fn = np.poly1d(z)
            x_fit = np.linspace(min(all_track_t_interp), max(all_track_t_interp), 100)
            ax_corr.plot(x_fit, p_fn(x_fit), color=ACCENT, lw=2, ls="--",
                         label=f"Tendance ({z[0]:+.3f} s/°C)")
            ax_corr.legend(fontsize=8, framealpha=0.2, facecolor=PANEL_BG,
                           edgecolor="#444")

    ax_corr.set_xlabel("TrackTemp (°C)", fontsize=9)
    ax_corr.set_ylabel("Dégradation (Δsec vs tour 1 stint)", fontsize=9)
    ax_corr.set_title("Corrélation TrackTemp × dégradation", fontsize=10, pad=5)
    ax_corr.grid(True)
    ax_time.set_title("Évolution temporelle", fontsize=10, pad=5)

    _watermark(fig, "FastF1")
    return _save(fig, output_dir,
                 f"degradation_{gp_name.replace(' ','_').lower()}.png")


def plot_wind_topspeed(weather_df: pd.DataFrame,
                        laps_df, drivers: dict,
                        gp_name: str, output_dir: Path) -> Path:
    """
    2 graphes :
    - Rose des vents colorée par vitesse de pointe moyenne (polaire)
    - Scatter vitesse vent × vitesse de pointe, par tour
    """
    fig = plt.figure(figsize=(13, 6))
    fig.suptitle(f"VENT × VITESSE DE POINTE — {gp_name}",
                 fontsize=12, fontweight="bold", color=ACCENT)

    ax_rose   = fig.add_subplot(121, projection="polar")
    ax_scatter= fig.add_subplot(122)

    wind_dir_rad = np.deg2rad(weather_df["WindDirection"].values)
    wind_spd     = weather_df["WindSpeed"].values
    t            = weather_df["Time"].values

    # ── Rose des vents ───────────────────────────────────────────────────────
    # Discrétiser en 16 secteurs
    n_sectors  = 16
    sector_w   = 2 * np.pi / n_sectors
    bins        = np.linspace(0, 2 * np.pi, n_sectors + 1)
    sector_freq = np.histogram(wind_dir_rad, bins=bins)[0]
    sector_freq = sector_freq / sector_freq.max()  # normaliser

    theta = np.linspace(0, 2 * np.pi, n_sectors, endpoint=False) + sector_w / 2
    bars  = ax_rose.bar(theta, sector_freq, width=sector_w * 0.85,
                         bottom=0, alpha=0.7, align="center",
                         color=plt.cm.plasma(sector_freq))

    ax_rose.set_theta_zero_location("N")
    ax_rose.set_theta_direction(-1)
    ax_rose.set_facecolor(PANEL_BG)
    ax_rose.set_title("Rose des vents\n(fréquence par secteur)",
                       fontsize=9, color=SUB_COL, pad=12)
    ax_rose.tick_params(colors=SUB_COL, labelsize=7)
    ax_rose.set_yticklabels([])

    # ── Scatter vent × vitesse de pointe ────────────────────────────────────
    all_x, all_y, all_c = [], [], []
    if laps_df is not None and not (hasattr(laps_df, 'empty') and laps_df.empty):
        for code, info in drivers.items():
            dl = laps_df.pick_driver(code).pick_quicklaps()
            if dl.empty:
                continue
            lap_nums = dl["LapNumber"].values

            for i, lap_num in enumerate(lap_nums):
                # Vitesse max sur ce tour (proxy vitesse de pointe)
                try:
                    lap_obj = dl[dl["LapNumber"] == lap_num].iloc[0]
                    tel = lap_obj.get_car_data()
                    top_spd = tel["Speed"].max()
                except Exception:
                    continue

                # Vent moyen au moment de ce tour
                wind_at_lap = np.interp(lap_num, t, wind_spd,
                                         left=wind_spd[0], right=wind_spd[-1])
                all_x.append(wind_at_lap)
                all_y.append(top_spd)
                all_c.append(info["color"])

    if all_x:
        ax_scatter.scatter(all_x, all_y, c=all_c, s=25, alpha=0.7, zorder=3)
        # Trend
        z = np.polyfit(all_x, all_y, 1)
        p_fn = np.poly1d(z)
        x_fit = np.linspace(min(all_x), max(all_x), 100)
        ax_scatter.plot(x_fit, p_fn(x_fit), color=ACCENT, lw=1.8, ls="--",
                         label=f"Tendance ({z[0]:+.1f} km/h par km/h vent)")
        ax_scatter.legend(fontsize=8, framealpha=0.2, facecolor=PANEL_BG,
                           edgecolor="#444")
    else:
        ax_scatter.text(0.5, 0.5, "Données de télémétrie\nnon disponibles",
                         ha="center", va="center", color=SUB_COL,
                         transform=ax_scatter.transAxes, fontsize=10)

    ax_scatter.set_xlabel("Vitesse du vent (km/h)", fontsize=9)
    ax_scatter.set_ylabel("Vitesse de pointe (km/h)", fontsize=9)
    ax_scatter.set_title("Corrélation vent × vitesse de pointe", fontsize=10, pad=5)
    ax_scatter.grid(True)

    _watermark(fig, "FastF1")
    return _save(fig, output_dir,
                 f"wind_topspeed_{gp_name.replace(' ','_').lower()}.png")


def plot_pressure_performance(sessions_data: list[dict],
                               gp_name: str, output_dir: Path) -> Path:
    """
    Pression atmosphérique × temps au tour médian, sur plusieurs sessions.
    sessions_data : liste de dicts {"gp", "pressure_hpa", "median_laptime_s", "color"}

    Permet de voir l'impact de la pression sur l'aéro (densité d'air).
    """
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(13, 5))
    fig.suptitle(f"PRESSION ATMOSPHÉRIQUE × PERFORMANCE — {gp_name}",
                 fontsize=12, fontweight="bold", color=ACCENT)

    if not sessions_data:
        for ax in [ax1, ax2]:
            ax.text(0.5, 0.5, "Données insuffisantes",
                    ha="center", va="center", color=SUB_COL,
                    transform=ax.transAxes)
        _watermark(fig, "FastF1")
        return _save(fig, output_dir,
                     f"pressure_perf_{gp_name.replace(' ','_').lower()}.png")

    df = pd.DataFrame(sessions_data)

    # ── Scatter pression × temps au tour ────────────────────────────────────
    ax1.scatter(df["pressure_hpa"], df["median_laptime_s"],
                c=df.get("color", [ACCENT]*len(df)), s=40, alpha=0.8, zorder=3)

    for _, row in df.iterrows():
        ax1.annotate(row.get("gp", ""), (row["pressure_hpa"], row["median_laptime_s"]),
                     fontsize=6, color=SUB_COL,
                     xytext=(4, 4), textcoords="offset points")

    z = np.polyfit(df["pressure_hpa"], df["median_laptime_s"], 1)
    p_fn = np.poly1d(z)
    x_fit = np.linspace(df["pressure_hpa"].min(), df["pressure_hpa"].max(), 100)
    ax1.plot(x_fit, p_fn(x_fit), color=ACCENT, lw=1.8, ls="--",
              label=f"{z[0]:+.3f} s/hPa")
    ax1.legend(fontsize=8, framealpha=0.2, facecolor=PANEL_BG, edgecolor="#444")
    ax1.set_xlabel("Pression atmosphérique (hPa)", fontsize=9)
    ax1.set_ylabel("Temps médian au tour (sec)", fontsize=9)
    ax1.set_title("Pression vs performance", fontsize=10, pad=5)
    ax1.grid(True)

    # ── Densité d'air calculée (ρ = P / RT) ─────────────────────────────────
    # T supposée constante (20°C = 293K) pour simplifier
    R_air = 287.05  # J/kg/K
    T_K   = 293.15
    df["air_density"] = (df["pressure_hpa"] * 100) / (R_air * T_K)

    ax2.scatter(df["air_density"], df["median_laptime_s"],
                c=df.get("color", [ACCENT]*len(df)), s=40, alpha=0.8, zorder=3)
    z2 = np.polyfit(df["air_density"], df["median_laptime_s"], 1)
    p2 = np.poly1d(z2)
    x2 = np.linspace(df["air_density"].min(), df["air_density"].max(), 100)
    ax2.plot(x2, p2(x2), color="#1E80FF", lw=1.8, ls="--",
              label=f"{z2[0]:+.1f} s·kg⁻¹·m³")
    ax2.legend(fontsize=8, framealpha=0.2, facecolor=PANEL_BG, edgecolor="#444")
    ax2.set_xlabel("Densité d'air estimée (kg/m³)", fontsize=9)
    ax2.set_ylabel("Temps médian au tour (sec)", fontsize=9)
    ax2.set_title("Densité d'air vs performance\n(effet sur aéro + moteur)",
                  fontsize=10, pad=5)
    ax2.grid(True)

    _watermark(fig, "FastF1 + calcul ρ = P/RT")
    return _save(fig, output_dir,
                 f"pressure_perf_{gp_name.replace(' ','_').lower()}.png")


# ═══════════════════════════════════════════════════════════════════════════════
#  TÉLÉMÉTRIE SEC / HUMIDE
# ═══════════════════════════════════════════════════════════════════════════════

def plot_dry_vs_wet_telemetry(lap_dry, lap_wet, session,
                               driver: str, gp_name: str,
                               output_dir: Path) -> Path:
    """
    Superposition de la télémétrie complète du même pilote :
    un tour sec (meilleur tour sec) vs un tour sous la pluie.

    Panneau 1 : Vitesse
    Panneau 2 : Gaz (Throttle)
    Panneau 3 : Freinage
    Panneau 4 : Carte circuit — superposition des deux trajectoires
    """
    tel_dry = lap_dry.get_car_data().add_distance()
    tel_wet = lap_wet.get_car_data().add_distance()

    pos_dry = lap_dry.get_pos_data()
    pos_wet = lap_wet.get_pos_data()

    x_dry, y_dry = _align_pos_to_tel(tel_dry, pos_dry)
    x_wet, y_wet = _align_pos_to_tel(tel_wet, pos_wet)

    color_dry = "#FF6B35"
    color_wet = "#1E80FF"

    time_dry = lap_dry["LapTime"]
    time_wet = lap_wet["LapTime"]
    t_dry_s  = time_dry.total_seconds() if hasattr(time_dry, "total_seconds") else 0
    t_wet_s  = time_wet.total_seconds() if hasattr(time_wet, "total_seconds") else 0

    fig = plt.figure(figsize=(16, 11))
    fig.suptitle(
        f"SEC vs HUMIDE — {driver}  |  {gp_name}\n"
        f"Sec : {int(t_dry_s//60)}:{t_dry_s%60:06.3f}   "
        f"Humide : {int(t_wet_s//60)}:{t_wet_s%60:06.3f}   "
        f"Écart : +{abs(t_wet_s-t_dry_s):.3f}s",
        fontsize=11, fontweight="bold", color=ACCENT
    )

    gs = gridspec.GridSpec(4, 2, hspace=0.45, wspace=0.3,
                           left=0.07, right=0.97, top=0.88, bottom=0.05)

    ax_spd = fig.add_subplot(gs[0, :])
    ax_thr = fig.add_subplot(gs[1, 0])
    ax_brk = fig.add_subplot(gs[1, 1])
    ax_map = fig.add_subplot(gs[2:, :])

    dist_dry = tel_dry["Distance"].values
    dist_wet = tel_wet["Distance"].values
    dist_max = min(dist_dry.max(), dist_wet.max())

    # ── Vitesse ──────────────────────────────────────────────────────────────
    ax_spd.plot(dist_dry, tel_dry["Speed"], color=color_dry, lw=1.6,
                label=f"Sec  ({int(t_dry_s//60)}:{t_dry_s%60:05.2f})")
    ax_spd.plot(dist_wet, tel_wet["Speed"], color=color_wet, lw=1.6,
                label=f"Humide ({int(t_wet_s//60)}:{t_wet_s%60:05.2f})",
                alpha=0.85)
    ax_spd.set_ylabel("Vitesse (km/h)", fontsize=9)
    ax_spd.set_title("Comparaison vitesse SEC / HUMIDE", fontsize=10, pad=5)
    ax_spd.legend(fontsize=9, framealpha=0.2, facecolor=PANEL_BG, edgecolor="#444",
                  loc="upper right")
    ax_spd.grid(True)
    ax_spd.set_xlim(0, dist_max)

    # Différence de vitesse en fond
    d_common = np.linspace(0, dist_max, 500)
    spd_d = np.interp(d_common, dist_dry, tel_dry["Speed"].values)
    spd_w = np.interp(d_common, dist_wet, tel_wet["Speed"].values)
    ax_spd.fill_between(d_common, spd_d, spd_w,
                         where=(spd_d > spd_w), alpha=0.12, color=color_dry,
                         label=f"+ rapide SEC")
    ax_spd.fill_between(d_common, spd_d, spd_w,
                         where=(spd_d < spd_w), alpha=0.12, color=color_wet)

    # ── Throttle ─────────────────────────────────────────────────────────────
    ax_thr.plot(dist_dry, tel_dry["Throttle"], color=color_dry, lw=1.2)
    ax_thr.plot(dist_wet, tel_wet["Throttle"], color=color_wet, lw=1.2, alpha=0.85)
    ax_thr.set_ylabel("%", fontsize=9)
    ax_thr.set_title("Gaz", fontsize=10, pad=4)
    ax_thr.set_ylim(-5, 110)
    ax_thr.set_xlim(0, dist_max)
    ax_thr.grid(True)

    # ── Frein ────────────────────────────────────────────────────────────────
    brake_dry = tel_dry["Brake"].astype(int).values
    brake_wet = tel_wet["Brake"].astype(int).values
    ax_brk.fill_between(dist_dry, brake_dry, alpha=0.55, color=color_dry,
                         label="Sec", step="mid")
    ax_brk.fill_between(dist_wet, brake_wet * 0.65, alpha=0.45, color=color_wet,
                         label="Humide", step="mid")
    ax_brk.set_ylabel("Freinage", fontsize=9)
    ax_brk.set_title("Freinage", fontsize=10, pad=4)
    ax_brk.set_xlim(0, dist_max)
    ax_brk.set_yticks([0, 1])
    ax_brk.set_yticklabels(["Relâché", "Appuyé"], fontsize=7)
    ax_brk.legend(fontsize=8, framealpha=0.2, facecolor=PANEL_BG, edgecolor="#444")
    ax_brk.grid(True, axis="x")

    # ── Carte circuit : superposition ────────────────────────────────────────
    # Fond circuit
    ax_map.plot(x_dry, y_dry, color="#2a2a2a", lw=8, solid_capstyle="round", zorder=0)

    # Différence de vitesse sur le tracé
    tel_t_dry  = tel_dry["SessionTime"].dt.total_seconds().values
    tel_t_wet  = tel_wet["SessionTime"].dt.total_seconds().values
    pos_t_dry  = pos_dry["SessionTime"].dt.total_seconds().values
    pos_t_wet  = pos_wet["SessionTime"].dt.total_seconds().values

    # Resampler sur distance commune pour les cartes
    N = min(len(x_dry), len(x_wet), 500)
    xi_dry = np.interp(np.linspace(0, len(x_dry)-1, N),
                        np.arange(len(x_dry)), x_dry)
    yi_dry = np.interp(np.linspace(0, len(y_dry)-1, N),
                        np.arange(len(y_dry)), y_dry)
    xi_wet = np.interp(np.linspace(0, len(x_wet)-1, N),
                        np.arange(len(x_wet)), x_wet)
    yi_wet = np.interp(np.linspace(0, len(y_wet)-1, N),
                        np.arange(len(y_wet)), y_wet)

    spd_dry_map = np.interp(np.linspace(0, dist_dry.max(), N), dist_dry,
                              tel_dry["Speed"].values)
    spd_wet_map = np.interp(np.linspace(0, dist_wet.max(), N), dist_wet,
                              tel_wet["Speed"].values)
    diff_map = spd_dry_map - spd_wet_map  # positif = sec plus rapide ici

    sym = max(abs(diff_map.min()), abs(diff_map.max()))
    lc_diff = _circuit_line_collection(xi_dry, yi_dry, diff_map,
                                        CMAPS["diff"], vmin=-sym, vmax=sym,
                                        linewidth=6)
    ax_map.add_collection(lc_diff)
    cb = fig.colorbar(lc_diff, ax=ax_map, fraction=0.02, pad=0.01, shrink=0.6)
    cb.set_label(f"Δ vitesse sec−humide (km/h)\n← humide plus vite | sec plus vite →",
                  fontsize=7, color=SUB_COL)
    cb.ax.yaxis.set_tick_params(color=SUB_COL, labelsize=6)
    plt.setp(cb.ax.yaxis.get_ticklabels(), color=SUB_COL)

    _style_map_ax(ax_map)
    ax_map.set_xlim(xi_dry.min() - 200, xi_dry.max() + 200)
    ax_map.set_ylim(yi_dry.min() - 200, yi_dry.max() + 200)
    ax_map.set_title("Carte de différence de vitesse SEC / HUMIDE",
                      fontsize=10, pad=5, color=TEXT_COL)

    # Légende carte
    from matplotlib.patches import Patch
    ax_map.legend(handles=[
        Patch(color=color_dry, label=f"Sec ({int(t_dry_s//60)}:{t_dry_s%60:05.2f})"),
        Patch(color=color_wet, label=f"Humide ({int(t_wet_s//60)}:{t_wet_s%60:05.2f})"),
    ], fontsize=8, framealpha=0.2, facecolor=PANEL_BG, edgecolor="#444",
       loc="lower right")

    _watermark(fig, "FastF1")
    return _save(fig, output_dir,
                 f"dry_vs_wet_{driver}_{gp_name.replace(' ','_').lower()}.png")


# ═══════════════════════════════════════════════════════════════════════════════
#  FIGURES EXISTANTES (maintenues pour compatibilité)
# ═══════════════════════════════════════════════════════════════════════════════

def plot_session_debrief(weather_df: pd.DataFrame, laps_df,
                          drivers: dict, session_name: str,
                          gp_name: str, output_dir: Path) -> Path:
    fig = plt.figure(figsize=(14, 10))
    fig.suptitle(f"DÉBRIEF MÉTÉO — {gp_name.upper()}  [{session_name}]",
                 fontsize=13, fontweight="bold", y=0.98, color=ACCENT)
    gs = gridspec.GridSpec(3, 2, hspace=0.5, wspace=0.35,
                           left=0.07, right=0.97, top=0.93, bottom=0.07)
    ax_temp = fig.add_subplot(gs[0, :])
    ax_rain = fig.add_subplot(gs[1, 0])
    ax_wind = fig.add_subplot(gs[1, 1])
    ax_laps = fig.add_subplot(gs[2, :])

    t = weather_df["Time"].values
    ax_temp.plot(t, weather_df["AirTemp"], color="#1E80FF", lw=1.6, label="Air (°C)")
    ax_temp.plot(t, weather_df["TrackTemp"], color="#FF6B35", lw=1.6, label="Piste (°C)")
    ax_temp.fill_between(t, weather_df["AirTemp"], weather_df["TrackTemp"],
                          alpha=0.08, color="#FF6B35")
    ax_temp.set_title("Température air & piste", pad=5)
    ax_temp.set_ylabel("°C")
    ax_temp.legend(fontsize=9, framealpha=0.2, facecolor=PANEL_BG, edgecolor="#444")
    ax_temp.grid(True)

    ax_rain.fill_between(t, weather_df["Rainfall"].astype(float),
                          alpha=0.6, color="#1E80FF", step="mid")
    ax_rain.set_title("Précipitations", pad=5)
    ax_rain.set_ylabel("mm")
    ax_rain.set_xlabel("Min depuis début")
    ax_rain.grid(True, axis="x")

    ax_wind.plot(t, weather_df["WindSpeed"], color="#1D9E75", lw=1.5)
    ax_wind.set_title("Vitesse du vent", pad=5)
    ax_wind.set_ylabel("km/h")
    ax_wind.set_xlabel("Min depuis début")
    ax_wind.grid(True)

    if laps_df is not None and not (hasattr(laps_df, 'empty') and laps_df.empty) and drivers:
        for code, info in drivers.items():
            dl = laps_df.pick_driver(code).pick_quicklaps()
            if dl.empty:
                continue
            ax_laps.plot(dl["LapNumber"], dl["LapTime"].dt.total_seconds(),
                          color=info["color"], lw=1.5,
                          label=f"{code} (P{info['position']})",
                          marker="o", markersize=2.5)
    ax_laps.set_title("Temps au tour (sec)", pad=5)
    ax_laps.set_ylabel("sec")
    ax_laps.set_xlabel("Tour")
    ax_laps.legend(fontsize=9, framealpha=0.2, facecolor=PANEL_BG, edgecolor="#444")
    ax_laps.grid(True)

    _watermark(fig, "FastF1")
    return _save(fig, output_dir,
                 f"debrief_{gp_name.replace(' ','_').lower()}.png")


def plot_prerace_forecast(models_data: dict, gp_name: str,
                           race_date: str, output_dir: Path) -> Path:
    colors = ["#1E80FF", "#FF6B35", "#1D9E75", "#A855F7"]
    fig, axes = plt.subplots(2, 1, figsize=(12, 7), sharex=True)
    fig.suptitle(f"PRÉVISION MÉTÉO — {gp_name.upper()}  |  {race_date}",
                 fontsize=13, fontweight="bold", y=0.97, color=ACCENT)
    ax_rain, ax_temp = axes

    for i, (model_name, df) in enumerate(models_data.items()):
        c = colors[i % len(colors)]
        if "rain_prob_pct" in df.columns:
            ax_rain.plot(df["time"], df["rain_prob_pct"],
                         color=c, lw=1.8, label=model_name,
                         marker="o", markersize=3, markevery=3)
        if "temp_c" in df.columns:
            ax_temp.plot(df["time"], df["temp_c"], color=c, lw=1.8, label=model_name)

    ax_rain.axhspan(50, 100, alpha=0.05, color="red", zorder=0)
    ax_rain.axhline(50, color="red", lw=0.8, ls="--", alpha=0.4)
    ax_rain.set_ylabel("Prob. précipitations (%)")
    ax_rain.set_ylim(0, 105)
    ax_rain.set_title("Probabilité de pluie — comparaison modèles", pad=6)
    ax_rain.legend(loc="upper right", fontsize=9, framealpha=0.2,
                   facecolor=PANEL_BG, edgecolor="#444")
    ax_rain.grid(True)

    ax_temp.set_ylabel("Température (°C)")
    ax_temp.set_title("Température de l'air", pad=6)
    ax_temp.grid(True)
    if not models_data:
        ax_temp.text(0.5, 0.5, "Aucun modèle disponible",
                     ha="center", va="center", color=SUB_COL,
                     transform=ax_temp.transAxes)
    else:
        try:
            ax_temp.xaxis.set_major_formatter(mdates.DateFormatter("%Hh"))
            ax_temp.xaxis.set_major_locator(mdates.HourLocator(interval=2))
            plt.xticks(rotation=30, ha="right")
        except Exception:
            pass

    _watermark(fig, "Open-Meteo")
    return _save(fig, output_dir,
                 f"prerace_{gp_name.replace(' ','_').lower()}.png")


def plot_circuit_climatology(clim_df: pd.DataFrame, circuit_name: str,
                              month_label: str, output_dir: Path) -> Path:
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(13, 5))
    fig.suptitle(f"CLIMATOLOGIE — {circuit_name.upper()} | {month_label}",
                 fontsize=13, fontweight="bold", y=1.01, color=ACCENT)

    if clim_df.empty:
        for ax in [ax1, ax2]:
            ax.text(0.5, 0.5, "Données insuffisantes", ha="center",
                    va="center", color=SUB_COL, transform=ax.transAxes)
        _watermark(fig, "Open-Meteo (ERA5)")
        return _save(fig, output_dir,
                     f"clim_{circuit_name.replace(' ','_').lower()}.png")

    daily = clim_df.groupby([clim_df["time"].dt.year,
                              clim_df["time"].dt.day])["rain_mm"].sum()
    annual_rain = daily.groupby(level=0).mean().reset_index()
    annual_rain.columns = ["year", "avg_daily_rain_mm"]

    ax1.bar(annual_rain["year"], annual_rain["avg_daily_rain_mm"],
            color=ACCENT, alpha=0.7, width=0.8)
    if len(annual_rain) > 2:
        z = np.polyfit(annual_rain["year"], annual_rain["avg_daily_rain_mm"], 1)
        p = np.poly1d(z)
        ax1.plot(annual_rain["year"], p(annual_rain["year"]),
                  color="#1E80FF", lw=2, ls="--",
                  label=f"Tendance ({z[0]:+.2f} mm/an)")
        ax1.legend(fontsize=9, framealpha=0.2, facecolor=PANEL_BG, edgecolor="#444")
    ax1.set_xlabel("Année")
    ax1.set_ylabel("Pluie journalière moyenne (mm)")
    ax1.set_title("Évolution des précipitations", pad=5)
    ax1.grid(True, axis="y")

    if "temp_c" in clim_df.columns:
        temps = clim_df["temp_c"].dropna()
        ax2.hist(temps, bins=30, color="#FF6B35", alpha=0.7, edgecolor="#333")
        ax2.axvline(temps.mean(), color="#1E80FF", lw=2,
                    label=f"Moy : {temps.mean():.1f}°C")
        ax2.axvline(temps.quantile(0.1), color="#888", lw=1.2, ls="--",
                    label=f"P10 : {temps.quantile(0.1):.1f}°C")
        ax2.axvline(temps.quantile(0.9), color="#888", lw=1.2, ls="--",
                    label=f"P90 : {temps.quantile(0.9):.1f}°C")
        ax2.set_xlabel("Température (°C)")
        ax2.set_ylabel("Fréquence")
        ax2.set_title("Distribution des températures", pad=5)
        ax2.legend(fontsize=9, framealpha=0.2, facecolor=PANEL_BG, edgecolor="#444")
        ax2.grid(True, axis="y")

    _watermark(fig, "Open-Meteo (ERA5)")
    return _save(fig, output_dir,
                 f"clim_{circuit_name.replace(' ','_').lower()}.png")