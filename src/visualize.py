"""
CyberGrid-AI | visualise.py

Publication-quality figures for IEEE SmartGridComm / IEEE ISGT submission.
All figures: Times New Roman, PDF vector, 600 DPI.

Figures produced
----------------
  Fig 1 (double-col, 178 mm)  CUSUM trajectories — the central result
  Fig 2 (single-col,  86 mm)  System architecture block diagram
  Fig 3 (single-col,  86 mm)  Detection delay vs attack magnitude λ
  Fig 4 (single-col,  86 mm)  Season-dependent detectability (bar chart)
  Fig 5 (single-col,  86 mm)  Temperature recovery performance vs λ

Usage (Windows CMD, cybergrid env active)
-----------------------------------------
    cd C:\\Users\\Albashir\\Documents\\CyberGrid-AI
    python src/visualise.py
    python src/visualise.py --outdir paper/figures/
    python src/visualise.py --skip fig3   # skip slow lambda sweep

Rule: every repo import (conformal, attacks, …) is inside a function,
never at module level — so the import block never fails even in a
restricted environment.
"""

from __future__ import annotations

import argparse
import sys
import warnings
from pathlib import Path

import numpy as np
import pandas as pd

import matplotlib
matplotlib.use("Agg")          # non-interactive, safe for all platforms
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
import matplotlib.lines   as mlines
from matplotlib.patches import FancyBboxPatch
from matplotlib           import rcParams

warnings.filterwarnings("ignore")

# ── Paths ────────────────────────────────────────────────────────────────────
BASE_DIR = Path(__file__).resolve().parent.parent
SRC_DIR  = BASE_DIR / "src"
DATA_DIR = BASE_DIR / "data" / "processed"
RES_DIR  = BASE_DIR / "results"
sys.path.insert(0, str(SRC_DIR))

# ── Typography ───────────────────────────────────────────────────────────────
# Times New Roman is the IEEE standard; DejaVu Serif is the fallback for
# environments where TNR is not installed (Linux CI, Docker).
# On Windows (the user's machine), TNR will be selected automatically.
rcParams.update({
    "font.family"        : "serif",
    "font.serif"         : ["Times New Roman", "Times", "DejaVu Serif"],
    "mathtext.fontset"   : "stix",          # STIXGeneral matches TNR weight
    "axes.titlesize"     : 9,
    "axes.titleweight"   : "normal",
    "axes.labelsize"     : 8,
    "xtick.labelsize"    : 7,
    "ytick.labelsize"    : 7,
    "legend.fontsize"    : 7,
    "legend.framealpha"  : 0.90,
    "legend.edgecolor"   : "0.65",
    "legend.borderpad"   : 0.4,
    "lines.linewidth"    : 1.2,
    "axes.linewidth"     : 0.75,
    "grid.linewidth"     : 0.4,
    "grid.alpha"         : 0.35,
    "figure.dpi"         : 600,
    "savefig.dpi"        : 600,
    "savefig.bbox"       : "tight",
    "savefig.pad_inches" : 0.04,
})

# ── Colorblind-safe palette (Wong 2011, Nature Methods) ──────────────────────
BLUE   = "#0072B2"   # sky / INPUT-DETECT
ORANGE = "#E69F00"   # orange / attack annotation
GREEN  = "#009E73"   # green / RESIDUAL-DETECT ★
RED    = "#D55E00"   # vermillion / alarm / corrupted
PURPLE = "#CC79A7"   # reddish-purple / fused
GREY   = "#999999"   # neutral
BLACK  = "#000000"

# ── IEEE column widths (inches) ───────────────────────────────────────────────
W2 = 7.00    # double-column: 178 mm exactly
W1 = 3.40    # single-column:  86 mm exactly

# ── Locked experimental results (from three conformal.py runs) ────────────────
# These are the numbers confirmed on real GEFCom2014 data.
_RECOVERY = {
    0.05: dict(mae_c=3.027,  mae_r=5.498,  pct=-81.6, lam_hat=-0.037),
    0.10: dict(mae_c=6.057,  mae_r=4.227,  pct=+30.2, lam_hat=+0.028),
    0.20: dict(mae_c=12.116, mae_r=3.171,  pct=+73.8, lam_hat=+0.140),
}

_DETECTABILITY = dict(
    sigma_T  = 15.760,   # °C  (from detectability_report, λ=0.05 run)
    dg_dT    =  3.503,   # kW/°C
    sigma_r  = 10.619,   # kW
    gain     =  5.20,    # constant across λ (theoretical result)
)


# =============================================================================
# Helpers
# =============================================================================

def _save(fig, path: Path) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(path, format="pdf", dpi=600,
                bbox_inches="tight", pad_inches=0.04)
    plt.close(fig)
    print(f"      → {path.name}")


def _first_post_attack(alarms: np.ndarray, attack_start: int):
    """Return index of first post-attack alarm, or None."""
    post = np.flatnonzero(alarms[attack_start:])
    return int(post[0]) + attack_start if len(post) > 0 else None


def _title(ax, text: str, **kw) -> None:
    """Apply a Times New Roman title with consistent styling."""
    ax.set_title(text, fontfamily="serif", fontsize=9,
                 fontweight="normal", pad=4, **kw)


def _build_detector():
    """Fit the three-signal detector on calibration data once.

    Returns (det, cal, test_05, cfg) so callers can use any of these
    without re-fitting. The full GBM fit takes ~90 s on first call;
    subsequent lambda runs reuse the fitted detector instantly.
    """
    from conformal import DetectorConfig, ThreeSignalDetector, build_from_repo
    cfg = DetectorConfig()
    cal, test05 = build_from_repo("scaling", lam=0.05, cfg=cfg)
    det = ThreeSignalDetector(cfg).fit(cal)
    return det, cal, test05, cfg


def _detect(det, lam: float, cfg):
    """Run detection for a single lambda without refitting."""
    from conformal import build_from_repo
    _, test = build_from_repo("scaling", lam=lam, cfg=cfg)
    res = det.detect(test)
    return res, test


# =============================================================================
# Figure 1 (double-column): CUSUM trajectories
# =============================================================================

def fig1_cusum_trajectory(det, test05, outdir: Path) -> None:
    """
    Three-panel CUSUM trajectory for λ=0.05 — the central result.

    Panel A  Temperature: clean vs. corrupted — shows attack subtlety
    Panel B  INPUT-DETECT normalised statistic G_t/h — slow, fires at 201 h
    Panel C  RESIDUAL-DETECT normalised statistic G_t/h — fires at 88 h
             + fused alarm event markers (purple triangles)
    """
    from conformal import DetectorConfig

    res = det.detect(test05)
    inp = res.signals.get("input")
    rsd = res.signals.get("residual")
    if inp is None or rsd is None:
        print("      [SKIP] missing signals")
        return

    a  = test05.attack_start                       # attack onset index
    d_i = _first_post_attack(inp.alarms, a)        # INPUT first alarm
    d_r = _first_post_attack(rsd.alarms, a)        # RESIDUAL first alarm
    d_f = a + int(res.metrics["detection_delay_hours"])  # fused alarm

    # Window: 300 h before attack to 400 h after fused alarm
    ws = max(0, a - 300)
    we = min(test05.n, d_f + 400)
    t  = np.arange(ws, we) - a                     # hours relative to onset

    # Clean temperature: re-read CSV (pre-attack ground truth)
    cfg = det.cfg
    feat_csv = DATA_DIR / "gefcom2014_features.csv"
    df_orig  = pd.read_csv(feat_csv, index_col="timestamp", parse_dates=True)
    df_orig  = df_orig.dropna(subset=["target_1h"])
    n_cal    = int(len(df_orig) * cfg.calib_frac)
    T_clean  = df_orig["temperature"].to_numpy(float)[n_cal:]
    T_corr   = test05.temperature

    # ── Layout ───────────────────────────────────────────────────────────
    fig, axes = plt.subplots(
        3, 1,
        figsize=(W2, 5.8),
        sharex=True,
        gridspec_kw={"height_ratios": [1.15, 1.15, 1.0], "hspace": 0.06},
    )

    # Background shading: pre-attack (white) / attack (very light red)
    for ax in axes:
        ax.axvspan(0, t[-1] + 1, color=RED, alpha=0.05, zorder=0)
        ax.axvline(0, color=RED, lw=0.9, ls="--", alpha=0.80, zorder=4,
                   label="Attack onset ($t_s$)")

    # ── Panel A: Temperature ─────────────────────────────────────────────
    ax = axes[0]
    ax.plot(t, T_clean[ws:we], color=BLUE,   lw=1.0, ls="-",
            label=r"Clean $T_t$", zorder=3)
    ax.plot(t, T_corr[ws:we],  color=ORANGE, lw=1.0, ls="--",
            label=r"Corrupted $\tilde{T}_t$  ($\lambda{=}0.05$)", zorder=3)
    ax.set_ylabel("Temperature (°C)", fontsize=8)
    ax.legend(loc="upper left", fontsize=6.5, ncol=2)
    ax.grid(True, axis="y")

    # Annotate: attack is visually subtle
    mid = int(len(t) * 0.65)
    ax.annotate("Scaling FDIA:\nstatistically\nundetectable\nby eye",
                xy=(t[mid], T_corr[ws + mid]),
                xytext=(t[mid] - 80, T_corr[ws + mid] + 4),
                fontsize=6, color=RED, fontfamily="serif",
                arrowprops=dict(arrowstyle="->", color=RED, lw=0.7),
                ha="center")

    # ── Panel B: INPUT-DETECT ────────────────────────────────────────────
    ax = axes[1]
    stat_i = inp.stat[ws:we] / inp.h
    ax.plot(t, stat_i, color=BLUE, lw=1.1, ls="-",
            label=r"INPUT-DETECT  $G_t/h$", zorder=3)
    ax.axhline(1.0, color=BLUE, lw=0.8, ls=":", alpha=0.75, zorder=3,
               label="Alarm threshold ($G_t/h = 1$)")
    ax.set_ylabel(r"INPUT-DETECT  $G_t/h$", fontsize=8)
    ax.set_ylim(bottom=-0.05)
    ax.legend(loc="upper left", fontsize=6.5)
    ax.grid(True, axis="y")

    if d_i is not None:
        xi = d_i - a
        ax.axvline(xi, color=BLUE, lw=0.9, ls="-.", alpha=0.90, zorder=4)
        ax.annotate(f"+{xi} h",
                    xy=(xi, 1.0), xytext=(xi + 20, 1.35),
                    fontsize=7, color=BLUE, fontfamily="serif",
                    arrowprops=dict(arrowstyle="->", color=BLUE, lw=0.7))

    # ── Panel C: RESIDUAL-DETECT + fused ─────────────────────────────────
    ax = axes[2]
    stat_r = rsd.stat[ws:we] / rsd.h
    ax.plot(t, stat_r, color=GREEN, lw=1.1, ls="-",
            label=r"RESIDUAL-DETECT  $G_t/h$  ★", zorder=3)
    ax.axhline(1.0, color=GREEN, lw=0.8, ls=":", alpha=0.75, zorder=3,
               label="Alarm threshold ($G_t/h = 1$)")
    ax.set_ylabel(r"RESIDUAL-DETECT  $G_t/h$", fontsize=8)
    ax.set_ylim(bottom=-0.05)
    ax.set_xlabel("Hours relative to attack onset $t - t_s$", fontsize=8)
    ax.grid(True, axis="y")

    if d_r is not None:
        xr = d_r - a
        ax.axvline(xr, color=GREEN, lw=0.9, ls="-.", alpha=0.90, zorder=4)
        ax.annotate(f"+{xr} h",
                    xy=(xr, 1.0), xytext=(xr + 20, 1.40),
                    fontsize=7, color=GREEN, fontfamily="serif",
                    arrowprops=dict(arrowstyle="->", color=GREEN, lw=0.7))

    # Fused alarm triangles at post-attack alarm events
    fused_post = np.flatnonzero(res.fused_alarms[a:we]) + a
    if len(fused_post) > 0:
        fused_t = fused_post - a
        ax.plot(fused_t, np.full(len(fused_t), -0.04),
                "v", color=PURPLE, ms=5, zorder=5,
                label=r"Fused alarm (vote, $\tau_v{=}2$, $W{=}250$ h)")

    # First fused alarm annotation
    xf = d_f - a
    if 0 <= xf < t[-1]:
        ax.annotate(f"Fused\n+{xf} h",
                    xy=(xf, -0.04), xytext=(xf + 20, 0.55),
                    fontsize=7, color=PURPLE, fontfamily="serif",
                    arrowprops=dict(arrowstyle="->", color=PURPLE, lw=0.7))

    ax.legend(loc="upper left", fontsize=6.5, ncol=2)

    # ── Super-title ───────────────────────────────────────────────────────
    fig.suptitle(
        r"CUSUM trajectories under $\lambda=0.05$ scaling FDIA: "
        r"RESIDUAL-DETECT fires at $+88$ h vs. INPUT-DETECT at $+201$ h "
        r"— detectability gain $5.20\times$",
        fontfamily="serif", fontsize=9, fontweight="normal", y=1.005,
    )

    _save(fig, outdir / "fig2_cusum_trajectory.pdf")


# =============================================================================
# Figure 2 (single-column): System architecture
# =============================================================================

def fig2_architecture(outdir: Path) -> None:
    """
    Block diagram of the CyberGrid-AI three-signal detection and
    recovery framework.

    Drawn entirely with matplotlib patches — no external drawing tools needed.
    """
    fig, ax = plt.subplots(1, 1, figsize=(W1, 4.0))
    ax.set_xlim(0, 10)
    ax.set_ylim(0, 10)
    ax.axis("off")

    # ── Box helper ────────────────────────────────────────────────────────
    def box(x, y, w, h, text, fc="#DDEEFF", ec="0.45",
            fs=6.5, fw="normal", lw=0.7, style="round,pad=0.08"):
        r = FancyBboxPatch((x, y), w, h, boxstyle=style,
                           facecolor=fc, edgecolor=ec,
                           linewidth=lw, zorder=2, clip_on=False)
        ax.add_patch(r)
        ax.text(x + w / 2, y + h / 2, text,
                ha="center", va="center",
                fontsize=fs, fontfamily="serif",
                fontweight=fw, zorder=3,
                multialignment="center", clip_on=False)

    # ── Arrow helper ──────────────────────────────────────────────────────
    def arr(x1, y1, x2, y2, lw=0.8, color="0.30"):
        ax.annotate("",
                    xy=(x2, y2), xytext=(x1, y1),
                    arrowprops=dict(arrowstyle="->", color=color,
                                   lw=lw, shrinkA=2, shrinkB=2),
                    zorder=5, clip_on=False)

    # ── INPUTS (left column) ──────────────────────────────────────────────
    box(0.10, 8.15, 2.45, 0.75, "Temperature $T_t$\n(attacker-corrupted)",
        fc="#FFE5D9", fs=6.5)
    box(0.10, 6.90, 2.45, 0.75, "Metered Load $y_t$\n(authentic channel)",
        fc="#D9FFE5", fs=6.5)
    box(0.10, 5.65, 2.45, 0.75, "Calendar Features\n$h_t, d_t, s_t$",
        fc="#E8E8F5", fs=6.5)

    # ── AE-CLSTM FORECASTER (centre-top) ──────────────────────────────────
    box(3.30, 7.40, 2.40, 1.30,
        "AE-CLSTM\nForecaster\n(absorbs attack)", fc="#D5E8FF", fs=6.5)
    arr(2.55, 8.53, 3.30, 8.05)    # Temp → forecaster
    arr(2.55, 7.28, 3.30, 7.72)    # Load → forecaster
    arr(2.55, 6.03, 3.30, 7.55)    # Cal → forecaster

    # Forecast output
    box(6.20, 7.65, 1.70, 0.75,
        "Load Forecast $\\hat{y}_t$", fc="#D5E8FF", fs=6.5)
    arr(5.70, 8.05, 6.20, 8.03)

    # ── THREE DETECTION SIGNALS (centre column) ───────────────────────────
    # AE-DETECT (blind — shown grey)
    box(3.30, 5.90, 2.40, 0.95,
        "AE-DETECT\n(blind at $\\lambda=0.05$)", fc="#EEEEEE", ec="0.6",
        fs=6.5)
    arr(4.50, 7.40, 4.50, 6.85)    # forecaster bottom → AE-DETECT

    # INPUT-DETECT (medium speed)
    box(3.30, 4.55, 2.40, 0.95,
        "INPUT-DETECT\n(delay: 201 h)",
        fc="#FFF0CC", ec=ORANGE, fs=6.5)
    arr(1.33, 8.15, 1.33, 4.55)    # Temp drop (left edge)
    ax.annotate("", xy=(3.30, 5.02), xytext=(1.33, 5.02),
                arrowprops=dict(arrowstyle="->", color="0.30",
                                lw=0.8, shrinkA=0, shrinkB=2), zorder=5)

    # RESIDUAL-DETECT (★ contribution)
    box(3.30, 3.10, 2.40, 1.05,
        "RESIDUAL-DETECT ★\n$r_t = y_t - g(T_t, c_t)$\n(delay: 88 h)",
        fc="#CCEECC", ec=GREEN, fs=6.2, fw="normal", lw=1.1)
    # Load → RESIDUAL-DETECT
    arr(2.55, 7.28, 2.85, 7.28)   # partial
    ax.annotate("", xy=(3.30, 3.62), xytext=(1.78, 3.62),
                arrowprops=dict(arrowstyle="->", color="0.30",
                                lw=0.8, shrinkA=0, shrinkB=2), zorder=5)
    ax.annotate("", xy=(1.78, 3.62), xytext=(1.78, 7.28),
                arrowprops=dict(arrowstyle="-", color="0.30",
                                lw=0.8), zorder=5)
    # Temp → RESIDUAL-DETECT
    ax.annotate("", xy=(3.30, 3.82), xytext=(0.90, 3.82),
                arrowprops=dict(arrowstyle="->", color="0.30",
                                lw=0.8, shrinkA=0, shrinkB=2), zorder=5)
    ax.annotate("", xy=(0.90, 3.82), xytext=(0.90, 8.15),
                arrowprops=dict(arrowstyle="-", color="0.30",
                                lw=0.8), zorder=5)

    # ── FUSION ────────────────────────────────────────────────────────────
    box(6.20, 4.15, 1.80, 1.80,
        "Vote\nFusion\n($\\tau_v=2$,\n$W=250$ h)",
        fc="#E5E5FF", ec=PURPLE, fs=6.5, lw=0.9)
    arr(5.70, 6.37, 6.40, 5.95)    # AE   → fusion
    arr(5.70, 5.02, 6.20, 5.35)    # INPUT → fusion
    arr(5.70, 3.62, 6.20, 4.50)    # RESIDUAL → fusion

    # ── OUTPUTS ───────────────────────────────────────────────────────────
    box(8.30, 5.35, 1.55, 0.80,
        "⚠  Alarm\n(+88 h)",
        fc="#FFD5D5", ec=RED, fs=6.5, fw="bold", lw=1.0)
    arr(8.00, 5.05, 8.30, 5.75)

    box(8.30, 4.10, 1.55, 0.80,
        "$\\hat{\\lambda}$ Estimation\n+ Recovery",
        fc="#CCEECC", ec=GREEN, fs=6.2)
    arr(8.00, 4.65, 8.60, 4.50)

    # ── Labels and legend ─────────────────────────────────────────────────
    _title(ax,
           "CyberGrid-AI: Three-Signal FDIA Detection and Recovery Framework")

    legend_items = [
        mpatches.Patch(fc="#CCEECC", ec=GREEN,   label="★ Primary contribution"),
        mpatches.Patch(fc="#EEEEEE", ec="0.6",   label="Blind to attack"),
        mpatches.Patch(fc="#D9FFE5", ec="0.45",  label="Authentic channel"),
        mpatches.Patch(fc="#FFE5D9", ec="0.45",  label="Corrupted channel"),
    ]
    ax.legend(handles=legend_items, loc="lower left",
              fontsize=5.5, framealpha=0.9, ncol=2,
              bbox_to_anchor=(0.0, 0.0))

    _save(fig, outdir / "fig1_architecture_PLACEHOLDER.pdf")


# =============================================================================
# Figure 3 (single-column): Detection delay vs λ
# =============================================================================

def fig3_delay_vs_lambda(det, cfg, outdir: Path) -> None:
    """
    Detection delay (hours) vs attack magnitude λ for all three signals
    and the fused vote result. Demonstrates that RESIDUAL-DETECT
    consistently achieves 2–3× lower delay than INPUT-DETECT, and that
    fused (vote) matches RESIDUAL's speed while controlling FA rate.
    """
    lambdas = [0.02, 0.05, 0.08, 0.10, 0.15, 0.20]

    rows: dict[str, list] = {
        "INPUT-DETECT":     [],
        "RESIDUAL-DETECT":  [],
        "Fused (vote)":     [],
    }

    print("      λ sweep (6 runs, ~30 s each)...")
    for lam in lambdas:
        res, test = _detect(det, lam, cfg)
        a = test.attack_start
        inp = res.signals.get("input")
        rsd = res.signals.get("residual")

        d_i = (_first_post_attack(inp.alarms, a) - a
               if inp is not None else None)
        d_r = (_first_post_attack(rsd.alarms, a) - a
               if rsd is not None else None)
        d_f = (int(res.metrics["detection_delay_hours"])
               if res.metrics.get("detected", 0) == 1.0 else None)

        rows["INPUT-DETECT"].append(d_i)
        rows["RESIDUAL-DETECT"].append(d_r)
        rows["Fused (vote)"].append(d_f)
        print(f"        λ={lam:.2f}  RESIDUAL={d_r}h  INPUT={d_i}h  FUSED={d_f}h")

    # ── Plot ─────────────────────────────────────────────────────────────
    fig, ax = plt.subplots(1, 1, figsize=(W1, 2.8))

    styles = {
        "INPUT-DETECT"   : (BLUE,   "-",  "o"),
        "RESIDUAL-DETECT": (GREEN,  "--", "s"),
        "Fused (vote)"   : (PURPLE, "-.", "^"),
    }

    for label, (color, ls, marker) in styles.items():
        xs, ys = [], []
        for lam, d in zip(lambdas, rows[label]):
            if d is not None:
                xs.append(lam)
                ys.append(d)
        if xs:
            ax.plot(xs, ys, color=color, ls=ls,
                    marker=marker, markersize=4.5,
                    label=label, lw=1.2, zorder=3)

    ax.set_xlabel(r"Attack magnitude $\lambda$", fontsize=8)
    ax.set_ylabel("Detection delay (hours)", fontsize=8)
    ax.set_xticks(lambdas)
    ax.set_xticklabels([f"{l:.2f}" for l in lambdas], fontsize=6.5)
    ax.legend(fontsize=6.5, loc="upper right")
    ax.grid(True, axis="y")
    _title(ax, r"Detection delay vs. attack magnitude $\lambda$")

    # Annotate the gain at λ=0.05
    d_i05 = rows["INPUT-DETECT"][1]
    d_r05 = rows["RESIDUAL-DETECT"][1]
    if d_i05 and d_r05:
        ax.annotate(
            "",
            xy=(0.05, d_r05), xytext=(0.05, d_i05),
            arrowprops=dict(arrowstyle="<->", color=GREY, lw=0.8),
            zorder=4,
        )
        ax.text(0.056, (d_i05 + d_r05) / 2, f"2.3×\nfaster",
                fontsize=6, color=GREY, fontfamily="serif", va="center")

    _save(fig, outdir / "fig3_delay_vs_lambda.pdf")


# =============================================================================
# Figure 4 (single-column): Season-dependent detectability
# =============================================================================

def fig4_season_detectability(outdir: Path) -> None:
    """
    Theoretical detectability shift δ by season for INPUT-DETECT and
    RESIDUAL-DETECT at λ=0.05, derived from GEFCom2014 seasonal
    temperature statistics and the fitted dg/dT.

    The V-shaped load–temperature curve means:
    • Winter T ≈ 2 °C  → 5 % of 2 °C = 0.1 °C displacement → small signal
    • Summer T ≈ 22 °C → 5 % of 22 °C = 1.1 °C             → large signal

    This is the "Celsius self-limiting" result: multiplicative scaling
    attacks are near-undetectable in winter.
    """
    # Seasonal mean |T| from GEFCom2014 features CSV
    feat_csv = DATA_DIR / "gefcom2014_features.csv"
    df = pd.read_csv(feat_csv, index_col="timestamp", parse_dates=True,
                     usecols=["timestamp", "temperature", "season"])

    # GEFCom temperature is in degrees Celsius (confirmed by σ_T = 15.76°C)
    season_T = (df.groupby("season")["temperature"]
                  .apply(lambda x: x.abs().mean())
                  .to_dict())

    seasons = ["Winter", "Spring", "Summer", "Autumn"]
    dg  = _DETECTABILITY["dg_dT"]
    sr  = _DETECTABILITY["sigma_r"]
    sT  = _DETECTABILITY["sigma_T"]
    lam = 0.05

    d_res = [dg * lam * season_T.get(s, 10.0) / sr for s in seasons]
    d_inp = [lam * season_T.get(s, 10.0) / sT       for s in seasons]

    x = np.arange(len(seasons))
    w = 0.35

    fig, ax = plt.subplots(1, 1, figsize=(W1, 2.8))

    b1 = ax.bar(x - w / 2, d_res, w, label=r"RESIDUAL-DETECT $\delta$",
                color=GREEN,  alpha=0.85, edgecolor="white", linewidth=0.4)
    b2 = ax.bar(x + w / 2, d_inp, w, label=r"INPUT-DETECT $\delta$",
                color=BLUE,   alpha=0.85, edgecolor="white", linewidth=0.4)

    # Detection threshold line: δ ≈ 1.0 considered reliably detectable
    ax.axhline(1.0, color=RED, lw=0.9, ls="--",
               label=r"Reliability threshold ($\delta = 1$)", zorder=4)

    # Value labels
    for bar in b1:
        h = bar.get_height()
        ax.text(bar.get_x() + bar.get_width() / 2, h + 0.015,
                f"{h:.2f}", ha="center", va="bottom",
                fontsize=5.5, color=GREEN)
    for bar in b2:
        h = bar.get_height()
        ax.text(bar.get_x() + bar.get_width() / 2, h + 0.015,
                f"{h:.2f}", ha="center", va="bottom",
                fontsize=5.5, color=BLUE)

    ax.set_xticks(x)
    ax.set_xticklabels(seasons, fontsize=7)
    ax.set_xlabel("Season", fontsize=8)
    ax.set_ylabel(r"Standardised shift $\delta$ ($\sigma$ units)", fontsize=8)
    ax.legend(fontsize=6.5, loc="upper left")
    ax.grid(True, axis="y")
    _title(ax, r"Season-dependent detectability at $\lambda=0.05$")

    # Annotation: winter blind-spot
    ax.annotate(r"Winter blind spot ($T\approx 2$°C)",
                xy=(0, d_res[0]), xytext=(0.6, 0.55),
                fontsize=6, color=GREY, fontfamily="serif",
                arrowprops=dict(arrowstyle="->", color=GREY, lw=0.7))

    _save(fig, outdir / "fig4_season_detectability.pdf")


# =============================================================================
# Figure 5 (single-column): Recovery performance
# =============================================================================

def fig5_recovery(outdir: Path) -> None:
    """
    Temperature recovery: MAE before/after vs λ, and improvement (%).

    Uses locked experimental results from the three conformal.py runs.
    Recovery via scalar λ̂ estimation; reliable for λ ≥ 0.10.
    """
    lambdas = [0.05, 0.10, 0.20]
    mae_c   = [_RECOVERY[l]["mae_c"] for l in lambdas]
    mae_r   = [_RECOVERY[l]["mae_r"] for l in lambdas]
    pct     = [_RECOVERY[l]["pct"]   for l in lambdas]

    x = np.arange(len(lambdas))
    w = 0.33

    fig, (ax1, ax2) = plt.subplots(
        1, 2, figsize=(W1, 2.8),
        gridspec_kw={"wspace": 0.42},
    )

    # ── Left: MAE bars ────────────────────────────────────────────────────
    b1 = ax1.bar(x - w / 2, mae_c, w,
                 label=r"Corrupted $\tilde{T}_t$",
                 color=RED, alpha=0.80, edgecolor="white", linewidth=0.4)
    b2 = ax1.bar(x + w / 2, mae_r, w,
                 label=r"Recovered $\hat{T}_t$",
                 color=GREEN, alpha=0.80, edgecolor="white", linewidth=0.4)

    ax1.set_xticks(x)
    ax1.set_xticklabels([f"$\\lambda={l}$" for l in lambdas], fontsize=6.5)
    ax1.set_ylabel("Temperature MAE (°C)", fontsize=8)
    ax1.legend(fontsize=6.0, loc="upper left")
    ax1.grid(True, axis="y")
    _title(ax1, "Recovery MAE")

    # Annotate λ=0.05 as below-threshold
    ax1.text(0, max(mae_c[0], mae_r[0]) * 1.04,
             "Below\nrecovery\nthreshold",
             ha="center", va="bottom",
             fontsize=5.2, color=GREY, style="italic", fontfamily="serif")

    # ── Right: Improvement % ──────────────────────────────────────────────
    colors_pct = [GREY if p < 0 else GREEN for p in pct]
    bars = ax2.bar(x, pct, color=colors_pct,
                   alpha=0.85, edgecolor="white", linewidth=0.4, zorder=3)
    ax2.axhline(0, color=BLACK, lw=0.7, zorder=4)

    for bar, p in zip(bars, pct):
        ypos = bar.get_height()
        sign = "+" if p > 0 else ""
        ax2.text(bar.get_x() + bar.get_width() / 2,
                 ypos + (2 if p >= 0 else -10),
                 f"{sign}{p:.0f}%",
                 ha="center", va="bottom" if p >= 0 else "top",
                 fontsize=7, fontfamily="serif",
                 color=BLACK)

    ax2.set_xticks(x)
    ax2.set_xticklabels([f"$\\lambda={l}$" for l in lambdas], fontsize=6.5)
    ax2.set_ylabel("MAE improvement (%)", fontsize=8)
    ax2.grid(True, axis="y")
    _title(ax2, "Improvement")

    fig.suptitle(
        r"Temperature recovery performance vs. attack magnitude $\lambda$"
        "\n"
        r"($\hat{\lambda}$ estimation over 168 h post-detection window)",
        fontfamily="serif", fontsize=9, fontweight="normal", y=1.03,
    )

    _save(fig, outdir / "fig5_recovery.pdf")


# =============================================================================
# Main
# =============================================================================

def main(argv=None):
    p = argparse.ArgumentParser(
        description="Generate all publication figures for the CyberGrid-AI paper.",
        formatter_class=argparse.RawDescriptionHelpFormatter,
    )
    p.add_argument("--outdir", default="figures",
                   help="Output directory (default: figures/)")
    p.add_argument("--skip", nargs="*", default=[],
                   choices=["fig1", "fig2", "fig3", "fig4", "fig5"],
                   help="Skip specific figures (e.g. --skip fig3 to skip slow sweep)")
    args = p.parse_args(argv)

    outdir = Path(args.outdir)
    outdir.mkdir(parents=True, exist_ok=True)

    print("=" * 62)
    print("  CyberGrid-AI  |  Generating publication figures (PDF)")
    print("=" * 62)
    print(f"  Output directory : {outdir.resolve()}")
    print(f"  Figures to skip  : {args.skip or 'none'}\n")

    # Fit detector once — reused by Fig 1, 3 (saves ~90 s)
    need_det = not all(f in args.skip for f in ["fig1", "fig3"])
    det = cal = test05 = cfg = None

    if need_det:
        print("[1/2] Fitting load-temperature model and calibrating"
              " thresholds...")
        print("      (this takes ~90 s — done once, reused for all figures)\n")
        det, cal, test05, cfg = _build_detector()
        print("      Done.\n")

    # ── Figure 1 ─────────────────────────────────────────────────────────
    if "fig1" not in args.skip:
        print("[Fig 2] CUSUM trajectory (double-column, 178 mm)")
        fig1_cusum_trajectory(det, test05, outdir)

    # ── Figure 2 ─────────────────────────────────────────────────────────
    if "fig2" not in args.skip:
        print("\n[Fig 1] System architecture (GitHub placeholder; paper version in LaTeX/TikZ) block diagram")
        fig2_architecture(outdir)

    # ── Figure 3 ─────────────────────────────────────────────────────────
    if "fig3" not in args.skip:
        print("\n[Fig 3] Detection delay vs lambda sweep")
        print("      (6 lambda values × detection run, ~3 minutes)")
        fig3_delay_vs_lambda(det, cfg, outdir)

    # ── Figure 4 ─────────────────────────────────────────────────────────
    if "fig4" not in args.skip:
        print("\n[Fig 4] Season-dependent detectability bar chart")
        fig4_season_detectability(outdir)

    # ── Figure 5 ─────────────────────────────────────────────────────────
    if "fig5" not in args.skip:
        print("\n[Fig 5] Temperature recovery performance")
        fig5_recovery(outdir)

    # ── Summary ──────────────────────────────────────────────────────────
    pdfs = sorted(outdir.glob("*.pdf"))
    print(f"\n{'=' * 62}")
    print(f"  All figures saved to: {outdir.resolve()}")
    for f in pdfs:
        size_kb = f.stat().st_size / 1024
        print(f"    {f.name:<35} {size_kb:>6.1f} KB")
    print(f"{'=' * 62}")
    print("\n  Next step: git commit then write the paper.\n")


if __name__ == "__main__":
    main()