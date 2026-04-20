"""
src/evaluation/black_swan.py

Black-swan event window evaluation and crisis lead/lag analysis.

Generates matplotlib figures for:
  - Lead/lag comparison: VQH vs LSTM around SVB 2023
  - Entanglement entropy time series with crisis overlay

These figures are saved to outputs/figures/ and also surfaced via the API.
"""

from __future__ import annotations

from pathlib import Path

import matplotlib
matplotlib.use("Agg")   # non-interactive backend for server environments
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
import numpy as np
import pandas as pd
import yaml
from loguru import logger

_ROOT = Path(__file__).resolve().parents[2]
_PATHS = yaml.safe_load((_ROOT / "config" / "paths.yaml").read_text())


def plot_lead_lag_svb(
    vqh_preds: pd.DataFrame,
    lstm_preds: pd.DataFrame,
    optimal_threshold: float = 0.4,
    out_path: Path | None = None,
) -> None:
    """
    Plot VQH vs LSTM crisis probability around SVB 2023.

    Chart spec:
      Date range : 2023-02-01 to 2023-05-31
      LSTM       : dashed blue line
      VQH        : solid violet line
      Threshold  : horizontal dashed grey line at optimal_threshold
      Crisis zone: red shading from 2023-03-10 onwards
      Reference  : vertical dashed line at 2023-03-10 labeled "SVB closure"

    Args:
        vqh_preds          : VQH prediction DataFrame (must have prob_crisis)
        lstm_preds         : LSTM prediction DataFrame (must have prob_crisis)
        optimal_threshold  : threshold value to draw horizontal line
        out_path           : output path (defaults to paths.yaml figure path)
    """
    if out_path is None:
        out_path = _ROOT / _PATHS["lead_lag_fig"]
    out_path.parent.mkdir(parents=True, exist_ok=True)

    start = pd.Timestamp("2023-02-01")
    end   = pd.Timestamp("2023-05-31")
    svb_date = pd.Timestamp("2023-03-10")

    def _slice(df: pd.DataFrame) -> pd.DataFrame:
        df = df.copy()
        df.index = pd.to_datetime(df.index)
        return df[(df.index >= start) & (df.index <= end)]

    vqh_slice  = _slice(vqh_preds)
    lstm_slice = _slice(lstm_preds)

    fig, ax = plt.subplots(figsize=(14, 5))
    fig.patch.set_facecolor("#0f172a")
    ax.set_facecolor("#1e293b")
    ax.tick_params(colors="#94a3b8")
    ax.spines[:].set_color("#475569")
    ax.title.set_color("#f1f5f9")
    ax.xaxis.label.set_color("#94a3b8")
    ax.yaxis.label.set_color("#94a3b8")

    # Crisis zone shading
    ax.axvspan(svb_date, end, alpha=0.15, color="#ef4444", label="Crisis window")

    # SVB closure vertical line
    ax.axvline(svb_date, color="#ef4444", linestyle="--", linewidth=1.5,
               label="SVB closure (2023-03-10)")

    # Threshold line
    ax.axhline(optimal_threshold, color="#94a3b8", linestyle=":", linewidth=1.2,
               label=f"Threshold ({optimal_threshold:.2f})")

    # LSTM
    if "prob_crisis" in lstm_slice.columns:
        ax.plot(lstm_slice.index, lstm_slice["prob_crisis"],
                color="#3b82f6", linestyle="--", linewidth=2,
                label="LSTM crisis prob", alpha=0.9)

    # VQH
    if "prob_crisis" in vqh_slice.columns:
        ax.plot(vqh_slice.index, vqh_slice["prob_crisis"],
                color="#8b5cf6", linestyle="-", linewidth=2.5,
                label="VQH crisis prob", alpha=0.95)

    ax.set_xlim(start, end)
    ax.set_ylim(-0.05, 1.05)
    ax.set_xlabel("Date")
    ax.set_ylabel("Crisis Probability")
    ax.set_title("Crisis Probability — VQH vs LSTM around SVB 2023",
                 fontsize=13, fontweight="bold")
    legend = ax.legend(loc="upper left", framealpha=0.3,
                       facecolor="#1e293b", labelcolor="#f1f5f9")
    ax.grid(axis="y", color="#475569", alpha=0.4)

    plt.tight_layout()
    plt.savefig(out_path, dpi=150, bbox_inches="tight", facecolor=fig.get_facecolor())
    plt.close()
    logger.info(f"Lead/lag figure saved → {out_path}")


def plot_entanglement_entropy(
    vqh_preds: pd.DataFrame,
    out_path: Path | None = None,
) -> None:
    """
    Plot Von Neumann entanglement entropy over the test window.

    Chart spec:
      X-axis   : dates in test window
      Y-axis   : entropy (0 to 1)
      Fill     : violet at 30% opacity above 0.5 (high entanglement zone)
      Overlay  : crisis_flag step function on secondary axis (red)
      Title    : "Quantum Entanglement Entropy — Proxy for Systemic Risk"

    Args:
        vqh_preds : DataFrame with entanglement_entropy and true_regime columns
        out_path  : output path (defaults to paths.yaml)
    """
    if out_path is None:
        out_path = _ROOT / _PATHS["entanglement_fig"]
    out_path.parent.mkdir(parents=True, exist_ok=True)

    df = vqh_preds.copy()
    df.index = pd.to_datetime(df.index)

    fig, ax1 = plt.subplots(figsize=(16, 5))
    fig.patch.set_facecolor("#0f172a")
    ax1.set_facecolor("#1e293b")
    ax1.tick_params(colors="#94a3b8")
    ax1.spines[:].set_color("#475569")

    # Entropy line + fill above 0.5
    if "entanglement_entropy" in df.columns:
        entropy = df["entanglement_entropy"]
        ax1.plot(df.index, entropy, color="#8b5cf6", linewidth=1.8, label="Entropy")
        ax1.fill_between(df.index, 0.5, entropy,
                         where=(entropy > 0.5),
                         alpha=0.3, color="#8b5cf6", label="High entanglement zone")

    ax1.set_ylim(0, 1.05)
    ax1.set_xlabel("Date", color="#94a3b8")
    ax1.set_ylabel("Von Neumann Entropy", color="#8b5cf6")
    ax1.set_title(
        "Quantum Entanglement Entropy — Proxy for Systemic Risk",
        fontsize=13, fontweight="bold", color="#f1f5f9"
    )

    # Secondary axis: crisis flag
    ax2 = ax1.twinx()
    if "true_regime" in df.columns:
        crisis_flag = (df["true_regime"] == 0).astype(float)
        ax2.fill_between(df.index, 0, crisis_flag,
                         step="pre", alpha=0.25, color="#ef4444", label="Crisis regime")
        ax2.set_ylim(0, 3)
        ax2.set_yticks([])
        ax2.spines[:].set_color("#475569")

    lines1, labels1 = ax1.get_legend_handles_labels()
    ax1.legend(lines1, labels1, loc="upper left",
               framealpha=0.3, facecolor="#1e293b", labelcolor="#f1f5f9")
    ax1.grid(axis="y", color="#475569", alpha=0.3)

    plt.tight_layout()
    plt.savefig(out_path, dpi=150, bbox_inches="tight", facecolor=fig.get_facecolor())
    plt.close()
    logger.info(f"Entanglement entropy figure saved → {out_path}")
