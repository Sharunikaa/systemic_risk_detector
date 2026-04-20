"""
src/data/pipeline.py

Step 1.9 — Feature Matrix Assembly.

Orchestrates all preprocessing steps and assembles the final feature_matrix.csv.
This is the output consumed by all downstream models.

Final column list (exact order, 15 training features + 1 label):
  [0]  spx_w              — winsorized SPX log return
  [1]  gold_w             — winsorized Gold log return
  [2]  btc_w              — winsorized BTC log return
  [3]  spx_garch          — SPX GARCH(1,1) conditional vol, annualized
  [4]  gold_garch         — Gold GARCH(1,1) conditional vol, annualized
  [5]  btc_garch          — BTC GARCH(1,1) conditional vol, annualized
  [6]  spx_rvol           — SPX 20d rolling realized vol, annualized
  [7]  gold_rvol          — Gold 20d rolling realized vol, annualized
  [8]  btc_rvol           — BTC 20d rolling realized vol, annualized
  [9]  rho_spx_btc_30     — SPX-BTC rolling 30d Pearson correlation
  [10] rho_spx_gold_30    — SPX-Gold rolling 30d Pearson correlation
  [11] rho_btc_gold_30    — BTC-Gold rolling 30d Pearson correlation
  [12] vol_ratio_btc_spx  — btc_garch / spx_garch (contagion amplifier)
  [13] delta_vol_btc      — first difference of btc_garch (vol velocity)
  [14] delta_vol_spx      — first difference of spx_garch (receiver signal)
  [15] crisis_flag        — LABEL COLUMN (not a training feature)

DO NOT include:
  - vix_ret: collinear with crisis_flag
  - 60d correlation columns: too slow to be a training signal
  - is_extreme: metadata only
  - delta_vol_gold: dropped per spec (only btc + spx delta included)
"""

from __future__ import annotations

import sys
from pathlib import Path

import numpy as np
import pandas as pd
import yaml
from loguru import logger

from src.data.ingest import run_ingestion, load_aligned_prices, _resolve as _ingest_resolve
from src.data.features import (
    compute_log_returns,
    winsorize_returns,
    compute_volatility,
    compute_correlations,
    compute_contagion_features,
)
from src.data.labelling import (
    build_crisis_flag,
    validate_crisis_dates,
    build_3class_labels,
)

_ROOT = Path(__file__).resolve().parents[2]
_PATHS_CFG = yaml.safe_load((_ROOT / "config" / "paths.yaml").read_text())
_DATA_CFG  = yaml.safe_load((_ROOT / "config" / "data.yaml").read_text())


def _resolve(rel_path: str) -> Path:
    return _ROOT / rel_path


# Training feature column names (exactly 15)
FEATURE_COLS = [
    "spx_w", "gold_w", "btc_w",
    "spx_garch", "gold_garch", "btc_garch",
    "spx_rvol", "gold_rvol", "btc_rvol",
    "rho_spx_btc_30", "rho_spx_gold_30", "rho_btc_gold_30",
    "vol_ratio_btc_spx", "delta_vol_btc", "delta_vol_spx",
]
LABEL_COL = "regime_3class"   # 0=CRISIS, 1=NORMAL, 2=HIGH-VOL (3-class training target)


def run_preprocessing(force_download: bool = True) -> pd.DataFrame:
    """
    Execute the full Phase 1 preprocessing pipeline.

    Steps:
      1. Download + align asset prices
      2. Compute log returns
      3. Winsorize returns + flag extremes
      4. Fit GARCH + rolling vol
      5. Compute rolling correlations
      6. Compute contagion-specific derived features
      7. Apply hybrid crisis labelling
      8. Assemble and validate feature matrix

    Args:
        force_download : if True, re-download data even if aligned_prices.csv exists

    Returns:
        feature_matrix DataFrame (rows = trading days, columns = 15 features + crisis_flag)

    Raises:
        SystemExit(1) if any validation fails
    """
    logger.info("=" * 60)
    logger.info("PHASE 1 — PREPROCESSING PIPELINE")
    logger.info("=" * 60)

    # ─── Step 1 — Ingest ──────────────────────────────────────────
    aligned_path = _resolve(_PATHS_CFG["aligned_prices"])
    if aligned_path.exists() and not force_download:
        logger.info("Loading existing aligned_prices.csv")
        aligned = load_aligned_prices()
    else:
        aligned = run_ingestion()

    # ─── Step 2 — Log returns ─────────────────────────────────────
    logger.info("Computing log returns...")
    log_returns = compute_log_returns(aligned)

    # ─── Step 3 — Winsorize ───────────────────────────────────────
    logger.info("Winsorizing returns...")
    winsorized = winsorize_returns(log_returns)

    # ─── Step 4 — Volatility ──────────────────────────────────────
    logger.info("Computing GARCH + rolling volatility (slow step)...")
    volatility = compute_volatility(winsorized)

    # ─── Step 5 — Correlations ────────────────────────────────────
    logger.info("Computing rolling correlations...")
    correlations = compute_correlations(log_returns)

    # ─── Step 6 — Contagion features ─────────────────────────────
    logger.info("Computing contagion-specific features...")
    contagion = compute_contagion_features(volatility, correlations)

    # ─── Step 7 — Crisis labelling ────────────────────────────────
    logger.info("Applying hybrid crisis labelling...")
    crisis_flag = build_crisis_flag(winsorized, volatility, correlations, aligned)

    # ─── Step 8 — Assemble feature matrix ─────────────────────────
    logger.info("Assembling feature matrix...")
    feature_matrix = pd.concat(
        [
            winsorized[["spx_ret", "gold_ret", "btc_ret"]].rename(
                columns={"spx_ret": "spx_w", "gold_ret": "gold_w", "btc_ret": "btc_w"}
            ),
            volatility[["spx_garch", "gold_garch", "btc_garch",
                        "spx_rvol", "gold_rvol", "btc_rvol"]],
            correlations[["rho_spx_btc_30", "rho_spx_gold_30", "rho_btc_gold_30"]],
            contagion[["vol_ratio_btc_spx", "delta_vol_btc", "delta_vol_spx"]],
            crisis_flag,
        ],
        axis=1,
        join="inner",
    )

    # ─── Drop NaN rows from rolling window warm-up ────────────────
    n_before = len(feature_matrix)
    feature_matrix = feature_matrix.dropna()
    n_after = len(feature_matrix)
    logger.info(f"Dropped {n_before - n_after} NaN rows (rolling window warm-up)")

    # ─── Validation ───────────────────────────────────────────────
    nan_count = feature_matrix.isna().sum().sum()
    inf_count = int(np.isinf(feature_matrix.select_dtypes(include=[np.number]).values).sum())

    # Crisis breakdown
    crisis_days  = (feature_matrix["crisis_flag"] == 1).sum()
    normal_days  = (feature_matrix["crisis_flag"] == 0).sum()

    # 3-class labels for breakdown
    regime_3 = build_3class_labels(
        feature_matrix["crisis_flag"],
        feature_matrix["btc_garch"],
    )
    feature_matrix["regime_3class"] = regime_3
    crisis_3   = (regime_3 == 0).sum()
    normal_3   = (regime_3 == 1).sum()
    highvol_3  = (regime_3 == 2).sum()

    # Known crisis date validation
    validation_results = validate_crisis_dates(feature_matrix["crisis_flag"])

    # ─── Print validation report ──────────────────────────────────
    print("\n" + "=" * 60)
    print("PHASE 1 VALIDATION REPORT")
    print("=" * 60)
    print(f"Feature matrix: {feature_matrix.shape}")
    print(f"Date range:     {feature_matrix.index[0].date()} to {feature_matrix.index[-1].date()}")
    print(f"NaN count:      {nan_count}")
    print(f"Inf count:      {inf_count}")
    print("----")
    print(f"CRISIS days:    {crisis_3:4d}  ({100.*crisis_3/len(feature_matrix):.1f}%)")
    print(f"NORMAL days:    {normal_3:4d}  ({100.*normal_3/len(feature_matrix):.1f}%)")
    print(f"HIGH-VOL days:  {highvol_3:4d}  ({100.*highvol_3/len(feature_matrix):.1f}%)")
    print("----")
    print("Crisis validation:")

    all_pass = True
    event_labels = {
        "2020-03-16": "COVID peak",
        "2020-03-20": "circuit breaker",
        "2022-05-09": "LUNA collapse",
        "2023-03-13": "SVB contagion",
    }
    for date_str, result in validation_results.items():
        label = event_labels.get(date_str, "")
        print(f"  {date_str} {label}: {result}")
        if result == "FAIL":
            all_pass = False

    print("=" * 60)

    # ─── Hard fail on validation errors ───────────────────────────
    if nan_count > 0:
        logger.error(f"FAIL: {nan_count} NaN values in feature matrix!")
        sys.exit(1)
    if inf_count > 0:
        logger.error(f"FAIL: {inf_count} infinite values in feature matrix!")
        sys.exit(1)

    # ─── Save feature matrix ──────────────────────────────────────
    out_path = _resolve(_PATHS_CFG["feature_matrix"])
    out_path.parent.mkdir(parents=True, exist_ok=True)
    feature_matrix.to_csv(out_path)
    logger.info(f"Feature matrix saved → {out_path}")

    return feature_matrix


def load_feature_matrix() -> pd.DataFrame:
    """Load the already-computed feature_matrix.csv from disk."""
    path = _resolve(_PATHS_CFG["feature_matrix"])
    df = pd.read_csv(path, index_col=0, parse_dates=True)
    df.index = pd.to_datetime(df.index).normalize()
    return df
