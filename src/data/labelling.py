"""
src/data/labelling.py

Step 1.8 — Hybrid Crisis Labelling.

This is the most important step in the entire project.
Pure GMM labelling fails because it has zero financial domain knowledge.
The hybrid rule combines two conditions that encode financial domain expertise:

  Condition A (market-structure rule — captures contagion mechanism):
    BTC GARCH vol > μ(non-crisis) + 2σ(non-crisis)
    AND rho_spx_btc_30 < 0
    AND spx_ret < -0.015

  Condition B (fear-regime rule — catches standalone panic events):
    VIX > 30

The μ/σ estimation uses an iterative bootstrap to avoid circular definition:
  Pass 1: compute μ/σ on all days
  Pass 2: compute μ/σ on non-crisis days only (using Pass 1 labels)
"""

from __future__ import annotations

from pathlib import Path

import numpy as np
import pandas as pd
import yaml
from loguru import logger

_ROOT = Path(__file__).resolve().parents[2]
_DATA_CFG  = yaml.safe_load((_ROOT / "config" / "data.yaml").read_text())


def build_crisis_flag(
    winsorized: pd.DataFrame,
    volatility: pd.DataFrame,
    correlations: pd.DataFrame,
    aligned_prices: pd.DataFrame,
) -> pd.Series:
    """
    Build binary crisis_flag using the hybrid labelling rule.

    The hybrid rule fires if EITHER condition is met:
      A: BTC_GARCH > μ + 2σ  AND  rho_spx_btc_30 < 0  AND  spx_ret < -1.5%
      B: VIX > 30

    μ/σ are estimated from NON-crisis days using a one-iteration bootstrap
    to avoid circular definition.

    Args:
        winsorized     : winsorized returns (contains spx_ret)
        volatility     : GARCH and rolling vol columns (contains btc_garch)
        correlations   : rolling correlations (contains rho_spx_btc_30)
        aligned_prices : raw aligned prices (contains VIX)

    Returns:
        pd.Series of int (0=normal, 1=crisis) with the same index as inputs

    Side-effects:
        Prints the rule breakdown and validation report to stdout.
    """
    cfg = _DATA_CFG["crisis_labelling"]
    vix_thresh    = cfg["vix_threshold"]      # 30
    spx_ret_thresh = cfg["spx_ret_threshold"] # -0.015
    sigma_mult    = cfg["btc_garch_sigma"]    # 2.0
    corr_thresh   = cfg["btc_spx_corr_neg"]  # 0.0

    # Align all inputs to common index
    common_idx = (
        winsorized.index
        .intersection(volatility.index)
        .intersection(correlations.index)
        .intersection(aligned_prices.index)
    )
    spx_ret      = winsorized["spx_ret"].reindex(common_idx)
    btc_garch    = volatility["btc_garch"].reindex(common_idx)
    rho_spx_btc  = correlations["rho_spx_btc_30"].reindex(common_idx)
    vix          = aligned_prices["VIX"].reindex(common_idx)

    # ─── Pass 1: initial μ/σ on all days ──────────────────────────
    mu1 = btc_garch.mean()
    sd1 = btc_garch.std()
    threshold1 = mu1 + sigma_mult * sd1

    cond_a_p1 = (
        (btc_garch > threshold1) &
        (rho_spx_btc < corr_thresh) &
        (spx_ret < spx_ret_thresh)
    )
    cond_b_p1 = vix > vix_thresh
    crisis_p1  = (cond_a_p1 | cond_b_p1).astype(int)

    # ─── Pass 2: refine μ/σ using non-crisis days only ────────────
    non_crisis_mask = crisis_p1 == 0
    mu2 = btc_garch[non_crisis_mask].mean()
    sd2 = btc_garch[non_crisis_mask].std()
    threshold2 = mu2 + sigma_mult * sd2

    logger.info(
        f"BTC GARCH threshold (non-crisis μ+2σ): "
        f"μ={mu2:.6f}, σ={sd2:.6f}, threshold={threshold2:.6f}"
    )

    cond_a = (
        (btc_garch > threshold2) &
        (rho_spx_btc < corr_thresh) &
        (spx_ret < spx_ret_thresh)
    )
    cond_b = vix > vix_thresh
    crisis_flag = (cond_a | cond_b).astype(int)

    # ─── Print rule breakdown ──────────────────────────────────────
    only_a = (cond_a & ~cond_b).sum()
    only_b = (~cond_a & cond_b).sum()
    both   = (cond_a & cond_b).sum()
    total  = crisis_flag.sum()

    print("=" * 60)
    print("CRISIS LABELLING BREAKDOWN")
    print("=" * 60)
    print(f"  Condition A alone fired:     {only_a:4d} days")
    print(f"  Condition B alone fired:     {only_b:4d} days")
    print(f"  Both fired:                  {both:4d} days")
    print(f"  Total crisis_flag = 1:       {total:4d} days  ({100.*total/len(crisis_flag):.1f}%)")
    print("=" * 60)

    return crisis_flag.rename("crisis_flag")


def validate_crisis_dates(
    crisis_flag: pd.Series,
    validation_dates: list[str] | None = None,
) -> dict:
    """
    Validate that known crisis dates are correctly labelled as crises.

    Args:
        crisis_flag      : output of build_crisis_flag()
        validation_dates : list of ISO date strings to check

    Returns:
        dict mapping date → "PASS" or "FAIL"
    """
    if validation_dates is None:
        validation_dates = _DATA_CFG["crisis_validation_dates"]

    results = {}
    for date_str in validation_dates:
        ts = pd.Timestamp(date_str)
        # Look for date in index (exact or within 1 business day)
        if ts in crisis_flag.index:
            flag = crisis_flag.loc[ts]
        else:
            # Find nearest trading day
            nearest_idx = crisis_flag.index.get_indexer([ts], method="nearest")
            if nearest_idx[0] >= 0:
                nearest_ts = crisis_flag.index[nearest_idx[0]]
                flag = crisis_flag.iloc[nearest_idx[0]]
                logger.info(f"  {date_str} → nearest trading day {nearest_ts.date()}")
            else:
                flag = 0
        results[date_str] = "PASS" if flag == 1 else "FAIL"

    return results


def build_3class_labels(
    crisis_flag: pd.Series,
    btc_garch: pd.Series,
) -> pd.Series:
    """
    Derive 3-class regime labels from crisis_flag and BTC GARCH volatility.

    Class definitions:
      0 = CRISIS  : Systemic contagion event (crisis_flag=1)
      1 = NORMAL  : Low/medium BTC volatility, no crisis flag
      2 = HIGH-VOL: Elevated BTC volatility, no systemic crisis

    The HIGH-VOL threshold is the median of BTC GARCH on non-crisis days,
    which separates genuinely high-volatility periods from normal operation
    without using crisis information in the threshold calculation.

    Args:
        crisis_flag : binary Series (0/1)
        btc_garch   : BTC GARCH conditional volatility Series

    Returns:
        pd.Series of int with values in {0, 1, 2}
    """
    non_crisis_btc_garch = btc_garch[crisis_flag == 0]
    highvol_threshold = np.median(non_crisis_btc_garch)
    logger.info(f"HIGH-VOL threshold (median non-crisis BTC GARCH): {highvol_threshold:.6f}")

    regime = np.where(
        crisis_flag == 1,
        0,                                          # CRISIS
        np.where(btc_garch > highvol_threshold,
                 2,                                  # HIGH-VOL
                 1)                                  # NORMAL
    )
    return pd.Series(regime, index=crisis_flag.index, name="regime_3class")
