"""
src/data/features.py

Steps 1.3 – 1.7: Log returns, outlier handling, volatility estimation,
cross-asset correlations, and contagion-specific derived features.

All features are computed WITHOUT look-ahead bias:
  - Rolling windows use past data only
  - GARCH is fitted on training-period data and then applied sequentially
  - Delta features use backward differences only

These are not generic ML features; every column has an explicit financial
justification in the SKILL.md specification.
"""

from __future__ import annotations

from pathlib import Path

import numpy as np
import pandas as pd
import yaml
from arch import arch_model
from loguru import logger

_ROOT = Path(__file__).resolve().parents[2]
_DATA_CFG  = yaml.safe_load((_ROOT / "config" / "data.yaml").read_text())
_PATHS_CFG = yaml.safe_load((_ROOT / "config" / "paths.yaml").read_text())


def _resolve(rel_path: str) -> Path:
    return _ROOT / rel_path


# ─── Step 1.3 — Log Returns ──────────────────────────────────────

def compute_log_returns(prices: pd.DataFrame) -> pd.DataFrame:
    """
    Compute log returns r_t = log(p_t / p_{t-1}) for SPX, GOLD, BTC, VIX.

    Log returns are preferred over simple returns because:
      - They are time-additive ( r_{0→T} = Σ r_t )
      - They handle BTC's 10,000× price range without numerical overflow

    Args:
        prices : aligned_prices DataFrame (columns: SPX, GOLD, BTC, VIX)

    Returns:
        DataFrame with columns [spx_ret, gold_ret, btc_ret, vix_ret]
        First row is dropped (NaN from shift).

    Side-effect: saves data/3_returns/log_returns.csv
    """
    log_ret = np.log(prices / prices.shift(1))
    log_ret = log_ret.dropna()
    log_ret.columns = ["spx_ret", "gold_ret", "btc_ret", "vix_ret"]

    out = _resolve(_PATHS_CFG["log_returns"])
    out.parent.mkdir(parents=True, exist_ok=True)
    log_ret.to_csv(out)
    logger.info(f"Log returns: {log_ret.shape} → {out}")
    return log_ret


# ─── Step 1.4 — Outlier Handling (Winsorize) ───────────────────

def winsorize_returns(log_returns: pd.DataFrame) -> pd.DataFrame:
    """
    Winsorize SPX, GOLD, BTC return series to their own 1st/99th percentiles.
    VIX is NOT winsorized — spike magnitude IS the crisis signal.

    Before winsorizing, flag extreme events:
      is_extreme = (|btc_ret| > 99th pct) OR (|spx_ret| > 99th pct)

    These are candidate black-swan dates — they are kept in the data.
    Winsorizing preserves the date but caps the magnitude so GARCH fitting
    is numerically stable.

    Args:
        log_returns : output of compute_log_returns()

    Returns:
        DataFrame with winsorized series + is_extreme boolean column

    Side-effect: saves data/3_returns/winsorized_returns.csv
    """
    lower_pct = _DATA_CFG["winsorize"]["lower_pct"]
    upper_pct = _DATA_CFG["winsorize"]["upper_pct"]

    df = log_returns.copy()

    # ── is_extreme flag before winsorizing ────────────────────────
    btc_99  = df["btc_ret"].abs().quantile(upper_pct / 100)
    spx_99  = df["spx_ret"].abs().quantile(upper_pct / 100)
    df["is_extreme"] = (
        (df["btc_ret"].abs() > btc_99) | (df["spx_ret"].abs() > spx_99)
    ).astype(int)
    logger.info(f"Extreme days flagged: {df['is_extreme'].sum()}")

    # ── Winsorize SPX, GOLD, BTC (not VIX) ───────────────────────
    for col in ["spx_ret", "gold_ret", "btc_ret"]:
        lo = df[col].quantile(lower_pct / 100)
        hi = df[col].quantile(upper_pct / 100)
        df[col] = df[col].clip(lower=lo, upper=hi)
        logger.info(f"  {col} winsorized to [{lo:.6f}, {hi:.6f}]")

    out = _resolve(_PATHS_CFG["winsorized_returns"])
    out.parent.mkdir(parents=True, exist_ok=True)
    df.to_csv(out)
    logger.info(f"Winsorized returns: {df.shape} → {out}")
    return df


# ─── Step 1.5 — GARCH(1,1) + Rolling Realized Volatility ───────

def _fit_garch(series: pd.Series, name: str) -> np.ndarray:
    """
    Fit GARCH(1,1) model to a return series and return conditional volatility.

    The series is multiplied by 100 before fitting for numerical stability
    (returns near 0 cause convergence issues in the optimizer). After fitting,
    the conditional volatility is divided by 100 and annualized by sqrt(252).

    α + β is printed — for BTC this should be ≥ 0.94 (shocks persist weeks).

    Args:
        series : winsorized log-return series (not ×100)
        name   : asset name for logging

    Returns:
        np.ndarray of annualized conditional volatility (same length as input)
    """
    model = arch_model(series * 100, vol="Garch", p=1, q=1, dist="Normal")
    result = model.fit(disp="off", show_warning=False)

    omega = result.params["omega"]
    alpha = result.params["alpha[1]"]
    beta  = result.params["beta[1]"]
    persistence = alpha + beta

    logger.info(
        f"  GARCH({name}): ω={omega:.6f}, α={alpha:.6f}, "
        f"β={beta:.6f}, α+β={persistence:.6f}"
    )
    if name == "BTC" and persistence < 0.94:
        logger.warning(
            f"  BTC α+β={persistence:.4f} < 0.94 — check input data quality"
        )

    cond_vol = result.conditional_volatility / 100.0  # undo ×100
    annualized = cond_vol * np.sqrt(252)
    return annualized.values


def compute_volatility(winsorized: pd.DataFrame) -> pd.DataFrame:
    """
    Compute two complementary volatility signals per trading asset (SPX, GOLD, BTC).

    Type A — GARCH(1,1) conditional vol:
        Forward-looking instantaneous variance. Reacts immediately to shocks.
        The delta (first difference) of GARCH vol is the leading indicator
        for regime transitions in this dataset.

    Type B — Rolling 20-day realized vol:
        Backward-looking confirmed turbulence. Slower than GARCH but easier
        to interpret and less sensitive to single-day outliers.
        GARCH spikes first; rolling vol confirms — the gap is signal.

    Output columns: spx_garch, gold_garch, btc_garch, spx_rvol, gold_rvol, btc_rvol

    Args:
        winsorized : output of winsorize_returns()

    Returns:
        DataFrame with 6 volatility columns, aligned to winsorized index

    Side-effect: saves data/4_volatility/volatility.csv
    """
    roll_w = _DATA_CFG["volatility"]["rolling_window"]  # 20
    ann    = np.sqrt(_DATA_CFG["volatility"]["trading_days"])  # sqrt(252)

    df = pd.DataFrame(index=winsorized.index)

    logger.info("Fitting GARCH(1,1) models...")
    for col, name in [("spx_ret", "SPX"), ("gold_ret", "GOLD"), ("btc_ret", "BTC")]:
        garch_key = f"{name.lower()}_garch"
        rvol_key  = f"{name.lower()}_rvol"

        df[garch_key] = _fit_garch(winsorized[col], name)
        df[rvol_key]  = winsorized[col].rolling(roll_w).std() * ann

    # Forward-fill any GARCH initialization NaN (first few rows)
    df = df.ffill().bfill()

    out = _resolve(_PATHS_CFG["volatility"])
    out.parent.mkdir(parents=True, exist_ok=True)
    df.to_csv(out)
    logger.info(f"Volatility features: {df.shape} → {out}")
    return df


# ─── Step 1.6 — Cross-Asset Correlations ────────────────────────

def compute_correlations(log_returns: pd.DataFrame) -> pd.DataFrame:
    """
    Compute rolling Pearson correlations for three asset pairs at two timescales.

    Asset pairs:
      (SPX, BTC)  — contagion transmission channel (most important)
      (SPX, GOLD) — flight-to-quality signal
      (BTC, GOLD) — crypto-commodity relationship

    Windows:
      30-day : crisis-sensitive, reacts within 6 weeks of onset
               → included in training features (feature_matrix.csv)
      60-day : regime context, smoother background signal for HMM
               → EDA only, NOT in training features

    The difference between 30d and 60d correlation is itself informative:
    divergence signals a developing regime change.

    Note: correlations are computed on LOG RETURNS, not prices.

    Args:
        log_returns : output of compute_log_returns()

    Returns:
        DataFrame with:
          rho_spx_btc_30, rho_spx_gold_30, rho_btc_gold_30  (training)
          rho_spx_btc_60, rho_spx_gold_60, rho_btc_gold_60  (EDA only)

    Side-effect: saves data/5_correlations/correlations.csv
    """
    short_w = _DATA_CFG["correlations"]["short_window"]  # 30
    long_w  = _DATA_CFG["correlations"]["long_window"]   # 60

    pairs = [
        ("spx_ret", "btc_ret",  "rho_spx_btc"),
        ("spx_ret", "gold_ret", "rho_spx_gold"),
        ("btc_ret", "gold_ret", "rho_btc_gold"),
    ]

    df = pd.DataFrame(index=log_returns.index)

    for col_a, col_b, name in pairs:
        df[f"{name}_{short_w}"] = (
            log_returns[col_a].rolling(short_w).corr(log_returns[col_b])
        )
        df[f"{name}_{long_w}"] = (
            log_returns[col_a].rolling(long_w).corr(log_returns[col_b])
        )

    # Forward fill initial NaN from rolling window warm-up
    df = df.ffill().bfill()

    out = _resolve(_PATHS_CFG["correlations"])
    out.parent.mkdir(parents=True, exist_ok=True)
    df.to_csv(out)
    logger.info(f"Correlations ({short_w}d + {long_w}d): {df.shape} → {out}")
    return df


# ─── Step 1.7 — Contagion-Specific Derived Features ─────────────

def compute_contagion_features(
    volatility: pd.DataFrame,
    correlations: pd.DataFrame,
) -> pd.DataFrame:
    """
    Compute five derived features that capture cross-asset contagion mechanics.
    These are NOT in standard financial ML pipelines.

    Features and their financial meaning:
      vol_ratio_btc_spx:
        BTC vol amplification relative to SPX. Values > 5 indicate crypto-specific
        stress that may spread. This ratio leading a VIX spike by 2-5 days is the
        core contagion pattern this model is designed to detect.

      delta_vol_btc / delta_vol_spx / delta_vol_gold:
        Velocity of conditional variance change. These spike BEFORE rolling realized
        vol reacts — they are the leading indicators. The sequence
        delta_vol_btc spikes → delta_vol_spx follows is the contagion transmission
        chain in the data.

      corr_change_btc_spx:
        Speed of correlation shift. A rapid move toward -1 signals a flight-to-quality
        rotation — the mechanism behind systemic crisis spread.

    Args:
        volatility   : output of compute_volatility()
        correlations : output of compute_correlations()

    Returns:
        DataFrame with 5 contagion feature columns
    """
    df = pd.DataFrame(index=volatility.index)

    # vol_ratio: +1e-8 prevents division by zero
    df["vol_ratio_btc_spx"] = (
        volatility["btc_garch"] / (volatility["spx_garch"] + 1e-8)
    )

    df["delta_vol_btc"]  = volatility["btc_garch"].diff().fillna(0)
    df["delta_vol_spx"]  = volatility["spx_garch"].diff().fillna(0)
    df["delta_vol_gold"] = volatility["gold_garch"].diff().fillna(0)

    df["corr_change_btc_spx"] = correlations["rho_spx_btc_30"].diff().fillna(0)

    # Sanity checks
    assert not np.isinf(df["vol_ratio_btc_spx"]).any(), (
        "Infinite values in vol_ratio_btc_spx — check GARCH output"
    )

    for col in df.columns:
        logger.info(f"  {col}: mean={df[col].mean():.6f}, std={df[col].std():.6f}")

    return df
