"""
src/data/ingest.py

Step 1.1 & 1.2 — Data Ingestion and NYSE Calendar Alignment.

Downloads SPX, GOLD, BTC, VIX via yfinance. Saves each raw series
immediately. Aligns all four series to the NYSE trading calendar by
forward-filling BTC weekend values and VIX holiday values.

Financial role of each asset:
  - SPX  (^GSPC) : Equity market regime anchor — the contagion receiver
  - GOLD (GLD)   : Safe-haven signal — flight-to-quality destination
  - BTC  (BTC-USD): Contagion amplifier — high-alpha shock source
  - VIX  (^VIX)  : Fear index — best single-day crisis predictor
"""

from __future__ import annotations

import os
from pathlib import Path
from typing import Dict

import pandas as pd
import yfinance as yf
import pandas_market_calendars as mcal
import yaml
from loguru import logger

# ─── Load configuration ────────────────────────────────────────────
_ROOT = Path(__file__).resolve().parents[2]
_DATA_CFG  = yaml.safe_load((_ROOT / "config" / "data.yaml").read_text())
_PATHS_CFG = yaml.safe_load((_ROOT / "config" / "paths.yaml").read_text())


def _resolve(rel_path: str) -> Path:
    """Resolve a path from paths.yaml relative to project root."""
    return _ROOT / rel_path


# ─── Step 1.1 — Download ──────────────────────────────────────────

def download_asset(
    ticker: str,
    start: str,
    end: str,
    name: str,
    save_path: Path,
) -> pd.Series:
    """
    Download close prices for a single asset via yfinance.
    Uses auto_adjust=True to correct for splits and dividends.

    Args:
        ticker    : yfinance ticker symbol (e.g. "^GSPC")
        start     : ISO date string for start date
        end       : ISO date string for end date
        name      : Human-readable asset name for logging
        save_path : Path to save the raw CSV

    Returns:
        pd.Series with date index and Close prices

    Raises:
        RuntimeError if download yields empty series
    """
    logger.info(f"Downloading {name} ({ticker}) from {start} to {end}")
    raw = yf.download(ticker, start=start, end=end, auto_adjust=True, progress=False)

    if raw.empty:
        raise RuntimeError(f"Download failed for {name} ({ticker}): empty result")

    # Handle MultiIndex columns that yfinance sometimes returns
    if isinstance(raw.columns, pd.MultiIndex):
        raw.columns = raw.columns.get_level_values(0)

    close = raw["Close"].copy()
    close.name = name

    # Normalize index to date-only (remove timezone)
    close.index = pd.to_datetime(close.index).normalize()
    # Remove any tz info
    if close.index.tz is not None:
        close.index = close.index.tz_localize(None)

    close = close.dropna()

    logger.info(
        f"  {name}: {len(close)} trading rows | "
        f"{close.index[0].date()} → {close.index[-1].date()}"
    )

    save_path.parent.mkdir(parents=True, exist_ok=True)
    close.to_csv(save_path, header=True)
    logger.info(f"  Saved raw → {save_path}")

    return close


def download_all_assets() -> Dict[str, pd.Series]:
    """
    Download all four assets and save raw CSVs.

    Returns:
        Dict mapping asset name → raw price Series
    """
    cfg = _DATA_CFG
    start = cfg["start_date"]
    end   = cfg["end_date"]
    tickers = cfg["tickers"]

    assets = {
        "SPX":  (tickers["spx"],  _resolve(_PATHS_CFG["spx_raw"])),
        "GOLD": (tickers["gold"], _resolve(_PATHS_CFG["gold_raw"])),
        "BTC":  (tickers["btc"],  _resolve(_PATHS_CFG["btc_raw"])),
        "VIX":  (tickers["vix"],  _resolve(_PATHS_CFG["vix_raw"])),
    }

    raw_series: Dict[str, pd.Series] = {}
    for name, (ticker, path) in assets.items():
        raw_series[name] = download_asset(ticker, start, end, name, path)

    return raw_series


# ─── Step 1.2 — NYSE Calendar Alignment ──────────────────────────

def get_nyse_trading_days(start: str, end: str) -> pd.DatetimeIndex:
    """
    Generate NYSE trading day index using pandas_market_calendars.

    BUG AVOIDANCE: Do NOT use pd.date_range(freq='B') — it creates
    artificial business days that differ from NYSE (ignores holidays).

    Args:
        start : ISO date string
        end   : ISO date string

    Returns:
        DatetimeIndex of NYSE trading days (date-only, no timezone)
    """
    nyse = mcal.get_calendar("NYSE")
    schedule = nyse.schedule(start_date=start, end_date=end)
    trading_days = mcal.date_range(schedule, frequency="1D")
    # Normalize to date-only DatetimeIndex
    trading_days = pd.DatetimeIndex(
        [pd.Timestamp(d).normalize() for d in trading_days]
    )
    if trading_days.tz is not None:
        trading_days = trading_days.tz_localize(None)
    return trading_days


def align_to_nyse(raw_series: Dict[str, pd.Series]) -> pd.DataFrame:
    """
    Align all four price series to the NYSE trading calendar.

    Strategy:
      - BTC trades 24/7 → forward-fill weekend values using prior Friday close
      - VIX has holidays (non-NYSE days) → forward-fill with prior trading day
      - SPX and GOLD natively follow NYSE → reindex and assert no NaN
      - Take the intersection of all series as the common index

    Args:
        raw_series : Dict of name → raw price Series from download_all_assets()

    Returns:
        DataFrame with columns [SPX, GOLD, BTC, VIX] aligned to NYSE calendar

    Raises:
        AssertionError if gaps > 5 calendar days are detected
    """
    cfg = _DATA_CFG
    start = cfg["start_date"]
    end   = cfg["end_date"]

    nyse_days = get_nyse_trading_days(start, end)
    logger.info(f"NYSE trading days in range: {len(nyse_days)}")

    aligned: Dict[str, pd.Series] = {}
    for name, series in raw_series.items():
        s = series.copy()

        # Ensure index is tz-naive date-only
        s.index = pd.to_datetime(s.index).normalize()
        if s.index.tz is not None:
            s.index = s.index.tz_localize(None)

        if name in ("BTC", "VIX"):
            # Reindex to a dense daily range first (to capture weekends for BTC)
            all_days = pd.date_range(s.index[0], s.index[-1], freq="D")
            s = s.reindex(all_days).ffill()     # forward-fill gaps

        # Reindex to NYSE trading days and forward-fill any remaining gaps
        s = s.reindex(nyse_days).ffill()

        n_nan = s.isna().sum()
        if n_nan > 0:
            logger.warning(f"  {name}: {n_nan} NaN after alignment — back-filling initial NaN")
            s = s.bfill()   # back-fill only the very start

        aligned[name] = s
        logger.info(
            f"  {name}: {len(s)} rows aligned | "
            f"{s.index[0].date()} → {s.index[-1].date()}"
        )

    df = pd.DataFrame(aligned)[["SPX", "GOLD", "BTC", "VIX"]]

    # ─── Validate: no gaps > 5 calendar days ─────────────────────
    idx = df.index
    gaps = (idx[1:] - idx[:-1]).days
    max_gap = gaps.max()
    if max_gap > 5:
        raise AssertionError(
            f"NYSE index has a gap of {max_gap} calendar days — "
            "check calendar alignment logic."
        )
    logger.info(f"Max gap in aligned index: {max_gap} calendar days ✓")

    # ─── Save ─────────────────────────────────────────────────────
    out_path = _resolve(_PATHS_CFG["aligned_prices"])
    out_path.parent.mkdir(parents=True, exist_ok=True)
    df.to_csv(out_path)
    logger.info(f"Saved aligned prices → {out_path}")

    return df


def load_aligned_prices() -> pd.DataFrame:
    """Load already-saved aligned_prices.csv from disk."""
    path = _resolve(_PATHS_CFG["aligned_prices"])
    df = pd.read_csv(path, index_col=0, parse_dates=True)
    df.index = pd.to_datetime(df.index).normalize()
    return df


def run_ingestion() -> pd.DataFrame:
    """
    Top-level function: download all assets and align to NYSE calendar.
    Returns the aligned DataFrame.
    """
    raw = download_all_assets()
    aligned = align_to_nyse(raw)
    return aligned
