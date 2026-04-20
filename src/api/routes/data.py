"""
src/api/routes/data.py

API routes for raw data endpoints:
  GET /api/health   — startup check
  GET /api/data/prices  — normalized asset prices indexed to 100
  GET /api/data/regimes — regime labels + crisis flag time series
"""

from __future__ import annotations

from typing import Any, Dict, List

import numpy as np
import pandas as pd
from fastapi import APIRouter, Request
from loguru import logger

from src.api.schemas import HealthResponse, PricePoint, RegimePoint

router = APIRouter()

REGIME_LABELS = {0: "CRISIS", 1: "NORMAL", 2: "HIGH-VOL"}


@router.get("/health", response_model=HealthResponse)
async def health_check(request: Request) -> HealthResponse:
    """
    Health check endpoint.
    Returns the number of models whose metrics are loaded.
    """
    metrics = getattr(request.app.state, "metrics", {})
    loaded  = sum(1 for v in metrics.values() if v is not None)
    return HealthResponse(status="ok", models_loaded=loaded)


@router.get("/data/prices", response_model=List[PricePoint])
async def get_prices(
    request: Request,
    start_date: str = None,
    end_date: str = None,
) -> List[PricePoint]:
    """
    Return SPX, BTC, GOLD normalized to base 100 at start date.
    Optionally filtered by start_date / end_date (YYYY-MM-DD).
    """
    prices: pd.DataFrame = request.app.state.aligned_prices
    if prices is None or prices.empty:
        return []

    # Apply date filter
    if start_date:
        prices = prices[prices.index >= pd.Timestamp(start_date)]
    if end_date:
        prices = prices[prices.index <= pd.Timestamp(end_date)]

    if prices.empty:
        return []

    # Normalize to 100 at the first date in the filtered range
    cols  = ["SPX", "GOLD", "BTC"]
    base  = prices[cols].iloc[0]
    normd = (prices[cols] / base * 100).round(4)

    result = []
    for date, row in normd.iterrows():
        result.append(PricePoint(
            date=str(date.date()),
            spx=float(row.get("SPX", 100.0)),
            gold=float(row.get("GOLD", 100.0)),
            btc=float(row.get("BTC", 100.0)),
        ))
    return result


@router.get("/data/regimes", response_model=List[RegimePoint])
async def get_regimes(
    request: Request,
    start_date: str = None,
    end_date: str = None,
) -> List[RegimePoint]:
    """
    Return the regime sequence over the full date range.
    Optionally filtered by start_date / end_date (YYYY-MM-DD).
    """
    fm: pd.DataFrame = request.app.state.feature_matrix
    if fm is None or fm.empty:
        return []

    # Apply date filter
    if start_date:
        fm = fm[fm.index >= pd.Timestamp(start_date)]
    if end_date:
        fm = fm[fm.index <= pd.Timestamp(end_date)]

    if fm.empty:
        return []

    result = []
    for date, row in fm.iterrows():
        crisis = int(row.get("crisis_flag", 0))
        if "regime_3class" in row:
            label = REGIME_LABELS.get(int(row["regime_3class"]), "NORMAL")
        else:
            label = "CRISIS" if crisis == 1 else "NORMAL"

        result.append(RegimePoint(
            date=str(date.date()),
            regime_label=label,
            crisis_flag=crisis,
            btc_garch=float(row.get("btc_garch", 0.0)),
        ))
    return result
