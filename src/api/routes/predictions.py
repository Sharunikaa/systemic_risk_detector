"""
src/api/routes/predictions.py

API routes for quantum and prediction endpoints:
  GET /api/quantum/entanglement   — entanglement entropy time series
  GET /api/quantum/lead-lag       — VQH vs LSTM crisis prob around SVB
  GET /api/predictions/latest     — last 30 days of VQH predictions
"""

from __future__ import annotations

from typing import Any, Dict, List

import pandas as pd
from fastapi import APIRouter, HTTPException, Request

from src.api.schemas import EntanglementPoint, LeadLagPoint, LatestPrediction

router = APIRouter()

REGIME_LABELS = {0: "CRISIS", 1: "NORMAL", 2: "HIGH-VOL"}


@router.get("/quantum/entanglement", response_model=List[EntanglementPoint])
async def get_entanglement(request: Request) -> List[EntanglementPoint]:
    """
    Return the entanglement entropy time series for the test window.
    Requires VQH predictions to have run (Phase 3).
    """
    predictions = getattr(request.app.state, "predictions", {})
    df = predictions.get("vqh")
    if df is None:
        raise HTTPException(
            status_code=404,
            detail="VQH predictions not found. Run scripts/run_phase3_qml.py first.",
        )
    if "entanglement_entropy" not in df.columns:
        raise HTTPException(
            status_code=404,
            detail="entanglement_entropy column not present in VQH predictions.",
        )

    result = []
    for date, row in df.iterrows():
        result.append(EntanglementPoint(
            date        = str(pd.Timestamp(date).date()),
            entropy     = float(row.get("entanglement_entropy", 0.5)),
            true_regime = int(row.get("true_regime", 1)),
        ))
    return result


@router.get("/quantum/lead-lag", response_model=List[LeadLagPoint])
async def get_lead_lag(request: Request) -> List[LeadLagPoint]:
    """
    Return VQH and LSTM crisis probabilities for the SVB 2023 window.
    used by the Quantum Results lead/lag chart.
    """
    predictions = getattr(request.app.state, "predictions", {})
    vqh_df  = predictions.get("vqh")
    lstm_df = predictions.get("lstm")

    if vqh_df is None:
        raise HTTPException(status_code=404, detail="VQH predictions not found.")

    # Filter to SVB analysis window
    start = pd.Timestamp("2023-02-01")
    end   = pd.Timestamp("2023-05-31")

    vqh_slice  = vqh_df[(vqh_df.index >= start) & (vqh_df.index <= end)]
    lstm_slice = lstm_df[(lstm_df.index >= start) & (lstm_df.index <= end)] if lstm_df is not None else None

    result = []
    for date, row in vqh_slice.iterrows():
        ts = pd.Timestamp(date)
        lstm_p = None
        if lstm_slice is not None and ts in lstm_slice.index:
            lstm_p = float(lstm_slice.loc[ts, "prob_crisis"])

        result.append(LeadLagPoint(
            date             = str(ts.date()),
            vqh_prob_crisis  = float(row.get("prob_crisis", 0.0)),
            lstm_prob_crisis = lstm_p,
        ))
    return result


@router.get("/predictions/latest", response_model=List[LatestPrediction])
async def get_latest_predictions(request: Request, days: int = 30) -> List[LatestPrediction]:
    """
    Return the last `days` predictions from the VQH model.
    Used by the Predictions page log table.
    """
    predictions = getattr(request.app.state, "predictions", {})
    df = predictions.get("vqh")
    if df is None:
        raise HTTPException(
            status_code=404,
            detail="VQH predictions not found. Run Phase 3 first.",
        )

    recent = df.tail(days)
    result = []
    for date, row in recent.iterrows():
        true_r = int(row.get("true_regime", 1))
        pred_r = int(row.get("pred_regime", 1))
        result.append(LatestPrediction(
            date         = str(pd.Timestamp(date).date()),
            true_regime  = true_r,
            pred_regime  = pred_r,
            prob_crisis  = float(row.get("prob_crisis", 0.0)),
            prob_normal  = float(row.get("prob_normal", 1.0)),
            prob_highvol = float(row.get("prob_highvol", 0.0)),
            match        = (true_r == pred_r),
            entanglement_entropy = (
                float(row["entanglement_entropy"])
                if "entanglement_entropy" in row else None
            ),
        ))
    return result
