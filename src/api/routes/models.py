"""
src/api/routes/models.py

API routes for model data:
  GET  /api/models/metrics            — all 5 model metric dicts
  GET  /api/models/benchmark          — cross-model comparison
  GET  /api/models/{name}/predictions — prediction time series for one model
  GET  /api/models/{name}/confusion-matrix — 3×3 confusion matrix
  GET  /api/models/{name}/pr-curve    — precision-recall curve data
  POST /api/models/predict            — run all models on custom input features
"""

from __future__ import annotations

from typing import Any, Dict, List

import numpy as np
import pandas as pd
from fastapi import APIRouter, HTTPException, Request
from loguru import logger

from src.api.schemas import (
    BenchmarkComparison,
    ConfusionMatrixResponse,
    CustomPredictionInput,
    CustomPredictionResponse,
    ModelPrediction,
    PredictionPoint,
)

router  = APIRouter()
MODELS  = ["hmm", "xgboost", "lstm", "tft", "vqh"]
CLASSES = ["CRISIS", "NORMAL", "HIGH-VOL"]


@router.get("/models/metrics")
async def get_all_metrics(request: Request) -> Dict[str, Any]:
    """Return all loaded model metrics dicts."""
    metrics = getattr(request.app.state, "metrics", {})
    return {k: v for k, v in metrics.items() if v is not None}


@router.get("/models/benchmark")
async def get_benchmark(request: Request) -> Dict[str, Any]:
    """Return cross-model benchmark comparison."""
    benchmark = getattr(request.app.state, "benchmark", None)
    if benchmark is None:
        raise HTTPException(
            status_code=404,
            detail="Benchmark results not found. Run scripts/run_phase2_classical.py first.",
        )
    return benchmark


@router.get("/models/{model_name}/predictions", response_model=List[PredictionPoint])
async def get_model_predictions(
    model_name: str,
    request: Request,
) -> List[PredictionPoint]:
    """
    Return prediction time series for a specific model.

    Args:
        model_name : one of hmm, xgboost, lstm, tft, vqh
    """
    name = model_name.lower()
    if name not in MODELS:
        raise HTTPException(status_code=404, detail=f"Unknown model: {model_name}")

    predictions = getattr(request.app.state, "predictions", {})
    df = predictions.get(name)
    if df is None:
        raise HTTPException(
            status_code=404,
            detail=f"{model_name} predictions not found. Run the corresponding training script.",
        )

    result = []
    for date, row in df.iterrows():
        result.append(PredictionPoint(
            date       = str(pd.Timestamp(date).date()),
            true_regime = int(row.get("true_regime", 1)),
            pred_regime = int(row.get("pred_regime", 1)),
            prob_crisis = float(row.get("prob_crisis", 0.0)),
            prob_normal = float(row.get("prob_normal", 1.0)),
            prob_highvol = float(row.get("prob_highvol", 0.0)),
        ))
    return result


@router.get("/models/{model_name}/confusion-matrix", response_model=ConfusionMatrixResponse)
async def get_confusion_matrix(
    model_name: str,
    request: Request,
) -> ConfusionMatrixResponse:
    """Return the 3×3 confusion matrix for a specific model."""
    name = model_name.lower()
    metrics = getattr(request.app.state, "metrics", {})
    m = metrics.get(name)
    if m is None:
        raise HTTPException(
            status_code=404,
            detail=f"{model_name} metrics not found.",
        )

    cm = m.get("confusion_matrix", [[0, 0, 0], [0, 0, 0], [0, 0, 0]])
    return ConfusionMatrixResponse(
        matrix     = cm,
        class_names = CLASSES,
        model_name = model_name.upper(),
    )


@router.get("/models/{model_name}/pr-curve")
async def get_pr_curve(model_name: str, request: Request) -> Dict[str, Any]:
    """Return precision-recall curve data for crisis class."""
    name = model_name.lower()
    pr_curves = getattr(request.app.state, "pr_curves", {})
    curve = pr_curves.get(name)
    if curve is None:
        raise HTTPException(
            status_code=404,
            detail=f"PR curve not found for {model_name}.",
        )
    return curve


@router.post("/models/predict", response_model=CustomPredictionResponse)
async def predict_custom(
    body: CustomPredictionInput,
    request: Request,
) -> CustomPredictionResponse:
    """
    Run all available models on a single custom feature vector.

    The input is mapped to the 15-feature vector used during training:
      [spx_w, gold_w, btc_w,
       spx_garch, gold_garch, btc_garch,
       spx_rvol, gold_rvol, btc_rvol,
       rho_spx_btc_30, rho_spx_gold_30, rho_btc_gold_30,
       vol_ratio_btc_spx, delta_vol_btc, delta_vol_spx]

    Missing features are filled with dataset medians loaded at startup.
    """
    import torch

    # Build feature vector from input
    # Map the 10 user-facing fields to the 15 training features
    btc_vol  = float(body.btc_volatility)
    spx_vol  = float(body.spx_volatility)
    gold_vol = float(body.gold_volatility)

    feature_vec = np.array([[
        float(body.spx_return),                          # spx_w
        float(body.gold_return),                         # gold_w
        float(body.btc_return),                          # btc_w
        spx_vol,                                         # spx_garch (proxy)
        gold_vol,                                        # gold_garch (proxy)
        btc_vol,                                         # btc_garch (proxy)
        spx_vol,                                         # spx_rvol (proxy)
        gold_vol,                                        # gold_rvol (proxy)
        btc_vol,                                         # btc_rvol (proxy)
        float(body.btc_spx_corr),                        # rho_spx_btc_30
        float(body.spx_gold_corr),                       # rho_spx_gold_30
        float(body.btc_gold_corr),                       # rho_btc_gold_30
        btc_vol / max(spx_vol, 1e-6),                    # vol_ratio_btc_spx
        0.0,                                             # delta_vol_btc (no history)
        0.0,                                             # delta_vol_spx (no history)
    ]], dtype=np.float32)

    # Load models from app state
    loaded_models = getattr(request.app.state, "loaded_models", {})
    benchmark     = getattr(request.app.state, "benchmark", {}) or {}
    best_model    = benchmark.get("best_model_by_crisis_pr_auc", "XGBoost")

    predictions_out: List[ModelPrediction] = []

    # ── HMM ──────────────────────────────────────────────────────────────────
    hmm_model = loaded_models.get("hmm")
    if hmm_model is not None:
        try:
            proba = hmm_model.predict_proba(feature_vec)[0]
            pred  = int(proba.argmax())
            predictions_out.append(ModelPrediction(
                model_name  = "HMM",
                pred_regime = pred,
                prob_crisis = float(proba[0]),
                prob_normal = float(proba[1]),
                prob_highvol= float(proba[2]),
                confidence  = float(proba.max()),
            ))
        except Exception as e:
            logger.error(f"HMM predict failed: {type(e).__name__}: {e}")

    # ── XGBoost ──────────────────────────────────────────────────────────────
    xgb_model = loaded_models.get("xgboost")
    if xgb_model is not None:
        try:
            proba = xgb_model.predict_proba(feature_vec)[0]
            pred  = int(proba.argmax())
            predictions_out.append(ModelPrediction(
                model_name  = "XGBoost",
                pred_regime = pred,
                prob_crisis = float(proba[0]),
                prob_normal = float(proba[1]),
                prob_highvol= float(proba[2]),
                confidence  = float(proba.max()),
            ))
        except Exception as e:
            logger.error(f"XGBoost predict failed: {type(e).__name__}: {e}")

    # ── LSTM ─────────────────────────────────────────────────────────────────
    lstm_model = loaded_models.get("lstm")
    if lstm_model is not None:
        try:
            lstm_model.eval()
            # LSTM expects (batch, seq_len, features) — repeat single step 20x
            seq = torch.tensor(
                np.repeat(feature_vec, 20, axis=0)[np.newaxis, :, :],
                dtype=torch.float32
            )
            with torch.no_grad():
                logits = lstm_model(seq)
                proba  = torch.softmax(logits, dim=1).squeeze().numpy()
            pred = int(proba.argmax())
            predictions_out.append(ModelPrediction(
                model_name  = "LSTM",
                pred_regime = pred,
                prob_crisis = float(proba[0]),
                prob_normal = float(proba[1]),
                prob_highvol= float(proba[2]),
                confidence  = float(proba.max()),
            ))
        except Exception as e:
            logger.warning(f"LSTM predict failed: {e}")

    # ── TFT ──────────────────────────────────────────────────────────────────
    tft_model = loaded_models.get("tft")
    if tft_model is not None:
        try:
            tft_model.eval()
            seq = torch.tensor(
                np.repeat(feature_vec, 20, axis=0)[np.newaxis, :, :],
                dtype=torch.float32
            )
            with torch.no_grad():
                logits = tft_model(seq)
                proba  = torch.softmax(logits, dim=1).squeeze().numpy()
            pred = int(proba.argmax())
            predictions_out.append(ModelPrediction(
                model_name  = "TFT",
                pred_regime = pred,
                prob_crisis = float(proba[0]),
                prob_normal = float(proba[1]),
                prob_highvol= float(proba[2]),
                confidence  = float(proba.max()),
            ))
        except Exception as e:
            logger.warning(f"TFT predict failed: {e}")

    # ── VQH ──────────────────────────────────────────────────────────────────
    # VQH takes (prob_crisis, prob_normal, prob_highvol) from best classical model
    # Use XGBoost probs if available, else uniform
    xgb_pred = next((p for p in predictions_out if p.model_name == "XGBoost"), None)
    vqh_model = loaded_models.get("vqh")
    if vqh_model is not None and xgb_pred is not None:
        try:
            vqh_model.eval()
            inp = torch.tensor(
                [[xgb_pred.prob_crisis, xgb_pred.prob_normal, xgb_pred.prob_highvol]],
                dtype=torch.float32
            )
            with torch.no_grad():
                logits = vqh_model(inp)
                proba  = torch.softmax(logits, dim=1).squeeze().numpy()
            pred = int(proba.argmax())
            predictions_out.append(ModelPrediction(
                model_name  = "VQH",
                pred_regime = pred,
                prob_crisis = float(proba[0]),
                prob_normal = float(proba[1]),
                prob_highvol= float(proba[2]),
                confidence  = float(proba.max()),
            ))
        except Exception as e:
            logger.warning(f"VQH predict failed: {e}")

    if not predictions_out:
        raise HTTPException(
            status_code=503,
            detail="No models are loaded. Run the training pipeline first.",
        )

    return CustomPredictionResponse(
        input_features = body,
        predictions    = predictions_out,
        best_model     = best_model,
    )
