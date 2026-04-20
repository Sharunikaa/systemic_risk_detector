"""
src/api/main.py

FastAPI application entry point.

ALL data is loaded ONCE at startup — never read from disk per request.
This is critical for performance: CSV reads on each request would add 50-200ms
and hold file locks on Windows.

CORS is configured to allow the React dev server (localhost:5173) to make
API requests to the FastAPI server (localhost:8000).
"""

from __future__ import annotations

import json
from contextlib import asynccontextmanager
from pathlib import Path
from typing import Optional

import pandas as pd
import yaml
from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware
from loguru import logger

from src.api.routes import data, models, predictions
from src.models.hmm_model import HMMRegimeModel
from src.models.xgboost_model import XGBoostRegimeModel
from src.models.lstm_model import LSTMRegimeModel
from src.models.transformer_model import TemporalFusionTransformer
from src.models.vqh_model import VQHModel
from src.utils.io import load_pkl, load_pt

_ROOT = Path(__file__).resolve().parents[2]
_PATHS = yaml.safe_load((_ROOT / "config" / "paths.yaml").read_text())


def _resolve(rel: str) -> Path:
    return _ROOT / rel


def _load_json_safe(path: Path) -> Optional[dict]:
    """Load JSON file; return None if not found (model may not have run yet)."""
    if not path.exists():
        logger.warning(f"File not found (model may not have run): {path}")
        return None
    with open(path) as f:
        return json.load(f)


def _load_csv_safe(path: Path) -> Optional[pd.DataFrame]:
    """Load CSV; return None if not found."""
    if not path.exists():
        logger.warning(f"CSV not found: {path}")
        return None
    return pd.read_csv(path, index_col=0, parse_dates=True)


# ─── App factory ─────────────────────────────────────────────────

def _load_all_state(app: FastAPI) -> None:
    """Load all pipeline outputs once at startup."""
    logger.info("Loading all data files at startup...")

    # ── Feature matrix + aligned prices ─────────────────────
    app.state.feature_matrix = _load_csv_safe(_resolve(_PATHS["feature_matrix"]))
    app.state.aligned_prices = _load_csv_safe(_resolve(_PATHS["aligned_prices"]))

    # ── Model metrics ─────────────────────────────────────────
    app.state.metrics = {
        "hmm":     _load_json_safe(_resolve(_PATHS["hmm_metrics"])),
        "xgboost": _load_json_safe(_resolve(_PATHS["xgb_metrics"])),
        "lstm":    _load_json_safe(_resolve(_PATHS["lstm_metrics"])),
        "tft":     _load_json_safe(_resolve(_PATHS["tft_metrics"])),
        "vqh":     _load_json_safe(_resolve(_PATHS["vqh_metrics"])),
    }

    # ── Predictions ──────────────────────────────────────────
    app.state.predictions = {
        "hmm":     _load_csv_safe(_resolve(_PATHS["hmm_predictions"])),
        "xgboost": _load_csv_safe(_resolve(_PATHS["xgb_predictions"])),
        "lstm":    _load_csv_safe(_resolve(_PATHS["lstm_predictions"])),
        "tft":     _load_csv_safe(_resolve(_PATHS["tft_predictions"])),
        "vqh":     _load_csv_safe(_resolve(_PATHS["vqh_predictions"])),
    }

    # ── Benchmark ────────────────────────────────────────────
    final_path  = _resolve(_PATHS["benchmark_final"])
    phase2_path = _resolve(_PATHS["benchmark_results"])
    if final_path.exists():
        app.state.benchmark = _load_json_safe(final_path)
    elif phase2_path.exists():
        app.state.benchmark = _load_json_safe(phase2_path)
    else:
        app.state.benchmark = None

    # ── PR Curves ────────────────────────────────────────────
    pr_dir = _ROOT / "outputs" / "metrics"
    app.state.pr_curves = {}
    for name in ["hmm", "xgb", "lstm", "tft", "vqh"]:
        curve_path = pr_dir / f"{name}_pr_curve.json"
        key = "xgboost" if name == "xgb" else name
        app.state.pr_curves[key] = _load_json_safe(curve_path)

    # ── Load model objects for /api/models/predict ───────────
    app.state.loaded_models = {}
    _cfg = yaml.safe_load((_ROOT / "config" / "model.yaml").read_text())

    try:
        import pickle
        with open(_resolve(_PATHS["hmm_model"]), "rb") as f:
            app.state.loaded_models["hmm"] = pickle.load(f)
        logger.info("HMM model object loaded.")
    except Exception as e:
        logger.warning(f"Could not load HMM model object: {e}")

    try:
        import pickle
        with open(_resolve(_PATHS["xgb_model"]), "rb") as f:
            app.state.loaded_models["xgboost"] = pickle.load(f)
        logger.info("XGBoost model object loaded.")
    except Exception as e:
        logger.warning(f"Could not load XGBoost model object: {e}")

    try:
        import torch
        lstm_cfg = _cfg["lstm"]
        lstm = LSTMRegimeModel(
            input_size  = 15,
            hidden_size = lstm_cfg["hidden_size"],
            n_layers    = lstm_cfg["n_layers"],
            n_classes   = 3,
            dropout     = lstm_cfg["dropout"],
        )
        state = torch.load(_resolve(_PATHS["lstm_model"]), map_location="cpu")
        lstm.load_state_dict(state)
        lstm.eval()
        app.state.loaded_models["lstm"] = lstm
        logger.info("LSTM model object loaded.")
    except Exception as e:
        logger.warning(f"Could not load LSTM model object: {e}")

    try:
        import torch
        tft_cfg = _cfg["tft"]
        tft = TemporalFusionTransformer(
            input_size = 15,
            hidden     = tft_cfg["hidden"],
            n_heads    = tft_cfg["n_heads"],
            n_classes  = 3,
            dropout    = tft_cfg["dropout"],
        )
        state = torch.load(_resolve(_PATHS["tft_model"]), map_location="cpu")
        tft.load_state_dict(state)
        tft.eval()
        app.state.loaded_models["tft"] = tft
        logger.info("TFT model object loaded.")
    except Exception as e:
        logger.warning(f"Could not load TFT model object: {e}")

    try:
        import torch
        vqh = VQHModel(n_qubits=3, n_layers=2, n_classes=3)
        state = torch.load(_resolve(_PATHS["vqh_model"]), map_location="cpu")
        vqh.load_state_dict(state)
        vqh.eval()
        app.state.loaded_models["vqh"] = vqh
        logger.info("VQH model object loaded.")
    except Exception as e:
        logger.warning(f"Could not load VQH model object: {e}")

    loaded = sum(1 for v in app.state.metrics.values() if v is not None)
    n_obj  = len(app.state.loaded_models)
    logger.info(f"Startup complete. {loaded}/5 model metrics loaded. {n_obj}/5 model objects loaded.")
    logger.info(
        f"Aligned prices: {'✓' if app.state.aligned_prices is not None else '✗'}  "
        f"Feature matrix: {'✓' if app.state.feature_matrix is not None else '✗'}  "
        f"Benchmark: {'✓' if app.state.benchmark is not None else '✗'}"
    )


def create_app() -> FastAPI:
    @asynccontextmanager
    async def lifespan(app: FastAPI):
        """Load all data once at startup. Nothing to clean up at shutdown."""
        _load_all_state(app)
        yield

    app = FastAPI(
        title       = "QML Financial Contagion API",
        description = "Detection and prediction of systemic financial contagion "
                      "via Variational Quantum HMM",
        version     = "1.0.0",
        lifespan    = lifespan,
    )

    app.add_middleware(
        CORSMiddleware,
        allow_origins     = ["http://localhost:5173", "http://127.0.0.1:5173"],
        allow_credentials = True,
        allow_methods     = ["*"],
        allow_headers     = ["*"],
    )

    app.include_router(data.router,        prefix="/api")
    app.include_router(models.router,      prefix="/api")
    app.include_router(predictions.router, prefix="/api")

    return app


app = create_app()
