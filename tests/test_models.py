"""tests/test_models.py — Smoke tests for all 5 model forward passes."""
import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

import numpy as np
import pytest
import torch


# ─── LSTM ────────────────────────────────────────────────────────

def test_lstm_forward():
    from src.models.lstm_model import LSTMRegimeModel
    model = LSTMRegimeModel(input_size=15, hidden_size=64, n_layers=2)
    model.eval()
    x = torch.randn(4, 20, 15)
    with torch.no_grad():
        out = model(x)
    assert out.shape == (4, 3), f"LSTM output shape: {out.shape}"


def test_focal_loss_forward():
    from src.models.lstm_model import FocalLoss, build_focal_loss
    y = np.array([0, 0, 1, 1, 1, 2, 2, 2, 2, 2])
    loss_fn = build_focal_loss(y, gamma=2.0, crisis_boost=4.0)
    logits  = torch.randn(10, 3)
    targets = torch.tensor(y, dtype=torch.long)
    loss    = loss_fn(logits, targets)
    assert loss.item() > 0, "Focal loss must be positive"


# ─── TFT ─────────────────────────────────────────────────────────

def test_tft_forward():
    from src.models.transformer_model import TemporalFusionTransformer, count_parameters
    model = TemporalFusionTransformer(input_size=15, hidden=32, n_heads=2, n_classes=3)
    model.eval()
    x = torch.randn(4, 20, 15)
    with torch.no_grad():
        out = model(x)
    assert out.shape == (4, 3), f"TFT output shape: {out.shape}"
    n_params = count_parameters(model)
    assert n_params < 500_000, f"TFT has {n_params:,} params — must be < 500k"


# ─── XGBoost smoke test ──────────────────────────────────────────

def test_xgboost_smoke():
    from src.models.xgboost_model import XGBoostRegimeModel
    rng = np.random.default_rng(42)
    X_train = rng.normal(0, 1, (80, 15)).astype(np.float32)
    y_train = rng.choice([0, 1, 2], 80, p=[0.1, 0.7, 0.2])
    X_val   = rng.normal(0, 1, (20, 15)).astype(np.float32)
    y_val   = rng.choice([0, 1, 2], 20, p=[0.1, 0.7, 0.2])

    model = XGBoostRegimeModel()
    model.fit(X_train, y_train, X_val, y_val)
    proba = model.predict_proba(X_val)
    assert proba.shape == (20, 3)
    assert np.allclose(proba.sum(axis=1), 1.0, atol=1e-5)


# ─── VQH smoke test ──────────────────────────────────────────────

def test_vqh_forward():
    from src.models.vqh_model import VQHModel
    model = VQHModel(n_qubits=3, n_layers=1, n_classes=3)
    model.eval()
    # Input: (B, 3) probability vectors
    x = torch.tensor([[0.1, 0.7, 0.2], [0.8, 0.1, 0.1]], dtype=torch.float32)
    with torch.no_grad():
        out = model(x)
    assert out.shape == (2, 3), f"VQH output shape: {out.shape}"


def test_vqh_parameter_count():
    from src.models.vqh_model import VQHModel
    model = VQHModel(n_qubits=3, n_layers=2, n_classes=3)
    n = sum(p.numel() for p in model.parameters())
    # ~18 quantum + ~115 classical = ~133
    assert n < 500, f"VQH has {n} params — expected ~133"


# ─── Sequence utilities ──────────────────────────────────────────

def test_build_sequences():
    from src.utils.sequences import build_sequences
    import pandas as pd
    rng  = np.random.default_rng(0)
    n    = 50
    X    = rng.normal(0, 1, (n, 15)).astype(np.float32)
    y    = rng.integers(0, 3, n).astype(np.int64)
    dates = pd.date_range("2020-01-01", periods=n, freq="B")

    X_seq, y_seq, d_seq = build_sequences(X, y, dates, seq_len=20)
    assert X_seq.shape == (31, 20, 15), f"Got: {X_seq.shape}"
    assert y_seq.shape == (31,)


def test_augment_crisis():
    from src.utils.sequences import augment_crisis_sequences
    rng = np.random.default_rng(42)
    X = rng.normal(0, 1, (5, 20, 15)).astype(np.float32)
    y = np.zeros(5, dtype=np.int64)
    X_aug, y_aug = augment_crisis_sequences(X, y, factor=3, noise_std=0.01)
    assert X_aug.shape == (15, 20, 15)
    assert (y_aug == 0).all()
