"""
src/training/train_classical.py

Training orchestrator for all 4 classical models:
  1. HMM (GaussianHMM)
  2. XGBoost
  3. LSTM + Attention
  4. Temporal Fusion Transformer

All models use the SAME chronological splits (never shuffle time series):
  Train: 2015-01-12 → 2021-12-31  (includes COVID 2020)
  Val:   2022-01-01 → 2022-12-31  (Crypto Winter 2022, threshold tuning only)
  Test:  2023-01-01 → 2024-12-31  (SVB 2023 + 2024, fully out-of-sample)

StandardScaler fits on training data ONLY — transformed on val and test.
Crisis augmentation on training data ONLY.
Optimal threshold selection on val set ONLY.
"""

from __future__ import annotations

import random
from pathlib import Path
from typing import Optional

import numpy as np
import pandas as pd
import torch
import torch.nn as nn
from torch.utils.data import DataLoader, TensorDataset
import yaml
from loguru import logger
from sklearn.metrics import f1_score
from sklearn.preprocessing import StandardScaler

from src.data.pipeline import load_feature_matrix, FEATURE_COLS, LABEL_COL
from src.models.hmm_model import HMMRegimeModel
from src.models.xgboost_model import XGBoostRegimeModel
from src.models.lstm_model import LSTMRegimeModel, FocalLoss, build_focal_loss
from src.models.transformer_model import TemporalFusionTransformer, count_parameters
from src.evaluation.metrics import compute_all_metrics, compute_pr_curve
from src.evaluation.benchmark import save_benchmark
from src.utils.sequences import build_sequences, augment_crisis_sequences
from src.utils.io import save_pkl, save_pt, save_metrics, save_predictions, save_json

_ROOT = Path(__file__).resolve().parents[2]
_DATA_CFG  = yaml.safe_load((_ROOT / "config" / "data.yaml").read_text())
_MODEL_CFG = yaml.safe_load((_ROOT / "config" / "model.yaml").read_text())
_PATHS     = yaml.safe_load((_ROOT / "config" / "paths.yaml").read_text())


def _set_seeds() -> None:
    seeds = _MODEL_CFG["seeds"]
    np.random.seed(seeds["numpy"])
    torch.manual_seed(seeds["torch"])
    random.seed(seeds["python"])
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seeds["torch"])


def _get_splits():
    cfg = _DATA_CFG["splits"]
    return {
        "train": (pd.Timestamp(cfg["train_start"]), pd.Timestamp(cfg["train_end"])),
        "val":   (pd.Timestamp(cfg["val_start"]),   pd.Timestamp(cfg["val_end"])),
        "test":  (pd.Timestamp(cfg["test_start"]),  pd.Timestamp(cfg["test_end"])),
    }


def split_data(feature_matrix: pd.DataFrame) -> dict:
    """
    Apply chronological train/val/test split. Never shuffle.

    Args:
        feature_matrix : loaded feature_matrix.csv

    Returns:
        dict with keys: X_train, y_train, X_val, y_val, X_test, y_test,
                        d_train, d_val, d_test, feature_train (full df slice)
    """
    splits = _get_splits()
    idx    = feature_matrix.index
    X_all  = feature_matrix[FEATURE_COLS].values
    y_all  = feature_matrix[LABEL_COL].values

    def _mask(start, end):
        return (idx >= start) & (idx <= end)

    mask_train = _mask(*splits["train"])
    mask_val   = _mask(*splits["val"])
    mask_test  = _mask(*splits["test"])

    logger.info(f"Train: {mask_train.sum()} days | Val: {mask_val.sum()} days | Test: {mask_test.sum()} days")

    return {
        "X_train":       X_all[mask_train],
        "y_train":       y_all[mask_train],
        "X_val":         X_all[mask_val],
        "y_val":         y_all[mask_val],
        "X_test":        X_all[mask_test],
        "y_test":        y_all[mask_test],
        "d_train":       idx[mask_train],
        "d_val":         idx[mask_val],
        "d_test":        idx[mask_test],
        "feature_train": feature_matrix[mask_train],
    }


# ─── HMM Training ────────────────────────────────────────────────

def train_hmm(data: dict) -> dict:
    """
    Train HMM on the full training time series.
    HMM does NOT use windowed sequences — it needs the temporal chain.
    """
    logger.info("=" * 50)
    logger.info("TRAINING: Gaussian HMM")
    logger.info("=" * 50)
    _set_seeds()

    model = HMMRegimeModel()
    model.fit(data["X_train"], data["feature_train"])

    # Predictions on all splits
    proba_test = model.predict_proba(data["X_test"])
    pred_test  = model.predict(data["X_test"])

    # Save model
    save_pkl(model, "hmm_model")

    # Metrics
    metrics = compute_all_metrics(
        y_true=data["y_test"],
        y_pred=pred_test,
        probs=proba_test,
        model_name="HMM",
        version="v1",
        d_test=data["d_test"],
        optimal_threshold=model.cfg.get("n_components", 3),  # not applicable
        hyperparams=_MODEL_CFG["hmm"],
    )
    save_metrics(metrics, "hmm_metrics")

    # Predictions CSV
    proba_train = model.predict_proba(data["X_train"])
    pred_train  = model.predict(data["X_train"])
    proba_val   = model.predict_proba(data["X_val"])
    pred_val    = model.predict(data["X_val"])

    pred_df = pd.concat([
        pd.DataFrame({
            "true_regime": data["y_train"],
            "pred_regime": pred_train,
            "prob_crisis": proba_train[:, 0],
            "prob_normal": proba_train[:, 1],
            "prob_highvol": proba_train[:, 2],
        }, index=data["d_train"]),
        pd.DataFrame({
            "true_regime": data["y_val"],
            "pred_regime": pred_val,
            "prob_crisis": proba_val[:, 0],
            "prob_normal": proba_val[:, 1],
            "prob_highvol": proba_val[:, 2],
        }, index=data["d_val"]),
        pd.DataFrame({
            "true_regime": data["y_test"],
            "pred_regime": pred_test,
            "prob_crisis": proba_test[:, 0],
            "prob_normal": proba_test[:, 1],
            "prob_highvol": proba_test[:, 2],
        }, index=data["d_test"]),
    ]).sort_index()
    pred_df.index.name = "date"
    save_predictions(pred_df, "hmm_predictions")

    pr_curve = compute_pr_curve(data["y_test"], proba_test)
    save_json(pr_curve, _ROOT / "outputs/metrics/hmm_pr_curve.json")

    print(f"\nHMM: crisis PR-AUC = {metrics['crisis_specific']['pr_auc']:.4f}")
    return metrics


# ─── XGBoost Training ────────────────────────────────────────────

def train_xgboost(data: dict) -> dict:
    logger.info("=" * 50)
    logger.info("TRAINING: XGBoost")
    logger.info("=" * 50)
    _set_seeds()

    model = XGBoostRegimeModel()
    model.fit(
        data["X_train"], data["y_train"],
        data["X_val"],   data["y_val"],
        feature_names=FEATURE_COLS,
    )

    # Save model
    save_pkl(model, "xgb_model")

    # Save feature importance
    fi = model.get_feature_importance()
    save_json(fi, _ROOT / _PATHS["xgb_feature_importance"])

    # Test predictions
    proba_test = model.predict_proba(data["X_test"])
    pred_test  = model.predict(data["X_test"])

    metrics = compute_all_metrics(
        y_true=data["y_test"],
        y_pred=pred_test,
        probs=proba_test,
        model_name="XGBoost",
        version="v1",
        d_test=data["d_test"],
        optimal_threshold=model.optimal_threshold,
        hyperparams=_MODEL_CFG["xgboost"],
    )
    save_metrics(metrics, "xgb_metrics")

    proba_train = model.predict_proba(data["X_train"])
    pred_train  = model.predict(data["X_train"])
    proba_val   = model.predict_proba(data["X_val"])
    pred_val    = model.predict(data["X_val"])

    pred_df = pd.concat([
        pd.DataFrame({
            "true_regime": data["y_train"],
            "pred_regime": pred_train,
            "prob_crisis": proba_train[:, 0],
            "prob_normal": proba_train[:, 1],
            "prob_highvol": proba_train[:, 2],
        }, index=data["d_train"]),
        pd.DataFrame({
            "true_regime": data["y_val"],
            "pred_regime": pred_val,
            "prob_crisis": proba_val[:, 0],
            "prob_normal": proba_val[:, 1],
            "prob_highvol": proba_val[:, 2],
        }, index=data["d_val"]),
        pd.DataFrame({
            "true_regime": data["y_test"],
            "pred_regime": pred_test,
            "prob_crisis": proba_test[:, 0],
            "prob_normal": proba_test[:, 1],
            "prob_highvol": proba_test[:, 2],
        }, index=data["d_test"]),
    ]).sort_index()
    pred_df.index.name = "date"
    save_predictions(pred_df, "xgb_predictions")

    pr_curve = compute_pr_curve(data["y_test"], proba_test)
    save_json(pr_curve, _ROOT / "outputs/metrics/xgb_pr_curve.json")

    print(f"\nXGBoost: crisis PR-AUC = {metrics['crisis_specific']['pr_auc']:.4f}")
    return metrics


# ─── Deep Learning Training Loop ────────────────────────────────

def _train_dl_model(
    model_name: str,
    model: nn.Module,
    train_loader: DataLoader,
    val_loader: DataLoader,
    X_val_seq: np.ndarray,
    y_val: np.ndarray,
    criterion: nn.Module,
    cfg: dict,
) -> tuple[nn.Module, float]:
    """
    Shared training loop for LSTM and TFT.
    Early stopping tracks val MACRO-F1 (NOT loss).

    Returns:
        (best_model_state, best_val_f1)
    """
    _set_seeds()
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model  = model.to(device)

    optimizer = torch.optim.Adam(
        model.parameters(),
        lr=cfg["lr"],
        weight_decay=cfg["weight_decay"],
    )
    scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(
        optimizer, mode="max", patience=10, factor=0.5
    )

    best_f1, best_state, patience = 0.0, None, 0
    max_patience = cfg["early_stop_patience"]

    for epoch in range(1, cfg["epochs"] + 1):
        # ── Train ─────────────────────────────────────────────────
        model.train()
        total_loss = 0.0
        for X_batch, y_batch in train_loader:
            X_batch, y_batch = X_batch.to(device), y_batch.to(device)
            optimizer.zero_grad()
            logits = model(X_batch)
            loss   = criterion(logits, y_batch)
            loss.backward()
            nn.utils.clip_grad_norm_(model.parameters(), cfg["grad_clip"])
            optimizer.step()
            total_loss += loss.item()

        # ── Val ────────────────────────────────────────────────────
        model.eval()
        all_preds = []
        with torch.no_grad():
            for X_b, _ in val_loader:
                out = model(X_b.to(device))
                preds = out.argmax(dim=1).cpu().numpy()
                all_preds.extend(preds)

        val_f1 = f1_score(y_val, all_preds, average="macro", zero_division=0)
        scheduler.step(val_f1)

        if val_f1 > best_f1:
            best_f1    = val_f1
            best_state = {k: v.cpu().clone() for k, v in model.state_dict().items()}
            patience   = 0
        else:
            patience += 1

        if epoch % 10 == 0 or epoch == 1:
            logger.info(
                f"  [{model_name}] Epoch {epoch:3d}: "
                f"loss={total_loss/len(train_loader):.4f}, val_macro_f1={val_f1:.4f} "
                f"(best={best_f1:.4f})"
            )

        if patience >= max_patience:
            logger.info(f"  [{model_name}] Early stopping at epoch {epoch} (patience={max_patience})")
            break

    model.load_state_dict(best_state)
    logger.info(f"  [{model_name}] Best val macro-F1: {best_f1:.4f}")
    return model, best_f1


def _make_dl_loaders(
    X_train: np.ndarray, y_train: np.ndarray,
    X_val: np.ndarray,   y_val: np.ndarray,
    cfg: dict,
) -> tuple[DataLoader, DataLoader]:
    """
    Build DataLoaders for train and val.
    Crisis augmentation is applied to TRAINING SEQUENCES ONLY.
    """
    # Identify crisis sequences in training
    crisis_mask = y_train == 0
    X_crisis    = X_train[crisis_mask]
    y_crisis    = y_train[crisis_mask]

    rng = np.random.default_rng(cfg["random_state"])
    X_aug, y_aug = augment_crisis_sequences(
        X_crisis, y_crisis,
        factor=cfg["augment_factor"],
        noise_std=cfg["augment_noise"],
        rng=rng,
    )

    X_train_aug = np.concatenate([X_train, X_aug], axis=0)
    y_train_aug = np.concatenate([y_train, y_aug], axis=0)

    # Convert to tensors
    Xt = torch.tensor(X_train_aug, dtype=torch.float32)
    yt = torch.tensor(y_train_aug, dtype=torch.long)
    Xv = torch.tensor(X_val, dtype=torch.float32)
    yv = torch.tensor(y_val, dtype=torch.long)

    train_ds = TensorDataset(Xt, yt)
    val_ds   = TensorDataset(Xv, yv)

    train_loader = DataLoader(train_ds, batch_size=cfg["batch_size"], shuffle=True)
    val_loader   = DataLoader(val_ds,   batch_size=cfg["batch_size"], shuffle=False)

    logger.info(
        f"  Training sequences: {len(X_train)} + {len(X_aug)} augmented = {len(X_train_aug)}"
    )
    return train_loader, val_loader


# ─── LSTM Training ────────────────────────────────────────────────

def train_lstm(data: dict) -> dict:
    logger.info("=" * 50)
    logger.info("TRAINING: LSTM + Attention")
    logger.info("=" * 50)
    _set_seeds()

    cfg = _MODEL_CFG["lstm"]
    seq_len = cfg["seq_len"]

    # Scale (fit on train ONLY)
    scaler = StandardScaler()
    X_tr = scaler.fit_transform(data["X_train"])
    X_v  = scaler.transform(data["X_val"])
    X_te = scaler.transform(data["X_test"])

    # Build sequences
    X_train_seq, y_train_seq, d_train_seq = build_sequences(X_tr, data["y_train"], data["d_train"], seq_len)
    X_val_seq,   y_val_seq,   d_val_seq   = build_sequences(X_v,  data["y_val"],   data["d_val"],   seq_len)
    X_test_seq,  y_test_seq,  d_test_seq  = build_sequences(X_te, data["y_test"],  data["d_test"],  seq_len)

    # Focal loss
    criterion = build_focal_loss(y_train_seq, cfg["focal_gamma"], cfg["crisis_weight_multiplier"])

    # Model
    n_features = X_train_seq.shape[2]
    model = LSTMRegimeModel(
        input_size  = n_features,
        hidden_size = cfg["hidden_size"],
        n_layers    = cfg["n_layers"],
        dropout     = cfg["dropout"],
    )
    logger.info(f"  LSTM parameters: {sum(p.numel() for p in model.parameters() if p.requires_grad):,}")

    # Loaders
    train_loader, val_loader = _make_dl_loaders(X_train_seq, y_train_seq, X_val_seq, y_val_seq, cfg)

    # Train
    model, best_f1 = _train_dl_model(
        "LSTM", model, train_loader, val_loader, X_val_seq, y_val_seq, criterion, cfg
    )

    # Save
    save_pt(model.state_dict(), "lstm_model")

    # Test predictions
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model  = model.to(device).eval()
    Xt = torch.tensor(X_test_seq, dtype=torch.float32)
    test_ds = DataLoader(TensorDataset(Xt), batch_size=256, shuffle=False)

    all_preds, all_proba = [], []
    with torch.no_grad():
        for (X_b,) in test_ds:
            out    = model(X_b.to(device))
            probs  = torch.softmax(out, dim=1).cpu().numpy()
            preds  = out.argmax(dim=1).cpu().numpy()
            all_proba.extend(probs)
            all_preds.extend(preds)

    proba_test = np.array(all_proba)
    pred_test  = np.array(all_preds)

    # Tune threshold on val
    val_proba = _predict_proba_dl(model, X_val_seq, device)
    opt_thresh = _tune_threshold_dl(y_val_seq, val_proba, pred_test)

    metrics = compute_all_metrics(
        y_true=y_test_seq,
        y_pred=pred_test,
        probs=proba_test,
        model_name="LSTM",
        version="v5",
        d_test=d_test_seq,
        optimal_threshold=opt_thresh,
        hyperparams=cfg,
    )
    save_metrics(metrics, "lstm_metrics")

    proba_train = _predict_proba_dl(model, X_train_seq, device)
    pred_train  = proba_train.argmax(axis=1)
    proba_val   = _predict_proba_dl(model, X_val_seq, device)
    pred_val    = proba_val.argmax(axis=1)

    pred_df = pd.concat([
        pd.DataFrame({
            "true_regime": y_train_seq,
            "pred_regime": pred_train,
            "prob_crisis": proba_train[:, 0],
            "prob_normal": proba_train[:, 1],
            "prob_highvol": proba_train[:, 2],
        }, index=d_train_seq),
        pd.DataFrame({
            "true_regime": y_val_seq,
            "pred_regime": pred_val,
            "prob_crisis": proba_val[:, 0],
            "prob_normal": proba_val[:, 1],
            "prob_highvol": proba_val[:, 2],
        }, index=d_val_seq),
        pd.DataFrame({
            "true_regime": y_test_seq,
            "pred_regime": pred_test,
            "prob_crisis": proba_test[:, 0],
            "prob_normal": proba_test[:, 1],
            "prob_highvol": proba_test[:, 2],
        }, index=d_test_seq),
    ]).sort_index()
    pred_df.index.name = "date"
    save_predictions(pred_df, "lstm_predictions")

    pr_curve = compute_pr_curve(y_test_seq, proba_test)
    save_json(pr_curve, _ROOT / "outputs/metrics/lstm_pr_curve.json")

    print(f"\nLSTM: crisis PR-AUC = {metrics['crisis_specific']['pr_auc']:.4f}")
    return metrics


# ─── TFT Training ────────────────────────────────────────────────

def train_tft(data: dict) -> dict:
    logger.info("=" * 50)
    logger.info("TRAINING: Temporal Fusion Transformer")
    logger.info("=" * 50)
    _set_seeds()

    cfg = _MODEL_CFG["tft"]
    seq_len = cfg["seq_len"]

    # Scale
    scaler = StandardScaler()
    X_tr = scaler.fit_transform(data["X_train"])
    X_v  = scaler.transform(data["X_val"])
    X_te = scaler.transform(data["X_test"])

    # Build sequences
    X_train_seq, y_train_seq, d_train_seq = build_sequences(X_tr, data["y_train"], data["d_train"], seq_len)
    X_val_seq,   y_val_seq,   d_val_seq   = build_sequences(X_v,  data["y_val"],   data["d_val"],   seq_len)
    X_test_seq,  y_test_seq,  d_test_seq  = build_sequences(X_te, data["y_test"],  data["d_test"],  seq_len)

    n_features = X_train_seq.shape[2]
    model = TemporalFusionTransformer(
        input_size = n_features,
        hidden     = cfg["hidden"],
        n_heads    = cfg["n_heads"],
        n_classes  = 3,
        dropout    = cfg["dropout"],
    )
    n_params = count_parameters(model)
    logger.info(f"  TFT parameters: {n_params:,} (target: <500,000)")
    if n_params >= 500_000:
        logger.warning(f"  TFT exceeds 500k parameter target: {n_params:,}")

    criterion    = build_focal_loss(y_train_seq, cfg["focal_gamma"], cfg["crisis_weight_multiplier"])
    train_loader, val_loader = _make_dl_loaders(X_train_seq, y_train_seq, X_val_seq, y_val_seq, cfg)

    model, best_f1 = _train_dl_model(
        "TFT", model, train_loader, val_loader, X_val_seq, y_val_seq, criterion, cfg
    )

    save_pt(model.state_dict(), "tft_model")

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model  = model.to(device).eval()
    Xt  = torch.tensor(X_test_seq, dtype=torch.float32)
    ds  = DataLoader(TensorDataset(Xt), batch_size=256, shuffle=False)

    all_preds, all_proba = [], []
    with torch.no_grad():
        for (X_b,) in ds:
            out   = model(X_b.to(device))
            probs = torch.softmax(out, dim=1).cpu().numpy()
            preds = out.argmax(dim=1).cpu().numpy()
            all_proba.extend(probs)
            all_preds.extend(preds)

    proba_test = np.array(all_proba)
    pred_test  = np.array(all_preds)

    val_proba  = _predict_proba_dl(model, X_val_seq, device)
    opt_thresh = _tune_threshold_dl(y_val_seq, val_proba, pred_test)

    metrics = compute_all_metrics(
        y_true=y_test_seq,
        y_pred=pred_test,
        probs=proba_test,
        model_name="TFT",
        version="v1",
        d_test=d_test_seq,
        optimal_threshold=opt_thresh,
        hyperparams=cfg,
    )
    save_metrics(metrics, "tft_metrics")

    proba_train = _predict_proba_dl(model, X_train_seq, device)
    pred_train  = proba_train.argmax(axis=1)
    proba_val   = _predict_proba_dl(model, X_val_seq, device)
    pred_val    = proba_val.argmax(axis=1)

    pred_df = pd.concat([
        pd.DataFrame({
            "true_regime": y_train_seq,
            "pred_regime": pred_train,
            "prob_crisis": proba_train[:, 0],
            "prob_normal": proba_train[:, 1],
            "prob_highvol": proba_train[:, 2],
        }, index=d_train_seq),
        pd.DataFrame({
            "true_regime": y_val_seq,
            "pred_regime": pred_val,
            "prob_crisis": proba_val[:, 0],
            "prob_normal": proba_val[:, 1],
            "prob_highvol": proba_val[:, 2],
        }, index=d_val_seq),
        pd.DataFrame({
            "true_regime": y_test_seq,
            "pred_regime": pred_test,
            "prob_crisis": proba_test[:, 0],
            "prob_normal": proba_test[:, 1],
            "prob_highvol": proba_test[:, 2],
        }, index=d_test_seq),
    ]).sort_index()
    pred_df.index.name = "date"
    save_predictions(pred_df, "tft_predictions")

    pr_curve = compute_pr_curve(y_test_seq, proba_test)
    save_json(pr_curve, _ROOT / "outputs/metrics/tft_pr_curve.json")

    print(f"\nTFT: crisis PR-AUC = {metrics['crisis_specific']['pr_auc']:.4f}")
    return metrics


# ─── Helpers ─────────────────────────────────────────────────────

def _predict_proba_dl(model, X_seq: np.ndarray, device) -> np.ndarray:
    model.eval()
    ds = DataLoader(TensorDataset(torch.tensor(X_seq, dtype=torch.float32)),
                    batch_size=256, shuffle=False)
    all_proba = []
    with torch.no_grad():
        for (X_b,) in ds:
            out   = model(X_b.to(device))
            probs = torch.softmax(out, dim=1).cpu().numpy()
            all_proba.extend(probs)
    return np.array(all_proba)


def _tune_threshold_dl(y_val: np.ndarray, val_proba: np.ndarray, _) -> float:
    """Find crisis probability threshold that maximizes val crisis F1."""
    best_f1, best_thresh = 0.0, 0.5
    crisis_proba = val_proba[:, 0]
    for thresh in np.arange(0.1, 0.9, 0.05):
        y_pred = val_proba.argmax(axis=1).copy()
        y_pred[crisis_proba >= thresh] = 0
        f1 = f1_score(y_val, y_pred, labels=[0], average="macro", zero_division=0)
        if f1 > best_f1:
            best_f1, best_thresh = f1, float(thresh)
    logger.info(f"  Optimal crisis threshold: {best_thresh:.2f} (val crisis F1={best_f1:.4f})")
    return best_thresh


# ─── Full Phase 2 Run ────────────────────────────────────────────

def run_all_classical() -> dict:
    """
    Train all 4 classical models, save all outputs, return benchmark dict.

    Returns:
        benchmark_dict containing comparison across all models
    """
    _set_seeds()

    logger.info("Loading feature matrix...")
    feature_matrix = load_feature_matrix()
    data = split_data(feature_matrix)

    metrics = {}
    metrics["HMM"]     = train_hmm(data)
    metrics["XGBoost"] = train_xgboost(data)
    metrics["LSTM"]    = train_lstm(data)
    metrics["TFT"]     = train_tft(data)

    # Save benchmark (Phase 2 — no VQH yet)
    benchmark = save_benchmark(metrics, include_vqh=False)

    best_name  = benchmark["best_model_by_crisis_pr_auc"]
    best_prauc = benchmark["comparison"]["crisis_pr_auc"].get(best_name, 0)
    print(f"\nBest model: {best_name} (crisis PR-AUC = {best_prauc:.4f})")
    print("Predictions ready for VQH encoder.")

    return benchmark
