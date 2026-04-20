"""
src/training/train_vqh.py

Phase 3 — VQH Training.

Task: Next-step regime forecasting.
  Input : regime probability vector at time t (from best classical model)
  Target: true regime at time t+1

WHY THIS TASK:
  The classical HMM transition matrix A_{ij} = P(S_j | S_i) is what the VQH
  replaces. So the VQH must do next-step prediction to be a direct replacement.
  This is distinct from the classical models which predict the current regime.

KEY CONSTRAINT: Use DIFFERENT learning rates for quantum and classical parts.
  quantum params: lr=0.01  (10× higher — quantum gradients are sparse)
  classical params: lr=0.001 (standard)
  This is mandatory per spec. Using the same lr causes convergence failure.

MAX 100 EPOCHS:
  3-qubit circuits converge faster than classical networks.
  Beyond 100 epochs, the model is fitting noise in the ~1000-sample dataset.
"""

from __future__ import annotations

import random
from pathlib import Path

import numpy as np
import pandas as pd
import torch
import torch.nn as nn
from torch.utils.data import DataLoader, TensorDataset
import yaml
from loguru import logger
from sklearn.metrics import f1_score

from src.models.vqh_model import VQHModel
from src.models.lstm_model import FocalLoss, build_focal_loss
from src.data.pipeline import LABEL_COL
from src.evaluation.metrics import compute_all_metrics, compute_pr_curve, crisis_timing_lead
from src.evaluation.black_swan import plot_lead_lag_svb, plot_entanglement_entropy
from src.evaluation.benchmark import save_benchmark
from src.utils.io import save_pt, save_metrics, save_predictions, save_json, load_metrics

_ROOT = Path(__file__).resolve().parents[2]
_MODEL_CFG = yaml.safe_load((_ROOT / "config" / "model.yaml").read_text())
_DATA_CFG  = yaml.safe_load((_ROOT / "config" / "data.yaml").read_text())
_PATHS     = yaml.safe_load((_ROOT / "config" / "paths.yaml").read_text())


def _set_seeds():
    seeds = _MODEL_CFG["seeds"]
    np.random.seed(seeds["numpy"])
    torch.manual_seed(seeds["torch"])
    random.seed(seeds["python"])


def load_best_classical_predictions() -> tuple[str, pd.DataFrame]:
    """
    Load the best classical model's prediction CSV.

    Returns:
        (model_name, predictions_df)
        predictions_df columns: prob_crisis, prob_normal, prob_highvol, true_regime
    """
    best_path = _ROOT / _PATHS["best_model"]
    if not best_path.exists():
        raise FileNotFoundError(
            f"best_model.txt not found at {best_path}. "
            "Run scripts/run_phase2_classical.py first."
        )

    best_name = best_path.read_text().strip()
    pred_key = f"{best_name.lower()}_predictions"

    # Some model naming conventions differ between benchmark output and path keys.
    if pred_key not in _PATHS:
        alias_map = {
            "xgboost": "xgb_predictions",
        }
        pred_key = alias_map.get(best_name.lower(), pred_key)

    if pred_key not in _PATHS:
        raise KeyError(
            f"Prediction path key not found for best model '{best_name}': {pred_key}"
        )

    pred_path = _ROOT / _PATHS[pred_key]

    logger.info(f"Loading {best_name} predictions from {pred_path}")
    preds = pd.read_csv(pred_path, index_col=0, parse_dates=True)

    required = {"prob_crisis", "prob_normal", "prob_highvol", "true_regime"}
    missing  = required - set(preds.columns)
    if missing:
        raise ValueError(f"Missing columns in {best_name} predictions: {missing}")

    return best_name, preds


def _apply_temporal_split(preds: pd.DataFrame) -> dict:
    """Apply same train/val/test split as Phase 2 to predictions."""
    splits = _DATA_CFG["splits"]

    def _mask(start, end):
        return (preds.index >= pd.Timestamp(start)) & (preds.index <= pd.Timestamp(end))

    return {
        "train": preds[_mask(splits["train_start"], splits["train_end"])],
        "val":   preds[_mask(splits["val_start"],   splits["val_end"])],
        "test":  preds[_mask(splits["test_start"],  splits["test_end"])],
    }


def build_vqh_pairs(preds: pd.DataFrame) -> tuple[np.ndarray, np.ndarray, pd.DatetimeIndex]:
    """
    Build (X_t, y_{t+1}) pairs for next-step prediction.

    X_vqh[i] = posterior probs at time t        → used as VQH input
    y_vqh[i] = true regime at time t+1           → prediction target

    The last row is dropped (no t+1 for the final day).

    Args:
        preds : prediction DataFrame with prob_crisis, prob_normal, prob_highvol, true_regime

    Returns:
        X : (T-1, 3) input probability arrays
        y : (T-1,) next-step regime labels
        d : DatetimeIndex of length T-1 (date of y targets)
    """
    probs   = preds[["prob_crisis", "prob_normal", "prob_highvol"]].values
    regimes = preds["true_regime"].values.astype(int)
    dates   = preds.index

    X = probs[:-1].astype(np.float32)           # t=0..T-2
    y = regimes[1:].astype(np.int64)            # t=1..T-1 (next-step)
    d = dates[1:]                               # dates of y

    return X, y, pd.DatetimeIndex(d)


def train_vqh(
    X_train: np.ndarray, y_train: np.ndarray,
    X_val:   np.ndarray, y_val:   np.ndarray,
) -> tuple[VQHModel, float]:
    """
    Train the Variational Quantum HMM model.

    Uses DUAL learning rates:
      quantum_weights: lr=0.01  (10× classical — quantum gradients are sparse)
      post_head:       lr=0.001 (standard classical rate)

    Early stopping on val macro-F1 (patience=15).
    Maximum 100 epochs — beyond this, quantum circuits fit noise.

    Args:
        X_train, y_train : training inputs and next-step regime labels
        X_val, y_val     : validation inputs and labels

    Returns:
        (model, optimal_threshold on val)
    """
    _set_seeds()
    cfg = _MODEL_CFG["vqh"]

    criterion = build_focal_loss(y_train, cfg["focal_gamma"], cfg["crisis_weight_multiplier"])

    model = VQHModel(n_qubits=3, n_layers=2, n_classes=3)
    logger.info(f"VQH total parameters: {sum(p.numel() for p in model.parameters())}")

    optimizer = torch.optim.Adam([
        {"params": model.qnode.parameters(),     "lr": cfg["lr_quantum"]},
        {"params": model.post_head.parameters(), "lr": cfg["lr_classical"]},
    ], weight_decay=cfg["weight_decay"])

    Xt = torch.tensor(X_train, dtype=torch.float32)
    yt = torch.tensor(y_train, dtype=torch.long)
    Xv = torch.tensor(X_val,   dtype=torch.float32)
    yv = torch.tensor(y_val,   dtype=torch.long)

    train_ds     = TensorDataset(Xt, yt)
    val_ds       = TensorDataset(Xv, yv)
    train_loader = DataLoader(train_ds, batch_size=cfg["batch_size"], shuffle=True)
    val_loader   = DataLoader(val_ds,   batch_size=cfg["batch_size"], shuffle=False)

    best_f1, best_state, patience = 0.0, None, 0

    for epoch in range(1, cfg["epochs"] + 1):
        model.train()
        ep_loss = 0.0
        for X_b, y_b in train_loader:
            optimizer.zero_grad()
            logits = model(X_b)
            loss   = criterion(logits, y_b)
            loss.backward()
            optimizer.step()
            ep_loss += loss.item()

        model.eval()
        all_preds = []
        with torch.no_grad():
            for X_b, _ in val_loader:
                preds = model(X_b).argmax(dim=1).numpy()
                all_preds.extend(preds)

        val_f1 = f1_score(y_val, all_preds, average="macro", zero_division=0)

        if val_f1 > best_f1:
            best_f1    = val_f1
            best_state = {k: v.clone() for k, v in model.state_dict().items()}
            patience   = 0
        else:
            patience += 1

        if epoch % 10 == 0 or epoch == 1:
            logger.info(
                f"  [VQH] Epoch {epoch:3d}: loss={ep_loss/len(train_loader):.4f}, "
                f"val_f1={val_f1:.4f} (best={best_f1:.4f})"
            )

        if patience >= cfg["early_stop_patience"]:
            logger.info(f"  [VQH] Early stopping at epoch {epoch}")
            break

    model.load_state_dict(best_state)
    logger.info(f"  [VQH] Best val macro-F1: {best_f1:.4f}")

    return model, best_f1


def run_vqh_training() -> dict:
    """
    Full Phase 3 execution:
      1. Load best classical model predictions
      2. Build (X_t, y_{t+1}) pairs
      3. Train VQH
      4. Compute entanglement entropy
      5. Lead/lag analysis vs LSTM
      6. Save all outputs + figures
      7. Update benchmark_results_final.json

    Returns:
        VQH metrics dict
    """
    _set_seeds()
    logger.info("=" * 60)
    logger.info("PHASE 3 — QUANTUM ML (VQH)")
    logger.info("=" * 60)

    best_name, preds = load_best_classical_predictions()
    splits = _apply_temporal_split(preds)

    X_train, y_train, d_train = build_vqh_pairs(splits["train"])
    X_val,   y_val,   d_val   = build_vqh_pairs(splits["val"])
    X_test,  y_test,  d_test  = build_vqh_pairs(splits["test"])

    logger.info(f"VQH pairs: train={len(X_train)}, val={len(X_val)}, test={len(X_test)}")

    # ─── Train VQH ──────────────────────────────────────────────
    model, best_val_f1 = train_vqh(X_train, y_train, X_val, y_val)

    # Save model
    save_pt(model.state_dict(), "vqh_model")

    # ─── Test predictions ────────────────────────────────────────
    model.eval()
    Xt = torch.tensor(X_test, dtype=torch.float32)
    with torch.no_grad():
        out        = model(Xt)
        proba_test = torch.softmax(out, dim=1).numpy()
        pred_test  = out.argmax(dim=1).numpy()

    # ─── Entanglement entropy ────────────────────────────────────
    from src.models.vqh_model import compute_entanglement_entropy_batch
    logger.info("Computing entanglement entropy (this may take a few minutes)...")
    entropy = compute_entanglement_entropy_batch(model, X_test)
    logger.info(f"  Mean entropy: {entropy.mean():.4f}")

    # ─── Build predictions CSV ───────────────────────────────────
    pred_df = pd.DataFrame({
        "true_regime":          y_test,
        "pred_regime":          pred_test,
        "prob_crisis":          proba_test[:, 0],
        "prob_normal":          proba_test[:, 1],
        "prob_highvol":         proba_test[:, 2],
        "entanglement_entropy": entropy,
    }, index=d_test)
    pred_df.index.name = "date"
    save_predictions(pred_df, "vqh_predictions")

    # ─── Lead/lag analysis ───────────────────────────────────────
    lstm_pred_path = _ROOT / _PATHS["lstm_predictions"]
    lead_lag_result = {}
    if lstm_pred_path.exists():
        lstm_preds = pd.read_csv(lstm_pred_path, index_col=0, parse_dates=True)
        lstm_preds_test = lstm_preds[
            (lstm_preds.index >= d_test[0]) & (lstm_preds.index <= d_test[-1])
        ]
        # Align indices
        common_idx  = d_test.intersection(lstm_preds_test.index)
        mask_vqh    = pd.Series(d_test).isin(common_idx).values
        mask_lstm   = lstm_preds_test.index.isin(common_idx)

        lead_lag_result = crisis_timing_lead(
            vqh_prob_crisis  = proba_test[mask_vqh, 0],
            lstm_prob_crisis = lstm_preds_test.loc[mask_lstm, "prob_crisis"].values,
            true_regime      = y_test[mask_vqh],
            dates            = common_idx,
            threshold        = 0.4,
        )
        logger.info(f"Lead/lag: VQH={lead_lag_result.get('vqh_mean_lead', 0):.2f}d, "
                    f"LSTM={lead_lag_result.get('lstm_mean_lead', 0):.2f}d, "
                    f"Δ={lead_lag_result.get('lead_delta', 0):.2f}d")

        # Plot lead/lag figure
        plot_lead_lag_svb(pred_df, lstm_preds, optimal_threshold=0.4)
    else:
        logger.warning("LSTM predictions not found — skipping lead/lag plot")

    # Plot entanglement entropy
    plot_entanglement_entropy(pred_df)

    # ─── Compute metrics ─────────────────────────────────────────
    # Tune threshold on val
    Xv = torch.tensor(X_val, dtype=torch.float32)
    with torch.no_grad():
        out_val    = model(Xv)
        proba_val  = torch.softmax(out_val, dim=1).numpy()

    from src.training.train_classical import _tune_threshold_dl
    opt_thresh = _tune_threshold_dl(y_val, proba_val, None)

    metrics = compute_all_metrics(
        y_true=y_test,
        y_pred=pred_test,
        probs=proba_test,
        model_name="VQH",
        version="v1",
        d_test=d_test,
        optimal_threshold=opt_thresh,
        hyperparams=_MODEL_CFG["vqh"],
    )

    # Add quantum-specific fields
    crisis_mask     = y_test == 0
    noncris_mask    = y_test != 0
    metrics["quantum_architecture"] = {
        "n_qubits":           3,
        "n_layers":           2,
        "n_quantum_parameters": sum(p.numel() for n, p in model.qnode.named_parameters() if "weights" in n),
        "entangling_structure": "StronglyEntanglingLayers (all-to-all CNOT)",
    }
    metrics["quantum_metrics"] = {
        "mean_entanglement_entropy_crisis":  float(entropy[crisis_mask].mean()) if crisis_mask.any() else 0.0,
        "mean_entanglement_entropy_noncris": float(entropy[noncris_mask].mean()) if noncris_mask.any() else 0.0,
        "crisis_timing_lead_days":   lead_lag_result.get("lead_delta", 0.0),
        "lead_lag_detail":           lead_lag_result,
    }

    save_metrics(metrics, "vqh_metrics")

    pr_curve = compute_pr_curve(y_test, proba_test)
    save_json(pr_curve, _ROOT / "outputs/metrics/vqh_pr_curve.json")

    # ─── Update benchmark ────────────────────────────────────────
    from src.utils.io import load_json
    all_metrics = {}
    alias_map = {
        "xgboost": "xgb",
    }

    for name in ["hmm", "xgboost", "lstm", "tft"]:
        path_key = alias_map.get(name, name)
        path = _ROOT / _PATHS[f"{path_key}_metrics"]
        if path.exists():
            import json
            with open(path) as f:
                m = json.load(f)
            all_metrics[m["model_name"]] = m

    all_metrics["VQH"] = metrics
    save_benchmark(all_metrics, include_vqh=True)

    print(f"\nVQH: crisis PR-AUC = {metrics['crisis_specific']['pr_auc']:.4f}")
    print(f"VQH lead vs {best_name}: {lead_lag_result.get('lead_delta', 0):.2f} days")

    return metrics
