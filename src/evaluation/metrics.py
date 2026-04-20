"""
src/evaluation/metrics.py

Comprehensive metrics computation for all 5 models.

Primary metric: Crisis PR-AUC (precision-recall area under curve).
  Rationale: accuracy is misleading for imbalanced datasets where
  crisis = ~6% of days. A model that always predicts NORMAL gets 94% accuracy
  but zero crisis recall. PR-AUC captures the precision/recall tradeoff
  for the rare but most important class.

Secondary metrics: macro-F1, crisis recall, ROC-AUC, confusion matrix.

Black-swan evaluation windows:
  COVID_2020  : 2020-02-01 to 2020-05-31  (falls in training for most models)
  Crypto_2022 : 2022-05-01 to 2022-07-31  (in test window)
  SVB_2023    : 2023-03-01 to 2023-05-31  (in test window)
"""

from __future__ import annotations

from datetime import datetime
from pathlib import Path
from typing import Any, Dict

import numpy as np
import pandas as pd
import yaml
from loguru import logger
from sklearn.metrics import (
    accuracy_score,
    average_precision_score,
    classification_report,
    confusion_matrix,
    f1_score,
    precision_recall_curve,
    roc_auc_score,
)

_ROOT = Path(__file__).resolve().parents[2]
_DATA_CFG  = yaml.safe_load((_ROOT / "config" / "data.yaml").read_text())
_PATHS_CFG = yaml.safe_load((_ROOT / "config" / "paths.yaml").read_text())

CLASS_NAMES = ["CRISIS", "NORMAL", "HIGH-VOL"]


def compute_all_metrics(
    y_true: np.ndarray,
    y_pred: np.ndarray,
    probs: np.ndarray,
    model_name: str,
    version: str,
    d_test: pd.DatetimeIndex,
    optimal_threshold: float = 0.5,
    hyperparams: dict | None = None,
) -> dict:
    """
    Compute and return a complete metrics dictionary for one model.

    Args:
        y_true            : (N,) ground truth labels {0,1,2}
        y_pred            : (N,) predicted labels {0,1,2}
        probs             : (N,3) predicted probabilities [P(crisis), P(normal), P(highvol)]
        model_name        : model identifier string (e.g. "LSTM")
        version           : model version string (e.g. "v5")
        d_test            : DatetimeIndex corresponding to y_true rows
        optimal_threshold : crisis threshold tuned on val set
        hyperparams       : dict of model hyperparameters to record

    Returns:
        Complete metrics dict matching the JSON schema in SKILL.md
    """
    # ── Overall metrics ──────────────────────────────────────────
    acc        = float(accuracy_score(y_true, y_pred))
    macro_f1   = float(f1_score(y_true, y_pred, average="macro", zero_division=0))
    weighted_f1 = float(f1_score(y_true, y_pred, average="weighted", zero_division=0))

    # ── Per-class metrics ─────────────────────────────────────────
    report = classification_report(
        y_true, y_pred, target_names=CLASS_NAMES, output_dict=True, zero_division=0
    )
    per_class = {
        cls: {
            "precision": float(report[cls]["precision"]),
            "recall":    float(report[cls]["recall"]),
            "f1":        float(report[cls]["f1-score"]),
            "support":   int(report[cls]["support"]),
        }
        for cls in CLASS_NAMES
    }

    # ── Crisis-specific metrics ───────────────────────────────────
    y_binary = (y_true == 0).astype(int)         # 1 = CRISIS, 0 = other
    crisis_proba = probs[:, 0]                   # P(crisis)

    pr_auc  = float(average_precision_score(y_binary, crisis_proba))
    roc_auc = float(roc_auc_score(y_binary, crisis_proba))

    # ── Confusion matrix ──────────────────────────────────────────
    cm = confusion_matrix(y_true, y_pred, labels=[0, 1, 2]).tolist()

    # ── Black-swan evaluation ─────────────────────────────────────
    black_swan_results = evaluate_black_swan_windows(
        y_true, y_pred, d_test, _DATA_CFG["black_swan_windows"]
    )

    # ── Test window metadata ──────────────────────────────────────
    test_window = {
        "start": str(d_test.min().date()),
        "end":   str(d_test.max().date()),
    }

    metrics = {
        "model_name":        model_name,
        "version":           version,
        "timestamp":         datetime.utcnow().isoformat(),
        "test_window":       test_window,
        "hyperparameters":   hyperparams or {},
        "overall": {
            "accuracy":    round(acc, 4),
            "macro_f1":    round(macro_f1, 4),
            "weighted_f1": round(weighted_f1, 4),
        },
        "per_class":         per_class,
        "crisis_specific": {
            "pr_auc":             round(pr_auc, 4),
            "roc_auc":            round(roc_auc, 4),
            "optimal_threshold":  round(optimal_threshold, 4),
        },
        "black_swan":        black_swan_results,
        "confusion_matrix":  cm,
    }

    logger.info(
        f"{model_name}: acc={acc:.3f}, macro_f1={macro_f1:.3f}, "
        f"crisis_pr_auc={pr_auc:.3f}, crisis_recall={per_class['CRISIS']['recall']:.3f}"
    )
    return metrics


def evaluate_black_swan_windows(
    y_true: np.ndarray,
    y_pred: np.ndarray,
    dates: pd.DatetimeIndex,
    windows: dict,
) -> dict:
    """
    Per-event evaluation for known systemic crisis windows.

    For each window, reports:
      - in_test   : whether the window falls in the test set
      - n_crisis_days : actual crisis days (y_true == 0) in the window
      - n_detected    : crisis days with correct prediction
      - recall_pct    : recall percentage

    Args:
        y_true  : ground truth labels
        y_pred  : predicted labels
        dates   : DatetimeIndex aligned to y_true
        windows : dict of {name: {start, end}} from config

    Returns:
        dict matching the black_swan JSON schema
    """
    results = {}
    date_series = pd.Series(dates, name="date")

    for event_name, window in windows.items():
        ws = pd.Timestamp(window["start"])
        we = pd.Timestamp(window["end"])

        mask = (dates >= ws) & (dates <= we)
        if not mask.any():
            results[event_name] = {
                "in_test": False,
                "note":    f"Window {window['start']}–{window['end']} not in test set",
            }
            continue

        y_win    = y_true[mask]
        yp_win   = y_pred[mask]
        crisis_mask = y_win == 0

        n_crisis   = int(crisis_mask.sum())
        n_detected = int(((yp_win == 0) & crisis_mask).sum())
        recall     = n_detected / max(n_crisis, 1)

        results[event_name] = {
            "in_test":      True,
            "n_crisis_days": n_crisis,
            "n_detected":    n_detected,
            "recall":        round(recall, 4),
        }

    return results


def compute_pr_curve(
    y_true: np.ndarray,
    probs: np.ndarray,
) -> dict:
    """
    Compute precision-recall curve for the crisis class.

    Args:
        y_true : ground truth labels
        probs  : (N, 3) predicted probabilities

    Returns:
        dict with keys: precision, recall, thresholds, pr_auc
    """
    y_binary     = (y_true == 0).astype(int)
    crisis_proba = probs[:, 0]

    precision, recall, thresholds = precision_recall_curve(y_binary, crisis_proba)
    pr_auc = float(average_precision_score(y_binary, crisis_proba))

    return {
        "precision":   precision.tolist(),
        "recall":      recall.tolist(),
        "thresholds":  thresholds.tolist(),
        "pr_auc":      pr_auc,
        "baseline":    float(y_binary.mean()),  # random classifier line
    }


def crisis_timing_lead(
    vqh_prob_crisis: np.ndarray,
    lstm_prob_crisis: np.ndarray,
    true_regime: np.ndarray,
    dates: pd.DatetimeIndex,
    threshold: float = 0.4,
) -> dict:
    """
    Measure how many days before crisis onset each model first crosses
    the probability threshold.

    For each true crisis onset (first day of a crisis sequence),
    look backward for the earliest day the probability crossed `threshold`.
    Positive lead = model predicts crisis BEFORE it begins (ideal).

    Args:
        vqh_prob_crisis  : (T,) VQH probability of crisis at each timestep
        lstm_prob_crisis : (T,) LSTM probability of crisis
        true_regime      : (T,) true regime labels {0,1,2}
        dates            : DatetimeIndex of length T
        threshold        : probability threshold to consider as "detected"

    Returns:
        dict with mean/std lead times for both models and the delta
    """
    # Find crisis onset indices (first day of each crisis window)
    in_crisis = (true_regime == 0)
    onset_indices = []
    for i in range(1, len(in_crisis)):
        if in_crisis[i] and not in_crisis[i - 1]:
            onset_indices.append(i)

    if not onset_indices:
        return {
            "vqh_mean_lead": 0.0,
            "lstm_mean_lead": 0.0,
            "lead_delta": 0.0,
            "n_crises": 0,
        }

    max_lookback = 30  # look up to 30 days before onset
    vqh_leads, lstm_leads = [], []

    for onset in onset_indices:
        start_look = max(0, onset - max_lookback)

        # VQH lead: how many days before onset did VQH first cross threshold?
        vqh_lead = _compute_lead(vqh_prob_crisis, onset, start_look, threshold)
        lstm_lead = _compute_lead(lstm_prob_crisis, onset, start_look, threshold)

        vqh_leads.append(vqh_lead)
        lstm_leads.append(lstm_lead)

    vqh_mean  = float(np.mean(vqh_leads))
    lstm_mean = float(np.mean(lstm_leads))

    return {
        "vqh_mean_lead":  round(vqh_mean, 2),
        "lstm_mean_lead": round(lstm_mean, 2),
        "lead_delta":     round(vqh_mean - lstm_mean, 2),
        "n_crises":       len(onset_indices),
        "vqh_leads":      vqh_leads,
        "lstm_leads":     lstm_leads,
    }


def _compute_lead(proba: np.ndarray, onset: int, start: int, threshold: float) -> float:
    """
    Compute lead time in days for a single crisis onset.
    Returns the number of days before `onset` that `proba` first exceeded `threshold`.
    Returns 0 if never exceeded (late detection).
    """
    window = proba[start:onset]
    crossed = np.where(window >= threshold)[0]
    if len(crossed) == 0:
        return 0.0
    first_cross = crossed[0]
    lead = onset - (start + first_cross)
    return float(lead)
