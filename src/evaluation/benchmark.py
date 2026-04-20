"""
src/evaluation/benchmark.py

Cross-model comparison and best model selection.

Compares all 4 classical models (HMM, XGBoost, LSTM, TFT) by crisis PR-AUC
and selects the best model to feed into Phase 3 (VQH).

The comparison table is saved as benchmark_results.json (Phase 2) and
benchmark_results_final.json (Phase 3, includes VQH).
"""

from __future__ import annotations

from pathlib import Path
from typing import Any

import yaml
from loguru import logger

from src.utils.io import save_json, load_json

_ROOT = Path(__file__).resolve().parents[2]
_PATHS = yaml.safe_load((_ROOT / "config" / "paths.yaml").read_text())


def compile_benchmark(
    metrics_dict: dict[str, dict],
    include_vqh: bool = False,
) -> dict:
    """
    Compile cross-model benchmark comparison from individual metrics dicts.

    Args:
        metrics_dict : {model_name: metrics_dict} for all trained models
        include_vqh  : if True, include VQH in the comparison

    Returns:
        benchmark dict matching the JSON schema in SKILL.md
    """
    from datetime import datetime

    model_names = ["HMM", "XGBoost", "LSTM", "TFT"]
    if include_vqh:
        model_names.append("VQH")

    comparison: dict[str, dict] = {
        "accuracy":       {},
        "macro_f1":       {},
        "crisis_pr_auc":  {},
        "crisis_recall":  {},
        "roc_auc":        {},
    }

    for name in model_names:
        m = metrics_dict.get(name)
        if m is None:
            logger.warning(f"Missing metrics for {name} — skipping")
            continue
        comparison["accuracy"][name]      = m["overall"]["accuracy"]
        comparison["macro_f1"][name]      = m["overall"]["macro_f1"]
        comparison["crisis_pr_auc"][name] = m["crisis_specific"]["pr_auc"]
        comparison["crisis_recall"][name] = m["per_class"]["CRISIS"]["recall"]
        comparison["roc_auc"][name]       = m["crisis_specific"]["roc_auc"]

    # Select best model by crisis PR-AUC
    pr_aucs = comparison["crisis_pr_auc"]
    best_model = max(pr_aucs, key=pr_aucs.get, default="LSTM")

    benchmark = {
        "generated_at":             datetime.utcnow().isoformat(),
        "primary_metric":           "crisis_pr_auc",
        "best_model_by_crisis_pr_auc": best_model,
        "models":                   model_names,
        "comparison":               comparison,
    }

    logger.info(f"Best model: {best_model} (crisis PR-AUC = {pr_aucs.get(best_model, 'N/A'):.4f})")
    return benchmark


def save_benchmark(
    metrics_dict: dict[str, dict],
    include_vqh: bool = False,
) -> dict:
    """
    Compile and save benchmark results to JSON file.

    Args:
        metrics_dict : {model_name: metrics_dict} for all models
        include_vqh  : if True, save to benchmark_results_final.json

    Returns:
        benchmark dict
    """
    benchmark = compile_benchmark(metrics_dict, include_vqh=include_vqh)

    path_key  = "benchmark_final" if include_vqh else "benchmark_results"
    path      = _ROOT / _PATHS[path_key]
    save_json(benchmark, path)

    # Write best_model.txt
    best_name = benchmark["best_model_by_crisis_pr_auc"]
    best_path = _ROOT / _PATHS["best_model"]
    best_path.parent.mkdir(parents=True, exist_ok=True)
    best_path.write_text(best_name)
    logger.info(f"Best model name written → {best_path}")

    return benchmark
