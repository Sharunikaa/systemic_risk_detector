"""
scripts/run_phase2_classical.py

Phase 2 entry point — Classical ML/DL Training.
Expected runtime: 15-45 minutes (TFT is the bottleneck).
Outputs:
  outputs/metrics/benchmark_results.json
  outputs/metrics/best_model.txt
"""

import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from loguru import logger
from src.training.train_classical import run_all_classical

if __name__ == "__main__":
    logger.info("Starting Phase 2: Classical ML/DL Training")
    benchmark = run_all_classical()
    best_name  = benchmark["best_model_by_crisis_pr_auc"]
    best_prauc = benchmark["comparison"]["crisis_pr_auc"].get(best_name, 0)
    print(f"\nBest model: {best_name} (crisis PR-AUC = {best_prauc:.4f})")
    print("Predictions ready for VQH encoder.")
