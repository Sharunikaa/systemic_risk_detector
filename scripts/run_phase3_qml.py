"""
scripts/run_phase3_qml.py

Phase 3 entry point — Quantum ML (VQH).
Expected runtime: 20-60 minutes (quantum simulation is CPU-bound).
Uses pennylane-lightning for 2-3x speedup over default.qubit.

Required outputs:
  outputs/models/vqh_v1.pt
  outputs/predictions/vqh_predictions.csv  (includes entanglement_entropy column)
  outputs/metrics/vqh_metrics.json
  outputs/metrics/benchmark_results_final.json  (all 5 models)
  outputs/figures/lead_lag_SVB2023.png
  outputs/figures/entanglement_entropy.png
"""

import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from loguru import logger
from src.training.train_vqh import run_vqh_training

if __name__ == "__main__":
    logger.info("Starting Phase 3: Quantum ML (VQH)")
    metrics = run_vqh_training()
    print(f"\nVQH crisis PR-AUC: {metrics['crisis_specific']['pr_auc']:.4f}")
    qm = metrics.get("quantum_metrics", {})
    print(f"Mean entanglement entropy (crisis):   {qm.get('mean_entanglement_entropy_crisis', 0):.4f}")
    print(f"Mean entanglement entropy (non-crisis): {qm.get('mean_entanglement_entropy_noncris', 0):.4f}")
    print(f"Crisis timing lead: {qm.get('crisis_timing_lead_days', 0):.2f} days vs best classical")
    print("\nPhase 3 complete.")
