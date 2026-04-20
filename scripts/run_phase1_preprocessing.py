"""
scripts/run_phase1_preprocessing.py

Phase 1 entry point — Data Preprocessing.
Expected runtime: 3-8 minutes (GARCH fitting is the bottleneck).
Output: data/6_features/feature_matrix.csv
"""

import sys
from pathlib import Path

# Add project root to sys.path so imports work
sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from loguru import logger
from src.data.pipeline import run_preprocessing

if __name__ == "__main__":
    logger.info("Starting Phase 1: Data Preprocessing")
    feature_matrix = run_preprocessing(force_download=True)
    logger.info("Phase 1 complete.")
    print(f"\nOutput: data/6_features/feature_matrix.csv")
    print(f"Shape:  {feature_matrix.shape}")
