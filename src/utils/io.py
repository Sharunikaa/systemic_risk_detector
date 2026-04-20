"""
src/utils/io.py

I/O helpers for saving and loading all pipeline artifacts.

Strict naming conventions:
  models/hmm_v1.pkl        ← scikit-learn / hmmlearn models (pickle)
  models/lstm_v5.pt        ← PyTorch state_dict
  models/vqh_v1.pt
  predictions/hmm_predictions.csv
  metrics/hmm_metrics.json
  metrics/benchmark_results.json
"""

from __future__ import annotations

import json
import pickle
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd
import torch
import yaml
from loguru import logger

_ROOT  = Path(__file__).resolve().parents[2]
_PATHS = yaml.safe_load((_ROOT / "config" / "paths.yaml").read_text())


def _resolve(key: str) -> Path:
    """Resolve a paths.yaml key to an absolute Path and create parent dir."""
    p = _ROOT / _PATHS[key]
    p.parent.mkdir(parents=True, exist_ok=True)
    return p


# ─── Pickle (sklearn / hmmlearn) ────────────────────────────────

def save_pkl(obj: Any, key: str) -> Path:
    """Save object as pickle."""
    path = _resolve(key)
    with open(path, "wb") as f:
        pickle.dump(obj, f, protocol=pickle.HIGHEST_PROTOCOL)
    logger.info(f"Saved pkl → {path}")
    return path


def load_pkl(key: str) -> Any:
    """Load object from pickle."""
    path = _resolve(key)
    with open(path, "rb") as f:
        return pickle.load(f)


# ─── PyTorch state_dict ──────────────────────────────────────────

def save_pt(state_dict: dict, key: str) -> Path:
    """Save PyTorch state_dict."""
    path = _resolve(key)
    torch.save(state_dict, path)
    logger.info(f"Saved pt → {path}")
    return path


def load_pt(key: str) -> dict:
    """Load PyTorch state_dict."""
    path = _resolve(key)
    return torch.load(path, map_location="cpu")


# ─── JSON ────────────────────────────────────────────────────────

def save_json(obj: Any, path: Path | str) -> Path:
    """Save dict as JSON. Handles numpy types automatically."""
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)

    def _default(o):
        if isinstance(o, (np.integer,)):  return int(o)
        if isinstance(o, (np.floating,)): return float(o)
        if isinstance(o, np.ndarray):     return o.tolist()
        raise TypeError(f"Not JSON serializable: {type(o)}")

    with open(path, "w") as f:
        json.dump(obj, f, indent=2, default=_default)
    logger.debug(f"Saved json → {path}")
    return path


def load_json(path: Path | str) -> Any:
    """Load JSON file."""
    path = Path(path)
    if not path.exists():
        return None
    with open(path) as f:
        return json.load(f)


# ─── Metrics ─────────────────────────────────────────────────────

def save_metrics(metrics: dict, key: str) -> Path:
    """Save metrics dict to its canonical path (from paths.yaml)."""
    path = _resolve(key)
    return save_json(metrics, path)


def load_metrics(key: str) -> dict | None:
    """Load metrics dict from paths.yaml path."""
    path = _resolve(key)
    return load_json(path)


# ─── Predictions CSV ─────────────────────────────────────────────

def save_predictions(df: pd.DataFrame, key: str) -> Path:
    """Save prediction DataFrame to CSV."""
    path = _resolve(key)
    df.to_csv(path)
    logger.info(f"Saved predictions → {path}  ({len(df)} rows)")
    return path


def load_predictions(key: str) -> pd.DataFrame | None:
    """Load prediction DataFrame from CSV."""
    path = _resolve(key)
    if not path.exists():
        logger.warning(f"Predictions not found: {path}")
        return None
    return pd.read_csv(path, index_col=0, parse_dates=True)
