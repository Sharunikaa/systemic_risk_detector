"""
src/utils/sequences.py

Sliding window sequence construction and crisis augmentation.

Sequences:
  For LSTM and TFT (seq_len = 20):
    X_seq[i] = X[i : i + seq_len]         shape: (seq_len, n_features)
    y_seq[i] = y[i + seq_len - 1]          label at END of window
    d_seq[i] = dates[i + seq_len - 1]      date at END of window

  The label corresponds to the LAST day of the window.
  This is "classify the current regime given the last 20 days of history."

Crisis augmentation:
  Gaussian noise is injected into crisis sequences during training ONLY.
  This prevents the model from memorising exact crisis feature values and
  forces it to learn the distributional properties of crisis regimes.
  Applied to: X_train (sequences where y == 0).
  NOT applied to val or test sets.
"""

from __future__ import annotations

from typing import Tuple

import numpy as np
import pandas as pd


def build_sequences(
    X: np.ndarray,
    y: np.ndarray,
    dates: pd.DatetimeIndex,
    seq_len: int = 20,
) -> Tuple[np.ndarray, np.ndarray, pd.DatetimeIndex]:
    """
    Build sliding window sequences from flat feature arrays.

    Args:
        X       : (N, n_features) feature array, already scaled
        y       : (N,) regime labels
        dates   : DatetimeIndex of length N
        seq_len : window length in trading days (default: 20)

    Returns:
        X_seq : (N - seq_len + 1, seq_len, n_features)
        y_seq : (N - seq_len + 1,) — label at the last timestep of each window
        d_seq : DatetimeIndex — dates at the last timestep of each window
    """
    n = len(X)
    n_out = n - seq_len + 1

    if n_out <= 0:
        raise ValueError(
            f"Not enough data: {n} rows with seq_len={seq_len}. "
            f"Need at least {seq_len} rows."
        )

    X_seq = np.stack([X[i : i + seq_len] for i in range(n_out)])  # (n_out, seq_len, F)
    y_seq = y[seq_len - 1 :]                                        # (n_out,)
    d_seq = dates[seq_len - 1 :]                                    # DatetimeIndex of len n_out

    assert len(X_seq) == len(y_seq) == len(d_seq), (
        f"Shape mismatch: X={len(X_seq)}, y={len(y_seq)}, d={len(d_seq)}"
    )

    return X_seq, y_seq, d_seq


def augment_crisis_sequences(
    X_crisis: np.ndarray,
    y_crisis: np.ndarray,
    factor: int = 3,
    noise_std: float = 0.01,
    rng: np.random.Generator | None = None,
) -> Tuple[np.ndarray, np.ndarray]:
    """
    Augment crisis sequences by injecting Gaussian noise.

    Creates `factor` copies of each crisis sequence, each with independent
    Gaussian noise (mean=0, std=noise_std) added to all feature values.

    Applied to training data ONLY. Never val or test.

    Args:
        X_crisis  : (N_crisis, seq_len, n_features) crisis sequences
        y_crisis  : (N_crisis,) all zeros (crisis label)
        factor    : number of augmented copies per sequence (3 per spec)
        noise_std : standard deviation of Gaussian noise (0.01 per spec)
        rng       : numpy random Generator (for reproducibility)

    Returns:
        X_aug : (N_crisis * factor, seq_len, n_features)
        y_aug : (N_crisis * factor,) all zeros
    """
    if rng is None:
        rng = np.random.default_rng(42)

    if len(X_crisis) == 0:
        return (
            np.empty((0,) + X_crisis.shape[1:], dtype=X_crisis.dtype),
            np.empty(0, dtype=y_crisis.dtype),
        )

    augmented_X = []
    for _ in range(factor):
        noise = rng.normal(0.0, noise_std, size=X_crisis.shape).astype(X_crisis.dtype)
        augmented_X.append(X_crisis + noise)

    X_aug = np.concatenate(augmented_X, axis=0)
    y_aug = np.zeros(len(X_aug), dtype=y_crisis.dtype)

    return X_aug, y_aug
