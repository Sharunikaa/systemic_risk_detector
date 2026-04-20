"""tests/test_features.py — Unit tests for Phase 1 feature engineering."""
import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

import numpy as np
import pandas as pd
import pytest

from src.data.features import compute_log_returns, winsorize_returns, compute_correlations
from src.data.labelling import build_3class_labels, build_crisis_flag


def _make_prices(n=200, seed=42):
    rng = np.random.default_rng(seed)
    idx = pd.date_range("2020-01-01", periods=n, freq="B")
    data = {}
    for col in ["SPX", "GOLD", "BTC", "VIX"]:
        p0 = 100.0
        rets = rng.normal(0, 0.01, n)
        prices = p0 * np.exp(np.cumsum(rets))
        data[col] = prices
    return pd.DataFrame(data, index=idx)


def test_log_returns_shape():
    prices = _make_prices(100)
    rets = compute_log_returns(prices)
    assert rets.shape == (99, 4), f"Expected 99 rows, got {rets.shape[0]}"
    assert list(rets.columns) == ["spx_ret", "gold_ret", "btc_ret", "vix_ret"]


def test_log_returns_no_nan():
    prices = _make_prices(100)
    rets = compute_log_returns(prices)
    assert rets.isna().sum().sum() == 0, "Log returns should have no NaN"


def test_winsorize_preserves_shape():
    prices = _make_prices(100)
    rets = compute_log_returns(prices)
    winsorized = winsorize_returns(rets)
    # Should have same number of rows + is_extreme column
    assert len(winsorized) == len(rets)
    assert "is_extreme" in winsorized.columns


def test_winsorize_vix_unchanged():
    """VIX must NOT be winsorized."""
    prices = _make_prices(150)
    rets = compute_log_returns(prices)
    winsorized = winsorize_returns(rets)
    pd.testing.assert_series_equal(rets["vix_ret"], winsorized["vix_ret"],
                                   check_names=False)


def test_correlations_columns():
    prices = _make_prices(200)
    rets = compute_log_returns(prices)
    corr = compute_correlations(rets)
    expected = [
        "rho_spx_btc_30", "rho_spx_gold_30", "rho_btc_gold_30",
        "rho_spx_btc_60", "rho_spx_gold_60", "rho_btc_gold_60",
    ]
    for col in expected:
        assert col in corr.columns, f"Missing column: {col}"


def test_correlations_range():
    prices = _make_prices(300)
    rets = compute_log_returns(prices)
    corr = compute_correlations(rets)
    short_cols = [c for c in corr.columns if "_30" in c]
    for col in short_cols:
        vals = corr[col].dropna()
        assert (vals >= -1.0).all() and (vals <= 1.0).all(), f"{col} out of [-1,1]"


def test_3class_labels_valid():
    n = 100
    crisis_flag = pd.Series(np.random.choice([0, 1], n, p=[0.9, 0.1]))
    btc_garch   = pd.Series(np.abs(np.random.normal(0.3, 0.1, n)))
    labels = build_3class_labels(crisis_flag, btc_garch)
    assert set(labels.unique()).issubset({0, 1, 2}), "Labels must be in {0,1,2}"
    # All crisis_flag=1 → label=0
    assert (labels[crisis_flag == 1] == 0).all()
