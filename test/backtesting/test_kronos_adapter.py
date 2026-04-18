"""Tests for KronosAdapter — mock-based, no real model download needed."""

import numpy as np
import pandas as pd
import pytest


def _make_ohlcv(n: int = 600) -> pd.DataFrame:
    """Synthetic 1-min OHLCV DataFrame."""
    idx = pd.date_range("2024-01-01", periods=n, freq="1min")
    close = 1.1000 + np.cumsum(np.random.randn(n) * 0.0001)
    df = pd.DataFrame({
        "open": close + np.random.randn(n) * 0.00005,
        "high": close + np.abs(np.random.randn(n) * 0.0001),
        "low": close - np.abs(np.random.randn(n) * 0.0001),
        "close": close,
        "volume": np.abs(np.random.randn(n) * 100),
    }, index=idx)
    return df


def test_ohlcv_conversion():
    """_ohlcv_from_predix renames $ columns correctly."""
    from rdagent.components.coder.kronos_adapter import _ohlcv_from_predix

    idx = pd.MultiIndex.from_arrays(
        [pd.date_range("2024-01-01", periods=3, freq="1min"), ["EURUSD"] * 3],
        names=["datetime", "instrument"],
    )
    predix_df = pd.DataFrame({
        "$open": [1.1, 1.2, 1.3],
        "$high": [1.15, 1.25, 1.35],
        "$low": [1.05, 1.15, 1.25],
        "$close": [1.12, 1.22, 1.32],
        "$volume": [100.0, 200.0, 300.0],
    }, index=idx)

    ohlcv = _ohlcv_from_predix(predix_df)
    assert "close" in ohlcv.columns
    assert "$close" not in ohlcv.columns
    assert list(ohlcv.columns) == ["open", "high", "low", "close", "volume"]


def test_kronos_adapter_load_skipped_without_repo(tmp_path, monkeypatch):
    """KronosAdapter gracefully reports unavailable when repo is missing."""
    import rdagent.components.coder.kronos_adapter as mod
    monkeypatch.setattr(mod, "KRONOS_REPO", tmp_path / "nonexistent")
    monkeypatch.setattr(mod, "_KRONOS_AVAILABLE", None)

    from rdagent.components.coder.kronos_adapter import KronosAdapter, _ensure_kronos
    available = _ensure_kronos()
    assert available is False


def test_build_kronos_factor_mock(tmp_path, monkeypatch):
    """build_kronos_factor produces correct MultiIndex output with mocked predictor."""
    import rdagent.components.coder.kronos_adapter as mod

    # Mock the adapter so no real Kronos load happens
    class MockAdapter:
        def load(self): return self
        def predict_next_bars(self, ohlcv_df, context_bars, pred_bars, **kw):
            idx = pd.date_range(ohlcv_df.index[-1], periods=pred_bars + 1, freq="1min")[1:]
            last_close = float(ohlcv_df["close"].iloc[-1])
            return pd.DataFrame({
                "open": last_close * (1 + np.random.randn(pred_bars) * 0.001),
                "close": last_close * (1 + np.random.randn(pred_bars) * 0.001),
                "high": last_close * 1.001,
                "low": last_close * 0.999,
                "volume": 100.0,
            }, index=idx)

    monkeypatch.setattr(mod, "KronosAdapter", lambda **kw: MockAdapter())

    # Write minimal HDF5
    n = 300
    idx = pd.MultiIndex.from_arrays(
        [pd.date_range("2024-01-01", periods=n, freq="1min"), ["EURUSD"] * n],
        names=["datetime", "instrument"],
    )
    df = pd.DataFrame({
        "$open": np.random.rand(n).astype("float32") + 1.1,
        "$close": np.random.rand(n).astype("float32") + 1.1,
        "$high": np.random.rand(n).astype("float32") + 1.11,
        "$low": np.random.rand(n).astype("float32") + 1.09,
        "$volume": np.random.rand(n).astype("float32") * 100,
    }, index=idx)
    h5_path = tmp_path / "intraday_pv.h5"
    df.to_hdf(h5_path, key="data", mode="w")

    result = mod.build_kronos_factor(
        hdf5_path=h5_path,
        context_bars=100,
        pred_bars=20,
        stride_bars=20,
        device="cpu",
    )

    assert isinstance(result, pd.DataFrame)
    assert result.index.names == ["datetime", "instrument"]
    assert "KronosPredReturn" in result.columns
    assert result["KronosPredReturn"].notna().sum() > 0
