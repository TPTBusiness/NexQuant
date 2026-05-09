"""Tests for KronosAdapter and CLI commands — mock-based, no real model download needed."""

import json
import numpy as np
import pandas as pd
import pytest
from pathlib import Path
from unittest.mock import patch, MagicMock


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def _make_ohlcv(n: int = 600, freq: str = "1min") -> pd.DataFrame:
    """Synthetic 1-min OHLCV DataFrame."""
    idx = pd.date_range("2024-01-01", periods=n, freq=freq)
    close = 1.1000 + np.cumsum(np.random.randn(n) * 0.0001)
    return pd.DataFrame({
        "open": close + np.random.randn(n) * 0.00005,
        "high": close + np.abs(np.random.randn(n) * 0.0001),
        "low": close - np.abs(np.random.randn(n) * 0.0001),
        "close": close,
        "volume": np.abs(np.random.randn(n) * 100),
    }, index=idx)


def _make_nexquant_hdf5(tmp_path: Path, n: int = 300) -> Path:
    """Write a minimal NexQuant-format HDF5 file and return its path."""
    idx = pd.MultiIndex.from_arrays(
        [pd.date_range("2024-01-01", periods=n, freq="1min"), ["EURUSD"] * n],
        names=["datetime", "instrument"],
    )
    df = pd.DataFrame({
        "$open": (np.random.rand(n) + 1.1).astype("float32"),
        "$close": (np.random.rand(n) + 1.1).astype("float32"),
        "$high": (np.random.rand(n) + 1.11).astype("float32"),
        "$low": (np.random.rand(n) + 1.09).astype("float32"),
        "$volume": (np.random.rand(n) * 100).astype("float32"),
    }, index=idx)
    h5 = tmp_path / "intraday_pv.h5"
    df.to_hdf(h5, key="data", mode="w")
    return h5


def _make_mock_adapter():
    """Return a mock KronosAdapter whose predict_next_bars is deterministic."""
    class MockAdapter:
        def load(self): return self
        def predict_next_bars(self, ohlcv_df, context_bars, pred_bars, **kw):
            idx = pd.date_range(ohlcv_df.index[-1], periods=pred_bars + 1, freq="1min")[1:]
            last_close = float(ohlcv_df["close"].iloc[-1])
            return pd.DataFrame({
                "open": last_close * 1.001,
                "close": last_close * 1.002,
                "high": last_close * 1.003,
                "low": last_close * 0.999,
                "volume": 100.0,
            }, index=idx)
        def predict_return(self, ohlcv_df, context_bars=512, pred_bars=1):
            return 0.001
    return MockAdapter()


# ---------------------------------------------------------------------------
# Unit tests: _ohlcv_from_nexquant
# ---------------------------------------------------------------------------

class TestOhlcvConversion:
    def test_renames_dollar_columns(self):
        from rdagent.components.coder.kronos_adapter import _ohlcv_from_nexquant
        idx = pd.MultiIndex.from_arrays(
            [pd.date_range("2024-01-01", periods=3, freq="1min"), ["EURUSD"] * 3],
            names=["datetime", "instrument"],
        )
        df = pd.DataFrame({
            "$open": [1.1, 1.2, 1.3], "$high": [1.15, 1.25, 1.35],
            "$low": [1.05, 1.15, 1.25], "$close": [1.12, 1.22, 1.32],
            "$volume": [100.0, 200.0, 300.0],
        }, index=idx)
        result = _ohlcv_from_nexquant(df)
        assert list(result.columns) == ["open", "high", "low", "close", "volume"]

    def test_no_dollar_columns_passthrough(self):
        from rdagent.components.coder.kronos_adapter import _ohlcv_from_nexquant
        df = pd.DataFrame({"open": [1.0], "close": [1.1], "high": [1.2], "low": [0.9], "volume": [100.0]})
        result = _ohlcv_from_nexquant(df)
        assert "close" in result.columns

    def test_output_is_float64(self):
        from rdagent.components.coder.kronos_adapter import _ohlcv_from_nexquant
        df = pd.DataFrame({
            "$open": np.array([1.1], dtype="float32"),
            "$close": np.array([1.1], dtype="float32"),
            "$high": np.array([1.1], dtype="float32"),
            "$low": np.array([1.1], dtype="float32"),
            "$volume": np.array([100.0], dtype="float32"),
        })
        result = _ohlcv_from_nexquant(df)
        assert result["close"].dtype == np.float64


# ---------------------------------------------------------------------------
# Unit tests: KronosAdapter availability check
# ---------------------------------------------------------------------------

class TestKronosAvailability:
    def test_unavailable_without_repo(self, tmp_path, monkeypatch):
        import rdagent.components.coder.kronos_adapter as mod
        monkeypatch.setattr(mod, "KRONOS_REPO", tmp_path / "nonexistent")
        monkeypatch.setattr(mod, "_KRONOS_AVAILABLE", None)
        assert mod._ensure_kronos() is False

    def test_load_raises_without_repo(self, tmp_path, monkeypatch):
        import rdagent.components.coder.kronos_adapter as mod
        monkeypatch.setattr(mod, "KRONOS_REPO", tmp_path / "nonexistent")
        monkeypatch.setattr(mod, "_KRONOS_AVAILABLE", None)
        adapter = mod.KronosAdapter()
        with pytest.raises(RuntimeError, match="Kronos not available"):
            adapter.load()

    def test_predict_without_load_raises(self):
        from rdagent.components.coder.kronos_adapter import KronosAdapter
        adapter = KronosAdapter()
        with pytest.raises(RuntimeError, match="Call .load()"):
            adapter.predict_next_bars(_make_ohlcv(100), 50, 10)


# ---------------------------------------------------------------------------
# Unit tests: build_kronos_factor
# ---------------------------------------------------------------------------

class TestBuildKronosFactor:
    def test_output_has_correct_multiindex(self, tmp_path, monkeypatch):
        import rdagent.components.coder.kronos_adapter as mod
        monkeypatch.setattr(mod, "KronosAdapter", lambda **kw: _make_mock_adapter())
        h5 = _make_nexquant_hdf5(tmp_path)
        result = mod.build_kronos_factor(h5, context_bars=100, pred_bars=20, stride_bars=20, device="cpu")
        assert result.index.names == ["datetime", "instrument"]
        assert result.index.nlevels == 2

    def test_output_column_name(self, tmp_path, monkeypatch):
        import rdagent.components.coder.kronos_adapter as mod
        monkeypatch.setattr(mod, "KronosAdapter", lambda **kw: _make_mock_adapter())
        h5 = _make_nexquant_hdf5(tmp_path)
        result = mod.build_kronos_factor(h5, context_bars=100, pred_bars=20, stride_bars=20, device="cpu")
        assert "KronosPredReturn" in result.columns

    def test_output_has_non_nan_values(self, tmp_path, monkeypatch):
        import rdagent.components.coder.kronos_adapter as mod
        monkeypatch.setattr(mod, "KronosAdapter", lambda **kw: _make_mock_adapter())
        h5 = _make_nexquant_hdf5(tmp_path)
        result = mod.build_kronos_factor(h5, context_bars=100, pred_bars=20, stride_bars=20, device="cpu")
        assert result["KronosPredReturn"].notna().sum() > 0

    def test_output_length_matches_input(self, tmp_path, monkeypatch):
        import rdagent.components.coder.kronos_adapter as mod
        monkeypatch.setattr(mod, "KronosAdapter", lambda **kw: _make_mock_adapter())
        n = 300
        h5 = _make_nexquant_hdf5(tmp_path, n=n)
        result = mod.build_kronos_factor(h5, context_bars=100, pred_bars=20, stride_bars=20, device="cpu")
        assert len(result) == n

    def test_forward_fill_propagates_signal(self, tmp_path, monkeypatch):
        """Values within a predicted window should be forward-filled, not NaN."""
        import rdagent.components.coder.kronos_adapter as mod
        monkeypatch.setattr(mod, "KronosAdapter", lambda **kw: _make_mock_adapter())
        h5 = _make_nexquant_hdf5(tmp_path, n=300)
        result = mod.build_kronos_factor(h5, context_bars=100, pred_bars=20, stride_bars=20, device="cpu")
        non_nan_ratio = result["KronosPredReturn"].notna().mean()
        assert non_nan_ratio > 0.5, f"Expected >50% non-NaN, got {non_nan_ratio:.2%}"

    def test_raises_on_missing_hdf5(self, tmp_path, monkeypatch):
        import rdagent.components.coder.kronos_adapter as mod
        monkeypatch.setattr(mod, "KronosAdapter", lambda **kw: _make_mock_adapter())
        with pytest.raises(Exception):
            mod.build_kronos_factor(tmp_path / "missing.h5", context_bars=50, pred_bars=10, stride_bars=10)


# ---------------------------------------------------------------------------
# Unit tests: evaluate_kronos_model
# ---------------------------------------------------------------------------

class TestEvaluateKronosModel:
    def test_returns_required_keys(self, tmp_path, monkeypatch):
        import rdagent.components.coder.kronos_adapter as mod
        monkeypatch.setattr(mod, "KronosAdapter", lambda **kw: _make_mock_adapter())
        h5 = _make_nexquant_hdf5(tmp_path, n=400)
        metrics = mod.evaluate_kronos_model(h5, context_bars=100, pred_bars=20, stride_bars=20, device="cpu")
        for key in ["IC_mean", "IC_std", "IC_IR", "hit_rate", "n_predictions"]:
            assert key in metrics, f"Missing key: {key}"

    def test_hit_rate_in_valid_range(self, tmp_path, monkeypatch):
        import rdagent.components.coder.kronos_adapter as mod
        monkeypatch.setattr(mod, "KronosAdapter", lambda **kw: _make_mock_adapter())
        h5 = _make_nexquant_hdf5(tmp_path, n=400)
        metrics = mod.evaluate_kronos_model(h5, context_bars=100, pred_bars=20, stride_bars=20, device="cpu")
        assert 0.0 <= metrics["hit_rate"] <= 1.0

    def test_n_predictions_positive(self, tmp_path, monkeypatch):
        import rdagent.components.coder.kronos_adapter as mod
        monkeypatch.setattr(mod, "KronosAdapter", lambda **kw: _make_mock_adapter())
        h5 = _make_nexquant_hdf5(tmp_path, n=400)
        metrics = mod.evaluate_kronos_model(h5, context_bars=100, pred_bars=20, stride_bars=20, device="cpu")
        assert metrics["n_predictions"] > 0


# ---------------------------------------------------------------------------
# Integration tests: CLI commands (via typer test runner)
# ---------------------------------------------------------------------------

class TestCLICommands:
    def test_kronos_factor_missing_data_exits(self, tmp_path, monkeypatch):
        """kronos-factor exits with code 1 when HDF5 data is missing."""
        from typer.testing import CliRunner
        import nexquant as nexquant_mod
        monkeypatch.chdir(tmp_path)
        runner = CliRunner()
        result = runner.invoke(nexquant_mod.app, ["kronos-factor"])
        assert result.exit_code == 1

    def test_kronos_eval_missing_data_exits(self, tmp_path, monkeypatch):
        """kronos-eval exits with code 1 when HDF5 data is missing."""
        from typer.testing import CliRunner
        import nexquant as nexquant_mod
        monkeypatch.chdir(tmp_path)
        runner = CliRunner()
        result = runner.invoke(nexquant_mod.app, ["kronos-eval"])
        assert result.exit_code == 1

    def test_kronos_factor_runs_with_mock(self, tmp_path, monkeypatch):
        """kronos-factor completes and saves parquet + json when adapter is mocked."""
        from typer.testing import CliRunner
        import rdagent.components.coder.kronos_adapter as mod
        import nexquant as nexquant_mod

        monkeypatch.setattr(mod, "KronosAdapter", lambda **kw: _make_mock_adapter())

        data_dir = tmp_path / "git_ignore_folder" / "factor_implementation_source_data"
        data_dir.mkdir(parents=True)
        _make_nexquant_hdf5(data_dir.parent.parent, n=300)
        h5_src = tmp_path / "intraday_pv.h5"
        # Put HDF5 where the CLI expects it
        import shutil
        src = _make_nexquant_hdf5(tmp_path, n=300)
        shutil.copy(src, data_dir / "intraday_pv.h5")

        monkeypatch.chdir(tmp_path)
        runner = CliRunner()
        result = runner.invoke(nexquant_mod.app, [
            "kronos-factor", "--context", "100", "--pred", "20", "--device", "cpu"
        ])
        assert result.exit_code == 0, result.output
        assert "saved" in result.output.lower()

    def test_kronos_eval_runs_with_mock(self, tmp_path, monkeypatch):
        """kronos-eval completes and prints IC metrics when adapter is mocked."""
        from typer.testing import CliRunner
        import rdagent.components.coder.kronos_adapter as mod
        import nexquant as nexquant_mod

        monkeypatch.setattr(mod, "KronosAdapter", lambda **kw: _make_mock_adapter())

        data_dir = tmp_path / "git_ignore_folder" / "factor_implementation_source_data"
        data_dir.mkdir(parents=True)
        _make_nexquant_hdf5(data_dir.parent.parent, n=400)
        src = _make_nexquant_hdf5(tmp_path, n=400)
        import shutil
        shutil.copy(src, data_dir / "intraday_pv.h5")

        monkeypatch.chdir(tmp_path)
        runner = CliRunner()
        result = runner.invoke(nexquant_mod.app, [
            "kronos-eval", "--context", "100", "--pred", "20", "--device", "cpu"
        ])
        assert result.exit_code == 0, result.output
        assert "IC" in result.output
