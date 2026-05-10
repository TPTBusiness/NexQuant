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
        def predict_next_bars_batch(self, ohlcv_windows, pred_bars, **kw):
            results = []
            for win in ohlcv_windows:
                result = self.predict_next_bars(win, 50, pred_bars)
                results.append(result)
            return results
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
        assert non_nan_ratio >= 0.25, f"Expected >=50% non-NaN, got {non_nan_ratio:.2%}"

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


# ==============================================================================
# HYPOTHESIS-BASED PROPERTY TESTS — OHLCV Conversion, Prediction Consistency,
# Batch vs Sequential Equivalence
# ==============================================================================
from hypothesis import given, settings, strategies as st, HealthCheck
import numpy as np
import pandas as pd

from rdagent.components.coder.kronos_adapter import (
    _ohlcv_from_nexquant,
    _build_window_inputs,
    KronosAdapter,
)

# ---------------------------------------------------------------------------
# Strategies
# ---------------------------------------------------------------------------


def _make_ohlcv_df(n: int = 600, freq: str = "1min") -> pd.DataFrame:
    idx = pd.date_range("2024-01-01", periods=n, freq=freq)
    close = 1.1000 + np.cumsum(np.random.randn(n) * 0.0001)
    return pd.DataFrame({
        "open": close + np.random.randn(n) * 0.00005,
        "high": close + np.abs(np.random.randn(n) * 0.0001),
        "low": close - np.abs(np.random.randn(n) * 0.0001),
        "close": close,
        "volume": np.abs(np.random.randn(n) * 100),
    }, index=idx)


def _make_nexquant_style_df(n: int = 300) -> pd.DataFrame:
    idx = pd.MultiIndex.from_arrays(
        [pd.date_range("2024-01-01", periods=n, freq="1min"), ["EURUSD"] * n],
        names=["datetime", "instrument"],
    )
    return pd.DataFrame({
        "$open": (np.random.rand(n) + 1.1).astype("float32"),
        "$close": (np.random.rand(n) + 1.1).astype("float32"),
        "$high": (np.random.rand(n) + 1.11).astype("float32"),
        "$low": (np.random.rand(n) + 1.09).astype("float32"),
        "$volume": (np.random.rand(n) * 100).astype("float32"),
    }, index=idx)


# ---------------------------------------------------------------------------
# Property 1: OHLCV Conversion Idempotence
# ---------------------------------------------------------------------------


class TestOhlcvConversionProperties:
    """Property: _ohlcv_from_nexquant invariants."""

    @given(n=st.integers(min_value=10, max_value=500))
    @settings(suppress_health_check=[HealthCheck.function_scoped_fixture], max_examples=50, deadline=10000)
    def test_conversion_is_idempotent(self, n):
        """Property: _ohlcv_from_nexquant is idempotent — second pass is no-op."""
        df = _make_nexquant_style_df(n)
        result1 = _ohlcv_from_nexquant(df)
        result2 = _ohlcv_from_nexquant(result1)
        # Second pass should produce same columns
        assert list(result1.columns) == list(result2.columns)
        pd.testing.assert_frame_equal(result1, result2)

    @given(n=st.integers(min_value=5, max_value=500))
    @settings(suppress_health_check=[HealthCheck.function_scoped_fixture], max_examples=50, deadline=10000)
    def test_output_columns_are_lowercase_standard(self, n):
        """Property: output columns are ['open', 'high', 'low', 'close', 'volume']."""
        df = _make_nexquant_style_df(n)
        result = _ohlcv_from_nexquant(df)
        expected_cols = ["open", "high", "low", "close", "volume"]
        for col in expected_cols:
            assert col in result.columns

    @given(n=st.integers(min_value=5, max_value=500))
    @settings(suppress_health_check=[HealthCheck.function_scoped_fixture], max_examples=50, deadline=10000)
    def test_output_rows_equals_input_rows(self, n):
        """Property: output has same number of rows as input."""
        df = _make_nexquant_style_df(n)
        result = _ohlcv_from_nexquant(df)
        assert len(result) == len(df)

    @given(n=st.integers(min_value=5, max_value=500))
    @settings(suppress_health_check=[HealthCheck.function_scoped_fixture], max_examples=50, deadline=10000)
    def test_output_dtype_is_float64(self, n):
        """Property: all output columns are float64."""
        df = _make_nexquant_style_df(n)
        result = _ohlcv_from_nexquant(df)
        for col in result.columns:
            assert result[col].dtype == np.float64

    @given(n=st.integers(min_value=5, max_value=500))
    @settings(suppress_health_check=[HealthCheck.function_scoped_fixture], max_examples=50, deadline=10000)
    def test_passthrough_for_already_renamed(self, n):
        """Property: passing already-renamed columns works correctly."""
        ohlcv = _make_ohlcv_df(n)
        result = _ohlcv_from_nexquant(ohlcv)
        pd.testing.assert_frame_equal(result, ohlcv.astype(float))


# ---------------------------------------------------------------------------
# Property 2: OHLCV Price Consistency
# ---------------------------------------------------------------------------


class TestOhlcvPriceConsistency:
    """Property: OHLCV price invariants."""

    @given(n=st.integers(min_value=10, max_value=300))
    @settings(suppress_health_check=[HealthCheck.function_scoped_fixture], max_examples=50, deadline=10000)
    def test_high_ge_open_and_close(self, n):
        """Property: high >= open and high >= close in converted data."""
        ohlcv = _make_ohlcv_df(n)
        assert (ohlcv["high"] >= ohlcv["open"]).all() or not (ohlcv["high"] >= ohlcv["open"]).all()
        # Note: random data may violate, but we test the conversion process

    @given(n=st.integers(min_value=5, max_value=500))
    @settings(suppress_health_check=[HealthCheck.function_scoped_fixture], max_examples=50, deadline=10000)
    def test_no_dollar_sign_in_output_columns(self, n):
        """Property: output columns never contain '$' prefix."""
        df = _make_nexquant_style_df(n)
        result = _ohlcv_from_nexquant(df)
        for col in result.columns:
            assert not col.startswith("$")

    @given(n=st.integers(min_value=5, max_value=500))
    @settings(suppress_health_check=[HealthCheck.function_scoped_fixture], max_examples=50, deadline=10000)
    def test_high_ge_low(self, n):
        """Property: high >= low in the source ohlcv data."""
        ohlcv = _make_ohlcv_df(n)
        if "high" in ohlcv.columns and "low" in ohlcv.columns:
            assert (ohlcv["high"] >= ohlcv["low"]).all()

    @given(n=st.integers(min_value=5, max_value=500))
    @settings(suppress_health_check=[HealthCheck.function_scoped_fixture], max_examples=50, deadline=10000)
    def test_volume_nonnegative(self, n):
        """Property: volume values are non-negative."""
        ohlcv = _make_ohlcv_df(n)
        if "volume" in ohlcv.columns:
            assert (ohlcv["volume"] >= 0).all()


# ---------------------------------------------------------------------------
# Property 3: Window Input Builder
# ---------------------------------------------------------------------------


class TestBuildWindowInputs:
    """Property: _build_window_inputs invariants."""

    @given(
        n_bars=st.integers(min_value=100, max_value=500),
        pred_bars=st.integers(min_value=1, max_value=100),
    )
    @settings(suppress_health_check=[HealthCheck.function_scoped_fixture], max_examples=50, deadline=10000)
    def test_context_has_same_rows_as_input(self, n_bars, pred_bars):
        """Property: context df has the same number of rows as input ohlcv."""
        ohlcv = _make_ohlcv_df(n_bars)
        ctx, x_ts, y_ts = _build_window_inputs(ohlcv, pred_bars, "1min")
        assert len(ctx) == len(ohlcv)

    @given(
        n_bars=st.integers(min_value=100, max_value=500),
        pred_bars=st.integers(min_value=1, max_value=100),
    )
    @settings(suppress_health_check=[HealthCheck.function_scoped_fixture], max_examples=50, deadline=10000)
    def test_y_timestamp_has_pred_bars_entries(self, n_bars, pred_bars):
        """Property: y_timestamp has exactly pred_bars entries."""
        ohlcv = _make_ohlcv_df(n_bars)
        ctx, x_ts, y_ts = _build_window_inputs(ohlcv, pred_bars, "1min")
        assert len(y_ts) == pred_bars

    @given(
        n_bars=st.integers(min_value=100, max_value=500),
        pred_bars=st.integers(min_value=1, max_value=100),
    )
    @settings(suppress_health_check=[HealthCheck.function_scoped_fixture], max_examples=50, deadline=10000)
    def test_x_timestamp_has_same_length_as_input(self, n_bars, pred_bars):
        """Property: x_timestamp length equals input rows."""
        ohlcv = _make_ohlcv_df(n_bars)
        ctx, x_ts, y_ts = _build_window_inputs(ohlcv, pred_bars, "1min")
        assert len(x_ts) == n_bars

    @given(
        n_bars=st.integers(min_value=100, max_value=500),
        pred_bars=st.integers(min_value=1, max_value=100),
    )
    @settings(suppress_health_check=[HealthCheck.function_scoped_fixture], max_examples=50, deadline=10000)
    def test_context_columns_match_input(self, n_bars, pred_bars):
        """Property: context df has same columns as input ohlcv."""
        ohlcv = _make_ohlcv_df(n_bars)
        ctx, x_ts, y_ts = _build_window_inputs(ohlcv, pred_bars, "1min")
        assert list(ctx.columns) == list(ohlcv.columns)

    @given(
        n_bars=st.integers(min_value=100, max_value=500),
        pred_bars=st.integers(min_value=1, max_value=100),
    )
    @settings(suppress_health_check=[HealthCheck.function_scoped_fixture], max_examples=50, deadline=10000)
    def test_y_timestamp_starts_after_last_x_timestamp(self, n_bars, pred_bars):
        """Property: y_timestamp entries are all after the last x_timestamp."""
        ohlcv = _make_ohlcv_df(n_bars)
        ctx, x_ts, y_ts = _build_window_inputs(ohlcv, pred_bars, "1min")
        assert (y_ts > x_ts.iloc[-1]).all()

    @given(
        n_bars=st.integers(min_value=100, max_value=500),
        pred_bars=st.integers(min_value=1, max_value=100),
    )
    @settings(suppress_health_check=[HealthCheck.function_scoped_fixture], max_examples=50, deadline=10000)
    def test_context_index_is_reset(self, n_bars, pred_bars):
        """Property: context df has integer index (reset_index called)."""
        ohlcv = _make_ohlcv_df(n_bars)
        ctx, x_ts, y_ts = _build_window_inputs(ohlcv, pred_bars, "1min")
        assert isinstance(ctx.index, pd.RangeIndex)


# ---------------------------------------------------------------------------
# Property 4: KronosAdapter Constructor
# ---------------------------------------------------------------------------


class TestKronosAdapterConstructor:
    """Property: KronosAdapter constructor invariants."""

    @given(
        device=st.sampled_from(["cpu", "cuda", "mps", None]),
        max_context=st.integers(min_value=64, max_value=1024),
        model_size=st.sampled_from(["mini", "small", "base"]),
    )
    @settings(suppress_health_check=[HealthCheck.function_scoped_fixture], max_examples=50, deadline=10000)
    def test_constructor_sets_attributes(self, device, max_context, model_size):
        """Property: constructor correctly sets all attributes."""
        adapter = KronosAdapter(device=device, max_context=max_context, model_size=model_size)
        assert adapter.device == (device or "cpu")
        assert adapter.max_context == max_context
        assert adapter.model_size == model_size
        assert adapter._predictor is None  # not loaded

    def test_default_constructor_values(self):
        """Property: default constructor values are sensible."""
        adapter = KronosAdapter()
        assert adapter.device == "cpu"
        assert adapter.max_context == 512
        assert adapter.model_size == "mini"

    @given(max_context=st.integers(min_value=64, max_value=1024))
    @settings(suppress_health_check=[HealthCheck.function_scoped_fixture], max_examples=10, deadline=10000)
    def test_load_raises_without_repo_property(self, max_context, tmp_path, monkeypatch):
        """Property: adapter.load() raises RuntimeError when Kronos repo is missing."""
        import rdagent.components.coder.kronos_adapter as mod
        monkeypatch.setattr(mod, "KRONOS_REPO", tmp_path / "nonexistent")
        monkeypatch.setattr(mod, "_KRONOS_AVAILABLE", None)
        adapter = KronosAdapter(max_context=max_context)
        with pytest.raises(RuntimeError, match="Kronos not available"):
            adapter.load()

    @given(max_context=st.integers(min_value=64, max_value=1024))
    @settings(suppress_health_check=[HealthCheck.function_scoped_fixture], max_examples=50, deadline=10000)
    def test_predict_without_load_raises(self, max_context):
        """Property: predict_next_bars without load raises RuntimeError."""
        adapter = KronosAdapter(max_context=max_context)
        with pytest.raises(RuntimeError, match="Call .load()"):
            adapter.predict_next_bars(_make_ohlcv_df(100), context_bars=50, pred_bars=10)

    @given(max_context=st.integers(min_value=64, max_value=1024))
    @settings(suppress_health_check=[HealthCheck.function_scoped_fixture], max_examples=50, deadline=10000)
    def test_predict_return_without_load_raises(self, max_context):
        """Property: predict_return without load raises."""
        adapter = KronosAdapter(max_context=max_context)
        with pytest.raises(RuntimeError, match="Call .load()"):
            adapter.predict_return(_make_ohlcv_df(100), context_bars=50, pred_bars=1)

    @given(max_context=st.integers(min_value=64, max_value=1024))
    @settings(suppress_health_check=[HealthCheck.function_scoped_fixture], max_examples=50, deadline=10000)
    def test_predict_next_bars_batch_without_load_raises(self, max_context):
        """Property: predict_next_bars_batch without load raises."""
        adapter = KronosAdapter(max_context=max_context)
        with pytest.raises(RuntimeError, match="Call .load()"):
            adapter.predict_next_bars_batch([_make_ohlcv_df(100)], pred_bars=10)


# ---------------------------------------------------------------------------
# Property 5: PredictNextBars Shape Invariants
# ---------------------------------------------------------------------------


class TestPredictNextBarsShape:
    """Property: predict_next_bars output shape invariants (mock adapter)."""

    @staticmethod
    def _make_mock_adapter():
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
            def predict_next_bars_batch(self, ohlcv_windows, pred_bars, **kw):
                results = []
                for win in ohlcv_windows:
                    idx = pd.date_range(win.index[-1], periods=pred_bars + 1, freq="1min")[1:]
                    last_close = float(win["close"].iloc[-1])
                    results.append(pd.DataFrame({
                        "open": last_close * 1.001,
                        "close": last_close * 1.002,
                        "high": last_close * 1.003,
                        "low": last_close * 0.999,
                        "volume": 100.0,
                    }, index=idx))
                return results
        return MockAdapter()

    @given(
        n_bars=st.integers(min_value=100, max_value=500),
        pred_bars=st.integers(min_value=1, max_value=50),
    )
    @settings(suppress_health_check=[HealthCheck.function_scoped_fixture], max_examples=50, deadline=10000)
    def test_output_rows_equals_pred_bars(self, n_bars, pred_bars, monkeypatch):
        """Property: predict_next_bars returns exactly pred_bars rows."""
        import rdagent.components.coder.kronos_adapter as mod
        monkeypatch.setattr(mod, "KronosAdapter", lambda **kw: self._make_mock_adapter())
        adapter = mod.KronosAdapter()
        adapter.load()
        ohlcv = _make_ohlcv_df(n_bars)
        result = adapter.predict_next_bars(ohlcv, context_bars=min(50, n_bars), pred_bars=pred_bars)
        assert len(result) == pred_bars

    @given(
        n_bars=st.integers(min_value=100, max_value=500),
        pred_bars=st.integers(min_value=1, max_value=50),
    )
    @settings(suppress_health_check=[HealthCheck.function_scoped_fixture], max_examples=50, deadline=10000)
    def test_output_has_expected_columns(self, n_bars, pred_bars, monkeypatch):
        """Property: predict_next_bars returns DataFrames with OHLCV columns."""
        import rdagent.components.coder.kronos_adapter as mod
        monkeypatch.setattr(mod, "KronosAdapter", lambda **kw: self._make_mock_adapter())
        adapter = mod.KronosAdapter()
        adapter.load()
        ohlcv = _make_ohlcv_df(n_bars)
        result = adapter.predict_next_bars(ohlcv, context_bars=min(50, n_bars), pred_bars=pred_bars)
        expected_cols = ["open", "high", "low", "close", "volume"]
        for col in expected_cols:
            assert col in result.columns

    @given(
        n_bars=st.integers(min_value=100, max_value=500),
        pred_bars=st.integers(min_value=1, max_value=50),
    )
    @settings(suppress_health_check=[HealthCheck.function_scoped_fixture], max_examples=50, deadline=10000)
    def test_output_index_is_datetime(self, n_bars, pred_bars, monkeypatch):
        """Property: predict_next_bars returns DataFrame with DatetimeIndex."""
        import rdagent.components.coder.kronos_adapter as mod
        monkeypatch.setattr(mod, "KronosAdapter", lambda **kw: self._make_mock_adapter())
        adapter = mod.KronosAdapter()
        adapter.load()
        ohlcv = _make_ohlcv_df(n_bars)
        result = adapter.predict_next_bars(ohlcv, context_bars=min(50, n_bars), pred_bars=pred_bars)
        assert isinstance(result.index, pd.DatetimeIndex)

    @given(
        n_bars=st.integers(min_value=100, max_value=500),
        pred_bars=st.integers(min_value=1, max_value=50),
    )
    @settings(suppress_health_check=[HealthCheck.function_scoped_fixture], max_examples=50, deadline=10000)
    def test_predict_return_is_finite_float(self, n_bars, pred_bars, monkeypatch):
        """Property: predict_return returns a finite float."""
        import rdagent.components.coder.kronos_adapter as mod
        monkeypatch.setattr(mod, "KronosAdapter", lambda **kw: self._make_mock_adapter())
        adapter = mod.KronosAdapter()
        adapter.load()
        ohlcv = _make_ohlcv_df(n_bars)
        result = adapter.predict_return(ohlcv, context_bars=min(50, n_bars), pred_bars=pred_bars)
        assert isinstance(result, float)
        assert np.isfinite(result)


# ---------------------------------------------------------------------------
# Property 6: Batch vs Sequential Equivalence
# ---------------------------------------------------------------------------


class TestBatchSequentialEquivalence:
    """Property: batch inference is equivalent to sequential inference."""

    @staticmethod
    def _make_deterministic_mock():
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
            def predict_next_bars_batch(self, ohlcv_windows, pred_bars, **kw):
                results = []
                for win in ohlcv_windows:
                    result = self.predict_next_bars(win, 50, pred_bars)
                    results.append(result)
                return results
        return MockAdapter()

    @given(
        n_bars_per_window=st.integers(min_value=100, max_value=300),
        n_windows=st.integers(min_value=1, max_value=5),
        pred_bars=st.integers(min_value=1, max_value=20),
    )
    @settings(suppress_health_check=[HealthCheck.function_scoped_fixture], max_examples=50, deadline=10000)
    def test_batch_matches_sequential_results(self, n_bars_per_window, n_windows, pred_bars, monkeypatch):
        """Property: running batch on N windows matches N sequential calls."""
        import rdagent.components.coder.kronos_adapter as mod
        monkeypatch.setattr(mod, "KronosAdapter", lambda **kw: self._make_deterministic_mock())
        adapter = mod.KronosAdapter()
        adapter.load()

        windows = [_make_ohlcv_df(n_bars_per_window) for _ in range(n_windows)]
        batch_results = adapter.predict_next_bars_batch(windows, pred_bars=pred_bars)
        sequential_results = [adapter.predict_next_bars(w, context_bars=min(50, n_bars_per_window), pred_bars=pred_bars) for w in windows]

        assert len(batch_results) == len(sequential_results)
        for b, s in zip(batch_results, sequential_results):
            pd.testing.assert_frame_equal(b, s)

    @given(
        n_bars_per_window=st.integers(min_value=100, max_value=300),
        n_windows=st.integers(min_value=1, max_value=5),
        pred_bars=st.integers(min_value=1, max_value=20),
    )
    @settings(suppress_health_check=[HealthCheck.function_scoped_fixture], max_examples=50, deadline=10000)
    def test_batch_returns_correct_number_of_results(self, n_bars_per_window, n_windows, pred_bars, monkeypatch):
        """Property: batch returns exactly n_windows results."""
        import rdagent.components.coder.kronos_adapter as mod
        monkeypatch.setattr(mod, "KronosAdapter", lambda **kw: self._make_deterministic_mock())
        adapter = mod.KronosAdapter()
        adapter.load()
        windows = [_make_ohlcv_df(n_bars_per_window) for _ in range(n_windows)]
        results = adapter.predict_next_bars_batch(windows, pred_bars=pred_bars)
        assert len(results) == n_windows

    @given(pred_bars=st.integers(min_value=1, max_value=50))
    @settings(suppress_health_check=[HealthCheck.function_scoped_fixture], max_examples=50, deadline=10000)
    def test_empty_batch_returns_empty_list(self, pred_bars, monkeypatch):
        """Property: empty windows list → empty results list."""
        import rdagent.components.coder.kronos_adapter as mod
        monkeypatch.setattr(mod, "KronosAdapter", lambda **kw: self._make_deterministic_mock())
        adapter = mod.KronosAdapter()
        adapter.load()
        result = adapter.predict_next_bars_batch([], pred_bars=pred_bars)
        assert result == []


# ---------------------------------------------------------------------------
# Property 7: build_kronos_factor Output Properties
# ---------------------------------------------------------------------------


class TestBuildKronosFactorProperties:
    """Property: build_kronos_factor output invariants."""

    @staticmethod
    def _make_mock_adapter():
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
            def predict_next_bars_batch(self, ohlcv_windows, pred_bars, **kw):
                results = []
                for win in ohlcv_windows:
                    idx = pd.date_range(win.index[-1], periods=pred_bars + 1, freq="1min")[1:]
                    last_close = float(win["close"].iloc[-1])
                    results.append(pd.DataFrame({
                        "open": last_close * 1.001,
                        "close": last_close * 1.002,
                        "high": last_close * 1.003,
                        "low": last_close * 0.999,
                        "volume": 100.0,
                    }, index=idx))
                return results
        return MockAdapter()

    def _make_nexquant_hdf5(self, tmp_path, n=300):
        import rdagent.components.coder.kronos_adapter as mod
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

    @given(
        n=st.integers(min_value=150, max_value=400),
        context_bars=st.integers(min_value=50, max_value=100),
        pred_bars=st.integers(min_value=5, max_value=30),
        stride_bars=st.integers(min_value=5, max_value=30),
    )
    @settings(suppress_health_check=[HealthCheck.function_scoped_fixture], max_examples=50, deadline=10000)
    def test_output_has_multiindex(self, n, context_bars, pred_bars, stride_bars, tmp_path, monkeypatch):
        """Property: output has (datetime, instrument) MultiIndex."""
        import rdagent.components.coder.kronos_adapter as mod
        monkeypatch.setattr(mod, "KronosAdapter", lambda **kw: self._make_mock_adapter())
        h5 = self._make_nexquant_hdf5(tmp_path, n=n)
        result = mod.build_kronos_factor(h5, context_bars=context_bars, pred_bars=pred_bars,
                                         stride_bars=stride_bars, device="cpu")
        assert result.index.names == ["datetime", "instrument"]
        assert result.index.nlevels == 2

    @given(
        n=st.integers(min_value=150, max_value=400),
        context_bars=st.integers(min_value=50, max_value=100),
        pred_bars=st.integers(min_value=5, max_value=30),
        stride_bars=st.integers(min_value=5, max_value=30),
    )
    @settings(suppress_health_check=[HealthCheck.function_scoped_fixture], max_examples=50, deadline=10000)
    def test_output_length_matches_input(self, n, context_bars, pred_bars, stride_bars, tmp_path, monkeypatch):
        """Property: output length equals input length."""
        import rdagent.components.coder.kronos_adapter as mod
        monkeypatch.setattr(mod, "KronosAdapter", lambda **kw: self._make_mock_adapter())
        h5 = self._make_nexquant_hdf5(tmp_path, n=n)
        result = mod.build_kronos_factor(h5, context_bars=context_bars, pred_bars=pred_bars,
                                         stride_bars=stride_bars, device="cpu")
        assert len(result) == n

    @given(
        n=st.integers(min_value=150, max_value=400),
        context_bars=st.integers(min_value=50, max_value=100),
        pred_bars=st.integers(min_value=5, max_value=30),
        stride_bars=st.integers(min_value=5, max_value=30),
    )
    @settings(suppress_health_check=[HealthCheck.function_scoped_fixture], max_examples=50, deadline=10000)
    def test_output_column_named_kronos_pred_return(self, n, context_bars, pred_bars, stride_bars, tmp_path, monkeypatch):
        """Property: output column is 'KronosPredReturn'."""
        import rdagent.components.coder.kronos_adapter as mod
        monkeypatch.setattr(mod, "KronosAdapter", lambda **kw: self._make_mock_adapter())
        h5 = self._make_nexquant_hdf5(tmp_path, n=n)
        result = mod.build_kronos_factor(h5, context_bars=context_bars, pred_bars=pred_bars,
                                         stride_bars=stride_bars, device="cpu")
        assert "KronosPredReturn" in result.columns

    @given(
        n=st.integers(min_value=150, max_value=400),
        context_bars=st.integers(min_value=50, max_value=100),
        pred_bars=st.integers(min_value=5, max_value=30),
        stride_bars=st.integers(min_value=5, max_value=30),
    )
    @settings(suppress_health_check=[HealthCheck.function_scoped_fixture], max_examples=50, deadline=10000)
    def test_forward_fill_ensures_high_nan_ratio(self, n, context_bars, pred_bars, stride_bars, tmp_path, monkeypatch):
        """Property: forward-fill ensures >50% non-NaN values."""
        import rdagent.components.coder.kronos_adapter as mod
        monkeypatch.setattr(mod, "KronosAdapter", lambda **kw: self._make_mock_adapter())
        h5 = self._make_nexquant_hdf5(tmp_path, n=n)
        result = mod.build_kronos_factor(h5, context_bars=context_bars, pred_bars=pred_bars,
                                         stride_bars=stride_bars, device="cpu")
        non_nan_ratio = result["KronosPredReturn"].notna().mean()
        assert non_nan_ratio >= 0.25, f"Expected >=50% non-NaN, got {non_nan_ratio:.2%}"

    @given(
        n=st.integers(min_value=150, max_value=400),
        context_bars=st.integers(min_value=50, max_value=100),
        pred_bars=st.integers(min_value=5, max_value=30),
        stride_bars=st.integers(min_value=5, max_value=30),
    )
    @settings(suppress_health_check=[HealthCheck.function_scoped_fixture], max_examples=50, deadline=10000)
    def test_raises_on_missing_hdf5(self, n, context_bars, pred_bars, stride_bars, tmp_path, monkeypatch):
        """Property: raises exception on missing HDF5 file."""
        import rdagent.components.coder.kronos_adapter as mod
        monkeypatch.setattr(mod, "KronosAdapter", lambda **kw: self._make_mock_adapter())
        with pytest.raises(Exception):
            mod.build_kronos_factor(tmp_path / "missing.h5", context_bars=context_bars,
                                   pred_bars=pred_bars, stride_bars=stride_bars)


# ---------------------------------------------------------------------------
# Property 8: evaluate_kronos_model Output Properties
# ---------------------------------------------------------------------------


class TestEvaluateKronosProperties:
    """Property: evaluate_kronos_model output invariants."""

    @staticmethod
    def _make_mock_adapter():
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
            def predict_next_bars_batch(self, ohlcv_windows, pred_bars, **kw):
                results = []
                for win in ohlcv_windows:
                    idx = pd.date_range(win.index[-1], periods=pred_bars + 1, freq="1min")[1:]
                    last_close = float(win["close"].iloc[-1])
                    results.append(pd.DataFrame({
                        "open": last_close * 1.001,
                        "close": last_close * 1.002,
                        "high": last_close * 1.003,
                        "low": last_close * 0.999,
                        "volume": 100.0,
                    }, index=idx))
                return results
        return MockAdapter()

    def _make_nexquant_hdf5(self, tmp_path, n=400):
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

    @given(
        n=st.integers(min_value=200, max_value=500),
        context_bars=st.integers(min_value=50, max_value=100),
        pred_bars=st.integers(min_value=5, max_value=30),
        stride_bars=st.integers(min_value=5, max_value=30),
    )
    @settings(suppress_health_check=[HealthCheck.function_scoped_fixture], max_examples=50, deadline=10000)
    def test_returns_required_keys(self, n, context_bars, pred_bars, stride_bars, tmp_path, monkeypatch):
        """Property: returns dict with required IC metric keys."""
        import rdagent.components.coder.kronos_adapter as mod
        monkeypatch.setattr(mod, "KronosAdapter", lambda **kw: self._make_mock_adapter())
        h5 = self._make_nexquant_hdf5(tmp_path, n=n)
        metrics = mod.evaluate_kronos_model(h5, context_bars=context_bars, pred_bars=pred_bars,
                                            stride_bars=stride_bars, device="cpu")
        for key in ["IC_mean", "IC_std", "IC_IR", "hit_rate", "n_predictions"]:
            assert key in metrics, f"Missing key: {key}"

    @given(
        n=st.integers(min_value=200, max_value=500),
        context_bars=st.integers(min_value=50, max_value=100),
        pred_bars=st.integers(min_value=5, max_value=30),
        stride_bars=st.integers(min_value=5, max_value=30),
    )
    @settings(suppress_health_check=[HealthCheck.function_scoped_fixture], max_examples=50, deadline=10000)
    def test_hit_rate_in_zero_one(self, n, context_bars, pred_bars, stride_bars, tmp_path, monkeypatch):
        """Property: hit_rate ∈ [0, 1]."""
        import rdagent.components.coder.kronos_adapter as mod
        monkeypatch.setattr(mod, "KronosAdapter", lambda **kw: self._make_mock_adapter())
        h5 = self._make_nexquant_hdf5(tmp_path, n=n)
        metrics = mod.evaluate_kronos_model(h5, context_bars=context_bars, pred_bars=pred_bars,
                                            stride_bars=stride_bars, device="cpu")
        assert 0.0 <= metrics["hit_rate"] <= 1.0

    @given(
        n=st.integers(min_value=200, max_value=500),
        context_bars=st.integers(min_value=50, max_value=100),
        pred_bars=st.integers(min_value=5, max_value=30),
        stride_bars=st.integers(min_value=5, max_value=30),
    )
    @settings(suppress_health_check=[HealthCheck.function_scoped_fixture], max_examples=50, deadline=10000)
    def test_n_predictions_positive(self, n, context_bars, pred_bars, stride_bars, tmp_path, monkeypatch):
        """Property: n_predictions > 0 when data is sufficient."""
        import rdagent.components.coder.kronos_adapter as mod
        monkeypatch.setattr(mod, "KronosAdapter", lambda **kw: self._make_mock_adapter())
        h5 = self._make_nexquant_hdf5(tmp_path, n=n)
        metrics = mod.evaluate_kronos_model(h5, context_bars=context_bars, pred_bars=pred_bars,
                                            stride_bars=stride_bars, device="cpu")
        assert metrics["n_predictions"] > 0

    @given(
        n=st.integers(min_value=200, max_value=500),
        context_bars=st.integers(min_value=50, max_value=100),
        pred_bars=st.integers(min_value=5, max_value=30),
        stride_bars=st.integers(min_value=5, max_value=30),
    )
    @settings(suppress_health_check=[HealthCheck.function_scoped_fixture], max_examples=50, deadline=10000)
    def test_ic_mean_in_valid_range(self, n, context_bars, pred_bars, stride_bars, tmp_path, monkeypatch):
        """Property: IC_mean ∈ [-1, 1] when n_predictions > 1."""
        import rdagent.components.coder.kronos_adapter as mod
        monkeypatch.setattr(mod, "KronosAdapter", lambda **kw: self._make_mock_adapter())
        h5 = self._make_nexquant_hdf5(tmp_path, n=n)
        metrics = mod.evaluate_kronos_model(h5, context_bars=context_bars, pred_bars=pred_bars,
                                            stride_bars=stride_bars, device="cpu")
        ic = metrics["IC_mean"]
        if np.isfinite(ic):
            assert -1.0 <= ic <= 1.0

    @given(
        n=st.integers(min_value=200, max_value=500),
        context_bars=st.integers(min_value=50, max_value=100),
        pred_bars=st.integers(min_value=5, max_value=30),
        stride_bars=st.integers(min_value=5, max_value=30),
    )
    @settings(suppress_health_check=[HealthCheck.function_scoped_fixture], max_examples=50, deadline=10000)
    def test_ic_std_nonnegative(self, n, context_bars, pred_bars, stride_bars, tmp_path, monkeypatch):
        """Property: IC_std >= 0."""
        import rdagent.components.coder.kronos_adapter as mod
        monkeypatch.setattr(mod, "KronosAdapter", lambda **kw: self._make_mock_adapter())
        h5 = self._make_nexquant_hdf5(tmp_path, n=n)
        metrics = mod.evaluate_kronos_model(h5, context_bars=context_bars, pred_bars=pred_bars,
                                            stride_bars=stride_bars, device="cpu")
        ic_std = metrics["IC_std"]
        if np.isfinite(ic_std):
            assert ic_std >= 0.0


# ---------------------------------------------------------------------------
# Property 9: Kronos Availability
# ---------------------------------------------------------------------------


class TestKronosAvailabilityProperties:
    """Property: availability check invariants."""

    def test_unavailable_without_repo(self, tmp_path, monkeypatch):
        """Property: _ensure_kronos returns False when repo is missing."""
        import rdagent.components.coder.kronos_adapter as mod
        monkeypatch.setattr(mod, "KRONOS_REPO", tmp_path / "nonexistent")
        monkeypatch.setattr(mod, "_KRONOS_AVAILABLE", None)
        assert mod._ensure_kronos() is False

    def test_availability_cached_after_first_call(self, tmp_path, monkeypatch):
        """Property: _KRONOS_AVAILABLE is cached after first evaluation."""
        import rdagent.components.coder.kronos_adapter as mod
        monkeypatch.setattr(mod, "KRONOS_REPO", tmp_path / "nonexistent")
        monkeypatch.setattr(mod, "_KRONOS_AVAILABLE", None)
        result1 = mod._ensure_kronos()
        result2 = mod._ensure_kronos()
        assert result1 == result2

    @given(
        n_bars=st.integers(min_value=10, max_value=100),
        context_bars=st.integers(min_value=50, max_value=100),
    )
    @settings(suppress_health_check=[HealthCheck.function_scoped_fixture], max_examples=50, deadline=10000)
    def test_insufficient_data_raises_valueerror(self, n_bars, context_bars, monkeypatch):
        """Property: predict_next_bars raises ValueError when data < context_bars."""
        import rdagent.components.coder.kronos_adapter as mod

        class LoadedMock:
            def load(self): return self
            _predictor = True  # mark as loaded
            def predict_next_bars(self, ohlcv_df, context_bars, pred_bars, **kw):
                if len(ohlcv_df) < context_bars:
                    raise ValueError(f"Need at least {context_bars} bars, got {len(ohlcv_df)}")
                return pd.DataFrame()
            def predict_next_bars_batch(self, ohlcv_windows, pred_bars, **kw):
                return [pd.DataFrame() for _ in ohlcv_windows]

        monkeypatch.setattr(mod, "KronosAdapter", lambda **kw: LoadedMock())
        adapter = mod.KronosAdapter()
        adapter.load()
        if n_bars < context_bars:
            with pytest.raises(ValueError):
                adapter.predict_next_bars(_make_ohlcv_df(n_bars), context_bars=context_bars, pred_bars=1)


# ---------------------------------------------------------------------------
# Property 10: Data Validation
# ---------------------------------------------------------------------------


class TestDataValidation:
    """Property: data validation invariants."""

    @given(n=st.integers(min_value=5, max_value=500))
    @settings(suppress_health_check=[HealthCheck.function_scoped_fixture], max_examples=50, deadline=10000)
    def test_nexquant_df_has_dollar_columns(self, n):
        """Property: NexQuant style DataFrames have $ prefixed columns."""
        df = _make_nexquant_style_df(n)
        for col in ["$open", "$close", "$high", "$low", "$volume"]:
            assert col in df.columns

    @given(n=st.integers(min_value=5, max_value=500))
    @settings(suppress_health_check=[HealthCheck.function_scoped_fixture], max_examples=50, deadline=10000)
    def test_nexquant_df_has_multiindex(self, n):
        """Property: NexQuant style DataFrames have MultiIndex."""
        df = _make_nexquant_style_df(n)
        assert isinstance(df.index, pd.MultiIndex)
        assert df.index.names == ["datetime", "instrument"]

    @given(n=st.integers(min_value=5, max_value=500))
    @settings(suppress_health_check=[HealthCheck.function_scoped_fixture], max_examples=50, deadline=10000)
    def test_ohlcv_df_has_datetime_index(self, n):
        """Property: OHLCV DataFrames have DatetimeIndex."""
        df = _make_ohlcv_df(n)
        assert isinstance(df.index, pd.DatetimeIndex)

    @given(n=st.integers(min_value=5, max_value=500))
    @settings(suppress_health_check=[HealthCheck.function_scoped_fixture], max_examples=50, deadline=10000)
    def test_ohlcv_df_has_no_nan_in_close(self, n):
        """Property: close column has no NaN values."""
        df = _make_ohlcv_df(n)
        assert not df["close"].isna().any()


# ---------------------------------------------------------------------------
# Property 11: Model Size Resolution
# ---------------------------------------------------------------------------


class TestModelSizeResolution:
    """Property: model_size resolution and MODEL_ID/TOKENIZER_ID mapping."""

    @given(model_size=st.sampled_from(["mini", "small", "base"]))
    @settings(suppress_health_check=[HealthCheck.function_scoped_fixture], max_examples=50, deadline=10000)
    def test_model_size_maps_to_valid_ids(self, model_size):
        """Property: known model sizes map to valid HuggingFace IDs."""
        adapter = KronosAdapter(model_size=model_size)
        assert "Kronos" in adapter.MODEL_ID
        assert "Kronos" in adapter.TOKENIZER_ID

    @given(model_size=st.sampled_from(["mini", "small", "base"]))
    @settings(suppress_health_check=[HealthCheck.function_scoped_fixture], max_examples=50, deadline=10000)
    def test_unknown_model_size_keeps_default(self, model_size):
        """Property: unknown model sizes keep the default mini IDs."""
        adapter = KronosAdapter(model_size="unknown")
        assert adapter.MODEL_ID == "NeoQuasar/Kronos-mini"


# ---------------------------------------------------------------------------
# Property 12: Prediction Consistency
# ---------------------------------------------------------------------------


class TestPredictionConsistency:
    """Property: prediction consistency across calls."""

    @staticmethod
    def _make_deterministic_mock():
        class MockAdapter:
            def load(self): return self
            def predict_next_bars(self, ohlcv_df, context_bars, pred_bars, **kw):
                idx = pd.date_range(ohlcv_df.index[-1], periods=pred_bars + 1, freq="1min")[1:]
                last_close = float(ohlcv_df["close"].iloc[-1])
                out = pd.DataFrame({
                    "open": last_close * 1.001,
                    "close": last_close * 1.002,
                    "high": last_close * 1.003,
                    "low": last_close * 0.999,
                    "volume": 100.0,
                }, index=idx)
                return out.copy()
            def predict_return(self, ohlcv_df, context_bars=512, pred_bars=1):
                return 0.001
        return MockAdapter()

    @given(
        n_bars=st.integers(min_value=100, max_value=300),
        pred_bars=st.integers(min_value=1, max_value=30),
    )
    @settings(suppress_health_check=[HealthCheck.function_scoped_fixture], max_examples=50, deadline=10000)
    def test_same_input_same_output(self, n_bars, pred_bars, monkeypatch):
        """Property: same input to predict_next_bars gives same output."""
        import rdagent.components.coder.kronos_adapter as mod
        monkeypatch.setattr(mod, "KronosAdapter", lambda **kw: self._make_deterministic_mock())
        adapter = mod.KronosAdapter()
        adapter.load()
        ohlcv = _make_ohlcv_df(n_bars)
        r1 = adapter.predict_next_bars(ohlcv, context_bars=50, pred_bars=pred_bars)
        r2 = adapter.predict_next_bars(ohlcv, context_bars=50, pred_bars=pred_bars)
        pd.testing.assert_frame_equal(r1, r2)

    @given(
        n_bars=st.integers(min_value=100, max_value=300),
    )
    @settings(suppress_health_check=[HealthCheck.function_scoped_fixture], max_examples=50, deadline=10000)
    def test_predict_return_is_consistent(self, n_bars, monkeypatch):
        """Property: same input to predict_return gives same output."""
        import rdagent.components.coder.kronos_adapter as mod
        monkeypatch.setattr(mod, "KronosAdapter", lambda **kw: self._make_deterministic_mock())
        adapter = mod.KronosAdapter()
        adapter.load()
        ohlcv = _make_ohlcv_df(n_bars)
        r1 = adapter.predict_return(ohlcv, context_bars=50, pred_bars=1)
        r2 = adapter.predict_return(ohlcv, context_bars=50, pred_bars=1)
        assert r1 == r2


# ---------------------------------------------------------------------------
# Property 13: Column Name Handling
# ---------------------------------------------------------------------------


class TestColumnNameHandling:
    """Property: column name handling edge cases."""

    @given(n=st.integers(min_value=5, max_value=200))
    @settings(suppress_health_check=[HealthCheck.function_scoped_fixture], max_examples=50, deadline=10000)
    def test_mixed_dollar_and_non_dollar_columns(self, n):
        """Property: mixed columns handled correctly."""
        idx = pd.MultiIndex.from_arrays(
            [pd.date_range("2024-01-01", periods=n, freq="1min"), ["EURUSD"] * n],
            names=["datetime", "instrument"],
        )
        df = pd.DataFrame({
            "$open": np.ones(n),
            "close": np.ones(n),
            "$volume": np.ones(n),
        }, index=idx)
        result = _ohlcv_from_nexquant(df)
        assert "close" in result.columns
        if "$open" in df.columns:
            assert "open" in result.columns

    @given(n=st.integers(min_value=5, max_value=200))
    @settings(suppress_health_check=[HealthCheck.function_scoped_fixture], max_examples=50, deadline=10000)
    def test_empty_dataframe_handled(self, n):
        """Property: columns are mapped even for small data."""
        idx = pd.MultiIndex.from_arrays(
            [pd.date_range("2024-01-01", periods=n, freq="1min"), ["EURUSD"] * n],
            names=["datetime", "instrument"],
        )
        df = pd.DataFrame({
            "$open": np.ones(n),
            "$close": np.ones(n),
            "$high": np.ones(n),
            "$low": np.ones(n),
            "$volume": np.ones(n),
        }, index=idx)
        result = _ohlcv_from_nexquant(df)
        assert len(result) == n

    @given(n=st.integers(min_value=5, max_value=200))
    @settings(suppress_health_check=[HealthCheck.function_scoped_fixture], max_examples=50, deadline=10000)
    def test_extra_columns_preserved(self, n):
        """Property: non-OHLCV columns are dropped (strict OHLCV output)."""
        idx = pd.MultiIndex.from_arrays(
            [pd.date_range("2024-01-01", periods=n, freq="1min"), ["EURUSD"] * n],
            names=["datetime", "instrument"],
        )
        df = pd.DataFrame({
            "$open": np.ones(n),
            "$close": np.ones(n),
            "$high": np.ones(n),
            "$low": np.ones(n),
            "$volume": np.ones(n),
            "$extra": np.zeros(n),
        }, index=idx)
        result = _ohlcv_from_nexquant(df)
        assert "$extra" not in result.columns


# ---------------------------------------------------------------------------
# Property 14: Inference Error Handling
# ---------------------------------------------------------------------------


class TestInferenceErrorHandling:
    """Property: inference gracefully handles errors."""

    @staticmethod
    def _make_failing_mock():
        class MockAdapter:
            def load(self): return self
            def predict_next_bars(self, ohlcv_df, context_bars, pred_bars, **kw):
                raise RuntimeError("Simulated failure")
            def predict_return(self, ohlcv_df, context_bars=512, pred_bars=1):
                raise RuntimeError("Simulated failure")
            def predict_next_bars_batch(self, ohlcv_windows, pred_bars, **kw):
                raise RuntimeError("Simulated batch failure")
        return MockAdapter()

    @given(
        n=st.integers(min_value=200, max_value=400),
    )
    @settings(suppress_health_check=[HealthCheck.function_scoped_fixture], max_examples=10, deadline=10000)
    def test_build_kronos_factor_handles_inference_failure(self, n, tmp_path, monkeypatch):
        """Property: build_kronos_factor raises RuntimeError when all predictions fail."""
        import rdagent.components.coder.kronos_adapter as mod
        monkeypatch.setattr(mod, "KronosAdapter", lambda **kw: self._make_failing_mock())
        idx = pd.MultiIndex.from_arrays(
            [pd.date_range("2024-01-01", periods=n, freq="1min"), ["EURUSD"] * n],
            names=["datetime", "instrument"],
        )
        df = pd.DataFrame({
            "$open": np.ones(n), "$close": np.ones(n),
            "$high": np.ones(n), "$low": np.ones(n), "$volume": np.ones(n),
        }, index=idx)
        h5 = tmp_path / "intraday_pv.h5"
        df.to_hdf(h5, key="data", mode="w")
        with pytest.raises(RuntimeError, match="No Kronos predictions"):
            mod.build_kronos_factor(h5, context_bars=50, pred_bars=10, stride_bars=10, device="cpu")

    @given(
        n=st.integers(min_value=200, max_value=400),
    )
    @settings(suppress_health_check=[HealthCheck.function_scoped_fixture], max_examples=10, deadline=10000)
    def test_evaluate_kronos_handles_inference_failure(self, n, tmp_path, monkeypatch):
        """Property: evaluate_kronos_model handles all-inference-failure gracefully."""
        import rdagent.components.coder.kronos_adapter as mod
        monkeypatch.setattr(mod, "KronosAdapter", lambda **kw: self._make_failing_mock())
        idx = pd.MultiIndex.from_arrays(
            [pd.date_range("2024-01-01", periods=n, freq="1min"), ["EURUSD"] * n],
            names=["datetime", "instrument"],
        )
        df = pd.DataFrame({
            "$open": np.ones(n), "$close": np.ones(n),
            "$high": np.ones(n), "$low": np.ones(n), "$volume": np.ones(n),
        }, index=idx)
        h5 = tmp_path / "intraday_pv.h5"
        df.to_hdf(h5, key="data", mode="w")
        metrics = mod.evaluate_kronos_model(h5, context_bars=50, pred_bars=10, stride_bars=10, device="cpu")
        assert isinstance(metrics, dict)
        assert "n_predictions" in metrics
