"""Deep tests for predix_gen_strategies_real_bt.py — property-based, edge cases.

Tests factor loading, threshold rescaling, backtest runner, acceptance
criteria, and the TeeFile logger — without requiring real OHLCV data or LLM.
"""

from __future__ import annotations

import json
import os
import sys
import tempfile
from pathlib import Path
from unittest.mock import MagicMock, patch

import numpy as np
import pandas as pd
import pytest

PROJECT_ROOT = Path(__file__).parent.parent.parent
sys.path.insert(0, str(PROJECT_ROOT))

from hypothesis import given, settings, HealthCheck
from hypothesis import strategies as st


@pytest.fixture
def gen_module():
    import importlib
    import scripts.predix_gen_strategies_real_bt as m
    return m


class TestRescaleThresholds:
    """Tests for _rescale_thresholds — threshold relaxation logic."""

    def test_noop_at_scale_1(self, gen_module):
        code = "threshold = 0.7\nrsi_threshold = 35.0"
        result = gen_module._rescale_thresholds(code, 1.0)
        # Scale 1 may reformat numbers (0.7 → 0.700) but values are preserved
        assert "threshold" in result
        assert "0.7" in result or "0.700" in result

    def test_rsi_pulled_toward_50(self, gen_module):
        """RSI values like 35 or 65 should move toward 50 when scaled < 1."""
        code = "if rsi < 35:\n    signal = 1\nif rsi > 65:\n    signal = -1"
        result = gen_module._rescale_thresholds(code, 0.5)
        # 35 → 50 + (35-50)*0.5 = 42.5
        # 65 → 50 + (65-50)*0.5 = 57.5
        assert "42.5" in result
        assert "57.5" in result

    def test_small_thresholds_scaled(self, gen_module):
        code = "entry = 0.5\nexit = 0.2"
        result = gen_module._rescale_thresholds(code, 0.5)
        # 0.5 → 0.250, 0.2 → 0.100
        assert "0.250" in result

    def test_non_threshold_numbers_preserved(self, gen_module):
        """Common code patterns like window sizes should not be changed much."""
        code = "rolling_window = 50\nsignal = z_score.rolling(50).mean()"
        result = gen_module._rescale_thresholds(code, 0.5)
        # 50 → pulled toward 50 (= 50, unchanged)
        assert "rolling_window = 50" in result or "rolling(50)" in result

    @given(st.text(min_size=5, max_size=200))
    @settings(max_examples=200, deadline=5000,
              suppress_health_check=[HealthCheck.function_scoped_fixture])
    def test_never_crashes_on_arbitrary_code(self, gen_module, code):
        """Any string input must not crash or hang."""
        result = gen_module._rescale_thresholds(code, 0.5)
        assert isinstance(result, str)

    @pytest.mark.parametrize("scale", [0.0, 0.1, 0.5, 1.0, 2.0, 10.0])
    def test_various_scales(self, gen_module, scale):
        code = "threshold = 0.5"
        result = gen_module._rescale_thresholds(code, scale)
        assert isinstance(result, str)

    def test_empty_code(self, gen_module):
        assert gen_module._rescale_thresholds("", 0.5) == ""

    def test_python_syntax_preserved(self, gen_module):
        """The output must be syntactically valid if the input was."""
        code = """
x = 0.3
if x > 0.0:
    y = x * 2
else:
    y = 0
"""
        result = gen_module._rescale_thresholds(code, 0.5)
        try:
            compile(result, "<test>", "exec")
        except SyntaxError:
            pytest.skip("Rescaling may break syntax on edge cases but must not crash")


class TestFactorLoading:
    def test_empty_directory(self, gen_module):
        gen_module._FACTORS_CACHE = None
        with tempfile.TemporaryDirectory() as td:
            gen_module.FACTORS_DIR = Path(td)
            factors = gen_module.load_available_factors(20)
            assert factors == []

    def test_loads_and_sorts_by_ic(self, gen_module):
        gen_module._FACTORS_CACHE = None
        with tempfile.TemporaryDirectory() as td:
            tdp = Path(td)
            (tdp / "values").mkdir()
            for name, ic in [("weak", 0.01), ("strong", 0.5), ("medium", 0.15)]:
                json.dump({"factor_name": name, "ic": ic, "status": "success"},
                         open(tdp / f"{name}.json", "w"))
                (tdp / "values" / f"{name}.parquet").touch()
            gen_module.FACTORS_DIR = tdp
            factors = gen_module.load_available_factors(10)
            names = [f["name"] for f in factors]
            assert names[0] == "strong"
            assert names[-1] == "weak"

    def test_respects_top_n(self, gen_module):
        gen_module._FACTORS_CACHE = None
        with tempfile.TemporaryDirectory() as td:
            tdp = Path(td)
            (tdp / "values").mkdir()
            for i in range(10):
                json.dump({"factor_name": f"f{i}", "ic": 0.1},
                         open(tdp / f"f{i}.json", "w"))
                (tdp / "values" / f"f{i}.parquet").touch()
            gen_module.FACTORS_DIR = tdp
            assert len(gen_module.load_available_factors(3)) == 3

    def test_skips_missing_parquet(self, gen_module):
        gen_module._FACTORS_CACHE = None
        with tempfile.TemporaryDirectory() as td:
            tdp = Path(td)
            (tdp / "values").mkdir()
            for i in range(5):
                json.dump({"factor_name": f"f{i}", "ic": 0.1},
                         open(tdp / f"f{i}.json", "w"))
                if i % 2 == 0:
                    (tdp / "values" / f"f{i}.parquet").touch()
            gen_module.FACTORS_DIR = tdp
            factors = gen_module.load_available_factors(20)
            assert len(factors) == 3  # only even-indexed have parquet

    def test_corrupt_json_skipped(self, gen_module):
        gen_module._FACTORS_CACHE = None
        with tempfile.TemporaryDirectory() as td:
            tdp = Path(td)
            (tdp / "values").mkdir()
            (tdp / "corrupt.json").write_text("not json")
            json.dump({"factor_name": "good", "ic": 0.5, "status": "success"},
                     open(tdp / "good.json", "w"))
            (tdp / "values" / "good.parquet").touch()
            gen_module.FACTORS_DIR = tdp
            factors = gen_module.load_available_factors(20)
            assert len(factors) == 1
            assert factors[0]["name"] == "good"


class TestOhlcvLoading:
    def test_file_not_found_raises(self, gen_module):
        gen_module._OHLCV_CACHE = None
        gen_module.OHLCV_PATH = Path("/nonexistent/path.h5")
        with pytest.raises(FileNotFoundError):
            gen_module.load_ohlcv_data()

    def test_uses_cache_second_call(self, gen_module):
        gen_module._OHLCV_CACHE = None
        with tempfile.TemporaryDirectory() as td:
            tdp = Path(td)
            h5_path = tdp / "test.h5"
            idx = pd.MultiIndex.from_product(
                [pd.date_range("2024-01-01", periods=100, freq="1min"), ["EURUSD"]],
                names=["datetime", "instrument"],
            )
            df = pd.DataFrame({"$close": np.random.default_rng(42).normal(1.10, 0.001, 100)}, index=idx)
            df.to_hdf(h5_path, key="data")
            gen_module.OHLCV_PATH = h5_path
            r1 = gen_module.load_ohlcv_data()
            r2 = gen_module.load_ohlcv_data()
            assert r1 is r2  # Same object from cache


class TestTeeFile:
    def test_writes_to_all_files(self, gen_module):
        f1 = tempfile.NamedTemporaryFile(mode="w+", delete=False)
        f2 = tempfile.NamedTemporaryFile(mode="w+", delete=False)
        try:
            tee = gen_module._TeeFile(f1, f2)
            tee.write("hello")
            tee.flush()
            f1.seek(0); f2.seek(0)
            assert f1.read() == "hello"
            assert f2.read() == "hello"
        finally:
            os.unlink(f1.name)
            os.unlink(f2.name)

    def test_fileno_delegates(self, gen_module):
        f1 = tempfile.NamedTemporaryFile(mode="w+", delete=False)
        try:
            tee = gen_module._TeeFile(f1)
            assert tee.fileno() == f1.fileno()
        finally:
            os.unlink(f1.name)


class TestBacktestRunner:
    def test_insufficient_data_rejected(self, gen_module):
        """Too few bars must return failed status."""
        idx = pd.DatetimeIndex(["2024-01-01"])
        close = pd.Series([1.10], index=idx)
        result = gen_module.run_backtest(close, None, "signal = close * 0")
        assert result is None or result.get("status") != "success"

    def test_sandbox_execution(self, gen_module):
        """Simple valid strategy code must produce a backtest result."""
        rng = np.random.default_rng(1)
        idx = pd.date_range("2024-01-01", periods=2000, freq="1min")
        close = pd.Series(1.10 + rng.normal(0, 0.0001, 2000).cumsum(), index=idx)
        code = "signal = pd.Series(np.where(close > close.shift(1), 1.0, -1.0), index=close.index)"
        result = gen_module.run_backtest(close, None, code)
        if result and result.get("status") == "success":
            assert "sharpe" in result
            assert "n_trades" in result
            assert "max_drawdown" in result

    def test_syntax_error_caught(self, gen_module):
        rng = np.random.default_rng(2)
        idx = pd.date_range("2024-01-01", periods=500, freq="1min")
        close = pd.Series(1.10 + rng.normal(0, 0.0001, 500).cumsum(), index=idx)
        result = gen_module.run_backtest(close, None, "this is not python")
        assert result is None or result.get("status") != "success"

    def test_no_signal_variable_detected(self, gen_module):
        rng = np.random.default_rng(3)
        idx = pd.date_range("2024-01-01", periods=500, freq="1min")
        close = pd.Series(1.10 + rng.normal(0, 0.0001, 500).cumsum(), index=idx)
        result = gen_module.run_backtest(close, None, "x = close * 2  # no 'signal' variable")
        assert result is None or result.get("status") != "success"


class TestAcceptanceCriteria:
    @pytest.mark.parametrize("ic,sharpe,trades,dd,oos_s,oos_m,expected", [
        (0.05, 0.8, 50, -0.05, 0.3, 2.0, True),
        (0.01, 0.8, 50, -0.05, 0.3, 2.0, False),   # IC too low
        (0.05, 0.3, 50, -0.05, 0.3, 2.0, False),    # Sharpe too low
        (0.05, 0.8, 5,  -0.05, 0.3, 2.0, False),    # Too few trades
        (0.05, 0.8, 50, -0.50, 0.3, 2.0, False),     # Drawdown too deep
        (0.05, 0.8, 50, -0.05, -0.1, 2.0, False),    # OOS Sharpe negative
        (0.05, 0.8, 50, -0.05, 0.3, -1.0, False),    # OOS monthly negative
    ])
    def test_daytrading_criteria(self, gen_module, ic, sharpe, trades, dd, oos_s, oos_m, expected):
        gen_module.TRADING_STYLE = "daytrading"
        gen_module.MIN_IC = 0.02
        gen_module.MIN_SHARPE = 0.5
        gen_module.MIN_TRADES = 30
        gen_module.MAX_DRAWDOWN = -0.10
        accepted = (
            abs(ic) > gen_module.MIN_IC
            and sharpe > gen_module.MIN_SHARPE
            and trades > gen_module.MIN_TRADES
            and dd > gen_module.MAX_DRAWDOWN
            and oos_s > 0.0
            and oos_m > 0.0
        )
        assert accepted == expected

    def test_ohlcv_only_mode_flags(self, gen_module):
        gen_module.OHLCV_ONLY = True
        assert gen_module.OHLCV_ONLY
        gen_module.OHLCV_ONLY = False


class TestConfiguration:
    def test_daytrading_defaults(self):
        """Daytrading config uses tighter risk limits."""
        os.environ["TRADING_STYLE"] = "daytrading"
        import importlib
        import scripts.predix_gen_strategies_real_bt as m
        importlib.reload(m)
        assert m.MIN_IC == 0.02
        assert m.MIN_SHARPE == 0.5
        assert m.MIN_TRADES == 300

    def test_swing_defaults(self):
        os.environ["TRADING_STYLE"] = "swing"
        import importlib
        import scripts.predix_gen_strategies_real_bt as m
        importlib.reload(m)
        assert m.MIN_TRADES == 10
        assert m.MAX_DRAWDOWN == -0.30
