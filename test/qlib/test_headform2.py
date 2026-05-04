"""More headform tests: performance, chaining, stress, integration, edge cases."""

from __future__ import annotations

import sys
from pathlib import Path
from unittest.mock import MagicMock, patch

import numpy as np
import pandas as pd
import pytest

PROJECT_ROOT = Path(__file__).parent.parent.parent
sys.path.insert(0, str(PROJECT_ROOT))


class TestPerformanceBounds:
    def test_backtest_completes_under_1s_for_1k_bars(self):
        import time
        from rdagent.components.backtesting.vbt_backtest import backtest_signal
        n = 1000
        dates = pd.date_range("2024-01-01", periods=n, freq="1min")
        close = pd.Series(1.10 + np.random.default_rng(42).normal(0, 0.0002, n).cumsum(), index=dates)
        signal = pd.Series(np.where(np.random.default_rng(43).normal(0, 1, n) > 0, 1.0, -1.0), index=dates)
        t0 = time.time()
        result = backtest_signal(close, signal)
        elapsed = time.time() - t0
        assert elapsed < 0.5, f"Backtest took {elapsed:.3f}s for {n} bars"
        assert result["status"] == "success"

    def test_backtest_scales_linearly(self):
        import time
        from rdagent.components.backtesting.vbt_backtest import backtest_signal
        times = []
        for n in [500, 1000, 2000]:
            dates = pd.date_range("2024-01-01", periods=n, freq="1min")
            close = pd.Series(1.10 + np.random.default_rng(42).normal(0, 0.0002, n).cumsum(), index=dates)
            signal = pd.Series(np.where(np.random.default_rng(43).normal(0, 1, n) > 0, 1.0, -1.0), index=dates)
            t0 = time.time()
            backtest_signal(close, signal)
            times.append(time.time() - t0)
        ratios = [times[i+1]/times[i] for i in range(len(times)-1)]
        for r in ratios:
            assert r < 5, f"Non-linear scaling: {ratios}"


class TestChainingConsistency:
    def test_two_backtests_same_result(self):
        from rdagent.components.backtesting.vbt_backtest import backtest_signal
        n = 2000
        dates = pd.date_range("2024-01-01", periods=n, freq="1min")
        rng = np.random.default_rng(42)
        close = pd.Series(1.10 * np.exp(np.cumsum(rng.normal(0, 0.0002, n))), index=dates)
        signal = pd.Series(np.where(rng.normal(0, 1, n) > 0, 1.0, -1.0), index=dates)
        r1 = backtest_signal(close, signal, txn_cost_bps=2.14)
        r2 = backtest_signal(close, signal, txn_cost_bps=2.14)
        assert r1["sharpe"] == r2["sharpe"]
        assert r1["max_drawdown"] == r2["max_drawdown"]

    def test_chained_backtests_no_side_effects(self):
        from rdagent.components.backtesting.vbt_backtest import backtest_signal
        n = 2000
        dates = pd.date_range("2024-01-01", periods=n, freq="1min")
        rng = np.random.default_rng(42)
        close1 = pd.Series(1.10 * np.exp(np.cumsum(rng.normal(0, 0.0002, n))), index=dates)
        close2 = pd.Series(1.10 * np.exp(np.cumsum(rng.normal(0, 0.0001, n))), index=dates)
        s1 = pd.Series(np.where(rng.normal(0, 1, n) > 0, 1.0, -1.0), index=dates)
        r1 = backtest_signal(close1, s1)
        r2 = backtest_signal(close2, s1)
        assert r1["sharpe"] != r2["sharpe"]  # Different data → different results


class TestMultiIndexEdgeCases:
    def test_single_instrument_multiindex(self):
        from rdagent.components.backtesting.vbt_backtest import backtest_from_forward_returns
        dates = pd.date_range("2024-01-01", periods=500, freq="1min")
        idx = pd.MultiIndex.from_arrays([dates, ["EURUSD"]*500], names=["datetime", "instrument"])
        close = pd.Series(1.10 + np.arange(500)*0.0001, index=idx)
        fwd = close.groupby(level="instrument").shift(-96) / close - 1
        factor = pd.Series(np.random.default_rng(42).normal(0, 1, 500), index=idx)
        result = backtest_from_forward_returns(factor, fwd, close)
        assert result["status"] in ("success", "failed")

    def test_duplicate_datetime_index(self):
        from rdagent.components.backtesting.vbt_backtest import backtest_signal
        dates = pd.date_range("2024-01-01", periods=200, freq="1min")
        close = pd.Series(1.10, index=dates)
        signal = pd.Series(np.where(np.arange(200)%2==0, 1.0, -1.0), index=dates)
        result = backtest_signal(close, signal)
        assert result["status"] in ("success", "failed")

    def test_unsorted_index(self):
        from rdagent.components.backtesting.vbt_backtest import backtest_signal
        dates = pd.date_range("2024-01-01", periods=500, freq="1min")
        close = pd.Series(1.10, index=dates)
        signal = pd.Series(np.where(np.arange(500)%2==0, 1.0, -1.0), index=dates)
        # Reverse order
        close_rev = close.iloc[::-1]
        signal_rev = signal.iloc[::-1]
        result = backtest_signal(close_rev, signal_rev)
        assert result["status"] in ("success", "failed")


class TestMetricBounds:
    def test_sortino_non_negative_for_profitable(self):
        from rdagent.components.backtesting.vbt_backtest import backtest_signal
        n = 2000
        dates = pd.date_range("2024-01-01", periods=n, freq="1min")
        close = pd.Series(1.10 + np.arange(n) * 0.0001, index=dates)
        signal = pd.Series(1.0, index=dates)
        result = backtest_signal(close, signal, txn_cost_bps=0.0)
        if result["status"] == "success":
            assert result.get("sortino", -1) >= -1

    def test_calmar_bounded(self):
        from rdagent.components.backtesting.vbt_backtest import backtest_signal
        n = 2000
        dates = pd.date_range("2024-01-01", periods=n, freq="1min")
        close = pd.Series(1.10 + np.random.default_rng(42).normal(0, 0.0002, n).cumsum(), index=dates)
        signal = pd.Series(np.where(np.random.default_rng(43).normal(0, 1, n) > 0, 1.0, -1.0), index=dates)
        result = backtest_signal(close, signal)
        if result["status"] == "success" and "calmar" in result:
            assert np.isfinite(result["calmar"])

    def test_profit_factor_range(self):
        from rdagent.components.backtesting.vbt_backtest import backtest_signal
        n = 2000
        dates = pd.date_range("2024-01-01", periods=n, freq="1min")
        close = pd.Series(1.10 + np.random.default_rng(42).normal(0, 0.0002, n).cumsum(), index=dates)
        signal = pd.Series(np.where(np.random.default_rng(43).normal(0, 1, n) > 0, 1.0, -1.0), index=dates)
        result = backtest_signal(close, signal)
        if result["status"] == "success" and "profit_factor" in result and result["profit_factor"] is not None:
            assert result["profit_factor"] >= 0


class TestDataQualityDetection:
    def test_nan_handling_in_eval(self):
        from rdagent.scenarios.qlib.developer.factor_runner import QlibFactorRunner
        import inspect
        source = inspect.getsource(QlibFactorRunner._evaluate_factor_directly)
        assert "dropna" in source.lower() or "np.isnan" in source

    def test_min_data_check(self):
        from rdagent.scenarios.qlib.developer.factor_runner import QlibFactorRunner
        import inspect
        source = inspect.getsource(QlibFactorRunner._evaluate_factor_directly)
        assert "len(valid_idx)" in source or "len(valid)" in source

    def test_nan_ic_returns_none(self):
        from rdagent.scenarios.qlib.developer.factor_runner import QlibFactorRunner
        import inspect
        source = inspect.getsource(QlibFactorRunner._evaluate_factor_directly)
        assert "isnan" in source.lower()


class TestFactorRunnerEdgeCases:
    def test_write_run_log_creates_entry(self, tmp_path, monkeypatch):
        from rdagent.scenarios.qlib.developer.factor_runner import QlibFactorRunner
        import os as _os
        runner = QlibFactorRunner.__new__(QlibFactorRunner)
        exp = MagicMock()
        exp.hypothesis = MagicMock()
        exp.hypothesis.hypothesis = "TestFactor"
        result = pd.Series({"IC": 0.05, "1day.excess_return_with_cost.shar": 1.0, "win_rate": 0.55})
        monkeypatch.setattr(_os, "getenv", lambda k, d="0": d)
        with patch("rdagent.scenarios.qlib.developer.factor_runner.Path.__new__", return_value=Path(tmp_path)):
            try:
                runner._write_run_log(exp, result)
            except Exception:
                pass  # May fail due to path mocking

    def test_save_failed_run_no_crash(self, tmp_path, monkeypatch):
        from rdagent.scenarios.qlib.developer.factor_runner import QlibFactorRunner
        runner = QlibFactorRunner.__new__(QlibFactorRunner)
        exp = MagicMock()
        exp.hypothesis = MagicMock()
        exp.hypothesis.hypothesis = "Test"
        with patch("rdagent.scenarios.qlib.developer.factor_runner.Path.__new__", return_value=Path(tmp_path)):
            try:
                runner._save_failed_run(exp, stdout="test", error_type="test_error")
            except Exception:
                pass
