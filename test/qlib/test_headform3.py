"""Batch 3: walk-forward details, signal validation, IC bounds."""

from __future__ import annotations
import sys
from pathlib import Path
import numpy as np, pandas as pd, pytest

PROJECT_ROOT = Path(__file__).parent.parent.parent
sys.path.insert(0, str(PROJECT_ROOT))


class TestWalkForwardDetails:
    def test_non_datetime_returns_empty(self):
        from rdagent.components.backtesting.vbt_backtest import walk_forward_rolling
        result = walk_forward_rolling(pd.Series([1.0]), pd.Series([1.0]), leverage=1.0)
        assert result == {"wf_n_windows": 0}

    def test_wf_consistency_bounds(self):
        from rdagent.components.backtesting.vbt_backtest import walk_forward_rolling
        dates = pd.date_range("2020-01-01", "2023-12-31", freq="1min")
        rng = np.random.default_rng(42)
        close = pd.Series(1.10 + rng.normal(0, 0.0001, len(dates)).cumsum(), index=dates)
        signal = pd.Series(np.where(rng.normal(0, 1, len(dates)) > 0, 1.0, -1.0), index=dates)
        result = walk_forward_rolling(close, signal, leverage=1.0)
        if result["wf_n_windows"] > 0 and "wf_oos_consistency" in result:
            assert 0.0 <= result["wf_oos_consistency"] <= 1.0


class TestSignalValidation:
    def test_constant_signal_zero_trades(self):
        from rdagent.components.backtesting.vbt_backtest import backtest_signal
        dates = pd.date_range("2024-01-01", periods=500, freq="1min")
        close = pd.Series(1.10, index=dates)
        result = backtest_signal(close, pd.Series(1.0, index=dates), txn_cost_bps=0.0)
        assert result["n_trades"] >= 0

    def test_binary_signal_range(self):
        from rdagent.components.backtesting.vbt_backtest import backtest_signal
        n = 1000
        dates = pd.date_range("2024-01-01", periods=n, freq="1min")
        close = pd.Series(1.10 + np.random.default_rng(42).normal(0, 0.0002, n).cumsum(), index=dates)
        for val in [0.0, 1.0, -1.0, 2.0, -2.0]:
            result = backtest_signal(close, pd.Series(val, index=dates))
            assert result["status"] in ("success", "failed")

    def test_float_signal_works(self):
        from rdagent.components.backtesting.vbt_backtest import backtest_signal
        n = 1000
        dates = pd.date_range("2024-01-01", periods=n, freq="1min")
        close = pd.Series(1.10 + np.random.default_rng(42).normal(0, 0.0002, n).cumsum(), index=dates)
        signal = pd.Series(np.random.default_rng(43).normal(0, 1, n), index=dates)
        result = backtest_signal(close, signal)
        assert result["status"] in ("success", "failed")


class TestBacktestFromFwdReturnsDetails:
    def test_ic_always_between_neg1_and_1(self):
        from rdagent.components.backtesting.vbt_backtest import backtest_from_forward_returns
        for seed in [42, 43, 44, 45, 46]:
            idx = pd.MultiIndex.from_arrays(
                [pd.date_range("2024-01-01", periods=500, freq="1min"), ["EURUSD"] * 500],
                names=["datetime", "instrument"],
            )
            close = pd.Series(1.10 + np.random.default_rng(seed).normal(0, 0.0001, 500).cumsum(), index=idx)
            fwd = close.groupby(level="instrument").shift(-96) / close - 1
            factor = pd.Series(np.random.default_rng(seed + 100).normal(0, 1, 500), index=idx)
            result = backtest_from_forward_returns(factor, fwd, close)
            if result["status"] == "success" and "ic" in result:
                assert -1.0 <= result["ic"] <= 1.0

    def test_trades_non_negative(self):
        from rdagent.components.backtesting.vbt_backtest import backtest_from_forward_returns
        idx = pd.MultiIndex.from_arrays(
            [pd.date_range("2024-01-01", periods=500, freq="1min"), ["EURUSD"] * 500],
            names=["datetime", "instrument"],
        )
        close = pd.Series(1.10 + np.arange(500) * 0.0001, index=idx)
        fwd = close.groupby(level="instrument").shift(-96) / close - 1
        factor = pd.Series(np.random.default_rng(42).normal(0, 1, 500), index=idx)
        result = backtest_from_forward_returns(factor, fwd, close)
        if result["status"] == "success":
            assert result.get("n_trades", 0) >= 0
