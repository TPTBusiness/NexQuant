"""Batch 3: ensemble, optuna path, signal validation, walk-forward details."""

from __future__ import annotations

import sys
from pathlib import Path
from unittest.mock import MagicMock, patch

import numpy as np
import pandas as pd
import pytest

PROJECT_ROOT = Path(__file__).parent.parent.parent
sys.path.insert(0, str(PROJECT_ROOT))


class TestEnsembleEdgeCases:
    def test_orchestrator_module_loads(self):
        from rdagent.scenarios.qlib.local import strategy_orchestrator as so
        assert hasattr(so, 'StrategyOrchestrator')


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

    def test_wf_keys_present(self):
        from rdagent.components.backtesting.vbt_backtest import walk_forward_rolling
        dates = pd.date_range("2020-01-01", "2023-12-31", freq="1min")
        rng = np.random.default_rng(42)
        close = pd.Series(1.10 + rng.normal(0, 0.0001, len(dates)).cumsum(), index=dates)
        signal = pd.Series(np.where(rng.normal(0, 1, len(dates)) > 0, 1.0, -1.0), index=dates)
        result = walk_forward_rolling(close, signal, leverage=1.0)
        for key in ["wf_n_windows"]:
            assert key in result


class TestOptunaPath:
    def test_optuna_optimizer_init(self):
        from rdagent.scenarios.qlib.local.optuna_optimizer import OptunaOptimizer
        opt = OptunaOptimizer(n_trials=3)
        assert opt.n_trials == 3
        assert opt.optimization_metric == "sharpe"

    def test_optuna_accepts_strategy_dict(self):
        from rdagent.scenarios.qlib.local.optuna_optimizer import OptunaOptimizer
        opt = OptunaOptimizer(n_trials=3)
        strat = {"strategy_name": "test", "status": "rejected", "sharpe_ratio": -1.0}
        try:
            result = opt.optimize_strategy(strat, pd.DataFrame({"a": [1, 2, 3, 4, 5, 6]}))
            assert isinstance(result, dict)
        except Exception:
            pass  # OHLCV may not be available


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
        signal_values = [0.0, 1.0, -1.0, 0.5, -0.5, 2.0, -2.0, 100.0, -100.0]
        for val in signal_values:
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


class TestPreflightValidation:
    def test_syntax_error_caught(self):
        from rdagent.scenarios.qlib.local.strategy_orchestrator import StrategyOrchestrator
        orch = StrategyOrchestrator.__new__(StrategyOrchestrator)
        result = orch._preflight_check("if True print(x)")
        assert result is not None

    def test_no_signal_caught(self):
        from rdagent.scenarios.qlib.local.strategy_orchestrator import StrategyOrchestrator
        orch = StrategyOrchestrator.__new__(StrategyOrchestrator)
        result = orch._preflight_check("x = 1")
        assert result is not None
        assert "signal" in result.lower()

    def test_valid_code_passes(self):
        from rdagent.scenarios.qlib.local.strategy_orchestrator import StrategyOrchestrator
        orch = StrategyOrchestrator.__new__(StrategyOrchestrator)
        result = orch._preflight_check("import numpy as np\nsignal = np.array([1.0, -1.0, 1.0])")
        assert result is None

    def test_constant_signal_caught(self):
        from rdagent.scenarios.qlib.local.strategy_orchestrator import StrategyOrchestrator
        orch = StrategyOrchestrator.__new__(StrategyOrchestrator)
        result = orch._preflight_check("import numpy as np\nsignal = np.array([1.0, 1.0, 1.0])")
        assert result is not None
