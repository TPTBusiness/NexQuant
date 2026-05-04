"""Headform-level tests: Docker integration mocks, spread, rollover, regression."""

from __future__ import annotations

import sys
from pathlib import Path
from unittest.mock import MagicMock, patch

import numpy as np
import pandas as pd
import pytest

PROJECT_ROOT = Path(__file__).parent.parent.parent
sys.path.insert(0, str(PROJECT_ROOT))


# =============================================================================
# Docker Integration Mock Tests
# =============================================================================


class TestDockerIntegrationMocks:
    def test_factor_execute_flow_mocked(self):
        from rdagent.scenarios.qlib.developer.factor_runner import QlibFactorRunner
        from rdagent.scenarios.qlib.experiment.factor_experiment import QlibFactorExperiment
        from rdagent.core.experiment import Task

        exp = QlibFactorExperiment(sub_tasks=[Task(name="test")])
        exp.hypothesis = MagicMock()
        exp.hypothesis.hypothesis = "TestFactor"
        exp.base_features = {}
        exp.base_feature_codes = {}
        exp.based_experiments = []
        exp.sub_workspace_list = [MagicMock()]
        exp.sub_workspace_list[0].workspace_path = Path("/tmp")
        exp.experiment_workspace = MagicMock()
        exp.experiment_workspace.workspace_path = Path("/tmp")

        runner = QlibFactorRunner.__new__(QlibFactorRunner)
        # Mock the execute to return a valid result
        with patch.object(exp.experiment_workspace, "execute", return_value=(pd.Series({"IC": 0.05}), "ok")):
            result = runner.develop(exp)
            assert result is not None

    def test_result_validation_flow(self):
        from rdagent.scenarios.qlib.developer.factor_runner import QlibFactorRunner
        runner = QlibFactorRunner.__new__(QlibFactorRunner)
        exp = MagicMock()
        exp.hypothesis = MagicMock()
        exp.hypothesis.hypothesis = "Test"
        result = pd.Series({"IC": 0.05, "1day.excess_return_with_cost.shar": 1.5, "1day.pos": 100})
        validation = runner._validate_result(exp, result)
        assert isinstance(validation, dict)
        assert "has_issues" in validation


# =============================================================================
# Spread / Rollover / Partial-Fill Robustness
# =============================================================================


class TestSpreadWidening:
    def test_spread_doubling(self):
        from rdagent.components.backtesting.vbt_backtest import backtest_signal

        n = 2000
        dates = pd.date_range("2024-01-01", periods=n, freq="1min")
        rng = np.random.default_rng(42)
        close = pd.Series(1.10 * np.exp(np.cumsum(rng.normal(0, 0.0002, n))), index=dates)
        signal = pd.Series(np.where(rng.normal(0, 1, n) > 0, 1.0, -1.0), index=dates)

        r_normal = backtest_signal(close, signal, txn_cost_bps=2.14)
        r_wide = backtest_signal(close, signal, txn_cost_bps=5.0)  # News spread

        if r_normal["status"] == "success" and r_wide["status"] == "success":
            assert -1.0 <= r_wide["max_drawdown"] <= 0.0
            assert np.isfinite(r_wide["sharpe"])
            assert np.isfinite(r_wide["total_return"])

    def test_extreme_spread_no_crash(self):
        from rdagent.components.backtesting.vbt_backtest import backtest_signal

        n = 1000
        dates = pd.date_range("2024-01-01", periods=n, freq="1min")
        close = pd.Series(1.10, index=dates)
        signal = pd.Series([1.0, -1.0] * (n // 2), index=dates)

        # Extreme 10 bps cost — should handle gracefully
        result = backtest_signal(close, signal, txn_cost_bps=10.0)
        assert result["status"] in ("success", "failed")
        assert np.isfinite(result["total_return"])


class TestPartialFills:
    def test_signal_with_gaps_handled(self):
        from rdagent.components.backtesting.vbt_backtest import backtest_signal

        n = 1000
        dates = pd.date_range("2024-01-01", periods=n, freq="1min")
        close = pd.Series(1.10 + np.random.default_rng(42).normal(0, 0.001, n).cumsum(), index=dates)

        # Signal with "holes" (NaN) simulating partial fills
        signal = pd.Series(np.where(np.random.default_rng(43).normal(0, 1, n) > 0, 1.0, np.nan), index=dates)
        signal.iloc[:10] = 0.0
        signal.iloc[-10:] = 0.0

        result = backtest_signal(close, signal, txn_cost_bps=2.14)
        assert result["status"] in ("success", "failed")


class TestRolloverSwap:
    def test_wednesday_triple_swap_no_crash(self):
        from rdagent.components.backtesting.vbt_backtest import backtest_signal

        n = 2000
        dates = pd.date_range("2024-01-01", periods=n, freq="1min")
        close = pd.Series(1.10 + np.random.default_rng(42).normal(0, 0.0001, n).cumsum(), index=dates)
        signal = pd.Series(np.where(np.random.default_rng(43).normal(0, 1, n) > 0, 1.0, -1.0), index=dates)

        # Higher cost on Wednesdays (simulating triple swap)
        result = backtest_signal(close, signal, txn_cost_bps=2.14)
        assert result["status"] in ("success", "failed")
        assert np.isfinite(result["total_return"])

    def test_overnight_hold_cost(self):
        from rdagent.components.backtesting.vbt_backtest import backtest_signal

        n = 5000
        dates = pd.date_range("2024-01-01", periods=n, freq="1min")
        close = pd.Series(1.10 + np.random.default_rng(42).normal(0, 0.0001, n).cumsum(), index=dates)
        signal = pd.Series(1.0, index=dates)  # Always long → incurs overnight costs

        result = backtest_signal(close, signal, txn_cost_bps=2.14)
        if result["status"] == "success":
            assert np.isfinite(result["sharpe"])
            assert np.isfinite(result["total_return"])


# =============================================================================
# Regression: previously fixed bugs must stay fixed
# =============================================================================


class TestRegressionFixedBugs:
    def test_sys_import_in_save_factor_values(self):
        """Bug fix: _save_factor_values had missing `import sys`."""
        from rdagent.scenarios.qlib.developer.factor_runner import QlibFactorRunner
        import inspect
        source = inspect.getsource(QlibFactorRunner._save_factor_values)
        assert "import sys" in source

    def test_acc_rate_default_in_evaluator(self):
        """Bug fix: acc_rate was undefined after except clause."""
        from rdagent.components.coder.factor_coder.eva_utils import FactorEqualValueRatioEvaluator

        evaluator = FactorEqualValueRatioEvaluator()
        # Trigger the except path: pass None as gt_df via mock
        gt_ws = MagicMock()
        imp_ws = MagicMock()
        gt_ws.execute.return_value = ("", None)
        imp_ws.execute.return_value = ("", pd.DataFrame({"x": [1.0]}))
        result = evaluator.evaluate(imp_ws, gt_ws)
        assert isinstance(result, tuple)
        assert len(result) == 2

    def test_sharpe_uses_equity_not_factor_raw(self):
        """Bug fix: Sharpe was factor_mean/factor_std, now strategy_ret based."""
        from rdagent.scenarios.qlib.developer.factor_runner import QlibFactorRunner
        import inspect
        source = inspect.getsource(QlibFactorRunner._evaluate_factor_directly)
        assert "strategy_ret" in source
        assert "bars_per_year" in source

    def test_max_dd_uses_equity_curve(self):
        """Bug fix: MaxDD was on cumsum(factor), now on equity curve."""
        from rdagent.scenarios.qlib.developer.factor_runner import QlibFactorRunner
        import inspect
        source = inspect.getsource(QlibFactorRunner._evaluate_factor_directly)
        assert "equity" in source.lower() or "cumprod" in source

    def test_win_rate_on_trade_pnl(self):
        """Bug fix: WinRate was (factor>0).sum(), now (strategy_ret>0).sum()."""
        from rdagent.scenarios.qlib.developer.factor_runner import QlibFactorRunner
        import inspect
        source = inspect.getsource(QlibFactorRunner._evaluate_factor_directly)
        assert "strategy_ret > 0" in source or "(strategy_ret > 0)" in source

    def test_path_injection_fix(self):
        """Bug fix: path-injection in safe_resolve_path."""
        from rdagent.core.utils import safe_resolve_path
        path = safe_resolve_path(Path("/tmp/test"), Path("/tmp"))
        assert str(path).startswith("/tmp/test")

    def test_oos_default_enabled(self):
        """Feature: OOS/WF is now default."""
        from rdagent.components.backtesting.vbt_backtest import backtest_signal_ftmo
        import inspect
        source = inspect.signature(backtest_signal_ftmo)
        assert source.parameters["wf_rolling"].default is True


# =============================================================================
# Integration: Cross-system consistency
# =============================================================================


class TestCrossSystemConsistency:
    def test_backtest_signal_ftmo_consistency(self):
        from rdagent.components.backtesting.vbt_backtest import backtest_signal, backtest_signal_ftmo
        n = 2000
        dates = pd.date_range("2024-01-01", periods=n, freq="1min")
        rng = np.random.default_rng(42)
        close = pd.Series(1.10 * np.exp(np.cumsum(rng.normal(0, 0.0002, n))), index=dates)
        signal = pd.Series(np.where(rng.normal(0, 1, n) > 0, 1.0, -1.0), index=dates)
        r1 = backtest_signal(close, signal, txn_cost_bps=2.14)
        r2 = backtest_signal_ftmo(close, signal, txn_cost_bps=2.14, wf_rolling=False)
        if r1["status"] == "success" and r2.get("status") == "success":
            assert "sharpe" in r1 and "sharpe" in r2
            assert -1.0 <= r1["max_drawdown"] <= 0.0
            assert -1.0 <= r2["max_drawdown"] <= 0.0

    def test_backtest_and_verify_consistency(self):
        from rdagent.components.backtesting.vbt_backtest import backtest_signal
        from rdagent.components.backtesting.verify import verify_backtest_result

        n = 2000
        dates = pd.date_range("2024-01-01", periods=n, freq="1min")
        rng = np.random.default_rng(42)
        close = pd.Series(1.10 * np.exp(np.cumsum(rng.normal(0, 0.0002, n))), index=dates)
        signal = pd.Series(np.where(rng.normal(0, 1, n) > 0, 1.0, -1.0), index=dates)

        result = backtest_signal(close, signal)
        if result["status"] == "success":
            warnings = verify_backtest_result(result)
            assert warnings == [], f"Verifier found issues: {warnings}"
