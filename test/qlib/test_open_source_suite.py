"""Open-source test suite V2 — fixed assertions."""

from __future__ import annotations

import json
import sys
import tempfile
from pathlib import Path
from unittest.mock import MagicMock, patch

import numpy as np
import pandas as pd
import pytest

PROJECT_ROOT = Path(__file__).parent.parent.parent
sys.path.insert(0, str(PROJECT_ROOT))


class TestNexQuantCLI:
    def test_cli_commands_available(self):
        import subprocess
        r = subprocess.run([sys.executable, "nexquant.py", "--help"], capture_output=True, text=True, timeout=10)
        assert r.returncode == 0
        for cmd in ["evaluate", "top", "best", "portfolio", "build-strategies", "generate-strategies", "health"]:
            assert cmd in r.stdout.lower(), f"Missing command: {cmd}"


class TestBacktestEdgeCases:
    def test_all_zero_signal(self):
        from rdagent.components.backtesting.vbt_backtest import backtest_signal
        n = 500
        close = pd.Series(1.10 + np.arange(n) * 0.0001, index=pd.date_range("2024-01-01", periods=n, freq="1min"))
        result = backtest_signal(close, pd.Series(0.0, index=close.index))
        assert result["n_trades"] == 0

    def test_sortino_present(self):
        from rdagent.components.backtesting.vbt_backtest import backtest_signal
        n = 2000
        dates = pd.date_range("2024-01-01", periods=n, freq="1min")
        close = pd.Series(1.10 * np.exp(np.cumsum(np.random.default_rng(42).normal(0, 0.0002, n))), index=dates)
        signal = pd.Series(np.where(np.random.default_rng(43).normal(0, 1, n) > 0, 1.0, -1.0), index=dates)
        result = backtest_signal(close, signal)
        assert "sortino" in result
        assert result["sortino"] is not None

    def test_calmar_present(self):
        from rdagent.components.backtesting.vbt_backtest import backtest_signal
        n = 2000
        dates = pd.date_range("2024-01-01", periods=n, freq="1min")
        close = pd.Series(1.10 * np.exp(np.cumsum(np.random.default_rng(42).normal(0, 0.0002, n))), index=dates)
        signal = pd.Series(np.where(np.random.default_rng(44).normal(0, 1, n) > 0, 1.0, -1.0), index=dates)
        result = backtest_signal(close, signal)
        assert "calmar" in result

    def test_all_required_keys_present(self):
        from rdagent.components.backtesting.vbt_backtest import backtest_signal
        n = 2000
        dates = pd.date_range("2024-01-01", periods=n, freq="1min")
        close = pd.Series(1.10 * np.exp(np.cumsum(np.random.default_rng(42).normal(0, 0.0002, n))), index=dates)
        signal = pd.Series(np.where(np.random.default_rng(43).normal(0, 1, n) > 0, 1.0, -1.0), index=dates)
        result = backtest_signal(close, signal)
        required = ["sharpe", "max_drawdown", "win_rate", "total_return", "n_trades",
                    "annual_return_pct", "monthly_return_pct", "sortino", "calmar"]
        for key in required:
            assert key in result, f"Missing: {key}"


class TestCoreUtils:
    def test_multiprocessing_wrapper(self):
        from rdagent.core.utils import multiprocessing_wrapper
        def fn(x):
            return x * 2
        results = multiprocessing_wrapper([(fn, (5,))], n=1)
        assert results[0] == 10

    def test_import_class_valid(self):
        from rdagent.core.utils import import_class
        cls = import_class("rdagent.core.exception.WorkflowError")
        assert cls is not None

    def test_singleton(self):
        from rdagent.core.utils import SingletonBaseClass
        class A(SingletonBaseClass):
            pass
        assert A() is A()


class TestBacktestFromFwdReturns:
    def test_all_nan(self):
        from rdagent.components.backtesting.vbt_backtest import backtest_from_forward_returns
        idx = pd.MultiIndex.from_arrays([pd.date_range("2024-01-01", periods=500, freq="1min"),
                                          ["EURUSD"] * 500], names=["datetime", "instrument"])
        close = pd.Series(1.10 + np.arange(500) * 0.0001, index=idx)
        fwd = close.groupby(level="instrument").shift(-96) / close - 1
        result = backtest_from_forward_returns(pd.Series([np.nan] * 500, index=idx), fwd, close)
        assert result["status"] == "failed"

    def test_ic_bounds(self):
        from rdagent.components.backtesting.vbt_backtest import backtest_from_forward_returns
        idx = pd.MultiIndex.from_arrays([pd.date_range("2024-01-01", periods=500, freq="1min"),
                                          ["EURUSD"] * 500], names=["datetime", "instrument"])
        close = pd.Series(1.10 + np.arange(500) * 0.0001, index=idx)
        fwd = close.groupby(level="instrument").shift(-96) / close - 1
        factor = pd.Series(np.random.default_rng(42).normal(0, 1, 500), index=idx)
        result = backtest_from_forward_returns(factor, fwd, close)
        if result["status"] == "success" and "ic" in result:
            assert -1.0 <= result["ic"] <= 1.0


class TestProtectionEdgeCases:
    def test_empty_manager(self):
        from rdagent.components.backtesting.protections import ProtectionManager
        pm = ProtectionManager()
        r = pm.check_all(returns=[0.01], timestamps=[], current_equity=100000, peak_equity=100000)
        assert not r.should_block

    def test_with_defaults(self):
        from rdagent.components.backtesting.protections import ProtectionManager
        pm = ProtectionManager()
        pm.create_default_protections()
        r = pm.check_all(returns=[0.01], timestamps=[pd.Timestamp.now()], current_equity=100000, peak_equity=101000)
        assert not r.should_block

    def test_get_stats(self):
        from rdagent.components.backtesting.protections import ProtectionManager
        pm = ProtectionManager()
        pm.create_default_protections()
        stats = pm.get_stats()
        assert isinstance(stats, dict)

    def test_protection_result_active(self):
        from rdagent.components.backtesting.protections import ProtectionResult
        from datetime import datetime, timedelta
        pr = ProtectionResult(should_block=True, reason="test", until=datetime.now() + timedelta(hours=1))
        assert pr.is_active


class TestEnvImports:
    def test_all_importable(self):
        from rdagent.utils.env import Env, QTDockerEnv, QlibCondaConf, QlibCondaEnv, KGDockerEnv
        assert all([Env, QTDockerEnv, QlibCondaConf, QlibCondaEnv, KGDockerEnv])


class TestLogInfra:
    def test_all_importable(self):
        from rdagent.log.conf import LOG_SETTINGS
        from rdagent.log.logger import RDAgentLog
        from rdagent.log.daily_log import session
        from rdagent.log.timer import RD_Agent_TIMER_wrapper
        assert LOG_SETTINGS is not None
        assert RDAgentLog is not None
        assert callable(session)
        assert RD_Agent_TIMER_wrapper is not None


class TestCoreExperiment:
    def test_task_and_experiment(self):
        from rdagent.core.experiment import Task, Experiment, FBWorkspace
        t = Task(name="t", description="d")
        exp = Experiment(sub_tasks=[t])
        assert len(exp.sub_tasks) == 1
        ws = FBWorkspace()
        assert ws.workspace_path is not None


class TestPromptLoader:
    def test_loads_strategy_generation(self):
        from rdagent.components.prompt_loader import load_prompt
        result = load_prompt("strategy_generation")
        assert isinstance(result, dict)

    def test_missing_raises(self):
        from rdagent.components.prompt_loader import load_prompt
        with pytest.raises(FileNotFoundError):
            load_prompt("xyz_nonexistent")


class TestApplyFTMOMask:
    def test_output_same_length(self):
        from rdagent.components.backtesting.vbt_backtest import _apply_ftmo_mask
        dates = pd.date_range("2024-01-01", periods=100, freq="1min")
        close = pd.Series(1.10, index=dates)
        signal = pd.Series(np.where(np.arange(100) % 2 == 0, 1.0, -1.0), index=dates)
        masked, metrics = _apply_ftmo_mask(signal, close, leverage=1.0, txn_cost_bps=2.14)
        assert len(masked) == len(signal)
        assert isinstance(metrics, dict)

    def test_flat_signal(self):
        from rdagent.components.backtesting.vbt_backtest import _apply_ftmo_mask
        dates = pd.date_range("2024-01-01", periods=200, freq="1min")
        close = pd.Series(1.10, index=dates)
        signal = pd.Series(0.0, index=dates)
        masked, metrics = _apply_ftmo_mask(signal, close, leverage=1.0, txn_cost_bps=2.14)
        assert isinstance(metrics, dict)


class TestBacktestSignalMetrics:
    def test_flat_signal_zero_trades(self):
        from rdagent.components.backtesting.vbt_backtest import backtest_signal
        dates = pd.date_range("2024-01-01", periods=500, freq="1min")
        close = pd.Series(1.10, index=dates)
        result = backtest_signal(close, pd.Series(0.0, index=dates))
        assert result["n_trades"] == 0

    def test_nan_signal_handled(self):
        from rdagent.components.backtesting.vbt_backtest import backtest_signal
        n = 200
        dates = pd.date_range("2024-01-01", periods=n, freq="1min")
        close = pd.Series(1.10 + np.random.default_rng(42).normal(0, 0.001, n).cumsum(), index=dates)
        signal = pd.Series(np.where(np.random.default_rng(42).normal(0, 1, n) > 0, 1.0, np.nan), index=dates)
        result = backtest_signal(close, signal)
        assert result["status"] in ("success", "failed")
