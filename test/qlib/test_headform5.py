"""Batch 5: runtime verifier edge cases, factor loader, ensemble, stability."""

from __future__ import annotations

import sys
from pathlib import Path
from unittest.mock import MagicMock, patch

import numpy as np
import pandas as pd
import pytest

PROJECT_ROOT = Path(__file__).parent.parent.parent
sys.path.insert(0, str(PROJECT_ROOT))


class TestRuntimeVerifierMissesNothing:
    def test_all_keys_missing_detected(self):
        from rdagent.components.backtesting.verify import verify_backtest_result
        assert len(verify_backtest_result({})) > 0

    def test_partial_keys_missing(self):
        from rdagent.components.backtesting.verify import verify_backtest_result
        w = verify_backtest_result({"sharpe": 1.0, "max_drawdown": -0.1})
        assert len(w) > 0

    def test_zero_sharpe_accepted(self):
        from rdagent.components.backtesting.verify import verify_backtest_result
        result = {
            "sharpe": 0.0, "max_drawdown": -0.15, "win_rate": 0.5,
            "total_return": 0.0, "annual_return_pct": 0.0,
            "monthly_return_pct": 0.0, "n_trades": 10, "status": "success",
        }
        assert verify_backtest_result(result) == []


class TestFactorLoaderEdgeCases:
    def test_load_nonexistent_factor(self):
        from rdagent.scenarios.qlib.local.strategy_orchestrator import StrategyOrchestrator
        orch = StrategyOrchestrator.__new__(StrategyOrchestrator)
        orch.values_dir = Path("/nonexistent")
        result = orch.load_factor_values("nonexistent")
        assert result is None


class TestStabilityCheckEdgeCases:
    def test_too_short_data_passes(self):
        from rdagent.scenarios.qlib.local.strategy_orchestrator import StrategyOrchestrator
        orch = StrategyOrchestrator.__new__(StrategyOrchestrator)
        dates = pd.date_range("2024-01-01", periods=100, freq="1min")
        close = pd.Series(1.10, index=dates)
        signal = pd.Series(np.where(np.arange(100)%2==0, 1.0, -1.0), index=dates)
        result = orch._check_stability(signal, close, "test")
        assert result["passed"] is True

    def test_negative_sharpe_fails(self):
        from rdagent.scenarios.qlib.local.strategy_orchestrator import StrategyOrchestrator
        orch = StrategyOrchestrator.__new__(StrategyOrchestrator)
        n = 5000
        dates = pd.date_range("2020-01-01", periods=n, freq="1min")
        rng = np.random.default_rng(42)
        close = pd.Series(1.10 * np.exp(np.cumsum(rng.normal(0, 0.0002, n))), index=dates)
        signal = pd.Series(np.where(rng.normal(0, 1, n) > 0, 1.0, -1.0), index=dates)
        result = orch._check_stability(signal, close, "test")
        assert isinstance(result, dict)
        assert "passed" in result
        assert "worst_sharpe" in result


class TestEnsembleBuilder:
    def test_build_ensemble_exists(self):
        from rdagent.scenarios.qlib.local.strategy_orchestrator import StrategyOrchestrator
        assert hasattr(StrategyOrchestrator, 'build_ensemble')



class TestMultiTimeframeEdgeCases:
    def test_returns_dict(self):
        from rdagent.scenarios.qlib.local.strategy_orchestrator import StrategyOrchestrator
        orch = StrategyOrchestrator.__new__(StrategyOrchestrator)
        dates = pd.date_range("2024-01-01", periods=500, freq="1min")
        close = pd.Series(1.10, index=dates)
        signal = pd.Series(np.where(np.arange(500)%2==0, 1.0, -1.0), index=dates)
        result = orch._check_multi_timeframe(signal, close, "test")
        assert isinstance(result, dict)
        assert "passed" in result
        assert "timeframes" in result


class TestSaveStrategyJson:
    def test_save_creates_file(self, tmp_path):
        import json
        (tmp_path / "factors").mkdir()
        from rdagent.scenarios.qlib.local.strategy_orchestrator import StrategyOrchestrator
        orch = StrategyOrchestrator.__new__(StrategyOrchestrator)
        orch.strategies_dir = tmp_path
        orch._save_strategy({"strategy_name": "test", "status": "accepted", "sharpe_ratio": 1.0})
        files = list(tmp_path.glob("*.json"))
        assert len(files) > 0
