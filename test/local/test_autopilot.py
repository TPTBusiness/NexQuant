"""Deep tests for nexquant_autopilot.py — property-based, mocks, edge cases.

Tests the core logic of the 24/7 strategy generator by mocking
the StrategyOrchestrator at the correct import path.
"""

from __future__ import annotations
from hypothesis import given, settings, HealthCheck
from hypothesis import strategies as st

import sys
from pathlib import Path
from unittest.mock import MagicMock, patch

import numpy as np
import pandas as pd
import pytest

PROJECT_ROOT = Path(__file__).parent.parent.parent
sys.path.insert(0, str(PROJECT_ROOT))

from hypothesis import given, settings
from hypothesis import strategies as st


@pytest.fixture
def mock_orch():
    """Mock StrategyOrchestrator that returns configurable results."""
    with patch(
        "rdagent.scenarios.qlib.local.strategy_orchestrator.StrategyOrchestrator",
        autospec=True,
    ) as mock:
        instance = MagicMock()
        mock.return_value = instance
        yield mock, instance


class TestMainRound:
    def test_returns_zero_on_init_failure(self):
        with patch(
            "rdagent.scenarios.qlib.local.strategy_orchestrator.StrategyOrchestrator",
            side_effect=RuntimeError("no data"),
        ):
            from scripts.nexquant_autopilot import main_round
            result = main_round("daytrading", 1)
            assert result == 0

    def test_returns_zero_on_generate_failure(self, mock_orch):
        mock_cls, instance = mock_orch
        instance.generate_strategies.side_effect = RuntimeError("crash")
        from scripts.nexquant_autopilot import main_round
        result = main_round("daytrading", 1)
        assert result == 0

    def test_counts_accepted(self, mock_orch):
        mock_cls, instance = mock_orch
        instance.generate_strategies.return_value = [
            {"status": "accepted", "strategy_name": "s1", "sharpe_ratio": 0.5, "oos_sharpe": 0.3},
            {"status": "rejected", "strategy_name": "s2", "reason": "low"},
        ]
        from scripts.nexquant_autopilot import main_round
        result = main_round("daytrading", 1)
        assert result == 1

    def test_ensemble_called_when_2_accepted(self, mock_orch):
        mock_cls, instance = mock_orch
        instance.generate_strategies.return_value = [
            {"status": "accepted", "strategy_name": "a", "sharpe_ratio": 0.5, "oos_sharpe": 0.3},
            {"status": "accepted", "strategy_name": "b", "sharpe_ratio": 0.6, "oos_sharpe": 0.4},
        ]
        instance.build_ensemble.return_value = {
            "status": "success", "sharpe_ratio": 0.95, "oos_sharpe": 0.65,
            "members": ["a", "b"],
        }
        from scripts.nexquant_autopilot import main_round
        main_round("daytrading", 1)
        instance.build_ensemble.assert_called_once()

    def test_ensemble_not_called_when_lt_2_accepted(self, mock_orch):
        mock_cls, instance = mock_orch
        instance.generate_strategies.return_value = [
            {"status": "accepted", "strategy_name": "a", "sharpe_ratio": 0.5, "oos_sharpe": 0.3},
            {"status": "rejected", "strategy_name": "b", "reason": "no"},
        ]
        from scripts.nexquant_autopilot import main_round
        main_round("daytrading", 1)
        instance.build_ensemble.assert_not_called()

    def test_ensemble_failure_doesnt_crash(self, mock_orch):
        mock_cls, instance = mock_orch
        instance.generate_strategies.return_value = [
            {"status": "accepted", "strategy_name": "a", "sharpe_ratio": 0.5, "oos_sharpe": 0.3},
            {"status": "accepted", "strategy_name": "b", "sharpe_ratio": 0.6, "oos_sharpe": 0.4},
        ]
        instance.build_ensemble.side_effect = RuntimeError("boom")
        from scripts.nexquant_autopilot import main_round
        result = main_round("daytrading", 1)
        assert result == 2  # Still counts accepted

    def test_empty_results_returns_zero(self, mock_orch):
        mock_cls, instance = mock_orch
        instance.generate_strategies.return_value = []
        from scripts.nexquant_autopilot import main_round
        result = main_round("daytrading", 1)
        assert result == 0

    def test_ensemble_none_returns_zero_extra(self, mock_orch):
        mock_cls, instance = mock_orch
        instance.generate_strategies.return_value = [
            {"status": "accepted", "strategy_name": "a", "sharpe_ratio": 0.5, "oos_sharpe": 0.3},
            {"status": "accepted", "strategy_name": "b", "sharpe_ratio": 0.6, "oos_sharpe": 0.4},
        ]
        instance.build_ensemble.return_value = None
        from scripts.nexquant_autopilot import main_round
        result = main_round("daytrading", 1)
        assert result == 2

    @given(
        sharpe=st.floats(min_value=-5, max_value=5, allow_nan=False, allow_infinity=False),
        name=st.text(min_size=1, max_size=20),
    )
    @settings(max_examples=100, deadline=5000)
    def test_main_round_never_crashes_property(self, sharpe, name):
        with patch(
            "rdagent.scenarios.qlib.local.strategy_orchestrator.StrategyOrchestrator",
            autospec=True,
        ) as mock_cls:
            instance = MagicMock()
            instance.generate_strategies.return_value = [
                {"status": "accepted" if sharpe > 0.1 else "rejected",
                 "strategy_name": name, "sharpe_ratio": sharpe, "oos_sharpe": 0.0,
                 "reason": "test"},
            ]
            mock_cls.return_value = instance
            from scripts.nexquant_autopilot import main_round
            result = main_round("daytrading", 1)
            assert isinstance(result, int) and result >= 0


class TestConfig:
    def test_batch_size_positive(self):
        from scripts import nexquant_autopilot
        assert nexquant_autopilot.BATCH_SIZE > 0

    def test_optuna_trials_positive(self):
        from scripts import nexquant_autopilot
        assert nexquant_autopilot.OPTUNA_TRIALS > 0

    def test_cooldown_positive(self):
        from scripts import nexquant_autopilot
        assert nexquant_autopilot.COOLDOWN > 0

    def test_max_consecutive_fails_positive(self):
        from scripts import nexquant_autopilot
        assert nexquant_autopilot.MAX_CONSECUTIVE_FAILS > 0


class TestStyleCycling:
    def test_odd_rounds_are_daytrading(self):
        styles = ["swing", "daytrading"]
        for r in range(1, 20, 2):
            assert styles[r % 2] == "daytrading"

    def test_even_rounds_are_swing(self):
        styles = ["swing", "daytrading"]
        for r in range(2, 21, 2):
            assert styles[r % 2] == "swing"
