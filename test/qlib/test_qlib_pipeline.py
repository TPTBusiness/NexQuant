"""Tests for qlib pipeline — feedback, bandit, quant_loop_factory."""

from __future__ import annotations

import sys
import tempfile
from pathlib import Path
from unittest.mock import MagicMock, patch

import numpy as np
import pandas as pd
import pytest

PROJECT_ROOT = Path(__file__).parent.parent.parent
sys.path.insert(0, str(PROJECT_ROOT))


# =============================================================================
# process_results (feedback.py)
# =============================================================================


class TestProcessResults:
    def test_process_results_handles_named_series(self):
        """process_results renames column "0" to "Current Result" — this works
        when the Series name is '0' (string), which matches the rename dict."""
        from rdagent.scenarios.qlib.developer.feedback import process_results
        import pandas as pd

        # process_results expects the Series to produce a DataFrame column named "0" (string)
        # This happens when the Series has name '0'
        current = pd.Series(
            {"IC": 0.05, "1day.excess_return_with_cost.annualized_return": 0.12,
             "1day.excess_return_with_cost.max_drawdown": -0.08},
            name="0",
        )
        sota = pd.Series(
            {"IC": 0.03, "1day.excess_return_with_cost.annualized_return": 0.10,
             "1day.excess_return_with_cost.max_drawdown": -0.05},
            name="0",
        )

        result = process_results(current, sota)
        assert "IC of Current Result is" in result
        assert "of SOTA Result is" in result

    def test_raises_on_missing_metrics(self):
        from rdagent.scenarios.qlib.developer.feedback import process_results

        current = pd.Series({"IC": 0.05})
        sota = pd.Series({"IC": 0.03})
        with pytest.raises(KeyError):
            process_results(current, sota)


# =============================================================================
# bandit.py — Metrics and extract_metrics_from_experiment
# =============================================================================


class TestBanditMetrics:
    def test_default_values_are_zero(self):
        from rdagent.scenarios.qlib.proposal.bandit import Metrics
        m = Metrics()
        assert m.ic == 0.0
        assert m.sharpe == 0.0
        assert m.mdd == 0.0

    def test_as_vector_length(self):
        from rdagent.scenarios.qlib.proposal.bandit import Metrics
        m = Metrics(ic=0.1, sharpe=1.5)
        v = m.as_vector()
        assert len(v) == 8
        assert v[0] == 0.1
        assert v[7] == 1.5

    def test_mdd_negated_in_vector(self):
        from rdagent.scenarios.qlib.proposal.bandit import Metrics
        m = Metrics(mdd=0.15)
        v = m.as_vector()
        assert v[6] == -0.15  # -self.mdd

    def test_extract_metrics_from_experiment(self):
        from rdagent.scenarios.qlib.proposal.bandit import extract_metrics_from_experiment

        mock_exp = MagicMock()
        mock_exp.result = {
            "IC": 0.04,
            "ICIR": 0.5,
            "Rank IC": 0.03,
            "Rank ICIR": 0.4,
            "1day.excess_return_with_cost.annualized_return ": 0.10,
            "1day.excess_return_with_cost.information_ratio": 0.6,
            "1day.excess_return_with_cost.max_drawdown": -0.12,
        }
        m = extract_metrics_from_experiment(mock_exp)
        assert m.ic == 0.04
        assert m.rank_ic == 0.03
        assert m.mdd == -0.12

    def test_extract_metrics_returns_default_on_error(self):
        from rdagent.scenarios.qlib.proposal.bandit import extract_metrics_from_experiment

        mock_exp = MagicMock()
        mock_exp.result = None  # Will cause AttributeError
        m = extract_metrics_from_experiment(mock_exp)
        assert m.ic == 0.0
        assert m.sharpe == 0.0

    def test_sharpe_computation(self):
        from rdagent.scenarios.qlib.proposal.bandit import extract_metrics_from_experiment

        mock_exp = MagicMock()
        mock_exp.result = {
            "IC": 0.0, "ICIR": 0.0, "Rank IC": 0.0, "Rank ICIR": 0.0,
            "1day.excess_return_with_cost.annualized_return ": 0.15,
            "1day.excess_return_with_cost.information_ratio": 0.0,
            "1day.excess_return_with_cost.max_drawdown": -0.10,
        }
        m = extract_metrics_from_experiment(mock_exp)
        assert m.sharpe == pytest.approx(1.5)  # 0.15 / 0.10


# =============================================================================
# LinearThompsonTwoArm
# =============================================================================


class TestLinearThompsonTwoArm:
    def test_initialization(self):
        from rdagent.scenarios.qlib.proposal.bandit import LinearThompsonTwoArm
        bandit = LinearThompsonTwoArm(dim=5)
        assert bandit.dim == 5
        assert bandit.noise_var == 1.0
        assert bandit.mean["factor"].shape == (5,)
        assert bandit.mean["model"].shape == (5,)
        assert bandit.precision["factor"].shape == (5, 5)

    def test_sample_reward_returns_float(self):
        from rdagent.scenarios.qlib.proposal.bandit import LinearThompsonTwoArm
        bandit = LinearThompsonTwoArm(dim=3)
        x = np.ones(3)
        reward = bandit.sample_reward("factor", x)
        assert isinstance(reward, float)

    def test_arms_are_initialized_identically(self):
        from rdagent.scenarios.qlib.proposal.bandit import LinearThompsonTwoArm
        bandit = LinearThompsonTwoArm(dim=4)
        assert np.array_equal(bandit.mean["factor"], bandit.mean["model"])
        assert np.array_equal(bandit.precision["factor"], bandit.precision["model"])


# =============================================================================
# quant_loop_factory.py
# =============================================================================


class TestHasLocalComponents:
    def test_returns_bool(self):
        from rdagent.scenarios.qlib.quant_loop_factory import has_local_components
        result = has_local_components()
        assert isinstance(result, bool)

    def test_returns_false_with_no_local_dir(self, monkeypatch):
        from rdagent.scenarios.qlib import quant_loop_factory
        monkeypatch.setattr(quant_loop_factory.Path, "exists", lambda self: False)
        assert quant_loop_factory.has_local_components() is False


class TestCountValidFactors:
    def test_returns_zero_when_no_dir(self):
        from rdagent.scenarios.qlib.quant_loop_factory import count_valid_factors
        with patch("rdagent.scenarios.qlib.quant_loop_factory.Path.exists", return_value=False):
            assert count_valid_factors() == 0

    def test_returns_int(self):
        from rdagent.scenarios.qlib.quant_loop_factory import count_valid_factors
        result = count_valid_factors()
        assert isinstance(result, int)
        assert result >= 0


class TestAdvancedLoopThreshold:
    def test_constant_is_defined(self):
        from rdagent.scenarios.qlib.quant_loop_factory import ADVANCED_LOOP_FACTOR_THRESHOLD
        assert ADVANCED_LOOP_FACTOR_THRESHOLD == 5000
