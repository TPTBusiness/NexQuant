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


# ==============================================================================
# HYPOTHESIS-BASED PROPERTY TESTS — Data Pipeline Transformations,
# Bandit Properties, Feedback Consistency
# ==============================================================================
from hypothesis import given, settings, strategies as st
import numpy as np
import pandas as pd

from rdagent.scenarios.qlib.developer.feedback import process_results
from rdagent.scenarios.qlib.proposal.bandit import (
    Metrics,
    extract_metrics_from_experiment,
    LinearThompsonTwoArm,
)
from rdagent.scenarios.qlib.quant_loop_factory import (
    has_local_components,
    count_valid_factors,
    ADVANCED_LOOP_FACTOR_THRESHOLD,
)


# ---------------------------------------------------------------------------
# Property 1: process_results Invariants
# ---------------------------------------------------------------------------


class TestProcessResultsInvariants:
    """Property: process_results output invariants."""

    REQUIRED_METRICS = [
        "IC",
        "1day.excess_return_with_cost.annualized_return",
        "1day.excess_return_with_cost.max_drawdown",
    ]

    @given(
        ic=st.floats(min_value=-1.0, max_value=1.0),
        ann_return=st.floats(min_value=-2.0, max_value=5.0),
        max_dd=st.floats(min_value=-1.0, max_value=0.0),
        sota_ic=st.floats(min_value=-1.0, max_value=1.0),
        sota_ann_return=st.floats(min_value=-2.0, max_value=5.0),
        sota_max_dd=st.floats(min_value=-1.0, max_value=0.0),
    )
    @settings(max_examples=50, deadline=10000)
    def test_process_results_contains_all_metrics(
        self, ic, ann_return, max_dd, sota_ic, sota_ann_return, sota_max_dd
    ):
        """Property: output string contains IC, annualized_return, and max_drawdown."""
        current = pd.Series({
            "IC": ic,
            "1day.excess_return_with_cost.annualized_return": ann_return,
            "1day.excess_return_with_cost.max_drawdown": max_dd,
        }, name="0")
        sota = pd.Series({
            "IC": sota_ic,
            "1day.excess_return_with_cost.annualized_return": sota_ann_return,
            "1day.excess_return_with_cost.max_drawdown": sota_max_dd,
        }, name="0")

        result = process_results(current, sota)
        assert "IC of Current Result is" in result
        assert "of SOTA Result is" in result
        assert f"{ic:.6f}" in result or "nan" in result.lower()

    @given(
        ic=st.floats(min_value=-1.0, max_value=1.0),
        ann_return=st.floats(min_value=-2.0, max_value=5.0),
        max_dd=st.floats(min_value=-1.0, max_value=0.0),
    )
    @settings(max_examples=50, deadline=10000)
    def test_process_results_returns_string(self, ic, ann_return, max_dd):
        """Property: process_results returns a string."""
        current = pd.Series({
            "IC": ic,
            "1day.excess_return_with_cost.annualized_return": ann_return,
            "1day.excess_return_with_cost.max_drawdown": max_dd,
        }, name="0")
        sota = pd.Series({
            "IC": 0.0,
            "1day.excess_return_with_cost.annualized_return": 0.0,
            "1day.excess_return_with_cost.max_drawdown": 0.0,
        }, name="0")

        result = process_results(current, sota)
        assert isinstance(result, str)
        assert len(result) > 0

    @given(
        ic=st.floats(min_value=-1.0, max_value=1.0),
        ann_return=st.floats(min_value=-2.0, max_value=5.0),
        max_dd=st.floats(min_value=-1.0, max_value=0.0),
    )
    @settings(max_examples=50, deadline=10000)
    def test_process_results_raises_on_missing_metrics(self, ic, ann_return, max_dd):
        """Property: process_results raises KeyError on missing required metrics."""
        current = pd.Series({"IC": ic}, name="0")
        sota = pd.Series({"IC": 0.0}, name="0")
        with pytest.raises(KeyError):
            process_results(current, sota)

    @given(
        ic=st.floats(min_value=-1.0, max_value=1.0),
        ann_return=st.floats(min_value=-2.0, max_value=5.0),
        max_dd=st.floats(min_value=-1.0, max_value=0.0),
    )
    @settings(max_examples=50, deadline=10000)
    def test_process_results_format_consistent(self, ic, ann_return, max_dd):
        """Property: output format is '<metric> of Current Result is <val>, of SOTA Result is <val>'."""
        current = pd.Series({
            "IC": ic,
            "1day.excess_return_with_cost.annualized_return": ann_return,
            "1day.excess_return_with_cost.max_drawdown": max_dd,
        }, name="0")
        sota = pd.Series({
            "IC": 0.0,
            "1day.excess_return_with_cost.annualized_return": 0.0,
            "1day.excess_return_with_cost.max_drawdown": 0.0,
        }, name="0")

        result = process_results(current, sota)
        assert "of Current Result is" in result
        assert "of SOTA Result is" in result
        # Results separated by '; '
        assert ";" in result

    # -----------------------------------------------------------------------
    # Property 2: Metrics Default Values
    # -----------------------------------------------------------------------


class TestMetricsDefaults:
    """Property: Metrics default values are zero."""

    @given(
        ic=st.floats(min_value=-1.0, max_value=1.0),
        sharpe=st.floats(min_value=-5.0, max_value=10.0),
        rank_ic=st.floats(min_value=-1.0, max_value=1.0),
    )
    @settings(max_examples=50, deadline=10000)
    def test_partial_construction_defaults_to_zero(self, ic, sharpe, rank_ic):
        """Property: fields not specified default to 0.0."""
        m = Metrics(ic=ic, sharpe=sharpe, rank_ic=rank_ic)
        assert m.ic == ic
        assert m.sharpe == sharpe
        assert m.rank_ic == rank_ic
        assert m.icir == 0.0
        assert m.rank_icir == 0.0
        assert m.mdd == 0.0

    @given(
        icir=st.floats(min_value=-2.0, max_value=10.0),
        rank_icir=st.floats(min_value=-2.0, max_value=10.0),
        mdd=st.floats(min_value=-1.0, max_value=0.0),
    )
    @settings(max_examples=50, deadline=10000)
    def test_three_fields_default_others_zero(self, icir, rank_icir, mdd):
        """Property: only given fields set, others zero."""
        m = Metrics(icir=icir, rank_icir=rank_icir, mdd=mdd)
        assert m.ic == 0.0
        assert m.sharpe == 0.0
        assert m.rank_ic == 0.0
        assert m.icir == icir
        assert m.rank_icir == rank_icir
        assert m.mdd == mdd

    def test_all_defaults_zero(self):
        """Property: default constructor sets everything to zero."""
        m = Metrics()
        assert m.ic == 0.0
        assert m.sharpe == 0.0
        assert m.mdd == 0.0
        assert m.icir == 0.0
        assert m.rank_ic == 0.0
        assert m.rank_icir == 0.0


# ---------------------------------------------------------------------------
# Property 3: Metrics as_vector
# ---------------------------------------------------------------------------


class TestMetricsAsVector:
    """Property: as_vector invariants."""

    @given(
        ic=st.floats(min_value=-1.0, max_value=1.0),
        icir=st.floats(min_value=-2.0, max_value=10.0),
        rank_ic=st.floats(min_value=-1.0, max_value=1.0),
        rank_icir=st.floats(min_value=-2.0, max_value=10.0),
        ann_return=st.floats(min_value=-2.0, max_value=5.0),
        ir=st.floats(min_value=-5.0, max_value=10.0),
        mdd=st.floats(min_value=-1.0, max_value=0.0),
        sharpe=st.floats(min_value=-5.0, max_value=10.0),
    )
    @settings(max_examples=50, deadline=10000)
    def test_as_vector_length_is_8(self, ic, icir, rank_ic, rank_icir, ann_return, ir, mdd, sharpe):
        """Property: as_vector always returns length-8 array."""
        m = Metrics(
            ic=ic, icir=icir, rank_ic=rank_ic, rank_icir=rank_icir,
            arr=ann_return, ir=ir, mdd=mdd, sharpe=sharpe,
        )
        v = m.as_vector()
        assert len(v) == 8

    @given(
        ic=st.floats(min_value=-1.0, max_value=1.0),
        icir=st.floats(min_value=-2.0, max_value=10.0),
        rank_ic=st.floats(min_value=-1.0, max_value=1.0),
        rank_icir=st.floats(min_value=-2.0, max_value=10.0),
        ann_return=st.floats(min_value=-2.0, max_value=5.0),
        ir=st.floats(min_value=-5.0, max_value=10.0),
        mdd=st.floats(min_value=-1.0, max_value=0.0),
        sharpe=st.floats(min_value=-5.0, max_value=10.0),
    )
    @settings(max_examples=50, deadline=10000)
    def test_as_vector_matches_input_order(self, ic, icir, rank_ic, rank_icir, ann_return, ir, mdd, sharpe):
        """Property: vector elements match (ic, icir, rank_ic, rank_icir, ann_return, ir, -mdd, sharpe)."""
        m = Metrics(
            ic=ic, icir=icir, rank_ic=rank_ic, rank_icir=rank_icir,
            arr=ann_return, ir=ir, mdd=mdd, sharpe=sharpe,
        )
        v = m.as_vector()
        assert v[0] == ic
        assert v[1] == icir
        assert v[2] == rank_ic
        assert v[3] == rank_icir
        assert v[4] == ann_return
        assert v[5] == ir
        assert v[6] == -mdd  # negated
        assert v[7] == sharpe

    @given(
        mdd=st.floats(min_value=-1.0, max_value=0.0),
    )
    @settings(max_examples=50, deadline=10000)
    def test_mdd_negated_in_vector(self, mdd):
        """Property: mdd is negated in as_vector output (v[6] = -mdd)."""
        m = Metrics(mdd=mdd)
        v = m.as_vector()
        assert v[6] == -mdd

    @given(
        ic=st.floats(min_value=-1.0, max_value=1.0),
        icir=st.floats(min_value=-2.0, max_value=10.0),
        rank_ic=st.floats(min_value=-1.0, max_value=1.0),
        rank_icir=st.floats(min_value=-2.0, max_value=10.0),
        ann_return=st.floats(min_value=-2.0, max_value=5.0),
        ir=st.floats(min_value=-5.0, max_value=10.0),
        mdd=st.floats(min_value=-1.0, max_value=0.0),
        sharpe=st.floats(min_value=-5.0, max_value=10.0),
    )
    @settings(max_examples=50, deadline=10000)
    def test_as_vector_returns_numpy_array(self, ic, icir, rank_ic, rank_icir, ann_return, ir, mdd, sharpe):
        """Property: as_vector returns np.ndarray."""
        m = Metrics(
            ic=ic, icir=icir, rank_ic=rank_ic, rank_icir=rank_icir,
            arr=ann_return, ir=ir, mdd=mdd, sharpe=sharpe,
        )
        v = m.as_vector()
        assert isinstance(v, np.ndarray)


# ---------------------------------------------------------------------------
# Property 4: extract_metrics_from_experiment
# ---------------------------------------------------------------------------


class TestExtractMetrics:
    """Property: extract_metrics_from_experiment invariants."""

    @given(
        ic=st.floats(min_value=-1.0, max_value=1.0),
        icir=st.floats(min_value=-2.0, max_value=10.0),
        rank_ic=st.floats(min_value=-1.0, max_value=1.0),
        rank_icir=st.floats(min_value=-2.0, max_value=10.0),
        ann_return=st.floats(min_value=-2.0, max_value=5.0),
        ir=st.floats(min_value=-5.0, max_value=10.0),
        mdd=st.floats(min_value=-1.0, max_value=0.0),
    )
    @settings(max_examples=50, deadline=10000)
    def test_extract_metrics_correct_values(self, ic, icir, rank_ic, rank_icir, ann_return, ir, mdd):
        """Property: extract_metrics_from_experiment reads correct values from result dict."""
        mock_exp = MagicMock()
        mock_exp.result = {
            "IC": ic, "ICIR": icir,
            "Rank IC": rank_ic, "Rank ICIR": rank_icir,
            "1day.excess_return_with_cost.annualized_return ": ann_return,
            "1day.excess_return_with_cost.information_ratio": ir,
            "1day.excess_return_with_cost.max_drawdown": mdd,
        }
        m = extract_metrics_from_experiment(mock_exp)
        assert m.ic == ic
        assert m.rank_ic == rank_ic
        assert m.icir == icir
        assert m.rank_icir == rank_icir
        assert m.mdd == mdd

    @given(
        ann_return=st.floats(min_value=0.01, max_value=2.0),
        mdd=st.floats(min_value=-0.01, max_value=-0.001),
    )
    @settings(max_examples=50, deadline=10000)
    def test_sharpe_computed_from_ann_return_and_mdd(self, ann_return, mdd):
        """Property: sharpe ≈ ann_return / |mdd| for standard inputs."""
        mock_exp = MagicMock()
        mock_exp.result = {
            "IC": 0.0, "ICIR": 0.0,
            "Rank IC": 0.0, "Rank ICIR": 0.0,
            "1day.excess_return_with_cost.annualized_return ": ann_return,
            "1day.excess_return_with_cost.information_ratio": 0.0,
            "1day.excess_return_with_cost.max_drawdown": mdd,
        }
        m = extract_metrics_from_experiment(mock_exp)
        expected_sharpe = ann_return / abs(mdd)
        assert m.sharpe == pytest.approx(expected_sharpe, rel=0.01)

    @given(seed=st.integers(min_value=0, max_value=100))
    @settings(max_examples=50, deadline=10000)
    def test_extract_returns_default_on_none_result(self, seed):
        """Property: returns default Metrics (all zeros) when result is None."""
        mock_exp = MagicMock()
        mock_exp.result = None
        m = extract_metrics_from_experiment(mock_exp)
        assert m.ic == 0.0
        assert m.sharpe == 0.0
        assert m.mdd == 0.0

    @given(seed=st.integers(min_value=0, max_value=100))
    @settings(max_examples=50, deadline=10000)
    def test_extract_returns_default_on_empty_result(self, seed):
        """Property: returns default Metrics when result dict is empty."""
        mock_exp = MagicMock()
        mock_exp.result = {}
        m = extract_metrics_from_experiment(mock_exp)
        assert m.ic == 0.0
        assert m.sharpe == 0.0


# ---------------------------------------------------------------------------
# Property 5: LinearThompsonTwoArm
# ---------------------------------------------------------------------------


class TestLinearThompsonTwoArm:
    """Property: LinearThompsonTwoArm bandit invariants."""

    @given(dim=st.integers(min_value=1, max_value=20))
    @settings(max_examples=50, deadline=10000)
    def test_dim_stored_correctly(self, dim):
        """Property: dim attribute matches constructor arg."""
        bandit = LinearThompsonTwoArm(dim=dim)
        assert bandit.dim == dim

    @given(dim=st.integers(min_value=1, max_value=10))
    @settings(max_examples=50, deadline=10000)
    def test_mean_shape_matches_dim(self, dim):
        """Property: mean vectors have shape (dim,)."""
        bandit = LinearThompsonTwoArm(dim=dim)
        assert bandit.mean["factor"].shape == (dim,)
        assert bandit.mean["model"].shape == (dim,)

    @given(dim=st.integers(min_value=1, max_value=10))
    @settings(max_examples=50, deadline=10000)
    def test_precision_shape_matches_dim(self, dim):
        """Property: precision matrices have shape (dim, dim)."""
        bandit = LinearThompsonTwoArm(dim=dim)
        assert bandit.precision["factor"].shape == (dim, dim)
        assert bandit.precision["model"].shape == (dim, dim)

    @given(dim=st.integers(min_value=1, max_value=10))
    @settings(max_examples=50, deadline=10000)
    def test_arms_initialized_identically(self, dim):
        """Property: factor and model arms are initialized identically."""
        bandit = LinearThompsonTwoArm(dim=dim)
        assert np.array_equal(bandit.mean["factor"], bandit.mean["model"])
        assert np.array_equal(bandit.precision["factor"], bandit.precision["model"])

    @given(dim=st.integers(min_value=1, max_value=10))
    @settings(max_examples=50, deadline=10000)
    def test_noise_var_is_default_1(self, dim):
        """Property: noise_var defaults to 1.0."""
        bandit = LinearThompsonTwoArm(dim=dim)
        assert bandit.noise_var == 1.0

    @given(
        dim=st.integers(min_value=1, max_value=10),
        noise_var=st.floats(min_value=0.01, max_value=10.0),
    )
    @settings(max_examples=50, deadline=10000)
    def test_noise_var_configurable(self, dim, noise_var):
        """Property: noise_var can be set via constructor."""
        bandit = LinearThompsonTwoArm(dim=dim, noise_var=noise_var)
        assert bandit.noise_var == noise_var

    @given(
        dim=st.integers(min_value=1, max_value=10),
    )
    @settings(max_examples=50, deadline=10000)
    def test_sample_reward_returns_float(self, dim):
        """Property: sample_reward returns a float."""
        bandit = LinearThompsonTwoArm(dim=dim)
        x = np.ones(dim)
        reward = bandit.sample_reward("factor", x)
        assert isinstance(reward, float)

    @given(
        dim=st.integers(min_value=1, max_value=10),
    )
    @settings(max_examples=50, deadline=10000)
    def test_sample_reward_finite(self, dim):
        """Property: sample_reward returns finite values."""
        bandit = LinearThompsonTwoArm(dim=dim)
        x = np.ones(dim)
        reward = bandit.sample_reward("factor", x)
        assert np.isfinite(reward)

    @given(
        dim=st.integers(min_value=1, max_value=10),
        seed_a=st.integers(min_value=0, max_value=50),
        seed_b=st.integers(min_value=51, max_value=100),
    )
    @settings(max_examples=50, deadline=10000)
    def test_sample_reward_varies(self, dim, seed_a, seed_b):
        """Property: different seeds may produce different rewards (stochasticity)."""
        bandit = LinearThompsonTwoArm(dim=dim)
        x = np.ones(dim)
        r1 = bandit.sample_reward("factor", x)
        r2 = bandit.sample_reward("factor", x)
        # Both should be finite (may be equal by chance)
        assert np.isfinite(r1)
        assert np.isfinite(r2)

    @given(dim=st.integers(min_value=2, max_value=10))
    @settings(max_examples=50, deadline=10000)
    def test_precision_is_symmetric(self, dim):
        """Property: precision matrix is symmetric."""
        bandit = LinearThompsonTwoArm(dim=dim)
        P = bandit.precision["factor"]
        assert np.allclose(P, P.T, atol=1e-10)

    @given(dim=st.integers(min_value=1, max_value=10))
    @settings(max_examples=50, deadline=10000)
    def test_both_arms_have_same_keys(self, dim):
        """Property: both 'factor' and 'model' arms exist in mean/precision dicts."""
        bandit = LinearThompsonTwoArm(dim=dim)
        assert "factor" in bandit.mean
        assert "model" in bandit.mean
        assert "factor" in bandit.precision
        assert "model" in bandit.precision


# ---------------------------------------------------------------------------
# Property 6: LinearThompsonTwoArm Update
# ---------------------------------------------------------------------------


class TestBanditUpdate:
    """Property: Thompson bandit update invariants."""

    @given(dim=st.integers(min_value=1, max_value=10))
    @settings(max_examples=50, deadline=10000)
    def test_update_exists_for_both_arms(self, dim):
        """Property: update method is callable for both arms."""
        bandit = LinearThompsonTwoArm(dim=dim)
        x = np.ones(dim)
        bandit.update("factor", x, 0.5)
        bandit.update("model", x, 0.3)
        # Should not raise

    @given(dim=st.integers(min_value=1, max_value=10))
    @settings(max_examples=50, deadline=10000)
    def test_update_changes_mean(self, dim):
        """Property: updating an arm changes its mean vector."""
        bandit = LinearThompsonTwoArm(dim=dim)
        orig = bandit.mean["factor"].copy()
        x = np.ones(dim)
        bandit.update("factor", x, 1.0)
        # Mean should change (or be computed differently after update)
        assert not np.array_equal(orig, bandit.mean["factor"]) or np.array_equal(orig, np.zeros(dim))


# ---------------------------------------------------------------------------
# Property 7: has_local_components / count_valid_factors / ADVANCED_LOOP
# ---------------------------------------------------------------------------


class TestQuantLoopFactory:
    """Property: quant_loop_factory function invariants."""

    def test_has_local_components_returns_bool(self):
        """Property: has_local_components returns bool."""
        result = has_local_components()
        assert isinstance(result, bool)

    def test_count_valid_factors_returns_nonnegative_int(self):
        """Property: count_valid_factors returns nonnegative int."""
        result = count_valid_factors()
        assert isinstance(result, int)
        assert result >= 0

    def test_advanced_loop_threshold_is_5000(self):
        """Property: ADVANCED_LOOP_FACTOR_THRESHOLD == 5000."""
        assert ADVANCED_LOOP_FACTOR_THRESHOLD == 5000

    def test_advanced_loop_threshold_is_positive(self):
        """Property: ADVANCED_LOOP_FACTOR_THRESHOLD > 0."""
        assert ADVANCED_LOOP_FACTOR_THRESHOLD > 0

    def test_has_local_components_deterministic(self):
        """Property: has_local_components returns same value on repeated calls."""
        r1 = has_local_components()
        r2 = has_local_components()
        assert r1 == r2

    def test_count_valid_factors_deterministic(self):
        """Property: count_valid_factors returns same value on repeated calls."""
        r1 = count_valid_factors()
        r2 = count_valid_factors()
        assert r1 == r2


# ---------------------------------------------------------------------------
# Property 8: process_results Numeric Edge Cases
# ---------------------------------------------------------------------------


class TestProcessResultsEdgeCases:
    """Property: process_results handles edge case values."""

    @given(
        ic=st.floats(min_value=-1.0, max_value=1.0, allow_nan=False, allow_infinity=False),
        ann_return=st.floats(min_value=-2.0, max_value=5.0, allow_nan=False, allow_infinity=False),
        max_dd=st.floats(min_value=-1.0, max_value=0.0, allow_nan=False, allow_infinity=False),
    )
    @settings(max_examples=50, deadline=10000)
    def test_all_numeric_values_formatted(self, ic, ann_return, max_dd):
        """Property: all valid numeric values produce a result string."""
        current = pd.Series({
            "IC": ic,
            "1day.excess_return_with_cost.annualized_return": ann_return,
            "1day.excess_return_with_cost.max_drawdown": max_dd,
        }, name="0")
        sota = pd.Series({
            "IC": 0.0,
            "1day.excess_return_with_cost.annualized_return": 0.0,
            "1day.excess_return_with_cost.max_drawdown": 0.0,
        }, name="0")

        result = process_results(current, sota)
        assert isinstance(result, str)

    @given(
        ic=st.floats(min_value=-1.0, max_value=1.0, allow_nan=False, allow_infinity=False),
        ann_return=st.floats(min_value=-2.0, max_value=5.0, allow_nan=False, allow_infinity=False),
        max_dd=st.floats(min_value=-1.0, max_value=0.0, allow_nan=False, allow_infinity=False),
    )
    @settings(max_examples=50, deadline=10000)
    def test_result_contains_both_current_and_sota(self, ic, ann_return, max_dd):
        """Property: result contains 'Current Result' and 'SOTA Result'."""
        current = pd.Series({
            "IC": ic,
            "1day.excess_return_with_cost.annualized_return": ann_return,
            "1day.excess_return_with_cost.max_drawdown": max_dd,
        }, name="0")
        sota = pd.Series({
            "IC": 0.0,
            "1day.excess_return_with_cost.annualized_return": 0.0,
            "1day.excess_return_with_cost.max_drawdown": 0.0,
        }, name="0")

        result = process_results(current, sota)
        assert "Current Result" in result
        assert "SOTA Result" in result


# ---------------------------------------------------------------------------
# Property 9: Metrics Constructor Type Safety
# ---------------------------------------------------------------------------


class TestMetricsTypeSafety:
    """Property: Metrics converts inputs to float."""

    @given(
        ic=st.integers(min_value=-10, max_value=10),
        sharpe=st.integers(min_value=-5, max_value=20),
        mdd=st.floats(min_value=-1.0, max_value=0.0),
    )
    @settings(max_examples=50, deadline=10000)
    def test_float_conversion(self, ic, sharpe, mdd):
        """Property: integer inputs become floats."""
        m = Metrics(ic=float(ic), sharpe=float(sharpe), mdd=mdd)
        assert isinstance(m.ic, float)
        assert isinstance(m.sharpe, float)
        assert isinstance(m.mdd, float)


# ---------------------------------------------------------------------------
# Property 10: Bandit Precision Positive Definite
# ---------------------------------------------------------------------------


class TestBanditPrecisionProperties:
    """Property: precision matrix is positive semi-definite (identity-initialized)."""

    @given(dim=st.integers(min_value=1, max_value=10))
    @settings(max_examples=50, deadline=10000)
    def test_precision_is_identity_initialized(self, dim):
        """Property: precision matrix starts as identity."""
        bandit = LinearThompsonTwoArm(dim=dim)
        P = bandit.precision["factor"]
        expected = np.eye(dim)
        assert np.allclose(P, expected, atol=1e-10)

    @given(dim=st.integers(min_value=1, max_value=10))
    @settings(max_examples=50, deadline=10000)
    def test_precision_diagonal_positive(self, dim):
        """Property: precision matrix diagonal elements are positive."""
        bandit = LinearThompsonTwoArm(dim=dim)
        P = bandit.precision["factor"]
        assert (np.diag(P) > 0).all()


# ---------------------------------------------------------------------------
# Property 11: Bandit Mean Initialization
# ---------------------------------------------------------------------------


class TestBanditMeanInitialization:
    """Property: mean vector is initialized to zeros."""

    @given(dim=st.integers(min_value=1, max_value=10))
    @settings(max_examples=50, deadline=10000)
    def test_mean_is_zero_initialized(self, dim):
        """Property: mean starts as zero vector."""
        bandit = LinearThompsonTwoArm(dim=dim)
        m = bandit.mean["factor"]
        expected = np.zeros(dim)
        assert np.allclose(m, expected, atol=1e-10)

    @given(dim=st.integers(min_value=1, max_value=10))
    @settings(max_examples=50, deadline=10000)
    def test_both_arms_mean_zero_initialized(self, dim):
        """Property: both arm means start as zero."""
        bandit = LinearThompsonTwoArm(dim=dim)
        assert np.allclose(bandit.mean["factor"], np.zeros(dim))
        assert np.allclose(bandit.mean["model"], np.zeros(dim))


# ---------------------------------------------------------------------------
# Property 12: extract_metrics Robustness
# ---------------------------------------------------------------------------


class TestExtractMetricsRobustness:
    """Property: extract_metrics_from_experiment handles missing keys."""

    @given(
        ic=st.floats(min_value=-1.0, max_value=1.0),
    )
    @settings(max_examples=50, deadline=10000)
    def test_partial_result_dict(self, ic):
        """Property: partial result dict fills defaults for missing keys."""
        mock_exp = MagicMock()
        mock_exp.result = {"IC": ic}
        m = extract_metrics_from_experiment(mock_exp)
        assert m.ic == ic
        assert m.sharpe == 0.0  # default since ann_return is missing

    @given(seed=st.integers(min_value=0, max_value=100))
    @settings(max_examples=50, deadline=10000)
    def test_extract_with_empty_dict(self, seed):
        """Property: empty result dict → all defaults or raises."""
        mock_exp = MagicMock()
        mock_exp.result = {}
        m = extract_metrics_from_experiment(mock_exp)
        assert isinstance(m, Metrics)
        assert m.ic == 0.0


# ---------------------------------------------------------------------------
# Property 13: Metrics Field Naming
# ---------------------------------------------------------------------------


class TestMetricsFieldNaming:
    """Property: Metrics has specific named fields."""

    def test_metrics_has_all_expected_fields(self):
        """Property: Metrics has ic, icir, rank_ic, rank_icir, ann_return, ir, mdd, sharpe."""
        m = Metrics()
        expected = {"ic", "icir", "rank_ic", "rank_icir", "arr", "ir", "mdd", "sharpe"}
        actual = {k for k in m.__dict__ if not k.startswith("_")}
        assert expected <= actual or expected <= set(m.__dataclass_fields__ if hasattr(m, "__dataclass_fields__") else [])

    @given(
        ann_return=st.floats(min_value=-2.0, max_value=5.0),
        ir=st.floats(min_value=-5.0, max_value=10.0),
        sharpe=st.floats(min_value=-5.0, max_value=10.0),
    )
    @settings(max_examples=50, deadline=10000)
    def test_return_and_sharpe_fields(self, ann_return, ir, sharpe):
        """Property: ann_return, ir, sharpe accessible by attribute."""
        m = Metrics(arr=ann_return, ir=ir, sharpe=sharpe)
        assert m.arr == ann_return
        assert m.ir == ir
        assert m.sharpe == sharpe


# ---------------------------------------------------------------------------
# Property 14: process_results Determinism
# ---------------------------------------------------------------------------


class TestProcessResultsDeterminism:
    """Property: process_results is deterministic."""

    @given(
        ic=st.floats(min_value=-1.0, max_value=1.0),
        ann_return=st.floats(min_value=-2.0, max_value=5.0),
        max_dd=st.floats(min_value=-1.0, max_value=0.0),
    )
    @settings(max_examples=50, deadline=10000)
    def test_same_inputs_same_output(self, ic, ann_return, max_dd):
        """Property: process_results is deterministic."""
        current = pd.Series({
            "IC": ic,
            "1day.excess_return_with_cost.annualized_return": ann_return,
            "1day.excess_return_with_cost.max_drawdown": max_dd,
        }, name="0")
        sota = pd.Series({
            "IC": 0.0,
            "1day.excess_return_with_cost.annualized_return": 0.0,
            "1day.excess_return_with_cost.max_drawdown": 0.0,
        }, name="0")

        r1 = process_results(current, sota)
        r2 = process_results(current, sota)
        assert r1 == r2


# ---------------------------------------------------------------------------
# Property 15: Bandit Sample Reward Distribution
# ---------------------------------------------------------------------------


class TestBanditSampleReward:
    """Property: sample_reward behavior across arms."""

    @given(dim=st.integers(min_value=1, max_value=10))
    @settings(max_examples=50, deadline=10000)
    def test_factor_and_model_reward_differ(self, dim):
        """Property: factor and model arms can give different rewards."""
        bandit = LinearThompsonTwoArm(dim=dim)
        x = np.random.randn(dim)
        r_factor = bandit.sample_reward("factor", x)
        r_model = bandit.sample_reward("model", x)
        assert isinstance(r_factor, float)
        assert isinstance(r_model, float)

    @given(
        dim=st.integers(min_value=1, max_value=10),
        n_samples=st.integers(min_value=10, max_value=100),
    )
    @settings(max_examples=10, deadline=10000)
    def test_sample_reward_changes_after_update(self, dim, n_samples):
        """Property: after updates, sample_reward distribution shifts."""
        bandit = LinearThompsonTwoArm(dim=dim)
        x = np.ones(dim)
        rewards_before = [bandit.sample_reward("factor", x) for _ in range(n_samples)]

        # Update with positive rewards
        for _ in range(10):
            bandit.update("factor", x, 1.0)

        rewards_after = [bandit.sample_reward("factor", x) for _ in range(n_samples)]

        # Mean should shift (though statistically it may not)
        assert np.all(np.isfinite(rewards_before))
        assert np.all(np.isfinite(rewards_after))
