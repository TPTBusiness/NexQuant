"""
Tests for ML Feedback Integrator

Tests the MLFeedbackMixin class for correct trigger logic,
factor counting, and prompt feedback generation.

15 tests covering:
- Initialization and configuration
- Trigger condition logic
- Factor counting methods
- Feature importance extraction
- Prompt suggestion generation
- Graceful error handling
"""

import json
import os
import sys
import tempfile
from pathlib import Path
from unittest.mock import MagicMock, patch

import pytest


# ---------------------------------------------------------------------------
# Fixtures
# ---------------------------------------------------------------------------


@pytest.fixture
def mock_project_root(tmp_path: Path) -> Path:
    """Create a temporary project structure for testing."""
    # Create directory structure
    (tmp_path / "results" / "factors").mkdir(parents=True)
    (tmp_path / "results" / "models").mkdir(parents=True)
    (tmp_path / "prompts" / "local").mkdir(parents=True)
    return tmp_path


@pytest.fixture
def mock_factor_data() -> dict:
    """Return sample factor data for testing."""
    return {
        "name": "test_momentum_factor",
        "status": "success",
        "ic": 0.15,
        "sharpe_ratio": 1.8,
        "max_drawdown": -0.12,
        "win_rate": 0.55,
        "code": "def factor(): ...",
    }


@pytest.fixture
def mock_importance_data() -> dict:
    """Return sample feature importance data."""
    return {
        "importance": {
            "momentum_5d": 0.25,
            "volatility_10d": 0.18,
            "mean_reversion_3d": 0.12,
            "volume_spike": 0.08,
            "trend_strength": 0.05,
        },
        "model_type": "lightgbm",
        "n_factors": 50,
    }


# ---------------------------------------------------------------------------
# Tests
# ---------------------------------------------------------------------------


class TestMLFeedbackMixinInit:
    """Test MLFeedbackMixin initialization."""

    def test_default_initialization(self, mock_project_root):
        """Test default configuration values."""
        from rdagent.scenarios.qlib.local.feedback_integrator import MLFeedbackMixin

        # Create a mock parent class
        class MockParent:
            def __init__(self):
                pass

        class TestMixin(MLFeedbackMixin, MockParent):
            pass

        mixin = TestMixin(ml_feedback=True)

        assert mixin.ml_feedback_enabled is True
        assert mixin.ml_train_interval == 500
        assert mixin.strategy_gen_interval == 1000
        assert mixin.portfolio_opt_interval == 2000

    def test_custom_intervals(self, mock_project_root):
        """Test custom interval configuration."""
        from rdagent.scenarios.qlib.local.feedback_integrator import MLFeedbackMixin

        class MockParent:
            def __init__(self):
                pass

        class TestMixin(MLFeedbackMixin, MockParent):
            pass

        mixin = TestMixin(
            ml_feedback=True,
            ml_train_interval=1000,
            strategy_gen_interval=2000,
            portfolio_opt_interval=4000,
        )

        assert mixin.ml_train_interval == 1000
        assert mixin.strategy_gen_interval == 2000
        assert mixin.portfolio_opt_interval == 4000

    def test_disabled_feedback(self, mock_project_root):
        """Test disabled feedback mode."""
        from rdagent.scenarios.qlib.local.feedback_integrator import MLFeedbackMixin

        class MockParent:
            def __init__(self):
                pass

        class TestMixin(MLFeedbackMixin, MockParent):
            pass

        mixin = TestMixin(ml_feedback=False)
        assert mixin.ml_feedback_enabled is False


class TestTriggerConditions:
    """Test trigger condition logic."""

    def test_should_trigger_ml_train_at_threshold(self):
        """Test ML train trigger at exact threshold."""
        from rdagent.scenarios.qlib.local.feedback_integrator import MLFeedbackMixin

        class MockParent:
            def __init__(self):
                pass

        class TestMixin(MLFeedbackMixin, MockParent):
            pass

        mixin = TestMixin(ml_train_interval=500)
        mixin._last_ml_train_factor = 0

        assert mixin._should_trigger_ml_train(500) is True
        assert mixin._should_trigger_ml_train(499) is False
        assert mixin._should_trigger_ml_train(1000) is True

    def test_no_duplicate_trigger(self):
        """Test that duplicate triggers are prevented."""
        from rdagent.scenarios.qlib.local.feedback_integrator import MLFeedbackMixin

        class MockParent:
            def __init__(self):
                pass

        class TestMixin(MLFeedbackMixin, MockParent):
            pass

        mixin = TestMixin(ml_train_interval=500)
        mixin._last_ml_train_factor = 500

        # Should not trigger again at 500
        assert mixin._should_trigger_ml_train(500) is False
        # Should trigger at 1000
        assert mixin._should_trigger_ml_train(1000) is True

    def test_strategy_gen_trigger(self):
        """Test strategy generation trigger logic."""
        from rdagent.scenarios.qlib.local.feedback_integrator import MLFeedbackMixin

        class MockParent:
            def __init__(self):
                pass

        class TestMixin(MLFeedbackMixin, MockParent):
            pass

        mixin = TestMixin(strategy_gen_interval=1000)
        mixin._last_strategy_gen_factor = 0

        assert mixin._should_trigger_strategy_gen(1000) is True
        assert mixin._should_trigger_strategy_gen(999) is False

    def test_portfolio_opt_trigger(self):
        """Test portfolio optimization trigger logic."""
        from rdagent.scenarios.qlib.local.feedback_integrator import MLFeedbackMixin

        class MockParent:
            def __init__(self):
                pass

        class TestMixin(MLFeedbackMixin, MockParent):
            pass

        mixin = TestMixin(portfolio_opt_interval=2000)
        mixin._last_portfolio_opt_factor = 0

        assert mixin._should_trigger_portfolio_opt(2000) is True
        assert mixin._should_trigger_portfolio_opt(1999) is False


class TestFactorCounting:
    """Test factor counting methods."""

    def test_count_from_results_dir(self, mock_project_root, mock_factor_data):
        """Test counting factors from results directory."""
        from rdagent.scenarios.qlib.local.feedback_integrator import MLFeedbackMixin

        # Write test factor files
        for i in range(5):
            factor_file = mock_project_root / "results" / "factors" / f"factor_{i}.json"
            data = mock_factor_data.copy()
            data["name"] = f"factor_{i}"
            data["ic"] = 0.1 + i * 0.01
            with open(factor_file, "w") as f:
                json.dump(data, f)

        class MockParent:
            def __init__(self):
                pass

        class TestMixin(MLFeedbackMixin, MockParent):
            def _get_project_root(self):
                return mock_project_root

        mixin = TestMixin()
        count = mixin._count_factors_from_results()

        assert count == 5

    def test_count_skips_failed_factors(self, mock_project_root, mock_factor_data):
        """Test that failed factors are not counted."""
        from rdagent.scenarios.qlib.local.feedback_integrator import MLFeedbackMixin

        # Write successful factors
        for i in range(3):
            factor_file = mock_project_root / "results" / "factors" / f"success_{i}.json"
            data = mock_factor_data.copy()
            with open(factor_file, "w") as f:
                json.dump(data, f)

        # Write failed factors
        for i in range(2):
            factor_file = mock_project_root / "results" / "factors" / f"failed_{i}.json"
            data = mock_factor_data.copy()
            data["status"] = "failed"
            data["ic"] = None
            with open(factor_file, "w") as f:
                json.dump(data, f)

        class MockParent:
            def __init__(self):
                pass

        class TestMixin(MLFeedbackMixin, MockParent):
            def _get_project_root(self):
                return mock_project_root

        mixin = TestMixin()
        count = mixin._count_factors_from_results()

        assert count == 3  # Only successful factors

    def test_count_empty_directory(self, mock_project_root):
        """Test counting with empty factors directory."""
        from rdagent.scenarios.qlib.local.feedback_integrator import MLFeedbackMixin

        class MockParent:
            def __init__(self):
                pass

        class TestMixin(MLFeedbackMixin, MockParent):
            def _get_project_root(self):
                return mock_project_root

        mixin = TestMixin()
        count = mixin._count_factors_from_results()

        assert count == 0


class TestFeatureImportance:
    """Test feature importance extraction and prompt suggestions."""

    def test_generate_prompt_suggestions_top_features(self, mock_importance_data):
        """Test prompt suggestions from feature importance."""
        from rdagent.scenarios.qlib.local.feedback_integrator import MLFeedbackMixin

        class MockParent:
            def __init__(self):
                pass

        class TestMixin(MLFeedbackMixin, MockParent):
            pass

        mixin = TestMixin()
        suggestions = mixin._generate_prompt_suggestions(mock_importance_data)

        assert len(suggestions) >= 1
        # Should mention top features
        assert any("momentum_5d" in s for s in suggestions)

    def test_generate_suggestions_low_performing_features(self, mock_importance_data):
        """Test suggestions for avoiding low-performing features."""
        from rdagent.scenarios.qlib.local.feedback_integrator import MLFeedbackMixin

        class MockParent:
            def __init__(self):
                pass

        class TestMixin(MLFeedbackMixin, MockParent):
            pass

        mixin = TestMixin()
        suggestions = mixin._generate_prompt_suggestions(mock_importance_data)

        # Should suggest avoiding low-performing features
        assert any("Avoid" in s or "avoid" in s or "reduce" in s.lower() for s in suggestions)

    def test_suggestions_empty_importance(self):
        """Test suggestions with empty importance data."""
        from rdagent.scenarios.qlib.local.feedback_integrator import MLFeedbackMixin

        class MockParent:
            def __init__(self):
                pass

        class TestMixin(MLFeedbackMixin, MockParent):
            pass

        mixin = TestMixin()
        suggestions = mixin._generate_prompt_suggestions({"importance": {}})

        assert len(suggestions) == 1
        assert "No feature importance" in suggestions[0]

    def test_suggestions_low_diversity(self):
        """Test suggestions when factor diversity is low."""
        from rdagent.scenarios.qlib.local.feedback_integrator import MLFeedbackMixin

        class MockParent:
            def __init__(self):
                pass

        class TestMixin(MLFeedbackMixin, MockParent):
            pass

        mixin = TestMixin()
        importance = {
            "importance": {
                "momentum_1d": 0.3,
                "momentum_2d": 0.25,
                "momentum_3d": 0.2,
                "momentum_4d": 0.15,
            }
        }
        suggestions = mixin._generate_prompt_suggestions(importance)

        # Should suggest more diversity
        assert any("diversity" in s.lower() or "Diversity" in s for s in suggestions)


class TestLoadTopFactors:
    """Test loading top factors by IC."""

    def test_load_top_factors(self, mock_project_root, mock_factor_data):
        """Test loading top N factors."""
        from rdagent.scenarios.qlib.local.feedback_integrator import MLFeedbackMixin

        # Write factor files with varying IC
        for i in range(10):
            factor_file = mock_project_root / "results" / "factors" / f"factor_{i}.json"
            data = mock_factor_data.copy()
            data["name"] = f"factor_{i}"
            data["ic"] = 0.01 * (i + 1)  # IC from 0.01 to 0.10
            with open(factor_file, "w") as f:
                json.dump(data, f)

        class MockParent:
            def __init__(self):
                pass

        class TestMixin(MLFeedbackMixin, MockParent):
            def _get_project_root(self):
                return mock_project_root

        mixin = TestMixin()
        top_factors = mixin._load_top_factors(n=5)

        assert len(top_factors) == 5
        # Should be sorted by IC (descending)
        assert top_factors[0]["ic"] >= top_factors[-1]["ic"]

    def test_load_top_factors_empty_dir(self, mock_project_root):
        """Test loading from empty directory."""
        from rdagent.scenarios.qlib.local.feedback_integrator import MLFeedbackMixin

        class MockParent:
            def __init__(self):
                pass

        class TestMixin(MLFeedbackMixin, MockParent):
            def _get_project_root(self):
                return mock_project_root

        mixin = TestMixin()
        top_factors = mixin._load_top_factors(n=5)

        assert top_factors == []


class TestErrorHandling:
    """Test graceful error handling."""

    def test_feedback_with_exception(self):
        """Test that feedback handles exceptions gracefully."""
        from rdagent.scenarios.qlib.local.feedback_integrator import MLFeedbackMixin

        class MockParent:
            def __init__(self):
                pass

            def feedback(self, prev_out):
                return "parent_feedback"

            def _get_factor_count(self):
                raise RuntimeError("Simulated error")

        class TestMixin(MLFeedbackMixin, MockParent):
            pass

        mixin = TestMixin(ml_feedback=True)

        # Should not raise exception
        result = mixin.feedback({})
        assert result == "parent_feedback"


class TestIntegration:
    """Integration tests for full workflow."""

    def test_full_feedback_cycle(self, mock_project_root, mock_factor_data, mock_importance_data):
        """Test complete feedback cycle with triggers."""
        from rdagent.scenarios.qlib.local.feedback_integrator import MLFeedbackMixin

        # Write importance file
        importance_file = mock_project_root / "results" / "models" / "feature_importance.json"
        with open(importance_file, "w") as f:
            json.dump(mock_importance_data, f)

        call_log = []

        class MockParent:
            def __init__(self):
                pass

            def feedback(self, prev_out):
                call_log.append("parent_feedback")
                return "feedback_result"

            def _get_factor_count(self):
                return 500

        class TestMixin(MLFeedbackMixin, MockParent):
            def _get_project_root(self):
                return mock_project_root

            def _count_factors_from_results(self):
                return 500

            def _trigger_ml_training(self, factor_count):
                call_log.append(f"ml_train_{factor_count}")
                self._last_ml_train_factor = factor_count

        mixin = TestMixin(ml_feedback=True, ml_train_interval=500)
        mixin._last_ml_train_factor = 0

        result = mixin.feedback({})

        assert result == "feedback_result"
        assert "parent_feedback" in call_log
        assert "ml_train_500" in call_log
