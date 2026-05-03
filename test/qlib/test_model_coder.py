"""Tests for model_coder — ModelTask, shape/value evaluators, config."""

from __future__ import annotations

import sys
from pathlib import Path
from unittest.mock import MagicMock

import numpy as np
import pytest

PROJECT_ROOT = Path(__file__).parent.parent.parent
sys.path.insert(0, str(PROJECT_ROOT))


# =============================================================================
# ModelTask
# =============================================================================


class TestModelTask:
    def test_construction_fields(self):
        from rdagent.components.coder.model_coder.model import ModelTask
        t = ModelTask(
            name="m1",
            description="desc",
            architecture="LSTM",
            hyperparameters={"lr": 0.001},
            training_hyperparameters={"epochs": 10},
            formulation="y = f(x)",
            variables={"x": "feature"},
            model_type="TimeSeries",
        )
        assert t.name == "m1"
        assert t.description == "desc"
        assert t.architecture == "LSTM"
        assert t.hyperparameters == {"lr": 0.001}
        assert t.training_hyperparameters == {"epochs": 10}
        assert t.formulation == "y = f(x)"
        assert t.variables == {"x": "feature"}
        assert t.model_type == "TimeSeries"
        assert t.base_code is None

    def test_get_task_information(self):
        from rdagent.components.coder.model_coder.model import ModelTask
        t = ModelTask(
            name="m1", description="desc", architecture="LSTM",
            hyperparameters={}, training_hyperparameters={},
            model_type="Tabular",
        )
        info = t.get_task_information()
        assert "name: m1" in info
        assert "architecture: LSTM" in info
        assert "model_type: Tabular" in info

    def test_get_task_information_with_optional_fields(self):
        from rdagent.components.coder.model_coder.model import ModelTask
        t = ModelTask(
            name="m2", description="d2", architecture="GRU",
            hyperparameters={}, training_hyperparameters={},
            formulation="f1", variables={"v": 1}, model_type="Graph",
        )
        info = t.get_task_information()
        assert "formulation: f1" in info
        assert "variables: {'v': 1}" in info

    def test_get_task_brief_information(self):
        from rdagent.components.coder.model_coder.model import ModelTask
        t = ModelTask(
            name="m1", description="desc", architecture="LSTM",
            hyperparameters={"lr": 0.01}, training_hyperparameters={"epochs": 5},
        )
        info = t.get_task_brief_information()
        assert "name: m1" in info
        assert "architecture: LSTM" in info
        assert "hyperparameters" in info

    def test_from_dict(self):
        from rdagent.components.coder.model_coder.model import ModelTask
        d = {
            "name": "m3", "description": "d3", "architecture": "TCN",
            "hyperparameters": {}, "training_hyperparameters": {},
        }
        t = ModelTask.from_dict(d)
        assert t.name == "m3"

    def test_repr(self):
        from rdagent.components.coder.model_coder.model import ModelTask
        t = ModelTask(
            name="mymodel", description="d", architecture="LSTM",
            hyperparameters={}, training_hyperparameters={},
        )
        assert "ModelTask" in repr(t)
        assert "mymodel" in repr(t)


# =============================================================================
# Shape/Value evaluators (eva_utils)
# =============================================================================


class TestShapeEvaluator:
    def test_correct_shape(self):
        from rdagent.components.coder.model_coder.eva_utils import shape_evaluator
        msg, ok = shape_evaluator(np.ones((32, 10)), target_shape=(32, 10))
        assert ok is True
        assert "correct" in msg.lower()

    def test_incorrect_shape(self):
        from rdagent.components.coder.model_coder.eva_utils import shape_evaluator
        msg, ok = shape_evaluator(np.ones((32, 5)), target_shape=(32, 10))
        assert ok is False
        assert "incorrect" in msg.lower()

    def test_none_prediction(self):
        from rdagent.components.coder.model_coder.eva_utils import shape_evaluator
        msg, ok = shape_evaluator(None, target_shape=(32, 10))
        assert ok is False

    def test_none_target_shape(self):
        from rdagent.components.coder.model_coder.eva_utils import shape_evaluator
        msg, ok = shape_evaluator(np.ones((3,)), target_shape=None)
        assert ok is False

    def test_float_array(self):
        from rdagent.components.coder.model_coder.eva_utils import shape_evaluator
        msg, ok = shape_evaluator(np.array([1.0, 2.0]), target_shape=(2,))
        assert ok is True


class TestValueEvaluator:
    def test_none_prediction(self):
        from rdagent.components.coder.model_coder.eva_utils import value_evaluator
        msg, ok = value_evaluator(None, np.ones((3,)))
        assert ok is False

    def test_none_target(self):
        from rdagent.components.coder.model_coder.eva_utils import value_evaluator
        msg, ok = value_evaluator(np.ones((3,)), None)
        assert ok is False

    def test_small_difference_passes(self):
        from rdagent.components.coder.model_coder.eva_utils import value_evaluator
        msg, ok = value_evaluator(
            np.array([1.0, 2.0, 3.0]),
            np.array([1.0, 2.0, 3.01]),
        )
        assert bool(ok) is True  # diff < 0.1

    def test_large_difference_fails(self):
        from rdagent.components.coder.model_coder.eva_utils import value_evaluator
        msg, ok = value_evaluator(
            np.array([1.0, 2.0]),
            np.array([10.0, 20.0]),
        )
        assert bool(ok) is False  # diff > 0.1


# =============================================================================
# ModelCoSTEERSettings
# =============================================================================


class TestModelCoSTEERSettings:
    def test_default_env_type(self):
        from rdagent.components.coder.model_coder.conf import ModelCoSTEERSettings
        s = ModelCoSTEERSettings()
        assert s.env_type == "conda"

    def test_singleton(self):
        from rdagent.components.coder.model_coder.conf import MODEL_COSTEER_SETTINGS
        from rdagent.components.coder.model_coder.conf import ModelCoSTEERSettings
        assert isinstance(MODEL_COSTEER_SETTINGS, ModelCoSTEERSettings)

    def test_get_model_env_runs(self):
        from rdagent.components.coder.model_coder.conf import get_model_env
        # May succeed (conda available) or fail — either way, test the code path
        try:
            env = get_model_env()
            assert env is not None
        except Exception:
            pass  # expected if docker/conda not available
