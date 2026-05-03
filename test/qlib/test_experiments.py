"""Tests for factor_experiment, model_experiment, workspace."""

from __future__ import annotations

import sys
from pathlib import Path
from unittest.mock import MagicMock, patch

import pytest

PROJECT_ROOT = Path(__file__).parent.parent.parent
sys.path.insert(0, str(PROJECT_ROOT))

_MOCK_ENV = "Mock environment info"


# =============================================================================
# QlibFactorScenario
# =============================================================================


class TestQlibFactorScenario:
    @pytest.fixture(autouse=True)
    def _mock_env(self, monkeypatch):
        monkeypatch.setattr(
            "rdagent.scenarios.qlib.experiment.factor_experiment.get_runtime_environment_by_env",
            lambda env: _MOCK_ENV,
        )
        monkeypatch.setattr(
            "rdagent.scenarios.qlib.experiment.factor_experiment.get_factor_env",
            lambda: MagicMock(),
        )

    def test_background_returns_string(self):
        from rdagent.scenarios.qlib.experiment.factor_experiment import QlibFactorScenario
        scen = QlibFactorScenario()
        assert isinstance(scen.background, str)
        assert len(scen.background) > 0

    def test_get_source_data_desc(self):
        from rdagent.scenarios.qlib.experiment.factor_experiment import QlibFactorScenario
        scen = QlibFactorScenario()
        desc = scen.get_source_data_desc()
        assert isinstance(desc, str)

    def test_output_format(self):
        from rdagent.scenarios.qlib.experiment.factor_experiment import QlibFactorScenario
        scen = QlibFactorScenario()
        assert isinstance(scen.output_format, str)

    def test_interface(self):
        from rdagent.scenarios.qlib.experiment.factor_experiment import QlibFactorScenario
        scen = QlibFactorScenario()
        assert isinstance(scen.interface, str)

    def test_simulator(self):
        from rdagent.scenarios.qlib.experiment.factor_experiment import QlibFactorScenario
        scen = QlibFactorScenario()
        assert isinstance(scen.simulator, str)

    def test_rich_style_description(self):
        from rdagent.scenarios.qlib.experiment.factor_experiment import QlibFactorScenario
        scen = QlibFactorScenario()
        assert isinstance(scen.rich_style_description, str)

    def test_experiment_setting(self):
        from rdagent.scenarios.qlib.experiment.factor_experiment import QlibFactorScenario
        scen = QlibFactorScenario()
        assert isinstance(scen.experiment_setting, str)

    def test_get_scenario_all_desc(self):
        from rdagent.scenarios.qlib.experiment.factor_experiment import QlibFactorScenario
        scen = QlibFactorScenario()
        desc = scen.get_scenario_all_desc()
        assert "Background" in desc
        assert "source data" in desc.lower()
        assert "interface" in desc.lower()
        assert "simulator" in desc.lower()

    def test_get_scenario_all_desc_simple_background(self):
        from rdagent.scenarios.qlib.experiment.factor_experiment import QlibFactorScenario
        scen = QlibFactorScenario()
        desc = scen.get_scenario_all_desc(simple_background=True)
        assert "Background" in desc
        # simple_background returns ONLY background, without interface/simulator sections

    def test_get_runtime_environment(self):
        from rdagent.scenarios.qlib.experiment.factor_experiment import QlibFactorScenario
        scen = QlibFactorScenario()
        env = scen.get_runtime_environment()
        assert env == _MOCK_ENV


# =============================================================================
# QlibModelScenario
# =============================================================================


class TestQlibModelScenario:
    @pytest.fixture(autouse=True)
    def _mock_env(self, monkeypatch):
        monkeypatch.setattr(
            "rdagent.scenarios.qlib.experiment.model_experiment.get_runtime_environment_by_env",
            lambda env: _MOCK_ENV,
        )
        monkeypatch.setattr(
            "rdagent.scenarios.qlib.experiment.model_experiment.get_model_env",
            lambda: MagicMock(),
        )

    def test_background_returns_string(self):
        from rdagent.scenarios.qlib.experiment.model_experiment import QlibModelScenario
        scen = QlibModelScenario()
        assert isinstance(scen.background, str)

    def test_source_data_raises_not_implemented(self):
        from rdagent.scenarios.qlib.experiment.model_experiment import QlibModelScenario
        scen = QlibModelScenario()
        with pytest.raises(NotImplementedError):
            _ = scen.source_data

    def test_output_format(self):
        from rdagent.scenarios.qlib.experiment.model_experiment import QlibModelScenario
        scen = QlibModelScenario()
        assert isinstance(scen.output_format, str)

    def test_interface(self):
        from rdagent.scenarios.qlib.experiment.model_experiment import QlibModelScenario
        scen = QlibModelScenario()
        assert isinstance(scen.interface, str)

    def test_simulator(self):
        from rdagent.scenarios.qlib.experiment.model_experiment import QlibModelScenario
        scen = QlibModelScenario()
        assert isinstance(scen.simulator, str)

    def test_rich_style_description(self):
        from rdagent.scenarios.qlib.experiment.model_experiment import QlibModelScenario
        scen = QlibModelScenario()
        assert isinstance(scen.rich_style_description, str)

    def test_experiment_setting(self):
        from rdagent.scenarios.qlib.experiment.model_experiment import QlibModelScenario
        scen = QlibModelScenario()
        assert isinstance(scen.experiment_setting, str)

    def test_get_scenario_all_desc(self):
        from rdagent.scenarios.qlib.experiment.model_experiment import QlibModelScenario
        scen = QlibModelScenario()
        desc = scen.get_scenario_all_desc()
        assert "Background" in desc
        assert "interface" in desc.lower()
        assert "simulator" in desc.lower()

    def test_get_runtime_environment(self):
        from rdagent.scenarios.qlib.experiment.model_experiment import QlibModelScenario
        scen = QlibModelScenario()
        env = scen.get_runtime_environment()
        assert env == _MOCK_ENV


# =============================================================================
# QlibFactorExperiment
# =============================================================================


class TestQlibFactorExperiment:
    def test_init_with_subtasks(self):
        from rdagent.scenarios.qlib.experiment.factor_experiment import QlibFactorExperiment
        from rdagent.core.experiment import Task
        exp = QlibFactorExperiment(sub_tasks=[Task(name="t1")])
        assert exp.stdout == ""
        assert exp.base_features == {}
        assert exp.experiment_workspace is not None

    def test_stdout_settable(self):
        from rdagent.scenarios.qlib.experiment.factor_experiment import QlibFactorExperiment
        from rdagent.core.experiment import Task
        exp = QlibFactorExperiment(sub_tasks=[Task(name="t1")])
        exp.stdout = "test output"
        assert exp.stdout == "test output"

    def test_base_features_default_empty(self):
        from rdagent.scenarios.qlib.experiment.factor_experiment import QlibFactorExperiment
        from rdagent.core.experiment import Task
        exp = QlibFactorExperiment(sub_tasks=[Task(name="t1")])
        assert exp.base_feature_codes == {}


# =============================================================================
# QlibModelExperiment
# =============================================================================


class TestQlibModelExperiment:
    def test_init_with_subtasks(self):
        from rdagent.scenarios.qlib.experiment.model_experiment import QlibModelExperiment
        from rdagent.core.experiment import Task
        exp = QlibModelExperiment(sub_tasks=[Task(name="t1")])
        assert exp.stdout == ""
        assert exp.base_features == {}
        assert exp.experiment_workspace is not None

    def test_stdout_settable(self):
        from rdagent.scenarios.qlib.experiment.model_experiment import QlibModelExperiment
        from rdagent.core.experiment import Task
        exp = QlibModelExperiment(sub_tasks=[Task(name="t1")])
        exp.stdout = "model output"
        assert exp.stdout == "model output"
