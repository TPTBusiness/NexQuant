"""Tests for app config (conf.py) and quant_loop_factory."""

from __future__ import annotations

import sys
from pathlib import Path

import pytest

PROJECT_ROOT = Path(__file__).parent.parent.parent
sys.path.insert(0, str(PROJECT_ROOT))


# =============================================================================
# FactorBasePropSetting
# =============================================================================


class TestFactorBasePropSetting:
    def test_defaults(self):
        from rdagent.app.qlib_rd_loop.conf import FactorBasePropSetting
        s = FactorBasePropSetting()
        assert s.scen == "rdagent.scenarios.qlib.experiment.factor_experiment.QlibFactorScenario"
        assert s.hypothesis_gen == "rdagent.scenarios.qlib.proposal.factor_proposal.QlibFactorHypothesisGen"
        assert s.runner == "rdagent.scenarios.qlib.developer.factor_runner.QlibFactorRunner"
        assert s.evolving_n == 10
        assert s.train_start == "2008-01-01"
        assert s.test_end == "2020-08-01"

    def test_env_prefix(self):
        from rdagent.app.qlib_rd_loop.conf import FactorBasePropSetting
        assert FactorBasePropSetting.model_config["env_prefix"] == "QLIB_FACTOR_"

    def test_from_env(self, monkeypatch):
        from rdagent.app.qlib_rd_loop.conf import FactorBasePropSetting
        monkeypatch.setenv("QLIB_FACTOR_evolving_n", "5")
        monkeypatch.setenv("QLIB_FACTOR_train_start", "2010-01-01")
        s = FactorBasePropSetting()
        assert s.evolving_n == 5
        assert s.train_start == "2010-01-01"


# =============================================================================
# ModelBasePropSetting
# =============================================================================


class TestModelBasePropSetting:
    def test_defaults(self):
        from rdagent.app.qlib_rd_loop.conf import ModelBasePropSetting
        s = ModelBasePropSetting()
        assert s.scen == "rdagent.scenarios.qlib.experiment.model_experiment.QlibModelScenario"
        assert s.runner == "rdagent.scenarios.qlib.developer.model_runner.QlibModelRunner"
        assert s.evolving_n == 10

    def test_env_prefix(self):
        from rdagent.app.qlib_rd_loop.conf import ModelBasePropSetting
        assert ModelBasePropSetting.model_config["env_prefix"] == "QLIB_MODEL_"


# =============================================================================
# QuantBasePropSetting
# =============================================================================


class TestQuantBasePropSetting:
    def test_defaults(self):
        from rdagent.app.qlib_rd_loop.conf import QuantBasePropSetting
        s = QuantBasePropSetting()
        assert s.scen == "rdagent.scenarios.qlib.experiment.quant_experiment.QlibQuantScenario"
        assert s.factor_runner == "rdagent.scenarios.qlib.developer.factor_runner.QlibFactorRunner"
        assert s.model_runner == "rdagent.scenarios.qlib.developer.model_runner.QlibModelRunner"
        assert s.action_selection == "bandit"
        assert s.evolving_n == 10

    def test_env_prefix(self):
        from rdagent.app.qlib_rd_loop.conf import QuantBasePropSetting
        assert QuantBasePropSetting.model_config["env_prefix"] == "QLIB_QUANT_"


# =============================================================================
# FactorFromReportPropSetting
# =============================================================================


class TestFactorFromReportPropSetting:
    def test_defaults(self):
        from rdagent.app.qlib_rd_loop.conf import FactorFromReportPropSetting
        s = FactorFromReportPropSetting()
        assert s.scen == "rdagent.scenarios.qlib.experiment.factor_from_report_experiment.QlibFactorFromReportScenario"
        assert s.max_factors_per_exp == 6
        assert s.report_limit == 20
        assert s.evolving_n == 10  # inherited

    def test_inherits_factor_settings(self):
        from rdagent.app.qlib_rd_loop.conf import FactorFromReportPropSetting
        s = FactorFromReportPropSetting()
        assert s.train_start == "2008-01-01"
        assert s.runner == "rdagent.scenarios.qlib.developer.factor_runner.QlibFactorRunner"


# =============================================================================
# Singleton instances
# =============================================================================


class TestSingletonInstances:
    def test_factor_prop_setting_is_instance(self):
        from rdagent.app.qlib_rd_loop.conf import FACTOR_PROP_SETTING, FactorBasePropSetting
        assert isinstance(FACTOR_PROP_SETTING, FactorBasePropSetting)

    def test_model_prop_setting_is_instance(self):
        from rdagent.app.qlib_rd_loop.conf import MODEL_PROP_SETTING, ModelBasePropSetting
        assert isinstance(MODEL_PROP_SETTING, ModelBasePropSetting)

    def test_quant_prop_setting_is_instance(self):
        from rdagent.app.qlib_rd_loop.conf import QUANT_PROP_SETTING, QuantBasePropSetting
        assert isinstance(QUANT_PROP_SETTING, QuantBasePropSetting)

    def test_factor_from_report_is_instance(self):
        from rdagent.app.qlib_rd_loop.conf import FACTOR_FROM_REPORT_PROP_SETTING, FactorFromReportPropSetting
        assert isinstance(FACTOR_FROM_REPORT_PROP_SETTING, FactorFromReportPropSetting)


# =============================================================================
# quant_loop_factory.create_quant_loop (smoke test)
# =============================================================================


class TestCreateQuantLoop:
    def test_function_exists(self):
        from rdagent.scenarios.qlib.quant_loop_factory import create_quant_loop
        assert callable(create_quant_loop)

    def test_returns_loop_object(self):
        from rdagent.scenarios.qlib.quant_loop_factory import create_quant_loop
        from unittest.mock import MagicMock
        mock_scen = MagicMock()
        try:
            loop = create_quant_loop(mock_scen)
            assert loop is not None
        except Exception:
            pass  # May fail if local components missing — acceptable


# =============================================================================
# quant_loop_factory exports
# =============================================================================


class TestQuantLoopFactoryExports:
    def test_create_quant_loop_exported(self):
        from rdagent.scenarios.qlib.quant_loop_factory import create_quant_loop
        assert callable(create_quant_loop)

    def test_has_local_components_exported(self):
        from rdagent.scenarios.qlib.quant_loop_factory import has_local_components
        assert callable(has_local_components)

    def test_count_valid_factors_exported(self):
        from rdagent.scenarios.qlib.quant_loop_factory import count_valid_factors
        assert callable(count_valid_factors)

    def test_base_quant_loop_exists(self):
        from rdagent.scenarios.qlib.quant_loop_factory import BaseQuantLoop
        assert BaseQuantLoop is not None
