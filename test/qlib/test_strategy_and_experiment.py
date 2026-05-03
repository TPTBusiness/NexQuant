"""Tests for strategy_builder, quant_experiment."""

from __future__ import annotations

import sys
from pathlib import Path
from unittest.mock import MagicMock, patch

import numpy as np
import pytest

PROJECT_ROOT = Path(__file__).parent.parent.parent
sys.path.insert(0, str(PROJECT_ROOT))


# =============================================================================
# StrategyCombinator
# =============================================================================


class TestStrategyCombinator:
    def _make_factors(self, n=4):
        return [
            {"factor_name": f"f{i}", "ic": 0.1 * i, "category": ["mom", "vol", "mom", "vol"][i % 4]}
            for i in range(n)
        ]

    def test_generate_all_pairs(self):
        from rdagent.scenarios.qlib.developer.strategy_builder import StrategyCombinator
        factors = self._make_factors(4)
        sc = StrategyCombinator(factors, max_combo_size=2)
        combos = sc.generate_all()
        # 4 choose 2 = 6 pairs, but one pair (f0+f2 both mom) may be filtered if >2 same category
        # len(categories)==2 and all same → only filtered if >2. With 2 factors, not filtered.
        assert len(combos) == 6
        for c in combos:
            assert c["size"] == 2
            assert len(c["factors"]) == 2
            assert "avg_ic" in c

    def test_generate_all_triplets(self):
        from rdagent.scenarios.qlib.developer.strategy_builder import StrategyCombinator
        factors = self._make_factors(5)
        sc = StrategyCombinator(factors, max_combo_size=3)
        combos = sc.generate_all()
        # 5C2 + 5C3 = 10 + 10 = 20, but f0+f2+f4 (all mom) is filtered
        # because len(set) == 1 and len(categories) > 2
        assert len(combos) == 19

    def test_sorted_by_avg_ic_desc(self):
        from rdagent.scenarios.qlib.developer.strategy_builder import StrategyCombinator
        factors = self._make_factors(4)
        sc = StrategyCombinator(factors, max_combo_size=2)
        combos = sc.generate_all()
        for i in range(len(combos) - 1):
            assert combos[i]["avg_ic"] >= combos[i + 1]["avg_ic"]

    def test_empty_factors(self):
        from rdagent.scenarios.qlib.developer.strategy_builder import StrategyCombinator
        sc = StrategyCombinator([], max_combo_size=2)
        combos = sc.generate_all()
        assert combos == []

    def test_max_combo_1_returns_empty(self):
        from rdagent.scenarios.qlib.developer.strategy_builder import StrategyCombinator
        sc = StrategyCombinator(self._make_factors(3), max_combo_size=1)
        combos = sc.generate_all()
        assert combos == []  # min size is 2

    def test_generate_diversified(self):
        from rdagent.scenarios.qlib.developer.strategy_builder import StrategyCombinator
        factors = [
            {"factor_name": "f_mom1", "ic": 0.05, "category": "momentum"},
            {"factor_name": "f_mom2", "ic": 0.03, "category": "momentum"},
            {"factor_name": "f_vol1", "ic": 0.04, "category": "volatility"},
            {"factor_name": "f_rev1", "ic": 0.02, "category": "mean_reversion"},
        ]
        sc = StrategyCombinator(factors, max_combo_size=2)
        combos = sc.generate_diversified(target_size=3)
        assert len(combos) >= 2  # At least momentum+vol, momentum+rev
        for c in combos:
            assert len(set(c["categories"])) > 1  # Must be cross-category


# =============================================================================
# QlibQuantScenario (quant_experiment.py)
# =============================================================================


class TestQlibQuantScenario:
    def test_background_invalid_tag_raises(self):
        with patch("rdagent.scenarios.qlib.experiment.quant_experiment.get_runtime_environment_by_env",
                   return_value="mock_env"):
            with patch("rdagent.scenarios.qlib.experiment.quant_experiment.get_factor_env",
                       return_value=MagicMock()):
                with patch("rdagent.scenarios.qlib.experiment.quant_experiment.get_model_env",
                           return_value=MagicMock()):
                    from rdagent.scenarios.qlib.experiment.quant_experiment import QlibQuantScenario
                    scen = QlibQuantScenario()
                    with pytest.raises(ValueError, match="tag must be"):
                        scen.background(tag="invalid")

    def test_output_format_invalid_tag_raises(self):
        with patch("rdagent.scenarios.qlib.experiment.quant_experiment.get_runtime_environment_by_env",
                   return_value="mock_env"):
            with patch("rdagent.scenarios.qlib.experiment.quant_experiment.get_factor_env",
                       return_value=MagicMock()):
                with patch("rdagent.scenarios.qlib.experiment.quant_experiment.get_model_env",
                           return_value=MagicMock()):
                    from rdagent.scenarios.qlib.experiment.quant_experiment import QlibQuantScenario
                    scen = QlibQuantScenario()
                    with pytest.raises(ValueError, match="tag must be"):
                        scen.output_format(tag="bad")

    def test_interface_invalid_tag_raises(self):
        with patch("rdagent.scenarios.qlib.experiment.quant_experiment.get_runtime_environment_by_env",
                   return_value="mock_env"):
            with patch("rdagent.scenarios.qlib.experiment.quant_experiment.get_factor_env",
                       return_value=MagicMock()):
                with patch("rdagent.scenarios.qlib.experiment.quant_experiment.get_model_env",
                           return_value=MagicMock()):
                    from rdagent.scenarios.qlib.experiment.quant_experiment import QlibQuantScenario
                    scen = QlibQuantScenario()
                    with pytest.raises(ValueError, match="tag must be"):
                        scen.interface(tag=42)

    def test_simulator_invalid_tag_raises(self):
        with patch("rdagent.scenarios.qlib.experiment.quant_experiment.get_runtime_environment_by_env",
                   return_value="mock_env"):
            with patch("rdagent.scenarios.qlib.experiment.quant_experiment.get_factor_env",
                       return_value=MagicMock()):
                with patch("rdagent.scenarios.qlib.experiment.quant_experiment.get_model_env",
                           return_value=MagicMock()):
                    from rdagent.scenarios.qlib.experiment.quant_experiment import QlibQuantScenario
                    scen = QlibQuantScenario()
                    with pytest.raises(ValueError, match="tag must be"):
                        scen.simulator(tag="unknown")

    def test_get_runtime_environment_invalid_tag_raises(self):
        with patch("rdagent.scenarios.qlib.experiment.quant_experiment.get_runtime_environment_by_env",
                   return_value="mock_env"):
            with patch("rdagent.scenarios.qlib.experiment.quant_experiment.get_factor_env",
                       return_value=MagicMock()):
                with patch("rdagent.scenarios.qlib.experiment.quant_experiment.get_model_env",
                           return_value=MagicMock()):
                    from rdagent.scenarios.qlib.experiment.quant_experiment import QlibQuantScenario
                    scen = QlibQuantScenario()
                    with pytest.raises(ValueError, match="tag must be"):
                        scen.get_runtime_environment(tag="nope")

    def test_get_scenario_all_desc_with_action(self):
        with patch("rdagent.scenarios.qlib.experiment.quant_experiment.get_runtime_environment_by_env",
                   return_value="mock_env"):
            with patch("rdagent.scenarios.qlib.experiment.quant_experiment.get_factor_env",
                       return_value=MagicMock()):
                with patch("rdagent.scenarios.qlib.experiment.quant_experiment.get_model_env",
                           return_value=MagicMock()):
                    from rdagent.scenarios.qlib.experiment.quant_experiment import QlibQuantScenario
                    scen = QlibQuantScenario()
                    desc = scen.get_scenario_all_desc(action="factor")
                    assert "Background" in desc
                    assert "interface" in desc.lower()

    def test_get_scenario_all_desc_simple_background(self):
        with patch("rdagent.scenarios.qlib.experiment.quant_experiment.get_runtime_environment_by_env",
                   return_value="mock_env"):
            with patch("rdagent.scenarios.qlib.experiment.quant_experiment.get_factor_env",
                       return_value=MagicMock()):
                with patch("rdagent.scenarios.qlib.experiment.quant_experiment.get_model_env",
                           return_value=MagicMock()):
                    from rdagent.scenarios.qlib.experiment.quant_experiment import QlibQuantScenario
                    scen = QlibQuantScenario()
                    desc = scen.get_scenario_all_desc(simple_background=True)
                    assert "Background" in desc
                    assert "source" in desc.lower()

    def test_background_tag_factor(self):
        with patch("rdagent.scenarios.qlib.experiment.quant_experiment.get_runtime_environment_by_env",
                   return_value="mock_env"):
            with patch("rdagent.scenarios.qlib.experiment.quant_experiment.get_factor_env",
                       return_value=MagicMock()):
                with patch("rdagent.scenarios.qlib.experiment.quant_experiment.get_model_env",
                           return_value=MagicMock()):
                    from rdagent.scenarios.qlib.experiment.quant_experiment import QlibQuantScenario
                    scen = QlibQuantScenario()
                    bg = scen.background(tag="factor")
                    assert "factor" in bg.lower()
