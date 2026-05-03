"""Tests for LLM-dependent components with mock backends."""

from __future__ import annotations

import sys
from pathlib import Path
from unittest.mock import MagicMock, patch

import numpy as np
import pytest

PROJECT_ROOT = Path(__file__).parent.parent.parent
sys.path.insert(0, str(PROJECT_ROOT))


# =============================================================================
# ModelCoSTEEREvaluator (model_coder/evaluators.py)
# =============================================================================


class TestModelCoSTEEREvaluator:
    def test_init(self):
        from rdagent.components.coder.model_coder.evaluators import ModelCoSTEEREvaluator
        eva = ModelCoSTEEREvaluator(scen=MagicMock())
        assert eva.scen is not None

    def test_returns_cached_feedback(self):
        from rdagent.components.coder.model_coder.evaluators import ModelCoSTEEREvaluator
        eva = ModelCoSTEEREvaluator(scen=MagicMock())
        qk = MagicMock()
        qk.success_task_to_knowledge_dict = {
            "info_task": MagicMock(feedback="cached_fb"),
        }
        t = MagicMock()
        t.get_task_information.return_value = "info_task"
        fb = eva.evaluate(target_task=t, implementation=None, gt_implementation=None, queried_knowledge=qk)
        assert fb == "cached_fb"

    def test_returns_failed_feedback(self):
        from rdagent.components.coder.model_coder.evaluators import ModelCoSTEEREvaluator
        eva = ModelCoSTEEREvaluator(scen=MagicMock())
        qk = MagicMock()
        qk.success_task_to_knowledge_dict = {}
        qk.failed_task_info_set = {"info_task"}
        t = MagicMock()
        t.get_task_information.return_value = "info_task"
        fb = eva.evaluate(target_task=t, implementation=None, gt_implementation=None, queried_knowledge=qk)
        assert fb.final_decision is False
        assert "failed too many times" in fb.execution_feedback

    def test_raises_on_wrong_task_type(self):
        from rdagent.components.coder.model_coder.evaluators import ModelCoSTEEREvaluator
        eva = ModelCoSTEEREvaluator(scen=MagicMock())
        qk = MagicMock()
        qk.success_task_to_knowledge_dict = {}
        qk.failed_task_info_set = set()
        t = MagicMock()
        t.get_task_information.return_value = "new_task"
        with pytest.raises(TypeError, match="Expected ModelTask"):
            eva.evaluate(target_task=t, implementation=None, gt_implementation=None, queried_knowledge=qk)

    def test_raises_on_wrong_workspace_type(self):
        from rdagent.components.coder.model_coder.evaluators import ModelCoSTEEREvaluator
        from rdagent.components.coder.model_coder.model import ModelTask

        eva = ModelCoSTEEREvaluator(scen=MagicMock())
        qk = MagicMock()
        qk.success_task_to_knowledge_dict = {}
        qk.failed_task_info_set = set()

        t = ModelTask(
            name="m1", description="d", architecture="LSTM",
            hyperparameters={}, training_hyperparameters={},
        )
        t.get_task_information = MagicMock(return_value="new")

        with pytest.raises(TypeError, match="Expected ModelFBWorkspace"):
            eva.evaluate(target_task=t, implementation="not_a_workspace", gt_implementation=None, queried_knowledge=qk)


# =============================================================================
# FactorMultiProcessEvolvingStrategy (factor_coder/evolving_strategy.py)
# =============================================================================


class TestFactorEvolvingStrategy:
    def test_init_sets_fields(self):
        from rdagent.components.coder.factor_coder.evolving_strategy import FactorMultiProcessEvolvingStrategy
        strat = FactorMultiProcessEvolvingStrategy(scen=MagicMock(), settings=MagicMock())
        assert strat.num_loop == 0
        assert strat.haveSelected is False
        assert strat.improve_mode is False

    def test_assign_code_list_to_evo_str_input(self):
        """assign_code_list_to_evo handles string code (not dict)."""
        from rdagent.components.coder.factor_coder.evolving_strategy import FactorMultiProcessEvolvingStrategy
        from rdagent.components.coder.CoSTEER.evolvable_subjects import EvolvingItem
        from rdagent.components.coder.factor_coder.factor import FactorTask, FactorFBWorkspace

        strat = FactorMultiProcessEvolvingStrategy(scen=MagicMock(), settings=MagicMock())
        evo = EvolvingItem(sub_tasks=[FactorTask("f1", "desc", "formula")])
        evo.sub_workspace_list = [None]

        with patch(
            "rdagent.components.coder.factor_coder.evolving_strategy.auto_fix_factor_code",
            return_value="fixed_code",
        ):
            strat.assign_code_list_to_evo(["raw_code"], evo)
            assert evo.sub_workspace_list[0] is not None
            # Should be a FactorFBWorkspace
            from rdagent.components.coder.factor_coder.factor import FactorFBWorkspace
            assert isinstance(evo.sub_workspace_list[0], FactorFBWorkspace)

    def test_assign_code_list_to_evo_dict_input(self):
        """assign_code_list_to_evo handles dict code."""
        from rdagent.components.coder.factor_coder.evolving_strategy import FactorMultiProcessEvolvingStrategy
        from rdagent.components.coder.CoSTEER.evolvable_subjects import EvolvingItem
        from rdagent.components.coder.factor_coder.factor import FactorTask

        strat = FactorMultiProcessEvolvingStrategy(scen=MagicMock(), settings=MagicMock())
        evo = EvolvingItem(sub_tasks=[FactorTask("f1", "desc", "formula")])
        evo.sub_workspace_list = [None]

        with patch(
            "rdagent.components.coder.factor_coder.evolving_strategy.auto_fix_factor_code",
            return_value="fixed",
        ):
            strat.assign_code_list_to_evo([{"factor.py": "code", "utils.py": "util_code"}], evo)
            assert evo.sub_workspace_list[0] is not None

    def test_assign_code_list_skips_none(self):
        from rdagent.components.coder.factor_coder.evolving_strategy import FactorMultiProcessEvolvingStrategy
        from rdagent.components.coder.CoSTEER.evolvable_subjects import EvolvingItem
        from rdagent.components.coder.factor_coder.factor import FactorTask

        strat = FactorMultiProcessEvolvingStrategy(scen=MagicMock(), settings=MagicMock())
        evo = EvolvingItem(sub_tasks=[FactorTask("f1", "desc", "formula")])
        evo.sub_workspace_list = [None]
        strat.assign_code_list_to_evo([None], evo)
        assert evo.sub_workspace_list[0] is None  # unchanged


# =============================================================================
# Eurusd_llm prompt class (eurusd_llm.py)
# =============================================================================


class TestEurusdLLM:
    def test_eurusd_llm_importable(self):
        from rdagent.components.coder.factor_coder import eurusd_llm
        assert eurusd_llm is not None

    def test_eurusd_risk_importable(self):
        from rdagent.components.coder.factor_coder import eurusd_risk
        assert eurusd_risk is not None

    def test_eurusd_regime_importable(self):
        from rdagent.components.coder.factor_coder import eurusd_regime
        assert eurusd_regime is not None

    def test_eurusd_debate_importable(self):
        from rdagent.components.coder.factor_coder import eurusd_debate
        assert eurusd_debate is not None

    # eurusd_macro needs yfinance (optional)
    # eurusd_memory needs rank_bm25 (optional)
    # eurusd_reflection needs eurusd_memory (chain dependency)


# =============================================================================
# model_coder/evolving_strategy.py import
# =============================================================================


class TestModelEvolvingStrategy:
    def test_model_evolving_strategy_importable(self):
        from rdagent.components.coder.model_coder import evolving_strategy
        assert evolving_strategy is not None


# =============================================================================
# model_coder/eva_utils.py ModelCodeEvaluator + ModelFinalEvaluator
# =============================================================================


class TestModelCodeFinalEvaluators:
    def test_model_code_evaluator_init(self):
        from rdagent.components.coder.model_coder.eva_utils import ModelCodeEvaluator
        eva = ModelCodeEvaluator(scen=MagicMock())
        assert eva.scen is not None

    def test_model_final_evaluator_init(self):
        from rdagent.components.coder.model_coder.eva_utils import ModelFinalEvaluator
        eva = ModelFinalEvaluator(scen=MagicMock())
        assert eva.scen is not None
