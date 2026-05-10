"""Deep tests for CoSTEER components: knowledge_management, evaluators, eva_utils, evolving_strategy, auto_fixer."""

from __future__ import annotations

import json
import pickle
import sys
import tempfile
from copy import deepcopy
from pathlib import Path
from typing import Any
from unittest.mock import MagicMock, patch

import numpy as np
import pandas as pd
import pytest

PROJECT_ROOT = Path(__file__).parent.parent.parent
sys.path.insert(0, str(PROJECT_ROOT))


def _make_mock_fb_workspace(codes: str = "print('hello')") -> Any:
    ws = MagicMock()
    ws.all_codes = codes
    ws.copy.return_value = ws
    return ws


def _make_mock_task(task_info: str = "factor task info") -> Any:
    t = MagicMock()
    t.get_task_information.return_value = task_info
    return t


# =============================================================================
# Import safety
# =============================================================================

COSTEER_MODULES = [
    "rdagent.components.coder.CoSTEER.knowledge_management",
    "rdagent.components.coder.CoSTEER.evaluators",
    "rdagent.components.coder.CoSTEER.evolvable_subjects",
    "rdagent.components.coder.CoSTEER.evolving_strategy",
    "rdagent.components.coder.CoSTEER.config",
    "rdagent.components.coder.factor_coder.evolving_strategy",
    "rdagent.components.coder.factor_coder.eva_utils",
    "rdagent.components.coder.factor_coder.auto_fixer",
    "rdagent.components.coder.factor_coder.factor",
    "rdagent.components.coder.factor_coder.config",
    "rdagent.components.knowledge_management.graph",
]


class TestCosteerImports:
    @pytest.mark.parametrize("mod_name", COSTEER_MODULES)
    def test_module_is_importable(self, mod_name: str) -> None:
        """Verify each CoSTEER submodule can be imported without error."""
        import importlib
        mod = importlib.import_module(mod_name)
        assert mod is not None


# =============================================================================
# CoSTEERKnowledge
# =============================================================================


class TestCoSTEERKnowledge:
    def test_construction_stores_task_implementation_feedback(self) -> None:
        """Knowledge stores target_task, implementation, and feedback."""
        from rdagent.components.coder.CoSTEER.knowledge_management import CoSTEERKnowledge
        from rdagent.components.coder.CoSTEER.evaluators import CoSTEERSingleFeedback
        task = _make_mock_task()
        ws = _make_mock_fb_workspace("def f(): pass")
        fb = CoSTEERSingleFeedback(execution="OK", return_checking="pass", code="good", final_decision=True)
        k = CoSTEERKnowledge(target_task=task, implementation=ws, feedback=fb)
        assert k.target_task is task
        assert k.implementation is ws
        assert k.feedback is fb

    def test_get_implementation_and_feedback_str_contains_code(self) -> None:
        """The formatted string includes implementation code and feedback."""
        from rdagent.components.coder.CoSTEER.knowledge_management import CoSTEERKnowledge
        from rdagent.components.coder.CoSTEER.evaluators import CoSTEERSingleFeedback
        task = _make_mock_task()
        ws = _make_mock_fb_workspace("def my_factor(): return df")
        fb = CoSTEERSingleFeedback(execution="ran", return_checking="ok", code="fine", final_decision=True)
        k = CoSTEERKnowledge(target_task=task, implementation=ws, feedback=fb)
        s = k.get_implementation_and_feedback_str()
        assert "def my_factor" in s

    def test_copy_implementation_is_called(self) -> None:
        """Knowledge copies the implementation workspace in __init__."""
        from rdagent.components.coder.CoSTEER.knowledge_management import CoSTEERKnowledge
        from rdagent.components.coder.CoSTEER.evaluators import CoSTEERSingleFeedback
        task = _make_mock_task()
        ws = _make_mock_fb_workspace("code")
        ws._copy_called = False
        def side_effect():
            ws._copy_called = True
            return ws
        ws.copy = MagicMock(side_effect=side_effect)
        fb = CoSTEERSingleFeedback(execution="x", return_checking="x", code="x", final_decision=False)
        CoSTEERKnowledge(target_task=task, implementation=ws, feedback=fb)
        assert ws._copy_called


# =============================================================================
# CoSTEERRAGStrategy — load, init, dump
# =============================================================================


class TestCoSTEERRAGStrategy:
    def test_load_or_init_creates_v2_when_no_file(self) -> None:
        """Creates a fresh CoSTEERKnowledgeBaseV2 when no former file exists."""
        from rdagent.components.coder.CoSTEER.knowledge_management import (
            CoSTEERRAGStrategyV2,
            CoSTEERKnowledgeBaseV2,
        )
        strategy = CoSTEERRAGStrategyV2(settings=MagicMock(), dump_knowledge_base_path=Path("/nonexistent_12345.pkl"))
        kb = strategy.load_or_init_knowledge_base(former_knowledge_base_path=None, evolving_version=2)
        assert isinstance(kb, CoSTEERKnowledgeBaseV2)

    def test_load_or_init_creates_v1_when_no_file(self) -> None:
        """Creates a fresh CoSTEERKnowledgeBaseV1 when no former file exists."""
        from rdagent.components.coder.CoSTEER.knowledge_management import (
            CoSTEERRAGStrategyV1,
            CoSTEERKnowledgeBaseV1,
        )
        strategy = CoSTEERRAGStrategyV1(settings=MagicMock(), dump_knowledge_base_path=None)
        kb = strategy.load_or_init_knowledge_base(former_knowledge_base_path=None, evolving_version=1)
        assert isinstance(kb, CoSTEERKnowledgeBaseV1)

    def test_dump_knowledge_base_creates_dir_and_file(self) -> None:
        """dump_knowledge_base writes pickle file when path is set."""
        from rdagent.components.coder.CoSTEER.knowledge_management import (
            CoSTEERRAGStrategyV2,
            CoSTEERKnowledgeBaseV2,
        )
        with tempfile.TemporaryDirectory() as tmpdir:
            dump_path = Path(tmpdir) / "sub" / "kb.pkl"
            strategy = CoSTEERRAGStrategyV2(settings=MagicMock(), dump_knowledge_base_path=dump_path)
            strategy.knowledgebase = CoSTEERKnowledgeBaseV2()
            strategy.dump_knowledge_base()
            assert dump_path.exists()

    def test_dump_knowledge_base_skips_when_path_is_none(self) -> None:
        """No error when dump path is None."""
        from rdagent.components.coder.CoSTEER.knowledge_management import CoSTEERRAGStrategyV2
        strategy = CoSTEERRAGStrategyV2(settings=MagicMock(), dump_knowledge_base_path=None)
        strategy.dump_knowledge_base()

    def test_load_dumped_knowledge_base_restores(self) -> None:
        """Loading from a dumped file restores the knowledge base."""
        from rdagent.components.coder.CoSTEER.knowledge_management import (
            CoSTEERRAGStrategyV2,
            CoSTEERKnowledgeBaseV2,
        )
        with tempfile.TemporaryDirectory() as tmpdir:
            dump_path = Path(tmpdir) / "kb.pkl"
            s1 = CoSTEERRAGStrategyV2(settings=MagicMock(), dump_knowledge_base_path=dump_path)
            s1.knowledgebase = CoSTEERKnowledgeBaseV2()
            s1.knowledgebase.success_task_to_knowledge_dict["k"] = "v"
            s1.dump_knowledge_base()
            s2 = CoSTEERRAGStrategyV2(settings=MagicMock(), dump_knowledge_base_path=dump_path)
            s2.load_dumped_knowledge_base()
            assert s2.knowledgebase is not None
            assert s2.knowledgebase.success_task_to_knowledge_dict["k"] == "v"


# =============================================================================
# CoSTEERQueriedKnowledge and variants
# =============================================================================


class TestCoSTEERQueriedKnowledge:
    def test_default_construction_empty_dicts(self) -> None:
        from rdagent.components.coder.CoSTEER.knowledge_management import CoSTEERQueriedKnowledge
        qk = CoSTEERQueriedKnowledge()
        assert qk.success_task_to_knowledge_dict == {}
        assert qk.failed_task_info_set == set()

    def test_construction_with_data(self) -> None:
        from rdagent.components.coder.CoSTEER.knowledge_management import CoSTEERQueriedKnowledge
        qk = CoSTEERQueriedKnowledge(
            success_task_to_knowledge_dict={"a": 1}, failed_task_info_set={"b"},
        )
        assert qk.success_task_to_knowledge_dict == {"a": 1}
        assert qk.failed_task_info_set == {"b"}

    @pytest.mark.parametrize("dict_val", [{}, {"k": None}, {"a": 1, "b": 2}])
    def test_success_task_to_knowledge_dict_variants(self, dict_val: dict) -> None:
        from rdagent.components.coder.CoSTEER.knowledge_management import CoSTEERQueriedKnowledge
        qk = CoSTEERQueriedKnowledge(success_task_to_knowledge_dict=dict_val)
        assert qk.success_task_to_knowledge_dict == dict_val

    @pytest.mark.parametrize("set_val", [set(), {"x"}, {"a", "b", "c"}])
    def test_failed_task_info_set_variants(self, set_val: set) -> None:
        from rdagent.components.coder.CoSTEER.knowledge_management import CoSTEERQueriedKnowledge
        qk = CoSTEERQueriedKnowledge(failed_task_info_set=set_val)
        assert qk.failed_task_info_set == set_val


class TestCoSTEERQueriedKnowledgeV1:
    def test_extra_fields_default_to_empty(self) -> None:
        from rdagent.components.coder.CoSTEER.knowledge_management import CoSTEERQueriedKnowledgeV1
        qk = CoSTEERQueriedKnowledgeV1()
        assert qk.task_to_former_failed_traces == {}
        assert qk.task_to_similar_task_successful_knowledge == {}

    def test_custom_extra_fields(self) -> None:
        from rdagent.components.coder.CoSTEER.knowledge_management import CoSTEERQueriedKnowledgeV1
        qk = CoSTEERQueriedKnowledgeV1(
            task_to_former_failed_traces={"t": []}, task_to_similar_task_successful_knowledge={"t": ["k"]},
        )
        assert qk.task_to_former_failed_traces == {"t": []}

    def test_inherits_from_base_queried_knowledge(self) -> None:
        from rdagent.components.coder.CoSTEER.knowledge_management import (
            CoSTEERQueriedKnowledge, CoSTEERQueriedKnowledgeV1,
        )
        qk = CoSTEERQueriedKnowledgeV1()
        assert isinstance(qk, CoSTEERQueriedKnowledge)


class TestCoSTEERQueriedKnowledgeV2:
    def test_extra_field_defaults_to_empty(self) -> None:
        from rdagent.components.coder.CoSTEER.knowledge_management import CoSTEERQueriedKnowledgeV2
        qk = CoSTEERQueriedKnowledgeV2()
        assert qk.task_to_similar_error_successful_knowledge == {}

    def test_inherits_from_v1(self) -> None:
        from rdagent.components.coder.CoSTEER.knowledge_management import (
            CoSTEERQueriedKnowledgeV1, CoSTEERQueriedKnowledgeV2,
        )
        qk = CoSTEERQueriedKnowledgeV2()
        assert isinstance(qk, CoSTEERQueriedKnowledgeV1)


# =============================================================================
# CoSTEERKnowledgeBaseV1
# =============================================================================


class TestCoSTEERKnowledgeBaseV1:
    def test_default_construction(self) -> None:
        from rdagent.components.coder.CoSTEER.knowledge_management import CoSTEERKnowledgeBaseV1
        kb = CoSTEERKnowledgeBaseV1()
        assert kb.implementation_trace == {}
        assert kb.success_task_info_set == set()
        assert kb.task_to_embedding == {}

    def test_query_raises_not_implemented(self) -> None:
        from rdagent.components.coder.CoSTEER.knowledge_management import CoSTEERKnowledgeBaseV1
        kb = CoSTEERKnowledgeBaseV1()
        with pytest.raises(NotImplementedError):
            kb.query()


# =============================================================================
# CoSTEERKnowledgeBaseV2
# =============================================================================


class TestCoSTEERKnowledgeBaseV2:
    def test_default_construction_has_attributes(self) -> None:
        from rdagent.components.coder.CoSTEER.knowledge_management import CoSTEERKnowledgeBaseV2
        kb = CoSTEERKnowledgeBaseV2()
        assert kb.working_trace_knowledge == {}
        assert kb.working_trace_error_analysis == {}
        assert kb.success_task_to_knowledge_dict == {}
        assert kb.node_to_implementation_knowledge_dict == {}
        assert kb.task_to_component_nodes == {}

    def test_v2_importable_and_instantiable(self) -> None:
        from rdagent.components.coder.CoSTEER.knowledge_management import CoSTEERKnowledgeBaseV2
        kb = CoSTEERKnowledgeBaseV2()
        assert kb is not None
        assert kb.working_trace_knowledge == {}
        assert kb.success_task_to_knowledge_dict == {}

    def test_has_update_success_task_method(self) -> None:
        from rdagent.components.coder.CoSTEER.knowledge_management import CoSTEERKnowledgeBaseV2
        kb = CoSTEERKnowledgeBaseV2()
        assert hasattr(kb, "update_success_task")

    def test_has_graph_query_methods(self) -> None:
        from rdagent.components.coder.CoSTEER.knowledge_management import CoSTEERKnowledgeBaseV2
        assert hasattr(CoSTEERKnowledgeBaseV2, "graph_query_by_content")
        assert hasattr(CoSTEERKnowledgeBaseV2, "graph_query_by_node")
        assert hasattr(CoSTEERKnowledgeBaseV2, "graph_query_by_intersection")


# =============================================================================
# CoSTEERSingleFeedback
# =============================================================================


class TestCoSTEERSingleFeedback:
    def test_construction_with_all_fields(self) -> None:
        from rdagent.components.coder.CoSTEER.evaluators import CoSTEERSingleFeedback
        fb = CoSTEERSingleFeedback(execution="exec ok", return_checking="return ok", code="code ok", final_decision=True)
        assert fb.execution == "exec ok"
        assert fb.return_checking == "return ok"
        assert fb.code == "code ok"
        assert fb.final_decision is True

    def test_construction_default_final_decision_none(self) -> None:
        from rdagent.components.coder.CoSTEER.evaluators import CoSTEERSingleFeedback
        fb = CoSTEERSingleFeedback(execution="x", return_checking="x", code="x")
        assert fb.final_decision is None

    def test_val_and_update_init_dict_converts_false_string(self) -> None:
        from rdagent.components.coder.CoSTEER.evaluators import CoSTEERSingleFeedback
        for s in ("false", "False"):
            result = CoSTEERSingleFeedback.val_and_update_init_dict({
                "execution": "x", "return_checking": "x", "code": "x", "final_decision": s,
            })
            assert result["final_decision"] is False

    def test_val_and_update_init_dict_converts_true_string(self) -> None:
        from rdagent.components.coder.CoSTEER.evaluators import CoSTEERSingleFeedback
        for s in ("true", "True"):
            result = CoSTEERSingleFeedback.val_and_update_init_dict({
                "execution": "x", "return_checking": "x", "code": "x", "final_decision": s,
            })
            assert result["final_decision"] is True

    def test_val_and_update_init_dict_raises_on_missing_final_decision(self) -> None:
        from rdagent.components.coder.CoSTEER.evaluators import CoSTEERSingleFeedback
        with pytest.raises(ValueError):
            CoSTEERSingleFeedback.val_and_update_init_dict({
                "execution": "x", "return_checking": "x", "code": "x",
            })

    def test_val_and_update_init_dict_raises_on_invalid_type(self) -> None:
        from rdagent.components.coder.CoSTEER.evaluators import CoSTEERSingleFeedback
        with pytest.raises(ValueError):
            CoSTEERSingleFeedback.val_and_update_init_dict({
                "execution": "x", "return_checking": "x", "code": "x", "final_decision": 1,
            })

    def test_val_and_update_init_dict_jsonifies_non_string_attrs(self) -> None:
        from rdagent.components.coder.CoSTEER.evaluators import CoSTEERSingleFeedback
        result = CoSTEERSingleFeedback.val_and_update_init_dict({
            "execution": ["line1"], "return_checking": {"status": "ok"}, "code": ["def f(): pass"],
            "final_decision": True,
        })
        assert isinstance(result["execution"], str)

    def test_val_and_update_init_dict_preserves_none_attrs(self) -> None:
        from rdagent.components.coder.CoSTEER.evaluators import CoSTEERSingleFeedback
        result = CoSTEERSingleFeedback.val_and_update_init_dict({
            "execution": "x", "return_checking": None, "code": "x", "final_decision": False,
        })
        assert result["return_checking"] is None

    def test_merge_all_true_makes_true(self) -> None:
        from rdagent.components.coder.CoSTEER.evaluators import CoSTEERSingleFeedback
        fb1 = CoSTEERSingleFeedback(execution="a", return_checking="a", code="a", final_decision=True)
        fb2 = CoSTEERSingleFeedback(execution="b", return_checking="b", code="b", final_decision=True)
        merged = CoSTEERSingleFeedback.merge([fb1, fb2])
        assert merged.final_decision is True

    def test_merge_one_false_makes_false(self) -> None:
        from rdagent.components.coder.CoSTEER.evaluators import CoSTEERSingleFeedback
        fb1 = CoSTEERSingleFeedback(execution="a", return_checking="a", code="a", final_decision=True)
        fb2 = CoSTEERSingleFeedback(execution="b", return_checking="b", code="b", final_decision=False)
        merged = CoSTEERSingleFeedback.merge([fb1, fb2])
        assert merged.final_decision is False

    def test_merge_concatenates_execution_strings(self) -> None:
        from rdagent.components.coder.CoSTEER.evaluators import CoSTEERSingleFeedback
        fb1 = CoSTEERSingleFeedback(execution="ex1", return_checking="r1", code="c1", final_decision=False)
        fb2 = CoSTEERSingleFeedback(execution="ex2", return_checking="r2", code="c2", final_decision=False)
        merged = CoSTEERSingleFeedback.merge([fb1, fb2])
        assert "ex1\n\nex2" in merged.execution

    def test_merge_preserves_source_feedback(self) -> None:
        from rdagent.components.coder.CoSTEER.evaluators import CoSTEERSingleFeedback
        fb1 = CoSTEERSingleFeedback(execution="a", return_checking="a", code="a", final_decision=True, source_feedback={"e1": True})
        fb2 = CoSTEERSingleFeedback(execution="b", return_checking="b", code="b", final_decision=True, source_feedback={"e2": False})
        merged = CoSTEERSingleFeedback.merge([fb1, fb2])
        assert merged.source_feedback["e1"] is True
        assert merged.source_feedback["e2"] is False

    def test_str_contains_success_on_true(self) -> None:
        from rdagent.components.coder.CoSTEER.evaluators import CoSTEERSingleFeedback
        fb = CoSTEERSingleFeedback(execution="x", return_checking="y", code="z", final_decision=True)
        assert "SUCCESS" in str(fb)

    def test_str_contains_fail_on_false(self) -> None:
        from rdagent.components.coder.CoSTEER.evaluators import CoSTEERSingleFeedback
        fb = CoSTEERSingleFeedback(execution="x", return_checking="y", code="z", final_decision=False)
        assert "FAIL" in str(fb)

    def test_str_no_return_checking_when_none(self) -> None:
        from rdagent.components.coder.CoSTEER.evaluators import CoSTEERSingleFeedback
        fb = CoSTEERSingleFeedback(execution="x", return_checking=None, code="z", final_decision=False)
        assert "No return checking" in str(fb)

    def test_bool_returns_final_decision(self) -> None:
        from rdagent.components.coder.CoSTEER.evaluators import CoSTEERSingleFeedback
        fb_true = CoSTEERSingleFeedback(execution="x", return_checking="x", code="x", final_decision=True)
        fb_false = CoSTEERSingleFeedback(execution="x", return_checking="x", code="x", final_decision=False)
        assert bool(fb_true) is True
        assert bool(fb_false) is False

    def test_source_feedback_defaults_to_empty_dict(self) -> None:
        from rdagent.components.coder.CoSTEER.evaluators import CoSTEERSingleFeedback
        fb = CoSTEERSingleFeedback(execution="x", return_checking="x", code="x", final_decision=True)
        assert fb.source_feedback == {}

    def test_raw_execution_default_empty_str(self) -> None:
        from rdagent.components.coder.CoSTEER.evaluators import CoSTEERSingleFeedback
        fb = CoSTEERSingleFeedback(execution="x", return_checking="x", code="x", final_decision=True)
        assert fb.raw_execution == ""

    def test_pickle_safety(self) -> None:
        from rdagent.components.coder.CoSTEER.evaluators import CoSTEERSingleFeedback
        fb = CoSTEERSingleFeedback(execution="exec", return_checking="ret", code="code", final_decision=True, source_feedback={"src": True})
        data = pickle.dumps(fb)
        fb2 = pickle.loads(data)
        assert fb2.execution == "exec"
        assert fb2.final_decision is True

    def test_merge_single_item_is_deepcopy(self) -> None:
        from rdagent.components.coder.CoSTEER.evaluators import CoSTEERSingleFeedback
        fb = CoSTEERSingleFeedback(execution="ex", return_checking="rc", code="cd", final_decision=True)
        merged = CoSTEERSingleFeedback.merge([fb])
        assert merged is not fb

    def test_final_decision_bool_conversion(self) -> None:
        """True/False boolean is passed through unchanged."""
        from rdagent.components.coder.CoSTEER.evaluators import CoSTEERSingleFeedback
        result = CoSTEERSingleFeedback.val_and_update_init_dict({
            "execution": "x", "return_checking": "x", "code": "x", "final_decision": True,
        })
        assert result["final_decision"] is True


# =============================================================================
# CoSTEERSingleFeedbackDeprecated
# =============================================================================


class TestCoSTEERSingleFeedbackDeprecated:
    def test_construction_with_all_kwargs(self) -> None:
        from rdagent.components.coder.CoSTEER.evaluators import CoSTEERSingleFeedbackDeprecated
        fb = CoSTEERSingleFeedbackDeprecated(execution_feedback="e", shape_feedback="s", code_feedback="c",
            value_feedback="v", final_decision=True, final_feedback="f",
            value_generated_flag=True, final_decision_based_on_gt=True, source_feedback={"src": True})
        assert fb.execution_feedback == "e"

    def test_execution_property_returns_execution_feedback(self) -> None:
        from rdagent.components.coder.CoSTEER.evaluators import CoSTEERSingleFeedbackDeprecated
        fb = CoSTEERSingleFeedbackDeprecated(execution_feedback="hello")
        assert fb.execution == "hello"

    def test_execution_setter_sets_execution_feedback(self) -> None:
        from rdagent.components.coder.CoSTEER.evaluators import CoSTEERSingleFeedbackDeprecated
        fb = CoSTEERSingleFeedbackDeprecated()
        fb.execution = "world"
        assert fb.execution_feedback == "world"

    def test_return_checking_returns_feedback_when_generated(self) -> None:
        from rdagent.components.coder.CoSTEER.evaluators import CoSTEERSingleFeedbackDeprecated
        fb = CoSTEERSingleFeedbackDeprecated(value_feedback="val ok", shape_feedback="shape ok", value_generated_flag=True)
        rc = fb.return_checking
        assert "val ok" in rc

    def test_return_checking_returns_none_when_not_generated(self) -> None:
        from rdagent.components.coder.CoSTEER.evaluators import CoSTEERSingleFeedbackDeprecated
        fb = CoSTEERSingleFeedbackDeprecated(value_generated_flag=False)
        assert fb.return_checking is None

    def test_code_property_returns_code_feedback(self) -> None:
        from rdagent.components.coder.CoSTEER.evaluators import CoSTEERSingleFeedbackDeprecated
        fb = CoSTEERSingleFeedbackDeprecated(code_feedback="my code")
        assert fb.code == "my code"

    def test_code_setter_sets_code_feedback(self) -> None:
        from rdagent.components.coder.CoSTEER.evaluators import CoSTEERSingleFeedbackDeprecated
        fb = CoSTEERSingleFeedbackDeprecated()
        fb.code = "new code"
        assert fb.code_feedback == "new code"

    def test_str_contains_all_sections(self) -> None:
        from rdagent.components.coder.CoSTEER.evaluators import CoSTEERSingleFeedbackDeprecated
        fb = CoSTEERSingleFeedbackDeprecated(execution_feedback="exec", shape_feedback="shape",
            code_feedback="code", value_feedback="val", final_feedback="final", final_decision=True)
        s = str(fb)
        assert "exec" in s
        assert "SUCCESS" in s

    def test_default_values_are_none(self) -> None:
        from rdagent.components.coder.CoSTEER.evaluators import CoSTEERSingleFeedbackDeprecated
        fb = CoSTEERSingleFeedbackDeprecated()
        assert fb.execution_feedback is None
        assert fb.final_decision is None


# =============================================================================
# CoSTEERMultiFeedback
# =============================================================================


class TestCoSTEERMultiFeedback:
    def test_empty_construction(self) -> None:
        from rdagent.components.coder.CoSTEER.evaluators import CoSTEERMultiFeedback
        mf = CoSTEERMultiFeedback([])
        assert len(mf) == 0

    def test_getitem_returns_single_feedback(self) -> None:
        from rdagent.components.coder.CoSTEER.evaluators import CoSTEERMultiFeedback, CoSTEERSingleFeedback
        fb = CoSTEERSingleFeedback(execution="x", return_checking="x", code="x", final_decision=True)
        mf = CoSTEERMultiFeedback([fb])
        assert mf[0] is fb

    def test_append_adds_feedback(self) -> None:
        from rdagent.components.coder.CoSTEER.evaluators import CoSTEERMultiFeedback, CoSTEERSingleFeedback
        mf = CoSTEERMultiFeedback([])
        fb = CoSTEERSingleFeedback(execution="x", return_checking="x", code="x", final_decision=True)
        mf.append(fb)
        assert len(mf) == 1

    def test_iter_yields_all_feedbacks(self) -> None:
        from rdagent.components.coder.CoSTEER.evaluators import CoSTEERMultiFeedback, CoSTEERSingleFeedback
        fbs = [
            CoSTEERSingleFeedback(execution="a", return_checking="a", code="a", final_decision=True),
            CoSTEERSingleFeedback(execution="b", return_checking="b", code="b", final_decision=False),
        ]
        mf = CoSTEERMultiFeedback(fbs)
        assert list(mf) == fbs

    def test_is_acceptable_all_true(self) -> None:
        from rdagent.components.coder.CoSTEER.evaluators import CoSTEERMultiFeedback, CoSTEERSingleFeedback
        fbs = [
            CoSTEERSingleFeedback(execution="a", return_checking="a", code="a", final_decision=True),
            CoSTEERSingleFeedback(execution="b", return_checking="b", code="b", final_decision=True),
        ]
        assert CoSTEERMultiFeedback(fbs).is_acceptable()

    def test_is_acceptable_any_false(self) -> None:
        from rdagent.components.coder.CoSTEER.evaluators import CoSTEERMultiFeedback, CoSTEERSingleFeedback
        fbs = [
            CoSTEERSingleFeedback(execution="a", return_checking="a", code="a", final_decision=True),
            CoSTEERSingleFeedback(execution="b", return_checking="b", code="b", final_decision=False),
        ]
        assert not CoSTEERMultiFeedback(fbs).is_acceptable()

    def test_finished_succeeds_with_none_feedbacks(self) -> None:
        from rdagent.components.coder.CoSTEER.evaluators import CoSTEERMultiFeedback, CoSTEERSingleFeedback
        fbs = [
            CoSTEERSingleFeedback(execution="a", return_checking="a", code="a", final_decision=True),
            None,
        ]
        assert CoSTEERMultiFeedback(fbs).finished()

    def test_bool_all_true(self) -> None:
        from rdagent.components.coder.CoSTEER.evaluators import CoSTEERMultiFeedback, CoSTEERSingleFeedback
        fbs = [CoSTEERSingleFeedback(execution="a", return_checking="a", code="a", final_decision=True)]
        assert bool(CoSTEERMultiFeedback(fbs))


# =============================================================================
# CoSTEERMultiEvaluator
# =============================================================================


class TestCoSTEERMultiEvaluator:
    def test_initialization_with_single_evaluator(self) -> None:
        from rdagent.components.coder.CoSTEER.evaluators import CoSTEERMultiEvaluator
        mock_eval = MagicMock()
        evaluator = CoSTEERMultiEvaluator(single_evaluator=mock_eval, scen=MagicMock())
        assert evaluator.single_evaluator is mock_eval

    def test_initialization_with_list_of_evaluators(self) -> None:
        from rdagent.components.coder.CoSTEER.evaluators import CoSTEERMultiEvaluator
        mock_evals = [MagicMock(), MagicMock()]
        evaluator = CoSTEERMultiEvaluator(single_evaluator=mock_evals, scen=MagicMock())
        assert evaluator.single_evaluator == mock_evals


# =============================================================================
# Factor Evaluators
# =============================================================================


class TestFactorInfEvaluator:
    def test_no_inf_values_returns_true(self) -> None:
        from rdagent.components.coder.factor_coder.eva_utils import FactorInfEvaluator
        evaluator = FactorInfEvaluator()
        imp = MagicMock()
        df = pd.DataFrame({"f": [1.0, 2.0]}, index=pd.MultiIndex.from_tuples(
            [("2020-01-01", "EUR"), ("2020-01-02", "EUR")], names=["datetime", "instrument"]))
        imp.execute.return_value = (None, df)
        _, result_bool = evaluator.evaluate(implementation=imp, gt_implementation=None)
        assert result_bool is True

    def test_with_inf_values_returns_false(self) -> None:
        from rdagent.components.coder.factor_coder.eva_utils import FactorInfEvaluator
        evaluator = FactorInfEvaluator()
        imp = MagicMock()
        df = pd.DataFrame({"f": [float("inf"), 2.0]}, index=pd.MultiIndex.from_tuples(
            [("2020-01-01", "EUR"), ("2020-01-02", "EUR")], names=["datetime", "instrument"]))
        imp.execute.return_value = (None, df)
        _, result_bool = evaluator.evaluate(implementation=imp, gt_implementation=None)
        assert result_bool is False

    def test_none_dataframe_returns_false(self) -> None:
        from rdagent.components.coder.factor_coder.eva_utils import FactorInfEvaluator
        evaluator = FactorInfEvaluator()
        imp = MagicMock()
        imp.execute.return_value = (None, None)
        _, result_bool = evaluator.evaluate(implementation=imp, gt_implementation=None)
        assert result_bool is False


class TestFactorSingleColumnEvaluator:
    def test_single_column_returns_true(self) -> None:
        from rdagent.components.coder.factor_coder.eva_utils import FactorSingleColumnEvaluator
        evaluator = FactorSingleColumnEvaluator()
        imp = MagicMock()
        df = pd.DataFrame({"col": [1]}, index=pd.MultiIndex.from_tuples(
            [("2020-01-01", "EUR")], names=["datetime", "instrument"]))
        imp.execute.return_value = (None, df)
        _, result = evaluator.evaluate(implementation=imp, gt_implementation=None)
        assert result is True

    def test_multi_column_returns_false(self) -> None:
        from rdagent.components.coder.factor_coder.eva_utils import FactorSingleColumnEvaluator
        evaluator = FactorSingleColumnEvaluator()
        imp = MagicMock()
        df = pd.DataFrame({"a": [1], "b": [2]}, index=pd.MultiIndex.from_tuples(
            [("2020-01-01", "EUR")], names=["datetime", "instrument"]))
        imp.execute.return_value = (None, df)
        _, result = evaluator.evaluate(implementation=imp, gt_implementation=None)
        assert result is False


class TestFactorRowCountEvaluator:
    def test_equal_row_count_returns_ratio_one(self) -> None:
        from rdagent.components.coder.factor_coder.eva_utils import FactorRowCountEvaluator
        evaluator = FactorRowCountEvaluator()
        imp = MagicMock()
        gt = MagicMock()
        idx = pd.MultiIndex.from_tuples([("2020-01-01", "EUR")], names=["datetime", "instrument"])
        imp.execute.return_value = (None, pd.DataFrame({"f": [1]}, index=idx))
        gt.execute.return_value = (None, pd.DataFrame({"f": [2]}, index=idx))
        _, ratio = evaluator.evaluate(implementation=imp, gt_implementation=gt)
        assert ratio == 1.0

    def test_different_row_count_returns_ratio_below_one(self) -> None:
        from rdagent.components.coder.factor_coder.eva_utils import FactorRowCountEvaluator
        evaluator = FactorRowCountEvaluator()
        imp = MagicMock()
        gt = MagicMock()
        idx_a = pd.MultiIndex.from_tuples([("2020-01-01", "EUR")], names=["datetime", "instrument"])
        idx_b = pd.MultiIndex.from_tuples(
            [("2020-01-01", "EUR"), ("2020-01-02", "EUR")], names=["datetime", "instrument"])
        imp.execute.return_value = (None, pd.DataFrame({"f": [1]}, index=idx_a))
        gt.execute.return_value = (None, pd.DataFrame({"f": [2, 3]}, index=idx_b))
        _, ratio = evaluator.evaluate(implementation=imp, gt_implementation=gt)
        assert ratio < 1.0

    @pytest.mark.parametrize("gen_rows,gt_rows,expected", [
        (5, 5, 1.0),
        (100, 5, 0.05),
        (500, 500, 1.0),
    ])
    def test_row_count_variants(self, gen_rows: int, gt_rows: int, expected: float) -> None:
        from rdagent.components.coder.factor_coder.eva_utils import FactorRowCountEvaluator
        evaluator = FactorRowCountEvaluator()
        imp = MagicMock()
        gt = MagicMock()
        gen_idx = pd.MultiIndex.from_tuples(
            [(f"2020-01-{i+1:02d}", "EUR") for i in range(max(gen_rows, 1))],
            names=["datetime", "instrument"])
        gt_idx = pd.MultiIndex.from_tuples(
            [(f"2020-01-{i+1:02d}", "EUR") for i in range(max(gt_rows, 1))],
            names=["datetime", "instrument"])
        imp.execute.return_value = (None, pd.DataFrame({"f": list(range(len(gen_idx)))}, index=gen_idx))
        gt.execute.return_value = (None, pd.DataFrame({"f": list(range(len(gt_idx)))}, index=gt_idx))
        _, ratio = evaluator.evaluate(implementation=imp, gt_implementation=gt)
        assert ratio == pytest.approx(expected)


class TestFactorIndexEvaluator:
    def test_identical_index_returns_one(self) -> None:
        from rdagent.components.coder.factor_coder.eva_utils import FactorIndexEvaluator
        evaluator = FactorIndexEvaluator()
        imp = MagicMock()
        gt = MagicMock()
        idx = pd.MultiIndex.from_tuples(
            [("2020-01-01", "EUR"), ("2020-01-02", "EUR")], names=["datetime", "instrument"])
        imp.execute.return_value = (None, pd.DataFrame({"f": [1, 2]}, index=idx))
        gt.execute.return_value = (None, pd.DataFrame({"f": [3, 4]}, index=idx))
        _, sim = evaluator.evaluate(implementation=imp, gt_implementation=gt)
        assert sim == 1.0

    def test_disjoint_index_returns_zero(self) -> None:
        from rdagent.components.coder.factor_coder.eva_utils import FactorIndexEvaluator
        evaluator = FactorIndexEvaluator()
        imp = MagicMock()
        gt = MagicMock()
        idx_a = pd.MultiIndex.from_tuples([("2020-01-01", "EUR")], names=["datetime", "instrument"])
        idx_b = pd.MultiIndex.from_tuples([("2020-01-02", "GBP")], names=["datetime", "instrument"])
        imp.execute.return_value = (None, pd.DataFrame({"f": [1]}, index=idx_a))
        gt.execute.return_value = (None, pd.DataFrame({"f": [2]}, index=idx_b))
        _, sim = evaluator.evaluate(implementation=imp, gt_implementation=gt)
        assert sim == 0.0


class TestFactorEqualValueRatioEvaluator:
    def test_identical_values_return_accuracy_one(self) -> None:
        from rdagent.components.coder.factor_coder.eva_utils import FactorEqualValueRatioEvaluator
        evaluator = FactorEqualValueRatioEvaluator()
        imp = MagicMock()
        gt = MagicMock()
        idx = pd.MultiIndex.from_tuples(
            [("2020-01-01", "EUR"), ("2020-01-02", "EUR")], names=["datetime", "instrument"])
        df = pd.DataFrame({"f": [1.0, 2.0]}, index=idx)
        imp.execute.return_value = (None, df)
        gt.execute.return_value = (None, df.copy())
        _, acc = evaluator.evaluate(implementation=imp, gt_implementation=gt)
        assert acc == 1.0

    def test_different_values_return_lower_accuracy(self) -> None:
        from rdagent.components.coder.factor_coder.eva_utils import FactorEqualValueRatioEvaluator
        evaluator = FactorEqualValueRatioEvaluator()
        imp = MagicMock()
        gt = MagicMock()
        idx = pd.MultiIndex.from_tuples(
            [("2020-01-01", "EUR"), ("2020-01-02", "EUR")], names=["datetime", "instrument"])
        imp.execute.return_value = (None, pd.DataFrame({"f": [1.0, 2.0]}, index=idx))
        gt.execute.return_value = (None, pd.DataFrame({"f": [1.0, 3.0]}, index=idx))
        _, acc = evaluator.evaluate(implementation=imp, gt_implementation=gt)
        assert acc < 1.0

    def test_none_dataframe_returns_negative_one(self) -> None:
        from rdagent.components.coder.factor_coder.eva_utils import FactorEqualValueRatioEvaluator
        evaluator = FactorEqualValueRatioEvaluator()
        imp = MagicMock()
        imp.execute.return_value = (None, None)
        _, acc = evaluator.evaluate(implementation=imp, gt_implementation=None)
        assert acc == -1


class TestFactorCorrelationEvaluator:
    def test_is_constructible(self) -> None:
        from rdagent.components.coder.factor_coder.eva_utils import FactorCorrelationEvaluator
        ev = FactorCorrelationEvaluator(hard_check=True)
        assert ev.hard_check is True

    def test_none_dataframe_returns_false(self) -> None:
        from rdagent.components.coder.factor_coder.eva_utils import FactorCorrelationEvaluator
        evaluator = FactorCorrelationEvaluator(hard_check=False)
        imp = MagicMock()
        imp.execute.return_value = (None, None)
        _, result = evaluator.evaluate(implementation=imp, gt_implementation=None)
        assert result is False


class TestFactorDatetimeDailyEvaluator:
    def test_valid_datetime_index_returns_true(self) -> None:
        from rdagent.components.coder.factor_coder.eva_utils import FactorDatetimeDailyEvaluator
        evaluator = FactorDatetimeDailyEvaluator()
        imp = MagicMock()
        idx = pd.MultiIndex.from_tuples(
            [("2020-01-01 09:00:00", "EUR"), ("2020-01-01 10:00:00", "EUR")],
            names=["datetime", "instrument"])
        imp.execute.return_value = (None, pd.DataFrame({"f": [1, 2]}, index=idx))
        _, result = evaluator.evaluate(implementation=imp, gt_implementation=None)
        assert result is True

    def test_no_datetime_index_returns_false(self) -> None:
        from rdagent.components.coder.factor_coder.eva_utils import FactorDatetimeDailyEvaluator
        evaluator = FactorDatetimeDailyEvaluator()
        imp = MagicMock()
        imp.execute.return_value = (None, pd.DataFrame({"f": [1]}, index=[0]))
        _, result = evaluator.evaluate(implementation=imp, gt_implementation=None)
        assert result is False


class TestFactorValueEvaluator:
    def test_evaluator_is_importable(self) -> None:
        from rdagent.components.coder.factor_coder.eva_utils import FactorValueEvaluator
        assert FactorValueEvaluator is not None

    def test_evaluate_method_exists(self) -> None:
        from rdagent.components.coder.factor_coder.eva_utils import FactorValueEvaluator
        assert hasattr(FactorValueEvaluator, "evaluate")


class TestFactorFinalDecisionEvaluator:
    def test_evaluator_is_importable(self) -> None:
        from rdagent.components.coder.factor_coder.eva_utils import FactorFinalDecisionEvaluator
        assert FactorFinalDecisionEvaluator is not None


# =============================================================================
# FactorEvaluator base class
# =============================================================================


class TestFactorEvaluatorBase:
    def test_str_returns_class_name(self) -> None:
        from rdagent.components.coder.factor_coder.eva_utils import FactorInfEvaluator
        ev = FactorInfEvaluator()
        assert str(ev) == "FactorInfEvaluator"


# =============================================================================
# Auto-fixer (extends existing tests)
# =============================================================================


class TestAutoFixerEdgeCases:
    @pytest.fixture
    def fixer(self):
        from rdagent.components.coder.factor_coder.auto_fixer import FactorAutoFixer
        return FactorAutoFixer()

    def test_empty_code_returns_empty(self, fixer) -> None:
        result = fixer.fix("")
        assert result == ""

    def test_whitespace_only_code_preserved(self, fixer) -> None:
        result = fixer.fix("   \n  \n  ")
        assert "   " in result

    def test_none_task_info_does_not_crash(self, fixer) -> None:
        result = fixer.fix("x = 1", factor_task_info=None)
        assert "x = 1" in result

    def test_very_long_code_handled(self, fixer) -> None:
        long_code = "x = 1\n" * 100 + "df['x'] = df.groupby(level=1)['y'].mean()\n" + "y = 2\n" * 100
        result = fixer.fix(long_code)
        assert "groupby" in result

    def test_convenience_function_returns_string(self, fixer) -> None:
        from rdagent.components.coder.factor_coder.auto_fixer import auto_fix_factor_code
        result = auto_fix_factor_code("x = 1")
        assert isinstance(result, str)

    def test_fixes_applied_list_tracks_changes(self, fixer) -> None:
        code = "df.groupby(['instrument'])['x'].mean()"
        fixer.fix(code)
        assert len(fixer.fixes_applied) > 0


# =============================================================================
# FactorMultiProcessEvolvingStrategy
# =============================================================================


class TestFactorMultiProcessEvolvingStrategy:
    def test_strategy_is_importable(self) -> None:
        from rdagent.components.coder.factor_coder.evolving_strategy import (
            FactorMultiProcessEvolvingStrategy,
        )
        assert FactorMultiProcessEvolvingStrategy is not None


# =============================================================================
# FactorFBWorkspace / FactorTask
# =============================================================================


class TestFactorWorkspaceImport:
    def test_factor_fb_workspace_importable(self) -> None:
        from rdagent.components.coder.factor_coder.factor import FactorFBWorkspace
        assert FactorFBWorkspace is not None

    def test_factor_task_importable(self) -> None:
        from rdagent.components.coder.factor_coder.factor import FactorTask
        assert FactorTask is not None


# =============================================================================
# UndirectedGraph / UndirectedNode
# =============================================================================


class TestUndirectedGraphIntegration:
    def test_undirected_graph_importable(self) -> None:
        from rdagent.components.knowledge_management.graph import UndirectedGraph
        assert UndirectedGraph is not None

    def test_undirected_node_importable(self) -> None:
        from rdagent.components.knowledge_management.graph import UndirectedNode
        assert UndirectedNode is not None
