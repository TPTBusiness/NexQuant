"""Tests for CoSTEER feedback types and EvolvingItem."""

from __future__ import annotations

import sys
from pathlib import Path
from unittest.mock import MagicMock

import pytest

PROJECT_ROOT = Path(__file__).parent.parent.parent
sys.path.insert(0, str(PROJECT_ROOT))


# =============================================================================
# CoSTEERSingleFeedback
# =============================================================================


class TestCoSTEERSingleFeedback:
    def test_construction_with_valid_fields(self):
        from rdagent.components.coder.CoSTEER.evaluators import CoSTEERSingleFeedback
        fb = CoSTEERSingleFeedback(
            execution="exec ok",
            return_checking="return ok",
            code="code ok",
            final_decision=True,
        )
        assert fb.execution == "exec ok"
        assert fb.return_checking == "return ok"
        assert fb.code == "code ok"
        assert fb.final_decision is True

    def test_bool_returns_final_decision(self):
        from rdagent.components.coder.CoSTEER.evaluators import CoSTEERSingleFeedback
        fb_true = CoSTEERSingleFeedback(execution="x", return_checking="x", code="x", final_decision=True)
        fb_false = CoSTEERSingleFeedback(execution="x", return_checking="x", code="x", final_decision=False)
        assert bool(fb_true) is True
        assert bool(fb_false) is False

    def test_default_values(self):
        from rdagent.components.coder.CoSTEER.evaluators import CoSTEERSingleFeedback
        fb = CoSTEERSingleFeedback(execution="x", return_checking=None, code="x")
        assert fb.final_decision is None
        assert fb.raw_execution == ""
        assert fb.source_feedback == {}

    def test_val_and_update_init_dict_converts_boolean(self):
        from rdagent.components.coder.CoSTEER.evaluators import CoSTEERSingleFeedback
        assert CoSTEERSingleFeedback.val_and_update_init_dict(
            {"execution": "x", "return_checking": "y", "code": "z", "final_decision": "true"}
        )["final_decision"] is True
        assert CoSTEERSingleFeedback.val_and_update_init_dict(
            {"execution": "x", "return_checking": "y", "code": "z", "final_decision": "false"}
        )["final_decision"] is False
        assert CoSTEERSingleFeedback.val_and_update_init_dict(
            {"execution": "x", "return_checking": "y", "code": "z", "final_decision": "True"}
        )["final_decision"] is True
        assert CoSTEERSingleFeedback.val_and_update_init_dict(
            {"execution": "x", "return_checking": "y", "code": "z", "final_decision": "False"}
        )["final_decision"] is False

    def test_val_and_update_init_dict_rejects_non_boolean(self):
        from rdagent.components.coder.CoSTEER.evaluators import CoSTEERSingleFeedback
        with pytest.raises(ValueError):
            CoSTEERSingleFeedback.val_and_update_init_dict(
                {"execution": "x", "return_checking": "y", "code": "z", "final_decision": 42}
            )

    def test_val_and_update_init_dict_missing_final_decision_raises(self):
        from rdagent.components.coder.CoSTEER.evaluators import CoSTEERSingleFeedback
        with pytest.raises(ValueError, match="final_decision"):
            CoSTEERSingleFeedback.val_and_update_init_dict(
                {"execution": "x", "return_checking": "y", "code": "z"}
            )

    def test_val_and_update_init_dict_json_dumps_non_string_attrs(self):
        from rdagent.components.coder.CoSTEER.evaluators import CoSTEERSingleFeedback
        import json
        data = {
            "execution": {"key": "val"},
            "return_checking": ["list"],
            "code": 123,
            "final_decision": True,
        }
        result = CoSTEERSingleFeedback.val_and_update_init_dict(data)
        for attr in ("execution", "return_checking", "code"):
            # Should have been converted to JSON string
            assert isinstance(result[attr], str)
            _ = json.loads(result[attr])  # valid JSON

    def test_merge_all_true_decisions(self):
        from rdagent.components.coder.CoSTEER.evaluators import CoSTEERSingleFeedback
        fb1 = CoSTEERSingleFeedback(execution="a", return_checking="ra", code="c1", final_decision=True)
        fb2 = CoSTEERSingleFeedback(execution="b", return_checking="rb", code="c2", final_decision=True)
        merged = CoSTEERSingleFeedback.merge([fb1, fb2])
        assert merged.final_decision is True

    def test_merge_one_false_decision(self):
        from rdagent.components.coder.CoSTEER.evaluators import CoSTEERSingleFeedback
        fb1 = CoSTEERSingleFeedback(execution="a", return_checking="ra", code="c1", final_decision=True)
        fb2 = CoSTEERSingleFeedback(execution="b", return_checking="rb", code="c2", final_decision=False)
        merged = CoSTEERSingleFeedback.merge([fb1, fb2])
        assert merged.final_decision is False

    def test_merge_concatenates_fields(self):
        from rdagent.components.coder.CoSTEER.evaluators import CoSTEERSingleFeedback
        fb1 = CoSTEERSingleFeedback(execution="A", return_checking="RA", code="C1", final_decision=True)
        fb2 = CoSTEERSingleFeedback(execution="B", return_checking="RB", code="C2", final_decision=True)
        merged = CoSTEERSingleFeedback.merge([fb1, fb2])
        assert "A\n\nB" in merged.execution
        assert "RA\n\nRB" in merged.return_checking
        assert "C1\n\nC2" in merged.code

    def test_merge_skips_none_fields(self):
        from rdagent.components.coder.CoSTEER.evaluators import CoSTEERSingleFeedback
        fb1 = CoSTEERSingleFeedback(execution="A", return_checking="RA", code="C1", final_decision=True)
        fb2 = CoSTEERSingleFeedback(execution="B", return_checking=None, code="C2", final_decision=True)
        merged = CoSTEERSingleFeedback.merge([fb1, fb2])
        assert merged.execution == "A\n\nB"
        assert merged.return_checking == "RA"
        assert merged.code == "C1\n\nC2"

    def test_merge_aggregates_source_feedback(self):
        from rdagent.components.coder.CoSTEER.evaluators import CoSTEERSingleFeedback
        fb1 = CoSTEERSingleFeedback(execution="a", return_checking="x", code="c", final_decision=True,
                                     source_feedback={"eval1": True})
        fb2 = CoSTEERSingleFeedback(execution="b", return_checking="y", code="d", final_decision=True,
                                     source_feedback={"eval2": False})
        merged = CoSTEERSingleFeedback.merge([fb1, fb2])
        assert merged.source_feedback == {"eval1": True, "eval2": False}

    def test_str_contains_all_sections(self):
        from rdagent.components.coder.CoSTEER.evaluators import CoSTEERSingleFeedback
        fb = CoSTEERSingleFeedback(execution="exec", return_checking="ret", code="code", final_decision=True)
        s = str(fb)
        assert "Execution" in s
        assert "Return Checking" in s
        assert "Code" in s
        assert "Final Decision" in s
        assert "SUCCESS" in s

    def test_str_shows_fail_for_false(self):
        from rdagent.components.coder.CoSTEER.evaluators import CoSTEERSingleFeedback
        fb = CoSTEERSingleFeedback(execution="exec", return_checking="ret", code="code", final_decision=False)
        assert "FAIL" in str(fb)


# =============================================================================
# CoSTEERSingleFeedbackDeprecated
# =============================================================================


class TestCoSTEERSingleFeedbackDeprecated:
    def test_property_getters(self):
        from rdagent.components.coder.CoSTEER.evaluators import CoSTEERSingleFeedbackDeprecated
        fb = CoSTEERSingleFeedbackDeprecated(
            execution_feedback="exec",
            code_feedback="code",
            value_feedback="val",
            shape_feedback="shape",
            final_decision=True,
            final_feedback="final",
            value_generated_flag=True,
            final_decision_based_on_gt=True,
        )
        assert fb.execution == "exec"
        assert fb.code == "code"
        assert fb.final_decision is True
        assert fb.value_generated_flag is True
        assert fb.final_decision_based_on_gt is True

    def test_return_checking_when_value_generated(self):
        from rdagent.components.coder.CoSTEER.evaluators import CoSTEERSingleFeedbackDeprecated
        fb = CoSTEERSingleFeedbackDeprecated(
            value_generated_flag=True, value_feedback="vals", shape_feedback="shapes",
        )
        rc = fb.return_checking
        assert "vals" in rc
        assert "shapes" in rc

    def test_return_checking_when_no_value_generated(self):
        from rdagent.components.coder.CoSTEER.evaluators import CoSTEERSingleFeedbackDeprecated
        fb = CoSTEERSingleFeedbackDeprecated(value_generated_flag=False)
        assert fb.return_checking is None

    def test_setters_work(self):
        from rdagent.components.coder.CoSTEER.evaluators import CoSTEERSingleFeedbackDeprecated
        fb = CoSTEERSingleFeedbackDeprecated()
        fb.execution = "new_exec"
        fb.code = "new_code"
        fb.return_checking = "new_rc"
        assert fb.execution_feedback == "new_exec"
        assert fb.code_feedback == "new_code"
        assert fb.value_feedback == "new_rc"
        assert fb.shape_feedback == "new_rc"

    def test_str_contains_all_sections(self):
        from rdagent.components.coder.CoSTEER.evaluators import CoSTEERSingleFeedbackDeprecated
        fb = CoSTEERSingleFeedbackDeprecated(
            execution_feedback="exec", shape_feedback="shape",
            code_feedback="code", value_feedback="val",
            final_feedback="final", final_decision=True,
        )
        s = str(fb)
        for keyword in ("Execution", "Shape", "Code", "Value", "Final Decision", "SUCCESS"):
            assert keyword in s


# =============================================================================
# CoSTEERMultiFeedback
# =============================================================================


class TestCoSTEERMultiFeedback:
    def _make_fb(self, decision=True):
        from rdagent.components.coder.CoSTEER.evaluators import CoSTEERSingleFeedback
        return CoSTEERSingleFeedback(execution="x", return_checking="x", code="x", final_decision=decision)

    def test_getitem(self):
        from rdagent.components.coder.CoSTEER.evaluators import CoSTEERMultiFeedback
        fb1, fb2 = self._make_fb(True), self._make_fb(False)
        mf = CoSTEERMultiFeedback([fb1, fb2])
        assert mf[0].final_decision is True
        assert mf[1].final_decision is False

    def test_len(self):
        from rdagent.components.coder.CoSTEER.evaluators import CoSTEERMultiFeedback
        assert len(CoSTEERMultiFeedback([self._make_fb()])) == 1
        assert len(CoSTEERMultiFeedback([])) == 0

    def test_append(self):
        from rdagent.components.coder.CoSTEER.evaluators import CoSTEERMultiFeedback
        mf = CoSTEERMultiFeedback([])
        mf.append(self._make_fb(True))
        assert len(mf) == 1
        assert mf[0].final_decision is True

    def test_iter(self):
        from rdagent.components.coder.CoSTEER.evaluators import CoSTEERMultiFeedback
        fbs = [self._make_fb(True), self._make_fb(True)]
        mf = CoSTEERMultiFeedback(fbs)
        assert list(mf) == fbs

    def test_finished_all_true(self):
        from rdagent.components.coder.CoSTEER.evaluators import CoSTEERMultiFeedback
        mf = CoSTEERMultiFeedback([self._make_fb(True), self._make_fb(True)])
        assert mf.finished() is True

    def test_finished_one_false(self):
        from rdagent.components.coder.CoSTEER.evaluators import CoSTEERMultiFeedback
        mf = CoSTEERMultiFeedback([self._make_fb(True), self._make_fb(False)])
        assert mf.finished() is False

    def test_finished_with_none_skips(self):
        from rdagent.components.coder.CoSTEER.evaluators import CoSTEERMultiFeedback
        mf = CoSTEERMultiFeedback([self._make_fb(True), None])
        assert mf.finished() is True  # None = skipped = accepted

    def test_bool_all_true(self):
        from rdagent.components.coder.CoSTEER.evaluators import CoSTEERMultiFeedback
        assert bool(CoSTEERMultiFeedback([self._make_fb(True), self._make_fb(True)])) is True

    def test_bool_one_false(self):
        from rdagent.components.coder.CoSTEER.evaluators import CoSTEERMultiFeedback
        assert bool(CoSTEERMultiFeedback([self._make_fb(True), self._make_fb(False)])) is False

    def test_is_acceptable_delegates(self):
        from rdagent.components.coder.CoSTEER.evaluators import CoSTEERMultiFeedback
        mf = CoSTEERMultiFeedback([self._make_fb(True), self._make_fb(True)])
        assert mf.is_acceptable() is True


# =============================================================================
# EvolvingItem
# =============================================================================


class TestEvolvingItem:
    def test_construction_without_gt(self):
        from rdagent.components.coder.CoSTEER.evolvable_subjects import EvolvingItem
        from rdagent.core.experiment import Task

        t1 = Task(name="task1")
        t2 = Task(name="task2")
        ei = EvolvingItem(sub_tasks=[t1, t2])
        assert len(ei.sub_tasks) == 2
        assert ei.sub_gt_implementations is None

    def test_construction_with_matching_gt(self):
        from rdagent.components.coder.CoSTEER.evolvable_subjects import EvolvingItem
        from rdagent.core.experiment import Task, FBWorkspace
        t1, t2 = Task(name="t1"), Task(name="t2")
        ws1, ws2 = FBWorkspace(), FBWorkspace()
        ei = EvolvingItem(sub_tasks=[t1, t2], sub_gt_implementations=[ws1, ws2])
        assert ei.sub_gt_implementations == [ws1, ws2]

    def test_mismatched_gt_length_resets_to_none(self):
        from rdagent.components.coder.CoSTEER.evolvable_subjects import EvolvingItem
        from rdagent.core.experiment import Task, FBWorkspace
        t1, t2 = Task(name="t1"), Task(name="t2")
        ei = EvolvingItem(sub_tasks=[t1, t2], sub_gt_implementations=[FBWorkspace()])
        assert ei.sub_gt_implementations is None

    def test_from_experiment(self):
        from rdagent.components.coder.CoSTEER.evolvable_subjects import EvolvingItem
        from rdagent.core.experiment import Experiment, Task

        exp = Experiment(sub_tasks=[Task(name="x")])
        exp.based_experiments = ["base"]
        exp.experiment_workspace = "ws"
        ei = EvolvingItem.from_experiment(exp)
        assert len(ei.sub_tasks) == 1
        assert ei.based_experiments == ["base"]
        assert ei.experiment_workspace == "ws"


# =============================================================================
# CoSTEERQueriedKnowledge
# =============================================================================


class TestCoSTEERQueriedKnowledge:
    def test_default_construction(self):
        from rdagent.components.coder.CoSTEER.knowledge_management import CoSTEERQueriedKnowledge
        qk = CoSTEERQueriedKnowledge()
        assert qk.success_task_to_knowledge_dict == {}
        assert qk.failed_task_info_set == set()

    def test_with_data(self):
        from rdagent.components.coder.CoSTEER.knowledge_management import CoSTEERQueriedKnowledge
        qk = CoSTEERQueriedKnowledge(
            success_task_to_knowledge_dict={"a": "knowledge_a"},
            failed_task_info_set={"fail1", "fail2"},
        )
        assert qk.success_task_to_knowledge_dict["a"] == "knowledge_a"
        assert "fail1" in qk.failed_task_info_set
        assert "fail2" in qk.failed_task_info_set
