"""Deep tests for workflow components: rd_loop.py, proposal, trace, hypothesis systems."""

from __future__ import annotations

import asyncio
import pickle
import sys
from multiprocessing import Queue
from pathlib import Path
from typing import Any
from unittest.mock import MagicMock, patch

import pytest

PROJECT_ROOT = Path(__file__).parent.parent.parent
sys.path.insert(0, str(PROJECT_ROOT))


def _make_mock_prop_setting(**overrides: Any) -> Any:
    ps = MagicMock()
    ps.scen = "rdagent.scenarios.qlib.scenario.QlibQuantScenario"
    for k, v in overrides.items():
        setattr(ps, k, v)
    ps.model_dump.return_value = {}
    return ps


# =============================================================================
# Import safety
# =============================================================================

WORKFLOW_MODULES = [
    "rdagent.components.workflow.rd_loop",
    "rdagent.components.workflow.conf",
    "rdagent.core.proposal",
    "rdagent.core.developer",
    "rdagent.core.experiment",
    "rdagent.core.scenario",
    "rdagent.core.evolving_framework",
    "rdagent.core.evolving_agent",
    "rdagent.core.utils",
    "rdagent.utils.workflow",
    "rdagent.utils.qlib",
]


class TestWorkflowImports:
    @pytest.mark.parametrize("module_name", WORKFLOW_MODULES)
    def test_module_importable(self, module_name: str) -> None:
        import importlib
        mod = importlib.import_module(module_name)
        assert mod is not None


# =============================================================================
# LoopBase and LoopMeta
# =============================================================================


class TestLoopBase:
    def test_loop_base_is_importable(self) -> None:
        from rdagent.utils.workflow import LoopBase
        assert LoopBase is not None

    def test_loop_meta_is_importable(self) -> None:
        from rdagent.utils.workflow import LoopMeta
        assert LoopMeta is not None

    def test_loop_base_can_be_instantiated(self) -> None:
        from rdagent.utils.workflow import LoopBase
        loop = LoopBase()
        assert loop is not None


# =============================================================================
# RDLoop — construction
# =============================================================================


class TestRDLoopConstruction:
    @patch("rdagent.components.workflow.rd_loop.logger.log_object")
    @patch("rdagent.components.workflow.rd_loop.import_class")
    def test_init_imports_scenario(self, mock_import: MagicMock, mock_log: MagicMock) -> None:
        from rdagent.components.workflow.rd_loop import RDLoop
        mock_scen = MagicMock()
        mock_import.return_value = mock_scen
        props = _make_mock_prop_setting()
        loop = RDLoop(props)
        assert loop.trace is not None

    @patch("rdagent.components.workflow.rd_loop.logger.log_object")
    @patch("rdagent.components.workflow.rd_loop.import_class")
    def test_init_creates_trace(self, mock_import: MagicMock, mock_log: MagicMock) -> None:
        from rdagent.components.workflow.rd_loop import RDLoop
        mock_import.return_value = MagicMock()
        props = _make_mock_prop_setting()
        loop = RDLoop(props)
        assert hasattr(loop, "trace")

    @patch("rdagent.components.workflow.rd_loop.logger.log_object")
    @patch("rdagent.components.workflow.rd_loop.import_class")
    def test_init_sets_experiment_plan(self, mock_import: MagicMock, mock_log: MagicMock) -> None:
        from rdagent.components.workflow.rd_loop import RDLoop
        mock_import.return_value = MagicMock()
        props = _make_mock_prop_setting()
        loop = RDLoop(props)
        assert "features" in loop.plan
        assert "feature_codes" in loop.plan

    @patch("rdagent.components.workflow.rd_loop.logger.log_object")
    @patch("rdagent.components.workflow.rd_loop.import_class")
    def test_init_with_hypothesis_gen_setting(self, mock_import: MagicMock, mock_log: MagicMock) -> None:
        from rdagent.components.workflow.rd_loop import RDLoop
        mock_import.return_value = MagicMock()
        props = _make_mock_prop_setting(hypothesis_gen="some.path.ClassName")
        loop = RDLoop(props)
        assert loop.hypothesis_gen is not None

    @patch("rdagent.components.workflow.rd_loop.logger.log_object")
    @patch("rdagent.components.workflow.rd_loop.import_class")
    def test_init_without_hypothesis_gen_setting(self, mock_import: MagicMock, mock_log: MagicMock) -> None:
        from rdagent.components.workflow.rd_loop import RDLoop
        mock_import.return_value = MagicMock()
        props = _make_mock_prop_setting()
        props.hypothesis_gen = None
        loop = RDLoop(props)
        assert loop.hypothesis_gen is None

    @patch("rdagent.components.workflow.rd_loop.logger.log_object")
    @patch("rdagent.components.workflow.rd_loop.import_class")
    def test_init_with_coder_setting(self, mock_import: MagicMock, mock_log: MagicMock) -> None:
        from rdagent.components.workflow.rd_loop import RDLoop
        mock_import.return_value = MagicMock()
        props = _make_mock_prop_setting(coder="some.path.Coder")
        loop = RDLoop(props)
        assert loop.coder is not None

    @patch("rdagent.components.workflow.rd_loop.logger.log_object")
    @patch("rdagent.components.workflow.rd_loop.import_class")
    def test_init_with_runner_setting(self, mock_import: MagicMock, mock_log: MagicMock) -> None:
        from rdagent.components.workflow.rd_loop import RDLoop
        mock_import.return_value = MagicMock()
        props = _make_mock_prop_setting(runner="some.path.Runner")
        loop = RDLoop(props)
        assert loop.runner is not None


# =============================================================================
# RDLoop — step methods
# =============================================================================


class TestRDLoopPropose:
    @patch("rdagent.components.workflow.rd_loop.logger.log_object")
    @patch("rdagent.components.workflow.rd_loop.import_class")
    def test_propose_returns_hypothesis(self, mock_import: MagicMock, mock_log: MagicMock) -> None:
        from rdagent.components.workflow.rd_loop import RDLoop
        from rdagent.core.proposal import Hypothesis
        mock_scen = MagicMock()
        mock_import.return_value = mock_scen
        props = _make_mock_prop_setting(hypothesis_gen="some.path")
        loop = RDLoop(props)
        mock_hypo = Hypothesis(hypothesis="test", reason="test",
            concise_reason="cr", concise_observation="co",
            concise_justification="cj", concise_knowledge="ck")
        loop.hypothesis_gen = MagicMock()
        loop.hypothesis_gen.gen.return_value = mock_hypo
        result = loop._propose()
        assert result == mock_hypo

    @patch("rdagent.components.workflow.rd_loop.logger.log_object")
    @patch("rdagent.components.workflow.rd_loop.import_class")
    def test_propose_raises_loop_resume_on_llm_error(self, mock_import: MagicMock, mock_log: MagicMock) -> None:
        from rdagent.components.workflow.rd_loop import RDLoop
        from rdagent.core.exception import LLMUnavailableError
        mock_scen = MagicMock()
        mock_import.return_value = mock_scen
        props = _make_mock_prop_setting(hypothesis_gen="some.path")
        loop = RDLoop(props)
        loop.hypothesis_gen = MagicMock()
        loop.hypothesis_gen.gen.side_effect = LLMUnavailableError("timeout")
        with pytest.raises(loop.LoopResumeError):
            loop._propose()


class TestRDLoopExpGen:
    @patch("rdagent.components.workflow.rd_loop.logger.log_object")
    @patch("rdagent.components.workflow.rd_loop.import_class")
    def test_exp_gen_returns_experiment(self, mock_import: MagicMock, mock_log: MagicMock) -> None:
        from rdagent.components.workflow.rd_loop import RDLoop
        from rdagent.core.proposal import Hypothesis
        mock_scen = MagicMock()
        mock_import.return_value = mock_scen
        props = _make_mock_prop_setting(hypothesis2experiment="some.path")
        loop = RDLoop(props)
        mock_exp = MagicMock()
        loop.hypothesis2experiment = MagicMock()
        loop.hypothesis2experiment.convert.return_value = mock_exp
        hypo = Hypothesis(hypothesis="h", reason="r",
            concise_reason="cr", concise_observation="co",
            concise_justification="cj", concise_knowledge="ck")
        result = loop._exp_gen(hypo)
        assert result == mock_exp


class TestRDLoopSteps:
    @patch("rdagent.components.workflow.rd_loop.logger.log_object")
    @patch("rdagent.components.workflow.rd_loop.import_class")
    def test_direct_exp_gen_yields_dict(self, mock_import: MagicMock, mock_log: MagicMock) -> None:
        from rdagent.components.workflow.rd_loop import RDLoop
        mock_scen = MagicMock()
        mock_scen.patcher = None
        mock_import.return_value = mock_scen
        props = _make_mock_prop_setting(hypothesis_gen="p.HG", hypothesis2experiment="p.H2E")
        loop = RDLoop(props)
        mock_hypo = MagicMock()
        mock_hypo.action = "factor"
        loop.hypothesis_gen = MagicMock()
        loop.hypothesis_gen.gen.return_value = mock_hypo
        loop.hypothesis2experiment = MagicMock()
        mock_exp = MagicMock()
        mock_exp.sub_tasks = []
        mock_exp.based_experiments = None
        loop.hypothesis2experiment.convert.return_value = mock_exp
        result = asyncio.run(loop.direct_exp_gen({}))
        assert "propose" in result
        assert "exp_gen" in result

    @patch("rdagent.components.workflow.rd_loop.logger.log_object")
    @patch("rdagent.components.workflow.rd_loop.import_class")
    def test_coding_calls_coder_develop(self, mock_import: MagicMock, mock_log: MagicMock) -> None:
        from rdagent.components.workflow.rd_loop import RDLoop
        mock_scen = MagicMock()
        mock_import.return_value = mock_scen
        props = _make_mock_prop_setting(coder="p.Coder")
        loop = RDLoop(props)
        loop.coder = MagicMock()
        loop.coder.develop.return_value = MagicMock()
        prev_out = {"direct_exp_gen": {"exp_gen": MagicMock()}}
        loop.coding(prev_out)
        assert loop.coder.develop.called

    @patch("rdagent.components.workflow.rd_loop.logger.log_object")
    @patch("rdagent.components.workflow.rd_loop.import_class")
    def test_running_calls_runner_develop(self, mock_import: MagicMock, mock_log: MagicMock) -> None:
        from rdagent.components.workflow.rd_loop import RDLoop
        mock_scen = MagicMock()
        mock_import.return_value = mock_scen
        props = _make_mock_prop_setting(runner="p.Runner")
        loop = RDLoop(props)
        loop.runner = MagicMock()
        loop.runner.develop.return_value = MagicMock()
        prev_out = {"coding": MagicMock()}
        loop.running(prev_out)
        assert loop.runner.develop.called

    @patch("rdagent.components.workflow.rd_loop.logger.log_object")
    @patch("rdagent.components.workflow.rd_loop.import_class")
    def test_feedback_on_exception_returns_reject_feedback(self, mock_import: MagicMock, mock_log: MagicMock) -> None:
        from rdagent.components.workflow.rd_loop import RDLoop
        from rdagent.core.proposal import HypothesisFeedback
        mock_scen = MagicMock()
        mock_import.return_value = mock_scen
        props = _make_mock_prop_setting(summarizer="p.Summarizer")
        loop = RDLoop(props)
        prev_out = {loop.EXCEPTION_KEY: "test error"}
        result = loop.feedback(prev_out)
        assert isinstance(result, HypothesisFeedback)
        assert result.decision is False

    @patch("rdagent.components.workflow.rd_loop.logger.log_object")
    @patch("rdagent.components.workflow.rd_loop.import_class")
    def test_feedback_normal_path_calls_summarizer(self, mock_import: MagicMock, mock_log: MagicMock) -> None:
        from rdagent.components.workflow.rd_loop import RDLoop
        from rdagent.core.proposal import HypothesisFeedback
        mock_scen = MagicMock()
        mock_import.return_value = mock_scen
        props = _make_mock_prop_setting(summarizer="p.Summarizer")
        loop = RDLoop(props)
        loop.summarizer = MagicMock()
        loop.summarizer.generate_feedback.return_value = HypothesisFeedback(
            reason="ok", decision=True, code_change_summary="done", acceptable=True)
        prev_out = {"running": MagicMock()}
        result = loop.feedback(prev_out)
        assert isinstance(result, HypothesisFeedback)

    @patch("rdagent.components.workflow.rd_loop.logger.log_object")
    @patch("rdagent.components.workflow.rd_loop.import_class")
    def test_record_syncs_trace_dag(self, mock_import: MagicMock, mock_log: MagicMock) -> None:
        from rdagent.components.workflow.rd_loop import RDLoop
        mock_scen = MagicMock()
        mock_import.return_value = mock_scen
        props = _make_mock_prop_setting()
        loop = RDLoop(props)
        loop.trace = MagicMock()
        mock_exp = MagicMock()
        mock_exp.hypothesis = "hypo"
        mock_fb = MagicMock()
        prev_out = {"feedback": mock_fb, "running": mock_exp, loop.LOOP_IDX_KEY: 0}
        loop.record(prev_out)
        loop.trace.sync_dag_parent_and_hist.assert_called_once()

    @patch("rdagent.components.workflow.rd_loop.logger.log_object")
    @patch("rdagent.components.workflow.rd_loop.import_class")
    def test_record_with_none_exp_does_not_crash(self, mock_import: MagicMock, mock_log: MagicMock) -> None:
        from rdagent.components.workflow.rd_loop import RDLoop
        mock_scen = MagicMock()
        mock_import.return_value = mock_scen
        props = _make_mock_prop_setting()
        loop = RDLoop(props)
        loop.trace = MagicMock()
        prev_out = {"feedback": MagicMock(), "running": MagicMock(hypothesis=None), loop.LOOP_IDX_KEY: 0}
        loop.record(prev_out)
        loop.trace.sync_dag_parent_and_hist.assert_not_called()


# =============================================================================
# RDLoop — interaction methods
# =============================================================================


class TestRDLoopInteractions:
    @patch("rdagent.components.workflow.rd_loop.logger.log_object")
    @patch("rdagent.components.workflow.rd_loop.import_class")
    def test_set_interactor_stores_queues(self, mock_import: MagicMock, mock_log: MagicMock) -> None:
        from rdagent.components.workflow.rd_loop import RDLoop
        mock_scen = MagicMock()
        mock_import.return_value = mock_scen
        props = _make_mock_prop_setting()
        loop = RDLoop(props)
        q1, q2 = Queue(), Queue()
        loop._set_interactor(q1, q2)
        assert loop.user_request_q is q1
        assert loop.user_response_q is q2

    @patch("rdagent.components.workflow.rd_loop.logger.log_object")
    @patch("rdagent.components.workflow.rd_loop.import_class")
    def test_interact_hypo_no_queues_returns_original(self, mock_import: MagicMock, mock_log: MagicMock) -> None:
        from rdagent.components.workflow.rd_loop import RDLoop
        from rdagent.core.proposal import Hypothesis
        mock_scen = MagicMock()
        mock_import.return_value = mock_scen
        props = _make_mock_prop_setting()
        loop = RDLoop(props)
        hypo = Hypothesis(hypothesis="h", reason="r",
            concise_reason="cr", concise_observation="co",
            concise_justification="cj", concise_knowledge="ck")
        result = loop._interact_hypo(hypo)
        assert result is hypo

    @patch("rdagent.components.workflow.rd_loop.logger.log_object")
    @patch("rdagent.components.workflow.rd_loop.import_class")
    def test_interact_feedback_no_queues_returns_original(self, mock_import: MagicMock, mock_log: MagicMock) -> None:
        from rdagent.components.workflow.rd_loop import RDLoop
        from rdagent.core.proposal import HypothesisFeedback
        mock_scen = MagicMock()
        mock_import.return_value = mock_scen
        props = _make_mock_prop_setting()
        loop = RDLoop(props)
        fb = HypothesisFeedback(reason="r", decision=True, code_change_summary="ok", acceptable=True)
        result = loop._interact_feedback(fb)
        assert result is fb


# =============================================================================
# BasePropSetting
# =============================================================================


class TestBasePropSetting:
    def test_base_prop_setting_is_pydantic_model(self) -> None:
        from rdagent.components.workflow.conf import BasePropSetting
        from pydantic import BaseModel
        assert issubclass(BasePropSetting, BaseModel)


# =============================================================================
# Hypothesis and HypothesisFeedback
# =============================================================================


class TestHypothesis:
    def test_construction_with_minimal_fields(self) -> None:
        from rdagent.core.proposal import Hypothesis
        h = Hypothesis(hypothesis="test", reason="because",
            concise_reason="cr", concise_observation="co",
            concise_justification="cj", concise_knowledge="ck")
        assert h.hypothesis == "test"
        assert h.reason == "because"

    def test_has_conciseness_fields(self) -> None:
        from rdagent.core.proposal import Hypothesis
        h = Hypothesis(hypothesis="h", reason="r",
            concise_reason="cr", concise_observation="co",
            concise_justification="cj", concise_knowledge="ck")
        assert h.concise_reason == "cr"
        assert h.concise_observation == "co"

    def test_pickle_safety(self) -> None:
        from rdagent.core.proposal import Hypothesis
        h = Hypothesis(hypothesis="h", reason="r",
            concise_reason="cr", concise_observation="co",
            concise_justification="cj", concise_knowledge="ck")
        data = pickle.dumps(h)
        h2 = pickle.loads(data)
        assert h2.hypothesis == "h"
        assert h2.concise_knowledge == "ck"

    def test_dict_conversion(self) -> None:
        from rdagent.core.proposal import Hypothesis
        h = Hypothesis(hypothesis="h", reason="r",
            concise_reason="cr", concise_observation="co",
            concise_justification="cj", concise_knowledge="ck")
        d = h.__dict__
        h2 = type(h)(**d)
        assert h2.hypothesis == h.hypothesis


class TestHypothesisFeedback:
    def test_construction_with_all_fields(self) -> None:
        from rdagent.core.proposal import HypothesisFeedback
        fb = HypothesisFeedback(reason="good", decision=True, code_change_summary="fixed", acceptable=True)
        assert fb.reason == "good"
        assert fb.decision is True

    def test_default_values(self) -> None:
        from rdagent.core.proposal import HypothesisFeedback
        fb = HypothesisFeedback(reason="reason", decision=False)
        assert fb.decision is False

    def test_pickle_safety(self) -> None:
        from rdagent.core.proposal import HypothesisFeedback
        fb = HypothesisFeedback(reason="r", decision=True, code_change_summary="c", acceptable=True)
        data = pickle.dumps(fb)
        fb2 = pickle.loads(data)
        assert fb2.decision is True


# =============================================================================
# Trace
# =============================================================================


class TestTrace:
    def test_trace_construction(self) -> None:
        from rdagent.core.proposal import Trace
        trace = Trace(scen=None)
        assert trace is not None

    def test_trace_has_hist_attribute(self) -> None:
        from rdagent.core.proposal import Trace
        trace = Trace(scen=None)
        assert hasattr(trace, "hist")
        assert isinstance(trace.hist, list)

    def test_trace_sync_dag_parent_and_hist(self) -> None:
        from rdagent.core.proposal import Trace
        trace = Trace(scen=None)
        exp = MagicMock()
        exp.based_experiments = []
        exp.hypothesis = "hypo"
        fb = MagicMock()
        trace.sync_dag_parent_and_hist((exp, fb), 0)
        assert len(trace.hist) > 0

    def test_trace_pickle_safety(self) -> None:
        from rdagent.core.proposal import Trace
        trace = Trace(scen=None)
        trace.hist = [("entry",)]
        data = pickle.dumps(trace)
        trace2 = pickle.loads(data)
        assert len(trace2.hist) == 1


# =============================================================================
# HypothesisGen, Hypothesis2Experiment, Experiment2Feedback
# =============================================================================


class TestProposalClasses:
    def test_hypothesis_gen_is_importable(self) -> None:
        from rdagent.core.proposal import HypothesisGen
        assert HypothesisGen is not None

    def test_hypothesis2experiment_is_importable(self) -> None:
        from rdagent.core.proposal import Hypothesis2Experiment
        assert Hypothesis2Experiment is not None

    def test_experiment2feedback_is_importable(self) -> None:
        from rdagent.core.proposal import Experiment2Feedback
        assert Experiment2Feedback is not None


# =============================================================================
# Developer
# =============================================================================


class TestDeveloper:
    def test_developer_is_importable(self) -> None:
        from rdagent.core.developer import Developer
        assert Developer is not None

    def test_developer_stores_scenario(self) -> None:
        from rdagent.core.developer import Developer
        from rdagent.core.experiment import ASpecificExp
        class ConcreteDev(Developer[ASpecificExp]):
            def develop(self, exp: ASpecificExp) -> ASpecificExp:
                return exp
        scen = MagicMock()
        dev = ConcreteDev(scen)
        assert dev.scen is scen


# =============================================================================
# Scenario base class
# =============================================================================


class TestScenarioBase:
    def test_scenario_is_abstract(self) -> None:
        from rdagent.core.scenario import Scenario
        assert hasattr(Scenario, "__abstractmethods__")

    def test_scenario_has_required_properties(self) -> None:
        from rdagent.core.scenario import Scenario
        assert hasattr(Scenario, "background")
        assert hasattr(Scenario, "rich_style_description")
        assert hasattr(Scenario, "source_data")


# =============================================================================
# Qlib utilities
# =============================================================================


class TestQlibUtils:
    def test_validate_qlib_features_importable(self) -> None:
        from rdagent.utils.qlib import validate_qlib_features
        assert callable(validate_qlib_features)

    def test_validate_valid_features(self) -> None:
        from rdagent.utils.qlib import validate_qlib_features
        result = validate_qlib_features(["$close", "$high / $low"])
        assert isinstance(result, bool)

    def test_validate_empty_list(self) -> None:
        from rdagent.utils.qlib import validate_qlib_features
        result = validate_qlib_features([])
        assert isinstance(result, bool)

    def test_alpha20_importable(self) -> None:
        from rdagent.utils.qlib import ALPHA20
        assert isinstance(ALPHA20, dict)
        assert len(ALPHA20) > 0

    @pytest.mark.parametrize("features", [
        ["$close"], ["$open", "$high", "$low", "$close"], ["$close / $open", "$high - $low"], [],
    ])
    def test_validate_qlib_features_variants(self, features: list) -> None:
        from rdagent.utils.qlib import validate_qlib_features
        result = validate_qlib_features(features)
        assert isinstance(result, bool)


# =============================================================================
# Experiment classes
# =============================================================================


class TestExperimentClasses:
    def test_task_is_importable(self) -> None:
        from rdagent.core.experiment import Task
        assert Task is not None

    def test_workspace_is_importable(self) -> None:
        from rdagent.core.experiment import Workspace
        assert Workspace is not None

    def test_fb_workspace_is_importable(self) -> None:
        from rdagent.core.experiment import FBWorkspace
        assert FBWorkspace is not None

    def test_fb_workspace_inject_files(self) -> None:
        from rdagent.core.experiment import FBWorkspace
        ws = FBWorkspace()
        ws.inject_files(**{"factor.py": "def calc(): pass"})
        code = ws.all_codes
        assert "def calc" in code

    def test_fb_workspace_pickle_safety(self) -> None:
        from rdagent.core.experiment import FBWorkspace
        ws = FBWorkspace()
        ws.inject_files(**{"factor.py": "x=1"})
        data = pickle.dumps(ws)
        ws2 = pickle.loads(data)
        assert isinstance(ws2, FBWorkspace)


# =============================================================================
# Evolving framework imports
# =============================================================================


class TestEvolvingFrameworkImports:
    @pytest.mark.parametrize("cls_name,module_path", [
        ("EvolvableSubjects", "rdagent.core.evolving_framework"),
        ("EvolvingKnowledgeBase", "rdagent.core.evolving_framework"),
        ("EvoStep", "rdagent.core.evolving_framework"),
        ("Knowledge", "rdagent.core.evolving_framework"),
        ("QueriedKnowledge", "rdagent.core.evolving_framework"),
        ("RAGStrategy", "rdagent.core.evolving_framework"),
        ("RAGEvaluator", "rdagent.core.evolving_agent"),
    ])
    def test_class_importable(self, cls_name: str, module_path: str) -> None:
        import importlib
        mod = importlib.import_module(module_path)
        assert hasattr(mod, cls_name)


# =============================================================================
# EvoStep — dataclass behavior
# =============================================================================


class TestEvoStep:
    def test_default_construction(self) -> None:
        from rdagent.core.evolving_framework import EvoStep
        es = EvoStep(evolvable_subjects="mock_evo")
        assert es.evolvable_subjects == "mock_evo"
        assert es.queried_knowledge is None
        assert es.feedback is None

    def test_full_construction(self) -> None:
        from rdagent.core.evolving_framework import EvoStep, QueriedKnowledge
        qk = QueriedKnowledge()
        es = EvoStep(evolvable_subjects="evo", queried_knowledge=qk, feedback="fb")
        assert es.queried_knowledge is qk
        assert es.feedback == "fb"

    def test_equality_by_reference(self) -> None:
        from rdagent.core.evolving_framework import EvoStep
        es1 = EvoStep(evolvable_subjects="a")
        es2 = EvoStep(evolvable_subjects="a")
        assert es1 == es2

    def test_pickle_safety(self) -> None:
        from rdagent.core.evolving_framework import EvoStep
        es = EvoStep(evolvable_subjects="subj", feedback="good")
        data = pickle.dumps(es)
        es2 = pickle.loads(data)
        assert es2.evolvable_subjects == "subj"
        assert es2.feedback == "good"


# =============================================================================
# import_class utility
# =============================================================================


class TestImportClass:
    def test_import_class_is_callable(self) -> None:
        from rdagent.core.utils import import_class
        assert callable(import_class)

    def test_import_class_resolves_known_class(self) -> None:
        from rdagent.core.utils import import_class
        cls = import_class("rdagent.core.proposal.Hypothesis")
        from rdagent.core.proposal import Hypothesis
        assert cls is Hypothesis

    def test_import_class_raises_on_bad_path(self) -> None:
        from rdagent.core.utils import import_class
        with pytest.raises((ValueError, ImportError, ModuleNotFoundError)):
            import_class("nonexistent.module.ClassName")


# =============================================================================
# LLMUnavailableError
# =============================================================================


class TestLLMUnavailableError:
    def test_is_importable(self) -> None:
        from rdagent.core.exception import LLMUnavailableError
        assert issubclass(LLMUnavailableError, Exception)

    def test_can_be_raised_and_caught(self) -> None:
        from rdagent.core.exception import LLMUnavailableError
        with pytest.raises(LLMUnavailableError):
            raise LLMUnavailableError("test error")

    def test_pickle_safety(self) -> None:
        from rdagent.core.exception import LLMUnavailableError
        e = LLMUnavailableError("pickle me")
        data = pickle.dumps(e)
        e2 = pickle.loads(data)
        assert str(e2) == "pickle me"


# =============================================================================
# Pickle safety for combined workflow objects
# =============================================================================


class TestPickleSafetyComposite:
    def test_combined_workflow_objects_pickle(self) -> None:
        from rdagent.core.proposal import Hypothesis, HypothesisFeedback, Trace
        h = Hypothesis(hypothesis="h", reason="r",
            concise_reason="cr", concise_observation="co",
            concise_justification="cj", concise_knowledge="ck")
        fb = HypothesisFeedback(reason="r", decision=True, code_change_summary="ok", acceptable=True)
        trace = Trace(scen=None)
        trace.hist = []
        bundle = {"hypothesis": h, "feedback": fb, "trace": trace}
        data = pickle.dumps(bundle)
        bundle2 = pickle.loads(data)
        assert bundle2["hypothesis"].hypothesis == "h"
