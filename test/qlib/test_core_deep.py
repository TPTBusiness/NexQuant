"""Deep tests for rdagent.core: developer.py, evaluation.py, and related abstractions."""

from __future__ import annotations

import pickle
import sys
from pathlib import Path
from typing import Any
from unittest.mock import MagicMock, patch

import pytest

PROJECT_ROOT = Path(__file__).parent.parent.parent
sys.path.insert(0, str(PROJECT_ROOT))


# =============================================================================
# Import safety
# =============================================================================

CORE_MODULES = [
    "rdagent.core.developer",
    "rdagent.core.evaluation",
    "rdagent.core.experiment",
    "rdagent.core.proposal",
    "rdagent.core.scenario",
    "rdagent.core.evolving_framework",
    "rdagent.core.evolving_agent",
    "rdagent.core.conf",
    "rdagent.core.exception",
    "rdagent.core.utils",
]


class TestCoreModuleImports:
    @pytest.mark.parametrize("module_name", CORE_MODULES)
    def test_module_importable(self, module_name: str) -> None:
        import importlib
        mod = importlib.import_module(module_name)
        assert mod is not None


# =============================================================================
# Feedback
# =============================================================================


class TestFeedback:
    def test_default_is_acceptable_returns_true(self) -> None:
        from rdagent.core.evaluation import Feedback
        fb = Feedback()
        assert fb.is_acceptable() is True

    def test_default_finished_returns_true(self) -> None:
        from rdagent.core.evaluation import Feedback
        fb = Feedback()
        assert fb.finished() is True

    def test_default_bool_is_true(self) -> None:
        from rdagent.core.evaluation import Feedback
        fb = Feedback()
        assert bool(fb) is True

    def test_is_acceptable_calls_bool(self) -> None:
        from rdagent.core.evaluation import Feedback

        class FalseFeedback(Feedback):
            def __bool__(self) -> bool:
                return False

        fb = FalseFeedback()
        assert fb.is_acceptable() is False

    def test_finished_can_be_overridden(self) -> None:
        from rdagent.core.evaluation import Feedback

        class CustomFinish(Feedback):
            def __bool__(self) -> bool:
                return False

            def finished(self) -> bool:
                return True

        fb = CustomFinish()
        assert fb.finished() is True
        assert bool(fb) is False

    def test_pickle_safety(self) -> None:
        from rdagent.core.evaluation import Feedback
        fb = Feedback()
        data = pickle.dumps(fb)
        fb2 = pickle.loads(data)
        assert isinstance(fb2, Feedback)
        assert bool(fb2) is True


# =============================================================================
# Evaluator / EvaluableObj
# =============================================================================


class TestEvaluator:
    def test_evaluator_is_abstract(self) -> None:
        from rdagent.core.evaluation import Evaluator
        assert hasattr(Evaluator, "evaluate")

    def test_concrete_evaluator_must_implement_evaluate(self) -> None:
        from rdagent.core.evaluation import Evaluator, Feedback

        class ConcreteEvaluator(Evaluator):
            def evaluate(self, eo) -> Feedback:
                return Feedback()

        ev = ConcreteEvaluator()
        result = ev.evaluate(None)
        assert isinstance(result, Feedback)

    @pytest.mark.parametrize("input_eo", [None, "string", {"key": "value"}, [1, 2, 3]])
    def test_concrete_evaluator_accepts_any_input(self, input_eo: Any) -> None:
        from rdagent.core.evaluation import Evaluator, Feedback

        class FlexibleEvaluator(Evaluator):
            def evaluate(self, eo) -> Feedback:
                return Feedback()

        ev = FlexibleEvaluator()
        result = ev.evaluate(input_eo)
        assert isinstance(result, Feedback)


class TestEvaluableObj:
    def test_evaluable_obj_is_importable(self) -> None:
        from rdagent.core.evaluation import EvaluableObj
        assert EvaluableObj is not None

    def test_evaluable_obj_can_be_instantiated(self) -> None:
        from rdagent.core.evaluation import EvaluableObj
        obj = EvaluableObj()
        assert isinstance(obj, EvaluableObj)


# =============================================================================
# Developer
# =============================================================================


class TestDeveloperBase:
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

    def test_develop_modifies_in_place(self) -> None:
        from rdagent.core.developer import Developer
        from rdagent.core.experiment import ASpecificExp

        class ModifyingDev(Developer[ASpecificExp]):
            def develop(self, exp: ASpecificExp) -> ASpecificExp:
                exp._modified = True
                return exp

        dev = ModifyingDev(MagicMock())
        exp = MagicMock()
        result = dev.develop(exp)
        assert result._modified is True


# =============================================================================
# Experiment classes
# =============================================================================


class TestExperiment:
    def test_task_has_get_task_information(self) -> None:
        from rdagent.core.experiment import Task
        assert hasattr(Task, "get_task_information")

    def test_workspace_has_execute(self) -> None:
        from rdagent.core.experiment import Workspace
        assert hasattr(Workspace, "execute")

    def test_fb_workspace_inject_files(self) -> None:
        from rdagent.core.experiment import FBWorkspace
        ws = FBWorkspace()
        ws.inject_files(**{"factor.py": "def calc(): pass"})
        code = ws.all_codes
        assert "def calc" in code

    def test_fb_workspace_copy_returns_new_instance(self) -> None:
        from rdagent.core.experiment import FBWorkspace
        ws = FBWorkspace()
        ws.inject_files(**{"test.py": "x=1"})
        ws2 = ws.copy()
        assert ws2 is not ws
        assert ws2.all_codes == ws.all_codes

    def test_fb_workspace_pickle_safety(self) -> None:
        from rdagent.core.experiment import FBWorkspace
        ws = FBWorkspace()
        ws.inject_files(**{"factor.py": "x=1", "utils.py": "y=2"})
        data = pickle.dumps(ws)
        ws2 = pickle.loads(data)
        assert isinstance(ws2, FBWorkspace)

    @pytest.mark.parametrize("files", [
        {},
        {"a.py": ""},
        {"a.py": "x=1", "b.py": "y=2", "c.py": "z=3"},
    ])
    def test_fb_workspace_file_variants(self, files: dict) -> None:
        from rdagent.core.experiment import FBWorkspace
        ws = FBWorkspace()
        ws.inject_files(**files)
        assert isinstance(ws.all_codes, str)

    def test_aspecific_exp_is_importable(self) -> None:
        from rdagent.core.experiment import ASpecificExp
        assert ASpecificExp is not None


# =============================================================================
# EvoStep
# =============================================================================


class TestEvoStep:
    def test_default_construction(self) -> None:
        from rdagent.core.evolving_framework import EvoStep
        es = EvoStep(evolvable_subjects="evo")
        assert es.evolvable_subjects == "evo"
        assert es.queried_knowledge is None
        assert es.feedback is None

    def test_full_construction(self) -> None:
        from rdagent.core.evolving_framework import EvoStep, QueriedKnowledge
        qk = QueriedKnowledge()
        es = EvoStep(evolvable_subjects="subj", queried_knowledge=qk, feedback="fb")
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

    def test_with_none_values(self) -> None:
        from rdagent.core.evolving_framework import EvoStep
        es = EvoStep(evolvable_subjects=None, queried_knowledge=None, feedback=None)
        assert es.evolvable_subjects is None


# =============================================================================
# Knowledge and QueriedKnowledge
# =============================================================================


class TestKnowledge:
    def test_knowledge_is_importable(self) -> None:
        from rdagent.core.evolving_framework import Knowledge
        assert Knowledge is not None

    def test_knowledge_can_be_instantiated(self) -> None:
        from rdagent.core.evolving_framework import Knowledge
        k = Knowledge()
        assert isinstance(k, Knowledge)

    def test_knowledge_pickle_safety(self) -> None:
        from rdagent.core.evolving_framework import Knowledge
        k = Knowledge()
        data = pickle.dumps(k)
        k2 = pickle.loads(data)
        assert isinstance(k2, Knowledge)


class TestQueriedKnowledge:
    def test_default_construction(self) -> None:
        from rdagent.core.evolving_framework import QueriedKnowledge
        qk = QueriedKnowledge()
        assert isinstance(qk, QueriedKnowledge)

    def test_pickle_safety(self) -> None:
        from rdagent.core.evolving_framework import QueriedKnowledge
        qk = QueriedKnowledge()
        data = pickle.dumps(qk)
        qk2 = pickle.loads(data)
        assert isinstance(qk2, QueriedKnowledge)


# =============================================================================
# RAGStrategy / RAGEvaluator
# =============================================================================


class TestRAGStrategy:
    def test_rag_strategy_has_methods(self) -> None:
        from rdagent.core.evolving_framework import RAGStrategy
        assert hasattr(RAGStrategy, "generate_knowledge")
        assert hasattr(RAGStrategy, "query")


class TestRAGEvaluator:
    def test_rage_evaluator_is_importable(self) -> None:
        from rdagent.core.evolving_agent import RAGEvaluator
        assert RAGEvaluator is not None


# =============================================================================
# EvolvingKnowledgeBase
# =============================================================================


class TestEvolvingKnowledgeBase:
    def test_has_query_method(self) -> None:
        from rdagent.core.evolving_framework import EvolvingKnowledgeBase
        assert hasattr(EvolvingKnowledgeBase, "query")

    def test_takes_optional_path_argument(self) -> None:
        from rdagent.core.evolving_framework import EvolvingKnowledgeBase
        kb = EvolvingKnowledgeBase(path=Path("/tmp/test"))
        assert kb.path == Path("/tmp/test")


# =============================================================================
# EvolvableSubjects
# =============================================================================


class TestEvolvableSubjects:
    def test_evolvable_subjects_is_importable(self) -> None:
        from rdagent.core.evolving_framework import EvolvableSubjects
        assert EvolvableSubjects is not None

    def test_evolvable_subjects_has_clone_method(self) -> None:
        from rdagent.core.evolving_framework import EvolvableSubjects
        assert hasattr(EvolvableSubjects, "clone")

    def test_evolvable_subjects_is_instantiable(self) -> None:
        from rdagent.core.evolving_framework import EvolvableSubjects
        es = EvolvableSubjects()
        assert es is not None


# =============================================================================
# Scenario
# =============================================================================


class TestScenario:
    def test_scenario_is_abstract(self) -> None:
        from rdagent.core.scenario import Scenario
        assert hasattr(Scenario, "__abstractmethods__")

    def test_scenario_has_rich_style_description(self) -> None:
        from rdagent.core.scenario import Scenario
        assert hasattr(Scenario, "rich_style_description")

    def test_scenario_has_background(self) -> None:
        from rdagent.core.scenario import Scenario
        assert hasattr(Scenario, "background")

    def test_scenario_source_data_default(self) -> None:
        from rdagent.core.scenario import Scenario

        class NoDataScen(Scenario):
            @property
            def background(self) -> str: return "bg"
            @property
            def rich_style_description(self) -> str: return "rd"
            def get_scenario_all_desc(self, **kw) -> str: return "ad"
            def get_runtime_environment(self) -> str: return "re"

        scen = NoDataScen()
        assert scen.source_data == ""


# =============================================================================
# Exception classes
# =============================================================================


class TestExceptionClasses:
    def test_llm_unavailable_error_is_exception(self) -> None:
        from rdagent.core.exception import LLMUnavailableError
        assert issubclass(LLMUnavailableError, Exception)

    def test_llm_unavailable_error_string_message(self) -> None:
        from rdagent.core.exception import LLMUnavailableError
        with pytest.raises(LLMUnavailableError, match="test message"):
            raise LLMUnavailableError("test message")

    def test_llm_unavailable_error_pickle(self) -> None:
        from rdagent.core.exception import LLMUnavailableError
        e = LLMUnavailableError("pickle me")
        data = pickle.dumps(e)
        e2 = pickle.loads(data)
        assert str(e2) == "pickle me"

    @pytest.mark.parametrize("message", ["", "short", "multi\nline\nmessage"])
    def test_llm_unavailable_error_message_variants(self, message: str) -> None:
        from rdagent.core.exception import LLMUnavailableError
        e = LLMUnavailableError(message)
        assert str(e) == message


# =============================================================================
# Conf module
# =============================================================================


class TestConfModule:
    def test_rd_agent_settings_is_importable(self) -> None:
        from rdagent.core.conf import RD_AGENT_SETTINGS
        assert RD_AGENT_SETTINGS is not None

    def test_rd_agent_settings_has_multi_proc_n(self) -> None:
        from rdagent.core.conf import RD_AGENT_SETTINGS
        assert hasattr(RD_AGENT_SETTINGS, "multi_proc_n")

    def test_rd_agent_settings_get_max_parallel(self) -> None:
        from rdagent.core.conf import RD_AGENT_SETTINGS
        result = RD_AGENT_SETTINGS.get_max_parallel()
        assert isinstance(result, int)


# =============================================================================
# Utils module
# =============================================================================


class TestCoreUtils:
    def test_import_class_with_valid_path(self) -> None:
        from rdagent.core.utils import import_class
        cls = import_class("rdagent.core.evaluation.Feedback")
        from rdagent.core.evaluation import Feedback
        assert cls is Feedback

    def test_import_class_raises_on_invalid_path(self) -> None:
        from rdagent.core.utils import import_class
        with pytest.raises((ValueError, ImportError, ModuleNotFoundError)):
            import_class("rdagent.nonexistent.Class")


# =============================================================================
# Pickle safety
# =============================================================================


class TestPickleSafety:
    def test_feedback_list_picklable(self) -> None:
        from rdagent.core.evaluation import Feedback
        items = [Feedback(), Feedback(), Feedback()]
        data = pickle.dumps(items)
        loaded = pickle.loads(data)
        assert [bool(x) for x in loaded] == [True, True, True]

    def test_evo_step_with_none_feedback_pickle(self) -> None:
        from rdagent.core.evolving_framework import EvoStep
        es = EvoStep(evolvable_subjects="s", queried_knowledge=None, feedback=None)
        data = pickle.dumps(es)
        es2 = pickle.loads(data)
        assert es2.feedback is None


# =============================================================================
# Edge cases
# =============================================================================


class TestEdgeCases:
    def test_evo_step_with_all_none(self) -> None:
        from rdagent.core.evolving_framework import EvoStep
        es = EvoStep(evolvable_subjects=None, queried_knowledge=None, feedback=None)
        assert es.evolvable_subjects is None

    def test_feedback_bool_edge_cases(self) -> None:
        from rdagent.core.evaluation import Feedback

        class AlwaysTrue(Feedback):
            def __bool__(self): return True

        class AlwaysFalse(Feedback):
            def __bool__(self): return False

        assert bool(AlwaysTrue()) is True
        assert bool(AlwaysFalse()) is False


# =============================================================================
# Generic types
# =============================================================================


class TestGenericTypes:
    def test_aspecific_exp_importable(self) -> None:
        from rdagent.core.experiment import ASpecificExp
        assert ASpecificExp is not None

    def test_developer_is_importable(self) -> None:
        from rdagent.core.developer import Developer
        assert Developer is not None
