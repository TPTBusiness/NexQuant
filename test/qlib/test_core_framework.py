"""Tests for rdagent.core — the core framework abstractions."""

from __future__ import annotations

import sys
from pathlib import Path
from unittest.mock import MagicMock

import pytest

PROJECT_ROOT = Path(__file__).parent.parent.parent
sys.path.insert(0, str(PROJECT_ROOT))


# =============================================================================
# Feedback base class
# =============================================================================


class TestFeedback:
    def test_default_is_acceptable_returns_true(self):
        from rdagent.core.evaluation import Feedback
        fb = Feedback()
        assert fb.is_acceptable() is True

    def test_default_finished_returns_true(self):
        from rdagent.core.evaluation import Feedback
        fb = Feedback()
        assert fb.finished() is True

    def test_default_bool_is_true(self):
        from rdagent.core.evaluation import Feedback
        fb = Feedback()
        assert bool(fb) is True


# =============================================================================
# EvoStep dataclass
# =============================================================================


class TestEvoStep:
    def test_default_construction(self):
        from rdagent.core.evolving_framework import EvoStep
        es = EvoStep(evolvable_subjects="mock_evo")
        assert es.evolvable_subjects == "mock_evo"
        assert es.queried_knowledge is None
        assert es.feedback is None

    def test_full_construction(self):
        from rdagent.core.evolving_framework import EvoStep, QueriedKnowledge
        qk = QueriedKnowledge()
        es = EvoStep(evolvable_subjects="evo", queried_knowledge=qk, feedback="fb")
        assert es.queried_knowledge is qk
        assert es.feedback == "fb"

    def test_equality_by_reference(self):
        from rdagent.core.evolving_framework import EvoStep
        es1 = EvoStep(evolvable_subjects="a")
        es2 = EvoStep(evolvable_subjects="a")
        assert es1 == es2


# =============================================================================
# Scenario base class
# =============================================================================


class TestScenario:
    def test_source_data_default_returns_empty_string(self):
        from rdagent.core.scenario import Scenario
        
        class MinimalScenario(Scenario):
            @property
            def background(self) -> str: return "bg"
            @property
            def rich_style_description(self) -> str: return "rich"
            def get_scenario_all_desc(self, **kwargs) -> str: return "all"
            def get_runtime_environment(self) -> str: return "env"

        scen = MinimalScenario()
        assert scen.source_data == ""

    def test_source_data_property_calls_get_source_data_desc(self):
        from rdagent.core.scenario import Scenario

        class FakeScenario(Scenario):
            @property
            def background(self) -> str: return "bg"
            @property
            def rich_style_description(self) -> str: return "rich"
            def get_scenario_all_desc(self, **kwargs) -> str: return "all"
            def get_runtime_environment(self) -> str: return "env"
            def get_source_data_desc(self, task=None) -> str: return "custom_data"

        scen = FakeScenario()
        assert scen.source_data == "custom_data"


# =============================================================================
# EvolvingStrategy base class
# =============================================================================


class TestEvolvingStrategy:
    def test_init_stores_scenario(self):
        from rdagent.core.evolving_framework import EvolvingStrategy
        
        class MinimalStrategy(EvolvingStrategy):
            def evolve_iter(self, evo, queried_knowledge=None, evolving_trace=None):
                yield evo
        
        mock_scen = MagicMock()
        es = MinimalStrategy(mock_scen)
        assert es.scen is mock_scen


# =============================================================================
# IterEvaluator base class
# =============================================================================


class TestIterEvaluator:
    def test_evaluate_returns_feedback(self):
        from rdagent.core.evaluation import Feedback
        from rdagent.core.evolving_framework import IterEvaluator, EvolvableSubjects

        class MyFeedback(Feedback):
            pass

        class MyEvaluator(IterEvaluator):
            def evaluate_iter(self):
                evo = yield MyFeedback()
                yield MyFeedback()
                return MyFeedback()

        eva = MyEvaluator()
        result = eva.evaluate(EvolvableSubjects())
        assert isinstance(result, MyFeedback)

    def test_evaluate_iter_send_none_stops(self):
        """Sending None mid-iteration triggers StopIteration with final feedback."""
        from rdagent.core.evaluation import Feedback
        from rdagent.core.evolving_framework import IterEvaluator

        class MyEvaluator(IterEvaluator):
            def evaluate_iter(self):
                yield Feedback()            # kick-off (none)
                evo_next = yield Feedback()  # partial eval
                if evo_next is None:
                    return Feedback()        # early return
                return Feedback()            # normal path

        eva = MyEvaluator()
        gen = eva.evaluate_iter()
        next(gen)           # kick-off → first Feedback
        gen.send("any")     # evo gets "any" → second Feedback (evo_next NOT assigned yet)
        with pytest.raises(StopIteration):
            gen.send(None)  # evo_next = None → return → StopIteration


# =============================================================================
# Developer base class
# =============================================================================


class TestDeveloper:
    def test_develop_raises_not_implemented(self):
        from rdagent.core.developer import Developer

        class MinimalDeveloper(Developer):
            def develop(self, exp):
                return super().develop(exp)

        dev = MinimalDeveloper(MagicMock())
        with pytest.raises(NotImplementedError):
            dev.develop(MagicMock())


# =============================================================================
# Knowledge / QueriedKnowledge
# =============================================================================


class TestKnowledgeHierarchy:
    def test_knowledge_pass_through(self):
        from rdagent.core.evolving_framework import Knowledge, QueriedKnowledge
        k = Knowledge()
        qk = QueriedKnowledge()
        assert isinstance(k, Knowledge)
        assert isinstance(qk, QueriedKnowledge)


# =============================================================================
# EvolvingAgent (abstract interface)
# =============================================================================


class TestEvolvingAgent:
    def test_ragevo_agent_init(self):
        from rdagent.core.evolving_agent import RAGEvoAgent
        mock_strategy = MagicMock()
        mock_rag = MagicMock()
        agent = RAGEvoAgent.__new__(RAGEvoAgent)
        RAGEvoAgent.__init__(agent, max_loop=5, evolving_strategy=mock_strategy, rag=mock_rag)
        assert agent.max_loop == 5
        assert agent.evolving_strategy is mock_strategy
        assert agent.rag is mock_rag

    def test_ragevo_agent_default_knowledge_flags(self):
        from rdagent.core.evolving_agent import RAGEvoAgent
        agent = RAGEvoAgent.__new__(RAGEvoAgent)
        RAGEvoAgent.__init__(agent, max_loop=3, evolving_strategy=MagicMock(), rag=MagicMock())
        assert agent.with_knowledge is False
        assert agent.knowledge_self_gen is False
        assert agent.enable_filelock is False

    def test_ragevo_agent_with_knowledge_enabled(self):
        from rdagent.core.evolving_agent import RAGEvoAgent
        agent = RAGEvoAgent.__new__(RAGEvoAgent)
        RAGEvoAgent.__init__(
            agent, max_loop=3, evolving_strategy=MagicMock(), rag=MagicMock(),
            with_knowledge=True, knowledge_self_gen=True,
            enable_filelock=True, filelock_path="/tmp/test.lock",
        )
        assert agent.with_knowledge is True
        assert agent.knowledge_self_gen is True
        assert agent.enable_filelock is True
        assert agent.filelock_path == "/tmp/test.lock"


# =============================================================================
# EvolvableSubjects clone
# =============================================================================


class TestEvolvableSubjects:
    def test_clone_produces_deep_copy(self):
        from rdagent.core.evolving_framework import EvolvableSubjects
        es = EvolvableSubjects()
        clone = es.clone()
        assert clone is not es
        assert type(clone) is type(es)
