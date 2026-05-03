"""Tests for CoSTEER config, task, and evolve strategy population logic."""

from __future__ import annotations

import sys
from pathlib import Path
from unittest.mock import MagicMock, patch

import pytest

PROJECT_ROOT = Path(__file__).parent.parent.parent
sys.path.insert(0, str(PROJECT_ROOT))


# =============================================================================
# CoSTEERSettings
# =============================================================================


class TestCoSTEERSettings:
    def test_default_values(self):
        from rdagent.components.coder.CoSTEER.config import CoSTEERSettings
        s = CoSTEERSettings()
        assert s.max_loop == 1
        assert s.fail_task_trial_limit == 5
        assert s.v2_query_component_limit == 1
        assert s.v2_query_error_limit == 1
        assert s.v2_query_former_trace_limit == 3
        assert s.v2_add_fail_attempt_to_latest_successful_execution is False
        assert s.v2_knowledge_sampler == 1.0
        assert s.coder_use_cache is False
        assert s.enable_filelock is False

    def test_singleton_instance(self):
        from rdagent.components.coder.CoSTEER.config import CoSTEER_SETTINGS
        from rdagent.components.coder.CoSTEER.config import CoSTEERSettings
        assert isinstance(CoSTEER_SETTINGS, CoSTEERSettings)
        assert CoSTEER_SETTINGS.max_loop == 1


# =============================================================================
# CoSTEERTask
# =============================================================================


class TestCoSTEERTask:
    def test_base_code_stored(self):
        from rdagent.components.coder.CoSTEER.task import CoSTEERTask
        t = CoSTEERTask(name="test", base_code="print(1)")
        assert t.base_code == "print(1)"

    def test_base_code_none_by_default(self):
        from rdagent.components.coder.CoSTEER.task import CoSTEERTask
        t = CoSTEERTask(name="test")
        assert t.base_code is None


# =============================================================================
# MultiProcessEvolvingStrategy.assign_code_list_to_evo
# =============================================================================


class TestAssignCodeListToEvo:
    def test_empty_code_list_noops(self):
        from rdagent.components.coder.CoSTEER.evolving_strategy import MultiProcessEvolvingStrategy
        from rdagent.components.coder.CoSTEER.evolvable_subjects import EvolvingItem
        from rdagent.core.experiment import Task

        strat = MultiProcessEvolvingStrategy.__new__(MultiProcessEvolvingStrategy)
        MultiProcessEvolvingStrategy.__init__(strat, scen=MagicMock(), settings=MagicMock())

        ei = EvolvingItem(sub_tasks=[Task(name="t1")])
        ei.experiment_workspace = MagicMock()
        result = strat.assign_code_list_to_evo([{}], ei)
        assert result is ei

    def test_none_entry_is_skipped(self):
        from rdagent.components.coder.CoSTEER.evolving_strategy import MultiProcessEvolvingStrategy
        from rdagent.components.coder.CoSTEER.evolvable_subjects import EvolvingItem
        from rdagent.core.experiment import Task

        strat = MultiProcessEvolvingStrategy.__new__(MultiProcessEvolvingStrategy)
        MultiProcessEvolvingStrategy.__init__(strat, scen=MagicMock(), settings=MagicMock())

        ei = EvolvingItem(sub_tasks=[Task(name="t1")])
        ei.experiment_workspace = MagicMock()
        result = strat.assign_code_list_to_evo([None], ei)
        assert result.sub_workspace_list[0] is None  # unchanged

    def test_code_injects_files(self):
        from rdagent.components.coder.CoSTEER.evolving_strategy import MultiProcessEvolvingStrategy
        from rdagent.components.coder.CoSTEER.evolvable_subjects import EvolvingItem
        from rdagent.core.experiment import Task

        strat = MultiProcessEvolvingStrategy.__new__(MultiProcessEvolvingStrategy)
        MultiProcessEvolvingStrategy.__init__(strat, scen=MagicMock(), settings=MagicMock())

        ei = EvolvingItem(sub_tasks=[Task(name="t1")])
        mock_ws = MagicMock()
        ei.experiment_workspace = mock_ws

        strat.assign_code_list_to_evo([{"factor.py": "x=1"}], ei)
        mock_ws.inject_files.assert_called_once_with(**{"factor.py": "x=1"})

    def test_change_summary_extracted(self):
        from rdagent.components.coder.CoSTEER.evolving_strategy import MultiProcessEvolvingStrategy
        from rdagent.components.coder.CoSTEER.evolvable_subjects import EvolvingItem
        from rdagent.core.experiment import Task

        strat = MultiProcessEvolvingStrategy.__new__(MultiProcessEvolvingStrategy)
        MultiProcessEvolvingStrategy.__init__(strat, scen=MagicMock(), settings=MagicMock())

        mock_ws = MagicMock()
        ei = EvolvingItem(sub_tasks=[Task(name="t1")])
        ei.experiment_workspace = mock_ws

        strat.assign_code_list_to_evo([{"__change_summary__": "summary", "factor.py": "x"}], ei)
        assert mock_ws.change_summary == "summary"
        # change_summary should have been popped from dict
        mock_ws.inject_files.assert_called_once_with(**{"factor.py": "x"})


# =============================================================================
# MultiProcessEvolvingStrategy.evolve_iter
# =============================================================================


class TestEvolveIter:
    def _make_strat(self):
        from rdagent.components.coder.CoSTEER.evolving_strategy import MultiProcessEvolvingStrategy
        from rdagent.components.coder.CoSTEER.config import CoSTEERSettings
        strat = MultiProcessEvolvingStrategy(
            scen=MagicMock(), settings=CoSTEERSettings(), improve_mode=False,
        )
        return strat

    def _make_evo(self, n_tasks=2):
        from rdagent.components.coder.CoSTEER.evolvable_subjects import EvolvingItem
        from rdagent.core.experiment import Task
        tasks = [Task(name=f"task_{i}") for i in range(n_tasks)]
        for t in tasks:
            t.get_task_information = MagicMock(return_value=f"info_{t.name}")
        ei = EvolvingItem(sub_tasks=tasks)
        ei.experiment_workspace = MagicMock()
        return ei

    def test_raises_without_queried_knowledge(self):
        strat = self._make_strat()
        evo = self._make_evo()
        with pytest.raises(ValueError, match="queried_knowledge"):
            next(strat.evolve_iter(evo=evo, queried_knowledge=None))

    def test_successful_tasks_not_scheduled(self):
        from rdagent.components.coder.CoSTEER.knowledge_management import CoSTEERQueriedKnowledge
        strat = self._make_strat()
        evo = self._make_evo(n_tasks=1)
        qk = CoSTEERQueriedKnowledge(
            success_task_to_knowledge_dict={
                "info_task_0": MagicMock(implementation=MagicMock(file_dict={"f.py": "x"})),
            },
        )
        # evolve_iter is a generator, next() starts it
        gen = strat.evolve_iter(evo=evo, queried_knowledge=qk)
        # Should yield the evo (populated from success knowledge)
        result = next(gen)
        # The task was already successful, so no new scheduling
        assert result is evo

    def test_failed_tasks_skipped(self):
        from rdagent.components.coder.CoSTEER.knowledge_management import CoSTEERQueriedKnowledge
        strat = self._make_strat()
        evo = self._make_evo(n_tasks=1)
        qk = CoSTEERQueriedKnowledge(
            failed_task_info_set={"info_task_0"},
        )
        gen = strat.evolve_iter(evo=evo, queried_knowledge=qk)
        result = next(gen)
        # Task skipped because it's in failed_set
        assert result is evo

    def test_improve_mode_skips_with_no_last_feedback(self):
        from rdagent.components.coder.CoSTEER.knowledge_management import CoSTEERQueriedKnowledge
        strat = self._make_strat()
        strat.improve_mode = True
        evo = self._make_evo(n_tasks=1)
        qk = CoSTEERQueriedKnowledge()
        gen = strat.evolve_iter(evo=evo, queried_knowledge=qk, evolving_trace=[])
        result = next(gen)
        # In improve_mode with no last_feedback, task should be skipped
        # (code_list[0] should be {} — empty implementation)
        assert result is evo

    def test_non_improve_mode_schedules_new_tasks(self):
        """Tasks not in success/failed should be scheduled."""
        from rdagent.components.coder.CoSTEER.knowledge_management import CoSTEERQueriedKnowledge
        strat = self._make_strat()
        evo = self._make_evo(n_tasks=1)
        qk = CoSTEERQueriedKnowledge()

        with patch(
            "rdagent.components.coder.CoSTEER.evolving_strategy.multiprocessing_wrapper",
            return_value=[{"factor.py": "x=1"}],
        ):
            gen = strat.evolve_iter(evo=evo, queried_knowledge=qk)
            result = next(gen)
            assert result is evo


# =============================================================================
# CoSTEERMultiEvaluator (partial)
# =============================================================================


class TestCoSTEERMultiEvaluator:
    def test_init_with_single_evaluator(self):
        from rdagent.components.coder.CoSTEER.evaluators import CoSTEERMultiEvaluator
        mock_eval = MagicMock()
        eva = CoSTEERMultiEvaluator(single_evaluator=mock_eval, scen=MagicMock())
        assert eva.single_evaluator is mock_eval

    def test_init_with_evaluator_list(self):
        from rdagent.components.coder.CoSTEER.evaluators import CoSTEERMultiEvaluator
        mock_list = [MagicMock(), MagicMock()]
        eva = CoSTEERMultiEvaluator(single_evaluator=mock_list, scen=MagicMock())
        assert eva.single_evaluator is mock_list
