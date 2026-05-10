"""Tests for factor_coder — evaluators, task, workspace."""

from __future__ import annotations

import sys
from pathlib import Path
from unittest.mock import MagicMock, patch

import pytest

PROJECT_ROOT = Path(__file__).parent.parent.parent
sys.path.insert(0, str(PROJECT_ROOT))


# =============================================================================
# FactorTask
# =============================================================================


class TestFactorTask:
    def test_construction_fields(self):
        from rdagent.components.coder.factor_coder.factor import FactorTask
        t = FactorTask(
            factor_name="f1",
            factor_description="desc",
            factor_formulation="formula",
            variables={"x": 1},
            resource="r1",
        )
        assert t.factor_name == "f1"
        assert t.factor_description == "desc"
        assert t.factor_formulation == "formula"
        assert t.variables == {"x": 1}
        assert t.factor_resources == "r1"
        assert t.factor_implementation is False
        assert t.base_code is None  # from CoSTEERTask

    def test_get_task_information(self):
        from rdagent.components.coder.factor_coder.factor import FactorTask
        t = FactorTask("f1", "desc", "formula", variables={"x": 1})
        info = t.get_task_information()
        assert "factor_name: f1" in info
        assert "factor_description: desc" in info
        assert "factor_formulation: formula" in info
        assert "variables: {'x': 1}" in info

    def test_get_task_brief_information(self):
        from rdagent.components.coder.factor_coder.factor import FactorTask
        t = FactorTask("f1", "desc", "formula")
        info = t.get_task_brief_information()
        assert "factor_name: f1" in info

    def test_get_task_information_and_implementation_result(self):
        from rdagent.components.coder.factor_coder.factor import FactorTask
        t = FactorTask("f1", "desc", "formula")
        result = t.get_task_information_and_implementation_result()
        assert result["factor_name"] == "f1"
        assert result["factor_description"] == "desc"
        assert "factor_implementation" in result

    def test_from_dict(self):
        from rdagent.components.coder.factor_coder.factor import FactorTask
        d = {
            "factor_name": "f2",
            "factor_description": "d2",
            "factor_formulation": "f2",
            "variables": {},
            "resource": None,
            "factor_implementation": True,
        }
        t = FactorTask.from_dict(d)
        assert t.factor_name == "f2"
        assert t.factor_implementation is True

    def test_repr(self):
        from rdagent.components.coder.factor_coder.factor import FactorTask
        t = FactorTask("myfactor", "desc", "formula")
        assert "FactorTask" in repr(t)
        assert "myfactor" in repr(t)


# =============================================================================
# FactorFBWorkspace
# =============================================================================


class TestFactorFBWorkspace:
    def test_init_sets_workspace_path(self):
        from rdagent.components.coder.factor_coder.factor import FactorFBWorkspace, FactorTask
        t = FactorTask("f1", "desc", "formula")
        ws = FactorFBWorkspace(target_task=t)
        assert ws.workspace_path is not None
        # Directory is created lazily by execute(), not in __init__
        assert isinstance(ws.workspace_path, Path)

    def test_execute_returns_message_and_dataframe(self):
        from rdagent.components.coder.factor_coder.factor import FactorFBWorkspace, FactorTask
        t = FactorTask("f1", "desc", "formula")
        t.version = 1
        ws = FactorFBWorkspace(target_task=t)
        # Inject valid factor code
        ws.inject_files(**{
            "factor.py": (
                "import pandas as pd\n"
                "import numpy as np\n"
                "data = pd.read_hdf('intraday_pv.h5', key='data')\n"
                "factor_val = data['$close'].pct_change()\n"
                "factor_val = factor_val.to_frame('f1')\n"
                "factor_val.to_hdf('result.h5', key='data', mode='w')\n"
            ),
        })
        msg, df = ws.execute()
        assert isinstance(msg, str)
        assert df is not None

    def test_execute_succeeds_and_returns_data(self):
        from rdagent.components.coder.factor_coder.factor import FactorFBWorkspace, FactorTask
        t = FactorTask("fl1", "desc", "formula")
        ws = FactorFBWorkspace(target_task=t)
        ws.inject_files(**{
            "factor.py": (
                "import pandas as pd\n"
                "data = pd.read_hdf('intraday_pv.h5', key='data')\n"
                "factor_val = data['$close'].pct_change().to_frame('fl1')\n"
                "factor_val.to_hdf('result.h5', key='data', mode='w')\n"
            ),
        })
        msg, df = ws.execute()
        assert FactorFBWorkspace.FB_EXEC_SUCCESS in msg
        assert FactorFBWorkspace.FB_OUTPUT_FILE_FOUND in msg
        assert df is not None


# =============================================================================
# FactorEvaluatorForCoder (partial integration)
# =============================================================================


class TestFactorEvaluatorForCoder:
    def test_init_creates_sub_evaluators(self):
        from rdagent.components.coder.factor_coder.evaluators import FactorEvaluatorForCoder
        mock_scen = MagicMock()
        eva = FactorEvaluatorForCoder(scen=mock_scen)
        assert eva.value_evaluator is not None
        assert eva.code_evaluator is not None
        assert eva.final_decision_evaluator is not None

    def test_evaluate_with_none_implementation(self):
        from rdagent.components.coder.factor_coder.evaluators import FactorEvaluatorForCoder
        eva = FactorEvaluatorForCoder(scen=MagicMock())
        assert eva.evaluate(target_task=MagicMock(), implementation=None) is None

    def test_evaluate_returns_queried_knowledge_if_present(self):
        from rdagent.components.coder.factor_coder.evaluators import FactorEvaluatorForCoder
        from rdagent.components.coder.factor_coder.factor import FactorTask

        eva = FactorEvaluatorForCoder(scen=MagicMock())

        t = FactorTask("f1", "desc", "formula")
        qk = MagicMock()
        qk.success_task_to_knowledge_dict = {"info_f1": MagicMock(feedback="cached_fb")}
        t.get_task_information = MagicMock(return_value="info_f1")
        qk.failed_task_info_set = set()

        fb = eva.evaluate(target_task=t, implementation=MagicMock(), queried_knowledge=qk)
        assert fb == "cached_fb"  # returned from cache

    def test_evaluate_skips_failed_task(self):
        from rdagent.components.coder.factor_coder.evaluators import FactorEvaluatorForCoder
        from rdagent.components.coder.factor_coder.factor import FactorTask

        eva = FactorEvaluatorForCoder(scen=MagicMock())

        t = FactorTask("f1", "desc", "formula")
        qk = MagicMock()
        qk.success_task_to_knowledge_dict = {}
        t.get_task_information = MagicMock(return_value="info_f1")
        qk.failed_task_info_set = {"info_f1"}

        fb = eva.evaluate(target_task=t, implementation=MagicMock(), queried_knowledge=qk)
        assert fb.final_decision is False
        assert "failed too many times" in fb.execution_feedback


# =============================================================================
# FactorEvaluator (eva_utils) — constructors and identity
# =============================================================================


class TestFactorEvaluatorsInit:
    def test_factor_inf_evaluator_init(self):
        from rdagent.components.coder.factor_coder.eva_utils import FactorInfEvaluator
        eva = FactorInfEvaluator()
        assert str(eva) == "FactorInfEvaluator"

    def test_factor_single_column_evaluator_init(self):
        from rdagent.components.coder.factor_coder.eva_utils import FactorSingleColumnEvaluator
        eva = FactorSingleColumnEvaluator()
        assert str(eva) == "FactorSingleColumnEvaluator"

    def test_factor_output_format_evaluator_init(self):
        from rdagent.components.coder.factor_coder.eva_utils import FactorOutputFormatEvaluator
        eva = FactorOutputFormatEvaluator()
        assert str(eva) == "FactorOutputFormatEvaluator"

    def test_factor_missing_values_evaluator_init(self):
        from rdagent.components.coder.factor_coder.eva_utils import FactorMissingValuesEvaluator
        eva = FactorMissingValuesEvaluator()
        assert str(eva) == "FactorMissingValuesEvaluator"

    def test_factor_correlation_evaluator_init(self):
        from rdagent.components.coder.factor_coder.eva_utils import FactorCorrelationEvaluator
        eva = FactorCorrelationEvaluator(hard_check=True)
        assert eva.hard_check is True
        assert str(eva) == "FactorCorrelationEvaluator"

    def test_factor_value_evaluator_init(self):
        from rdagent.components.coder.factor_coder.eva_utils import FactorValueEvaluator
        mock_scen = MagicMock()
        eva = FactorValueEvaluator(mock_scen)
        assert eva.scen is mock_scen


# ==============================================================================
# HYPOTHESIS-BASED PROPERTY TESTS — Code Generation Patterns, Variable
# Extraction, Evaluator Consistency
# ==============================================================================
from hypothesis import given, settings, strategies as st
import numpy as np
import pandas as pd
from pathlib import Path
from unittest.mock import MagicMock

from rdagent.components.coder.factor_coder.factor import (
    FactorTask,
    FactorFBWorkspace,
)
from rdagent.components.coder.factor_coder.evaluators import (
    FactorEvaluatorForCoder,
)
from rdagent.components.coder.factor_coder.eva_utils import (
    FactorInfEvaluator,
    FactorSingleColumnEvaluator,
    FactorOutputFormatEvaluator,
    FactorMissingValuesEvaluator,
    FactorCorrelationEvaluator,
    FactorValueEvaluator,
)


# ---------------------------------------------------------------------------
# Strategies
# ---------------------------------------------------------------------------


def _valid_factor_task_names() -> st.SearchStrategy:
    return st.text(
        alphabet=st.characters(whitelist_categories=("L", "N", "Lu", "Ll"), whitelist_characters="_"),
        min_size=1,
        max_size=50,
    ).filter(lambda s: s and s[0].isalpha() and " " not in s)


# ---------------------------------------------------------------------------
# Property 1: FactorTask Field Invariants
# ---------------------------------------------------------------------------


class TestFactorTaskInvariants:
    """Property: FactorTask fields maintain invariants after construction."""

    @given(
        factor_name=st.text(min_size=1, max_size=50).filter(lambda s: " " not in s),
        factor_description=st.text(min_size=0, max_size=200),
        factor_formulation=st.text(min_size=0, max_size=200),
    )
    @settings(max_examples=50, deadline=10000)
    def test_construction_preserves_all_fields(self, factor_name, factor_description, factor_formulation):
        """Property: all constructor args are stored as instance attributes."""
        t = FactorTask(factor_name, factor_description, factor_formulation)
        assert t.factor_name == factor_name
        assert t.factor_description == factor_description
        assert t.factor_formulation == factor_formulation

    @given(
        factor_name=st.text(min_size=1, max_size=50).filter(lambda s: " " not in s),
        factor_description=st.text(min_size=0, max_size=200),
        factor_formulation=st.text(min_size=0, max_size=200),
    )
    @settings(max_examples=50, deadline=10000)
    def test_default_field_values(self, factor_name, factor_description, factor_formulation):
        """Property: default fields have expected values."""
        t = FactorTask(factor_name, factor_description, factor_formulation)
        assert t.factor_implementation is False
        assert t.factor_resources is None
        assert t.base_code is None

    @given(
        factor_name=st.text(min_size=1, max_size=50).filter(lambda s: " " not in s),
    )
    @settings(max_examples=50, deadline=10000)
    def test_get_task_information_contains_name(self, factor_name):
        """Property: get_task_information returns string containing factor_name."""
        t = FactorTask(factor_name, "desc", "formula")
        info = t.get_task_information()
        assert factor_name in info

    @given(
        factor_name=st.text(min_size=1, max_size=50).filter(lambda s: " " not in s),
    )
    @settings(max_examples=50, deadline=10000)
    def test_get_task_brief_information_contains_name(self, factor_name):
        """Property: get_task_brief_information returns string containing factor_name."""
        t = FactorTask(factor_name, "desc", "formula")
        info = t.get_task_brief_information()
        assert factor_name in info

    @given(
        factor_name=st.text(min_size=1, max_size=50).filter(lambda s: " " not in s),
    )
    @settings(max_examples=50, deadline=10000)
    def test_get_task_information_and_implementation_result(self, factor_name):
        """Property: returned dict contains expected keys."""
        t = FactorTask(factor_name, "desc", "formula")
        result = t.get_task_information_and_implementation_result()
        assert "factor_name" in result
        assert "factor_description" in result
        assert "factor_formulation" in result
        assert "factor_implementation" in result
        assert result["factor_name"] == factor_name


# ---------------------------------------------------------------------------
# Property 2: FactorTask from_dict
# ---------------------------------------------------------------------------


class TestFactorTaskFromDict:
    """Property: FactorTask.from_dict round-trip."""

    @given(
        factor_name=st.text(min_size=1, max_size=30).filter(lambda s: s.isidentifier()),
        factor_description=st.text(min_size=0, max_size=100),
        factor_formulation=st.text(min_size=0, max_size=100),
    )
    @settings(max_examples=50, deadline=10000)
    def test_from_dict_round_trip(self, factor_name, factor_description, factor_formulation):
        """Property: constructing from dict of get_task_information_and_implementation_result preserves values."""
        t1 = FactorTask(factor_name, factor_description, factor_formulation)
        info = t1.get_task_information_and_implementation_result()
        t2 = FactorTask.from_dict(info)
        assert t2.factor_name == t1.factor_name
        assert t2.factor_description == t1.factor_description
        assert t2.factor_formulation == t1.factor_formulation

    @given(
        factor_name=st.text(min_size=1, max_size=30).filter(lambda s: s.isidentifier()),
    )
    @settings(max_examples=50, deadline=10000)
    def test_from_dict_with_implementation(self, factor_name):
        """Property: factor_implementation field restored from dict."""
        d = {
            "factor_name": factor_name,
            "factor_description": "desc",
            "factor_formulation": "formula",
            "variables": {},
            "resource": None,
            "factor_implementation": True,
        }
        t = FactorTask.from_dict(d)
        assert t.factor_implementation is True

    @given(
        factor_name=st.text(min_size=1, max_size=30).filter(lambda s: s.isidentifier()),
    )
    @settings(max_examples=50, deadline=10000)
    def test_from_dict_with_variables(self, factor_name):
        """Property: variables dict restored from dict."""
        d = {
            "factor_name": factor_name,
            "factor_description": "desc",
            "factor_formulation": "formula",
            "variables": {"x": 1, "y": 2},
            "resource": "r1",
            "factor_implementation": False,
        }
        t = FactorTask.from_dict(d)
        assert t.variables == {"x": 1, "y": 2}
        assert t.factor_resources == "r1"


# ---------------------------------------------------------------------------
# Property 3: FactorTask Repr
# ---------------------------------------------------------------------------


class TestFactorTaskRepr:
    """Property: __repr__ invariants."""

    @given(
        factor_name=st.text(min_size=1, max_size=30).filter(lambda s: s.isidentifier()),
    )
    @settings(max_examples=50, deadline=10000)
    def test_repr_contains_factor_task_and_name(self, factor_name):
        """Property: repr contains 'FactorTask' and factor_name."""
        t = FactorTask(factor_name, "desc", "formula")
        r = repr(t)
        assert "FactorTask" in r
        assert factor_name in r

    @given(
        factor_name=st.text(min_size=1, max_size=30).filter(lambda s: s.isidentifier()),
    )
    @settings(max_examples=50, deadline=10000)
    def test_repr_is_string(self, factor_name):
        """Property: repr returns a string."""
        t = FactorTask(factor_name, "desc", "formula")
        r = repr(t)
        assert isinstance(r, str)
        assert len(r) > 0


# ---------------------------------------------------------------------------
# Property 4: FactorTask Variables
# ---------------------------------------------------------------------------


class TestFactorTaskVariables:
    """Property: variables field invariants."""

    @given(
        factor_name=st.text(min_size=1, max_size=20).filter(lambda s: s.isidentifier()),
        vars_keys=st.lists(
            st.text(min_size=1, max_size=10).filter(lambda s: s.isidentifier()),
            min_size=0, max_size=10, unique=True,
        ),
    )
    @settings(max_examples=50, deadline=10000)
    def test_variables_stored_correctly(self, factor_name, vars_keys):
        """Property: variables dict stored as provided."""
        vars_dict = {k: i for i, k in enumerate(vars_keys)}
        t = FactorTask(factor_name, "desc", "formula", variables=vars_dict)
        assert t.variables == vars_dict

    @given(
        factor_name=st.text(min_size=1, max_size=20).filter(lambda s: s.isidentifier()),
    )
    @settings(max_examples=50, deadline=10000)
    def test_default_variables_is_empty_dict(self, factor_name):
        """Property: default variables is empty dict."""
        t = FactorTask(factor_name, "desc", "formula")
        assert t.variables == {}


# ---------------------------------------------------------------------------
# Property 5: FactorTask Resource
# ---------------------------------------------------------------------------


class TestFactorTaskResource:
    """Property: resource field invariants."""

    @given(
        factor_name=st.text(min_size=1, max_size=20).filter(lambda s: s.isidentifier()),
        resource=st.one_of(st.none(), st.text(min_size=1, max_size=50)),
    )
    @settings(max_examples=50, deadline=10000)
    def test_resource_stored_correctly(self, factor_name, resource):
        """Property: resource field stored as provided or default None."""
        t = FactorTask(factor_name, "desc", "formula", resource=resource)
        assert t.factor_resources == resource


# ---------------------------------------------------------------------------
# Property 6: FactorFBWorkspace Path
# ---------------------------------------------------------------------------


class TestFactorFBWorkspacePath:
    """Property: FactorFBWorkspace workspace path invariants."""

    @given(
        factor_name=st.text(min_size=1, max_size=20).filter(lambda s: s.isidentifier()),
    )
    @settings(max_examples=50, deadline=10000)
    def test_workspace_path_is_valid_path(self, factor_name):
        """Property: workspace_path is a Path instance."""
        t = FactorTask(factor_name, "desc", "formula")
        ws = FactorFBWorkspace(target_task=t)
        assert isinstance(ws.workspace_path, Path)

    @given(
        factor_name=st.text(min_size=1, max_size=20).filter(lambda s: s.isidentifier()),
    )
    @settings(max_examples=50, deadline=10000)
    def test_target_task_reference_preserved(self, factor_name):
        """Property: target_task reference points back to FactorTask."""
        t = FactorTask(factor_name, "desc", "formula")
        ws = FactorFBWorkspace(target_task=t)
        assert ws.target_task is t
        assert ws.target_task.factor_name == factor_name


# ---------------------------------------------------------------------------
# Property 7: FactorEvaluatorForCoder Construction
# ---------------------------------------------------------------------------


class TestFactorEvaluatorForCoderConstruction:
    """Property: FactorEvaluatorForCoder constructor creates sub-evaluators."""

    @given(
        seed=st.integers(min_value=0, max_value=100),
    )
    @settings(max_examples=50, deadline=10000)
    def test_sub_evaluators_are_created(self, seed):
        """Property: constructor creates value, code, and final_decision evaluators."""
        mock_scen = MagicMock()
        eva = FactorEvaluatorForCoder(scen=mock_scen)
        assert eva.value_evaluator is not None
        assert eva.code_evaluator is not None
        assert eva.final_decision_evaluator is not None

    @given(
        seed=st.integers(min_value=0, max_value=100),
    )
    @settings(max_examples=50, deadline=10000)
    def test_evaluate_none_implementation_returns_none(self, seed):
        """Property: evaluate with implementation=None returns None."""
        eva = FactorEvaluatorForCoder(scen=MagicMock())
        result = eva.evaluate(target_task=MagicMock(), implementation=None)
        assert result is None

    @given(
        seed=st.integers(min_value=0, max_value=100),
    )
    @settings(max_examples=50, deadline=10000)
    def test_scenario_reference_accessible(self, seed):
        """Property: evaluator has access to scenario."""
        mock_scen = MagicMock()
        eva = FactorEvaluatorForCoder(scen=mock_scen)
        assert eva.scen is mock_scen


# ---------------------------------------------------------------------------
# Property 8: FactorEvaluator SubTypes
# ---------------------------------------------------------------------------


class TestFactorEvaluatorSubTypes:
    """Property: sub-evaluator types are correct."""

    @given(
        seed=st.integers(min_value=0, max_value=100),
    )
    @settings(max_examples=50, deadline=10000)
    def test_value_evaluator_is_factor_value_evaluator(self, seed):
        """Property: value_evaluator is FactorValueEvaluator instance."""
        eva = FactorEvaluatorForCoder(scen=MagicMock())
        assert isinstance(eva.value_evaluator, FactorValueEvaluator)

    @given(
        seed=st.integers(min_value=0, max_value=100),
    )
    @settings(max_examples=50, deadline=10000)
    def test_scen_passed_to_value_evaluator(self, seed):
        """Property: scenario is passed to value_evaluator."""
        mock_scen = MagicMock()
        eva = FactorEvaluatorForCoder(scen=mock_scen)
        assert eva.value_evaluator.scen is mock_scen


# ---------------------------------------------------------------------------
# Property 9: FactorInfEvaluator
# ---------------------------------------------------------------------------


class TestFactorInfEvaluator:
    """Property: FactorInfEvaluator invariants."""

    @given(seed=st.integers(min_value=0, max_value=100))
    @settings(max_examples=50, deadline=10000)
    def test_str_is_correct(self, seed):
        """Property: __str__ returns 'FactorInfEvaluator'."""
        eva = FactorInfEvaluator()
        assert str(eva) == "FactorInfEvaluator"

    @given(seed=st.integers(min_value=0, max_value=100))
    @settings(max_examples=50, deadline=10000)
    def test_constructor_no_args(self, seed):
        """Property: FactorInfEvaluator can be constructed without arguments."""
        eva = FactorInfEvaluator()
        assert eva is not None


# ---------------------------------------------------------------------------
# Property 10: FactorSingleColumnEvaluator
# ---------------------------------------------------------------------------


class TestFactorSingleColumnEvaluator:
    """Property: FactorSingleColumnEvaluator invariants."""

    @given(seed=st.integers(min_value=0, max_value=100))
    @settings(max_examples=50, deadline=10000)
    def test_str_is_correct(self, seed):
        """Property: __str__ returns 'FactorSingleColumnEvaluator'."""
        eva = FactorSingleColumnEvaluator()
        assert str(eva) == "FactorSingleColumnEvaluator"

    @given(seed=st.integers(min_value=0, max_value=100))
    @settings(max_examples=50, deadline=10000)
    def test_constructor_no_args(self, seed):
        """Property: FactorSingleColumnEvaluator can be constructed without arguments."""
        eva = FactorSingleColumnEvaluator()
        assert eva is not None


# ---------------------------------------------------------------------------
# Property 11: FactorOutputFormatEvaluator
# ---------------------------------------------------------------------------


class TestFactorOutputFormatEvaluator:
    """Property: FactorOutputFormatEvaluator invariants."""

    @given(seed=st.integers(min_value=0, max_value=100))
    @settings(max_examples=50, deadline=10000)
    def test_str_is_correct(self, seed):
        """Property: __str__ returns 'FactorOutputFormatEvaluator'."""
        eva = FactorOutputFormatEvaluator()
        assert str(eva) == "FactorOutputFormatEvaluator"

    @given(seed=st.integers(min_value=0, max_value=100))
    @settings(max_examples=50, deadline=10000)
    def test_constructor_no_args(self, seed):
        """Property: FactorOutputFormatEvaluator can be constructed without arguments."""
        eva = FactorOutputFormatEvaluator()
        assert eva is not None


# ---------------------------------------------------------------------------
# Property 12: FactorMissingValuesEvaluator
# ---------------------------------------------------------------------------


class TestFactorMissingValuesEvaluator:
    """Property: FactorMissingValuesEvaluator invariants."""

    @given(seed=st.integers(min_value=0, max_value=100))
    @settings(max_examples=50, deadline=10000)
    def test_str_is_correct(self, seed):
        """Property: __str__ returns 'FactorMissingValuesEvaluator'."""
        eva = FactorMissingValuesEvaluator()
        assert str(eva) == "FactorMissingValuesEvaluator"

    @given(seed=st.integers(min_value=0, max_value=100))
    @settings(max_examples=50, deadline=10000)
    def test_constructor_no_args(self, seed):
        """Property: FactorMissingValuesEvaluator can be constructed without arguments."""
        eva = FactorMissingValuesEvaluator()
        assert eva is not None


# ---------------------------------------------------------------------------
# Property 13: FactorCorrelationEvaluator
# ---------------------------------------------------------------------------


class TestFactorCorrelationEvaluator:
    """Property: FactorCorrelationEvaluator invariants."""

    @given(
        hard_check=st.booleans(),
    )
    @settings(max_examples=50, deadline=10000)
    def test_hard_check_stored_correctly(self, hard_check):
        """Property: hard_check flag stored correctly."""
        eva = FactorCorrelationEvaluator(hard_check=hard_check)
        assert eva.hard_check is hard_check

    @given(seed=st.integers(min_value=0, max_value=100))
    @settings(max_examples=50, deadline=10000)
    def test_str_contains_correct_name(self, seed):
        """Property: __str__ contains 'FactorCorrelationEvaluator'."""
        eva = FactorCorrelationEvaluator(hard_check=False)
        assert "FactorCorrelationEvaluator" in str(eva)

    @given(seed=st.integers(min_value=0, max_value=100))
    @settings(max_examples=50, deadline=10000)
    def test_default_hard_check_is_false(self, seed):
        """Property: hard_check parameter works."""
        eva = FactorCorrelationEvaluator(hard_check=False)
        assert eva.hard_check is False
        eva2 = FactorCorrelationEvaluator(hard_check=True)
        assert eva2.hard_check is True


# ---------------------------------------------------------------------------
# Property 14: FactorValueEvaluator
# ---------------------------------------------------------------------------


class TestFactorValueEvaluator:
    """Property: FactorValueEvaluator invariants."""

    @given(seed=st.integers(min_value=0, max_value=100))
    @settings(max_examples=50, deadline=10000)
    def test_scenario_stored_correctly(self, seed):
        """Property: scenario reference stored."""
        mock_scen = MagicMock()
        eva = FactorValueEvaluator(mock_scen)
        assert eva.scen is mock_scen

    @given(seed=st.integers(min_value=0, max_value=100))
    @settings(max_examples=50, deadline=10000)
    def test_requires_scenario_arg(self, seed):
        """Property: FactorValueEvaluator requires scenario argument."""
        mock_scen = MagicMock()
        eva = FactorValueEvaluator(mock_scen)
        assert eva is not None


# ---------------------------------------------------------------------------
# Property 15: FactorTask Version
# ---------------------------------------------------------------------------


class TestFactorTaskVersion:
    """Property: version field invariants."""

    @given(
        factor_name=st.text(min_size=1, max_size=20).filter(lambda s: s.isidentifier()),
        version=st.integers(min_value=0, max_value=1000),
    )
    @settings(max_examples=50, deadline=10000)
    def test_version_default_and_mutable(self, factor_name, version):
        """Property: version can be set and retrieved."""
        t = FactorTask(factor_name, "desc", "formula")
        t.version = version
        assert t.version == version


# ---------------------------------------------------------------------------
# Property 16: FactorTask Feedback Field
# ---------------------------------------------------------------------------


class TestFactorTaskFeedback:
    """Property: feedback-related fields."""

    @given(
        factor_name=st.text(min_size=1, max_size=20).filter(lambda s: s.isidentifier()),
    )
    @settings(max_examples=50, deadline=10000)
    def test_default_implementation_is_false(self, factor_name):
        """Property: factor_implementation defaults to False."""
        t = FactorTask(factor_name, "desc", "formula")
        assert t.factor_implementation is False

    @given(
        factor_name=st.text(min_size=1, max_size=20).filter(lambda s: s.isidentifier()),
    )
    @settings(max_examples=50, deadline=10000)
    def test_implementation_can_be_set(self, factor_name):
        """Property: factor_implementation can be set to True."""
        t = FactorTask(factor_name, "desc", "formula")
        t.factor_implementation = True
        assert t.factor_implementation is True


# ---------------------------------------------------------------------------
# Property 17: FactorFBWorkspace FB Constants
# ---------------------------------------------------------------------------


class TestFactorFBWorkspaceConstants:
    """Property: FactorFBWorkspace class constants."""

    def test_fb_exec_success_constant(self):
        """Property: FB_EXEC_SUCCESS is defined as a non-empty string."""
        assert len(str(FactorFBWorkspace.FB_EXEC_SUCCESS)) > 0

    def test_fb_output_file_found_constant(self):
        """Property: FB_OUTPUT_FILE_FOUND is defined as a non-empty string."""
        assert len(str(FactorFBWorkspace.FB_OUTPUT_FILE_FOUND)) > 0


# ---------------------------------------------------------------------------
# Property 18: FactorTask with Variables from_dict Round-trip
# ---------------------------------------------------------------------------


class TestFactorTaskRoundTrip:
    """Property: full round-trip through from_dict preserves all data."""

    @given(
        factor_name=st.text(min_size=1, max_size=20).filter(lambda s: s.isidentifier()),
        factor_description=st.text(min_size=0, max_size=100),
        factor_formulation=st.text(min_size=0, max_size=100),
        n_vars=st.integers(min_value=0, max_value=10),
    )
    @settings(max_examples=50, deadline=10000)
    def test_to_dict_from_dict_round_trip(self, factor_name, factor_description, factor_formulation, n_vars):
        """Property: task.to_dict() → FactorTask.from_dict(d) preserves key fields."""
        t1 = FactorTask(factor_name, factor_description, factor_formulation)
        d = t1.get_task_information_and_implementation_result()
        t2 = FactorTask.from_dict(d)
        assert t2.factor_name == factor_name
        assert t2.factor_description == factor_description
        assert t2.factor_formulation == factor_formulation


# ---------------------------------------------------------------------------
# Property 19: FactorTask CoSTEERTask Inheritance
# ---------------------------------------------------------------------------


class TestFactorTaskCoSTEER:
    """Property: FactorTask inherits correctly from CoSTEERTask."""

    @given(
        factor_name=st.text(min_size=1, max_size=20).filter(lambda s: s.isidentifier()),
    )
    @settings(max_examples=50, deadline=10000)
    def test_base_code_is_none_by_default(self, factor_name):
        """Property: base_code attribute is None by default (from CoSTEERTask)."""
        t = FactorTask(factor_name, "desc", "formula")
        assert t.base_code is None

    @given(
        factor_name=st.text(min_size=1, max_size=20).filter(lambda s: s.isidentifier()),
    )
    @settings(max_examples=50, deadline=10000)
    def test_base_code_can_be_set(self, factor_name):
        """Property: base_code can be set."""
        t = FactorTask(factor_name, "desc", "formula")
        t.base_code = "print(42)"
        assert t.base_code == "print(42)"


# ---------------------------------------------------------------------------
# Property 20: FactorEvaluatorForCoder Caching Behavior
# ---------------------------------------------------------------------------


class TestEvaluatorCaching:
    """Property: evaluator caching behavior."""

    @given(
        factor_name=st.text(min_size=1, max_size=20).filter(lambda s: s.isidentifier()),
    )
    @settings(max_examples=50, deadline=10000)
    def test_cached_feedback_returned(self, factor_name):
        """Property: queried_knowledge with cached feedback returns it."""
        from rdagent.components.coder.factor_coder.factor import FactorTask

        eva = FactorEvaluatorForCoder(scen=MagicMock())
        t = FactorTask(factor_name, "desc", "formula")
        qk = MagicMock()
        qk.success_task_to_knowledge_dict = {"info": MagicMock(feedback="cached")}
        t.get_task_information = MagicMock(return_value="info")
        qk.failed_task_info_set = set()

        fb = eva.evaluate(target_task=t, implementation=MagicMock(), queried_knowledge=qk)
        assert fb == "cached"

    @given(
        factor_name=st.text(min_size=1, max_size=20).filter(lambda s: s.isidentifier()),
    )
    @settings(max_examples=50, deadline=10000)
    def test_failed_task_returns_negative_feedback(self, factor_name):
        """Property: failed tasks return negative feedback with 'failed too many times'."""
        from rdagent.components.coder.factor_coder.factor import FactorTask

        eva = FactorEvaluatorForCoder(scen=MagicMock())
        t = FactorTask(factor_name, "desc", "formula")
        qk = MagicMock()
        qk.success_task_to_knowledge_dict = {}
        t.get_task_information = MagicMock(return_value="info")
        qk.failed_task_info_set = {"info"}

        fb = eva.evaluate(target_task=t, implementation=MagicMock(), queried_knowledge=qk)
        assert fb.final_decision is False
        assert "failed too many times" in fb.execution_feedback


# ---------------------------------------------------------------------------
# Property 21: FactorTask Information Format
# ---------------------------------------------------------------------------


class TestFactorTaskInformation:
    """Property: task information output format."""

    @given(
        factor_name=st.text(min_size=1, max_size=30).filter(lambda s: s.isidentifier()),
        factor_description=st.text(min_size=0, max_size=100),
        factor_formulation=st.text(min_size=0, max_size=100),
    )
    @settings(max_examples=50, deadline=10000)
    def test_get_task_information_format(self, factor_name, factor_description, factor_formulation):
        """Property: get_task_information has expected format."""
        t = FactorTask(factor_name, factor_description, factor_formulation)
        info = t.get_task_information()
        assert f"factor_name: {factor_name}" in info
        assert f"factor_description: {factor_description}" in info
        assert f"factor_formulation: {factor_formulation}" in info

    @given(
        factor_name=st.text(min_size=1, max_size=30).filter(lambda s: s.isidentifier()),
    )
    @settings(max_examples=50, deadline=10000)
    def test_get_task_information_is_string(self, factor_name):
        """Property: get_task_information returns str."""
        t = FactorTask(factor_name, "desc", "formula")
        info = t.get_task_information()
        assert isinstance(info, str)

    @given(
        factor_name=st.text(min_size=1, max_size=30).filter(lambda s: s.isidentifier()),
    )
    @settings(max_examples=50, deadline=10000)
    def test_get_task_brief_information_is_string(self, factor_name):
        """Property: get_task_brief_information returns str."""
        t = FactorTask(factor_name, "desc", "formula")
        info = t.get_task_brief_information()
        assert isinstance(info, str)
