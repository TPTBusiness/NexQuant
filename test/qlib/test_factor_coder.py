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
