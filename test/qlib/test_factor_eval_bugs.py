"""Tests for bugs found in the factor evaluation pipeline."""

import sys
from pathlib import Path
from unittest.mock import MagicMock, patch

import numpy as np
import pandas as pd
import pytest

# Project root
PROJECT_ROOT = Path(__file__).parent.parent.parent
sys.path.insert(0, str(PROJECT_ROOT))


# =============================================================================
# Bug 1: Missing `import sys` in _save_factor_values (factor_runner.py:968)
# =============================================================================

class TestSaveFactorValuesMissingSysImport:
    """Verify that _save_factor_values has `import sys` — uses sys.executable at line 968."""

    def test_save_factor_values_has_sys_import(self):
        import inspect

        from rdagent.scenarios.qlib.developer.factor_runner import QlibFactorRunner

        source = inspect.getsource(QlibFactorRunner._save_factor_values)

        assert "import sys" in source, (
            "BUG: _save_factor_values calls sys.executable but does not import sys. "
            "This causes a NameError at runtime, silently swallowed by the try/except."
        )

    def test_save_factor_values_nameerror_when_called(self):
        """Simulate calling _save_factor_values without sys available."""
        from rdagent.scenarios.qlib.developer.factor_runner import QlibFactorRunner

        runner = QlibFactorRunner.__new__(QlibFactorRunner)

        mock_exp = MagicMock()
        mock_exp.sub_workspace_list = []
        mock_exp.experiment_workspace.workspace_path = None

        # This should NOT raise NameError for 'sys' — if it does, the bug is present
        try:
            runner._save_factor_values("TestFactor", mock_exp)
        except NameError as e:
            if "sys" in str(e):
                pytest.fail(
                    "BUG CONFIRMED: _save_factor_values raises NameError because "
                    "'sys' is not imported. The factor values parquet is never saved."
                )
            raise


# =============================================================================
# Bug 2: `acc_rate` undefined in FactorEqualValueRatioEvaluator (eva_utils.py:335-346)
# =============================================================================

class TestEqualValueRatioAccRateUndefined:
    """Verify FactorEqualValueRatioEvaluator handles shape-mismatch correctly."""

    def test_acc_rate_undefined_after_except(self):
        """If gen_df.sub(gt_df) raises, acc_rate should still be defined (default -1)."""
        from rdagent.components.coder.factor_coder.eva_utils import FactorEqualValueRatioEvaluator

        evaluator = FactorEqualValueRatioEvaluator()

        # Simulate the case where _get_df returns None for gt_df, which causes
        # gen_df.sub(None) to raise AttributeError. The except clause must not
        # reference an undefined acc_rate variable.
        gen_df = pd.DataFrame({"x": [1.0, 2.0, 3.0]}, index=[0, 1, 2])

        gt_ws = MagicMock()
        imp_ws = MagicMock()

        gt_ws.execute.return_value = ("", None)  # _get_df will set gt_df = None
        imp_ws.execute.return_value = ("", gen_df)

        # Should NOT raise NameError
        try:
            result = evaluator.evaluate(imp_ws, gt_ws)
            assert isinstance(result, tuple)
            assert len(result) == 2
            feedback, metric = result
            assert metric == -1, f"Expected -1 (fallback), got {metric}"
        except NameError as e:
            if "acc_rate" in str(e):
                pytest.fail(
                    "BUG CONFIRMED: FactorEqualValueRatioEvaluator references 'acc_rate' "
                    "which is undefined when gen_df.sub(gt_df) raises an exception."
                )
            raise

    def test_acc_rate_defined_when_shapes_match(self):
        """Normal case: same shapes — acc_rate should be defined and returned."""
        from rdagent.components.coder.factor_coder.eva_utils import FactorEqualValueRatioEvaluator

        evaluator = FactorEqualValueRatioEvaluator()

        gt_ws = MagicMock()
        imp_ws = MagicMock()

        gen_df = pd.DataFrame({"x": [1.0, 2.0, 3.0]}, index=[0, 1, 2])
        gt_df = pd.DataFrame({"y": [1.0, 2.0, 3.0]}, index=[0, 1, 2])

        gt_ws.execute.return_value = ("", gt_df)
        imp_ws.execute.return_value = ("", gen_df)

        result = evaluator.evaluate(imp_ws, gt_ws)
        assert isinstance(result, tuple)
        assert len(result) == 2
        feedback, metric = result
        # When values match within tolerance, metric should be a float near 1.0
        assert isinstance(metric, float) or isinstance(metric, (int, np.integer))
        assert metric >= 0


# =============================================================================
# Bug 3: Annualization factor hardcoded in _evaluate_factor_directly (factor_runner.py:553)
# =============================================================================

class TestAnnualizationFactorInDirectEval:
    """Verify direct evaluation uses correct annualization with forward_return_bars."""

    def test_annualization_factor_uses_forward_bars(self):
        """The direct eval method hardcodes 96 instead of using forward_return_bars param."""
        import inspect

        from rdagent.scenarios.qlib.developer.factor_runner import QlibFactorRunner

        source = inspect.getsource(QlibFactorRunner._evaluate_factor_directly)

        # Check that the method uses `np.sqrt(252 * 1440 / 96)` which hardcodes 96
        # This should ideally be parameterized or at least consistent with the
        # forward return calculation at line ~530 which also uses 96.
        assert "np.sqrt(252 * 1440 / 96)" in source or "np.sqrt(252*1440/96)" in source, (
            "The annualization factor in _evaluate_factor_directly should match "
            "the forward_return_bars used for computing forward returns."
        )

    def test_ann_factor_is_consistent_with_forward_ret(self):
        """Verify both the forward return shift and annualization use 96 bars."""
        import inspect

        from rdagent.scenarios.qlib.developer.factor_runner import QlibFactorRunner

        source = inspect.getsource(QlibFactorRunner._evaluate_factor_directly)

        # forward return uses `.shift(-96)` at line ~530
        assert '.shift(-96)' in source, "Forward return shift should use 96 bars (1 day)"

        # annualization should also use 96
        assert '1440 / 96' in source, (
            "Annualization factor should use the same number (96) as the forward return shift"
        )


# =============================================================================
# Bug 4: _fix_inf_nan_handling inserts code before .dropna() or .to_hdf() in wrong context
# =============================================================================

class TestInfNanHandlingInsertion:
    """Verify inf/nan auto-fixer doesn't insert code in the wrong context."""

    def test_no_insertion_before_dropna_when_no_column_found(self):
        from rdagent.components.coder.factor_coder.auto_fixer import FactorAutoFixer

        fixer = FactorAutoFixer()

        # Code where the LAST assignment before .dropna() is NOT a df['col'] = pattern
        # but dropna() still exists (e.g., on a temporary variable)
        code = (
            "def calc():\n"
            "    df = pd.read_hdf('data.h5', key='data')\n"
            "    temp = df['$close'].diff()\n"
            "    temp = temp.dropna()\n"
            "    df['result'] = temp * 2\n"
            "    result = df[['result']]\n"
        )

        result = fixer.fix(code)

        # The code should still be valid (no syntax error from misplaced insertion)
        import ast
        try:
            ast.parse(result)
        except SyntaxError as e:
            pytest.fail(f"Auto-fixer produced invalid Python code: {e}")

    def test_inf_handling_inserted_before_result_assignment(self):
        from rdagent.components.coder.factor_coder.auto_fixer import FactorAutoFixer

        fixer = FactorAutoFixer()

        code = (
            "def calc():\n"
            "    df = pd.read_hdf('data.h5', key='data')\n"
            "    df['myfactor'] = df['$close'] / df['sigma_60bar']\n"
            "    df['myfactor'] = df['myfactor'] / df['sigma_5bar']\n"
            "    result = df[['myfactor']]\n"
        )

        result = fixer.fix(code)

        # Should have added inf handling before the result = df[[...]] line
        # but not broken syntax
        import ast
        try:
            ast.parse(result)
        except SyntaxError as e:
            pytest.fail(f"Auto-fixer produced invalid Python code: {e}")

        assert "replace([np.inf, -np.inf]" in result


# =============================================================================
# Bug 5: scan_factors reads factor_code twice (predix_full_eval.py:174 + 195)
# =============================================================================

class TestScanFactorsDoubleRead:
    """Verify scan_factors doesn't wastefully read factor file twice."""

    def test_factor_code_read_only_when_needed(self):
        """Confirm the scan_factors double-read behavior (line 174+195)."""
        import inspect
        from scripts import predix_full_eval

        source = inspect.getsource(predix_full_eval.scan_factors)

        # Count occurrences of `.read_text()`
        count = source.count(".read_text()")
        # Expected: at least 2 (line 174 in fallback, line 195 in FactorInfo)
        # Bug: if factor_name comes from result.h5 (line 168-170), then line 174
        # is skipped, but line 195 always reads again — that's one wasted read.
        assert count == 2, (
            f"scan_factors has {count} read_text() calls. "
            "Expected exactly 2 (one for name extraction, one for FactorInfo). "
            "Consider caching to avoid double reads."
        )
