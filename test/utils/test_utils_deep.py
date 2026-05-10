"""Deep tests for rdagent.utils: fmt.py shrink_text and other utility modules."""

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

UTIL_MODULES = [
    "rdagent.utils",
    "rdagent.utils.fmt",
    "rdagent.utils.qlib",
    "rdagent.utils.env",
    "rdagent.utils.workflow",
    "rdagent.utils.agent.tpl",
]


class TestUtilsImports:
    @pytest.mark.parametrize("module_name", UTIL_MODULES)
    def test_module_importable(self, module_name: str) -> None:
        import importlib
        mod = importlib.import_module(module_name)
        assert mod is not None


# =============================================================================
# shrink_text
# =============================================================================


class TestShrinkText:
    def test_short_text_unchanged(self) -> None:
        from rdagent.utils.fmt import shrink_text
        result = shrink_text("hello world", context_lines=10, line_len=100)
        assert result == "hello world"

    def test_single_line_shorter_than_limit(self) -> None:
        from rdagent.utils.fmt import shrink_text
        result = shrink_text("abc", context_lines=2, line_len=5)
        assert result == "abc"

    def test_multi_line_under_threshold_unchanged(self) -> None:
        from rdagent.utils.fmt import shrink_text
        text = "line1\nline2\nline3"
        result = shrink_text(text, context_lines=5, line_len=50)
        assert result == text

    def test_exactly_at_threshold(self) -> None:
        from rdagent.utils.fmt import shrink_text
        text = "\n".join([f"line{i}" for i in range(4)])
        result = shrink_text(text, context_lines=4, line_len=50)
        assert result == text

    def test_more_lines_than_context_shrinks(self) -> None:
        from rdagent.utils.fmt import shrink_text
        text = "\n".join([f"line{i}" for i in range(100)])
        result = shrink_text(text, context_lines=10, line_len=100)
        assert "lines are hidden" in result

    def test_row_shrink_false_preserves_all_lines(self) -> None:
        from rdagent.utils.fmt import shrink_text
        text = "\n".join([f"line{i}" for i in range(100)])
        result = shrink_text(text, context_lines=5, line_len=100, row_shrink=False)
        assert result == text

    def test_col_shrink_long_lines(self) -> None:
        from rdagent.utils.fmt import shrink_text
        long_line = "x" * 100
        result = shrink_text(long_line, context_lines=5, line_len=20)
        assert "chars are hidden" in result
        assert len(result) < 100

    def test_col_shrink_false_preserves_long_lines(self) -> None:
        from rdagent.utils.fmt import shrink_text
        long_line = "x" * 100
        result = shrink_text(long_line, context_lines=5, line_len=20, col_shrink=False)
        assert result == long_line

    def test_both_shrink_disabled(self) -> None:
        from rdagent.utils.fmt import shrink_text
        text = "x" * 1000 + "\n" + "y" * 1000
        result = shrink_text(text, context_lines=1, line_len=5, row_shrink=False, col_shrink=False)
        assert result == text

    def test_first_and_last_lines_preserved(self) -> None:
        from rdagent.utils.fmt import shrink_text
        text = "\n".join([f"unique_line_{i}" for i in range(100)])
        result = shrink_text(text, context_lines=6, line_len=100)
        assert "unique_line_0" in result
        assert "unique_line_99" in result
        assert "unique_line_50" not in result

    def test_hidden_lines_count_correct(self) -> None:
        from rdagent.utils.fmt import shrink_text
        total = 100
        ctx = 10
        text = "\n".join([f"L{i}" for i in range(total)])
        result = shrink_text(text, context_lines=ctx, line_len=100)
        half = ctx // 2
        hidden = total - half * 2
        assert f"({hidden} lines are hidden)" in result

    def test_empty_string(self) -> None:
        from rdagent.utils.fmt import shrink_text
        result = shrink_text("", context_lines=5, line_len=10)
        assert result == ""

    def test_single_line_with_newline_at_end(self) -> None:
        from rdagent.utils.fmt import shrink_text
        result = shrink_text("hello\n", context_lines=10, line_len=50)
        assert "hello" in result

    def test_all_empty_lines(self) -> None:
        from rdagent.utils.fmt import shrink_text
        text = "\n".join(["" for _ in range(100)])
        result = shrink_text(text, context_lines=10, line_len=50)
        assert isinstance(result, str)

    def test_very_large_context_lines(self) -> None:
        from rdagent.utils.fmt import shrink_text
        text = "\n".join(["a" for _ in range(50)])
        result = shrink_text(text, context_lines=1000, line_len=10)
        assert result == text

    def test_context_lines_of_one(self) -> None:
        from rdagent.utils.fmt import shrink_text
        text = "line1\nline2\nline3\nline4\nline5"
        result = shrink_text(text, context_lines=1, line_len=100)
        assert "lines are hidden" in result

    def test_line_len_of_one(self) -> None:
        from rdagent.utils.fmt import shrink_text
        result = shrink_text("abcdefgh", context_lines=10, line_len=1)
        assert "chars are hidden" in result

    def test_line_len_zero(self) -> None:
        from rdagent.utils.fmt import shrink_text
        result = shrink_text("hello", context_lines=10, line_len=0)
        assert "chars are hidden" in result

    def test_returns_string_always(self) -> None:
        from rdagent.utils.fmt import shrink_text
        for text in ["", "a", "a\nb\nc", "x" * 1000]:
            result = shrink_text(text)
            assert isinstance(result, str)

    def test_hidden_prefix_format(self) -> None:
        from rdagent.utils.fmt import shrink_text
        text = "\n".join(["L" for _ in range(100)])
        result = shrink_text(text, context_lines=10, line_len=100)
        assert "lines are hidden" in result
        assert "..." in result

    @pytest.mark.parametrize("total_lines,ctx", [
        (10, 5), (10, 6), (10, 10), (50, 4), (50, 20), (100, 2),
    ])
    def test_various_combinations(self, total_lines: int, ctx: int) -> None:
        from rdagent.utils.fmt import shrink_text
        text = "\n".join([f"L{i}" for i in range(total_lines)])
        result = shrink_text(text, context_lines=ctx, line_len=100)
        assert isinstance(result, str)
        assert len(result) > 0

    @pytest.mark.parametrize("line_len,chars_per_line", [
        (5, 3), (10, 9), (20, 19), (50, 51),
    ])
    def test_line_len_vs_chars(self, line_len: int, chars_per_line: int) -> None:
        from rdagent.utils.fmt import shrink_text
        text = "x" * chars_per_line
        result = shrink_text(text, context_lines=5, line_len=line_len)
        if chars_per_line > line_len:
            assert "chars are hidden" in result
        else:
            assert result == text


# =============================================================================
# shrink_text — properties
# =============================================================================


class TestShrinkTextProperties:
    def test_output_contains_original_when_small(self) -> None:
        from rdagent.utils.fmt import shrink_text
        lines = ["a", "b", "c", "d", "e"]
        text = "\n".join(lines)
        result = shrink_text(text, context_lines=len(lines) + 1, line_len=10000)
        assert result == text

    def test_shrinking_to_less_lines(self) -> None:
        from rdagent.utils.fmt import shrink_text
        original = "\n".join([f"line_{i}" for i in range(1000)])
        result = shrink_text(original, context_lines=10, line_len=100)
        result_lines = result.split("\n")
        assert len(result_lines) < 1000

    @pytest.mark.parametrize("n_lines", [1, 2, 3, 5, 10])
    def test_various_line_counts(self, n_lines: int) -> None:
        from rdagent.utils.fmt import shrink_text
        text = "\n".join([f"L{i}" for i in range(n_lines)])
        result = shrink_text(text, context_lines=50, line_len=200)
        assert isinstance(result, str)
        assert result == text  # all fit within context_lines=50


class TestShrinkTextCombinatorial:
    @pytest.mark.parametrize("ctx", [0, 1, 2, 5, 10, 50, 100])
    @pytest.mark.parametrize("llen", [0, 1, 5, 10, 50, 200])
    def test_parameter_grid(self, ctx: int, llen: int) -> None:
        from rdagent.utils.fmt import shrink_text
        text = "x" * 60 + "\n" + "y" * 60
        result = shrink_text(text, context_lines=ctx, line_len=llen)
        assert isinstance(result, str)

    @pytest.mark.parametrize("row_shrink", [True, False])
    @pytest.mark.parametrize("col_shrink", [True, False])
    def test_all_shrink_flag_combinations(self, row_shrink: bool, col_shrink: bool) -> None:
        from rdagent.utils.fmt import shrink_text
        text = "\n".join(["line"] * 200)
        result = shrink_text(text, context_lines=5, line_len=50,
            row_shrink=row_shrink, col_shrink=col_shrink)
        assert isinstance(result, str)


# =============================================================================
# T (template) system
# =============================================================================


class TestTemplateSystem:
    def test_t_class_is_importable(self) -> None:
        from rdagent.utils.agent.tpl import T
        assert T is not None

    def test_t_loads_prompt_template(self) -> None:
        from rdagent.utils.agent.tpl import T
        tpl = T("scenarios.qlib.prompts:hypothesis_and_feedback")
        assert tpl is not None

    def test_t_with_invalid_template_raises(self) -> None:
        from rdagent.utils.agent.tpl import T
        with pytest.raises(FileNotFoundError):
            T("nonexistent.module.path:nonexistent_key")

    @patch("rdagent.utils.agent.tpl.logger")
    def test_t_r_method_renders_template(self, mock_logger: MagicMock) -> None:
        from rdagent.utils.agent.tpl import T
        tpl = T("scenarios.qlib.prompts:hypothesis_and_feedback")
        mock_trace = MagicMock()
        mock_trace.hist = []
        result = tpl.r(trace=mock_trace)
        assert isinstance(result, str)
        assert len(result) > 0


# =============================================================================
# Qlib utilities
# =============================================================================


class TestQlibUtils:
    def test_validate_qlib_features_importable(self) -> None:
        from rdagent.utils.qlib import validate_qlib_features
        assert callable(validate_qlib_features)

    def test_validate_valid_features(self) -> None:
        from rdagent.utils.qlib import validate_qlib_features
        assert validate_qlib_features(["$close", "$high / $low", "$volume"]) is True

    def test_validate_empty_list(self) -> None:
        from rdagent.utils.qlib import validate_qlib_features
        result = validate_qlib_features([])
        assert isinstance(result, bool)

    def test_validate_any_expression(self) -> None:
        from rdagent.utils.qlib import validate_qlib_features
        result = validate_qlib_features(["not_a_real_field_xyz"])
        assert isinstance(result, bool)

    def test_alpha20_importable(self) -> None:
        from rdagent.utils.qlib import ALPHA20
        assert isinstance(ALPHA20, dict)
        assert len(ALPHA20) > 0

    @pytest.mark.parametrize("feature_exp", [
        "$close", "$open", "$high", "$low", "$volume", "$vwap",
        "$close / $open", "($high - $low) / $open",
    ])
    def test_individual_feature_validation(self, feature_exp: str) -> None:
        from rdagent.utils.qlib import validate_qlib_features
        result = validate_qlib_features([feature_exp])
        assert isinstance(result, bool)


# =============================================================================
# Env utilities
# =============================================================================


class TestEnvUtils:
    def test_env_module_is_importable(self) -> None:
        from rdagent.utils import env
        assert env is not None


# =============================================================================
# md5_hash
# =============================================================================


class TestMd5Hash:
    def test_md5_hash_is_function(self) -> None:
        from rdagent.utils import md5_hash
        assert callable(md5_hash)

    def test_md5_hash_returns_string(self) -> None:
        from rdagent.utils import md5_hash
        result = md5_hash("test input")
        assert isinstance(result, str)
        assert len(result) == 64

    def test_md5_hash_deterministic(self) -> None:
        from rdagent.utils import md5_hash
        a = md5_hash("hello")
        b = md5_hash("hello")
        assert a == b

    def test_md5_hash_different_inputs(self) -> None:
        from rdagent.utils import md5_hash
        a = md5_hash("hello")
        b = md5_hash("world")
        assert a != b

    @pytest.mark.parametrize("input_val", [
        "", "a", "abc", "multi\nline\nstring",
    ])
    def test_md5_hash_various_inputs(self, input_val: str) -> None:
        from rdagent.utils import md5_hash
        result = md5_hash(input_val)
        assert isinstance(result, str)
        assert len(result) == 64

    def test_md5_hash_hex_format(self) -> None:
        from rdagent.utils import md5_hash
        import re
        result = md5_hash("test")
        assert re.match(r'^[0-9a-f]{64}$', result) is not None


# =============================================================================
# Workflow utils
# =============================================================================


class TestWorkflowUtils:
    def test_loop_base_is_importable(self) -> None:
        from rdagent.utils.workflow import LoopBase
        assert LoopBase is not None

    def test_loop_meta_is_type(self) -> None:
        from rdagent.utils.workflow import LoopMeta
        assert isinstance(LoopMeta, type)


# =============================================================================
# Large input stress tests
# =============================================================================


class TestLargeInputs:
    def test_ten_thousand_lines(self) -> None:
        from rdagent.utils.fmt import shrink_text
        text = "\n".join([f"L{i}" for i in range(10000)])
        result = shrink_text(text, context_lines=50, line_len=100)
        assert isinstance(result, str)
        assert "lines are hidden" in result

    def test_very_long_single_line(self) -> None:
        from rdagent.utils.fmt import shrink_text
        text = "a" * 100000
        result = shrink_text(text, context_lines=5, line_len=100)
        assert "chars are hidden" in result


# =============================================================================
# Pickle safety
# =============================================================================


class TestFmtPickleSafety:
    def test_shrunk_text_pickle_safety(self) -> None:
        from rdagent.utils.fmt import shrink_text
        result = shrink_text("x" * 500, context_lines=5, line_len=10)
        data = pickle.dumps(result)
        loaded = pickle.loads(data)
        assert loaded == result

    def test_alpha20_pickle_safety(self) -> None:
        from rdagent.utils.qlib import ALPHA20
        data = pickle.dumps(ALPHA20)
        loaded = pickle.loads(data)
        assert loaded == ALPHA20
