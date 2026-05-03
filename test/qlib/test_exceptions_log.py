"""Tests for core/exception and log infrastructure."""

from __future__ import annotations

import sys
from pathlib import Path

import pytest

PROJECT_ROOT = Path(__file__).parent.parent.parent
sys.path.insert(0, str(PROJECT_ROOT))


# =============================================================================
# Exception hierarchy
# =============================================================================


class TestExceptionHierarchy:
    def test_workflow_error_is_exception(self):
        from rdagent.core.exception import WorkflowError
        with pytest.raises(WorkflowError):
            raise WorkflowError("test")

    def test_format_error_is_workflow_error(self):
        from rdagent.core.exception import FormatError, WorkflowError
        assert issubclass(FormatError, WorkflowError)

    def test_coder_error_is_workflow_error(self):
        from rdagent.core.exception import CoderError, WorkflowError
        assert issubclass(CoderError, WorkflowError)

    def test_code_format_error_is_coder_error(self):
        from rdagent.core.exception import CodeFormatError, CoderError
        assert issubclass(CodeFormatError, CoderError)

    def test_custom_runtime_error_is_coder_error(self):
        from rdagent.core.exception import CustomRuntimeError, CoderError
        assert issubclass(CustomRuntimeError, CoderError)

    def test_no_output_error_is_coder_error(self):
        from rdagent.core.exception import NoOutputError, CoderError
        assert issubclass(NoOutputError, CoderError)

    def test_runner_error_is_exception(self):
        from rdagent.core.exception import RunnerError
        with pytest.raises(RunnerError):
            raise RunnerError("test")

    def test_factor_empty_error_is_coder_error(self):
        from rdagent.core.exception import FactorEmptyError, CoderError
        assert FactorEmptyError is CoderError

    def test_model_empty_error_is_coder_error(self):
        from rdagent.core.exception import ModelEmptyError, CoderError
        assert ModelEmptyError is CoderError

    def test_llm_unavailable_error_is_runtime_error(self):
        from rdagent.core.exception import LLMUnavailableError
        with pytest.raises(LLMUnavailableError):
            raise LLMUnavailableError("LLM down")

    def test_code_block_parse_error(self):
        from rdagent.core.exception import CodeBlockParseError
        e = CodeBlockParseError("msg", "content", "python")
        assert e.message == "msg"
        assert e.content == "content"
        assert e.language == "python"
        assert isinstance(e, Exception)

    def test_coder_error_caused_by_timeout_default(self):
        from rdagent.core.exception import CoderError
        assert CoderError.caused_by_timeout is False


# =============================================================================
# RDAgentLog singleton
# =============================================================================


class TestRDAgentLog:
    def test_is_singleton(self):
        from rdagent.log.logger import RDAgentLog
        a = RDAgentLog()
        b = RDAgentLog()
        assert a is b

    def test_has_tag_context(self):
        from rdagent.log.logger import RDAgentLog
        assert hasattr(RDAgentLog, "_tag_ctx")


# =============================================================================
# Daily log session
# =============================================================================


class TestDailyLog:
    def test_session_importable(self):
        from rdagent.log.daily_log import session
        assert callable(session)
