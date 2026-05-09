"""Tests for scripts/nexquant_full_eval.py pure functions and dataclasses."""

from __future__ import annotations

import sys
from pathlib import Path

import pytest

PROJECT_ROOT = Path(__file__).parent.parent.parent
sys.path.insert(0, str(PROJECT_ROOT))


class TestFactorInfo:
    def test_construction(self):
        from scripts.nexquant_full_eval import FactorInfo
        fi = FactorInfo(
            workspace_hash="abc123",
            factor_name="test_factor",
            factor_code="x=1",
        )
        assert fi.workspace_hash == "abc123"
        assert fi.factor_name == "test_factor"
        assert fi.factor_code == "x=1"


class TestEvalResult:
    def test_defaults(self):
        from scripts.nexquant_full_eval import EvalResult
        er = EvalResult(factor_name="f1", workspace_hash="h1")
        assert er.status == ""
        assert er.ic is None
        assert er.error_message is None
        assert er.non_null_count == 0

    def test_failed_result(self):
        from scripts.nexquant_full_eval import EvalResult
        er = EvalResult(
            factor_name="f1", workspace_hash="h1",
            status="failed", error_message="timeout",
        )
        assert er.status == "failed"
        assert er.error_message == "timeout"

    def test_to_dict(self):
        from scripts.nexquant_full_eval import EvalResult
        er = EvalResult(factor_name="f1", workspace_hash="h1", status="success", ic=0.05)
        d = er.to_dict()
        assert d["factor_name"] == "f1"
        assert d["ic"] == 0.05
        assert d["status"] == "success"


class TestExtractFactorDescription:
    def test_docstring_extracted(self):
        from scripts.nexquant_full_eval import _extract_factor_description
        code = '"""This is a test factor.\nComputes momentum."""\nx=1'
        desc = _extract_factor_description(code)
        assert "test factor" in desc

    def test_comment_extraction(self):
        from scripts.nexquant_full_eval import _extract_factor_description
        code = "# Momentum factor\n# Uses 20-bar window\nx=1"
        desc = _extract_factor_description(code)
        assert "Momentum factor" in desc
        assert "20-bar window" in desc

    def test_no_docstring_or_comments(self):
        from scripts.nexquant_full_eval import _extract_factor_description
        code = "x = 1\ny = 2\n"
        desc = _extract_factor_description(code)
        assert desc == "No description available"

    def test_shebang_skipped(self):
        from scripts.nexquant_full_eval import _extract_factor_description
        code = "#!/usr/bin/env python\n# Real comment\nx=1"
        desc = _extract_factor_description(code)
        assert "Real comment" in desc
        assert "usr/bin" not in desc
