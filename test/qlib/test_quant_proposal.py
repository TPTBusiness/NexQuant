"""Tests for quant_proposal — QuantTrace, QlibQuantHypothesis."""

from __future__ import annotations

import sys
from pathlib import Path
from unittest.mock import MagicMock

import pytest

PROJECT_ROOT = Path(__file__).parent.parent.parent
sys.path.insert(0, str(PROJECT_ROOT))


class TestQuantTrace:
    def test_init_factor_count_zero(self):
        from rdagent.scenarios.qlib.proposal.quant_proposal import QuantTrace
        trace = QuantTrace(scen=MagicMock())
        assert trace.get_factor_count() == 0
        assert trace.controller is not None

    def test_increment_factor_count(self):
        from rdagent.scenarios.qlib.proposal.quant_proposal import QuantTrace
        trace = QuantTrace(scen=MagicMock())
        trace.increment_factor_count()
        assert trace.get_factor_count() == 1
        trace.increment_factor_count()
        assert trace.get_factor_count() == 2


class TestQlibQuantHypothesis:
    def test_construction_fields(self):
        from rdagent.scenarios.qlib.proposal.quant_proposal import QlibQuantHypothesis
        h = QlibQuantHypothesis(
            hypothesis="test hypothesis",
            reason="test reason",
            concise_reason="cr",
            concise_observation="co",
            concise_justification="cj",
            concise_knowledge="ck",
            action="factor",
        )
        assert h.hypothesis == "test hypothesis"
        assert h.reason == "test reason"
        assert h.action == "factor"
        assert h.concise_reason == "cr"

    def test_str_contains_action(self):
        from rdagent.scenarios.qlib.proposal.quant_proposal import QlibQuantHypothesis
        h = QlibQuantHypothesis(
            hypothesis="h", reason="r", concise_reason="cr",
            concise_observation="co", concise_justification="cj",
            concise_knowledge="ck", action="model",
        )
        s = str(h)
        assert "Chosen Action: model" in s
        assert "Hypothesis: h" in s
        assert "Reason: r" in s
