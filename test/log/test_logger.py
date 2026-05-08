"""Tests for rdagent/log/logger.py — RDAgentLog wrapper around loguru."""

from __future__ import annotations

import pytest
from rdagent.log.logger import RDAgentLog


class TestRDAgentLog:
    def test_singleton(self):
        a = RDAgentLog()
        b = RDAgentLog()
        assert a is b

    def test_has_debug_method(self):
        logger = RDAgentLog()
        assert hasattr(logger, "debug")
        assert callable(logger.debug)

    def test_debug_accepts_args(self):
        logger = RDAgentLog()
        logger.debug("test message")
        logger.debug("test message", tag="mytag")
        logger.debug("test message", raw=True)
        logger.debug("test message", tag="x", raw=False)

    def test_info_warning_error_exist(self):
        logger = RDAgentLog()
        for method in ("info", "warning", "error", "debug"):
            assert hasattr(logger, method), f"missing {method}"
            assert callable(getattr(logger, method)), f"{method} not callable"

    def test_log_object(self):
        logger = RDAgentLog()
        logger.log_object({"key": "value"})
        logger.log_object(["a", "b"], tag="test")

    def test_tag_context_manager(self):
        logger = RDAgentLog()
        with logger.tag("test_tag"):
            logger.info("inside tag")
        logger.info("outside tag")

    def test_debug_does_not_raise_on_empty(self):
        logger = RDAgentLog()
        logger.debug("")
        logger.debug("")
        logger.debug("")

    def test_debug_tag_propagation(self):
        logger = RDAgentLog()
        with logger.tag("debug_context"):
            logger.debug("debug with tag", tag="inner")
        logger.debug("debug outside")


class TestRDAgentLogMethods:
    """Verify all log-level methods exist and are callable."""

    def test_all_methods_present(self):
        logger = RDAgentLog()
        expected = {"debug", "info", "warning", "error", "log_object"}
        for name in expected:
            assert hasattr(logger, name), f"RDAgentLog missing method: {name}"

    def test_methods_are_bound(self):
        logger = RDAgentLog()
        for name in ("debug", "info", "warning", "error"):
            method = getattr(logger, name)
            assert callable(method)
            # Should accept at minimum a string message
            method("bound method test")
