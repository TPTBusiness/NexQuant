"""Tests for rdagent/oai/backend/litellm.py — LiteLLM API backend.

These are offline tests that don't require a running LLM server.
"""

from __future__ import annotations

from unittest.mock import MagicMock, patch

import pytest

from rdagent.oai.backend.litellm import LiteLLMAPIBackend


class TestLiteLLMAPIBackendInit:
    def test_creates_without_crash(self):
        backend = LiteLLMAPIBackend()
        assert backend is not None

    def test_has_inner_function(self):
        backend = LiteLLMAPIBackend()
        assert hasattr(backend, "_create_chat_completion_inner_function")

    def test_complete_kwargs_returns_dict_like(self):
        backend = LiteLLMAPIBackend()
        kwargs = backend.get_complete_kwargs()
        assert kwargs is not None

    def test_supports_response_schema_returns_bool(self):
        backend = LiteLLMAPIBackend()
        result = backend.supports_response_schema()
        assert isinstance(result, bool)


class TestLiteLLMAPIBackendTokenCounting:
    @patch("rdagent.oai.backend.litellm.token_counter")
    def test_calculate_token_from_messages_returns_int(self, mock_counter):
        mock_counter.return_value = 42
        backend = LiteLLMAPIBackend()
        result = backend._calculate_token_from_messages(
            [{"role": "user", "content": "hello"}]
        )
        assert isinstance(result, int)
        assert result == 42

    @patch("rdagent.oai.backend.litellm.token_counter")
    def test_calculate_token_from_messages_handles_empty(self, mock_counter):
        mock_counter.return_value = 0
        backend = LiteLLMAPIBackend()
        result = backend._calculate_token_from_messages([])
        assert result == 0


class TestLiteLLMAPIBackendStreaming:
    @patch("rdagent.oai.backend.litellm.completion")
    @patch("rdagent.oai.backend.litellm.token_counter")
    def test_non_streaming_response(self, mock_tokens, mock_completion):
        mock_tokens.return_value = 10
        from rdagent.oai.backend.litellm import LITELLM_SETTINGS
        LITELLM_SETTINGS.chat_stream = False
        try:
            # Build a proper mock response structure
            resp = MagicMock()
            choice = MagicMock()
            msg = MagicMock()
            msg.content = '{"key": "value"}'
            choice.message = msg
            choice.finish_reason = "stop"
            resp.choices = [choice]
            mock_completion.return_value = resp

            backend = LiteLLMAPIBackend()
            content, finish = backend._create_chat_completion_inner_function(
                messages=[{"role": "user", "content": "test"}],
            )
            assert '{"key": "value"}' in str(content)
            assert finish == "stop"
        finally:
            LITELLM_SETTINGS.chat_stream = True

    @patch("rdagent.oai.backend.litellm.completion")
    @patch("rdagent.oai.backend.litellm.token_counter")
    def test_streaming_response(self, mock_tokens, mock_completion):
        mock_tokens.return_value = 5

        chunk1 = {"choices": [{"finish_reason": None, "delta": {"content": "hello"}}]}
        chunk2 = {"choices": [{"finish_reason": "stop", "delta": {}}]}
        mock_completion.return_value = [chunk1, chunk2]

        backend = LiteLLMAPIBackend()
        from rdagent.oai.backend.litellm import LITELLM_SETTINGS
        LITELLM_SETTINGS.chat_stream = True
        try:
            content, finish = backend._create_chat_completion_inner_function(
                messages=[{"role": "user", "content": "hi"}],
            )
            assert "hello" in content
        finally:
            LITELLM_SETTINGS.chat_stream = False


class TestLiteLLMAPIBackendEdgeCases:
    def test_empty_messages_token_count(self):
        backend = LiteLLMAPIBackend()
        with patch("rdagent.oai.backend.litellm.token_counter", return_value=0):
            result = backend._calculate_token_from_messages([])
            assert result == 0

    def test_unicode_messages_token_count(self):
        backend = LiteLLMAPIBackend()
        messages = [{"role": "user", "content": "üéñ–—…€🦀"}]
        with patch("rdagent.oai.backend.litellm.token_counter", return_value=5):
            result = backend._calculate_token_from_messages(messages)
            assert result == 5

    def test_very_long_message_token_count(self):
        backend = LiteLLMAPIBackend()
        long_msg = "hello " * 10000
        messages = [{"role": "user", "content": long_msg}]
        with patch("rdagent.oai.backend.litellm.token_counter", return_value=20000):
            result = backend._calculate_token_from_messages(messages)
            assert result == 20000

    def test_build_log_messages_returns_string(self):
        backend = LiteLLMAPIBackend()
        messages = [
            {"role": "system", "content": "test system"},
            {"role": "user", "content": "test user"},
        ]
        result = backend._build_log_messages(messages)
        assert isinstance(result, str)

    def test_supports_response_schema_does_not_crash(self):
        backend = LiteLLMAPIBackend()
        for _ in range(10):
            backend.supports_response_schema()
