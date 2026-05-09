"""Deep tests for nexquant_parallel.py — property-based, state transitions, edge cases.

Tests RunState, ParallelRunner configuration, environment building,
command building, and API key loading logic.
"""

from __future__ import annotations
from hypothesis import given, settings, HealthCheck
from hypothesis import strategies as st

import os
import signal
import sys
import tempfile
from datetime import datetime
from pathlib import Path
from unittest.mock import MagicMock, patch

import pytest

PROJECT_ROOT = Path(__file__).parent.parent.parent
sys.path.insert(0, str(PROJECT_ROOT))

from hypothesis import given, settings, HealthCheck
from hypothesis import strategies as st


@pytest.fixture
def runstate():
    from scripts.nexquant_parallel import RunState
    return RunState(run_id=1, api_key_idx=0, model="local")


class TestRunState:
    def test_init_defaults(self, runstate):
        assert runstate.run_id == 1
        assert runstate.api_key_idx == 0
        assert runstate.model == "local"
        assert runstate.status == "pending"
        assert runstate.process is None
        assert runstate.exit_code is None

    def test_elapsed_not_started(self, runstate):
        assert runstate.elapsed == "--:--:--"

    def test_elapsed_running(self, runstate):
        runstate.start_time = datetime(2024, 1, 1, 12, 0, 0)
        with patch("scripts.nexquant_parallel.datetime") as mock_dt:
            mock_dt.now.return_value = datetime(2024, 1, 1, 13, 30, 45)
            assert runstate.elapsed == "01:30:45"

    def test_elapsed_completed(self, runstate):
        runstate.start_time = datetime(2024, 1, 1, 12, 0, 0)
        runstate.end_time = datetime(2024, 1, 1, 14, 5, 30)
        assert runstate.elapsed == "02:05:30"

    def test_elapsed_over_24h(self, runstate):
        runstate.start_time = datetime(2024, 1, 1, 0, 0, 0)
        runstate.end_time = datetime(2024, 1, 3, 6, 30, 15)
        assert runstate.elapsed == "54:30:15"

    @pytest.mark.parametrize("status,icon", [
        ("pending", "⏳"), ("running", "🔄"), ("success", "✅"),
        ("failed", "❌"), ("stopped", "⏹️"),
    ])
    def test_status_icons(self, runstate, status, icon):
        runstate.status = status
        assert runstate.status_icon == icon

    def test_unknown_status_icon(self, runstate):
        runstate.status = "weird_status"
        assert runstate.status_icon == "❓"

    @given(
        hours=st.integers(min_value=0, max_value=1000),
        mins=st.integers(min_value=0, max_value=59),
        secs=st.integers(min_value=0, max_value=59),
    )
    @settings(max_examples=200, deadline=5000,
              suppress_health_check=[HealthCheck.function_scoped_fixture])
    def test_elapsed_format_property(self, runstate, hours, mins, secs):
        """Elapsed time format must always be HH:MM:SS with zero-padding."""
        runstate.start_time = datetime(2024, 1, 1, 0, 0, 0)
        total_secs = hours * 3600 + mins * 60 + secs
        if total_secs < 365 * 86400:
            from datetime import timedelta
            runstate.end_time = runstate.start_time + timedelta(seconds=total_secs)
            e = runstate.elapsed
            parts = e.split(":")
            assert len(parts) == 3
            assert len(parts[1]) == 2 and len(parts[2]) == 2  # mins and secs always 2 digits
            assert all(p.isdigit() for p in parts)
            assert int(parts[0]) == hours


class TestParallelRunnerConfig:
    @patch.dict(os.environ, {}, clear=True)
    def test_load_api_keys_openrouter(self):
        from scripts.nexquant_parallel import ParallelRunner
        with patch.dict(os.environ, {
            "OPENROUTER_API_KEY": "sk-key1",
            "OPENROUTER_API_KEY_2": "sk-key2",
        }):
            runner = ParallelRunner(num_runs=2, num_api_keys=2, model="openrouter")
            keys = runner.api_keys
            assert len(keys) == 2
            assert keys[0].startswith("sk-")

    @patch.dict(os.environ, {}, clear=True)
    def test_load_api_keys_local(self):
        from scripts.nexquant_parallel import ParallelRunner
        runner = ParallelRunner(num_runs=1, num_api_keys=1, model="local")
        assert runner.api_keys == ["local"]

    @patch.dict(os.environ, {}, clear=True)
    def test_load_api_keys_round_robin(self):
        from scripts.nexquant_parallel import ParallelRunner
        runner = ParallelRunner(num_runs=5, num_api_keys=2, model="local")
        assert len(runner.runs) == 5
        idxs = [r.api_key_idx for r in runner.runs]
        assert idxs == [0, 1, 0, 1, 0]

    @patch.dict(os.environ, {}, clear=True)
    def test_build_env_local_model(self):
        from scripts.nexquant_parallel import ParallelRunner, RunState
        with patch.dict(os.environ, {
            "OPENAI_API_KEY": "local",
            "OPENAI_API_BASE": "http://localhost:8081/v1",
            "CHAT_MODEL": "openai/qwen3.5-35b",
        }):
            runner = ParallelRunner(num_runs=1, num_api_keys=1, model="local")
            rs = RunState(run_id=1, api_key_idx=0, model="local")
            env = runner._build_env(rs)
            assert env["OPENAI_API_KEY"] == "local"
            assert "localhost:8081" in env["OPENAI_API_BASE"]
            assert env["CHAT_MODEL"] == "openai/qwen3.5-35b"

    @patch.dict(os.environ, {}, clear=True)
    def test_build_env_openrouter(self):
        from scripts.nexquant_parallel import ParallelRunner, RunState
        with patch.dict(os.environ, {
            "OPENROUTER_API_KEY": "sk-test",
            "OPENROUTER_API_KEY_2": "sk-test2",
        }):
            runner = ParallelRunner(num_runs=1, num_api_keys=2, model="openrouter")
            rs = RunState(run_id=1, api_key_idx=0, model="openrouter")
            env = runner._build_env(rs)
            assert "openrouter" in env["OPENAI_API_BASE"]

    @patch.dict(os.environ, {}, clear=True)
    def test_build_env_sets_workspace(self):
        from scripts.nexquant_parallel import ParallelRunner, RunState
        runner = ParallelRunner(num_runs=1, num_api_keys=1, model="local")
        rs = RunState(run_id=42, api_key_idx=0, model="local")
        env = runner._build_env(rs)
        assert "run42" in env.get("RD_AGENT_WORKSPACE", "")
        assert env["PARALLEL_RUN_ID"] == "42"

    @patch.dict(os.environ, {}, clear=True)
    def test_build_command(self):
        from scripts.nexquant_parallel import ParallelRunner, RunState
        runner = ParallelRunner(num_runs=1, num_api_keys=1, model="local")
        rs = RunState(run_id=7, api_key_idx=0, model="local")
        cmd = runner._build_command(rs)
        assert "nexquant.py" in cmd[1] or "nexquant" in cmd[1]
        assert "quant" in cmd
        assert "--model" in cmd
        assert "local" in cmd
        assert "7" in [str(a) for a in cmd]

    @patch.dict(os.environ, {}, clear=True)
    def test_parallel_runner_init_counts(self):
        from scripts.nexquant_parallel import ParallelRunner
        for n in [1, 3, 10]:
            runner = ParallelRunner(num_runs=n, num_api_keys=2, model="local")
            assert len(runner.runs) == n
            assert runner.num_runs == n


class TestParallelRunnerEdgeCases:
    @patch.dict(os.environ, {}, clear=True)
    def test_max_runs_limit(self):
        from scripts.nexquant_parallel import ParallelRunner
        runner = ParallelRunner(num_runs=100, num_api_keys=1, model="local")
        assert len(runner.runs) == 100

    @patch.dict(os.environ, {}, clear=True)
    def test_api_keys_empty_uses_local(self):
        from scripts.nexquant_parallel import ParallelRunner
        runner = ParallelRunner(num_runs=1, num_api_keys=2, model="openrouter")
        assert len(runner.api_keys) >= 1

    @patch.dict(os.environ, {}, clear=True)
    def test_build_env_preserves_existing_env(self, monkeypatch):
        monkeypatch.setenv("MY_CUSTOM_VAR", "custom_value")
        from scripts.nexquant_parallel import ParallelRunner, RunState
        runner = ParallelRunner(num_runs=1, num_api_keys=1, model="local")
        rs = RunState(run_id=1, api_key_idx=0, model="local")
        env = runner._build_env(rs)
        assert "MY_CUSTOM_VAR" in env
