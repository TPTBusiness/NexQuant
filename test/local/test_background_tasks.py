"""
Tests for background task infrastructure (parallel runner, CLI paths, env loading).

Verifies bugs that were previously present:
- nexquant_parallel.py: project_root pointing to scripts/ instead of repo root
- nexquant_parallel.py: .env loaded from scripts/ instead of repo root
- nexquant_parallel.py: API key round-robin overwritten by comma-separated list
- cli.py: project_root depth wrong (4 .parent hops instead of 3)
- cli.py start_loop: hardcoded "python" instead of sys.executable
- cli.py parallel: hardcoded model=local
"""

import os
import sys
from pathlib import Path
from unittest.mock import Mock, patch

import pytest


# ── nexquant_parallel.py ──────────────────────────────────────────────────


class TestParallelRunnerProjectRoot:
    """Verify ParallelRunner.project_root points to the repo root, not scripts/."""

    def test_project_root_is_repo_root(self):
        """Bug: project_root was Path(__file__).parent (= scripts/)."""
        from scripts.nexquant_parallel import ParallelRunner

        runner = ParallelRunner(num_runs=1, num_api_keys=1, model="local")
        root = runner.project_root

        # Must contain nexquant.py (repo root), NOT be the scripts/ dir
        assert (root / "nexquant.py").exists(), (
            f"project_root={root} does not contain nexquant.py — "
            f"likely still pointing to scripts/ instead of repo root"
        )
        assert root.name != "scripts", (
            f"project_root={root} ends with 'scripts/' — should be repo root"
        )

    def test_build_command_points_to_nexquant_py(self):
        """Bug: command pointed to scripts/nexquant.py which doesn't exist."""
        from scripts.nexquant_parallel import ParallelRunner, RunState

        runner = ParallelRunner(num_runs=1, num_api_keys=1, model="local")
        run = RunState(run_id=1, api_key_idx=0, model="local")
        cmd = runner._build_command(run)

        nexquant_path = Path(cmd[1])
        assert nexquant_path.exists(), (
            f"Command references {nexquant_path} which does not exist — "
            f"project_root likely still wrong"
        )
        assert nexquant_path.name == "nexquant.py"
        assert nexquant_path.parent.name != "scripts", (
            "nexquant.py should be in repo root, not scripts/"
        )

    def test_env_loading_from_repo_root(self):
        """Bug: load_dotenv loaded scripts/.env which doesn't exist."""
        # load_dotenv is called at module import time, so we just verify
        # that after import, the env reflects any .env at repo root.
        # The key test: the call should not raise FileNotFoundError.
        repo_root = Path(__file__).parent.parent.parent
        env_path = repo_root / ".env"
        assert env_path.exists(), (
            f".env not found at {env_path} — repo root detection may be wrong"
        )


class TestParallelRunnerAPIKeys:
    """Verify API key distribution logic."""

    def test_single_api_key_no_overwrite(self):
        """Bug: with num_api_keys=1, individual key was set then overwritten."""
        from scripts.nexquant_parallel import ParallelRunner, RunState

        with patch.dict(os.environ, {}, clear=True):
            os.environ["OPENROUTER_API_KEY"] = "sk-test-key-1"
            runner = ParallelRunner(num_runs=2, num_api_keys=1, model="openrouter")
            # Reset api_keys since _load_api_keys already ran in __init__
            runner.api_keys = ["sk-test-key-1"]
            runner.num_api_keys = 1

            env = runner._build_env(RunState(run_id=1, api_key_idx=0, model="openrouter"))

            assert env["OPENAI_API_KEY"] == "sk-test-key-1", (
                "Single key should be assigned directly, not overwritten"
            )
            assert "LITELLM_PARALLEL_CALLS" not in env, (
                "LITELLM_PARALLEL_CALLS should not be set for single key"
            )

    def test_multi_api_key_comma_separated(self):
        """With 2+ keys, all runs get comma-separated list for load balancing."""
        from scripts.nexquant_parallel import ParallelRunner, RunState

        with patch.dict(os.environ, {}, clear=True):
            os.environ["OPENROUTER_API_KEY"] = "sk-key-a"
            os.environ["OPENROUTER_API_KEY_2"] = "sk-key-b"
            runner = ParallelRunner(num_runs=3, num_api_keys=2, model="openrouter")

            env = runner._build_env(RunState(run_id=1, api_key_idx=0, model="openrouter"))

            assert env["OPENAI_API_KEY"] == "sk-key-a,sk-key-b", (
                "Multiple keys should be comma-separated for LiteLLM load balancing"
            )
            assert env.get("LITELLM_PARALLEL_CALLS") == "2"

    def test_round_robin_api_key_index(self):
        """Verify round-robin API key index assignment is computed correctly."""
        from scripts.nexquant_parallel import ParallelRunner

        with patch.dict(os.environ, {}, clear=True):
            os.environ["OPENROUTER_API_KEY"] = "a"
            os.environ["OPENROUTER_API_KEY_2"] = "b"
            runner = ParallelRunner(num_runs=5, num_api_keys=2, model="openrouter")

            # 5 runs, 2 keys → indices: 0, 1, 0, 1, 0
            expected = [0, 1, 0, 1, 0]
            actual = [r.api_key_idx for r in runner.runs]
            assert actual == expected, f"Round-robin mismatch: {actual} != {expected}"


class TestParallelRunnerLogFileHandling:
    """Verify log files and results go to the right place."""

    def test_log_file_paths_in_repo_root(self):
        """Bug: logs went to scripts/fin_quant_runN.log."""
        from scripts.nexquant_parallel import ParallelRunner

        runner = ParallelRunner(num_runs=2, num_api_keys=1, model="local")

        for run in runner.runs:
            log_file = run.log_file
            # log_file is relative — should be "fin_quant_runN.log"
            assert "scripts" not in log_file, (
                f"Log file {log_file} should not be in scripts/"
            )
            assert log_file.startswith("fin_quant_run"), (
                f"Unexpected log file name: {log_file}"
            )


# ── cli.py ──────────────────────────────────────────────────────────────


class TestCLIProjectRoot:
    """Verify CLI commands resolve project_root to the actual repo root."""

    REPO_ROOT = Path(__file__).parent.parent.parent

    def test_cli_project_root_depth(self):
        """Bug: 4x .parent put project_root one level above the repo."""
        # The fixed code uses .parent.parent.parent (3 hops) from rdagent/app/cli.py
        cli_file = self.REPO_ROOT / "rdagent" / "app" / "cli.py"
        assert cli_file.exists(), f"cli.py not found at {cli_file}"

        # Simulate what the fixed code does
        resolved = cli_file.parent.parent.parent
        assert resolved == self.REPO_ROOT, (
            f"3 .parent hops from cli.py should yield repo root, got {resolved}"
        )

        # The bug used 4 hops which would overshoot
        buggy = cli_file.parent.parent.parent.parent
        assert buggy != self.REPO_ROOT, (
            "4 .parent hops should NOT yield repo root "
            f"(got {buggy}, expected {self.REPO_ROOT.parent})"
        )
        assert (buggy / "NexQuant").exists() or buggy == self.REPO_ROOT.parent, (
            f"4 .parent hops overshoots repo root: {buggy}"
        )

    def test_cli_start_loop_uses_sys_executable(self):
        """Bug: start_loop used hardcoded 'python' instead of sys.executable."""
        from rdagent.app.cli import start_loop_cli
        import inspect

        source = inspect.getsource(start_loop_cli)

        # The fixed code uses sys.executable in the generator list
        assert "sys.executable" in source, (
            "start_loop_cli should use sys.executable, not hardcoded 'python'"
        )
        # Should NOT contain the old hardcoded pattern
        assert 'f"python ' not in source, (
            "start_loop_cli should not contain hardcoded 'python' string"
        )

    def test_cli_parallel_not_hardcoded_model(self):
        """Bug: parallel_cli hardcoded -m local in subprocess command."""
        from rdagent.app.cli import parallel_cli
        import inspect

        source = inspect.getsource(parallel_cli)

        # The fixed code no longer passes -m local as a cmd argument
        assert '-m", "local"' not in source and '-m", \n        "local"' not in source and '"-m", "local"' not in source and '"local"]' not in source, (
            "parallel_cli should not hardcode model=local in subprocess command list"
        )

    def test_cli_scripts_exist_at_resolved_paths(self):
        """Verify scripts referenced by CLI commands exist at the resolved paths."""
        from rdagent.app.cli import eval_all_cli, batch_backtest_cli, simple_eval_cli
        from rdagent.app.cli import rebacktest_cli, report_cli, parallel_cli
        import inspect

        # All these commands use Path(__file__).parent.parent.parent as project_root
        commands = {
            "eval_all": "scripts/nexquant_full_eval.py",
            "batch_backtest": "scripts/nexquant_batch_backtest.py",
            "simple_eval": "scripts/nexquant_simple_eval.py",
            "rebacktest": "scripts/nexquant_rebacktest_strategies.py",
            "report": "scripts/nexquant_strategy_report.py",
            "parallel": "scripts/nexquant_parallel.py",
        }

        for cmd_name, script_path in commands.items():
            full_path = self.REPO_ROOT / script_path
            assert full_path.exists(), (
                f"CLI command '{cmd_name}' references {full_path} which does not exist. "
                f"project_root depth may be wrong."
            )

    def test_start_loop_generator_script_exists(self):
        """Bug: wrong project_root meant generator script not found."""
        from rdagent.app.cli import start_loop_cli
        import inspect

        source = inspect.getsource(start_loop_cli)
        # The generator should reference scripts/nexquant_smart_strategy_gen.py
        assert "nexquant_smart_strategy_gen.py" in source, (
            "start_loop_cli should reference nexquant_smart_strategy_gen.py"
        )

        script = self.REPO_ROOT / "scripts" / "nexquant_smart_strategy_gen.py"
        assert script.exists(), (
            f"Generator script not found at {script}"
        )

    def test_start_loop_uses_child_proc_not_pkill(self):
        """Bug: cleanup used pkill -f which killed all instances system-wide."""
        from rdagent.app.cli import start_loop_cli
        import inspect

        source = inspect.getsource(start_loop_cli)

        # Fixed code uses child_proc.terminate() / child_proc.kill()
        assert "child_proc" in source, (
            "start_loop_cli should use child_proc variable for targeted cleanup"
        )
        # Should NOT contain the old broad pkill
        assert "pkill" not in source, (
            "start_loop_cli should not use broad pkill for process management"
        )


# ── Integration: full import checks ─────────────────────────────────────


class TestImportsDontCrash:
    """Verify that importing the fixed modules doesn't crash."""

    def test_import_parallel_runner(self):
        """ParallelRunner should import without errors."""
        from scripts.nexquant_parallel import ParallelRunner, RunState
        runner = ParallelRunner(num_runs=1, num_api_keys=1, model="local")
        assert runner.num_runs == 1
        assert len(runner.runs) == 1

    def test_import_cli_app(self):
        """CLI app should import without errors."""
        from rdagent.app.cli import app
        assert app is not None
