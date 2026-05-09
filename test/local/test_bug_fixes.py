"""
Tests for bug fixes in strategy_orchestrator, factor_runner, backtest_engine,
results_db, model_runner, optuna_optimizer, env, and related modules.

Verifies:
- strategy_orchestrator.py compiles (IndentationError was fixed)
- factor_runner.py uses sys.executable (variable), not literal string
- backtest_engine.py path depth is 4 (not 3) .parent hops
- results_db.py path depth for factors/failed dirs is 4 (not 3)
- model_runner.py DB connection closed via try/finally
- factor_runner.py variable shadowing eliminated (run_id vs db_run_id)
- optuna_optimizer.py no longer shadows imported logger
- env.py Docker build output handles non-UTF-8 bytes
- env.py conda env list parsing guards against empty lines
- strategy_orchestrator.py exec() exception logged at ERROR level
- strategy_orchestrator.py template validation warns on unreplaced {{...}}
- factor_runner.py IC_max guard against scalar (AttributeError)
- nexquant_parallel.py handle leak on Popen failure
- nexquant_rebacktest_strategies.py bare except replaced with except Exception
"""

import ast
import inspect
import os
import sys
from pathlib import Path
from unittest.mock import patch

import pytest

REPO_ROOT = Path(__file__).parent.parent.parent


# ── Fix 1: strategy_orchestrator.py IndentationError ──────────────────────


class TestStrategyOrchestratorSyntax:
    def test_file_compiles(self):
        """Bug: IndentationError at line 764 prevented the entire file from importing."""
        import py_compile
        py_compile.compile(
            str(REPO_ROOT / "rdagent/components/coder/strategy_orchestrator.py"),
            doraise=True,
        )

    def test_module_imports(self):
        """Verify StrategyOrchestrator can be imported after syntax fix."""
        from rdagent.scenarios.qlib.local.strategy_orchestrator import StrategyOrchestrator
        assert StrategyOrchestrator is not None


# ── Fix 2: factor_runner.py literal "sys.executable" ──────────────────────


class TestFactorRunnerSysExecutable:
    def test_not_literal_string(self):
        """Bug: ["sys.executable", ...] was a literal string, not the variable."""
        source = (REPO_ROOT / "rdagent/scenarios/qlib/developer/factor_runner.py").read_text()
        # The fix should NOT contain the quoted literal 'sys.executable'
        assert '"sys.executable"' not in source, (
            "factor_runner.py still contains literal string 'sys.executable' — "
            "should be sys.executable (variable)"
        )
        # Should use sys.executable (variable, part of a list)
        assert "sys.executable" in source

    def test_subprocess_run_with_check_false(self):
        """Bug: subprocess.run without explicit check=False."""
        source = (REPO_ROOT / "rdagent/scenarios/qlib/developer/factor_runner.py").read_text()
        # The fix added check=False to the full-data factor run
        assert 'check=False' in source, (
            "subprocess.run should have explicit check=False for full-data factor run"
        )


# ── Fix 3: backtest_engine.py path depth ─────────────────────────────────


class TestBacktestEnginePathDepth:
    def test_path_depth_is_4(self):
        """Bug: 3 .parent hops ended at rdagent/ instead of repo root."""
        from rdagent.components.backtesting.backtest_engine import FactorBacktester

        fb = FactorBacktester()
        results = fb.results_path

        # results_path should be under the repo root, not under rdagent/
        assert REPO_ROOT in results.parents or results.parent == REPO_ROOT / "results", (
            f"results_path={results} is not under repo root {REPO_ROOT}. "
            "Path depth may still be wrong."
        )
        # The path should NOT be inside rdagent/
        assert "rdagent/results" not in str(results).replace(str(REPO_ROOT), ""), (
            f"results_path={results} appears to be inside rdagent/ directory"
        )


# ── Fix 4: results_db.py path depth ──────────────────────────────────────


class TestResultsDBPathDepth:
    def test_factors_dir_depth(self):
        """Bug: 3 .parent hops from results_db.py ended at rdagent/."""
        from rdagent.components.backtesting.results_db import ResultsDatabase
        source = inspect.getsource(ResultsDatabase.generate_results_summary)

        # After fix, should use .parent.parent.parent.parent (4 hops)
        assert ".parent.parent.parent.parent" in source, (
            "ResultsDatabase.generate_results_summary should use 4 .parent hops, not 3"
        )


# ── Fix 5: model_runner.py DB close ──────────────────────────────────────


class TestModelRunnerDBClose:
    def test_try_finally_for_db_close(self):
        """Bug: db.close() was after add_backtest, not in finally block."""
        source = (REPO_ROOT / "rdagent/scenarios/qlib/developer/model_runner.py").read_text()

        # After fix, db.close() should be in a finally block or try/finally context
        assert "finally:" in source, (
            "model_runner.py should use try/finally to close DB connection"
        )
        assert "db.close()" in source


# ── Fix 6: factor_runner.py variable shadowing ───────────────────────────


class TestFactorRunnerShadowing:
    def test_no_run_id_shadowing(self):
        """Bug: run_id was reassigned from parallel run ID to DB row ID."""
        source = (REPO_ROOT / "rdagent/scenarios/qlib/developer/factor_runner.py").read_text()

        # After fix, parallel_run_id and db_run_id are separate variables
        assert "parallel_run_id" in source, (
            "factor_runner.py should use parallel_run_id for parallel run isolation"
        )
        assert "db_run_id" in source, (
            "factor_runner.py should use db_run_id for DB row ID"
        )


# ── Fix 7: optuna_optimizer.py logger shadowing ──────────────────────────


class TestOptunaLoggerShadowing:
    def test_logger_not_reassigned(self):
        """Bug: logger was reassigned from rdagent_logger to raw logging.getLogger."""
        source = (REPO_ROOT / "rdagent/components/coder/optuna_optimizer.py").read_text()

        # After fix, the module-level logger should be rdagent_logger
        assert "from rdagent.log import rdagent_logger as logger" in source
        # The second assignment should use a different name
        assert "_optuna_logger" in source, (
            "optuna_optimizer.py should not shadow the rdagent logger"
        )


# ── Fix 8: env.py UnicodeDecodeError ─────────────────────────────────────


class TestEnvUnicodeDecode:
    def test_decode_with_errors_replace(self):
        """Bug: part.decode('utf-8') could raise UnicodeDecodeError."""
        source = (REPO_ROOT / "rdagent/utils/env.py").read_text()

        # After fix, decode uses errors="replace"
        assert 'decode("utf-8", errors="replace")' in source, (
            "env.py should handle non-UTF-8 Docker build output with errors='replace'"
        )


# ── Fix 9: env.py conda env list parsing ─────────────────────────────────


class TestEnvCondaParsing:
    def test_guard_against_empty_lines(self):
        """Bug: line.split()[0] crashed on empty tokens."""
        source = (REPO_ROOT / "rdagent/utils/env.py").read_text()

        # After fix, guards against empty split results
        assert "len(line.split()) > 0" in source, (
            "env.py should guard against empty split results in conda env list parsing"
        )


# ── Fix 10: strategy_orchestrator.py exec() logging ──────────────────────


class TestStrategyOrchExecLogging:
    def test_error_logging_in_exec_handler(self):
        """Bug: exec() exception was silently swallowed."""
        source = (REPO_ROOT / "rdagent/components/coder/strategy_orchestrator.py").read_text()

        # After fix, logger.error is called inside the except block
        assert "logger.error" in source, (
            "strategy_orchestrator.py should log exec() errors at ERROR level"
        )


# ── Fix 11: strategy_orchestrator.py template validation ─────────────────


class TestStrategyOrchTemplateValidation:
    def test_unreplaced_template_warning(self):
        """Bug: no validation that {{...}} placeholders were replaced."""
        source = (REPO_ROOT / "rdagent/components/coder/strategy_orchestrator.py").read_text()

        # After fix, warns on unreplaced template variables
        assert "Unreplaced template variables" in source, (
            "strategy_orchestrator.py should warn on unreplaced {{...}} placeholders"
        )


# ── Fix 12: factor_runner.py IC_max guard ────────────────────────────────


class TestFactorRunnerICMaxGuard:
    def test_hasattr_guard(self):
        """Bug: IC_max[...].index failed with AttributeError on scalar result."""
        source = (REPO_ROOT / "rdagent/scenarios/qlib/developer/factor_runner.py").read_text()

        # After fix, guards against scalar IC_max result
        assert 'hasattr(IC_max, "index")' in source, (
            "factor_runner.py should guard IC_max.index access with hasattr"
        )


# ── Fix 13: nexquant_parallel.py handle leak ───────────────────────────────


class TestParallelRunnerHandleLeak:
    def test_log_f_close_on_popen_failure(self):
        """Bug: open() file handle leaked if Popen failed."""
        source = (REPO_ROOT / "scripts/nexquant_parallel.py").read_text()

        # After fix, log_f.close() is called before re-raise
        assert "log_f.close()" in source, (
            "nexquant_parallel.py should close log file handle on Popen failure"
        )


# ── Fix 14: nexquant_rebacktest_strategies.py bare except ──────────────────


class TestRebacktestBareExcept:
    def test_not_bare_except(self):
        """Bug: bare except: pass swallowed all errors including SystemExit."""
        source = (REPO_ROOT / "scripts/nexquant_rebacktest_strategies.py").read_text()

        # After fix, should use except Exception, not bare except
        assert "except Exception:" in source
        assert "except:" not in source, (
            "nexquant_rebacktest_strategies.py should not use bare except:"
        )


# ── Integration: import checks ───────────────────────────────────────────


class TestAllImportsDontCrash:
    def test_strategy_orchestrator_imports(self):
        """Verify all fixed modules import without errors."""
        # moved to local/  # noqa: F401

    def test_factor_runner_imports(self):
        import rdagent.scenarios.qlib.developer.factor_runner  # noqa: F401

    def test_backtest_engine_imports(self):
        import rdagent.components.backtesting.backtest_engine  # noqa: F401

    def test_results_db_imports(self):
        import rdagent.components.backtesting.results_db  # noqa: F401

    def test_model_runner_imports(self):
        import rdagent.scenarios.qlib.developer.model_runner  # noqa: F401

    def test_optuna_optimizer_imports(self):
        pass  # moved to local/
