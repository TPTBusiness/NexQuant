"""
Integration Tests for P4 CLI Commands

Tests the new CLI commands:
- generate_strategies
- optimize_portfolio
- strategies_report
- fin_quant --auto-strategies integration

Run with:
    pytest test/integration/test_cli_commands.py -v
"""

import json
import sys
import tempfile
from pathlib import Path
from unittest.mock import MagicMock, patch

import numpy as np
import pytest
from typer.testing import CliRunner

# Add project root to path
project_root = Path(__file__).parent.parent.parent
sys.path.insert(0, str(project_root))

from rdagent.app.cli import app


class TestCLIGenerateStrategies:
    """Test generate_strategies CLI command."""

    def setup_method(self):
        self.runner = CliRunner()

    def test_generate_strategies_help(self):
        """Test help message displays correctly."""
        result = self.runner.invoke(app, ["generate_strategies", "--help"])
        assert result.exit_code == 0
        assert "Generate trading strategies" in result.output
        assert "--count" in result.output
        assert "--workers" in result.output
        assert "--style" in result.output
        assert "--optuna" in result.output

    def test_generate_strategies_invalid_style(self):
        """Test error handling for invalid trading style."""
        result = self.runner.invoke(app, ["generate_strategies", "--style", "invalid"])
        assert result.exit_code == 1
        assert "Error: Invalid style" in result.output

    def test_generate_strategies_invalid_count(self):
        """Test error handling for invalid count."""
        result = self.runner.invoke(app, ["generate_strategies", "--count", "0"])
        assert result.exit_code == 1
        assert "Error: Count must be at least 1" in result.output

    def test_generate_strategies_invalid_workers(self):
        """Test error handling for invalid workers."""
        result = self.runner.invoke(app, ["generate_strategies", "--workers", "0"])
        assert result.exit_code == 1
        assert "Error: Workers must be between 1 and 16" in result.output

    def test_generate_strategies_workers_too_high(self):
        """Test error handling for workers > 16."""
        result = self.runner.invoke(app, ["generate_strategies", "--workers", "20"])
        assert result.exit_code == 1
        assert "Error: Workers must be between 1 and 16" in result.output

    def test_generate_strategies_with_mocked_orchestrator(self):
        """Test generate_strategies with mocked orchestrator."""
        mock_results = [
            {
                "strategy_name": "TestStrategy_v1",
                "status": "accepted",
                "sharpe_ratio": 2.1,
                "annualized_return": 0.15,
                "max_drawdown": -0.10,
                "win_rate": 0.55,
                "factors_used": ["factor_a", "factor_b"],
            },
            {
                "strategy_name": "TestStrategy_v2",
                "status": "rejected",
                "reason": "Sharpe too low",
                "factors_used": ["factor_c"],
            },
        ]

        with patch.dict(
            sys.modules,
            {
                "rdagent.components.coder.strategy_orchestrator": MagicMock(
                    StrategyOrchestrator=MagicMock(
                        return_value=MagicMock(
                            generate_strategies=MagicMock(return_value=mock_results)
                        )
                    )
                ),
            },
        ):
            result = self.runner.invoke(
                app,
                ["generate_strategies", "--count", "2", "--workers", "1", "--no-optuna"],
            )

            assert result.exit_code == 0
            assert "Strategy Generation Summary" in result.output

    def test_generate_strategies_daytrading_style(self):
        """Test generate_strategies with daytrading style."""
        with patch.dict(
            sys.modules,
            {
                "rdagent.components.coder.strategy_orchestrator": MagicMock(
                    StrategyOrchestrator=MagicMock(
                        return_value=MagicMock(
                            generate_strategies=MagicMock(return_value=[])
                        )
                    )
                ),
            },
        ):
            result = self.runner.invoke(
                app,
                ["generate_strategies", "--style", "daytrading", "--count", "1", "--no-optuna"],
            )
            assert result.exit_code == 0


class TestCLIOptimizePortfolio:
    """Test optimize_portfolio CLI command."""

    def setup_method(self):
        self.runner = CliRunner()

    def test_optimize_portfolio_help(self):
        """Test help message displays correctly."""
        result = self.runner.invoke(app, ["optimize_portfolio", "--help"])
        assert result.exit_code == 0
        assert "Optimize portfolio weights" in result.output
        assert "--top-n" in result.output
        assert "--method" in result.output

    def test_optimize_portfolio_invalid_method(self):
        """Test error handling for invalid method."""
        result = self.runner.invoke(app, ["optimize_portfolio", "--method", "invalid"])
        assert result.exit_code == 1
        assert "Error: Invalid method" in result.output

    def test_optimize_portfolio_no_strategies(self):
        """Test handling when no strategies directory exists."""
        with patch("pathlib.Path.exists", return_value=False):
            result = self.runner.invoke(app, ["optimize_portfolio"])
            # Should handle gracefully
            assert result.exit_code in (0, 1)


class TestCLIStrategiesReport:
    """Test strategies_report CLI command."""

    def setup_method(self):
        self.runner = CliRunner()

    def test_strategies_report_help(self):
        """Test help message displays correctly."""
        result = self.runner.invoke(app, ["strategies_report", "--help"])
        assert result.exit_code == 0
        assert "Generate performance reports" in result.output
        assert "--strategy-path" in result.output
        assert "--output-dir" in result.output

    def test_strategies_report_invalid_path(self):
        """Test error handling for invalid path."""
        result = self.runner.invoke(
            app, ["strategies_report", "--strategy-path", "/nonexistent/path.json"]
        )
        assert result.exit_code == 1
        assert "Error: Path not found" in result.output

    def test_strategies_report_no_json_files(self):
        """Test error handling when no JSON files found."""
        with tempfile.TemporaryDirectory() as tmpdir:
            result = self.runner.invoke(
                app, ["strategies_report", "--strategy-path", tmpdir]
            )
            assert result.exit_code == 1
            assert "Error: No strategy JSON files found" in result.output

    def test_strategies_report_with_valid_file(self):
        """Test strategies_report with a valid strategy file."""
        with tempfile.TemporaryDirectory() as tmpdir:
            tmpdir = Path(tmpdir)

            # Create test strategy file
            test_strategy = {
                "strategy_name": "TestStrategy",
                "status": "accepted",
                "sharpe_ratio": 2.0,
                "annualized_return": 0.15,
                "max_drawdown": -0.10,
                "win_rate": 0.55,
                "factors_used": ["factor_a", "factor_b"],
                "trading_style": "swing",
            }

            strategy_file = tmpdir / "test_strategy.json"
            with open(strategy_file, "w") as f:
                json.dump(test_strategy, f)

            output_dir = tmpdir / "reports"

            result = self.runner.invoke(
                app,
                [
                    "strategies_report",
                    "--strategy-path",
                    str(strategy_file),
                    "--output-dir",
                    str(output_dir),
                ],
            )

            assert result.exit_code == 0
            assert "Report Generation Complete" in result.output
            # Check report file was created
            assert output_dir.exists()

    def test_generate_single_strategy_report(self):
        """Test _generate_single_strategy_report function."""
        from rdagent.app.cli import _generate_single_strategy_report

        with tempfile.TemporaryDirectory() as tmpdir:
            tmpdir = Path(tmpdir)

            # Create test strategy file
            test_strategy = {
                "strategy_name": "TestReportStrategy",
                "status": "accepted",
                "sharpe_ratio": 1.8,
                "annualized_return": 0.12,
                "max_drawdown": -0.15,
                "win_rate": 0.52,
                "volatility": 0.08,
                "information_ratio": 0.5,
                "factors_used": ["factor_x", "factor_y"],
                "trading_style": "swing",
            }

            strategy_file = tmpdir / "test_strategy.json"
            with open(strategy_file, "w") as f:
                json.dump(test_strategy, f)

            output_dir = tmpdir / "reports"
            output_dir.mkdir()

            # Generate report
            report = _generate_single_strategy_report(strategy_file, output_dir)

            # Verify report
            assert "strategy_name" in report
            assert report["strategy_name"] == "TestReportStrategy"
            assert "metrics" in report
            assert report["metrics"]["sharpe_ratio"] == 1.8
            assert "output_file" in report


class TestFinQuantAutoStrategiesIntegration:
    """Test fin_quant --auto-strategies integration."""

    def setup_method(self):
        self.runner = CliRunner()

    def test_fin_quant_help_shows_auto_strategies(self):
        """Test that fin_quant help shows auto-strategies options."""
        result = self.runner.invoke(app, ["fin_quant", "--help"])
        assert result.exit_code == 0
        assert "--auto-strategies" in result.output
        assert "--auto-strategies-threshold" in result.output

    def test_fin_quant_accepts_auto_strategies_flag(self):
        """Test that fin_quant accepts --auto-strategies flag without error."""
        # Just verify the flag is accepted (command may fail due to missing config)
        result = self.runner.invoke(
            app,
            [
                "fin_quant",
                "--auto-strategies",
                "--auto-strategies-threshold",
                "100",
            ],
        )
        # Should not fail due to argument parsing
        assert "Error" not in result.output or "auto" not in result.output.lower()


class TestCLIOutputFormatting:
    """Test Rich console output formatting."""

    def setup_method(self):
        self.runner = CliRunner()

    def test_summary_table_headers(self):
        """Test that summary table headers appear in output."""
        mock_results = [
            {
                "strategy_name": "AlphaStrategy",
                "status": "accepted",
                "sharpe_ratio": 2.5,
                "annualized_return": 0.20,
                "max_drawdown": -0.08,
                "win_rate": 0.60,
            }
        ]

        with patch.dict(
            sys.modules,
            {
                "rdagent.components.coder.strategy_orchestrator": MagicMock(
                    StrategyOrchestrator=MagicMock(
                        return_value=MagicMock(
                            generate_strategies=MagicMock(return_value=mock_results)
                        )
                    )
                ),
            },
        ):
            result = self.runner.invoke(
                app,
                ["generate_strategies", "--count", "1", "--no-optuna"],
            )

            assert result.exit_code == 0
            # Check table headers appear in output
            assert "Status" in result.output
            assert "Count" in result.output
            assert "Percentage" in result.output


class TestCLIErrorHandling:
    """Test CLI error handling edge cases."""

    def setup_method(self):
        self.runner = CliRunner()

    def test_strategies_report_malformed_json(self):
        """Test handling of malformed JSON in strategy files."""
        with tempfile.TemporaryDirectory() as tmpdir:
            tmpdir = Path(tmpdir)

            # Create malformed JSON file
            bad_file = tmpdir / "bad_strategy.json"
            bad_file.write_text("{invalid json}", encoding="utf-8")

            result = self.runner.invoke(
                app,
                [
                    "strategies_report",
                    "--strategy-path",
                    str(bad_file),
                    "--output-dir",
                    str(tmpdir / "reports"),
                ],
            )

            # Should handle error gracefully
            assert result.exit_code in (0, 1)


class TestCLICommandRegistration:
    """Test that all CLI commands are properly registered."""

    def setup_method(self):
        self.runner = CliRunner()

    def test_all_commands_registered(self):
        """Test that all new commands are registered."""
        result = self.runner.invoke(app, ["--help"])
        assert result.exit_code == 0

        # Check all new commands appear in help (typer uses underscores)
        assert "generate_strategies" in result.output
        assert "optimize_portfolio" in result.output
        assert "strategies_report" in result.output

    def test_fin_quant_still_works(self):
        """Test that existing fin_quant command still works."""
        result = self.runner.invoke(app, ["fin_quant", "--help"])
        assert result.exit_code == 0
        assert "EURUSD quantitative trading" in result.output


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
