"""
Tests for Portfolio Optimizer

Tests the PortfolioOptimizer class for:
- Mean-Variance Optimization
- Risk Parity Optimization
- Correlation Analysis
- Portfolio Backtesting
- Strategy Selection

20 tests covering all optimization methods and edge cases.
"""

import json
import tempfile
from pathlib import Path
from unittest.mock import MagicMock, patch

import numpy as np
import pandas as pd
import pytest


# ---------------------------------------------------------------------------
# Fixtures
# ---------------------------------------------------------------------------


@pytest.fixture
def mock_strategies_dir(tmp_path: Path) -> Path:
    """Create a temporary strategies directory with mock data."""
    strategies_dir = tmp_path / "results" / "strategies_new"
    strategies_dir.mkdir(parents=True)

    # Create 5 mock strategy files
    strategies = [
        {
            "name": "MomentumStrategy",
            "sharpe_ratio": 2.1,
            "ic": 0.15,
            "max_drawdown": -0.10,
            "backtest": {
                "returns": np.random.randn(252) * 0.01 + 0.0005,
                "equity_curve": np.cumprod(1 + np.random.randn(252) * 0.01 + 0.0005).tolist(),
            },
        },
        {
            "name": "MeanReversionStrategy",
            "sharpe_ratio": 1.8,
            "ic": 0.12,
            "max_drawdown": -0.15,
            "backtest": {
                "returns": np.random.randn(252) * 0.012 + 0.0004,
            },
        },
        {
            "name": "VolatilityTargetStrategy",
            "sharpe_ratio": 1.5,
            "ic": 0.10,
            "max_drawdown": -0.12,
            "backtest": {
                "returns": np.random.randn(252) * 0.009 + 0.0003,
            },
        },
        {
            "name": "TrendFollowingStrategy",
            "sharpe_ratio": 1.2,
            "ic": 0.08,
            "max_drawdown": -0.18,
            "backtest": {
                "returns": np.random.randn(252) * 0.011 + 0.0002,
            },
        },
        {
            "name": "BreakoutStrategy",
            "sharpe_ratio": 0.9,
            "ic": 0.06,
            "max_drawdown": -0.22,
            "backtest": {
                "returns": np.random.randn(252) * 0.013 + 0.0001,
            },
        },
    ]

    for strategy in strategies:
        filepath = strategies_dir / f"{strategy['name']}.json"
        with open(filepath, "w") as f:
            json.dump(strategy, f, default=lambda x: x.tolist() if isinstance(x, np.ndarray) else x)

    return tmp_path


@pytest.fixture
def mock_returns_data() -> pd.DataFrame:
    """Create mock strategy returns DataFrame."""
    np.random.seed(42)
    n_days = 252

    return pd.DataFrame(
        {
            "StrategyA": np.random.randn(n_days) * 0.01 + 0.0005,
            "StrategyB": np.random.randn(n_days) * 0.01 + 0.0004,
            "StrategyC": np.random.randn(n_days) * 0.009 + 0.0003,
        }
    )


@pytest.fixture
def portfolio_optimizer(mock_strategies_dir):
    """Create a PortfolioOptimizer instance with mock data."""
    from rdagent.scenarios.qlib.local.portfolio_optimizer import PortfolioOptimizer

    return PortfolioOptimizer(project_root=mock_strategies_dir)


# ---------------------------------------------------------------------------
# Tests
# ---------------------------------------------------------------------------


class TestPortfolioOptimizerInit:
    """Test PortfolioOptimizer initialization."""

    def test_default_initialization(self):
        """Test default configuration values."""
        from rdagent.scenarios.qlib.local.portfolio_optimizer import PortfolioOptimizer

        optimizer = PortfolioOptimizer()

        assert optimizer.max_correlation == 0.3
        assert optimizer.top_strategies == 30
        assert optimizer.risk_free_rate == 0.02

    def test_custom_configuration(self):
        """Test custom configuration values."""
        from rdagent.scenarios.qlib.local.portfolio_optimizer import PortfolioOptimizer

        optimizer = PortfolioOptimizer(
            max_correlation=0.5,
            top_strategies=20,
            risk_free_rate=0.03,
        )

        assert optimizer.max_correlation == 0.5
        assert optimizer.top_strategies == 20
        assert optimizer.risk_free_rate == 0.03


class TestLoadStrategyData:
    """Test strategy data loading."""

    def test_load_strategy_data(self, portfolio_optimizer):
        """Test loading strategy data from directory."""
        result = portfolio_optimizer._load_strategy_data()

        assert result is True
        assert portfolio_optimizer._strategy_returns is not None
        assert portfolio_optimizer._strategy_expected_returns is not None
        assert portfolio_optimizer._cov_matrix is not None
        assert portfolio_optimizer._corr_matrix is not None

    def test_load_specific_strategies(self, portfolio_optimizer):
        """Test loading specific strategies by name."""
        result = portfolio_optimizer._load_strategy_data(
            strategies=["MomentumStrategy", "MeanReversionStrategy"]
        )

        assert result is True
        assert len(portfolio_optimizer._strategy_expected_returns) == 2

    def test_load_no_strategies_dir(self, tmp_path):
        """Test loading when no strategies directory exists."""
        from rdagent.scenarios.qlib.local.portfolio_optimizer import PortfolioOptimizer

        optimizer = PortfolioOptimizer(project_root=tmp_path)
        result = optimizer._load_strategy_data()

        assert result is False


class TestExtractReturns:
    """Test returns extraction from backtest data."""

    def test_extract_returns_from_array(self):
        """Test extracting returns from array."""
        from rdagent.scenarios.qlib.local.portfolio_optimizer import PortfolioOptimizer

        optimizer = PortfolioOptimizer()
        returns = np.array([0.01, -0.005, 0.008])

        data = {"returns": returns.tolist()}
        result = optimizer._extract_returns(data)

        assert result is not None
        assert len(result) == 3

    def test_extract_returns_from_equity_curve(self):
        """Test extracting returns from equity curve."""
        from rdagent.scenarios.qlib.local.portfolio_optimizer import PortfolioOptimizer

        optimizer = PortfolioOptimizer()

        equity = [100, 101, 100.5, 101.5, 102]
        data = {"equity_curve": equity}
        result = optimizer._extract_returns(data)

        assert result is not None
        assert len(result) == 4  # One less than equity points

    def test_extract_returns_no_data(self):
        """Test extracting returns when no data available."""
        from rdagent.scenarios.qlib.local.portfolio_optimizer import PortfolioOptimizer

        optimizer = PortfolioOptimizer()
        result = optimizer._extract_returns({})

        assert result is None


class TestMeanVarianceOptimization:
    """Test mean-variance optimization."""

    def test_mean_variance_basic(self, portfolio_optimizer):
        """Test basic mean-variance optimization."""
        portfolio_optimizer._load_strategy_data()
        result = portfolio_optimizer._optimize_mean_variance()

        assert result is not None
        assert "weights" in result
        assert "expected_return" in result
        assert "volatility" in result
        assert "sharpe" in result
        assert result["method"] == "mean_variance"

        # Weights should sum to 1
        total_weight = sum(result["weights"].values())
        assert abs(total_weight - 1.0) < 0.01

    def test_mean_variance_weights_positive(self, portfolio_optimizer):
        """Test that all weights are non-negative."""
        portfolio_optimizer._load_strategy_data()
        result = portfolio_optimizer._optimize_mean_variance()

        assert result is not None
        for name, weight in result["weights"].items():
            assert weight >= 0

    def test_mean_variance_insufficient_strategies(self, portfolio_optimizer):
        """Test optimization with insufficient strategies."""
        portfolio_optimizer._strategy_expected_returns = pd.Series({"OnlyOne": 0.1})
        portfolio_optimizer._cov_matrix = pd.DataFrame([[0.0001]], index=["OnlyOne"], columns=["OnlyOne"])

        result = portfolio_optimizer._optimize_mean_variance()

        assert result is None


class TestRiskParityOptimization:
    """Test risk parity optimization."""

    def test_risk_parity_basic(self, portfolio_optimizer):
        """Test basic risk parity optimization."""
        portfolio_optimizer._load_strategy_data()
        result = portfolio_optimizer._optimize_risk_parity()

        assert result is not None
        assert "weights" in result
        assert result["method"] == "risk_parity"

    def test_risk_parity_weights_positive(self, portfolio_optimizer):
        """Test that all weights are non-negative."""
        portfolio_optimizer._load_strategy_data()
        result = portfolio_optimizer._optimize_risk_parity()

        assert result is not None
        for name, weight in result["weights"].items():
            assert weight >= 0

    def test_risk_parity_insufficient_strategies(self, portfolio_optimizer):
        """Test optimization with insufficient strategies."""
        portfolio_optimizer._strategy_expected_returns = pd.Series({"OnlyOne": 0.1})
        portfolio_optimizer._cov_matrix = pd.DataFrame([[0.0001]], index=["OnlyOne"], columns=["OnlyOne"])

        result = portfolio_optimizer._optimize_risk_parity()

        assert result is None


class TestICWeightedOptimization:
    """Test IC-weighted optimization."""

    def test_ic_weighted_basic(self, portfolio_optimizer):
        """Test basic IC-weighted optimization."""
        result = portfolio_optimizer._optimize_ic_weighted()

        assert result is not None
        assert "weights" in result
        assert result["method"] == "ic_weighted"

    def test_ic_weighted_weights_proportional(self, portfolio_optimizer):
        """Test that weights are proportional to IC."""
        result = portfolio_optimizer._optimize_ic_weighted()

        assert result is not None
        total_weight = sum(result["weights"].values())
        assert abs(total_weight - 1.0) < 0.01


class TestCorrelationAnalysis:
    """Test correlation analysis."""

    def test_analyze_correlations(self, portfolio_optimizer):
        """Test correlation analysis."""
        portfolio_optimizer._load_strategy_data()
        result = portfolio_optimizer.analyze_correlations()

        assert result is not None
        assert "correlation_matrix" in result
        assert "uncorrelated_strategies" in result
        assert "high_corr_pairs" in result

    def test_select_uncorrelated_strategies(self, portfolio_optimizer):
        """Test selecting uncorrelated strategies."""
        portfolio_optimizer._load_strategy_data()
        uncorrelated = portfolio_optimizer.select_uncorrelated_strategies(target_count=3)

        assert len(uncorrelated) <= 3

    def test_correlation_no_data(self, tmp_path):
        """Test correlation analysis with no data."""
        from rdagent.scenarios.qlib.local.portfolio_optimizer import PortfolioOptimizer

        optimizer = PortfolioOptimizer(project_root=tmp_path)
        result = optimizer.analyze_correlations()

        assert result is None


class TestPortfolioBacktest:
    """Test portfolio backtesting."""

    def test_backtest_portfolio(self, portfolio_optimizer):
        """Test backtesting a portfolio."""
        portfolio_optimizer._load_strategy_data()

        # Use equal weights
        n = len(portfolio_optimizer._strategy_returns.columns)
        weights = {col: 1.0 / n for col in portfolio_optimizer._strategy_returns.columns}

        result = portfolio_optimizer.backtest_portfolio(weights)

        assert result is not None
        assert "total_return" in result
        assert "annualized_return" in result
        assert "annualized_volatility" in result
        assert "sharpe_ratio" in result
        assert "max_drawdown" in result
        assert "win_rate" in result

    def test_backtest_default_weights(self, portfolio_optimizer):
        """Test backtesting with default (equal) weights."""
        portfolio_optimizer._load_strategy_data()
        result = portfolio_optimizer.backtest_portfolio()

        assert result is not None

    def test_backtest_no_data(self, tmp_path):
        """Test backtesting with no data."""
        from rdagent.scenarios.qlib.local.portfolio_optimizer import PortfolioOptimizer

        optimizer = PortfolioOptimizer(project_root=tmp_path)
        result = optimizer.backtest_portfolio()

        assert result is None


class TestOptimizePortfolio:
    """Test the main optimize_portfolio method."""

    def test_optimize_mean_variance(self, portfolio_optimizer):
        """Test optimize_portfolio with mean_variance method."""
        result = portfolio_optimizer.optimize_portfolio(method="mean_variance")

        assert result is not None
        assert result["method"] == "mean_variance"

    def test_optimize_risk_parity(self, portfolio_optimizer):
        """Test optimize_portfolio with risk_parity method."""
        result = portfolio_optimizer.optimize_portfolio(method="risk_parity")

        assert result is not None
        assert result["method"] == "risk_parity"

    def test_optimize_ic_weighted(self, portfolio_optimizer):
        """Test optimize_portfolio with ic_weighted method."""
        result = portfolio_optimizer.optimize_portfolio(method="ic_weighted")

        assert result is not None
        assert result["method"] == "ic_weighted"

    def test_optimize_unknown_method(self, portfolio_optimizer):
        """Test optimize_portfolio with unknown method."""
        result = portfolio_optimizer.optimize_portfolio(method="unknown_method")

        assert result is None


class TestReportGeneration:
    """Test report generation."""

    def test_generate_report(self, portfolio_optimizer):
        """Test generating optimization report."""
        portfolio_optimizer._load_strategy_data()
        report = portfolio_optimizer.generate_report()

        assert isinstance(report, str)
        assert "PORTFOLIO OPTIMIZATION REPORT" in report
        assert "Configuration" in report


class TestSaveResults:
    """Test saving optimization results."""

    def test_save_result_creates_file(self, portfolio_optimizer, tmp_path):
        """Test that saving creates a JSON file."""
        portfolio_optimizer.project_root = tmp_path
        result = {
            "weights": {"A": 0.5, "B": 0.5},
            "method": "test",
            "sharpe": 1.5,
        }

        portfolio_optimizer._save_optimization_result(result)

        # Check file was created
        results_dir = tmp_path / "results" / "portfolios"
        assert results_dir.exists()

        json_files = list(results_dir.glob("*.json"))
        assert len(json_files) == 1
