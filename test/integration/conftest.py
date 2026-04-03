"""
Shared fixtures for Predix integration tests.
Provides common test data, mock objects, and utilities.
"""
import pytest
import tempfile
import os
import sys
import numpy as np
import pandas as pd
from pathlib import Path
from datetime import datetime, timedelta
from unittest.mock import MagicMock, patch

# Project root
PROJECT_ROOT = Path(__file__).parent.parent.parent
sys.path.insert(0, str(PROJECT_ROOT))


# =============================================================================
# MOCK DATA FIXTURES
# =============================================================================


@pytest.fixture
def mock_factor_data():
    """Generate mock factor time series data."""
    np.random.seed(42)
    dates = pd.date_range("2024-01-01", periods=252, freq="B")
    factor_values = pd.Series(np.random.randn(252), index=dates, name="test_factor")
    forward_returns = pd.Series(np.random.randn(252) * 0.01, index=dates, name="returns")
    return factor_values, forward_returns


@pytest.fixture
def mock_portfolio_returns():
    """Generate mock portfolio return data."""
    np.random.seed(42)
    n_assets = 5
    n_days = 252
    dates = pd.date_range("2024-01-01", periods=n_days, freq="B")
    returns = pd.DataFrame(
        np.random.randn(n_days, n_assets) * 0.01,
        index=dates,
        columns=[f"asset_{i}" for i in range(n_assets)]
    )
    return returns


@pytest.fixture
def mock_expected_returns():
    """Generate mock expected returns."""
    return pd.Series({
        "asset_0": 0.10, "asset_1": 0.08, "asset_2": 0.06,
        "asset_3": 0.07, "asset_4": 0.12
    })


@pytest.fixture
def mock_covariance_matrix(mock_portfolio_returns):
    """Generate mock covariance matrix."""
    return mock_portfolio_returns.cov() * 252


@pytest.fixture
def mock_backtest_metrics():
    """Generate mock backtest metrics dictionary."""
    return {
        "ic": 0.08,
        "sharpe_ratio": 1.5,
        "annualized_return": 0.12,
        "max_drawdown": -0.08,
        "win_rate": 0.55,
        "total_trades": 252,
        "total_return": 0.15,
        "factor_name": "TestFactor",
        "timestamp": datetime.now().isoformat(),
    }


# =============================================================================
# TEMPORARY RESOURCE FIXTURES
# =============================================================================


@pytest.fixture
def temp_database_path():
    """Create a temporary database path for testing."""
    with tempfile.TemporaryDirectory() as tmpdir:
        db_path = os.path.join(tmpdir, "test.db")
        yield db_path


@pytest.fixture
def temp_output_dir():
    """Create a temporary output directory for testing."""
    with tempfile.TemporaryDirectory() as tmpdir:
        yield Path(tmpdir)


@pytest.fixture
def temp_env_file():
    """Create a temporary .env file for testing."""
    with tempfile.TemporaryDirectory() as tmpdir:
        env_path = os.path.join(tmpdir, ".env")
        with open(env_path, "w") as f:
            f.write("OPENAI_API_KEY=local\n")
            f.write("OPENAI_API_BASE=http://localhost:8081/v1\n")
            f.write("CHAT_MODEL=qwen3.5-35b\n")
            f.write("EMBEDD_MODEL=nomic-embed-text\n")
            f.write("LITELLM_PROXY_API_BASE=http://localhost:11434/v1\n")
        yield env_path


# =============================================================================
# COMPONENT FIXTURES
# =============================================================================


@pytest.fixture
def backtest_metrics_instance():
    """BacktestMetrics instance for testing."""
    from rdagent.components.backtesting.backtest_engine import BacktestMetrics
    return BacktestMetrics(risk_free_rate=0.02)


@pytest.fixture
def factor_backtester_instance(temp_output_dir):
    """FactorBacktester instance with temporary output directory."""
    from rdagent.components.backtesting.backtest_engine import FactorBacktester
    backtester = FactorBacktester()
    backtester.results_path = temp_output_dir
    return backtester


@pytest.fixture
def results_database_instance(temp_database_path):
    """ResultsDatabase instance with temporary database."""
    from rdagent.components.backtesting.results_db import ResultsDatabase
    db = ResultsDatabase(db_path=temp_database_path)
    yield db
    db.close()


@pytest.fixture
def populated_database_instance(results_database_instance):
    """ResultsDatabase pre-populated with test data."""
    db = results_database_instance

    # Add factors
    db.add_factor("Momentum", "price_based")
    db.add_factor("MeanReversion", "price_based")
    db.add_factor("Volatility", "risk_based")
    db.add_factor("ML_Factor", "ml_based")

    # Add backtest results
    db.add_backtest("Momentum", {
        "ic": 0.08, "sharpe_ratio": 1.5, "annualized_return": 0.12,
        "max_drawdown": -0.08, "win_rate": 0.55
    })
    db.add_backtest("MeanReversion", {
        "ic": 0.05, "sharpe_ratio": 1.2, "annualized_return": 0.08,
        "max_drawdown": -0.05, "win_rate": 0.52
    })
    db.add_backtest("Volatility", {
        "ic": -0.03, "sharpe_ratio": 0.8, "annualized_return": 0.04,
        "max_drawdown": -0.03, "win_rate": 0.48
    })
    db.add_backtest("ML_Factor", {
        "ic": 0.12, "sharpe_ratio": 2.1, "annualized_return": 0.18,
        "max_drawdown": -0.10, "win_rate": 0.60
    })

    # Add loop results
    db.add_loop(1, 4, 6, 0.08, "completed")
    db.add_loop(2, 5, 5, 0.10, "completed")

    return db


@pytest.fixture
def correlation_analyzer_instance():
    """CorrelationAnalyzer instance for testing."""
    from rdagent.components.backtesting.risk_management import CorrelationAnalyzer
    return CorrelationAnalyzer(lookback=60)


@pytest.fixture
def portfolio_optimizer_instance():
    """PortfolioOptimizer instance for testing."""
    from rdagent.components.backtesting.risk_management import PortfolioOptimizer
    return PortfolioOptimizer()


@pytest.fixture
def risk_manager_instance():
    """AdvancedRiskManager instance for testing."""
    from rdagent.components.backtesting.risk_management import AdvancedRiskManager
    return AdvancedRiskManager(max_pos=0.2, max_lev=5.0, max_dd=0.20)
