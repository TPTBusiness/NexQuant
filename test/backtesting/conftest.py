"""
Predix Backtesting Test Fixtures
Wiederverwendbare Test-Daten und Fixtures für alle Backtesting-Tests
"""
import pytest
import numpy as np
import pandas as pd
import tempfile
import os
from pathlib import Path
from datetime import datetime, timedelta

# Importiere die zu testenden Klassen
import sys
sys.path.insert(0, str(Path(__file__).parent.parent.parent))
from rdagent.components.backtesting.backtest_engine import BacktestMetrics, FactorBacktester
from rdagent.components.backtesting.results_db import ResultsDatabase
from rdagent.components.backtesting.risk_management import (
    CorrelationAnalyzer, PortfolioOptimizer, AdvancedRiskManager
)


# =============================================================================
# FIXTURES FÜR BACKTEST METRICS
# =============================================================================

@pytest.fixture
def sample_factor_data():
    """Normale Faktor-Daten für Standard-Tests"""
    np.random.seed(42)
    n = 252
    dates = pd.date_range(start='2024-01-01', periods=n, freq='B')
    factor_values = pd.Series(np.random.randn(n), index=dates, name='factor')
    forward_returns = pd.Series(np.random.randn(n) * 0.01 + 0.0001, index=dates, name='fwd_ret')
    return factor_values, forward_returns


@pytest.fixture
def sample_returns_data():
    """Returns-Daten für Sharpe und Drawdown Tests"""
    np.random.seed(42)
    n = 252
    dates = pd.date_range(start='2024-01-01', periods=n, freq='B')
    returns = pd.Series(np.random.randn(n) * 0.01 + 0.0005, index=dates)
    equity = (1 + returns).cumprod()
    return returns, equity


@pytest.fixture
def backtest_metrics():
    """BacktestMetrics Instanz mit Standard-Parametern"""
    return BacktestMetrics(risk_free_rate=0.02)


# =============================================================================
# FIXTURES FÜR EDGE CASES
# =============================================================================

@pytest.fixture
def empty_data():
    """Leere Daten für Edge-Case Tests"""
    return pd.Series([], dtype=float), pd.Series([], dtype=float)


@pytest.fixture
def nan_data():
    """Daten mit vielen NaN-Werten"""
    np.random.seed(42)
    n = 100
    dates = pd.date_range(start='2024-01-01', periods=n, freq='B')
    factor = pd.Series([np.nan] * 50 + list(np.random.randn(50)), index=dates)
    fwd_ret = pd.Series(list(np.random.randn(50)) + [np.nan] * 50, index=dates)
    return factor, fwd_ret


@pytest.fixture
def insufficient_data():
    """Zu wenig Daten (< 10 Punkte)"""
    np.random.seed(42)
    n = 5
    dates = pd.date_range(start='2024-01-01', periods=n, freq='B')
    factor = pd.Series(np.random.randn(n), index=dates)
    fwd_ret = pd.Series(np.random.randn(n), index=dates)
    return factor, fwd_ret


@pytest.fixture
def extreme_values_data():
    """Daten mit Extremwerten"""
    np.random.seed(42)
    n = 252
    dates = pd.date_range(start='2024-01-01', periods=n, freq='B')
    factor = pd.Series(np.random.randn(n), index=dates)
    factor.iloc[50] = 1000  # Extremwert
    factor.iloc[100] = -1000  # Extremwert negativ
    fwd_ret = pd.Series(np.random.randn(n) * 0.01, index=dates)
    return factor, fwd_ret


@pytest.fixture
def constant_data():
    """Konstante Daten (Std = 0)"""
    n = 252
    dates = pd.date_range(start='2024-01-01', periods=n, freq='B')
    factor = pd.Series([1.0] * n, index=dates)
    fwd_ret = pd.Series([0.001] * n, index=dates)
    return factor, fwd_ret


# =============================================================================
# FIXTURES FÜR DATABASE TESTS
# =============================================================================

@pytest.fixture
def temp_db_path():
    """Temporäre Datenbank für Tests"""
    with tempfile.TemporaryDirectory() as tmpdir:
        db_path = os.path.join(tmpdir, 'test_backtest.db')
        yield db_path


@pytest.fixture
def results_database(temp_db_path):
    """ResultsDatabase Instanz mit temporärer DB"""
    db = ResultsDatabase(db_path=temp_db_path)
    yield db
    db.close()


@pytest.fixture
def populated_database(results_database):
    """Datenbank mit Test-Daten befüllt"""
    db = results_database
    
    # Faktoren hinzufügen
    db.add_factor("Momentum", "price_based")
    db.add_factor("MeanReversion", "price_based")
    db.add_factor("Volatility", "risk_based")
    db.add_factor("Volume", "volume_based")
    db.add_factor("ML_Factor", "ml_based")
    
    # Backtest-Ergebnisse hinzufügen
    db.add_backtest("Momentum", {
        'ic': 0.08, 'sharpe_ratio': 1.5, 'annualized_return': 0.12,
        'max_drawdown': -0.08, 'win_rate': 0.55
    })
    db.add_backtest("MeanReversion", {
        'ic': 0.05, 'sharpe_ratio': 1.2, 'annualized_return': 0.08,
        'max_drawdown': -0.05, 'win_rate': 0.52
    })
    db.add_backtest("Volatility", {
        'ic': -0.03, 'sharpe_ratio': 0.8, 'annualized_return': 0.04,
        'max_drawdown': -0.03, 'win_rate': 0.48
    })
    db.add_backtest("ML_Factor", {
        'ic': 0.12, 'sharpe_ratio': 2.1, 'annualized_return': 0.18,
        'max_drawdown': -0.10, 'win_rate': 0.60
    })
    
    # Loop-Ergebnisse hinzufügen
    db.add_loop(1, 4, 6, 0.08, "completed")
    db.add_loop(2, 5, 5, 0.10, "completed")
    db.add_loop(3, 3, 7, 0.05, "completed")
    
    return db


# =============================================================================
# FIXTURES FÜR RISK MANAGEMENT TESTS
# =============================================================================

@pytest.fixture
def sample_returns_matrix():
    """Returns-Matrix für Korrelations-Analyse"""
    np.random.seed(42)
    n = 252
    dates = pd.date_range(start='2024-01-01', periods=n, freq='B')
    columns = ['Mom', 'MeanRev', 'Vol', 'Volu', 'ML']
    
    # Erzeuge korrelierte Returns
    data = np.random.randn(n, 5)
    data[:, 0] = data[:, 1] * 0.3 + data[:, 0] * 0.7  # Mom korreliert mit MeanRev
    data[:, 3] = data[:, 2] * 0.5 + data[:, 3] * 0.5  # Volu korreliert mit Vol
    
    return pd.DataFrame(data, columns=columns, index=dates)


@pytest.fixture
def correlation_analyzer():
    """CorrelationAnalyzer Instanz"""
    return CorrelationAnalyzer(lookback=60)


@pytest.fixture
def portfolio_optimizer():
    """PortfolioOptimizer Instanz"""
    return PortfolioOptimizer()


@pytest.fixture
def sample_expected_returns():
    """Erwartete Returns für Portfolio-Optimierung"""
    return pd.Series({
        'Mom': 0.10, 'MeanRev': 0.08, 'Vol': 0.06,
        'Volu': 0.07, 'ML': 0.12
    })


@pytest.fixture
def sample_covariance_matrix(sample_returns_matrix):
    """Kovarianz-Matrix aus Returns"""
    return sample_returns_matrix.cov() * 252


@pytest.fixture
def risk_manager():
    """AdvancedRiskManager Instanz"""
    return AdvancedRiskManager(max_pos=0.2, max_lev=5.0, max_dd=0.20)


@pytest.fixture
def sample_weights():
    """Test-Gewichtungen"""
    return np.array([0.25, 0.20, 0.15, 0.20, 0.20])


# =============================================================================
# FIXTURES FÜR BACKTESTER
# =============================================================================

@pytest.fixture
def factor_backtester():
    """FactorBacktester Instanz mit temporärem Output-Verzeichnis"""
    with tempfile.TemporaryDirectory() as tmpdir:
        backtester = FactorBacktester()
        backtester.results_path = Path(tmpdir)
        yield backtester


# =============================================================================
# ZUSÄTZLICHE HILFS-FIXTURES
# =============================================================================

@pytest.fixture
def realistic_market_data():
    """Realistischere Markt-Daten mit typischen Eigenschaften"""
    np.random.seed(42)
    n = 504  # 2 Jahre
    dates = pd.date_range(start='2023-01-01', periods=n, freq='B')
    
    # Faktor mit etwas Autokorrelation (wie echte Faktoren)
    factor = pd.Series(index=dates)
    factor.iloc[0] = 0
    for i in range(1, n):
        factor.iloc[i] = 0.3 * factor.iloc[i-1] + np.random.randn() * 0.7
    
    # Forward Returns mit leichtem positiven Drift
    fwd_ret = pd.Series(np.random.randn(n) * 0.015 + 0.0002, index=dates)
    
    # Füge einige Ausreißer hinzu (wie bei echten Marktdaten)
    fwd_ret.iloc[50] = -0.05  # Crash-Tag
    fwd_ret.iloc[150] = 0.04  # Rally-Tag
    
    return factor, fwd_ret


@pytest.fixture
def zero_variance_returns():
    """Returns mit Varianz = 0 (für Edge-Case Tests)"""
    n = 100
    dates = pd.date_range(start='2024-01-01', periods=n, freq='B')
    returns = pd.Series([0.001] * n, index=dates)
    equity = (1 + returns).cumprod()
    return returns, equity


@pytest.fixture
def negative_equity_data():
    """Equity-Daten mit Drawdowns"""
    np.random.seed(42)
    n = 252
    dates = pd.date_range(start='2024-01-01', periods=n, freq='B')
    
    # Erzeuge Equity mit signifikantem Drawdown
    returns = pd.Series(np.random.randn(n) * 0.02, index=dates)
    returns.iloc[50:80] = -0.03  # Drawdown-Periode
    equity = (1 + returns).cumprod()
    
    return returns, equity
