"""NexQuant Backtesting Package"""
from .backtest_engine import BacktestMetrics, FactorBacktester
from .results_db import ResultsDatabase
from .risk_management import CorrelationAnalyzer, PortfolioOptimizer, AdvancedRiskManager
from .vbt_backtest import (
    DEFAULT_BARS_PER_YEAR,
    DEFAULT_TXN_COST_BPS,
    INITIAL_CAPITAL,
    MAX_DAILY_LOSS,
    MAX_TOTAL_LOSS,
    MAX_LEVERAGE,
    RISK_PER_TRADE,
    OOS_START_DEFAULT,
    WF_IS_YEARS,
    WF_OOS_YEARS,
    WF_STEP_YEARS,
    backtest_from_forward_returns,
    backtest_signal,
    backtest_signal_risk,
    monte_carlo_trade_pvalue,
    walk_forward_rolling,
)

__all__ = [
    'BacktestMetrics', 'FactorBacktester', 'ResultsDatabase',
    'CorrelationAnalyzer', 'PortfolioOptimizer', 'AdvancedRiskManager',
    'backtest_signal', 'backtest_signal_risk', 'backtest_from_forward_returns',
    'monte_carlo_trade_pvalue', 'walk_forward_rolling',
    'DEFAULT_BARS_PER_YEAR', 'DEFAULT_TXN_COST_BPS',
    'INITIAL_CAPITAL', 'MAX_DAILY_LOSS', 'MAX_TOTAL_LOSS',
    'MAX_LEVERAGE', 'RISK_PER_TRADE', 'OOS_START_DEFAULT',
    'WF_IS_YEARS', 'WF_OOS_YEARS', 'WF_STEP_YEARS',
]
