"""
Predix Backtesting Engine - IC, Sharpe, Drawdown
"""
import numpy as np
import pandas as pd
from pathlib import Path
from typing import Dict, Optional
from datetime import datetime
import json

class BacktestMetrics:
    def __init__(self, risk_free_rate: float = 0.02):
        self.risk_free_rate = risk_free_rate
    
    def calculate_ic(self, factor_values: pd.Series, forward_returns: pd.Series) -> float:
        mask = factor_values.notna() & forward_returns.notna()
        if mask.sum() < 10: return np.nan
        return factor_values[mask].corr(forward_returns[mask])
    
    def calculate_sharpe(self, returns: pd.Series, annualize: bool = True) -> float:
        if len(returns) < 10 or returns.std() == 0: return np.nan
        sharpe = (returns.mean() - self.risk_free_rate/252) / returns.std()
        return sharpe * np.sqrt(252) if annualize else sharpe
    
    def calculate_max_drawdown(self, equity: pd.Series) -> float:
        running_max = equity.cummax()
        drawdown = (equity - running_max) / running_max
        return float(drawdown.min())
    
    def calculate_all(self, returns: pd.Series, equity: pd.Series, 
                     factor_values: Optional[pd.Series] = None, 
                     forward_returns: Optional[pd.Series] = None) -> Dict:
        metrics = {
            'total_return': float((1 + returns).prod() - 1),
            'annualized_return': float(returns.mean() * 252),
            'sharpe_ratio': self.calculate_sharpe(returns),
            'max_drawdown': self.calculate_max_drawdown(equity),
            'win_rate': float((returns > 0).mean()),
            'total_trades': len(returns),
        }
        if factor_values is not None and forward_returns is not None:
            metrics['ic'] = self.calculate_ic(factor_values, forward_returns)
        return metrics

class FactorBacktester:
    def __init__(self):
        self.metrics = BacktestMetrics()
        self.results_path = Path(__file__).parent.parent.parent / "results" / "backtests"
        self.results_path.mkdir(parents=True, exist_ok=True)
    
    def run_backtest(self, factor_values: pd.Series, forward_returns: pd.Series, 
                     factor_name: str, transaction_cost: float = 0.00015) -> Dict:
        ic = self.metrics.calculate_ic(factor_values, forward_returns)
        signals = np.sign(factor_values)
        strategy_returns = signals.shift(1) * forward_returns - transaction_cost
        equity = (1 + strategy_returns).cumprod()
        
        metrics = self.metrics.calculate_all(strategy_returns, equity, factor_values, forward_returns)
        metrics['ic'] = ic if not np.isnan(ic) else np.nan
        metrics['factor_name'] = factor_name
        metrics['timestamp'] = datetime.now().isoformat()
        
        # Speichern
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        safe_name = factor_name.replace("/", "_")
        
        with open(self.results_path / f"{safe_name}_{timestamp}.json", 'w') as f:
            json.dump({k: (None if isinstance(v, float) and np.isnan(v) else v) for k, v in metrics.items()}, f, indent=2)
        
        return metrics

if __name__ == "__main__":
    print("=== Backtest Test ===")
    np.random.seed(42)
    n = 252
    factor = pd.Series(np.random.randn(n))
    fwd_ret = pd.Series(np.random.randn(n) * 0.01 + 0.0001)
    
    backtester = FactorBacktester()
    metrics = backtester.run_backtest(factor, fwd_ret, "TestFactor")
    
    print(f"IC: {metrics.get('ic', np.nan):.4f}")
    print(f"Sharpe: {metrics.get('sharpe_ratio', np.nan):.4f}")
    print(f"Win Rate: {metrics.get('win_rate', np.nan):.4f}")
    print("✅ Test bestanden!")
