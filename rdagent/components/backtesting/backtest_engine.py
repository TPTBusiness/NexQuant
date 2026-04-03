"""
Predix Backtesting Engine - IC, Sharpe, Drawdown

Supports both factor-based backtesting and RL agent backtesting.
"""
import numpy as np
import pandas as pd
from pathlib import Path
from typing import Dict, Optional, Any, List
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

    def run_rl_backtest(
        self,
        rl_agent: Any,
        prices: pd.Series,
        indicators: Optional[pd.DataFrame] = None,
        initial_balance: float = 100000.0,
        transaction_cost: float = 0.00015,
        window_size: int = 60,
        enable_protections: bool = True,
    ) -> Dict:
        """
        Run backtest with RL agent.

        Parameters
        ----------
        rl_agent : Any
            Trained RL agent (RLTradingAgent or model with predict method)
        prices : pd.Series
            Price time series for backtesting
        indicators : pd.DataFrame, optional
            Technical indicators DataFrame
        initial_balance : float
            Starting balance
        transaction_cost : float
            Transaction cost per trade
        window_size : int
            Lookback window for observations
        enable_protections : bool
            Enable trading protections

        Returns
        -------
        dict
            Backtest metrics
        """
        from rdagent.components.coder.rl import RLCosteer

        # Create costeer with protections
        costeer = RLCosteer(
            model_path=None,
            algorithm=getattr(rl_agent, 'algorithm', 'PPO'),
            window_size=window_size,
            enable_protections=enable_protections,
        )

        # Attach trained model directly
        if hasattr(rl_agent, 'model'):
            costeer.model = rl_agent.model
            costeer.is_active = True
        elif hasattr(rl_agent, 'predict'):
            # Agent has predict method directly
            costeer.model = rl_agent
            costeer.is_active = True
        else:
            raise ValueError("RL agent must have 'model' or 'predict' attribute")

        # Initialize with price data
        costeer.initialize(
            prices=prices,
            indicators=indicators,
            initial_equity=initial_balance,
        )

        # Run simulation
        equity_curve: List[float] = [initial_balance]
        position = 0.0
        cash = initial_balance
        returns_history: List[float] = []

        price_values = prices.values if isinstance(prices, pd.Series) else np.array(prices)

        for step in range(len(price_values) - 1):
            # Ensure costeer doesn't go beyond available data
            if costeer.current_step >= len(price_values):
                break

            current_price = float(price_values[step])
            current_equity = cash + position * current_price

            # Get RL action with protections
            trade_info = costeer.step(
                current_equity=current_equity,
                cash=cash,
                position=position,
                returns_history=returns_history[-100:] if returns_history else None,  # Last 100 returns
            )

            # Execute trade (simplified)
            target_position = trade_info["target_position"]
            position_change = target_position - position

            # Calculate transaction cost
            trade_value = abs(position_change) * current_price
            cost = trade_value * transaction_cost

            # Update position and cash
            position = target_position
            cash -= cost

            # Calculate return for this step
            if step > 0:
                prev_price = float(price_values[step - 1]) if step > 0 else current_price
                if prev_price > 0:
                    step_return = (current_price - prev_price) / prev_price * position
                    returns_history.append(step_return)

            # Calculate new equity
            new_equity = cash + position * current_price
            equity_curve.append(new_equity)

        # Calculate metrics
        equity_series = pd.Series(equity_curve)
        returns_series = equity_series.pct_change().dropna()

        metrics = self.metrics.calculate_all(returns_series, equity_series)
        metrics["factor_name"] = f"RL_{getattr(rl_agent, 'algorithm', 'Unknown')}"
        metrics["timestamp"] = datetime.now().isoformat()
        metrics["initial_balance"] = initial_balance
        metrics["final_equity"] = equity_curve[-1]
        metrics["total_steps"] = len(price_values) - 1

        # Save results
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        rl_name = f"RL_{getattr(rl_agent, 'algorithm', 'Unknown')}"

        with open(self.results_path / f"{rl_name}_{timestamp}.json", 'w') as f:
            json.dump(
                {k: (None if isinstance(v, float) and np.isnan(v) else v) for k, v in metrics.items()},
                f, indent=2
            )

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
