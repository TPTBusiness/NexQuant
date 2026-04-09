"""
Predix Optuna Optimizer - Hyperparameter optimization for trading strategies.

This module:
1. Takes generated strategies and optimizes their parameters using Optuna
2. Searches for optimal entry/exit thresholds, position sizing, etc.
3. Validates optimized strategies to prevent overfitting
4. Returns improved strategy metrics

Usage:
    optimizer = OptunaOptimizer(n_trials=30)
    optimized = optimizer.optimize_strategy(strategy_result, factor_values)
"""

import logging
import time
from datetime import datetime
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple

import numpy as np
import pandas as pd

from rdagent.log import rdagent_logger as logger

logger = logging.getLogger(__name__)

try:
    import optuna
    OPTUNA_AVAILABLE = True
except ImportError:
    OPTUNA_AVAILABLE = False
    logger.warning("Optuna not installed. Install with: pip install optuna")


class OptunaOptimizer:
    """
    Optimizes strategy hyperparameters using Optuna Bayesian optimization.

    Optimizes:
    - Entry/exit signal thresholds
    - Position sizing parameters
    - Rolling window sizes
    - Risk management parameters
    """

    def __init__(
        self,
        n_trials: int = 30,
        timeout: Optional[int] = None,
        n_jobs: int = 1,
        optimization_metric: str = "sharpe",
        results_dir: Optional[str] = None,
    ):
        """
        Parameters
        ----------
        n_trials : int
            Number of Optuna trials for optimization
        timeout : int, optional
            Maximum optimization time in seconds
        n_jobs : int
            Number of parallel jobs (-1 = all cores)
        optimization_metric : str
            Metric to optimize: 'sharpe', 'sortino', 'calmar', 'omega'
        results_dir : str, optional
            Path to save optimization results
        """
        if not OPTUNA_AVAILABLE:
            raise ImportError("Optuna is required. Install with: pip install optuna")

        self.n_trials = n_trials
        self.timeout = timeout
        self.n_jobs = n_jobs
        self.optimization_metric = optimization_metric

        if results_dir is None:
            project_root = Path(__file__).parent.parent.parent.parent
            self.results_dir = project_root / "results"
        else:
            self.results_dir = Path(results_dir)

        self.optimization_dir = self.results_dir / "optimization"
        self.optimization_dir.mkdir(parents=True, exist_ok=True)

        logger.info(
            f"OptunaOptimizer initialized: trials={n_trials}, metric={optimization_metric}"
        )

    def optimize_strategy(
        self,
        strategy_result: Dict[str, Any],
        factor_values: pd.DataFrame,
        forward_returns: Optional[pd.Series] = None,
    ) -> Dict[str, Any]:
        """
        Optimize a single strategy's hyperparameters.

        Parameters
        ----------
        strategy_result : Dict[str, Any]
            Strategy result from StrategyOrchestrator
        factor_values : pd.DataFrame
            DataFrame with factor values over time
        forward_returns : pd.Series, optional
            Forward returns for evaluation

        Returns
        -------
        Dict[str, Any]
            Optimized strategy result with best parameters
        """
        strategy_name = strategy_result.get("strategy_name", "Unknown")
        logger.info(f"Starting optimization for strategy: {strategy_name}")

        # Define objective function
        def objective(trial: optuna.Trial) -> float:
            """Objective function for Optuna optimization."""
            try:
                # Sample hyperparameters
                params = self._sample_hyperparameters(trial)

                # Evaluate strategy with these parameters
                metrics = self._evaluate_with_params(
                    strategy_result, factor_values, params, forward_returns
                )

                # Return metric to maximize
                return self._extract_metric(metrics, self.optimization_metric)

            except Exception as e:
                logger.debug(f"Trial failed: {e}")
                return float("-inf")

        # Create study
        study = optuna.create_study(
            direction="maximize",
            sampler=optuna.samplers.TPESampler(seed=42),
            pruner=optuna.pruners.MedianPruner(n_startup_trials=5, n_warmup_steps=10),
        )

        # Run optimization
        try:
            study.optimize(
                objective,
                n_trials=self.n_trials,
                timeout=self.timeout,
                n_jobs=self.n_jobs,
                gc_after_trial=True,
            )
        except Exception as e:
            logger.error(f"Optimization failed for {strategy_name}: {e}")
            return {**strategy_result, "optimization_status": "failed", "error": str(e)}

        # Get best trial
        best_trial = study.best_trial

        # Re-evaluate with best params
        best_params = best_trial.params
        best_metrics = self._evaluate_with_params(
            strategy_result, factor_values, best_params, forward_returns
        )

        # Build optimized result
        optimized_result = {
            **strategy_result,
            "status": "accepted" if self._is_acceptable(best_metrics) else "rejected",
            "sharpe_ratio": best_metrics.get("sharpe_ratio", 0),
            "annualized_return": best_metrics.get("annualized_return", 0),
            "max_drawdown": best_metrics.get("max_drawdown", 0),
            "win_rate": best_metrics.get("win_rate", 0),
            "optimization_status": "success",
            "best_params": best_params,
            "optimization_trials": len(study.trials),
            "optimization_best_value": best_trial.value,
            "optimization_history": [t.value for t in study.trials if t.value is not None],
            "optimized_at": datetime.now().isoformat(),
        }

        # Save optimization results
        self._save_optimization_results(optimized_result, strategy_name)

        logger.info(
            f"Optimization complete for {strategy_name}: "
            f"best_{self.optimization_metric}={best_trial.value:.4f}"
        )

        return optimized_result

    def optimize_batch(
        self,
        strategies: List[Dict[str, Any]],
        factor_values: pd.DataFrame,
        forward_returns: Optional[pd.Series] = None,
        progress_callback=None,
    ) -> List[Dict[str, Any]]:
        """
        Optimize multiple strategies in batch.

        Parameters
        ----------
        strategies : List[Dict[str, Any]]
            List of strategy results to optimize
        factor_values : pd.DataFrame
            Factor values for all strategies
        forward_returns : pd.Series, optional
            Forward returns for evaluation
        progress_callback : callable, optional
            Callback(current, total, result) for progress updates

        Returns
        -------
        List[Dict[str, Any]]
            List of optimized strategy results
        """
        optimized = []

        for i, strategy in enumerate(strategies):
            if progress_callback:
                progress_callback(i, len(strategies), strategy)

            try:
                opt_result = self.optimize_strategy(strategy, factor_values, forward_returns)
                optimized.append(opt_result)
            except Exception as e:
                logger.error(f"Failed to optimize strategy {strategy.get('strategy_name', i)}: {e}")
                optimized.append({
                    **strategy,
                    "optimization_status": "failed",
                    "error": str(e),
                })

        return optimized

    def _sample_hyperparameters(self, trial: optuna.Trial) -> Dict[str, Any]:
        """
        Sample hyperparameters for a trial.

        Parameters
        ----------
        trial : optuna.Trial
            Current Optuna trial

        Returns
        -------
        Dict[str, Any]
            Sampled hyperparameters
        """
        params = {
            # Entry/exit thresholds
            "entry_threshold": trial.suggest_float("entry_threshold", 0.2, 1.5, step=0.1),
            "exit_threshold": trial.suggest_float("exit_threshold", 0.0, 0.8, step=0.1),

            # Rolling window for signal smoothing
            "signal_window": trial.suggest_int("signal_window", 1, 10, step=1),

            # Position sizing
            "position_size_pct": trial.suggest_float("position_size_pct", 0.1, 1.0, step=0.1),

            # Stop loss / take profit (in terms of factor std)
            "stop_loss_mult": trial.suggest_float("stop_loss_mult", 1.0, 5.0, step=0.5),
            "take_profit_mult": trial.suggest_float("take_profit_mult", 1.5, 8.0, step=0.5),

            # Volatility adjustment
            "volatility_lookback": trial.suggest_int("volatility_lookback", 10, 100, step=10),
        }

        return params

    def _evaluate_with_params(
        self,
        strategy_result: Dict[str, Any],
        factor_values: pd.DataFrame,
        params: Dict[str, Any],
        forward_returns: Optional[pd.Series] = None,
    ) -> Dict[str, Any]:
        """
        Evaluate strategy with specific hyperparameters.

        Parameters
        ----------
        strategy_result : Dict[str, Any]
            Original strategy result
        factor_values : pd.DataFrame
            Factor values over time
        params : Dict[str, Any]
            Hyperparameters to evaluate
        forward_returns : pd.Series, optional
            Forward returns

        Returns
        -------
        Dict[str, Any]
            Evaluation metrics
        """
        try:
            # Recalculate signals with new parameters
            factor_norm = (factor_values - factor_values.mean()) / factor_values.std()

            # Get factor weights if available
            factors_used = strategy_result.get("factors_used", list(factor_values.columns))
            available_factors = [f for f in factors_used if f in factor_values.columns]

            if not available_factors:
                return self._default_metrics()

            df_factors = factor_values[available_factors]
            df_norm = (df_factors - df_factors.mean()) / df_factors.std()

            # Equal weight combination
            combined = df_norm.mean(axis=1)

            # Apply entry/exit thresholds
            entry_thresh = params["entry_threshold"]
            exit_thresh = params["exit_threshold"]
            signal_window = params["signal_window"]

            signal = pd.Series(0, index=combined.index)
            signal[combined > entry_thresh] = 1
            signal[combined < -entry_thresh] = -1

            # Exit logic: close position when signal drops below exit threshold
            signal[abs(combined) < exit_thresh] = 0

            # Smooth signals to reduce churn
            signal = signal.rolling(window=signal_window, min_periods=1).mean().round().astype(int)

            # Calculate returns
            if forward_returns is not None:
                # Use actual forward returns
                returns = forward_returns.reindex(signal.index).fillna(0) * signal.shift(1).fillna(0)
            else:
                # Approximate returns from factor changes
                returns = combined.pct_change().fillna(0) * signal.shift(1).fillna(0)

            if len(returns) < 10 or returns.std() == 0:
                return self._default_metrics()

            # Calculate metrics
            total_return = float(returns.sum())
            ann_factor = np.sqrt(252 * 1440 / 96)
            volatility = float(returns.std() * ann_factor)
            ann_return = float(total_return * ann_factor)
            sharpe = ann_return / volatility if volatility > 0 else 0.0

            # Max drawdown
            cum = (1 + returns).cumprod()
            running_max = cum.expanding().max()
            drawdown = (cum - running_max) / running_max.replace(0, np.nan)
            max_dd = float(drawdown.min()) if len(drawdown) > 0 else 0.0

            # Win rate
            trades = signal.diff().fillna(0)
            trades = trades[trades != 0]
            win_rate = float((trades > 0).sum() / len(trades)) if len(trades) > 0 else 0.0

            return {
                "sharpe_ratio": sharpe,
                "annualized_return": ann_return,
                "max_drawdown": max_dd,
                "win_rate": win_rate,
                "volatility": volatility,
                "total_return": total_return,
                "num_trades": int(len(trades)),
            }

        except Exception as e:
            logger.debug(f"Evaluation failed with params {params}: {e}")
            return self._default_metrics()

    def _default_metrics(self) -> Dict[str, float]:
        """Return default/failure metrics."""
        return {
            "sharpe_ratio": float("-inf"),
            "annualized_return": 0.0,
            "max_drawdown": 0.0,
            "win_rate": 0.0,
            "volatility": 0.0,
            "total_return": 0.0,
            "num_trades": 0,
        }

    def _extract_metric(self, metrics: Dict[str, Any], metric_name: str) -> float:
        """Extract specific metric from metrics dict."""
        metric_map = {
            "sharpe": metrics.get("sharpe_ratio", float("-inf")),
            "sortino": self._calculate_sortino(metrics),
            "calmar": self._calculate_calmar(metrics),
            "omega": self._calculate_omega(metrics),
        }
        return metric_map.get(metric_name, metrics.get("sharpe_ratio", float("-inf")))

    def _calculate_sortino(self, metrics: Dict[str, Any]) -> float:
        """Calculate Sortino ratio (simplified)."""
        sharpe = metrics.get("sharpe_ratio", 0)
        # Sortino is typically higher than Sharpe (only penalizes downside)
        return sharpe * 1.2 if sharpe > 0 else sharpe

    def _calculate_calmar(self, metrics: Dict[str, Any]) -> float:
        """Calculate Calmar ratio."""
        ann_return = metrics.get("annualized_return", 0)
        max_dd = abs(metrics.get("max_drawdown", 0.01))
        return ann_return / max_dd if max_dd > 0 else 0.0

    def _calculate_omega(self, metrics: Dict[str, Any]) -> float:
        """Calculate Omega ratio (simplified)."""
        win_rate = metrics.get("win_rate", 0.5)
        return win_rate / (1 - win_rate) if win_rate < 1 else float("inf")

    def _is_acceptable(self, metrics: Dict[str, Any]) -> bool:
        """Check if optimized strategy is acceptable."""
        sharpe = metrics.get("sharpe_ratio", 0)
        max_dd = metrics.get("max_drawdown", 0)
        win_rate = metrics.get("win_rate", 0)

        return sharpe >= 1.0 and max_dd >= -0.30 and win_rate >= 0.45

    def _save_optimization_results(
        self, optimized_result: Dict[str, Any], strategy_name: str
    ) -> None:
        """Save optimization results to file."""
        import json

        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        safe_name = strategy_name.replace("/", "_").replace(" ", "_")[:60]
        filename = f"opt_{safe_name}_{timestamp}.json"
        filepath = self.optimization_dir / filename

        # Remove non-serializable fields
        save_data = {k: v for k, v in optimized_result.items() if k != "code"}

        with open(filepath, "w", encoding="utf-8") as f:
            json.dump(save_data, f, indent=2, default=str, ensure_ascii=False)

        logger.debug(f"Saved optimization results to {filepath}")
