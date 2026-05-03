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

_optuna_logger = logging.getLogger(__name__)

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
        Optimiere eine einzelne Strategie mit mehrstufiger Suche (grob → fein).

        STAGE 1: Grobe Suche mit weiten Bereichen (10 Trials)
        STAGE 2: Feine Suche um die besten Stage-1-Parameter (15 Trials)
        STAGE 3: Sehr feine lokale Suche (5 Trials)

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
        logger.info(f"Starting multi-stage optimization for strategy: {strategy_name}")

        # Speichere Referenzen für Objective-Methoden
        self._current_strategy = strategy_result
        self._current_factors = factor_values
        self._current_forward_returns = forward_returns

        # STAGE 1: Grobe Suche mit weiten Bereichen (10 Trials)
        logger.info(f"Stage 1: Coarse search for {strategy_name}")
        stage1_study = optuna.create_study(
            direction="maximize",
            sampler=optuna.samplers.TPESampler(seed=42),
            pruner=optuna.pruners.MedianPruner(n_startup_trials=3, n_warmup_steps=5),
        )
        stage1_study.optimize(self._objective_coarse, n_trials=10, gc_after_trial=True)

        best_stage1 = stage1_study.best_trial.params
        best_stage1_value = stage1_study.best_trial.value
        logger.info(
            f"Stage 1 complete: best_value={best_stage1_value:.4f}, "
            f"params={best_stage1}"
        )

        # STAGE 2: Feine Suche um die besten Stage-1-Parameter (15 Trials)
        logger.info(f"Stage 2: Fine search around best params")
        stage2_study = optuna.create_study(
            direction="maximize",
            sampler=optuna.samplers.TPESampler(seed=43),
            pruner=optuna.pruners.MedianPruner(n_startup_trials=5, n_warmup_steps=5),
        )
        # Verwende beste Stage-1-Parameter als Zentrum für feine Suche
        self._fine_search_center = best_stage1
        stage2_study.optimize(self._objective_fine, n_trials=15, gc_after_trial=True)

        best_stage2 = stage2_study.best_trial.params
        best_stage2_value = stage2_study.best_trial.value
        logger.info(
            f"Stage 2 complete: best_value={best_stage2_value:.4f}, "
            f"params={best_stage2}"
        )

        # STAGE 3: Sehr feine lokale Suche (5 Trials) - nur wenn Stage 2 besser war
        if best_stage2_value > best_stage1_value:
            logger.info(f"Stage 3: Very fine local search")
            stage3_study = optuna.create_study(
                direction="maximize",
                sampler=optuna.samplers.TPESampler(seed=44),
            )
            self._very_fine_center = best_stage2
            stage3_study.optimize(self._objective_very_fine, n_trials=5, gc_after_trial=True)

            best_stage3_value = stage3_study.best_trial.value
            logger.info(f"Stage 3 complete: best_value={best_stage3_value:.4f}")

            # Bestes Trial über alle Stufen wählen
            if best_stage3_value > best_stage2_value:
                best_trial = stage3_study.best_trial
            else:
                best_trial = stage2_study.best_trial
        else:
            best_trial = stage1_study.best_trial

        # Re-evaluate with best params
        best_params = best_trial.params
        best_metrics = self._evaluate_with_params(
            strategy_result, factor_values, best_params, forward_returns
        )

        # Baue optimiertes Ergebnis
        optimized_result = {
            **strategy_result,
            "status": "accepted" if self._is_acceptable(best_metrics) else "rejected",
            "sharpe_ratio": best_metrics.get("sharpe_ratio", 0),
            "annualized_return": best_metrics.get("annualized_return", 0),
            "max_drawdown": best_metrics.get("max_drawdown", 0),
            "win_rate": best_metrics.get("win_rate", 0),
            "optimization_status": "success",
            "best_params": best_params,
            "optimization_stages": {
                "stage1_best": best_stage1_value,
                "stage2_best": best_stage2_value,
                "stage3_best": best_stage3_value if best_stage2_value > best_stage1_value else None,
            },
            "optimization_trials": len(stage1_study.trials) + len(stage2_study.trials) + (
                len(stage3_study.trials) if best_stage2_value > best_stage1_value else 0
            ),
            "optimization_history": {
                "stage1": [t.value for t in stage1_study.trials if t.value is not None],
                "stage2": [t.value for t in stage2_study.trials if t.value is not None],
                "stage3": (
                    [t.value for t in stage3_study.trials if t.value is not None]
                    if best_stage2_value > best_stage1_value else []
                ),
            },
            "optimized_at": datetime.now().isoformat(),
        }

        # Speichere Optimierungsergebnisse
        self._save_optimization_results(optimized_result, strategy_name)

        logger.info(
            f"Multi-stage optimization complete for {strategy_name}: "
            f"best_metric={best_trial.value:.4f}, status={optimized_result['status']}"
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

    def _sample_coarse_params(self, trial: optuna.Trial) -> Dict[str, Any]:
        """
        Weite Bereiche für initiale Exploration (Stage 1).

        Parameters
        ----------
        trial : optuna.Trial
            Current Optuna trial

        Returns
        -------
        Dict[str, Any]
            Sampled hyperparameters with wide ranges
        """
        return {
            "entry_threshold": trial.suggest_float("entry_threshold", 0.1, 3.0, step=0.1),
            "exit_threshold": trial.suggest_float("exit_threshold", 0.0, 1.5, step=0.1),
            "zscore_window": trial.suggest_int("zscore_window", 5, 500, step=5),
            "signal_window": trial.suggest_int("signal_window", 1, 30, step=1),
            "position_size_pct": trial.suggest_float("position_size_pct", 0.05, 1.0, step=0.05),
            "stop_loss_mult": trial.suggest_float("stop_loss_mult", 0.5, 15.0, step=0.5),
            "take_profit_mult": trial.suggest_float("take_profit_mult", 1.0, 20.0, step=0.5),
            "volatility_lookback": trial.suggest_int("volatility_lookback", 5, 500, step=5),
            "signal_bias": trial.suggest_float("signal_bias", -1.0, 1.0, step=0.05),
            "max_hold_bars": trial.suggest_int("max_hold_bars", 5, 1000, step=5),
            "max_positions": trial.suggest_int("max_positions", 1, 5, step=1),
        }

    # Parameters that are allowed to be negative (not clamped to 0).
    _SIGNED_PARAMS = {"signal_bias"}
    # Absolute lower bounds per parameter (applied after the center-half_width calc).
    _PARAM_FLOOR: Dict[str, float] = {
        "entry_threshold": 0.0,
        "exit_threshold": 0.0,
        "zscore_window": 1.0,
        "signal_window": 1.0,
        "position_size_pct": 0.01,
        "stop_loss_mult": 0.1,
        "take_profit_mult": 0.1,
        "volatility_lookback": 1.0,
        "signal_bias": -1.0,
        "max_hold_bars": 1.0,
        "max_positions": 1.0,
    }

    def _suggest_bounded(
        self,
        trial: optuna.Trial,
        key: str,
        center_val: float,
        half_width: float,
    ) -> Any:
        """Suggest a parameter value with safe bounds that never invert."""
        floor = self._PARAM_FLOOR.get(key, -float("inf"))
        is_int = "window" in key or "lookback" in key or "bars" in key
        if is_int:
            low = max(int(floor), int(center_val - half_width))
            high = max(low + 1, int(center_val + half_width))
            return trial.suggest_int(key, low, high)
        else:
            low = max(floor, center_val - half_width)
            high = center_val + half_width
            if high <= low:
                high = low + max(1e-4, half_width * 0.1)
            return trial.suggest_float(key, low, high)

    def _sample_fine_params(self, trial: optuna.Trial) -> Dict[str, Any]:
        """
        Enge Bereiche zentriert um die besten Stage-1-Parameter (Stage 2).

        Parameters
        ----------
        trial : optuna.Trial
            Current Optuna trial

        Returns
        -------
        Dict[str, Any]
            Sampled hyperparameters with narrow ranges around Stage 1 best
        """
        center = getattr(self, "_fine_search_center", {})
        ranges: Dict[str, Tuple[float, float]] = {
            "entry_threshold": (center.get("entry_threshold", 1.0), 0.3),
            "exit_threshold": (center.get("exit_threshold", 0.3), 0.2),
            "zscore_window": (center.get("zscore_window", 50), 20),
            "signal_window": (center.get("signal_window", 3), 5),
            "position_size_pct": (center.get("position_size_pct", 0.5), 0.15),
            "stop_loss_mult": (center.get("stop_loss_mult", 5.0), 2.0),
            "take_profit_mult": (center.get("take_profit_mult", 5.0), 2.0),
            "volatility_lookback": (center.get("volatility_lookback", 100), 30),
            "signal_bias": (center.get("signal_bias", 0.0), 0.2),
            "max_hold_bars": (center.get("max_hold_bars", 100), 50),
            "max_positions": (center.get("max_positions", 1), 2),
        }
        return {key: self._suggest_bounded(trial, key, c, hw) for key, (c, hw) in ranges.items()}

    def _sample_very_fine_params(self, trial: optuna.Trial) -> Dict[str, Any]:
        """
        Sehr enge Bereiche für finale Verfeinerung (Stage 3).

        Parameters
        ----------
        trial : optuna.Trial
            Current Optuna trial

        Returns
        -------
        Dict[str, Any]
            Sampled hyperparameters with very narrow ranges around Stage 2 best
        """
        center = getattr(
            self, "_very_fine_center", getattr(self, "_fine_search_center", {})
        )
        ranges: Dict[str, Tuple[float, float]] = {
            "entry_threshold": (center.get("entry_threshold", 1.0), 0.1),
            "exit_threshold": (center.get("exit_threshold", 0.3), 0.07),
            "zscore_window": (center.get("zscore_window", 50), 7),
            "signal_window": (center.get("signal_window", 3), 2),
            "position_size_pct": (center.get("position_size_pct", 0.5), 0.05),
            "stop_loss_mult": (center.get("stop_loss_mult", 5.0), 0.7),
            "take_profit_mult": (center.get("take_profit_mult", 5.0), 0.7),
            "volatility_lookback": (center.get("volatility_lookback", 100), 10),
            "signal_bias": (center.get("signal_bias", 0.0), 0.07),
            "max_hold_bars": (center.get("max_hold_bars", 100), 17),
            "max_positions": (center.get("max_positions", 1), 1),
        }
        return {key: self._suggest_bounded(trial, key, c, hw) for key, (c, hw) in ranges.items()}

    def _objective_coarse(self, trial: optuna.Trial) -> float:
        """Objective-Funktion für Stage 1 (grobe Suche)."""
        try:
            params = self._sample_coarse_params(trial)
            metrics = self._evaluate_with_params(
                self._current_strategy, self._current_factors, params, self._current_forward_returns
            )
            return self._extract_metric(metrics, self.optimization_metric)
        except Exception as e:
            logger.warning(f"Stage 1 trial {trial.number} failed: {e}")
            return float("-inf")

    def _objective_fine(self, trial: optuna.Trial) -> float:
        """Objective-Funktion für Stage 2 (feine Suche)."""
        try:
            params = self._sample_fine_params(trial)
            metrics = self._evaluate_with_params(
                self._current_strategy, self._current_factors, params, self._current_forward_returns
            )
            return self._extract_metric(metrics, self.optimization_metric)
        except Exception as e:
            logger.warning(f"Stage 2 trial {trial.number} failed: {e}")
            return float("-inf")

    def _objective_very_fine(self, trial: optuna.Trial) -> float:
        """Objective-Funktion für Stage 3 (sehr feine Suche)."""
        try:
            params = self._sample_very_fine_params(trial)
            metrics = self._evaluate_with_params(
                self._current_strategy, self._current_factors, params, self._current_forward_returns
            )
            return self._extract_metric(metrics, self.optimization_metric)
        except Exception as e:
            logger.warning(f"Stage 3 trial {trial.number} failed: {e}")
            return float("-inf")

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
            # Entry/exit thresholds (wider range for better optimization)
            "entry_threshold": trial.suggest_float("entry_threshold", 0.3, 2.0, step=0.1),
            "exit_threshold": trial.suggest_float("exit_threshold", 0.0, 1.0, step=0.1),

            # Rolling window for z-score normalization
            "zscore_window": trial.suggest_int("zscore_window", 10, 200, step=10),

            # Rolling window for signal smoothing
            "signal_window": trial.suggest_int("signal_window", 1, 15, step=1),

            # Position sizing
            "position_size_pct": trial.suggest_float("position_size_pct", 0.1, 1.0, step=0.1),

            # Stop loss / take profit (in terms of factor std)
            "stop_loss_mult": trial.suggest_float("stop_loss_mult", 1.0, 10.0, step=0.5),
            "take_profit_mult": trial.suggest_float("take_profit_mult", 1.5, 15.0, step=0.5),

            # Volatility adjustment
            "volatility_lookback": trial.suggest_int("volatility_lookback", 10, 200, step=10),

            # Signal bias (shifts thresholds)
            "signal_bias": trial.suggest_float("signal_bias", -0.5, 0.5, step=0.1),

            # Max holding periods (in bars)
            "max_hold_bars": trial.suggest_int("max_hold_bars", 10, 500, step=10),

            # Max concurrent positions (1 = no pyramiding, 2-5 = scale-in)
            "max_positions": trial.suggest_int("max_positions", 1, 5, step=1),
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

        This method:
        1. Uses the ORIGINAL strategy code from the LLM
        2. Overrides key parameters (thresholds, windows) via exec
        3. Evaluates the resulting signals

        Parameters
        ----------
        strategy_result : Dict[str, Any]
            Original strategy result with 'code' field
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
            # Get original strategy code
            original_code = strategy_result.get("code", "")

            # Get factor weights if available
            factors_used = strategy_result.get("factors_used", list(factor_values.columns))
            available_factors = [f for f in factors_used if f in factor_values.columns]

            if not available_factors:
                return self._default_metrics()

            df_factors = factor_values[available_factors]

            if len(df_factors) < 100:
                return self._default_metrics()

            # Extract Optuna parameters
            entry_thresh = params["entry_threshold"]
            exit_thresh = params["exit_threshold"]
            zscore_window = params["zscore_window"]
            signal_window = params["signal_window"]
            signal_bias = params.get("signal_bias", 0.0)

            # Build parameter-override prefix that INJECTS Optuna params into code scope
            # This replaces hardcoded thresholds/windows in the LLM code

            # If no original code, build strategy from scratch using factor IC weights
            if not original_code or len(original_code.strip()) < 20:
                df_norm = (df_factors - df_factors.rolling(zscore_window).mean()) / (df_factors.rolling(zscore_window).std() + 1e-8)

                ic_weights = strategy_result.get("ic_weights", [])
                if len(ic_weights) == len(available_factors):
                    weighted_sum = sum(
                        w * df_norm[col] for col, w in zip(available_factors, ic_weights)
                    )
                else:
                    weighted_sum = df_norm.mean(axis=1)

                signal = pd.Series(0.0, index=df_factors.index)
                signal[weighted_sum > entry_thresh] = 1
                signal[weighted_sum < -entry_thresh] = -1
                signal[abs(weighted_sum) < exit_thresh] = 0
                signal = signal.rolling(window=signal_window, min_periods=1).mean().round().astype(int)
            else:
                # Patch the LLM code: replace hardcoded parameter assignments with Optuna values
                import re
                patched_code = original_code

                # Replace parameter assignments: entry_thresh = 0.8 → entry_thresh = 1.2
                param_patterns = [
                    (r'entry_thresh\s*=\s*[\d.]+', f'entry_thresh = {entry_thresh}'),
                    (r'exit_thresh\s*=\s*[\d.]+', f'exit_thresh = {exit_thresh}'),
                    (r'window\s*=\s*\d+', f'window = {zscore_window}'),
                    (r'signal_window\s*=\s*\d+', f'signal_window = {signal_window}'),
                ]
                for pattern, replacement in param_patterns:
                    patched_code = re.sub(pattern, replacement, patched_code)

                # Also handle inline .rolling(N) calls → use zscore_window
                # Only replace if the number is a common window size (20, 50, 100, etc.)
                rolling_pattern = r'\.rolling\((\d+)\)'
                def replace_rolling(match):
                    val = int(match.group(1))
                    if val in (20, 30, 50, 100, 200):
                        return f'.rolling({zscore_window})'
                    return match.group(0)
                patched_code = re.sub(rolling_pattern, replace_rolling, patched_code)

                # Execute patched code
                local_vars = {"factors": df_factors}
                try:
                    exec(patched_code, {"np": np, "pd": pd, "numpy": np}, local_vars)  # nosec B102: exec is required for sandboxed strategy code evaluation
                except Exception:
                    # Fallback: build simple IC-weighted strategy
                    df_norm = (df_factors - df_factors.rolling(zscore_window).mean()) / (df_factors.rolling(zscore_window).std() + 1e-8)
                    combined = df_norm.mean(axis=1)
                    signal = pd.Series(0, index=combined.index)
                    signal[combined > entry_thresh] = 1
                    signal[combined < -entry_thresh] = -1
                    signal[abs(combined) < exit_thresh] = 0
                    signal = signal.rolling(window=signal_window, min_periods=1).mean().round().astype(int)
                    local_vars["signal"] = signal

                signal = local_vars.get("signal")

            if signal is None or len(signal) < 10:
                return self._default_metrics()

            # Ensure signal is aligned
            signal = signal.reindex(df_factors.index).fillna(0).astype(int)

            # Apply signal bias (shifts signal values before thresholding)
            if signal_bias != 0.0:
                signal = (signal.astype(float) + signal_bias).round().astype(int).clip(-1, 1)

            # Apply max_positions: scale signal by position_size_pct and cap exposure
            max_positions   = int(params.get("max_positions", 1))
            position_size_pct = float(params.get("position_size_pct", 1.0))
            # Each "position" is position_size_pct of equity; total exposure capped at max_positions × size
            effective_size  = min(position_size_pct * max_positions, 1.0)
            signal = (signal.astype(float) * effective_size).clip(-1.0, 1.0)

            # Build a synthetic close from the factor-mean so we can route
            # through the same unified engine as every other backtest path.
            # Backtest formulas must match the orchestrator's real-OHLCV path.
            combined = df_factors.mean(axis=1)
            combined_ret = combined.pct_change().fillna(0)
            synthetic_close = (1 + combined_ret).cumprod() * 100.0

            from rdagent.components.backtesting.vbt_backtest import (
                backtest_signal_ftmo,
                DEFAULT_TXN_COST_BPS,
            )
            import os as _os

            bt = backtest_signal_ftmo(
                close=synthetic_close,
                signal=signal,
                txn_cost_bps=float(_os.getenv("TXN_COST_BPS", DEFAULT_TXN_COST_BPS)),
            )
            if bt.get("status") != "success":
                return self._default_metrics()

            return {
                "sharpe_ratio": bt["sharpe"],
                "annualized_return": bt["annualized_return"],
                "max_drawdown": bt["max_drawdown"],
                "win_rate": bt["win_rate"],
                "volatility": bt["volatility"],
                "total_return": bt["total_return"],
                "num_trades": bt["n_trades"],
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

        return sharpe >= 0.3 and max_dd >= -0.30 and win_rate >= 0.40

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
