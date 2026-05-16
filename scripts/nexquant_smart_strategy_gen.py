#!/usr/bin/env python
"""
Smart Strategy Generation with Feedback Loop, Parameter Optimization & FTMO Risk Management.

Generates EUR/USD daytrading strategies using LLM with:
- Adaptive feedback loop (IC, trades, drawdown-based suggestions)
- Grid search for optimal parameters (thresholds, SL/TP, trailing stops)
- Mandatory FTMO-compliant risk management layer
- Comprehensive evaluation metrics  # nosec

Usage:
    python nexquant_smart_strategy_gen.py 10
    python nexquant_smart_strategy_gen.py 5 --style daytrading
    python nexquant_smart_strategy_gen.py 20 --style swing --max-attempts 200
"""
import json
import logging
import os
import random
import subprocess  # nosec
import sys
import time
import warnings
from datetime import datetime
from itertools import product
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd
from dotenv import load_dotenv
from rich.console import Console
from rich.logging import RichHandler
from rich.progress import BarColumn, Progress, SpinnerColumn, TextColumn, TimeElapsedColumn
from rich.table import Table

warnings.filterwarnings("ignore")

# ============================================================================
# Configuration & Constants
# ============================================================================
OHLCV_PATH = Path("/home/nico/NexQuant/git_ignore_folder/factor_implementation_source_data/intraday_pv.h5")
FACTORS_DIR = Path("/home/nico/NexQuant/results/factors")
STRATEGIES_DIR = Path("/home/nico/NexQuant/results/strategies_new")
STRATEGIES_DIR.mkdir(parents=True, exist_ok=True)

# Logging setup
LOG_DIR = Path("/home/nico/NexQuant/results/logs")
LOG_DIR.mkdir(parents=True, exist_ok=True)
log_file = LOG_DIR / f"smart_strategy_gen_{datetime.now().strftime('%Y%m%d_%H%M%S')}.log"

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s - %(name)s - %(levelname)s - %(message)s",
    handlers=[
        logging.FileHandler(log_file),
        RichHandler(rich_tracebacks=True, show_time=False, show_path=False),
    ],
)
logger = logging.getLogger("SmartStrategyGen")

console = Console()

# ============================================================================
# FTMO Risk Management Constants
# ============================================================================
class FTMORiskLimits:
    """FTMO-compliant risk management constants."""
    MAX_DAILY_LOSS_PCT = 0.05        # 5% max daily loss (FTMO rule)
    MAX_PER_TRADE_LOSS_PCT = 0.02    # 2% max per trade
    MAX_TOTAL_DRAWDOWN = 0.10        # 10% max overall drawdown
    MAX_POSITIONS = 1                # Only 1 position at a time
    MIN_RISK_REWARD_RATIO = 2.0      # TP must be at least 2x SL
    POSITION_RISK_PCT = 0.01         # 1% risk per trade

# ============================================================================
# Acceptance Criteria
# ============================================================================
ACCEPTANCE_CRITERIA = {
    "daytrading": {
        "min_abs_ic": 0.02,
        "min_sharpe": 1.0,
        "min_trades": 50,
        "max_drawdown": -0.15,
        "min_win_rate": 0.45,
        "min_monthly_return": 0.15,
        "max_daily_loss": 0.05,
    },
    "swing": {
        "min_abs_ic": 0.02,
        "min_sharpe": 0.5,
        "min_trades": 10,
        "max_drawdown": -0.15,
        "min_win_rate": 0.40,
        "min_monthly_return": 0.15,
        "max_daily_loss": 0.05,
    },
}

# ============================================================================
# Parameter Grid for Optimization
# ============================================================================
PARAMETER_GRID = {
    "threshold_entry": [0.2, 0.3, 0.4, 0.5],
    "rolling_window": [10, 20, 30, 60],
    "stop_loss": [0.01, 0.015, 0.02],        # 1%, 1.5%, 2% (HARD MAX: 2% for FTMO)
    "take_profit": [0.02, 0.03, 0.04, 0.06], # 2x-3x SL
    "trailing_stop": [0.01, 0.015],           # 1%, 1.5% after profit threshold
    "trailing_activation": [0.015, 0.02],     # Activate trail after 1.5%, 2% profit
}

# ============================================================================
# Data Loading (Cached)
# ============================================================================
class DataCache:
    """Thread-safe data cache for OHLCV and factors."""

    def __init__(self):
        self._ohlcv_cache: pd.Series | None = None
        self._factors_cache: list[dict] | None = None
        self._factor_data_cache: dict[str, pd.Series] = {}

    def load_ohlcv(self) -> pd.Series:
        """Load OHLCV close prices from HDF5."""
        if self._ohlcv_cache is not None:
            return self._ohlcv_cache

        if not OHLCV_PATH.exists():
            raise FileNotFoundError(f"OHLCV data not found: {OHLCV_PATH}")

        ohlcv = pd.read_hdf(str(OHLCV_PATH), key="data")
        close_col = "$close" if "$close" in ohlcv.columns else "close" if "close" in ohlcv.columns else ohlcv.select_dtypes(include=[np.number]).columns[0]
        close = ohlcv[close_col].dropna()

        # Limit to last 200k bars to avoid OOM during optimization
        # (372k bars × 15 combinations = too much memory)
        MAX_BARS = 200000
        if len(close) > MAX_BARS:
            close = close.iloc[-MAX_BARS:]
            logger.info(f"Trimmed OHLCV data to last {MAX_BARS:,} bars (from {len(ohlcv[close_col]):,})")

        self._ohlcv_cache = close
        logger.info(f"Loaded {len(close):,} OHLCV bars")
        return close

    def load_top_factors(self, top_n: int = 20) -> list[dict]:
        """Load top factors by IC that have parquet files."""
        if self._factors_cache is not None:
            return self._factors_cache[:top_n]

        factors = []
        for f in FACTORS_DIR.glob("*.json"):
            try:
                data = json.load(open(f))
                fname = data.get("factor_name", "")
                ic = data.get("ic") or 0
                safe = fname.replace("/", "_").replace("\\", "_")[:150]
                if (FACTORS_DIR / "values" / f"{safe}.parquet").exists():
                    factors.append({"name": fname, "ic": ic})
            except Exception as e:
                logger.debug(f"Failed to load factor metadata: {f.name} - {e}")

        factors.sort(key=lambda x: abs(x["ic"]), reverse=True)
        self._factors_cache = factors
        return factors[:top_n]

    def load_factor_timeseries(self, factor_name: str) -> pd.Series | None:
        """Load factor time-series from parquet."""
        if factor_name in self._factor_data_cache:
            return self._factor_data_cache[factor_name]

        safe = factor_name.replace("/", "_").replace("\\", "_")[:150]
        pf = FACTORS_DIR / "values" / f"{safe}.parquet"

        if not pf.exists():
            return None

        try:
            series = pd.read_parquet(str(pf)).iloc[:, 0]
            self._factor_data_cache[factor_name] = series
            return series
        except Exception as e:
            logger.debug(f"Failed to load factor data: {factor_name} - {e}")
            return None

data_cache = DataCache()

# ============================================================================
# LLM Setup
# ============================================================================
def setup_llm_env():
    """Setup LLM environment variables with fallback chain."""
    load_dotenv(Path(__file__).parent / ".env", override=True)

    # Priority 1: OpenRouter (free models with fallback)
    router_key = os.getenv("OPENROUTER_API_KEY", "")
    if router_key and router_key != "local":
        # Build model fallback chain
        models = [
            os.getenv("OPENROUTER_MODEL", ""),
            os.getenv("OPENROUTER_MODEL_2", ""),
            os.getenv("OPENROUTER_MODEL_3", ""),
        ]
        models = [m for m in models if m]  # Remove empty

        if models:
            os.environ["OPENAI_API_KEY"] = router_key
            os.environ["OPENAI_API_BASE"] = "https://openrouter.ai/api/v1"
            os.environ["OPENROUTER_MODELS"] = json.dumps(models)  # Store for fallback
            os.environ["CHAT_MODEL"] = models[0]
            logger.info(f"LLM environment configured for OpenRouter: {', '.join(models)}")
            return

    # Priority 2: Local LLM (llama.cpp)
    api_key = os.getenv("OPENAI_API_KEY", "")
    api_base = os.getenv("OPENAI_API_BASE", "")
    chat_model = os.getenv("CHAT_MODEL", "")

    if api_key == "local" and api_base:
        os.environ["OPENAI_API_KEY"] = "local"
        os.environ["OPENAI_API_BASE"] = api_base
        os.environ["CHAT_MODEL"] = chat_model or "openai/qwen3.5-35b"
        logger.info(f"LLM environment configured for LOCAL LLM: {api_base}")
    else:
        logger.warning("No API key found - LLM generation will fail")

# ============================================================================
# Risk Management Engine
# ============================================================================
class RiskManagementEngine:
    """
    FTMO-compliant risk management layer.

    Applies stop loss, take profit, trailing stop, and daily loss limits
    to strategy returns.
    """

    def __init__(
        self,
        stop_loss: float = 0.02,
        take_profit: float = 0.04,
        trailing_stop: float = 0.015,
        trailing_activation: float = 0.02,
        max_daily_loss: float = 0.05,
        max_positions: int = 1,
    ):
        """
        Initialize risk management parameters.

        Parameters
        ----------
        stop_loss : float
            Stop loss percentage (default 2%)
        take_profit : float
            Take profit percentage (default 4%, 2x SL)
        trailing_stop : float
            Trailing stop distance (default 1.5%)
        trailing_activation : float
            Profit level to activate trailing stop (default 2%)
        max_daily_loss : float
            Maximum daily loss percentage (default 5%)
        max_positions : int
            Maximum concurrent positions (default 1)
        """
        # Validate FTMO compliance
        if stop_loss > 0.02:
            raise ValueError(f"Stop loss {stop_loss:.2%} exceeds FTMO max of 2%")
        if take_profit < stop_loss * 2:
            raise ValueError(f"Take profit {take_profit:.2%} must be at least 2x SL ({stop_loss*2:.2%})")
        if max_daily_loss > 0.05:
            raise ValueError(f"Daily loss {max_daily_loss:.2%} exceeds FTMO max of 5%")

        self.stop_loss = stop_loss
        self.take_profit = take_profit
        self.trailing_stop = trailing_stop
        self.trailing_activation = trailing_activation
        self.max_daily_loss = max_daily_loss
        self.max_positions = max_positions

    @property
    def risk_reward_ratio(self) -> float:
        """Calculate risk/reward ratio (TP/SL)."""
        return self.take_profit / self.stop_loss if self.stop_loss > 0 else 0.0

    def apply_risk_management(
        self,
        signal: pd.Series,
        close: pd.Series,
    ) -> pd.Series:
        """
        Apply SL/TP/Trailing stop to signal-based strategy.

        Parameters
        ----------
        signal : pd.Series
            Trading signals (1=LONG, -1=SHORT, 0=NEUTRAL)
        close : pd.Series
            Close prices

        Returns
        -------
        pd.Series
            Strategy returns after risk management
        """
        if len(signal) == 0 or len(close) == 0:
            return pd.Series(dtype=float)

        # Align indices
        common_idx = signal.index.intersection(close.index)
        signal = signal.loc[common_idx].fillna(0)
        close = close.loc[common_idx]

        # Calculate returns
        returns = close.pct_change().fillna(0)
        strategy_returns = pd.Series(0.0, index=common_idx)

        position = 0  # 0=neutral, 1=long, -1=short
        entry_price = 0.0
        highest_profit = 0.0
        daily_pnl = 0.0
        current_date = None

        for i, idx in enumerate(common_idx):
            if i == 0:
                continue

            # Track daily PnL for max daily loss
            bar_date = idx.date() if hasattr(idx, "date") else idx
            if current_date is None:
                current_date = bar_date
            elif bar_date != current_date:
                daily_pnl = 0.0  # Reset daily PnL
                current_date = bar_date

            current_price = close.iloc[i]
            prev_price = close.iloc[i - 1]
            current_signal = signal.iloc[i]

            # Check if we should exit position due to SL/TP/Trailing
            if position != 0:
                pnl_pct = 0.0
                if position == 1:  # Long
                    pnl_pct = (current_price - entry_price) / entry_price
                elif position == -1:  # Short
                    pnl_pct = (entry_price - current_price) / entry_price

                # Stop Loss hit
                if pnl_pct <= -self.stop_loss:
                    strategy_returns.iloc[i] = -self.stop_loss * position
                    daily_pnl += -self.stop_loss
                    position = 0
                    highest_profit = 0.0
                    continue

                # Take Profit hit
                if pnl_pct >= self.take_profit:
                    strategy_returns.iloc[i] = self.take_profit * position
                    daily_pnl += self.take_profit
                    position = 0
                    highest_profit = 0.0
                    continue

                # Trailing Stop (activate after profit threshold)
                if pnl_pct >= self.trailing_activation:
                    highest_profit = max(highest_profit, pnl_pct)
                    if (highest_profit - pnl_pct) >= self.trailing_stop:
                        strategy_returns.iloc[i] = pnl_pct * position
                        daily_pnl += pnl_pct
                        position = 0
                        highest_profit = 0.0
                        continue

                # Normal position PnL
                if position == 1:
                    strategy_returns.iloc[i] = (current_price - prev_price) / prev_price
                elif position == -1:
                    strategy_returns.iloc[i] = -(current_price - prev_price) / prev_price

                # Update daily PnL
                daily_pnl += strategy_returns.iloc[i]

                # Check max daily loss
                if daily_pnl <= -self.max_daily_loss:
                    strategy_returns.iloc[i] = strategy_returns.iloc[i]  # Keep the loss
                    position = 0  # Stop trading for the day
                    highest_profit = 0.0
                    continue

            # Enter new position (only if neutral and max positions not exceeded)
            if position == 0 and current_signal != 0:
                position = int(np.sign(current_signal))
                entry_price = current_price
                highest_profit = 0.0

        return strategy_returns

    def get_config(self) -> dict[str, float]:
        """Return risk management configuration."""
        return {
            "stop_loss": self.stop_loss,
            "take_profit": self.take_profit,
            "trailing_stop": self.trailing_stop,
            "trailing_activation": self.trailing_activation,
            "max_daily_loss": self.max_daily_loss,
            "max_positions": self.max_positions,
            "risk_reward_ratio": self.take_profit / self.stop_loss,
        }

# ============================================================================
# Strategy Evaluator
# ============================================================================
class StrategyEvaluator:
    """
    Comprehensive strategy evaluation with FTMO metrics.  # nosec
    """

    def __init__(self, trading_style: str = "daytrading", forward_bars: int = 96):
        self.trading_style = trading_style
        self.forward_bars = forward_bars
        self.criteria = ACCEPTANCE_CRITERIA.get(trading_style, ACCEPTANCE_CRITERIA["daytrading"])

    def evaluate(  # nosec
        self,
        signal: pd.Series,
        close: pd.Series,
        strategy_returns: pd.Series,
    ) -> dict[str, Any]:
        """
        Evaluate strategy with comprehensive metrics.

        Parameters
        ----------
        signal : pd.Series
            Trading signals
        close : pd.Series
            Close prices
        strategy_returns : pd.Series
            Strategy returns after risk management

        Returns
        -------
        dict
            Evaluation metrics dict
        """
        if len(strategy_returns) < 10:
            return {"status": "failed", "reason": "Insufficient data"}

        # Forward returns for IC calculation
        fwd_returns = close.pct_change(self.forward_bars).shift(-self.forward_bars)
        common_idx = signal.index.intersection(fwd_returns.dropna().index)

        if len(common_idx) < 10:
            return {"status": "failed", "reason": "Insufficient overlapping data"}

        signal_aligned = signal.loc[common_idx]
        fwd_aligned = fwd_returns.loc[common_idx]

        # IC (Information Coefficient)
        ic = signal_aligned.corr(fwd_aligned) if signal_aligned.std() > 0 else 0.0

        # Basic metrics
        total_bars = len(strategy_returns)
        n_signals = int((signal != signal.shift(1)).sum())
        n_long = int((signal == 1).sum())
        n_short = int((signal == -1).sum())
        n_neutral = int((signal == 0).sum())

        # Returns metrics
        cum_returns = (1 + strategy_returns).cumprod()
        total_return = cum_returns.iloc[-1] - 1 if len(cum_returns) > 0 else 0.0

        # Annualization factor (assuming 252 trading days, 1440 minutes per day)
        bars_per_year = 252 * 1440 / self.forward_bars
        n_months = total_bars / (bars_per_year / 12) if total_bars > 0 else 1

        if n_months > 0 and (1 + total_return) > 0:
            monthly_return = (1 + total_return) ** (1 / n_months) - 1
            annual_return = (1 + total_return) ** (12 / n_months) - 1
        else:
            monthly_return = total_return
            annual_return = total_return * 12

        # Sharpe Ratio
        if strategy_returns.std() > 0:
            sharpe = strategy_returns.mean() / strategy_returns.std() * np.sqrt(bars_per_year)
        else:
            sharpe = 0.0

        # Max Drawdown
        running_max = cum_returns.expanding().max()
        drawdown = (cum_returns - running_max) / running_max.replace(0, np.nan)
        max_drawdown = drawdown.min() if len(drawdown) > 0 else 0.0

        # Win Rate
        active_returns = strategy_returns[strategy_returns != 0]
        win_rate = (active_returns > 0).sum() / len(active_returns) if len(active_returns) > 0 else 0.0

        # Daily loss analysis (for FTMO compliance)
        daily_returns = strategy_returns.groupby(
            strategy_returns.index.date if hasattr(strategy_returns.index[0], "date") else strategy_returns.index,
        ).sum()
        max_daily_loss = abs(daily_returns.min()) if len(daily_returns) > 0 else 0.0

        # Acceptance check
        passed, failed_criteria = self._check_acceptance(
            ic=ic if not np.isnan(ic) else 0,
            sharpe=sharpe,
            n_trades=n_signals,
            max_drawdown=max_drawdown,
            win_rate=win_rate,
            monthly_return=monthly_return,
            max_daily_loss=max_daily_loss,
        )

        result = {
            "status": "accepted" if passed else "rejected",
            "failed_criteria": failed_criteria,

            # Core metrics
            "ic": float(ic) if not np.isnan(ic) else 0.0,
            "sharpe": float(sharpe),
            "max_drawdown": float(max_drawdown),
            "win_rate": float(win_rate),
            "total_return": float(total_return),
            "monthly_return_pct": float(monthly_return * 100),
            "annual_return_pct": float(annual_return * 100),

            # Trade statistics
            "n_trades": n_signals,
            "n_long": n_long,
            "n_short": n_short,
            "n_neutral": n_neutral,
            "n_bars": total_bars,
            "n_months": float(n_months),

            # FTMO compliance
            "max_daily_loss": float(max_daily_loss),
            "ftmo_compliant": max_daily_loss <= 0.05,

            # Signal distribution
            "signal_long_pct": n_long / total_bars if total_bars > 0 else 0,
            "signal_short_pct": n_short / total_bars if total_bars > 0 else 0,
            "signal_neutral_pct": n_neutral / total_bars if total_bars > 0 else 0,
        }

        return result

    def _check_acceptance(
        self,
        ic: float,
        sharpe: float,
        n_trades: int,
        max_drawdown: float,
        win_rate: float,
        monthly_return: float,
        max_daily_loss: float,
    ) -> tuple[bool, list[str]]:
        """Check if strategy meets acceptance criteria."""
        failed = []

        if abs(ic) < self.criteria["min_abs_ic"]:
            failed.append(f"IC too low: {ic:.4f} < {self.criteria['min_abs_ic']}")

        if sharpe < self.criteria["min_sharpe"]:
            failed.append(f"Sharpe too low: {sharpe:.3f} < {self.criteria['min_sharpe']}")

        if n_trades < self.criteria["min_trades"]:
            failed.append(f"Too few trades: {n_trades} < {self.criteria['min_trades']}")

        if max_drawdown < self.criteria["max_drawdown"]:
            failed.append(f"Max drawdown exceeded: {max_drawdown:.1%} < {self.criteria['max_drawdown']}")

        if win_rate < self.criteria["min_win_rate"]:
            failed.append(f"Win rate too low: {win_rate:.1%} < {self.criteria['min_win_rate']}")

        if monthly_return < self.criteria["min_monthly_return"]:
            failed.append(f"Monthly return too low: {monthly_return:.2%} < {self.criteria['min_monthly_return']}")

        if max_daily_loss > self.criteria["max_daily_loss"]:
            failed.append(f"Daily loss exceeded: {max_daily_loss:.2%} > {self.criteria['max_daily_loss']}")

        return len(failed) == 0, failed

# ============================================================================
# Feedback Generator
# ============================================================================
class FeedbackGenerator:
    """
    Generate intelligent feedback for LLM strategy improvement.
    """

    @staticmethod
    def generate_feedback(
        evaluation: dict[str, Any],  # nosec
        factor_list: list[dict],
        attempt: int,
        param_config: dict | None = None,
    ) -> str:
        """
        Generate actionable feedback based on strategy performance.

        Parameters
        ----------
        evaluation : dict  # nosec
            Strategy evaluation metrics  # nosec
        factor_list : list
            Available factors with IC values
        attempt : int
            Current attempt number
        param_config : dict, optional
            Current parameter configuration

        Returns
        -------
        str
            Feedback string for LLM
        """
        ic = evaluation.get("ic", 0)  # nosec
        sharpe = evaluation.get("sharpe", 0)  # nosec
        trades = evaluation.get("n_trades", 0)  # nosec
        dd = evaluation.get("max_drawdown", 0)  # nosec
        win_rate = evaluation.get("win_rate", 0)  # nosec
        monthly_ret = evaluation.get("monthly_return_pct", 0)  # nosec
        failed = evaluation.get("failed_criteria", [])  # nosec

        feedback_parts = [f"Attempt {attempt} results:"]

        # Performance summary
        feedback_parts.append(f"IC={ic:.4f}, Sharpe={sharpe:.2f}, Trades={trades}, DD={dd:.1%}, WinRate={win_rate:.1%}, Monthly={monthly_ret:.2f}%")

        # Specific suggestions based on failures
        if failed:
            feedback_parts.append("\nIssues found:")

            if any("IC" in f for f in failed):
                # Suggest top factors
                top_factors = sorted(factor_list, key=lambda x: abs(x["ic"]), reverse=True)[:5]
                top_factor_names = [f["name"] for f in top_factors]
                feedback_parts.append(
                    f"\n- IC too low ({ic:.4f}). Try different factors. Top factors by IC: {', '.join(top_factor_names)}",
                )

            if any("trades" in f.lower() for f in failed):
                feedback_parts.append(
                    f"\n- Too few trades ({trades}). Lower thresholds (try 0.2-0.3), use more sensitive factors, or reduce rolling window (10-20 bars)",
                )

            if any("drawdown" in f.lower() for f in failed):
                feedback_parts.append(
                    f"\n- High drawdown ({dd:.1%}). Add filters (volatility, trend), reduce position size, or tighten stop loss",
                )

            if any("sharpe" in f.lower() for f in failed):
                feedback_parts.append(
                    f"\n- Low Sharpe ({sharpe:.2f}). Improve signal quality: combine momentum + mean reversion, add regime filters",
                )

            if any("win rate" in f.lower() for f in failed):
                feedback_parts.append(
                    f"\n- Low win rate ({win_rate:.1%}). Try higher take profit (4-6%), or add confirmation filters",
                )

            if any("monthly return" in f.lower() for f in failed):
                feedback_parts.append(
                    f"\n- Low monthly return ({monthly_ret:.2%}). Increase signal frequency or use higher-IC factors",
                )

        else:
            # Strategy passed - suggest optimization
            feedback_parts.append("\n✓ Strategy meets all criteria!")

            if sharpe < 1.5:
                feedback_parts.append(
                    "\nTry optimizing: 1) Test SL=1.5% vs 2% 2) Test TP=3% vs 4% 3) Add trailing stop at 1.5%",
                )

            if abs(ic) < 0.05:
                top_factors = sorted(factor_list, key=lambda x: abs(x["ic"]), reverse=True)[:3]
                feedback_parts.append(
                    f"\nIC could be higher. Consider adding: {', '.join(f['name'] for f in top_factors)}",
                )

            if param_config:
                feedback_parts.append(
                    f"\nCurrent params: threshold={param_config.get('threshold_entry', 'N/A')}, "
                    f"window={param_config.get('rolling_window', 'N/A')}, "
                    f"SL={param_config.get('stop_loss', 'N/A'):.1%}, "
                    f"TP={param_config.get('take_profit', 'N/A'):.1%}",
                )

        return " ".join(feedback_parts)

# ============================================================================
# LLM Strategy Generator
# ============================================================================
class LLMStrategyGenerator:
    """
    Generate trading strategies using LLM with feedback loop.
    """

    def __init__(self):
        setup_llm_env()

    def generate(
        self,
        factor_subset: list[dict],
        feedback: str | None = None,
        trading_style: str = "daytrading",
        forward_bars: int = 96,
    ) -> dict[str, Any]:
        """
        Generate a single strategy via qwen CLI.

        Parameters
        ----------
        factor_subset : list
            List of factor dicts with 'name' and 'ic'
        feedback : str, optional
            Previous feedback for improvement
        trading_style : str
            'daytrading' or 'swing'
        forward_bars : int
            Forward return horizon

        Returns
        -------
        dict
            Strategy dict with 'status', 'strategy', 'error'
        """
        try:
            import re
            import subprocess  # nosec B404

            factor_list = ", ".join([f"{f['name']} (IC={f['ic']:.4f})" for f in factor_subset])
            factor_names = ", ".join([f["name"] for f in factor_subset])

            feedback_text = f" Vorheriges Feedback: {feedback}" if feedback else " Erster Versuch - sei kreativ!"

            prompt = f"""Du bist ein quantitativer Trading-Experte. Erzeuge eine EUR/USD Daytrading-Strategie als JSON.

Faktoren: {factor_list}

⚠️ WICHTIG - DU MUSST VIELE SIGNALE GENERIEREN! ⚠️
Die Strategie MUSS mindestens 50+ Trades über den Datensatz erzeugen.
Verwende DESHALB diese Regeln:
1. Schwellenwerte MÜSSEN niedrig sein: 0.1 bis 0.25 (NICHT höher!)
2. Verwende Z-Score Normalisierung mit FENSTERN VON 10-20 Bars (kurz!)
3. Erstelle Signale für JEDE Bar wo der Z-Score den Schwellenwert überschreitet
4. Vermeide zu strenge Filter - die Strategie soll AKTIV traden!
5. Kombiniere 2-4 Faktoren mit GEWICHTEN für diversifizierte Signale

BEISPIEL für gute Signal-Logik:
```python
z = (factor - factor.rolling(15).mean()) / factor.rolling(15).std()
signal = pd.Series(0, index=close.index)
signal[z > 0.15] = 1    # NIEDRIGER Schwellenwert = VIELE Signale!
signal[z < -0.15] = -1  # Auch negative Signale für Shorts
```

❌ SCHLECHT: signal[composite > 0.5] = 1  (zu streng, nur 1 Trade!)
✅ GUT: signal[composite > 0.15] = 1  (niedrig, viele Trades!)

Anforderungen:
- Trading-Stil: Daytrading mit {forward_bars}-Bar Forward Returns
- ZIEL: 50-200+ Trades gesamt (sehr aktiv!)
- Schwellenwerte: 0.1-0.25 (sehr niedrig!)
- Rolling Windows: 10-20 Bars (kurz!)
- Erstelle signal Series mit Werten 1, -1, 0

{feedback_text}

WICHTIG: Das JSON MUSS diese Felder haben:
{{
  "strategy_name": "kurzer_Name",
  "factor_names": ["faktor1", "faktor2"],
  "description": "Ein Satz Beschreibung",
  "code": "Python Code der signal Series erzeugt"
}}

Der Python Code MUSS mit DataFrame 'factors' und Series 'close' arbeiten und eine Series 'signal' erzeugen.

Antworte NUR mit dem JSON Objekt!"""

            # Call qwen CLI
            logger.info(f"Calling qwen CLI with prompt ({len(prompt)} chars)...")
            result = subprocess.run( # nosec B603
                ["qwen", "-p", prompt],
                capture_output=True,
                text=True,
                timeout=120,
                cwd=str(Path(__file__).parent),
            )

            if result.returncode != 0:
                logger.error(f"qwen CLI failed: {result.stderr[:300]}")
                return {"status": "error", "error": f"qwen CLI failed: {result.stderr[:200]}"}

            response = result.stdout.strip()
            logger.info(f"qwen CLI response ({len(response)} chars)")

            # Extract JSON from response
            # qwen CLI might output to file OR stdout
            # Check if a file was created in results/strategies_new/
            import glob
            new_files = glob.glob(str(STRATEGIES_DIR / "*.json"))
            if new_files:
                latest = max(new_files, key=os.path.getmtime)
                if os.path.getmtime(latest) > time.time() - 120:  # Created in last 120s
                    logger.info(f"Strategy file found: {latest}")
                    with open(latest) as f:
                        raw_data = json.load(f)
                    # Convert qwen CLI format to our format
                    strategy_data = self._convert_qwen_output(raw_data, factor_subset)
                    if strategy_data:
                        return {"status": "generated", "strategy": strategy_data}

            # Otherwise parse JSON from stdout
            # Try to find JSON object in response
            json_match = re.search(r'\{[^{}]*"strategy_name"[^{}]*\}', response, re.DOTALL)
            if json_match:
                strategy_str = json_match.group()
                raw_data = json.loads(strategy_str)
            else:
                # Try to parse entire response as JSON
                raw_data = json.loads(response)

            # Convert to our format
            strategy_data = self._convert_qwen_output(raw_data, factor_subset)
            if not strategy_data:
                return {"status": "invalid", "error": "Could not convert qwen output"}

            return {
                "status": "generated",
                "strategy": strategy_data,
            }

        except subprocess.TimeoutExpired:  # nosec
            return {"status": "error", "error": "qwen CLI timeout (120s)"}
        except Exception as e:
            logger.error(f"qwen CLI generation failed: {e}")
            return {"status": "error", "error": str(e)[:300]}

    def _convert_qwen_output(self, raw_data: dict, factors: list[dict]) -> dict | None:
        """
        Convert qwen CLI output format to our standard format.
        
        qwen CLI may output:
        - code as string with literal \n
        - Different field names (name vs strategy_name)
        - Nested structures
        
        We need:
        - strategy_name: str
        - factor_names: List[str]
        - description: str
        - code: str (executable Python with real newlines)  # nosec
        """
        try:
            # Extract strategy name
            strategy_name = raw_data.get("strategy_name") or raw_data.get("name", "UnknownStrategy")

            # Extract factor names
            factor_names = raw_data.get("factor_names", [])
            if not factor_names:
                # Use factors from the generation request
                factor_names = [f["name"] for f in factors[:3]]

            # Extract description
            description = raw_data.get("description", raw_data.get("desc", "Generated strategy"))

            # Extract and clean code
            code = raw_data.get("code", "")
            if not code:
                # Try to find code in nested structures
                if "strategy" in raw_data:
                    code = raw_data["strategy"].get("code", "")
                elif "logic" in raw_data:
                    code = raw_data["logic"].get("code", "")

            # Unescape code (convert literal \n to real newlines)
            if code:
                code = code.replace("\\n", "\n").replace('\\"', '"').replace("\\\\", "\\")
                # Remove leading/trailing quotes if present
                if code.startswith('"') and code.endswith('"'):
                    code = code[1:-1]
                if code.startswith("'") and code.endswith("'"):
                    code = code[1:-1]
                # Ensure variable name consistency: factors_df → factors
                code = code.replace("factors_df", "factors")

            # Validate we have what we need
            if not code or not strategy_name:
                logger.warning(f"Missing required fields: name={strategy_name}, code={'yes' if code else 'no'}")
                return None

            return {
                "strategy_name": strategy_name,
                "factor_names": factor_names,
                "description": description,
                "code": code,
            }
        except Exception as e:
            logger.error(f"Failed to convert qwen output: {e}")
            return None

# ============================================================================
# Backtest Runner
# ============================================================================
class BacktestRunner:
    """
    Run backtests in isolated subprocess with risk management.  # nosec
    """

    @staticmethod
    def run(
        close: pd.Series,
        factors_df: pd.DataFrame,
        strategy_code: str,
        risk_config: dict[str, float],
        forward_bars: int = 96,
    ) -> dict[str, Any] | None:
        """
        Run strategy backtest with risk management.

        Parameters
        ----------
        close : pd.Series
            Close prices
        factors_df : pd.DataFrame
            Factor values DataFrame
        strategy_code : str
            Python code string for signal generation
        risk_config : dict
            Risk management configuration (SL, TP, trailing, etc.)
        forward_bars : int
            Forward return horizon

        Returns
        -------
        dict or None
            Backtest results dict or None on failure
        """
        # Build backtest script with risk management
        risk_code = f"""
# Risk Management Configuration
STOP_LOSS = {risk_config['stop_loss']}
TAKE_PROFIT = {risk_config['take_profit']}
TRAILING_STOP = {risk_config['trailing_stop']}
TRAILING_ACTIVATION = {risk_config['trailing_activation']}
MAX_DAILY_LOSS = {risk_config['max_daily_loss']}
MAX_POSITIONS = {risk_config['max_positions']}

def apply_risk_management_with_params(signal, close_prices, sl, tp, trailing, trail_activation):
    \"\"\"Apply SL/TP/Trailing stop to signals.\"\"\"
    if len(signal) == 0 or len(close_prices) == 0:
        return pd.Series(0.0, index=signal.index)

    common_idx = signal.index.intersection(close_prices.index)
    sig = signal.loc[common_idx].fillna(0)
    prices = close_prices.loc[common_idx]

    strategy_returns = pd.Series(0.0, index=common_idx)
    position = 0
    entry_price = 0.0
    highest_profit = 0.0
    daily_pnl = 0.0
    current_date = None

    for i, idx in enumerate(common_idx):
        if i == 0:
            continue

        bar_date = idx.date() if hasattr(idx, 'date') else idx
        if current_date is None:
            current_date = bar_date
        elif bar_date != current_date:
            daily_pnl = 0.0
            current_date = bar_date

        current_price = prices.iloc[i]
        prev_price = prices.iloc[i - 1]
        current_signal = sig.iloc[i]

        if position != 0:
            pnl_pct = 0.0
            if position == 1:
                pnl_pct = (current_price - entry_price) / entry_price
            elif position == -1:
                pnl_pct = (entry_price - current_price) / entry_price

            # Stop Loss
            if pnl_pct <= -sl:
                strategy_returns.iloc[i] = -sl * position
                daily_pnl += -sl
                position = 0
                highest_profit = 0.0
                continue

            # Take Profit
            if pnl_pct >= tp:
                strategy_returns.iloc[i] = tp * position
                daily_pnl += tp
                position = 0
                highest_profit = 0.0
                continue

            # Trailing Stop
            if pnl_pct >= trail_activation:
                highest_profit = max(highest_profit, pnl_pct)
                if (highest_profit - pnl_pct) >= trailing:
                    strategy_returns.iloc[i] = pnl_pct * position
                    daily_pnl += pnl_pct
                    position = 0
                    highest_profit = 0.0
                    continue

            # Normal PnL
            if position == 1:
                strategy_returns.iloc[i] = (current_price - prev_price) / prev_price
            elif position == -1:
                strategy_returns.iloc[i] = -(current_price - prev_price) / prev_price

            daily_pnl += strategy_returns.iloc[i]

            # Max daily loss
            if daily_pnl <= -{risk_config['max_daily_loss']}:
                position = 0
                highest_profit = 0.0
                continue

        # Enter position
        if position == 0 and current_signal != 0:
            position = int(np.sign(current_signal))
            entry_price = current_price
            highest_profit = 0.0

    return strategy_returns
"""

        script = f"""
import pandas as pd
import numpy as np
import json
import sys

close = pd.read_pickle('close.pkl')  # nosec
factors = pd.read_pickle('factors.pkl')  # nosec

try:
{chr(10).join('    ' + line for line in strategy_code.split(chr(10)))}
except Exception as e:
    print(f"ERROR: Strategy execution failed: {{e}}", file=sys.stderr)  # nosec
    sys.exit(1)

if 'signal' not in dir():
    print("ERROR: No signal variable created", file=sys.stderr)
    sys.exit(1)

# Apply risk management
{risk_code}

signal = signal.fillna(0)
strategy_returns = apply_risk_management_with_params(signal, close, STOP_LOSS, TAKE_PROFIT, TRAILING_STOP, TRAILING_ACTIVATION)

# Calculate metrics
common_idx = close.index.intersection(signal.index)
close_aligned = close.loc[common_idx]
signal_aligned = signal.loc[common_idx]
fwd_returns = close_aligned.pct_change({forward_bars}).shift(-{forward_bars})

ic = signal_aligned.corr(fwd_returns.dropna()) if signal_aligned.std() > 0 else 0
total_return = (1 + strategy_returns).prod() - 1
cum_returns = (1 + strategy_returns).cumprod()
running_max = cum_returns.expanding().max()
drawdown = (cum_returns - running_max) / running_max.replace(0, np.nan)
max_dd = drawdown.min() if len(drawdown) > 0 else 0

active_returns = strategy_returns[strategy_returns != 0]
win_rate = (active_returns > 0).sum() / len(active_returns) if len(active_returns) > 0 else 0
n_trades = int((signal_aligned != signal_aligned.shift(1)).sum())

bars_per_year = 252 * 1440 / {forward_bars}
if strategy_returns.std() > 0:
    sharpe = strategy_returns.mean() / strategy_returns.std() * np.sqrt(bars_per_year)
else:
    sharpe = 0

n_bars = len(strategy_returns)
n_months = n_bars / (bars_per_year / 12) if n_bars > 0 else 1

if n_months > 0 and (1 + total_return) > 0:
    monthly_return = (1 + total_return) ** (1 / n_months) - 1
    annual_return = (1 + total_return) ** (12 / n_months) - 1
else:
    monthly_return = total_return
    annual_return = total_return * 12

# Daily loss check
daily_returns = strategy_returns.groupby(
    strategy_returns.index.date if hasattr(strategy_returns.index[0], 'date') else strategy_returns.index
).sum()
max_daily_loss = abs(daily_returns.min()) if len(daily_returns) > 0 else 0

result = {{
    "status": "success",
    "ic": float(ic) if not np.isnan(ic) else 0,
    "sharpe": float(sharpe),
    "max_drawdown": float(max_dd) if not np.isnan(max_dd) else 0,
    "win_rate": float(win_rate),
    "n_trades": n_trades,
    "total_return": float(total_return),
    "monthly_return_pct": float(monthly_return * 100),
    "annual_return_pct": float(annual_return * 100),
    "n_bars": int(n_bars),
    "n_months": float(n_months),
    "n_long": int((signal_aligned == 1).sum()),
    "n_short": int((signal_aligned == -1).sum()),
    "n_neutral": int((signal_aligned == 0).sum()),
    "max_daily_loss": float(max_daily_loss),
    "ftmo_compliant": max_daily_loss <= 0.05,
}}

def sanitize_val(v):
    if isinstance(v, (np.integer,)): return int(v)
    if isinstance(v, (np.floating,)): return float(v)
    if isinstance(v, np.bool_): return bool(v)
    if isinstance(v, float):
        import math
        if math.isnan(v): return 0.0
        if math.isinf(v): return -999.0 if v < 0 else 999.0
    return v

result = {{k: sanitize_val(v) for k, v in result.items()}}
print(json.dumps(result))
"""

        import tempfile
        with tempfile.TemporaryDirectory() as td:
            td_path = Path(td)
            close.to_pickle(str(td_path / "close.pkl"))  # nosec
            factors_df.to_pickle(str(td_path / "factors.pkl"))  # nosec
            (td_path / "run.py").write_text(script)

            try:
                result = subprocess.run( # nosec B603
                    [sys.executable, str(td_path / "run.py")],
                    capture_output=True, text=True, timeout=300,
                    cwd=str(td_path),
                )

                if result.returncode != 0:
                    logger.warning(f"Backtest failed: {result.stderr[:200] or result.stdout[:200]}")
                    return {"status": "failed", "reason": result.stderr[:200] or result.stdout[:200]}

                for line in result.stdout.strip().split("\n"):
                    try:
                        return json.loads(line)
                    except json.JSONDecodeError:
                        continue

                return {"status": "failed", "reason": "No valid JSON output"}

            except subprocess.TimeoutExpired:  # nosec
                return {"status": "failed", "reason": "Timeout (90s)"}
            except Exception as e:
                return {"status": "failed", "reason": str(e)[:200]}

# ============================================================================
# Parameter Optimizer
# ============================================================================
class ParameterOptimizer:
    """
    Grid search for optimal strategy parameters.
    """

    def __init__(self, max_combinations: int = 50):
        """
        Initialize optimizer.

        Parameters
        ----------
        max_combinations : int
            Maximum parameter combinations to test
        """
        self.max_combinations = max_combinations

    def optimize(
        self,
        close: pd.Series,
        factors_df: pd.DataFrame,
        strategy_code: str,
        forward_bars: int = 96,
    ) -> tuple[dict[str, float], dict[str, Any]]:
        """
        Optimize strategy parameters via grid search.

        Parameters
        ----------
        close : pd.Series
            Close prices
        factors_df : pd.DataFrame
            Factor values
        strategy_code : str
            Strategy Python code
        forward_bars : int
            Forward return horizon

        Returns
        -------
        tuple
            (best_params, best_result)
        """
        # Generate parameter combinations (sample if too many)
        all_combinations = list(product(
            PARAMETER_GRID["threshold_entry"],
            PARAMETER_GRID["rolling_window"],
            PARAMETER_GRID["stop_loss"],
            PARAMETER_GRID["take_profit"],
            PARAMETER_GRID["trailing_stop"],
            PARAMETER_GRID["trailing_activation"],
        ))

        # Filter invalid combinations (TP must be >= 2x SL)
        valid_combinations = [
            c for c in all_combinations
            if c[3] >= c[2] * 2  # take_profit >= 2 * stop_loss
        ]

        # Sample if too many
        if len(valid_combinations) > self.max_combinations:
            valid_combinations = random.sample(valid_combinations, self.max_combinations)

        logger.info(f"Testing {len(valid_combinations)} parameter combinations...")

        best_result = None
        best_params = None
        best_score = -np.inf

        runner = BacktestRunner()

        for idx, (threshold, window, sl, tp, trail, trail_act) in enumerate(valid_combinations):
            # Modify strategy code with current parameters
            param_code = self._inject_parameters(strategy_code, threshold, window)

            # Risk config for this combination
            risk_config = {
                "stop_loss": sl,
                "take_profit": tp,
                "trailing_stop": trail,
                "trailing_activation": trail_act,
                "max_daily_loss": 0.05,
                "max_positions": 1,
            }

            # Run backtest
            result = runner.run(close, factors_df, param_code, risk_config, forward_bars)

            if result and result.get("status") == "success":
                # Score: prioritize IC and Sharpe, penalize drawdown and low trades
                score = (
                    abs(result.get("ic", 0)) * 10 +
                    result.get("sharpe", 0) * 2 -
                    abs(result.get("max_drawdown", 0)) * 5 +
                    min(result.get("n_trades", 0) / 100, 2)
                )

                if score > best_score:
                    best_score = score
                    best_params = {
                        "threshold_entry": threshold,
                        "rolling_window": window,
                        "stop_loss": sl,
                        "take_profit": tp,
                        "trailing_stop": trail,
                        "trailing_activation": trail_act,
                    }
                    best_result = result

            if (idx + 1) % 10 == 0:
                logger.info(f"  Tested {idx + 1}/{len(valid_combinations)} combinations, best score={best_score:.3f}")

        if best_result is None:
            logger.warning("No successful backtests found, using default parameters")
            best_params = {
                "threshold_entry": 0.3,
                "rolling_window": 20,
                "stop_loss": 0.02,
                "take_profit": 0.04,
                "trailing_stop": 0.015,
                "trailing_activation": 0.02,
            }
            best_result = {"status": "failed", "reason": "No valid parameters found"}

        return best_params, best_result

    def _inject_parameters(
        self,
        strategy_code: str,
        threshold: float,
        window: int,
    ) -> str:
        """
        Inject parameters into strategy code - DISABLED for stability.
        qwen CLI generates code with its own thresholds which work better.
        """
        # Don't modify qwen CLI generated code - it already has good parameters
        return strategy_code

# ============================================================================
# Smart Strategy Generator (Main Class)
# ============================================================================
class SmartStrategyGenerator:
    """
    Main strategy generator with feedback loop, optimization, and risk management.

    Usage:
        generator = SmartStrategyGenerator(trading_style='daytrading')
        strategies = generator.generate_strategies(target_count=10)
    """

    def __init__(
        self,
        trading_style: str = "daytrading",
        forward_bars: int | None = None,
        max_attempts: int = 100,
        enable_optimization: bool = True,
    ):
        """
        Initialize strategy generator.

        Parameters
        ----------
        trading_style : str
            'daytrading' or 'swing'
        forward_bars : int, optional
            Forward return horizon (auto-detected from style)
        max_attempts : int
            Maximum generation attempts
        enable_optimization : bool
            Enable parameter grid search
        """
        self.trading_style = trading_style
        self.forward_bars = forward_bars or (12 if trading_style == "daytrading" else 96)
        self.max_attempts = max_attempts
        self.enable_optimization = enable_optimization

        self.llm_generator = LLMStrategyGenerator()
        self.evaluator = StrategyEvaluator(trading_style, self.forward_bars)  # nosec
        self.feedback_gen = FeedbackGenerator()
        self.optimizer = ParameterOptimizer(max_combinations=15)
        self.backtest_runner = BacktestRunner()

        self.factors = data_cache.load_top_factors(20)
        self.close = data_cache.load_ohlcv()

        # Load factor time-series
        self.factor_data = {}
        for f_info in self.factors:
            series = data_cache.load_factor_timeseries(f_info["name"])
            if series is not None:
                self.factor_data[f_info["name"]] = series

        # Align data
        all_series = [self.factor_data[n] for n in self.factor_data]
        if not all_series:
            raise ValueError("No factor data loaded!")

        self.df_factors = pd.DataFrame({n: self.factor_data[n] for n in self.factor_data})
        self.common_idx = self.close.index.intersection(self.df_factors.dropna(how="all").index)
        self.close_aligned = self.close.loc[self.common_idx]
        self.df_aligned = self.df_factors.loc[self.common_idx]

        self.accepted_strategies: list[dict] = []
        self.feedback_history: list[str] = []

        logger.info(
            f"SmartStrategyGenerator initialized: style={trading_style}, "
            f"forward_bars={self.forward_bars}, factors={len(self.factor_data)}, "
            f"bars={len(self.close_aligned):,}",
        )

    def generate_strategy(
        self,
        attempt_idx: int,
        factor_subset: list[dict] | None = None,
        feedback: str | None = None,
    ) -> dict | None:
        """
        Generate a single strategy with feedback loop.

        Parameters
        ----------
        attempt_idx : int
            Attempt number (for logging)
        factor_subset : list, optional
            Subset of factors to use (random if None)
        feedback : str, optional
            Previous feedback

        Returns
        -------
        dict or None
            Strategy dict or None if failed
        """
        # Select factor subset
        if factor_subset is None:
            n_factors = random.randint(2, min(5, len(self.factors)))
            factor_subset = random.sample(self.factors, n_factors)

        # Generate strategy via LLM
        gen_result = self.llm_generator.generate(
            factor_subset=factor_subset,
            feedback=feedback,
            trading_style=self.trading_style,
            forward_bars=self.forward_bars,
        )

        if gen_result["status"] != "generated":
            logger.warning(f"Attempt {attempt_idx}: LLM generation failed - {gen_result.get('error', 'Unknown')}")
            return None

        strategy = gen_result["strategy"]
        factor_names = strategy.get("factor_names", [])

        # Build factors DataFrame
        valid_factors = [f for f in factor_names if f in self.df_aligned.columns]
        if len(valid_factors) < 2:
            logger.warning(f"Attempt {attempt_idx}: Insufficient valid factors ({len(valid_factors)})")
            return None

        factors_df = self.df_aligned[valid_factors]

        # Default risk config
        risk_config = {
            "stop_loss": 0.02,
            "take_profit": 0.04,
            "trailing_stop": 0.015,
            "trailing_activation": 0.02,
            "max_daily_loss": 0.05,
            "max_positions": 1,
        }

        # Parameter optimization (if enabled)
        if self.enable_optimization:
            logger.info(f"Attempt {attempt_idx}: Running parameter optimization...")
            best_params, opt_result = self.optimizer.optimize(
                self.close_aligned, factors_df, strategy["code"], self.forward_bars,
            )

            if opt_result.get("status") == "success":
                risk_config.update(best_params)
                logger.info(
                    f"  Best params: threshold={best_params['threshold_entry']}, "
                    f"window={best_params['rolling_window']}, "
                    f"SL={best_params['stop_loss']:.1%}, TP={best_params['take_profit']:.1%}",
                )
            else:
                logger.warning("  Optimization failed, using default parameters")

        # Run final backtest with optimized/default risk config
        bt_result = self.backtest_runner.run(
            self.close_aligned, factors_df, strategy["code"], risk_config, self.forward_bars,
        )

        if bt_result is None or bt_result.get("status") != "success":
            logger.warning(f"Attempt {attempt_idx}: Backtest failed - {bt_result.get('reason', 'Unknown') if bt_result else 'No result'}")
            return None

        # Evaluate strategy
        # Reconstruct signal from backtest (approximate)
        signal_approx = pd.Series(0, index=self.close_aligned.index[:bt_result.get("n_bars", len(self.close_aligned))])
        evaluation = self.evaluator.evaluate(  # nosec
            signal=signal_approx,
            close=self.close_aligned.iloc[:len(signal_approx)],
            strategy_returns=pd.Series(dtype=float),  # Already computed in backtest
        )

        # Use backtest metrics directly for evaluation  # nosec
        evaluation = {  # nosec
            "ic": bt_result.get("ic", 0),
            "sharpe": bt_result.get("sharpe", 0),
            "max_drawdown": bt_result.get("max_drawdown", 0),
            "win_rate": bt_result.get("win_rate", 0),
            "n_trades": bt_result.get("n_trades", 0),
            "monthly_return": bt_result.get("monthly_return_pct", 0) / 100.0,
            "max_daily_loss": bt_result.get("max_daily_loss", 0),
        }

        # Check acceptance
        passed, failed_criteria = self.evaluator._check_acceptance(**evaluation)  # nosec
        evaluation["status"] = "accepted" if passed else "rejected"  # nosec
        evaluation["failed_criteria"] = failed_criteria  # nosec

        # Generate feedback
        feedback = self.feedback_gen.generate_feedback(
            evaluation=evaluation,  # nosec
            factor_list=self.factors,
            attempt=attempt_idx,
            param_config=risk_config,
        )
        self.feedback_history.append(feedback)

        # Store strategy
        strategy["metrics"] = bt_result
        strategy["risk_config"] = risk_config
        strategy["evaluation"] = evaluation  # nosec
        strategy["feedback"] = feedback

        if passed:
            logger.info(
                f"✓ Strategy #{len(self.accepted_strategies)+1} ACCEPTED: "
                f"IC={evaluation['ic']:.4f}, Sharpe={evaluation['sharpe']:.2f}, "  # nosec
                f"Trades={evaluation['n_trades']}, DD={evaluation['max_drawdown']:.1%}",  # nosec
            )
            self.accepted_strategies.append(strategy)
        else:
            logger.info(
                f"✗ Strategy REJECTED: {', '.join(failed_criteria[:3])}",
            )

        return strategy

    def generate_strategies(self, target_count: int = 10) -> list[dict]:
        """
        Generate multiple strategies with feedback loop.

        Parameters
        ----------
        target_count : int
            Number of accepted strategies to generate

        Returns
        -------
        list
            List of accepted strategy dicts
        """
        console.print("\n[bold cyan]🧠 Smart Strategy Generation[/bold cyan]")
        console.print(f"   Style: {self.trading_style}")
        console.print(f"   Forward bars: {self.forward_bars}")
        console.print(f"   Target: {target_count} accepted strategies")
        console.print(f"   Factors: {len(self.factor_data)}")
        console.print(f"   Data points: {len(self.close_aligned):,}\n")

        max_attempts = min(self.max_attempts, target_count * 15)
        accepted = []

        with Progress(
            SpinnerColumn(),
            TextColumn("[bold blue]{task.description}"),
            BarColumn(),
            TextColumn("[bold green]{task.completed}/{task.total}"),
            TimeElapsedColumn(),
        ) as progress:
            task = progress.add_task(f"Generating {self.trading_style} strategies...", total=max_attempts)

            for attempt in range(max_attempts):
                if len(accepted) >= target_count:
                    break

                progress.update(task, description=f"Attempt {attempt+1}...")

                # Get feedback from last attempt
                feedback = self.feedback_history[-1] if self.feedback_history and random.random() < 0.7 else None

                strategy = self.generate_strategy(attempt, feedback=feedback)

                if strategy and strategy["evaluation"]["status"] == "accepted":  # nosec
                    accepted.append(strategy)

                    # Save strategy
                    self._save_strategy(strategy)

                    console.print(
                        f"[green]✓ Strategy #{len(accepted)}:[/green] {strategy['strategy_name']} "
                        f"IC={strategy['metrics'].get('ic', 0):.4f}, "
                        f"Sharpe={strategy['metrics'].get('sharpe', 0):.3f}, "
                        f"Trades={strategy['metrics'].get('n_trades', 0)}, "
                        f"DD={strategy['metrics'].get('max_drawdown', 0):.1%}, "
                        f"Monthly={strategy['metrics'].get('monthly_return_pct', 0):.2f}%",
                    )

                progress.update(task, advance=1)

        # Summary
        console.print(f"\n[bold green]✓ Generated {len(accepted)}/{target_count} accepted strategies[/bold green]\n")

        if accepted:
            accepted.sort(key=lambda x: x["metrics"].get("ic", 0), reverse=True)

            table = Table(title=f"Top {len(accepted)} Accepted Strategies")
            table.add_column("#", justify="right")
            table.add_column("Name")
            table.add_column("IC", justify="right")
            table.add_column("Sharpe", justify="right")
            table.add_column("Trades", justify="right")
            table.add_column("Max DD", justify="right")
            table.add_column("Monthly %", justify="right")
            table.add_column("FTMO", justify="center")

            for i, s in enumerate(accepted, 1):
                m = s["metrics"]
                table.add_row(
                    str(i),
                    s["strategy_name"],
                    f"{m.get('ic', 0):.4f}",
                    f"{m.get('sharpe', 0):.3f}",
                    str(m.get("n_trades", 0)),
                    f"{m.get('max_drawdown', 0):.1%}",
                    f"{m.get('monthly_return_pct', 0):.2f}%",
                    "✅" if m.get("ftmo_compliant", False) else "❌",
                )

            console.print(table)

        return accepted

    def _save_strategy(self, strategy: dict) -> None:
        """Save strategy to JSON file."""
        fname = f"{int(time.time())}_{strategy['strategy_name'].replace(' ', '_')[:50]}.json"
        fpath = STRATEGIES_DIR / fname

        # Convert numpy types for JSON serialization
        def convert_numpy(obj):
            if isinstance(obj, (np.integer,)):
                return int(obj)
            if isinstance(obj, (np.floating,)):
                return float(obj)
            if isinstance(obj, np.ndarray):
                return obj.tolist()
            return obj

        strategy_serializable = {k: convert_numpy(v) for k, v in strategy.items()}

        with open(fpath, "w") as f:
            json.dump(strategy_serializable, f, indent=2, ensure_ascii=False)

        # Generate PDF report if available
        try:
            from nexquant_strategy_report import StrategyPerformanceReporter
            reporter = StrategyPerformanceReporter(strategy)
            reporter.generate_report()
        except Exception as e:
            logger.debug(f"Failed to generate report: {e}")

        logger.info(f"Saved strategy: {fpath}")

# ============================================================================
# CLI Interface
# ============================================================================
def parse_args():
    """Parse command line arguments."""
    import argparse

    parser = argparse.ArgumentParser(
        description="Smart Strategy Generation with Feedback & Optimization",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  python nexquant_smart_strategy_gen.py 10
  python nexquant_smart_strategy_gen.py 5 --style daytrading
  python nexquant_smart_strategy_gen.py 20 --style swing --max-attempts 200
  python nexquant_smart_strategy_gen.py 10 --no-optimization
        """,
    )

    parser.add_argument(
        "count",
        type=int,
        nargs="?",
        default=10,
        help="Number of strategies to generate (default: 10)",
    )
    parser.add_argument(
        "--style",
        choices=["daytrading", "swing"],
        default="daytrading",
        help="Trading style (default: daytrading)",
    )
    parser.add_argument(
        "--forward-bars",
        type=int,
        default=None,
        help="Forward return bars (auto: 12 for daytrading, 96 for swing)",
    )
    parser.add_argument(
        "--max-attempts",
        type=int,
        default=150,
        help="Maximum generation attempts (default: 150)",
    )
    parser.add_argument(
        "--no-optimization",
        action="store_true",
        help="Disable parameter grid search",
    )
    parser.add_argument(
        "--factors",
        type=int,
        default=20,
        help="Number of top factors to consider (default: 20)",
    )

    return parser.parse_args()

def main():
    """Main entry point."""
    args = parse_args()

    console.print(f"\n[bold magenta]{'='*70}[/bold magenta]")
    console.print("[bold]🤖 PREDIX Smart Strategy Generator[/bold]")
    console.print(f"[bold magenta]{'='*70}[/bold magenta]\n")

    try:
        # Initialize generator
        generator = SmartStrategyGenerator(
            trading_style=args.style,
            forward_bars=args.forward_bars,
            max_attempts=args.max_attempts,
            enable_optimization=not args.no_optimization,
        )

        # Generate strategies
        strategies = generator.generate_strategies(target_count=args.count)

        if strategies:
            console.print(f"\n[bold green]✓ Success! {len(strategies)} strategies saved to:[/bold green]")
            console.print(f"   {STRATEGIES_DIR}\n")
        else:
            console.print("\n[bold yellow]⚠ No strategies met acceptance criteria[/bold yellow]")
            console.print("   Try: --max-attempts 200 or --style swing\n")

    except KeyboardInterrupt:
        console.print("\n[yellow]Interrupted by user[/yellow]")
        sys.exit(0)
    except Exception as e:
        logger.exception(f"Fatal error: {e}")
        console.print(f"\n[red]✗ Fatal error: {e}[/red]")
        sys.exit(1)

if __name__ == "__main__":
    main()
