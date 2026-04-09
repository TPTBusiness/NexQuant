"""
Predix Strategy Orchestrator - Generate trading strategies from factors.

This module:
1. Loads top evaluated factors from the results database
2. Generates LLM-powered trading strategy code
3. Evaluates strategies using real OHLCV backtest
4. Accepts/rejects based on performance thresholds
5. Saves accepted strategies as JSON files

Usage:
    orchestrator = StrategyOrchestrator(
        top_factors=20,
        trading_style='swing',
        min_sharpe=1.5,
        max_drawdown=-0.20,
    )
    results = orchestrator.generate_strategies(count=10, workers=4)
"""

import json
import logging
import os
import time
import traceback
from concurrent.futures import ThreadPoolExecutor, as_completed
from datetime import datetime
from pathlib import Path
from typing import Any, Dict, List, Optional

import numpy as np
import pandas as pd

from rdagent.components.prompt_loader import load_prompt

# OHLCV data path
OHLCV_PATH = Path(os.getenv(
    'PREDIX_OHLCV_PATH',
    '/home/nico/Predix/git_ignore_folder/factor_implementation_source_data/intraday_pv.h5'
))
from rdagent.log import rdagent_logger as logger

logger = logging.getLogger(__name__)


class StrategyOrchestrator:
    """
    Orchestrates strategy generation from evaluated factors.

    Uses LLM to generate strategy code from factor combinations,
    then evaluates each strategy using real OHLCV backtest data.
    """

    def __init__(
        self,
        top_factors: int = 20,
        trading_style: str = "swing",
        min_sharpe: float = 1.5,
        max_drawdown: float = -0.20,
        min_win_rate: float = 0.50,
        results_dir: Optional[str] = None,
    ):
        """
        Parameters
        ----------
        top_factors : int
            Number of top factors to consider for strategy generation
        trading_style : str
            Trading style: 'daytrading' or 'swing'
        min_sharpe : float
            Minimum Sharpe ratio for strategy acceptance
        max_drawdown : float
            Maximum allowed drawdown (negative value)
        min_win_rate : float
            Minimum win rate for strategy acceptance
        results_dir : str, optional
            Path to results directory
        """
        self.top_factors = top_factors
        self.trading_style = trading_style.lower()
        self.min_sharpe = min_sharpe
        self.max_drawdown = max_drawdown
        self.min_win_rate = min_win_rate

        if results_dir is None:
            project_root = Path(__file__).parent.parent.parent.parent
            self.results_dir = project_root / "results"
        else:
            self.results_dir = Path(results_dir)

        self.strategies_dir = self.results_dir / "strategies_new"
        self.strategies_dir.mkdir(parents=True, exist_ok=True)

        self.factors_dir = self.results_dir / "factors"
        self.values_dir = self.factors_dir / "values"

        # Load prompt for strategy generation
        try:
            self.strategy_prompt = load_prompt("strategy_generation")
        except Exception:
            self.strategy_prompt = None
            logger.warning("Strategy generation prompt not found. Using fallback template.")

        logger.info(
            f"StrategyOrchestrator initialized: style={self.trading_style}, "
            f"top_factors={self.top_factors}, min_sharpe={self.min_sharpe}"
        )

    def load_ohlcv_close(self) -> pd.Series:
        """Load OHLCV close prices from HDF5 file."""
        if not OHLCV_PATH.exists():
            logger.warning(f"OHLCV data not found: {OHLCV_PATH}")
            return None
        
        try:
            ohlcv = pd.read_hdf(str(OHLCV_PATH), key='data')
            if '$close' in ohlcv.columns:
                close = ohlcv['$close'].dropna()
            elif 'close' in ohlcv.columns:
                close = ohlcv['close'].dropna()
            else:
                close = ohlcv.select_dtypes(include=[np.number]).iloc[:, 0].dropna()
            
            # Handle MultiIndex
            if isinstance(close.index, pd.MultiIndex):
                try:
                    close = close.xs('EURUSD', level='instrument')
                except KeyError:
                    idx = close.index.get_level_values('instrument') == 'EURUSD'
                    close = close[idx]
                    close.index = close.index.droplevel('instrument')
            
            return close
        except Exception as e:
            logger.warning(f"Failed to load OHLCV data: {e}")
            return None

    def load_top_factors(self) -> List[Dict[str, Any]]:
        """
        Load top evaluated factors from JSON files.

        Returns
        -------
        List[Dict[str, Any]]
            List of factor info dicts sorted by IC
        """
        if not self.factors_dir.exists():
            logger.warning(f"Factors directory not found: {self.factors_dir}")
            return []

        factors = []
        for f in self.factors_dir.glob("*.json"):
            try:
                with open(f, encoding="utf-8") as fh:
                    data = json.load(fh)
                if data.get("status") == "success" and data.get("ic") is not None:
                    data["_source_file"] = str(f)
                    factors.append(data)
            except Exception as e:
                logger.debug(f"Failed to load {f}: {e}")
                continue

        # Sort by absolute IC and take top N
        factors.sort(key=lambda x: abs(x.get("ic", 0) or 0), reverse=True)
        
        # Filter to only include factors that have parquet files
        factors_with_files = []
        for f in factors:
            fname = f.get("factor_name", "")
            safe = fname.replace("/", "_").replace("\\", "_").replace(" ", "_")[:100]
            pf = self.values_dir / f"{safe}.parquet"
            if pf.exists():
                factors_with_files.append(f)
            else:
                logger.debug(f"Skipping {fname} - no parquet file")
        
        return factors_with_files[: self.top_factors]

    def load_factor_values(self, factor_name: str) -> Optional[pd.Series]:
        """
        Load factor time-series values from parquet file.

        Parameters
        ----------
        factor_name : str
            Name of the factor

        Returns
        -------
        pd.Series or None
            Factor values indexed by timestamp
        """
        safe_name = factor_name.replace("/", "_").replace("\\", "_").replace(" ", "_")[:100]
        parquet_path = self.values_dir / f"{safe_name}.parquet"

        if not parquet_path.exists():
            return None

        try:
            df = pd.read_parquet(str(parquet_path))
            # Handle MultiIndex (datetime, instrument)
            if isinstance(df.index, pd.MultiIndex):
                # Get the factor column name (should be the only column)
                factor_col = df.columns[0]
                # Extract EURUSD series
                try:
                    series = df.xs('EURUSD', level='instrument')[factor_col]
                except KeyError:
                    # Try alternative extraction
                    df_reset = df.reset_index()
                    if 'instrument' in df_reset.columns:
                        df_eur = df_reset[df_reset['instrument'] == 'EURUSD'].set_index('datetime')
                        series = df_eur[factor_col] if factor_col in df_eur.columns else df_eur.iloc[:, -1]
                    else:
                        series = df.iloc[:, 0]
            else:
                series = df.iloc[:, 0]
            
            # Ensure numeric
            series = pd.to_numeric(series, errors='coerce')
            series.name = factor_name
            return series
        except Exception as e:
            logger.warning(f"Failed to load factor values for {factor_name}: {e}")
            return None

    def generate_strategy_code(self, factors: List[Dict[str, Any]], strategy_name: str) -> Optional[str]:
        """
        Generate strategy code using LLM from factor combinations.

        Parameters
        ----------
        factors : List[Dict[str, Any]]
            List of factor info dicts to combine
        strategy_name : str
            Name for the generated strategy

        Returns
        -------
        str or None
            Generated Python strategy code
        """
        factor_names = [f["factor_name"] for f in factors]
        factor_ics = {f["factor_name"]: f.get("ic", 0) for f in factors}

        # Build prompt context
        context = {
            "strategy_name": strategy_name,
            "factor_names": factor_names,
            "factor_ics": factor_ics,
            "trading_style": self.trading_style,
            "min_sharpe": self.min_sharpe,
            "max_drawdown": self.max_drawdown,
            "system_prompt": self.strategy_prompt.get("system", "") if isinstance(self.strategy_prompt, dict) else "",
            "user_prompt": self.strategy_prompt.get("user", "").replace("{{ factors }}", str(factor_ics)).replace("{{ additional_context }}", f"Strategy name: {strategy_name}") if isinstance(self.strategy_prompt, dict) else "",
        }

        # Try LLM first
        if self.strategy_prompt is not None:
            try:
                code = self._generate_with_llm(context)
                if code:
                    return code
            except Exception as e:
                logger.warning(f"LLM strategy generation failed: {e}")

        # Fallback: generate template code programmatically
        return self._generate_fallback_code(context)

    def _generate_with_llm(self, context: Dict[str, Any]) -> Optional[str]:
        """Generate strategy code using LLM."""
        import os
        import requests
        
        # Use local llama.cpp server (running on port 8081)
        api_url = "http://localhost:8081/v1"
        api_key = "local"
        model = ""
        
        headers = {
            "Authorization": f"Bearer {api_key}",
            "Content-Type": "application/json",
        }
        
        payload = {
            
            "model": "Qwen3.5-35B-A3B-Q3_K_M.gguf",
            "messages": [
                {"role": "system", "content": context.get("system_prompt", "")},
                {"role": "user", "content": context.get("user_prompt", "")},
            ],
            "max_tokens": 4096,
            "temperature": 0.5,
            "include_reasoning": False,
        }
        
        # Build API URL
        api_base = api_url.rstrip("/")
        if not api_base.endswith("/v1"):
            api_base = f"{api_base}/v1"
        api_endpoint = f"{api_base}/chat/completions"
        
        response = requests.post(
            api_endpoint,
            headers=headers,
            json=payload,
            timeout=120,
        )
        
        if response.status_code != 200:
            logger.warning(f"LLM API error: {response.text[:200]}")
            return None
        
        data = response.json()
        message = data.get("choices", [{}])[0].get("message", {})
        content = message.get("content", "") or message.get("reasoning_content", "")
        
        if not content:
            # Try fallback: some models put content in different fields
            content = data.get("output", "") or data.get("text", "")
            if not content:
                logger.warning(f"LLM returned empty response. Model: {model}, Full response: {str(data)[:500]}")
                return None
        
        # Debug: log what we got}, first 100 chars: {content[:100]}")
        
        code = content.strip()
        
        # Extract code from markdown blocks or reasoning content
        import re
        from collections import Counter
        
        # First try to find code between ``` markers
        if "```" in code:
            match = re.search(r'```python\s*\n(.*?)\n```', code, re.DOTALL)
            if match:
                code = match.group(1)
            else:
                match = re.search(r'```\s*\n(.*?)\n```', code, re.DOTALL)
                if match:
                    code = match.group(1)
        
        # If code has indent from reasoning, dedent it
        if code:
            # Find code before first ``` if present
            match = re.search(r'^(.*?)(?:```)', code, re.DOTALL)
            if match:
                code = match.group(1).strip()
            
            # Smart dedent: find most common indent
            lines = code.split('\n')
            indents = Counter()
            for line in lines:
                if line.strip():
                    indent = len(line) - len(line.lstrip())
                    indents[indent] += 1
            
            if len(indents) > 1 and indents.get(0, 0) <= 1:
                indents.pop(0, None)
            
            if indents:
                common_indent = indents.most_common(1)[0][0]
            else:
                common_indent = 0
            
            dedented = []
            for line in lines:
                if len(line) >= common_indent and line[:common_indent].isspace():
                    dedented.append(line[common_indent:])
                else:
                    dedented.append(line.lstrip())
            
            code = '\n'.join(dedented).strip()
            
            # Remove non-code lines (bullets, commentary after code)
            final_lines = []
            for line in code.split('\n'):
                stripped = line.strip()
                if not stripped:
                    continue
                if stripped.startswith('*') or stripped.startswith('\u2022') or stripped.startswith('Wait') or stripped.startswith('Also') or stripped.startswith('One more'):
                    break
                final_lines.append(line)
            
            code = '\n'.join(final_lines).strip()
        
        # Remove non-ASCII (emojis etc)
        code = code.encode('ascii', 'ignore').decode('ascii').strip()
        
        
        if not code:
            logger.warning("LLM returned empty code after cleaning")
            return None
        
        # Try to parse as JSON and extract code field
        import json
        if code.startswith('{'):
            try:
                data = json.loads(code)
                # Extract code from JSON response
                if 'code' in data:
                    code = data['code']
                    logger.info(f"Extracted code from JSON response ({len(code)} chars)")
                elif 'strategy_code' in data:
                    code = data['strategy_code']
                    logger.info(f"Extracted strategy_code from JSON response ({len(code)} chars)")
            except json.JSONDecodeError:
                pass  # Not valid JSON, treat as raw code
        
        # Validate it's valid Python}")
        try:
            compile(code, "<strategy>", "exec")
            
            return code
        except SyntaxError as e:
            logger.warning(f"LLM generated invalid Python code: {e}")
            logger.warning(f"Code was: {code[:500]}")
            return None
        
        system_prompt = """You are an expert quantitative trading developer.
Generate a complete Python trading strategy that:
1. Takes factor values as input
2. Produces trading signals (1=LONG, -1=SHORT, 0=NEUTRAL)
3. Includes proper risk management
4. Uses the provided factors optimally

The strategy code will be executed with a 'factors' DataFrame available in scope.
Output ONLY valid Python code, no markdown formatting."""

        user_prompt = f"""Generate a {context['trading_style']} trading strategy named '{context['strategy_name']}'.

Factors to use (with IC scores):
{json.dumps(context['factor_ics'], indent=2)}

Requirements:
- The strategy must output a 'signal' variable (1, -1, or 0)
- Use z-score normalization for factor combination
- Include entry/exit logic based on signal thresholds
- Add risk management: position sizing, stop loss awareness
- Target Sharpe ratio > {context['min_sharpe']}
- Maximum drawdown tolerance: {context['max_drawdown']}

Output the complete strategy code."""

        code = api.build_messages_and_create_chat_completion(
            user_prompt=user_prompt,
            system_prompt=system_prompt,
            json_mode=False,
        ).strip()

        # Remove markdown code blocks if present
        code = code.replace("```python\n", "").replace("```", "").strip()

        # Validate it's valid Python
        try:
            compile(code, "<strategy>", "exec")
            return code
        except SyntaxError:
            logger.warning("LLM generated invalid Python code")
            return None

    def _generate_fallback_code(self, context: Dict[str, Any]) -> str:
        """Generate fallback strategy code programmatically."""
        factor_names = context["factor_names"]
        style_config = "daytrading" if context["trading_style"] == "daytrading" else "swing"

        # Build factor assignment code
        factor_assignments = "\n    ".join(
            [f'"{name}": factors["{name}"]' for name in factor_names if name != "timestamp"]
        )

        code = f'''"""
{context['strategy_name']} - {style_config.title()} Strategy
Auto-generated by Predix Strategy Orchestrator
Factors: {', '.join(factor_names)}
"""
import numpy as np
import pandas as pd

# Strategy configuration
STRATEGY_NAME = "{context['strategy_name']}"
TRADING_STYLE = "{style_config}"
FACTOR_NAMES = {json.dumps(factor_names)}

# Calculate combined signal
factor_data = pd.DataFrame({{
    {factor_assignments}
}})

# Normalize factors to z-scores
factor_norm = (factor_data - factor_data.mean()) / factor_data.std()

# Weighted combination (weight by IC)
weights = np.array([{", ".join([str(abs(context["factor_ics"].get(n, 0.01))) for n in factor_names if n != "timestamp"])}])
weights = weights / weights.sum()

combined_signal = (factor_norm * weights).sum(axis=1)

# Generate trading signals
# Entry: signal crosses above/below threshold
# Exit: signal crosses back toward zero
entry_threshold = 0.5
exit_threshold = 0.2

signal = pd.Series(0, index=combined_signal.index)
signal[combined_signal > entry_threshold] = 1
signal[combined_signal < -entry_threshold] = -1
signal[abs(combined_signal) < exit_threshold] = 0

# Smooth signals to reduce turnover
signal = signal.rolling(window=3, min_periods=1).mean().round().astype(int)
'''
        return code

    def evaluate_strategy(
        self, strategy_code: str, strategy_name: str, factors: List[Dict[str, Any]]
    ) -> Dict[str, Any]:
        """
        Evaluate a strategy by executing its code and calculating metrics.

        Parameters
        ----------
        strategy_code : str
            Python strategy code to execute
        strategy_name : str
            Name of the strategy
        factors : List[Dict[str, Any]]
            List of factor info dicts used by this strategy

        Returns
        -------
        Dict[str, Any]
            Strategy evaluation metrics
        """
        try:
            # Load factor values
            factor_names = [f["factor_name"] for f in factors if f["factor_name"] != "timestamp"]
            factor_values = {}

            for fname in factor_names:
                series = self.load_factor_values(fname)
                if series is not None:
                    factor_values[fname] = series

            if not factor_values:
                return {
                    "strategy_name": strategy_name,
                    "status": "rejected",
                    "reason": "No factor values available",
                    "factors_used": factor_names,
                }

            # Align factor values with common index
            if not factor_values:
                df_factors = pd.DataFrame()
            else:
                # Find common index across all series
                common_idx = None
                for name, s in factor_values.items():
                    if common_idx is None:
                        common_idx = s.index
                    else:
                        common_idx = common_idx.intersection(s.index)
                
                if common_idx is not None and len(common_idx) > 100:
                    df_factors = pd.DataFrame({
                        name: s.reindex(common_idx) for name, s in factor_values.items()
                    }).dropna()
                else:
                    df_factors = pd.DataFrame()

            if len(df_factors) < 100:
                return {
                    "strategy_name": strategy_name,
                    "status": "rejected",
                    "reason": "Insufficient aligned data",
                    "factors_used": factor_names,
                }

            # Convert all factor columns to numeric
            for col in df_factors.columns:
                df_factors[col] = pd.to_numeric(df_factors[col], errors='coerce')
            df_factors = df_factors.dropna()
            
            if len(df_factors) < 100:
                return {
                    "strategy_name": strategy_name,
                    "status": "rejected",
                    "reason": "Insufficient numeric data after conversion",
                    "factors_used": factor_names,
                }
            
            # Load OHLCV close prices for strategies that need them
            close = self.load_ohlcv_close()
            if close is not None:
                # Reindex close to match factor index
                close = close.reindex(df_factors.index).ffill()
            
            # Execute strategy code with factor data and close prices
            local_vars = {"factors": df_factors}
            if close is not None:
                local_vars["close"] = close
            
            try:
                exec(strategy_code, {"np": np, "pd": pd, "numpy": np}, local_vars)
            except Exception as e:
                return {
                    "strategy_name": strategy_name,
                    "status": "rejected",
                    "reason": f"Code execution error: {str(e)}",
                    "factors_used": factor_names,
                }

            if "signal" not in local_vars:
                return {
                    "strategy_name": strategy_name,
                    "status": "rejected",
                    "reason": "Strategy did not produce 'signal' variable",
                    "factors_used": factor_names,
                }

            signal = local_vars["signal"]
            
            # Debug: check signal distribution

            # Calculate REAL returns using OHLCV data
            close = self.load_ohlcv_close()
            
            if close is not None:
                # Use factor timestamps as the base (signal is generated on factor data)
                # Resample OHLCV close to factor timestamps
                signal_index = signal.index
                close_aligned = close.reindex(signal_index).ffill()
                
                # Calculate real price returns
                price_returns = close_aligned.pct_change().fillna(0)
                
                # Apply signal positions to real returns (lagged signal)
                signal_positions = signal.shift(1).fillna(0)
                returns = price_returns * signal_positions
                
                # Include spread costs (1.5 bps per trade = 0.00015)
                combined_factor = df_factors.mean(axis=1)
                SPREAD_COST = 0.00015
                signal_changes = signal_positions.diff().abs().fillna(0)
                spread_costs = signal_changes * SPREAD_COST
                returns = returns - spread_costs
            else:
                # Fallback: use factor proxy if OHLCV unavailable
                logger.warning("OHLCV data unavailable, using factor proxy")
                signal_positions = signal.shift(1).fillna(0)
                combined_factor = df_factors.mean(axis=1)
                return_proxy = combined_factor * 0.0001
                returns = return_proxy * signal_positions
            
            # Returns calculated

            if returns.std() == 0:
                return {
                    "strategy_name": strategy_name,
                    "status": "rejected",
                    "reason": "Zero return variance",
                    "factors_used": factor_names,
                }

            # Calculate metrics
            total_return = float(returns.sum())
            n_periods = len(returns)
            
            # Annualization for 1-minute data
            # 252 trading days * 1440 minutes per day = 362880 minutes per year
            minutes_per_year = 252 * 1440
            ann_factor = np.sqrt(minutes_per_year)  # ~602 for 1-min data
            
            # Calculate years of data (minimum 0.1 years = ~36 days to avoid extreme values)
            years = max(n_periods / minutes_per_year, 0.1) if n_periods > 0 else 0.1
            
            # Annualized return (compound, not linear)
            # For short periods, scale linearly to avoid extreme values
            if years >= 1 and (1 + total_return) > 0:
                ann_return = (1 + total_return) ** (1 / years) - 1
            else:
                # For < 1 year, linear scaling is more appropriate
                ann_return = total_return / years
            
            volatility = float(returns.std() * ann_factor)
            sharpe = ann_return / volatility if volatility > 0 else 0.0

            # Max drawdown
            # Handle any NaN/inf in returns
            returns = returns.fillna(0).replace([np.inf, -np.inf], 0)
            cum_returns = (1 + returns).cumprod()
            running_max = cum_returns.expanding().max()
            drawdown = (cum_returns - running_max) / running_max.replace(0, np.nan)
            drawdown = drawdown.fillna(0).replace([np.inf, -np.inf], 0)
            max_dd = float(drawdown.min()) if len(drawdown) > 0 else 0.0

            # Win rate
            signal_changes = signal.diff().fillna(0)
            trades = signal_changes[signal_changes != 0]
            win_rate = float((trades > 0).sum() / len(trades)) if len(trades) > 0 else 0.0

            # Information ratio (signal vs buy-and-hold)
            if close is not None:
                benchmark_returns = price_returns
            else:
                benchmark_returns = combined_factor.pct_change().fillna(0)
            excess_returns = returns - benchmark_returns
            if excess_returns.std() > 0:
                ir = float(excess_returns.mean() / excess_returns.std() * ann_factor)
            else:
                ir = 0.0

            metrics = {
                "strategy_name": strategy_name,
                "status": "accepted" if self._check_acceptance(sharpe, max_dd, win_rate) else "rejected",
                "sharpe_ratio": round(sharpe, 4),
                "annualized_return": round(ann_return, 6),
                "max_drawdown": round(max_dd, 6),
                "win_rate": round(win_rate, 4),
                "volatility": round(volatility, 6),
                "information_ratio": round(ir, 4),
                "total_return": round(total_return, 6),
                "num_periods": n_periods,
                "factors_used": factor_names,
                "trading_style": self.trading_style,
                "generated_at": datetime.now().isoformat(),
            }

            if metrics["status"] == "rejected":
                metrics["reason"] = self._get_rejection_reason(sharpe, max_dd, win_rate)

            return metrics

        except Exception as e:
            logger.error(f"Strategy evaluation failed for {strategy_name}: {e}")
            logger.debug(traceback.format_exc())
            return {
                "strategy_name": strategy_name,
                "status": "rejected",
                "reason": f"Evaluation error: {str(e)}",
                "factors_used": [],
            }

    def _check_acceptance(self, sharpe: float, max_dd: float, win_rate: float) -> bool:
        """Check if strategy meets acceptance criteria."""
        return sharpe >= self.min_sharpe and max_dd >= self.max_drawdown and win_rate >= self.min_win_rate

    def _get_rejection_reason(self, sharpe: float, max_dd: float, win_rate: float) -> str:
        """Get human-readable rejection reason."""
        reasons = []
        if sharpe < self.min_sharpe:
            reasons.append(f"Sharpe {sharpe:.2f} < {self.min_sharpe}")
        if max_dd < self.max_drawdown:
            reasons.append(f"Max DD {max_dd:.2%} < {self.max_drawdown:.2%}")
        if win_rate < self.min_win_rate:
            reasons.append(f"Win Rate {win_rate:.2%} < {self.min_win_rate:.2%}")
        return "; ".join(reasons) if reasons else "Unknown"

    def _generate_strategy_name(self, factors: List[Dict[str, Any]], idx: int) -> str:
        """Generate a strategy name from its factors."""
        # Extract key words from factor names
        words = []
        for f in factors:
            name = f["factor_name"]
            # Split on underscores and camelCase
            parts = name.replace("_", " ").split()
            for p in parts:
                # Extract capitalized words
                cap_words = [w for w in p.split() if w[0:1].isupper()]
                words.extend(cap_words if cap_words else [p])

        # Take up to 3 unique words
        unique_words = list(dict.fromkeys(words))[:3]
        if unique_words:
            return f"{''.join(unique_words)}_v{idx}"
        return f"Strategy_{idx}"

    def generate_strategies(
        self,
        count: int = 10,
        workers: int = 4,
        progress_callback=None,
    ) -> List[Dict[str, Any]]:
        """
        Generate and evaluate trading strategies.

        Parameters
        ----------
        count : int
            Number of strategies to generate
        workers : int
            Number of parallel workers
        progress_callback : callable, optional
            Callback function(current, total, result) for progress updates

        Returns
        -------
        List[Dict[str, Any]]
            List of strategy results (accepted and rejected)
        """
        # Load factors
        factors = self.load_top_factors()
        if not factors:
            logger.warning("No factors available for strategy generation")
            return []

        logger.info(f"Loaded {len(factors)} top factors for strategy generation")

        results = []
        strategies_generated = 0
        strategies_accepted = 0

        # Generate strategies using factor combinations
        strategy_configs = self._generate_strategy_configs(factors, count)

        # Execute strategies with thread pool
        with ThreadPoolExecutor(max_workers=workers) as executor:
            futures = {}

            for i, config in enumerate(strategy_configs):
                future = executor.submit(self._generate_and_evaluate_single, i, config)
                futures[future] = config

            for future in as_completed(futures):
                strategies_generated += 1
                try:
                    result = future.result()
                    results.append(result)

                    if result["status"] == "accepted":
                        strategies_accepted += 1
                        self._save_strategy(result)
                        logger.info(
                            f"Strategy ACCEPTED: {result['strategy_name']} | "
                            f"Sharpe={result['sharpe_ratio']:.2f} | "
                            f"DD={result['max_drawdown']:.2%}"
                        )
                    else:
                        logger.debug(
                            f"Strategy rejected: {result['strategy_name']} - {result.get('reason', 'unknown')}"
                        )

                    if progress_callback:
                        progress_callback(strategies_generated, len(strategy_configs), result)

                except Exception as e:
                    logger.error(f"Strategy generation failed: {e}")
                    results.append({
                        "strategy_name": f"Failed_{strategies_generated}",
                        "status": "rejected",
                        "reason": str(e),
                    })

        logger.info(
            f"Strategy generation complete: {strategies_accepted}/{strategies_generated} accepted "
            f"({strategies_accepted/max(strategies_generated,1)*100:.1f}%)"
        )

        return results

    def _generate_strategy_configs(self, factors: List[Dict], count: int) -> List[List[Dict]]:
        """
        Generate strategy configurations from factor combinations.

        Creates combinations of 2-4 factors, prioritizing high-IC factors
        and diversity across factor categories.
        """
        from itertools import combinations

        configs = []

        # Generate 2-factor combinations
        for combo in combinations(factors, 2):
            if len(configs) >= count * 2:  # Generate extras for rejection buffer
                break
            configs.append(list(combo))

        # Generate 3-factor combinations if needed
        if len(configs) < count and len(factors) >= 3:
            for combo in combinations(factors, 3):
                if len(configs) >= count * 2:
                    break
                configs.append(list(combo))

        # Shuffle to add randomness, then take what we need
        np.random.shuffle(configs)
        return configs[: count * 2]  # Generate extras

    def _generate_and_evaluate_single(self, idx: int, factors: List[Dict]) -> Dict[str, Any]:
        """Generate and evaluate a single strategy."""
        strategy_name = self._generate_strategy_name(factors, idx + 1)

        # Generate code
        code = self.generate_strategy_code(factors, strategy_name)
        if not code:
            return {
                "strategy_name": strategy_name,
                "status": "rejected",
                "reason": "Code generation failed",
            }

        # Evaluate
        result = self.evaluate_strategy(code, strategy_name, factors)
        result["code"] = code

        return result

    def _save_strategy(self, result: Dict[str, Any]) -> None:
        """Save accepted strategy to JSON file."""
        timestamp = int(time.time())
        safe_name = result["strategy_name"].replace("/", "_").replace(" ", "_")[:60]
        filename = f"{timestamp}_{safe_name}.json"
        filepath = self.strategies_dir / filename

        # Prepare serializable result
        save_data = {k: v for k, v in result.items() if k != "code"}
        save_data["code"] = result.get("code", "")

        with open(filepath, "w", encoding="utf-8") as f:
            json.dump(save_data, f, indent=2, default=str, ensure_ascii=False)

        logger.info(f"Saved strategy to {filepath}")

    def get_strategy_summary(self, results: List[Dict[str, Any]]) -> Dict[str, Any]:
        """
        Generate summary statistics from strategy generation results.

        Parameters
        ----------
        results : List[Dict[str, Any]]
            List of strategy results

        Returns
        -------
        Dict[str, Any]
            Summary statistics
        """
        if not results:
            return {"total": 0, "accepted": 0, "rejected": 0}

        accepted = [r for r in results if r["status"] == "accepted"]
        rejected = [r for r in results if r["status"] == "rejected"]

        summary = {
            "total": len(results),
            "accepted": len(accepted),
            "rejected": len(rejected),
            "acceptance_rate": len(accepted) / len(results) if results else 0,
        }

        if accepted:
            sharpe_values = [r.get("sharpe_ratio", 0) for r in accepted if "sharpe_ratio" in r]
            dd_values = [r.get("max_drawdown", 0) for r in accepted if "max_drawdown" in r]
            wr_values = [r.get("win_rate", 0) for r in accepted if "win_rate" in r]

            summary["best_sharpe"] = max(sharpe_values) if sharpe_values else 0
            summary["avg_sharpe"] = np.mean(sharpe_values) if sharpe_values else 0
            summary["worst_drawdown"] = min(dd_values) if dd_values else 0
            summary["avg_win_rate"] = np.mean(wr_values) if wr_values else 0

        return summary
