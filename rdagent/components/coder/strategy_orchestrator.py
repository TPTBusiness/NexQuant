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
        min_sharpe=0.3,
        max_drawdown=-0.30,
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
import requests

from rdagent.components.prompt_loader import load_prompt
from rdagent.components.coder.optuna_optimizer import OptunaOptimizer

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
        min_sharpe: float = 0.3,
        max_drawdown: float = -0.30,
        min_win_rate: float = 0.40,
        results_dir: Optional[str] = None,
        use_optuna: bool = True,
        optuna_trials: int = 20,
        continuous_optimization: bool = True,
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
        use_optuna : bool
            Enable Optuna hyperparameter optimization
        optuna_trials : int
            Number of Optuna trials per strategy
        continuous_optimization : bool
            If True, optimize ALL strategies (including rejected ones)
            Optuna can often rescue strategies with bad initial parameters
        """
        self.top_factors = top_factors
        self.trading_style = trading_style.lower()
        self.min_sharpe = min_sharpe
        self.max_drawdown = max_drawdown
        self.min_win_rate = min_win_rate
        self.use_optuna = use_optuna
        self.optuna_trials = optuna_trials
        self.continuous_optimization = continuous_optimization

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
        
        # Select diverse factor TYPES, not just top IC
        # This ensures we get momentum, volatility, session, volume, etc.
        type_keywords = {
            "momentum": [], "trend": [], "volatility": [], "volume": [],
            "session": [], "london": [], "range": [], "vwap": [],
            "return": [], "ofi": [], "spread": [], "close": [],
            "divergence": [], "other": []
        }
        
        for f in factors_with_files:
            name = f.get("factor_name", "").lower()
            matched = False
            for kw in type_keywords:
                if kw in name:
                    type_keywords[kw].append(f)
                    matched = True
                    break
            if not matched:
                type_keywords["other"].append(f)
        
        # Select best from each type (ensures diversity)
        selected = []
        already_names = set()
        
        # Priority order: momentum, divergence, volatility, session, volume, etc.
        priority_types = ["momentum", "divergence", "volatility", "session", 
                         "london", "range", "vwap", "volume", "ofi", "spread", 
                         "return", "trend", "close", "other"]
        
        per_type = max(2, self.top_factors // len(priority_types))
        
        for kw in priority_types:
            for f in sorted(type_keywords[kw], key=lambda x: abs(x.get("ic", 0)), reverse=True):
                if f["factor_name"] not in already_names:
                    selected.append(f)
                    already_names.add(f["factor_name"])
                    if len([s for s in selected if s["factor_name"] in [x["factor_name"] for x in type_keywords[kw]]]) >= per_type:
                        break
        
        # Fill remaining with highest IC not yet selected
        if len(selected) < self.top_factors:
            remaining = [f for f in factors_with_files if f["factor_name"] not in already_names]
            remaining.sort(key=lambda x: abs(x.get("ic", 0)), reverse=True)
            selected.extend(remaining[:self.top_factors - len(selected)])
        
        # Log diversity
        type_counts = {}
        for f in selected:
            name = f.get("factor_name", "").lower()
            matched = False
            for kw in type_keywords:
                if kw in name:
                    type_counts[kw] = type_counts.get(kw, 0) + 1
                    matched = True
                    break
            if not matched:
                type_counts["other"] = type_counts.get("other", 0) + 1
        
        logger.info(f"Selected {len(selected)} diverse factors: {type_counts}")
        
        return selected[:self.top_factors]

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

    def generate_strategy_code(
        self,
        factors: List[Dict[str, Any]],
        strategy_name: str,
        max_retries: int = 3,
    ) -> Optional[str]:
        """
        Generate strategy code using LLM from factor combinations.

        Parameters
        ----------
        factors : List[Dict[str, Any]]
            List of factor info dicts to combine
        strategy_name : str
            Name for the generated strategy
        max_retries : int
            Maximum number of retry attempts with feedback (default: 3)

        Returns
        -------
        str or None
            Generated Python strategy code
        """
        factor_names = [f["factor_name"] for f in factors]
        factor_ics = {f["factor_name"]: f.get("ic", 0) for f in factors}

        # Build prompt context with proper template variable replacement
        if isinstance(self.strategy_prompt, dict):
            user_prompt_raw = self.strategy_prompt.get("user", "")
            # Replace ALL template variables
            user_prompt = user_prompt_raw
            user_prompt = user_prompt.replace("{{ factors }}", str(factor_ics))
            user_prompt = user_prompt.replace("{{ ic_values }}", str(factor_ics))  # IC values for each factor
            user_prompt = user_prompt.replace("{{ additional_context }}", f"Strategy name: {strategy_name}")
            user_prompt = user_prompt.replace("{{ trading_style }}", self.trading_style)
            user_prompt = user_prompt.replace("{{ min_sharpe }}", str(self.min_sharpe))
            user_prompt = user_prompt.replace("{{ max_drawdown }}", str(self.max_drawdown))
            system_prompt = self.strategy_prompt.get("system", "")
        else:
            user_prompt = ""
            system_prompt = ""

        context = {
            "strategy_name": strategy_name,
            "factor_names": factor_names,
            "factor_ics": factor_ics,
            "trading_style": self.trading_style,
            "min_sharpe": self.min_sharpe,
            "max_drawdown": self.max_drawdown,
            "system_prompt": system_prompt,
            "user_prompt": user_prompt,
        }

        # Try LLM first with retry logic
        if self.strategy_prompt is not None:
            last_error = None
            for attempt in range(1, max_retries + 1):
                try:
                    feedback = last_error if attempt > 1 else None
                    code = self._generate_with_llm(context, attempt=attempt, feedback=feedback)
                    if code:
                        logger.info(f"LLM generated valid code on attempt {attempt} ({len(code)} chars)")
                        return code
                    last_error = f"Attempt {attempt}: LLM returned empty or invalid code"
                    logger.warning(f"LLM attempt {attempt}/{max_retries} failed: {last_error}")
                except Exception as e:
                    last_error = f"Attempt {attempt}: {str(e)}"
                    logger.warning(f"LLM attempt {attempt}/{max_retries} failed with exception: {e}")

            logger.warning(
                f"LLM strategy generation failed after {max_retries} attempts. "
                f"Last error: {last_error}"
            )

        # Fallback: generate template code programmatically
        logger.info("Using fallback code generation")
        return self._generate_fallback_code(context)

    def _generate_with_llm(
        self,
        context: Dict[str, Any],
        attempt: int = 1,
        feedback: Optional[str] = None,
    ) -> Optional[str]:
        """
        Generate strategy code using LLM with APIBackend (same as Factor Coder).

        Parameters
        ----------
        context : Dict[str, Any]
            Prompt context including system and user prompts
        attempt : int
            Current retry attempt number (1-based)
        feedback : str, optional
            Feedback message from previous failed attempts

        Returns
        -------
        str or None
            Validated Python strategy code, or None if invalid
        """
        import json as json_module

        # Build user message with optional feedback
        user_content = context.get("user_prompt", "")
        if feedback:
            user_content += f"\n\nPREVIOUS ATTEMPT FAILED: {feedback}\n\nPlease output ONLY valid JSON with a 'code' field."

        system_content = context.get("system_prompt", "")

        # Use APIBackend factory function (same as Factor Coder)
        from rdagent.oai.llm_utils import APIBackend

        try:
            response = APIBackend().build_messages_and_create_chat_completion(
                user_prompt=user_content,
                system_prompt=system_content,
                json_mode=True,
            )
        except Exception as e:
            logger.warning(f"APIBackend call failed (attempt {attempt}): {e}")
            return None

        if not response:
            logger.warning(f"APIBackend returned empty response (attempt {attempt})")
            return None

        # Log response for debugging
        logger.info(f"[DEBUG] APIBackend response (attempt {attempt}): {len(response)} chars, preview: {response[:100]!r}")

        # === STEP 1: Try JSON extraction FIRST ===
        content = response.strip()
        json_data = self._extract_json(content)

        if json_data is not None:
            logger.info(f"[DEBUG] JSON extracted successfully, keys: {list(json_data.keys())}")
            code = self._extract_code_from_json(json_data)
            if code:
                if self._validate_python_code(code):
                    logger.info(f"[DEBUG] Valid Python code extracted ({len(code)} chars)")
                    return code
                else:
                    logger.warning(f"JSON 'code' field contains invalid Python (attempt {attempt}). Preview: {code[:200]}")
            else:
                logger.warning(f"JSON parsed but no valid 'code' field found (attempt {attempt}). Keys: {list(json_data.keys())}")

        # === STEP 2: Fallback - Extract Python code block directly (like Factor Coder) ===
        import re
        code_block_match = re.search(r'```python\s*\n(.*?)\n```', content, re.DOTALL)
        if code_block_match:
            code = code_block_match.group(1).strip()
            if code and self._validate_python_code(code):
                logger.info(f"[DEBUG] Python code block extracted ({len(code)} chars)")
                return code

        logger.warning(f"All extraction methods failed (attempt {attempt}). Response preview: {response[:200]}")
        return None

    def _extract_json(self, content: str) -> Optional[Dict[str, Any]]:
        """
        Extract JSON object from LLM response content.

        Tries multiple strategies:
        1. Direct parse if content starts with {
        2. Find ```json ... ``` blocks
        3. Find ```python ... ``` blocks (Qwen often wraps JSON in python blocks)
        4. Find first { to last } in content
        5. Find ``` ... ``` blocks (any language tag)

        Returns
        -------
        Dict or None
            Parsed JSON data, or None if no valid JSON found
        """
        import json as json_module
        import re

        # Strategy 1: Direct parse
        stripped = content.strip()
        if stripped.startswith("{") and stripped.endswith("}"):
            try:
                return json_module.loads(stripped)
            except json_module.JSONDecodeError:
                pass

        # Strategy 2: Find ```json ... ``` blocks
        json_block_match = re.search(r'```json\s*\n(.*?)\n```', content, re.DOTALL)
        if json_block_match:
            try:
                return json_module.loads(json_block_match.group(1))
            except json_module.JSONDecodeError:
                pass

        # Strategy 3: Find ```python ... ``` blocks (Qwen often puts JSON in python blocks)
        python_block_match = re.search(r'```python\s*\n(.*?)\n```', content, re.DOTALL)
        if python_block_match:
            block = python_block_match.group(1).strip()
            if block.startswith("{") and block.endswith("}"):
                try:
                    return json_module.loads(block)
                except json_module.JSONDecodeError:
                    pass

        # Strategy 4: Find first { to last }
        first_brace = content.find("{")
        last_brace = content.rfind("}")
        if first_brace != -1 and last_brace > first_brace:
            json_str = content[first_brace : last_brace + 1]
            try:
                return json_module.loads(json_str)
            except json_module.JSONDecodeError:
                # Try to fix common JSON issues (trailing commas, unescaped newlines)
                try:
                    # Remove trailing commas before } or ]
                    json_str_fixed = re.sub(r',\s*([}\]])', r'\1', json_str)
                    return json_module.loads(json_str_fixed)
                except json_module.JSONDecodeError:
                    pass

        # Strategy 5: Find ``` ... ``` blocks (any language tag)
        code_block_match = re.search(r'```\w*\s*\n(.*?)\n```', content, re.DOTALL)
        if code_block_match:
            block = code_block_match.group(1).strip()
            if block.startswith("{") and block.endswith("}"):
                try:
                    return json_module.loads(block)
                except json_module.JSONDecodeError:
                    pass

        return None

    def _extract_code_from_json(self, json_data: Dict[str, Any]) -> Optional[str]:
        """
        Extract Python code from parsed JSON data.

        Checks multiple possible field names: 'code', 'strategy_code', 'python_code'
        Also handles nested JSON where code might have literal \n characters.

        Returns
        -------
        str or None
            Extracted Python code
        """
        # Try known field names in priority order
        for key in ("code", "strategy_code", "python_code", "strategy"):
            if key in json_data:
                code_value = json_data[key]
                if isinstance(code_value, str) and code_value.strip():
                    # Handle JSON-escaped newlines
                    code = code_value.replace("\\n", "\n")
                    return code.strip()

        return None

    def _extract_code_from_raw(self, content: str) -> Optional[str]:
        """
        Extract Python code from raw (non-JSON) LLM response.

        Tries:
        1. Extract from ```python ... ``` blocks
        2. Extract from ``` ... ``` blocks
        3. Use entire content as code

        Returns
        -------
        str or None
            Extracted Python code
        """
        import re
        from collections import Counter

        code = content.strip()

        # Try to find ```python blocks
        python_match = re.search(r'```python\s*\n(.*?)\n```', code, re.DOTALL)
        if python_match:
            code = python_match.group(1)
        else:
            # Try generic ``` blocks
            block_match = re.search(r'```\s*\n(.*?)\n```', code, re.DOTALL)
            if block_match:
                code = block_match.group(1)

        if not code or not code.strip():
            return None

        # Remove markdown code fences if still present
        code = code.replace("```python", "").replace("```", "").strip()

        # Smart dedent: find most common non-zero indent
        lines = code.split("\n")
        indents = Counter()
        for line in lines:
            if line.strip():  # Non-empty line
                indent = len(line) - len(line.lstrip())
                indents[indent] += 1

        # Remove empty lines from consideration
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

        code = "\n".join(dedented).strip()

        # Remove non-code commentary lines
        final_lines = []
        for line in code.split("\n"):
            stripped = line.strip()
            if not stripped:
                continue
            # Stop at commentary (bullets, conversational text)
            commentary_patterns = ["*", "\u2022", "Wait", "Also", "One more", "Note:", "Here", "This strategy"]
            if any(stripped.startswith(p) for p in commentary_patterns):
                break
            final_lines.append(line)

        code = "\n".join(final_lines).strip()

        # Remove non-ASCII characters (emojis, etc.)
        code = code.encode("ascii", "ignore").decode("ascii").strip()

        return code if code else None

    def _validate_python_code(self, code: str) -> bool:
        """
        Validate that a string is valid Python code.

        Parameters
        ----------
        code : str
            Code string to validate

        Returns
        -------
        bool
            True if code compiles successfully
        """
        if not code or len(code) < 10:
            logger.debug("Code too short to be valid")
            return False

        try:
            compile(code, "<strategy>", "exec")
            return True
        except SyntaxError as e:
            logger.debug(f"Python syntax error: {e}")
            return False

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
            
            # Forward-fill daily factors to match OHLCV 1-min index
            # Many factors are daily (1 value per day), need to ffill to 1-min
            close = self.load_ohlcv_close()
            if close is not None:
                df_factors = df_factors.reindex(close.index).ffill()
            
            df_factors = df_factors.dropna()
            
            if len(df_factors) < 1000:
                return {
                    "strategy_name": strategy_name,
                    "status": "rejected",
                    "reason": f"Insufficient numeric data after conversion ({len(df_factors)} rows)",
                    "factors_used": factor_names,
                }
            
            # close is already loaded above for ffill, reuse it
            # Reindex close to match factor index
            if close is not None:
                close = close.reindex(df_factors.index)
            
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
        workers: int = 2,  # Reduced from 4 to 2 to avoid LLM server overload
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
                        # Also save rejected strategies for debugging
                        self._save_strategy(result)
                        logger.warning(
                            f"Strategy REJECTED: {result['strategy_name']} - {result.get('reason', 'unknown')} | "
                            f"Sharpe={result.get('sharpe_ratio', 'N/A')} | "
                            f"DD={result.get('max_drawdown', 'N/A')}"
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

        # Optimize with Optuna if enabled
        # KEY CHANGE: Optimize ALL strategies, not just accepted ones
        # Optuna can often rescue strategies with bad initial parameters
        # by finding optimal entry/exit thresholds, signal smoothing, etc.
        if self.use_optuna:
            initial_status = result.get("status", "rejected")
            initial_sharpe = result.get("sharpe_ratio", float('-inf'))
            logger.info(f"Running Optuna optimization for {strategy_name} (initial: {initial_status}, Sharpe={initial_sharpe:.4f})...")
            optimizer = OptunaOptimizer(n_trials=self.optuna_trials)

            # Prepare factor values for optimization
            factor_values = self._prepare_factor_values(factors)

            if factor_values is not None:
                optimized = optimizer.optimize_strategy(result, factor_values)
                optimized_sharpe = optimized.get("sharpe_ratio", float('-inf'))
                optimized_status = optimized.get("status", "rejected")

                # Check if Optuna improved the strategy
                if optimized_sharpe > initial_sharpe:
                    improvement = optimized_sharpe - initial_sharpe
                    logger.info(
                        f"Optuna {'RESCUED' if optimized_status == 'accepted' and initial_status == 'rejected' else 'improved'} "
                        f"{strategy_name}: Sharpe {initial_sharpe:.4f} → {optimized_sharpe:.4f} (+{improvement:.4f})"
                    )
                    result.update(optimized)
                else:
                    logger.debug(f"Optuna did not improve {strategy_name}: {initial_sharpe:.4f} vs {optimized_sharpe:.4f}")
            else:
                logger.warning(f"No factor values available for Optuna optimization of {strategy_name}")

        return result

    def _prepare_factor_values(self, factors: List[Dict]) -> Optional[pd.DataFrame]:
        """Prepare factor values DataFrame for Optuna optimization."""
        factor_values = {}
        for f in factors:
            fname = f.get("factor_name", "")
            if fname:
                series = self.load_factor_values(fname)
                if series is not None:
                    factor_values[fname] = series
        
        if factor_values:
            df = pd.DataFrame(factor_values)
            # Forward-fill to OHLCV index
            close = self.load_ohlcv_close()
            if close is not None:
                df = df.reindex(close.index).ffill()
            return df.dropna()
        return None

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
