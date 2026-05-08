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
from typing import Any

import numpy as np
import pandas as pd
from rdagent.scenarios.qlib.local.optuna_optimizer import OptunaOptimizer
from rdagent.components.prompt_loader import load_prompt

# OHLCV data path
OHLCV_PATH = Path(os.getenv(
    "PREDIX_OHLCV_PATH",
    "/home/nico/Predix/git_ignore_folder/factor_implementation_source_data/intraday_pv.h5",
))
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
        max_drawdown: float = -0.30,
        min_win_rate: float = 0.40,
        results_dir: str | None = None,
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
            project_root = Path(__file__).parent.parent.parent.parent.parent
            self.results_dir = project_root / "results"
        else:
            self.results_dir = Path(results_dir)

        self.strategies_dir = self.results_dir / "strategies_new"
        self.strategies_dir.mkdir(parents=True, exist_ok=True)

        self.factors_dir = self.results_dir / "factors"
        self.values_dir = self.factors_dir / "values"

        # Cache OHLCV close — loaded once per orchestrator instance
        self._ohlcv_close = None

        # Track generated combos to avoid duplicates
        self._generated_combos: set[tuple] = set()
        # Track rejected strategies for feedback loop
        self._rejected_history: list[dict] = []

        # Load prompt for strategy generation
        try:
            self.strategy_prompt = load_prompt("strategy_generation")
        except Exception:
            self.strategy_prompt = None
            logger.warning("Strategy generation prompt not found. Using fallback template.")

        logger.info(
            f"StrategyOrchestrator initialized: style={self.trading_style}, "
            f"top_factors={self.top_factors}, min_sharpe={self.min_sharpe}",
        )

    @property
    def ohlcv_close(self) -> pd.Series | None:
        """Lazy-load and cache OHLCV close prices."""
        if self._ohlcv_close is None:
            self._ohlcv_close = self._load_ohlcv_close()
        return self._ohlcv_close

    def _load_ohlcv_close(self) -> pd.Series | None:
        """Load OHLCV close prices from HDF5 file (internal, use ohlcv_close property)."""
        if not OHLCV_PATH.exists():
            logger.warning(f"OHLCV data not found: {OHLCV_PATH}")
            return None

        try:
            ohlcv = pd.read_hdf(str(OHLCV_PATH), key="data")
            if "$close" in ohlcv.columns:
                close = ohlcv["$close"].dropna()
            elif "close" in ohlcv.columns:
                close = ohlcv["close"].dropna()
            else:
                close = ohlcv.select_dtypes(include=[np.number]).iloc[:, 0].dropna()

            # Handle MultiIndex
            if isinstance(close.index, pd.MultiIndex):
                try:
                    close = close.xs("EURUSD", level="instrument")
                except KeyError:
                    idx = close.index.get_level_values("instrument") == "EURUSD"
                    close = close[idx]
                    close.index = close.index.droplevel("instrument")

            return close
        except Exception as e:
            logger.warning(f"Failed to load OHLCV data: {e}")
            return None

    def load_top_factors(self) -> list[dict[str, Any]]:
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
            "kronos": [], "momentum": [], "trend": [], "volatility": [],
            "volume": [], "session": [], "london": [], "range": [],
            "vwap": [], "return": [], "ofi": [], "spread": [],
            "close": [], "divergence": [], "other": [],
        }

        for f in factors_with_files:
            name = f.get("factor_name", "").lower()
            matched = False
            if "kronos" in name:
                type_keywords["kronos"].append(f)
                matched = True
            else:
                for kw in type_keywords:
                    if kw == "kronos":
                        continue
                    if kw in name:
                        type_keywords[kw].append(f)
                        matched = True
                        break
            if not matched:
                type_keywords["other"].append(f)

        # Select best from each type (ensures diversity)
        selected = []
        already_names = set()

        # Priority order: kronos first (foundation model), then momentum, divergence, etc.
        priority_types = ["kronos", "momentum", "divergence", "volatility", "session",
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

    def load_factor_values(self, factor_name: str) -> pd.Series | None:
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

            # Handle empty DataFrame
            if df.empty or len(df.columns) == 0:
                logger.warning(f"Empty parquet file: {parquet_path}")
                return None

            # Handle MultiIndex (datetime, instrument)
            if isinstance(df.index, pd.MultiIndex):
                # Get the factor column name (should be the only column)
                factor_col = df.columns[0]
                # Extract EURUSD series
                try:
                    series = df.xs("EURUSD", level="instrument")[factor_col]
                except KeyError:
                    # Try alternative extraction
                    df_reset = df.reset_index()
                    if "instrument" in df_reset.columns:
                        df_eur = df_reset[df_reset["instrument"] == "EURUSD"].set_index("datetime")
                        series = df_eur[factor_col] if factor_col in df_eur.columns else df_eur.iloc[:, -1]
                    else:
                        series = df.iloc[:, 0]
            else:
                series = df.iloc[:, 0]

            # Ensure numeric
            series = pd.to_numeric(series, errors="coerce")
            series.name = factor_name
            return series
        except Exception as e:
            logger.warning(f"Failed to load factor values for {factor_name}: {e}")
            return None

    def generate_strategy_code(
        self,
        factors: list[dict[str, Any]],
        strategy_name: str,
        max_retries: int = 3,
    ) -> str | None:
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

            if "{{" in user_prompt:
                unreplaced = [w for w in user_prompt.split() if "{{" in w]
                logger.warning(
                    f"Unreplaced template variables in prompt for '{strategy_name}': {unreplaced}"
                )
            user_prompt = user_prompt.replace("{{ max_drawdown }}", str(self.max_drawdown))

            # === FEEDBACK LOOP: Inject insights from previously rejected strategies ===
            if self._rejected_history:
                recent_rejects = self._rejected_history[-5:]
                feedback_lines = ["\n--- Lessons from previous failed strategies ---"]
                for r in recent_rejects:
                    feedback_lines.append(f"- '{r['name']}' rejected: {r['reason']} (Sharpe={r.get('sharpe', 0):.2f})")
                feedback_lines.append("Avoid repeating these mistakes. Generate DIFFERENT code.\n")
                user_prompt += "\n".join(feedback_lines)

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
                    last_error = f"Attempt {attempt}: {e!s}"
                    logger.warning(f"LLM attempt {attempt}/{max_retries} failed with exception: {e}")

            logger.warning(
                f"LLM strategy generation failed after {max_retries} attempts. "
                f"Last error: {last_error}",
            )

        # Fallback: generate template code programmatically
        logger.info("Using fallback code generation")
        return self._generate_fallback_code(context)

    def _generate_with_llm(
        self,
        context: dict[str, Any],
        attempt: int = 1,
        feedback: str | None = None,
    ) -> str | None:
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
                logger.warning(f"JSON 'code' field contains invalid Python (attempt {attempt}). Preview: {code[:200]}")
            else:
                logger.warning(f"JSON parsed but no valid 'code' field found (attempt {attempt}). Keys: {list(json_data.keys())}")

        # === STEP 2: Fallback - Extract Python code block directly (like Factor Coder) ===
        import re
        code_block_match = re.search(r"```python\s*\n(.*?)\n```", content, re.DOTALL)
        if code_block_match:
            code = code_block_match.group(1).strip()
            if code and self._validate_python_code(code):
                logger.info(f"[DEBUG] Python code block extracted ({len(code)} chars)")
                return code

        logger.warning(f"All extraction methods failed (attempt {attempt}). Response preview: {response[:200]}")
        return None

    def _extract_json(self, content: str) -> dict[str, Any] | None:
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
        json_block_match = re.search(r"```json\s*\n(.*?)\n```", content, re.DOTALL)
        if json_block_match:
            try:
                return json_module.loads(json_block_match.group(1))
            except json_module.JSONDecodeError:
                pass

        # Strategy 3: Find ```python ... ``` blocks (Qwen often puts JSON in python blocks)
        python_block_match = re.search(r"```python\s*\n(.*?)\n```", content, re.DOTALL)
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
                    json_str_fixed = re.sub(r",\s*([}\]])", r"\1", json_str)
                    return json_module.loads(json_str_fixed)
                except json_module.JSONDecodeError:
                    pass

        # Strategy 5: Find ``` ... ``` blocks (any language tag)
        code_block_match = re.search(r"```\w*\s*\n(.*?)\n```", content, re.DOTALL)
        if code_block_match:
            block = code_block_match.group(1).strip()
            if block.startswith("{") and block.endswith("}"):
                try:
                    return json_module.loads(block)
                except json_module.JSONDecodeError:
                    pass

        return None

    def _extract_code_from_json(self, json_data: dict[str, Any]) -> str | None:
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

    def _extract_code_from_raw(self, content: str) -> str | None:
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
        python_match = re.search(r"```python\s*\n(.*?)\n```", code, re.DOTALL)
        if python_match:
            code = python_match.group(1)
        else:
            # Try generic ``` blocks
            block_match = re.search(r"```\s*\n(.*?)\n```", code, re.DOTALL)
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

        return code or None

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

    def _generate_fallback_code(self, context: dict[str, Any]) -> str:
        """Generate fallback strategy code programmatically."""
        factor_names = context["factor_names"]
        style_config = "daytrading" if context["trading_style"] == "daytrading" else "swing"

        # Build factor assignment code
        factor_assignments = ",\n    ".join(
            [f'"{name}": factors["{name}"]' for name in factor_names if name != "timestamp"],
        )
        ics = context["factor_ics"]
        ic_weights = [str(abs(ics.get(n, 0.01))) for n in factor_names if n != "timestamp"]

        lines = []
        lines.append(f'"""{context["strategy_name"]} - {style_config.title()} Strategy')
        lines.append(f'Auto-generated by Predix Strategy Orchestrator')
        lines.append(f'Factors: {", ".join(factor_names)}')
        lines.append(f'"""')
        lines.append(f'import numpy as np')
        lines.append(f'import pandas as pd')
        lines.append(f'')
        lines.append(f'STRATEGY_NAME = "{context["strategy_name"]}"')
        lines.append(f'TRADING_STYLE = "{style_config}"')
        lines.append(f'FACTOR_NAMES = {json.dumps(factor_names)}')
        lines.append(f'')
        lines.append(f'factor_data = pd.DataFrame({{')
        lines.append(f'    {factor_assignments}')
        lines.append(f'}})')
        lines.append(f'')
        lines.append(f'factor_norm = (factor_data - factor_data.mean()) / factor_data.std()')
        lines.append(f'')
        lines.append(f'weights = np.array([{", ".join(ic_weights)}])')
        lines.append(f'weights = weights / weights.sum()')
        lines.append(f'')
        lines.append(f'combined_signal = (factor_norm * weights).sum(axis=1)')
        lines.append(f'')
        lines.append(f'entry_threshold = 0.5')
        lines.append(f'exit_threshold = 0.2')
        lines.append(f'')
        lines.append(f'signal = pd.Series(0, index=combined_signal.index)')
        lines.append(f'signal[combined_signal > entry_threshold] = 1')
        lines.append(f'signal[combined_signal < -entry_threshold] = -1')
        lines.append(f'signal[abs(combined_signal) < exit_threshold] = 0')
        lines.append(f'')
        lines.append(f'signal = signal.rolling(window=3, min_periods=1).mean().round().astype(int)')
        return "\n".join(lines)

    def _preflight_check(self, code: str) -> str | None:
        """Fast pre-backtest validation. Returns error string or None if OK."""
        # 1) Syntax check
        try:
            compile(code, "<strategy>", "exec")
        except SyntaxError as e:
            return f"Syntax error: {e}"

        # 2) Quick sandbox: run with tiny fake data, check signal exists and varies
        try:
            import numpy as np
            import pandas as pd

            fake_data = pd.DataFrame({
                "a": np.random.default_rng(0).normal(0, 1, 200),
                "b": np.random.default_rng(1).normal(0, 1, 200),
            })
            fake_close = pd.Series(1.0 + np.cumsum(np.random.default_rng(2).normal(0, 0.001, 200)))
            local_vars = {"factors": fake_data, "close": fake_close}
            exec(code, {"np": np, "pd": pd, "numpy": np}, local_vars)

            if "signal" not in local_vars:
                return "No 'signal' variable produced"
            signal = local_vars["signal"]
            if not isinstance(signal, (pd.Series, np.ndarray)):
                return f"signal is {type(signal).__name__}, not Series/ndarray"
            if isinstance(signal, pd.Series) and len(signal) < 2:
                return "signal has < 2 elements"
            # Check signal varies (not constant)
            if isinstance(signal, pd.Series):
                unique = signal.dropna().nunique()
            else:
                unique = len(set(signal[~np.isnan(signal)])) if len(signal) > 0 else 0
            if unique <= 1:
                return "signal is constant (no variation)"
        except SystemExit:
            pass  # OK, some templates call exit(1) intentionally
        except Exception as e:
            return f"Sandbox error: {type(e).__name__}: {e}"

        return None  # All clear

    def _load_benchmark_strategies(self, n: int = 3) -> list[dict]:
        """Load top-N existing strategies as benchmarks."""
        benchmarks = []
        if self.strategies_dir.exists():
            for f in sorted(self.strategies_dir.glob("*.json")):
                try:
                    data = json.loads(f.read_text())
                    s = data.get("sharpe_ratio") or data.get("summary", {}).get("sharpe")
                    if s and s > 0:
                        benchmarks.append({"name": data.get("strategy_name", f.stem)[:40], "sharpe": s})
                except Exception:
                    pass
        benchmarks.sort(key=lambda x: x["sharpe"], reverse=True)
        return benchmarks[:n]

    def evaluate_strategy(
        self, strategy_code: str, strategy_name: str, factors: list[dict[str, Any]],
        *, override_code: str | None = None,
    ) -> dict[str, Any]:
        """
        Evaluate a strategy by executing its code and calculating metrics.

        Parameters
        ----------
        strategy_code : str
            Python strategy code to execute (original)
        strategy_name : str
            Name of the strategy
        factors : List[Dict[str, Any]]
            List of factor info dicts used by this strategy
        override_code : str, optional
            If provided, use this code instead of strategy_code (for Optuna patching)
        """
        code_to_run = override_code or strategy_code
        try:
            # === PRE-FLIGHT VALIDATION ===
            preflight_issue = self._preflight_check(strategy_code)
            if preflight_issue:
                return {
                    "strategy_name": strategy_name,
                    "status": "rejected",
                    "reason": f"Pre-flight failed: {preflight_issue}",
                    "factors_used": [f["factor_name"] for f in factors],
                }

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
                df_factors[col] = pd.to_numeric(df_factors[col], errors="coerce")

            # Forward-fill daily factors to match OHLCV 1-min index
            # Many factors are daily (1 value per day), need to ffill to 1-min
            # FIX 6: Track ffill ratio for data quality monitoring
            close = self.ohlcv_close
            if close is not None:
                original_len = len(df_factors)
                df_factors = df_factors.reindex(close.index).ffill()

                # Log how much was ffill'd
                ffill_ratio = 1.0 - (original_len / len(df_factors)) if len(df_factors) > original_len else 0.0
                logger.info(
                    f"[DEBUG] {strategy_name}: data quality: "
                    f"original_rows={original_len}, "
                    f"ffill_rows={len(df_factors) - original_len}, "
                    f"ffill_ratio={ffill_ratio:.2%}",
                )

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
                exec(code_to_run, {"np": np, "pd": pd, "numpy": np}, local_vars)
            except Exception as e:
                import traceback
                logger.error(
                    f"Strategy code execution failed for '{strategy_name}': {e}\n"
                    f"{traceback.format_exc()[-2000:]}"
                )
                return {
                    "strategy_name": strategy_name,
                    "status": "rejected",
                    "reason": f"Code execution error: {e!s}",
                    "factors_used": factor_names,
                }

            signal = local_vars.get("signal")
            if signal is None or (isinstance(signal, pd.Series) and signal.empty):
                return {
                    "strategy_name": strategy_name,
                    "status": "rejected",
                    "reason": "Strategy did not produce valid 'signal' variable",
                    "factors_used": factor_names,
                }

            logger.info(
                f"[DEBUG] {strategy_name}: signal stats: "
                f"len={len(signal)}, "
                f"long={int((signal > 0).sum())}, "
                f"short={int((signal < 0).sum())}, "
                f"flat={int((signal == 0).sum())}, "
                f"unique={signal.nunique()}",
            )

            # Delegate all metric computation to the single source of truth.
            # Same formulas as every other backtest path in the repo.
            from rdagent.components.backtesting.vbt_backtest import (
                DEFAULT_TXN_COST_BPS,
                backtest_signal_ftmo,
            )

            # Reuse the already-loaded close from above; create a synthetic proxy if unavailable
            if close is None:
                logger.warning("OHLCV data unavailable, using factor-mean proxy")
                proxy = df_factors.mean(axis=1).astype(float)
                synthetic_close = (1 + proxy.pct_change().fillna(0) * 0.0001).cumprod() * 100.0
                close_for_bt = synthetic_close
            else:
                close_for_bt = close.reindex(signal.index).ffill()

            txn_cost_bps = float(os.getenv("TXN_COST_BPS", DEFAULT_TXN_COST_BPS))
            bt = backtest_signal_ftmo(
                close=close_for_bt,
                signal=signal,
                txn_cost_bps=txn_cost_bps,
                wf_rolling=True,
                mc_n_permutations=200,
            )

            if bt.get("status") != "success":
                return {
                    "strategy_name": strategy_name,
                    "status": "rejected",
                    "reason": bt.get("reason", "backtest failed"),
                    "factors_used": factor_names,
                }

            sharpe = bt["sharpe"]
            max_dd = bt["max_drawdown"]
            win_rate = bt["win_rate"]
            num_real_trades = bt["n_trades"]

            # === WALK-FORWARD OOS METRICS ===
            oos_sharpe = bt.get("wf_oos_sharpe_mean", 0)
            oos_consistency = bt.get("wf_oos_consistency", 0)
            oos_monthly = bt.get("wf_oos_monthly_return_mean", 0)
            wf_n_windows = bt.get("wf_n_windows", 0)

            # === MULTI-TIMEFRAME CHECK ===
            mtf_result = self._check_multi_timeframe(signal, close_for_bt, strategy_name)

            # === STABILITY CHECK ===
            stability = self._check_stability(signal, close_for_bt, strategy_name)

            logger.info(
                f"[DEBUG] {strategy_name}: bt stats: "
                f"sharpe={sharpe:.4f} dd={max_dd:.4f} wr={win_rate:.4f} "
                f"trades={num_real_trades} OOS={oos_sharpe:.2f} "
                f"MTF={mtf_result['passed']} stability={stability['passed']}",
            )

            accepted = self._check_acceptance(sharpe, max_dd, win_rate, oos_sharpe, stability, mtf_result)

            metrics = {
                "strategy_name": strategy_name,
                "status": "accepted" if accepted else "rejected",
                "sharpe_ratio": round(sharpe, 4),
                "annualized_return": round(bt["annualized_return"], 6),
                "annual_return_cagr": round(bt["annual_return_cagr"], 6),
                "max_drawdown": round(max_dd, 6),
                "win_rate": round(win_rate, 4),
                "volatility": round(bt["volatility"], 6),
                "total_return": round(bt["total_return"], 6),
                "num_periods": bt["n_bars"],
                "num_real_trades": num_real_trades,
                "profit_factor": round(bt["profit_factor"], 4) if np.isfinite(bt["profit_factor"]) else None,
                "sortino": round(bt["sortino"], 4),
                "calmar": round(bt["calmar"], 4),
                "factors_used": factor_names,
                "trading_style": self.trading_style,
                "generated_at": datetime.now().isoformat(),
                # Walk-Forward
                "oos_sharpe": round(oos_sharpe, 4),
                "oos_consistency": round(oos_consistency, 4),
                "oos_monthly_return": round(oos_monthly, 6),
                "wf_n_windows": wf_n_windows,
                # Multi-Тimeframe
                "mtf_passed": mtf_result["passed"],
                "mtf_timeframes": mtf_result["timeframes"],
                "mtf_sharpes": mtf_result["sharpes"],
                # Stability
                "stability_passed": stability["passed"],
                "stability_worst_sharpe": round(stability["worst_sharpe"], 4),
            }
            if "data_quality_flag" in bt:
                metrics["data_quality_flag"] = bt["data_quality_flag"]

            if metrics["status"] == "rejected":
                logger.info(
                    f"[DEBUG] {strategy_name}: rejection breakdown: "
                    f"sharpe={sharpe:.4f} (need>={self.min_sharpe}), "
                    f"dd={max_dd:.4f} (need>={self.max_drawdown}), "
                    f"wr={win_rate:.4f} (need>={self.min_win_rate}), "
                    f"OOS={oos_sharpe:.2f}, MTF={mtf_result['passed']}, Stab={stability['passed']}",
                )
                metrics["reason"] = self._get_rejection_reason(sharpe, max_dd, win_rate, oos_sharpe, stability, mtf_result)

            return metrics

        except Exception as e:
            logger.error(f"Strategy evaluation failed for {strategy_name}: {e}")
            logger.debug(traceback.format_exc())
            return {
                "strategy_name": strategy_name,
                "status": "rejected",
                "reason": f"Evaluation error: {e!s}",
                "factors_used": [],
            }

    def _check_acceptance(self, sharpe: float, max_dd: float, win_rate: float,
                          oos_sharpe: float = 0, stability: dict | None = None,
                          mtf_result: dict | None = None) -> bool:
        """Check if strategy meets ALL acceptance criteria."""
        if not (sharpe >= self.min_sharpe and max_dd >= self.max_drawdown and win_rate >= self.min_win_rate):
            return False
        if oos_sharpe <= 0:
            return False
        if stability and not stability["passed"]:
            return False
        if mtf_result and not mtf_result["passed"]:
            return False
        return True

    def _get_rejection_reason(self, sharpe: float, max_dd: float, win_rate: float,
                               oos_sharpe: float = 0, stability: dict | None = None,
                               mtf_result: dict | None = None) -> str:
        """Get human-readable rejection reason."""
        reasons = []
        if sharpe < self.min_sharpe:
            reasons.append(f"Sharpe {sharpe:.2f} < {self.min_sharpe}")
        if max_dd < self.max_drawdown:
            reasons.append(f"MaxDD {max_dd:.2%} < {self.max_drawdown:.2%}")
        if win_rate < self.min_win_rate:
            reasons.append(f"WinRate {win_rate:.1%} < {self.min_win_rate}")
        if oos_sharpe <= 0:
            reasons.append(f"OOS-Sharpe {oos_sharpe:.2f} <= 0 (overfit)")
        if stability and not stability["passed"]:
            reasons.append(f"Stability failed (worst 12M={stability['worst_sharpe']:.2f})")
        if mtf_result and not mtf_result["passed"]:
            reasons.append(f"MTF {mtf_result['passed_frames']}/{len(mtf_result['timeframes'])} timeframes")
        return "; ".join(reasons) if reasons else "Unknown"

    def _check_multi_timeframe(self, signal: pd.Series, close: pd.Series, name: str) -> dict:
        """Test strategy on multiple timeframes (1min, 5min, 15min, 1h)."""
        from rdagent.components.backtesting.vbt_backtest import backtest_signal
        timeframes = {"1min": "1min", "5min": "5min", "15min": "15min", "1h": "1h"}
        sharpes = {}
        passed = 0
        for tf_label, tf_code in timeframes.items():
            try:
                if tf_code == "1min":
                    sig_tf, close_tf = signal, close
                else:
                    sig_tf = signal.resample(tf_code).last().ffill()
                    close_tf = close.resample(tf_code).last().ffill()
                common = sig_tf.index.intersection(close_tf.index)
                if len(common) < 100:
                    sharpes[tf_label] = None
                    continue
                r = backtest_signal(close_tf.loc[common], sig_tf.loc[common], txn_cost_bps=2.14)
                sharpes[tf_label] = round(r["sharpe"], 3)
                if r["sharpe"] > 0:
                    passed += 1
            except Exception:
                sharpes[tf_label] = None
        return {"passed": passed >= 2, "timeframes": list(sharpes.keys()),
                "sharpes": sharpes, "passed_frames": passed}

    def _check_stability(self, signal: pd.Series, close: pd.Series, name: str) -> dict:
        """Rolling 12-month Sharpe — must never go negative."""
        from rdagent.components.backtesting.vbt_backtest import backtest_signal
        n = len(close)
        bars_per_year = 252 * 1440
        window = bars_per_year
        step = bars_per_year // 4
        if n < window * 2:
            return {"passed": True, "worst_sharpe": 0, "windows": 0}
        worst = float("inf")
        windows = 0
        for start in range(0, n - window, step):
            try:
                r = backtest_signal(close.iloc[start:start+window], signal.iloc[start:start+window], txn_cost_bps=2.14)
                if r["status"] == "success":
                    worst = min(worst, r["sharpe"])
                    windows += 1
            except Exception:
                pass
        return {"passed": worst >= 0 and windows >= 3, "worst_sharpe": round(worst, 4), "windows": windows}

    def build_ensemble(self, strategies: list[dict], close: pd.Series | None = None) -> dict | None:
        """Build Sharpe-weighted ensemble from top-N accepted strategies."""
        accepted = [s for s in strategies if s.get("status") == "accepted" and s.get("sharpe_ratio", 0) > 0]
        if len(accepted) < 2:
            return None
        accepted.sort(key=lambda s: s.get("sharpe_ratio", 0), reverse=True)
        top5 = accepted[:5]
        signals = {}
        for s in top5:
            sig = self._load_strategy_signal(s)
            if sig is not None:
                signals[s["strategy_name"]] = sig
        if len(signals) < 2:
            return None
        df_signals = pd.DataFrame(signals).ffill().dropna()
        if len(df_signals) < 1000:
            return None
        weights = np.array([s.get("sharpe_ratio", 1) for s in top5 if s["strategy_name"] in signals])
        weights = weights / weights.sum()
        ensemble_signal = np.sign((df_signals * weights).sum(axis=1))
        if close is None:
            close = self.ohlcv_close
        if close is None:
            return None
        common = ensemble_signal.index.intersection(close.index)
        from rdagent.components.backtesting.vbt_backtest import backtest_signal_ftmo
        bt = backtest_signal_ftmo(close=close.loc[common], signal=ensemble_signal.loc[common],
                                  txn_cost_bps=2.14, wf_rolling=True)
        return {"status": "success", "strategy_name": "Ensemble_Top5",
                "sharpe_ratio": round(bt["sharpe"], 4), "max_drawdown": round(bt["max_drawdown"], 4),
                "win_rate": round(bt["win_rate"], 4), "n_trades": bt["n_trades"],
                "oos_sharpe": round(bt.get("wf_oos_sharpe_mean", 0), 4),
                "members": list(signals.keys())}

    def _load_strategy_signal(self, strategy: dict) -> pd.Series | None:
        """Re-execute strategy code to get its signal series."""
        code = strategy.get("code", "")
        factor_names = strategy.get("factors_used") or strategy.get("factor_names", [])
        if not code or len(factor_names) < 2:
            return None
        factor_values = {}
        for fname in factor_names:
            series = self.load_factor_values(fname)
            if series is not None:
                factor_values[fname] = series
        if len(factor_values) < 2:
            return None
        df = pd.DataFrame(factor_values).sort_index()
        if isinstance(df.index, pd.MultiIndex):
            df = df.droplevel(-1)
        close = self.ohlcv_close
        if close is not None:
            df = df.reindex(close.index).ffill().dropna()
        if len(df) < 1000:
            return None
        local_vars = {"factors": df}
        try:
            exec(code, {"np": np, "pd": pd}, local_vars)
        except Exception:
            return None
        signal = local_vars.get("signal")
        if signal is None or not isinstance(signal, pd.Series):
            return None
        return signal.reindex(close.index).ffill().fillna(0) if close is not None else signal

    def _generate_strategy_name(self, factors: list[dict[str, Any]], idx: int) -> str:
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
                words.extend(cap_words or [p])

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
    ) -> list[dict[str, Any]]:
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
                            f"DD={result['max_drawdown']:.2%}",
                        )
                    else:
                        # Also save rejected strategies for debugging
                        self._save_strategy(result)
                        logger.warning(
                            f"Strategy REJECTED: {result['strategy_name']} - {result.get('reason', 'unknown')} | "
                            f"Sharpe={result.get('sharpe_ratio', 'N/A')} | "
                            f"DD={result.get('max_drawdown', 'N/A')}",
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
            f"({strategies_accepted/max(strategies_generated,1)*100:.1f}%)",
        )

        return results

    def _generate_strategy_configs(self, factors: list[dict], count: int) -> list[list[dict]]:
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

    def _generate_and_evaluate_single(self, idx: int, factors: list[dict]) -> dict[str, Any]:
        """Generate and evaluate a single strategy."""
        strategy_name = self._generate_strategy_name(factors, idx + 1)

        # === DUPLICATE DETECTION ===
        factor_names_tuple = tuple(sorted(f["factor_name"] for f in factors))
        if factor_names_tuple in self._generated_combos:
            return {
                "strategy_name": strategy_name,
                "status": "rejected",
                "reason": f"Duplicate factor combination: {factor_names_tuple}",
            }
        self._generated_combos.add(factor_names_tuple)

        # Generate code
        code = self.generate_strategy_code(factors, strategy_name)
        if not code:
            self._rejected_history.append({"name": strategy_name, "reason": "Code generation failed", "factors": factor_names_tuple})
            return {
                "strategy_name": strategy_name,
                "status": "rejected",
                "reason": "Code generation failed",
            }

        # Evaluate
        result = self.evaluate_strategy(code, strategy_name, factors)
        result["code"] = code

        # === BENCHMARK COMPARISON ===
        if result.get("status") == "accepted":
            benchmarks = self._load_benchmark_strategies(3)
            new_sharpe = result.get("sharpe_ratio", 0)
            for b in benchmarks:
                if new_sharpe > b["sharpe"]:
                    result["beats_benchmark"] = b["name"]
                    result["benchmark_delta"] = round(new_sharpe - b["sharpe"], 2)
                    break

        # Optimize with Optuna if enabled
        if self.use_optuna:
            initial_status = result.get("status", "rejected")
            initial_sharpe = result.get("sharpe_ratio", float("-inf"))
            logger.info(f"Running Optuna optimization for {strategy_name} (initial: {initial_status}, Sharpe={initial_sharpe:.4f})...")
            optimizer = OptunaOptimizer(n_trials=self.optuna_trials)

            # Prepare factor values for optimization
            factor_values = self._prepare_factor_values(factors)

            if factor_values is not None:
                optimized = optimizer.optimize_strategy(result, factor_values)
                optimized_sharpe = optimized.get("sharpe_ratio", float("-inf"))
                optimized_status = optimized.get("status", "rejected")
                best_params = optimized.get("best_params", {})

                # Check if Optuna improved the strategy
                if optimized_sharpe > initial_sharpe:
                    improvement = optimized_sharpe - initial_sharpe
                    logger.info(
                        f"Optuna {'RESCUED' if optimized_status == 'accepted' and initial_status == 'rejected' else 'improved'} "
                        f"{strategy_name}: Sharpe {initial_sharpe:.4f} → {optimized_sharpe:.4f} (+{improvement:.4f})",
                    )

                    # Re-evaluate with best parameters to get comparable metrics
                    if best_params:
                        patched_code = self._patch_strategy_code(code, best_params)
                        re_eval = self._evaluate_with_patched_code(patched_code, strategy_name, factors)
                        if re_eval.get("sharpe_ratio", float("-inf")) > initial_sharpe:
                            result.update(re_eval)
                            result["code"] = patched_code
                            result["best_params"] = best_params
                            # Clear old rejection reason if now accepted
                            if result.get("status") == "accepted":
                                result.pop("reason", None)
                            logger.info(
                                f"Re-evaluated {strategy_name} with best params: "
                                f"Sharpe {initial_sharpe:.4f} → {re_eval.get('sharpe_ratio', 0):.4f}",
                            )
                        else:
                            result.update(optimized)
                            result["best_params"] = best_params
                            # Clear old rejection reason if now accepted
                            if result.get("status") == "accepted":
                                result.pop("reason", None)
                    else:
                        result.update(optimized)
                        # Clear old rejection reason if now accepted
                        if result.get("status") == "accepted":
                            result.pop("reason", None)
                else:
                    logger.debug(f"Optuna did not improve {strategy_name}: {initial_sharpe:.4f} vs {optimized_sharpe:.4f}")
            else:
                logger.warning(f"No factor values available for Optuna optimization of {strategy_name}")

        # === FEEDBACK LOOP ===
        if result.get("status") != "accepted":
            self._rejected_history.append({
                "name": strategy_name,
                "reason": result.get("reason", "unknown"),
                "factors": factor_names_tuple,
                "sharpe": result.get("sharpe_ratio", 0),
            })
            # Keep last 20 rejected for prompt context
            if len(self._rejected_history) > 20:
                self._rejected_history = self._rejected_history[-20:]

        return result

    def _prepare_factor_values(self, factors: list[dict]) -> pd.DataFrame | None:
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
            close = self.ohlcv_close
            if close is not None:
                df = df.reindex(close.index).ffill()
            return df.dropna()
        return None

    def _patch_strategy_code(self, code: str, params: dict[str, Any]) -> str:
        """Patch strategy code with Optuna's best parameters."""
        import re
        patched = code

        entry_thresh = params.get("entry_threshold", 0.8)
        exit_thresh = params.get("exit_threshold", 0.3)
        zscore_window = params.get("zscore_window", 50)
        signal_window = params.get("signal_window", 3)

        param_patterns = [
            (r"entry_thresh\s*=\s*[\d.]+", f"entry_thresh = {entry_thresh}"),
            (r"exit_thresh\s*=\s*[\d.]+", f"exit_thresh = {exit_thresh}"),
            (r"window\s*=\s*\d+", f"window = {zscore_window}"),
            (r"signal_window\s*=\s*\d+", f"signal_window = {signal_window}"),
        ]
        for pattern, replacement in param_patterns:
            patched = re.sub(pattern, replacement, patched)

        # Patch .rolling(N) calls for common window sizes
        rolling_pattern = r"\.rolling\((\d+)\)"
        def replace_rolling(match):
            val = int(match.group(1))
            if val in (20, 30, 50, 100, 200):
                return f".rolling({zscore_window})"
            return match.group(0)
        patched = re.sub(rolling_pattern, replace_rolling, patched)

        return patched

    def _evaluate_with_patched_code(self, patched_code: str, strategy_name: str, factors: list[dict]) -> dict[str, Any]:
        """Re-evaluate strategy with patched parameters (delegates to evaluate_strategy)."""
        return self.evaluate_strategy(patched_code, strategy_name, factors, override_code=patched_code)

    def _save_strategy(self, result: dict[str, Any]) -> None:
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

    def get_strategy_summary(self, results: list[dict[str, Any]]) -> dict[str, Any]:
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
