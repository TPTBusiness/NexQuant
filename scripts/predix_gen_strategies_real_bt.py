#!/usr/bin/env python
import logging
"""
Parallel AI Strategy Generation with REAL OHLCV Backtest.

Generates multiple trading strategies in parallel using LLM,
each with real backtesting on OHLCV data.

Usage:
    # Swing trading (96-bar forward returns)
    python predix_gen_strategies_real_bt.py 10

    # Daytrading with FTMO constraints (12-bar forward returns)
    TRADING_STYLE=daytrading python predix_gen_strategies_real_bt.py 5

    # With parallel workers (default: CPU count)
    TRADING_STYLE=daytrading WORKERS=4 python predix_gen_strategies_real_bt.py 20
"""
import os, sys, json, time, math, random, logging, warnings, subprocess
from pathlib import Path
from datetime import datetime

import numpy as np
import pandas as pd
from rich.console import Console
from rich.progress import Progress, SpinnerColumn, TextColumn, BarColumn, TimeElapsedColumn
from dotenv import load_dotenv

# Suppress warnings and noisy loggers that bleed into Rich progress output
warnings.filterwarnings('ignore')
for _noisy in ('rdagent', 'litellm', 'LiteLLM', 'litellm.utils',
               'litellm.main', 'httpx', 'httpcore', 'openai', 'urllib3'):
    logging.getLogger(_noisy).setLevel(logging.CRITICAL)
# Suppress litellm verbose flag if already imported
try:
    import litellm as _ll
    _ll.suppress_debug_info = True
    _ll.verbose = False
    _ll.set_verbose = False
except Exception:
    logging.debug("Exception caught", exc_info=True)

# ============================================================================
# Configuration
# ============================================================================
OHLCV_PATH = Path('/home/nico/Predix/git_ignore_folder/factor_implementation_source_data/intraday_pv.h5')
FACTORS_DIR = Path('/home/nico/Predix/results/factors')
STRATEGIES_DIR = Path('/home/nico/Predix/results/strategies_new')
STRATEGIES_DIR.mkdir(parents=True, exist_ok=True)

# Trading style
TRADING_STYLE = os.getenv('TRADING_STYLE', 'swing')
N_WORKERS = int(os.getenv('WORKERS', os.cpu_count() or 4))

if TRADING_STYLE == 'daytrading':
    FORWARD_BARS = int(os.getenv('FORWARD_BARS', '12'))
    MIN_IC = 0.02
    MIN_SHARPE = 0.5
    MIN_TRADES = 300
    MAX_DRAWDOWN = -0.10
    STYLE_EMOJI = '🎯 Daytrading'
    STYLE_DESC = 'short-term intraday with FTMO compliance'
else:
    FORWARD_BARS = int(os.getenv('FORWARD_BARS', '96'))
    MIN_IC = 0.02
    MIN_SHARPE = 0.5
    MIN_TRADES = 10
    MAX_DRAWDOWN = -0.30
    STYLE_EMOJI = '📈 Swing'
    STYLE_DESC = 'medium-term intraday'

# Whether to use raw OHLCV-only strategies (no daily factors)
OHLCV_ONLY = os.getenv('OHLCV_ONLY', '0') == '1'

TXN_COST_BPS = float(os.getenv('TXN_COST_BPS', '2.14'))  # 2.35 pip realistic EUR/USD costs

# ── Logging setup: everything printed goes to log file + stdout ───────────────
_LOG_DIR = Path(__file__).parent.parent / "git_ignore_folder" / "logs"
_LOG_DIR.mkdir(parents=True, exist_ok=True)
_log_file_path = _LOG_DIR / f"gen_strategies_{datetime.now().strftime('%Y%m%d_%H%M%S')}.log"
_log_file = open(_log_file_path, "w", encoding="utf-8", buffering=1)  # line-buffered

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(message)s",
    handlers=[
        logging.StreamHandler(sys.stdout),
        logging.FileHandler(_log_file_path, encoding="utf-8"),
    ],
)

class _TeeFile:
    """Writes to both stdout and log file — used as Rich Console file."""
    def __init__(self, *files):
        self._files = files
    def write(self, data):
        for f in self._files:
            f.write(data)
    def flush(self):
        for f in self._files:
            f.flush()
    def fileno(self):
        return self._files[0].fileno()

console = Console(file=_TeeFile(sys.stdout, _log_file), highlight=False)

# ============================================================================
# LLM Configuration (Process-safe)
# ============================================================================
def setup_llm_env():
    """Setup LLM environment variables."""
    load_dotenv(Path(__file__).parent.parent / '.env')
    if os.getenv('OPENAI_API_KEY') == 'local' or os.getenv('LLM_BACKEND', '').lower() == 'local':
        return
    router_key = os.getenv('OPENROUTER_API_KEY', '')
    if router_key:
        os.environ['OPENAI_API_KEY'] = router_key
        os.environ['OPENAI_API_BASE'] = 'https://openrouter.ai/api/v1'
        os.environ['CHAT_MODEL'] = os.getenv('OPENROUTER_MODEL', 'openrouter/google/gemma-4-26b-a4b-it:free')

# ============================================================================
# Factor Loading (cached at module level for each process)
# ============================================================================
_FACTORS_CACHE = None

def load_available_factors(top_n=20):
    """Load top factors that have parquet time-series files."""
    global _FACTORS_CACHE
    if _FACTORS_CACHE is not None:
        return _FACTORS_CACHE[:top_n]
    
    factors = []
    for f in FACTORS_DIR.glob('*.json'):
        try:
            data = json.load(open(f))
            fname = data.get('factor_name', '')
            ic = data.get('ic') or 0
            safe = fname.replace('/','_').replace('\\','_')[:150]
            if (FACTORS_DIR / 'values' / f"{safe}.parquet").exists():
                factors.append({'name': fname, 'ic': ic})
        except:
            pass
    
    factors.sort(key=lambda x: abs(x['ic']), reverse=True)
    _FACTORS_CACHE = factors
    return factors[:top_n]

# ============================================================================
# OHLCV Data Loading (cached at module level)
# ============================================================================
_OHLCV_CACHE = None

def load_ohlcv_data():
    """Load OHLCV close prices."""
    global _OHLCV_CACHE
    if _OHLCV_CACHE is not None:
        return _OHLCV_CACHE
    
    if not OHLCV_PATH.exists():
        raise FileNotFoundError(f"OHLCV data not found: {OHLCV_PATH}")
    
    ohlcv = pd.read_hdf(str(OHLCV_PATH), key='data')
    if '$close' in ohlcv.columns:
        close = ohlcv['$close']
    elif 'close' in ohlcv.columns:
        close = ohlcv['close']
    else:
        close = ohlcv.select_dtypes(include=[np.number]).iloc[:, 0]
    
    _OHLCV_CACHE = close.dropna()
    return _OHLCV_CACHE

# ============================================================================
# Strategy Generation (LLM call - runs in separate process)
# ============================================================================
def generate_single_strategy(args):
    """Generate and backtest ONE strategy. Runs in separate process."""
    idx, factor_subset, feedback, attempt = args
    
    try:
        setup_llm_env()
        
        from rdagent.oai.llm_utils import APIBackend
        
        factor_list = "\n".join([f"- {f['name']} (IC={f['ic']:.4f})" for f in factor_subset])
        
        # Optimized prompts for daytrading vs swing
        if TRADING_STYLE == 'daytrading' and OHLCV_ONLY:
            system_prompt = """You are an expert EUR/USD intraday quant. You build strategies that work ONLY on raw price data (OHLCV), computing all indicators directly from the 1-minute close series.

CRITICAL RULES:
1. The code receives ONLY a pandas Series called 'close' (1-minute EUR/USD close prices, UTC timestamps).
2. 'factors' is NOT available — compute everything from 'close' directly.
3. Create a pandas Series called 'signal' with values: 1 (long), -1 (short), 0 (neutral).
4. signal.index MUST match close.index exactly.
5. signal.name must be 'signal'.
6. Use ONLY pandas/numpy — no external libraries.
7. MANDATORY: The signal MUST flip at least 300 times across the full dataset. Use low thresholds.

Allowed intraday techniques (pick 2-3 and combine):
- Session timing: London open (07:00-09:00 UTC), NY open (13:00-15:00 UTC), session overlap
- Short-window RSI (7-14 bars) on 1-min close
- EMA crossovers (fast=5-15 bars, slow=20-60 bars)
- Bollinger Bands (20-bar, 1.5σ) for mean reversion
- ATR-based volatility breakouts
- VWAP deviation (approximate with rolling mean)
- Time-of-day filters combined with momentum

Output ONLY valid JSON:
{"strategy_name": "short_name", "factor_names": [], "description": "one sentence", "code": "python code"}"""

            user_prompt = f"""Create a EUR/USD 1-minute intraday strategy using ONLY the raw close price series.

{f'Previous feedback: {feedback}' if feedback else 'First attempt — be creative and combine session timing with a momentum or mean-reversion indicator!'}

Hard requirements:
- Signal must change direction at least 300 times total (~4-8 trades per trading day)
- NEVER use ffill() or forward-fill on the signal — recompute fresh at every bar
- Use RSI thresholds between 35-45 (long) and 55-65 (short) — NOT extreme values like 10/90
- Use EMA crossover thresholds of 0 (cross above/below) for maximum trade frequency
- Use causal indicators only: rolling windows, shift(1) — NO look-ahead bias
- No factor data — compute everything from 'close'
- Keep it simple: 2-3 indicators max"""

        elif TRADING_STYLE == 'daytrading':
            system_prompt = f"""You are an expert daytrading quant specializing in EUR/USD scalping and intraday strategies.

CRITICAL RULES for {STYLE_DESC} (forward horizon: {FORWARD_BARS} bars = ~{FORWARD_BARS} minutes):
1. ONLY use the factors listed below - no others!
2. The code MUST work with a DataFrame called 'factors' and Series called 'close'
3. Create a pandas Series called 'signal' with values: 1 (long), -1 (short), 0 (neutral)
4. signal.index MUST match close.index
5. signal.name must be 'signal'
6. MANDATORY: signal must flip direction at least 300 times total — use low thresholds (0.1-0.3)
7. Use rolling z-scores with SHORT windows (5-20 bars) and TIGHT thresholds

Output ONLY valid JSON with these fields:
{{"strategy_name": "short_name", "factor_names": ["f1", "f2"], "description": "one sentence", "code": "python code"}}"""

            user_prompt = f"""Create a EUR/USD DAYTRADING strategy ({FORWARD_BARS}-minute horizon) using these factors:

{factor_list}

{f'Previous feedback: {feedback}' if feedback else 'First attempt - be creative!'}

Hard requirements:
- signal must change at least 300 times total (~4 trades/day) — use thresholds of 0.1-0.3
- NEVER use ffill() or forward-fill on the signal — recompute fresh at every bar
- Use rolling z-scores with windows of 5-20 bars (not 50-100), thresholds ±0.2 to ±0.5
- Combine 2 factors: one momentum, one mean-reversion
- NO global mean/std — always use rolling(window).mean() with shift(1) to avoid look-ahead bias"""

        else:
            system_prompt = f"""You are a quantitative trading expert specializing in EUR/USD daily swing strategies.

CRITICAL RULES for {STYLE_DESC} (forward horizon: {FORWARD_BARS} bars = ~{FORWARD_BARS/60:.1f} hours):
1. ONLY use the factors listed below - no others!
2. The code MUST work with a DataFrame called 'factors' and Series called 'close'
3. Create a pandas Series called 'signal' with values: 1 (long), -1 (short), 0 (neutral)
4. signal.index MUST match close.index
5. signal.name must be 'signal'
6. IMPORTANT: factors are DAILY values broadcast to every 1-minute bar — they change once per day.
   Use daily-level logic: compare today's factor value to a rolling daily mean (window 5-20 DAYS).
   To get daily rolling mean: group by date, take first value per day, compute rolling, then reindex back.
   Example: dates = factors[col].index.get_level_values('datetime').normalize()
            daily_vals = factors[col].groupby(dates).first()
            daily_mean = daily_vals.rolling(10).mean().shift(1)
            daily_signal = (daily_vals > daily_mean).astype(int) * 2 - 1
            signal = daily_signal.reindex(dates).values  (broadcast back to minute bars)
7. The signal should change roughly once per day — this produces ~250-500 trades over 6 years.
8. Keep conditions SIMPLE: one factor above/below its N-day rolling average. Avoid combining 3+ conditions.

Output ONLY valid JSON with these fields:
{{"strategy_name": "short_name", "factor_names": ["f1", "f2"], "description": "one sentence", "code": "python code"}}"""

            user_prompt = f"""Create a EUR/USD SWING trading strategy (hold ~{FORWARD_BARS/60:.0f} hours) using these factors:

{factor_list}

{f'Previous feedback: {feedback}' if feedback else 'First attempt - be creative!'}

Use daily-level signal logic (factor above/below rolling daily mean). Signal changes once per day."""
        
        api = APIBackend()
        response = api.build_messages_and_create_chat_completion(
            user_prompt=user_prompt, system_prompt=system_prompt, json_mode=True
        )
        strategy_data = json.loads(response)
        
        # Validate response
        if 'code' not in strategy_data or 'factor_names' not in strategy_data:
            return {'status': 'invalid', 'reason': 'Missing required fields', 'idx': idx}
        
        return {
            'status': 'generated',
            'strategy': strategy_data,
            'idx': idx
        }
        
    except Exception as e:
        return {'status': 'error', 'reason': str(e)[:200], 'idx': idx}

# ============================================================================
# Backtest Runner (runs in main process to avoid re-loading data)
# ============================================================================
def run_backtest(close, factors_df, strategy_code):
    """
    Execute LLM-generated strategy code in a sandboxed subprocess to produce
    the signal, then delegate all metric computation to the unified
    ``backtest_signal`` engine in the main process.
    """
    if close is None:
        return None
    if not OHLCV_ONLY and (factors_df is None or len(factors_df.columns) < 2):
        return None

    # Flatten MultiIndex — strategy code expects a plain DatetimeIndex
    if isinstance(close.index, pd.MultiIndex):
        close = close.droplevel(-1)
    close = close.sort_index()

    import tempfile

    # Subprocess stays minimal: it only runs the untrusted strategy code
    # and pickles the resulting signal. All numbers come from the shared engine.
    factors_line = "" if OHLCV_ONLY else "factors = pd.read_pickle('factors.pkl')"
    script = f"""
import pandas as pd
import numpy as np

close = pd.read_pickle('close.pkl')
{factors_line}

try:
{chr(10).join('    ' + l for l in strategy_code.split(chr(10)))}
except Exception as e:
    print(f"ERROR: Strategy execution failed: {{e}}")
    raise SystemExit(1)

if 'signal' not in dir():
    print("ERROR: No signal generated")
    raise SystemExit(1)

signal.fillna(0).to_pickle('signal.pkl')
"""

    with tempfile.TemporaryDirectory() as td:
        tdp = Path(td)
        close.to_pickle(str(tdp / 'close.pkl'))
        if not OHLCV_ONLY and factors_df is not None:
            factors_df.to_pickle(str(tdp / 'factors.pkl'))
        (tdp / 'run.py').write_text(script)

        try:
            result = subprocess.run( # nosec B603
                ['python', 'run.py'],
                capture_output=True, text=True, timeout=60,
                cwd=str(tdp)
            )
            if result.returncode != 0:
                return {'status': 'failed', 'reason': (result.stderr or result.stdout)[:200]}

            signal = pd.read_pickle(tdp / 'signal.pkl')
        except subprocess.TimeoutExpired:
            return {'status': 'failed', 'reason': 'Timeout (60s)'}
        except Exception as e:
            return {'status': 'failed', 'reason': str(e)[:200]}

    # Main process: FTMO-realistic backtest (leverage + daily/total loss limits).
    from rdagent.components.backtesting.vbt_backtest import backtest_signal_ftmo

    common = close.index.intersection(signal.index)
    if len(common) < 100:
        return {'status': 'failed', 'reason': f'Not enough aligned data ({len(common)} bars)'}

    close_a  = close.loc[common]
    signal_a = signal.reindex(common).fillna(0)
    fwd_returns = close_a.pct_change(FORWARD_BARS).shift(-FORWARD_BARS)

    from rdagent.components.backtesting.vbt_backtest import OOS_START_DEFAULT
    return backtest_signal_ftmo(
        close=close_a,
        signal=signal_a,
        txn_cost_bps=TXN_COST_BPS,
        forward_returns=fwd_returns,
        oos_start=OOS_START_DEFAULT,
        wf_rolling=False,   # too slow on 2M bars — run via rebacktest script instead
        mc_n_permutations=50,
    )

# ============================================================================
# Threshold Tuner — relax numeric thresholds until MIN_TRADES is reached
# ============================================================================
def _rescale_thresholds(code: str, scale: float) -> str:
    """
    Scale numeric literals in the strategy code that look like signal thresholds.
    RSI thresholds (30-70 range) are moved toward 50.
    Z-score / ratio thresholds (0.0–3.0 range) are multiplied by scale.
    """
    import re

    def replace_rsi(m):
        val = float(m.group(0))
        # Pull toward 50 by (1-scale) fraction
        new_val = 50 + (val - 50) * scale
        return f"{new_val:.1f}"

    def replace_small(m):
        val = float(m.group(0))
        return f"{val * scale:.3f}"

    # RSI-style thresholds: integers/floats between 10 and 90
    code = re.sub(r'\b([1-9]\d(?:\.\d+)?)\b', replace_rsi, code)
    # Small float thresholds: 0.05 – 2.99
    code = re.sub(r'\b(0\.\d+|[12]\.\d+)\b', replace_small, code)
    return code


def tune_thresholds(close, factors_df, code: str) -> tuple:
    """
    Binary-search scale factor (1.0 → 0.05) until n_trades >= MIN_TRADES.
    Returns (best_bt_result, tuned_code) where best_bt_result has max Sharpe
    among all runs that hit MIN_TRADES.
    """
    best_bt, best_code = None, code

    for scale in [1.0, 0.7, 0.5, 0.35, 0.2, 0.1, 0.05]:
        tuned = _rescale_thresholds(code, scale) if scale < 1.0 else code
        bt = run_backtest(close, factors_df, tuned)
        if bt is None or bt.get('status') != 'success':
            continue
        trades = bt.get('n_trades', 0)
        sharpe = bt.get('sharpe', -999)
        if trades >= MIN_TRADES:
            if best_bt is None or sharpe > best_bt.get('sharpe', -999):
                best_bt = bt
                best_code = tuned
            break  # first scale that hits MIN_TRADES wins (they get looser after this)

    return best_bt, best_code


# ============================================================================
# Main Parallel Strategy Generation
# ============================================================================
def main(target_count=10):
    """Generate strategies in parallel with real backtesting."""
    import sys as _sys
    _sys.path.insert(0, str(Path(__file__).parent.parent))
    from rdagent.log import daily_log as _dlog
    _log = _dlog.setup(
        "strategies_bt",
        style=TRADING_STYLE,
        forward_bars=FORWARD_BARS,
        target=target_count,
        workers=N_WORKERS,
    )

    console.print(f"\n[bold cyan]{STYLE_EMOJI} Parallel Strategy Generation[/bold cyan]")
    console.print(f"[dim]Log: {_log_file_path}[/dim]")
    console.print(f"   Style: {STYLE_DESC}")
    console.print(f"   Forward bars: {FORWARD_BARS}")
    console.print(f"   Target: {target_count} accepted strategies")
    console.print(f"   Workers: {N_WORKERS}\n")
    
    # Load data (main process only)
    close = load_ohlcv_data()
    factors = load_available_factors(20)
    
    console.print(f"[green]✓[/green] Loaded {len(factors)} factors, {len(close):,} OHLCV bars\n")
    
    # Load factor time-series
    factor_data = {}
    with Progress(SpinnerColumn(), TextColumn("[bold blue]Loading factors..."), BarColumn(), TimeElapsedColumn()) as progress:
        task = progress.add_task("Loading...", total=len(factors))
        for f_info in factors:
            safe = f_info['name'].replace('/','_').replace('\\','_')[:150]
            pf = FACTORS_DIR / 'values' / f"{safe}.parquet"
            if pf.exists():
                try:
                    series = pd.read_parquet(str(pf)).iloc[:, 0]
                    factor_data[f_info['name']] = series
                except:
                    pass
            progress.update(task, advance=1)
    
    # Align factors with close prices
    all_factor_series = [factor_data[n] for n in factor_data if n in factor_data]
    if not all_factor_series:
        console.print("[red]✗ No factor data loaded![/red]")
        return
    
    df_factors = pd.DataFrame({n: factor_data[n] for n in factor_data if n in factor_data})
    common_idx = close.index.intersection(df_factors.dropna(how='all').index)
    close_aligned = close.loc[common_idx]
    df_aligned = df_factors.loc[common_idx]
    
    console.print(f"[green]✓[/green] Aligned {len(df_aligned):,} data points\n")
    
    # Strategy generation loop
    accepted = []
    feedback_history = []
    max_attempts = target_count * 10  # Allow 10x attempts
    
    with Progress(
        SpinnerColumn(),
        TextColumn("[bold blue]{task.description}"),
        BarColumn(),
        TextColumn("[bold green]{task.completed}/{task.total}"),
        TimeElapsedColumn(),
        redirect_stdout=True,
        redirect_stderr=True,
    ) as progress:
        task = progress.add_task("Generating...", total=max_attempts)
        
        for attempt in range(max_attempts):
            if len(accepted) >= target_count:
                break
            
            # Select random factor subset (2-5 factors) — empty for OHLCV-only mode
            if OHLCV_ONLY:
                factor_subset = []
            else:
                n_factors = random.randint(2, min(5, len(factors)))
                factor_subset = random.sample(factors, n_factors)

            feedback = feedback_history[-1] if feedback_history and random.random() < 0.7 else None

            # Generate in main process (LLM doesn't parallelize well)
            gen_result = generate_single_strategy((attempt, factor_subset, feedback, attempt))

            if gen_result['status'] != 'generated':
                progress.update(task, advance=1)
                continue

            strategy = gen_result['strategy']

            # Backtest (main process - needs data access)
            if OHLCV_ONLY:
                strat_factors = None
                bt_result = run_backtest(close, None, strategy.get('code', ''))
            else:
                strat_factors = df_aligned[[f for f in strategy.get('factor_names', []) if f in df_aligned.columns]]
                if len(strat_factors.columns) < 2:
                    progress.update(task, advance=1)
                    continue
                bt_result = run_backtest(close_aligned, strat_factors, strategy.get('code', ''))
            
            if bt_result and bt_result.get('status') == 'success':
                ic = bt_result.get('ic', 0)
                sharpe = bt_result.get('sharpe', 0)
                trades = bt_result.get('n_trades', 0)
                dd = bt_result.get('max_drawdown', 0)

                # If too few trades, auto-tune thresholds before giving up
                original_code = strategy.get('code', '')
                if trades < MIN_TRADES and bt_result.get('status') == 'success':
                    _log.info(f"TUNING  trades={trades}<{MIN_TRADES} — trying looser thresholds")
                    tuned_bt, tuned_code = tune_thresholds(
                        close if OHLCV_ONLY else close_aligned,
                        None if OHLCV_ONLY else strat_factors,
                        original_code,
                    )
                    if tuned_bt and tuned_bt.get('n_trades', 0) >= MIN_TRADES:
                        bt_result = tuned_bt
                        strategy['code'] = tuned_code
                        ic = bt_result.get('ic', 0)
                        sharpe = bt_result.get('sharpe', 0)
                        trades = bt_result.get('n_trades', 0)
                        dd = bt_result.get('max_drawdown', 0)
                        _log.info(f"TUNED   Sharpe={sharpe:.2f}  Trades={trades}")

                # OOS metrics — mandatory, no fallback to IS values
                oos_sharpe  = bt_result.get('oos_sharpe')
                oos_monthly = bt_result.get('oos_monthly_return_pct')
                oos_trades  = bt_result.get('oos_n_trades', 0)

                # Reject if OOS data is missing (strategy trained on data without OOS period)
                if oos_sharpe is None or oos_monthly is None:
                    _log.info(f"REJECTED  no OOS data (data ends before {OOS_START_DEFAULT}?)")
                    feedback_history.append(f"Rejected: no out-of-sample data after {OOS_START_DEFAULT}.")
                    progress.update(task, advance=1)
                    continue

                # Monte Carlo p-value (edge significance)
                mc_pvalue = bt_result.get('mc_pvalue')

                # Rolling walk-forward metrics
                wf_consistency = bt_result.get('wf_oos_consistency')
                wf_sharpe_mean = bt_result.get('wf_oos_sharpe_mean')

                # Check acceptance criteria — OOS must be profitable + statistically significant
                mc_ok = mc_pvalue is None or mc_pvalue < 0.20  # lenient: top 20% non-random
                wf_ok = wf_consistency is None or wf_consistency >= 0.5  # ≥50% of WF windows profitable
                if (abs(ic or 0) > MIN_IC and sharpe > MIN_SHARPE and trades > MIN_TRADES and dd > MAX_DRAWDOWN
                        and oos_sharpe > 0.0 and oos_monthly > 0.0 and mc_ok and wf_ok):
                    # ACCEPT
                    strategy['real_backtest'] = bt_result
                    strategy['metrics'] = bt_result
                    strategy['summary'] = {
                        'sharpe': sharpe, 'max_drawdown': dd, 'win_rate': bt_result.get('win_rate', 0),
                        'monthly_return_pct': bt_result.get('monthly_return_pct', 0),
                        'annual_return_pct': bt_result.get('annual_return_pct', 0),
                        'real_ic': ic, 'real_n_trades': trades, 'real_backtest_status': 'success',
                        'n_bars': bt_result.get('n_bars', 0), 'n_months': bt_result.get('n_months', 0),
                        'trading_style': TRADING_STYLE,
                        'ohlcv_only': OHLCV_ONLY,
                        'engine': 'ftmo_v2',
                        'txn_cost_bps': TXN_COST_BPS,
                        # Walk-forward OOS split
                        'oos_sharpe': bt_result.get('oos_sharpe'),
                        'oos_monthly_return_pct': bt_result.get('oos_monthly_return_pct'),
                        'oos_max_drawdown': bt_result.get('oos_max_drawdown'),
                        'oos_win_rate': bt_result.get('oos_win_rate'),
                        'oos_n_trades': bt_result.get('oos_n_trades'),
                        'is_sharpe': bt_result.get('is_sharpe'),
                        'is_monthly_return_pct': bt_result.get('is_monthly_return_pct'),
                        'oos_start': bt_result.get('oos_start'),
                        # Rolling walk-forward
                        'wf_n_windows': bt_result.get('wf_n_windows'),
                        'wf_oos_sharpe_mean': wf_sharpe_mean,
                        'wf_oos_sharpe_std': bt_result.get('wf_oos_sharpe_std'),
                        'wf_oos_monthly_return_mean': bt_result.get('wf_oos_monthly_return_mean'),
                        'wf_oos_consistency': wf_consistency,
                        # Monte Carlo significance
                        'mc_pvalue': mc_pvalue,
                        'mc_n_permutations': bt_result.get('mc_n_permutations'),
                    }
                    
                    fname = f"{int(time.time())}_{strategy['strategy_name']}.json"
                    with open(STRATEGIES_DIR / fname, 'w') as f:
                        json.dump(strategy, f, indent=2, ensure_ascii=False)
                    
                    # Generate PDF report
                    try:
                        from predix_strategy_report import StrategyPerformanceReporter
                        reporter = StrategyPerformanceReporter(strategy)
                        reporter.generate_report()
                    except:
                        pass
                    
                    accepted.append(strategy)
                    _log.success(f"ACCEPTED  {strategy['strategy_name']}  IC={ic:.4f}  Sharpe={sharpe:.3f}  Trades={trades}  DD={dd:.1%}")
                    feedback_history.append(f"Excellent! IC={ic:.4f}, Sharpe={sharpe:.2f}, Trades={trades}. Try to improve further.")

                    progress.console.print(f"[green]✓ Strategy #{len(accepted)}:[/green] {strategy['strategy_name']} "
                                          f"IC={ic:.4f}, Sharpe={sharpe:.3f}, Trades={trades}, DD={dd:.1%}")
                else:
                    oos_info = f"OOS_Sharpe={oos_sharpe:+.2f} OOS_Mon={oos_monthly:+.2f}%" if oos_sharpe is not None else ""
                    mc_info  = f" MC_p={mc_pvalue:.2f}" if mc_pvalue is not None else ""
                    wf_info  = f" WF_consistency={wf_consistency:.0%}" if wf_consistency is not None else ""
                    _ic = ic or 0; _sh = sharpe or 0; _dd = dd or 0
                    _log.info(f"REJECTED  IC={_ic:.4f}  Sharpe={_sh:.2f}  Trades={trades}  DD={_dd:.1%}  {oos_info}{mc_info}{wf_info}")
                    feedback_history.append(
                        f"Failed: IC={_ic:.4f}, Sharpe={_sh:.2f}, Trades={trades}, DD={_dd:.1%}, "
                        f"OOS_Sharpe={oos_sharpe:+.2f}, OOS_Monthly={oos_monthly:+.2f}%"
                        + (f", MC_p={mc_pvalue:.2f}" if mc_pvalue is not None else "")
                        + (f", WF_consistency={wf_consistency:.0%}" if wf_consistency is not None else "")
                        + f". Need |IC|>{MIN_IC}, Sharpe>{MIN_SHARPE}, Trades>{MIN_TRADES}, "
                        f"OOS_Sharpe>0, OOS_Monthly>0, MC_p<0.20, WF_consistency≥50%."
                    )
            
            progress.update(task, advance=1)
    
    # Summary
    _log.info(f"DONE  accepted={len(accepted)}  target={target_count}")
    for i, s in enumerate(sorted(accepted, key=lambda x: x['real_backtest'].get('ic', 0), reverse=True), 1):
        bt = s['real_backtest']
        _log.info(f"  #{i}  {s['strategy_name']}  IC={bt.get('ic',0):.4f}  Sharpe={bt.get('sharpe',0):.3f}  Monthly={bt.get('monthly_return_pct',0):.2f}%")

    console.print(f"\n[bold green]✓ Generated {len(accepted)}/{target_count} accepted strategies[/bold green]\n")

    if accepted:
        accepted.sort(key=lambda x: x['real_backtest'].get('ic', 0), reverse=True)
        console.print("[bold]Results:[/bold]")
        for i, s in enumerate(accepted, 1):
            bt = s['real_backtest']
            console.print(f"  {i}. {s['strategy_name']:30s} IC={bt.get('ic',0):.4f} Sharpe={bt.get('sharpe',0):.3f} "
                          f"Monthly={bt.get('monthly_return_pct',0):.2f}% Trades={bt.get('n_trades',0)}")

if __name__ == '__main__':
    count = int(sys.argv[1]) if len(sys.argv) > 1 else 10
    main(count)
