#!/usr/bin/env python
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
    pass

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
    MIN_TRADES = 20
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

TXN_COST_BPS = float(os.getenv('TXN_COST_BPS', '2.14'))  # 2.35 pip realistic EUR/USD costs

console = Console()

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
        if TRADING_STYLE == 'daytrading':
            system_prompt = f"""You are an expert daytrading quant specializing in EUR/USD scalping and intraday strategies.

CRITICAL RULES for {STYLE_DESC} (forward horizon: {FORWARD_BARS} bars = ~{FORWARD_BARS} minutes):
1. ONLY use the factors listed below - no others!
2. The code MUST work with a DataFrame called 'factors' and Series called 'close'
3. Create a pandas Series called 'signal' with values: 1 (long), -1 (short), 0 (neutral)
4. signal.index MUST match close.index
5. signal.name must be 'signal'
6. Optimize for FREQUENT signals (many trades) since the horizon is only {FORWARD_BARS} minutes
7. Use LOWER thresholds (0.2-0.5) to generate more trades for daytrading

Output ONLY valid JSON with these fields:
{{"strategy_name": "short_name", "factor_names": ["f1", "f2"], "description": "one sentence", "code": "python code"}}"""
            
            user_prompt = f"""Create a EUR/USD DAYTRADING strategy ({FORWARD_BARS}-minute horizon) using these factors:

{factor_list}

{f'Previous feedback: {feedback}' if feedback else 'First attempt - be creative!'}

Requirements for daytrading:
- Use {FORWARD_BARS}-minute forward returns (not daily)
- Generate frequent signals (aim for 20+ trades in the dataset)
- Use rolling z-scores with short windows (10-30 bars)
- Apply tight thresholds (0.2-0.5) for more trades
- Combine momentum + mean-reversion effectively"""
        
        else:
            system_prompt = f"""You are a quantitative trading expert specializing in EUR/USD intraday strategies.

CRITICAL RULES for {STYLE_DESC} (forward horizon: {FORWARD_BARS} bars = ~{FORWARD_BARS/60:.1f} hours):
1. ONLY use the factors listed below - no others!
2. The code MUST work with a DataFrame called 'factors' and Series called 'close'
3. Create a pandas Series called 'signal' with values: 1 (long), -1 (short), 0 (neutral)
4. signal.index MUST match close.index
5. signal.name must be 'signal'

Output ONLY valid JSON with these fields:
{{"strategy_name": "short_name", "factor_names": ["f1", "f2"], "description": "one sentence", "code": "python code"}}"""
            
            user_prompt = f"""Create a EUR/USD trading strategy using these factors:

{factor_list}

{f'Previous feedback: {feedback}' if feedback else 'First attempt - be creative!'}"""
        
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
    if close is None or factors_df is None or len(factors_df.columns) < 2:
        return None

    import tempfile

    # Subprocess stays minimal: it only runs the untrusted strategy code
    # and pickles the resulting signal. All numbers come from the shared engine.
    script = f"""
import pandas as pd
import numpy as np

close = pd.read_pickle('close.pkl')
factors = pd.read_pickle('factors.pkl')

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
        factors_df.to_pickle(str(tdp / 'factors.pkl'))
        (tdp / 'run.py').write_text(script)

        try:
            result = subprocess.run(
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

    return backtest_signal_ftmo(
        close=close_a,
        signal=signal_a,
        txn_cost_bps=TXN_COST_BPS,
        forward_returns=fwd_returns,
    )

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
            
            # Select random factor subset (2-5 factors)
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
            # Build factors DataFrame for this strategy
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
                
                # Check acceptance criteria
                if abs(ic) > MIN_IC and sharpe > MIN_SHARPE and trades > MIN_TRADES and dd > MAX_DRAWDOWN:
                    # ACCEPT
                    strategy['real_backtest'] = bt_result
                    strategy['metrics'] = bt_result
                    strategy['summary'] = {
                        'sharpe': sharpe, 'max_drawdown': dd, 'win_rate': bt_result.get('win_rate', 0),
                        'monthly_return_pct': bt_result.get('monthly_return_pct', 0),
                        'annual_return_pct': bt_result.get('annual_return_pct', 0),
                        'real_ic': ic, 'real_n_trades': trades, 'real_backtest_status': 'success',
                        'n_bars': bt_result.get('n_bars', 0), 'n_months': bt_result.get('n_months', 0),
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
                    _log.info(f"REJECTED  IC={ic:.4f}  Sharpe={sharpe:.2f}  Trades={trades}  DD={dd:.1%}")
                    feedback_history.append(f"Failed: IC={ic:.4f}, Sharpe={sharpe:.2f}, Trades={trades}, DD={dd:.1%}. Need |IC|>{MIN_IC}, Sharpe>{MIN_SHARPE}, Trades>{MIN_TRADES}")
            
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
