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

# Suppress warnings
warnings.filterwarnings('ignore')
logging.getLogger('rdagent').setLevel(logging.WARNING)

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
    MAX_DRAWDOWN = -1.0
    STYLE_EMOJI = '📈 Swing'
    STYLE_DESC = 'medium-term intraday'

console = Console()

# ============================================================================
# LLM Configuration (Process-safe)
# ============================================================================
def setup_llm_env():
    """Setup LLM environment variables."""
    load_dotenv(Path(__file__).parent / '.env')
    router_key = os.getenv('OPENROUTER_API_KEY') or os.getenv('OPENAI_API_KEY', '')
    if not router_key or router_key == 'local':
        router_key = os.getenv('OPENROUTER_API_KEY', '')
    if router_key:
        os.environ['OPENAI_API_KEY'] = router_key
        os.environ['OPENAI_API_BASE'] = 'https://openrouter.ai/api/v1'
        os.environ['CHAT_MODEL'] = os.getenv('OPENROUTER_MODEL', 'openrouter/qwen/qwen3.6-plus:free')

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
    """Run real backtest with actual OHLCV data."""
    if close is None or factors_df is None or len(factors_df.columns) < 2:
        return None
    
    import tempfile
    
    script = f"""
import pandas as pd
import numpy as np
import json

close = pd.read_pickle('close.pkl')
factors = pd.read_pickle('factors.pkl')

try:
{chr(10).join('    ' + l for l in strategy_code.split(chr(10)))}
except:
    print("ERROR: Strategy execution failed")
    exit(1)

if 'signal' not in dir():
    print("ERROR: No signal generated")
    exit(1)

signal = signal.fillna(0)
common_idx = close.index.intersection(signal.index)
close = close.loc[common_idx]
signal = signal.loc[common_idx]

FORWARD_BARS = {FORWARD_BARS}
returns_fwd = close.pct_change(FORWARD_BARS).shift(-FORWARD_BARS)
signal_aligned = signal.loc[returns_fwd.dropna().index]
fwd_returns = returns_fwd.loc[signal_aligned.index]

if len(signal_aligned) < 100 or len(fwd_returns) < 100:
    print("ERROR: Not enough data after alignment")
    exit(1)

ic = signal_aligned.corr(fwd_returns)
strategy_returns = signal_aligned * fwd_returns

if strategy_returns.std() > 0:
    sharpe = strategy_returns.mean() / strategy_returns.std() * np.sqrt(252 * 1440 / {FORWARD_BARS})
else:
    sharpe = 0

cum = (1 + strategy_returns).cumprod()
running_max = cum.expanding().max()
drawdown = (cum - running_max) / running_max.replace(0, np.nan)
max_dd = drawdown.min() if len(drawdown) > 0 else 0

win_rate = (strategy_returns > 0).sum() / len(strategy_returns) if len(strategy_returns) > 0 else 0
n_trades = int((signal_aligned != signal_aligned.shift(1)).sum())

total_return = cum.iloc[-1] - 1
n_bars = len(strategy_returns)
n_months = n_bars / (252 * 1440 / {FORWARD_BARS} / 12) if n_bars > 0 else 1

if n_months > 0 and (1 + total_return) > 0:
    monthly_return = (1 + total_return) ** (1 / n_months) - 1
    annual_return = (1 + total_return) ** (12 / n_months) - 1
else:
    monthly_return = total_return
    annual_return = total_return * 12

result = {{
    "status": "success",
    "sharpe": float(sharpe),
    "max_drawdown": float(max_dd) if not np.isnan(max_dd) else -0.20,
    "win_rate": float(win_rate),
    "ic": float(ic) if not np.isnan(ic) else 0,
    "n_trades": n_trades,
    "total_return": float(total_return),
    "monthly_return_pct": float(monthly_return * 100),
    "annual_return_pct": float(annual_return * 100),
    "n_bars": int(n_bars),
    "n_months": float(n_months),
    "signal_long": int((signal_aligned == 1).sum()),
    "signal_short": int((signal_aligned == -1).sum()),
    "signal_neutral": int((signal_aligned == 0).sum()),
}}

print(json.dumps(result))
"""
    
    with tempfile.TemporaryDirectory() as td:
        tdp = Path(td)
        close.to_pickle(str(tdp / 'close.pkl'))
        factors_df.to_pickle(str(tdp / 'factors.pkl'))
        
        script_path = tdp / 'run.py'
        script_path.write_text(script)
        
        try:
            result = subprocess.run(
                ['python', str(script_path)],
                capture_output=True, text=True, timeout=60,
                cwd=str(tdp)
            )
            
            if result.returncode != 0:
                return {'status': 'failed', 'reason': result.stderr[:200] or result.stdout[:200]}
            
            for line in result.stdout.strip().split('\n'):
                try:
                    return json.loads(line)
                except:
                    continue
            
            return {'status': 'failed', 'reason': 'No valid JSON output'}
        except subprocess.TimeoutExpired:
            return {'status': 'failed', 'reason': 'Timeout (60s)'}
        except Exception as e:
            return {'status': 'failed', 'reason': str(e)[:200]}

# ============================================================================
# Main Parallel Strategy Generation
# ============================================================================
def main(target_count=10):
    """Generate strategies in parallel with real backtesting."""
    
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
                    feedback_history.append(f"Excellent! IC={ic:.4f}, Sharpe={sharpe:.2f}, Trades={trades}. Try to improve further.")
                    
                    progress.console.print(f"[green]✓ Strategy #{len(accepted)}:[/green] {strategy['strategy_name']} "
                                          f"IC={ic:.4f}, Sharpe={sharpe:.3f}, Trades={trades}, DD={dd:.1%}")
                else:
                    feedback_history.append(f"Failed: IC={ic:.4f}, Sharpe={sharpe:.2f}, Trades={trades}, DD={dd:.1%}. Need |IC|>{MIN_IC}, Sharpe>{MIN_SHARPE}, Trades>{MIN_TRADES}")
            
            progress.update(task, advance=1)
    
    # Summary
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
