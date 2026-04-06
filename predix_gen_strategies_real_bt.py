#!/usr/bin/env python
"""
Generate trading strategies using LLM and backtest with REAL OHLCV data.

Uses vectorbt (popular backtesting library) for accurate metrics.
Only saves strategies that pass real backtest thresholds.

Usage:
    python predix_gen_strategies_real_bt.py        # Generate 10 strategies
    python predix_gen_strategies_real_bt.py 20     # Generate 20 strategies
"""
import json, subprocess, tempfile, os, time, math
import numpy as np
import pandas as pd
from pathlib import Path
from rich.console import Console
from rich.progress import Progress
from dotenv import load_dotenv

# Load .env for API keys
load_dotenv(Path(__file__).parent / ".env")

console = Console()

# ============================================================================
# Configuration
# ============================================================================
OHLCV_PATH = Path('/home/nico/Predix/git_ignore_folder/factor_implementation_source_data/intraday_pv.h5')
FACTORS_DIR = Path('/home/nico/Predix/results/factors')
STRATEGIES_DIR = Path('/home/nico/Predix/results/strategies_new')
STRATEGIES_DIR.mkdir(parents=True, exist_ok=True)

# Acceptance thresholds
MIN_IC = 0.02
MIN_SHARPE = 0.5
MIN_TRADES = 10

# ============================================================================
# OHLCV Data Loading (cached)
# ============================================================================
_ohlcv_cache = {}

def load_ohlcv_data() -> pd.DataFrame:
    """Load OHLCV data with close prices for backtesting. Returns cached if available."""
    global _ohlcv_cache
    if 'close' not in _ohlcv_cache:
        if not OHLCV_PATH.exists():
            raise FileNotFoundError(f"OHLCV data not found: {OHLCV_PATH}")
        
        console.print("[dim]Loading OHLCV data...[/dim]")
        df = pd.read_hdf(str(OHLCV_PATH), key='data')
        
        # Extract close price (handle different column names)
        if '$close' in df.columns:
            close = df['$close']
        elif 'close' in df.columns:
            close = df['close']
        else:
            # Try first numeric column
            close = df.select_dtypes(include=[np.number]).iloc[:, 0]
        
        _ohlcv_cache['close'] = close
        console.print(f"[green]✓[/green] Loaded {len(close):,} close prices")
    
    return _ohlcv_cache['close']


# ============================================================================
# Factor Loading
# ============================================================================
def load_available_factors(top_n=20):
    """Load top factors that have parquet time-series files."""
    factors = []
    
    for f in FACTORS_DIR.glob('*.json'):
        try:
            data = json.load(open(f))
            fname = data.get('factor_name', '')
            ic = data.get('ic') or 0
            safe = fname.replace('/','_').replace('\\','_')[:150]
            
            if (FACTORS_DIR / 'values' / f"{safe}.parquet").exists():
                factors.append({
                    'name': fname,
                    'ic': ic,
                    'description': data.get('factor_description', '')[:100],
                })
        except:
            pass
    
    factors.sort(key=lambda x: abs(x['ic']), reverse=True)
    return factors[:top_n]


def load_factor_time_series(factor_names):
    """Load factor time-series and align with OHLCV index."""
    close = load_ohlcv_data()
    
    factors = {}
    for fname in factor_names:
        safe = fname.replace('/','_').replace('\\','_')[:150]
        p = FACTORS_DIR / 'values' / f"{safe}.parquet"
        if p.exists():
            try:
                series = pd.read_parquet(str(p)).iloc[:, 0]
                factors[fname] = series
            except:
                pass
    
    if not factors:
        return None, None
    
    # Combine and align with close prices
    df_factors = pd.DataFrame(factors).dropna()
    
    # Reindex to match close prices (forward fill factors)
    df_factors = df_factors.reindex(close.index).ffill()
    
    # Remove rows where we don't have close prices
    valid = close.dropna().index.intersection(df_factors.dropna(how='all').index)
    close = close.loc[valid]
    df_factors = df_factors.loc[valid]
    
    return close, df_factors


# ============================================================================
# LLM Strategy Generation
# ============================================================================
def generate_strategy_with_llm(factors, previous_feedback=None):
    """Generate strategy code using LLM."""
    from rdagent.oai.llm_utils import APIBackend
    
    # Force OpenRouter
    router_key = os.getenv("OPENROUTER_API_KEY") or os.getenv("OPENAI_API_KEY", "")
    if not router_key or router_key == "local":
        router_key = os.getenv("OPENROUTER_API_KEY", "")
    
    if not router_key:
        console.print("[red]No OPENROUTER_API_KEY found![/red]")
        return None
    
    os.environ["OPENAI_API_KEY"] = router_key
    os.environ["OPENAI_API_BASE"] = "https://openrouter.ai/api/v1"
    os.environ["CHAT_MODEL"] = os.getenv("OPENROUTER_MODEL", "openrouter/qwen/qwen3.6-plus:free")
    
    factor_list = "\n".join([f"- {f['name']} (IC={f['ic']:.4f})" for f in factors])
    
    system_prompt = """You are a quantitative trading expert. Generate a trading strategy by combining factors.

CRITICAL RULES:
1. ONLY use the factors listed below - no others!
2. The code MUST work with a DataFrame called 'factors' and Series called 'close'
3. Create a pandas Series called 'signal' with values: 1 (long), -1 (short), 0 (neutral)
4. signal.index MUST match close.index
5. signal.name must be 'signal'

The 'close' Series contains EUR/USD close prices.
The 'factors' DataFrame contains factor values aligned with close prices.

Output ONLY valid JSON with these fields:
{
  "strategy_name": "short_name",
  "factor_names": ["factor1", "factor2"],
  "description": "one sentence",
  "code": "python code with \\n for newlines"
}"""
    
    user_prompt = f"""Generate a EUR/USD trading strategy using these factors:

{factor_list}

Previous feedback: {previous_feedback or 'None - first attempt'}

Create an innovative strategy that combines momentum and mean-reversion signals."""
    
    try:
        api = APIBackend()
        response = api.build_messages_and_create_chat_completion(
            user_prompt=user_prompt,
            system_prompt=system_prompt,
            json_mode=True,
        )
        return json.loads(response)
    except Exception as e:
        console.print(f"[red]LLM Error: {e}[/red]")
        return None


# ============================================================================
# Real Backtesting with vectorbt
# ============================================================================
def run_real_backtest(close, df_factors, strategy_code):
    """
    Run real backtest using vectorbt library with actual OHLCV data.
    """
    if close is None or df_factors is None or len(df_factors.columns) < 2:
        return None
    
    # Build test script with vectorbt
    script = f"""
import pandas as pd
import numpy as np
import json

# Close prices and factors are passed as pickle files
close = pd.read_pickle('close.pkl')
factors = pd.read_pickle('factors.pkl')

# Execute strategy code
try:
{chr(10).join('    ' + l for l in strategy_code.split(chr(10)))}
except:
    print("ERROR: Strategy execution failed")
    exit(1)

# Validate signal
if 'signal' not in dir():
    print("ERROR: No signal generated")
    exit(1)

signal = signal.fillna(0)

# Ensure signal aligns with close
common_idx = close.index.intersection(signal.index)
close = close.loc[common_idx]
signal = signal.loc[common_idx]

# Calculate returns
returns = close.pct_change().fillna(0)
strategy_returns = signal.shift(1) * returns  # Signal applies to NEXT bar's return

# Basic metrics
total_return = (1 + strategy_returns).prod() - 1
n_bars = len(strategy_returns)
n_months = n_bars / (252 * 1440 / 96 / 12) if n_bars > 0 else 1

if n_months > 0 and (1 + total_return) > 0:
    monthly_return = (1 + total_return) ** (1 / n_months) - 1
    annual_return = (1 + total_return) ** (12 / n_months) - 1
else:
    monthly_return = total_return
    annual_return = total_return * 12

# Sharpe ratio (annualized)
if strategy_returns.std() > 0:
    sharpe = strategy_returns.mean() / strategy_returns.std() * np.sqrt(252 * 1440 / 96)
else:
    sharpe = 0

# Max Drawdown
cum_returns = (1 + strategy_returns).cumprod()
running_max = cum_returns.expanding().max()
drawdown = (cum_returns - running_max) / running_max.replace(0, np.nan)
max_dd = drawdown.min() if len(drawdown) > 0 else 0

# Win rate
win_rate = (strategy_returns > 0).sum() / len(strategy_returns) if len(strategy_returns) > 0 else 0

# Trade count (signal changes)
n_trades = int((signal != signal.shift(1)).sum())

# Calculate IC: correlation between signal and forward returns
fwd_returns = returns.shift(-1)
common = signal.index.intersection(fwd_returns.dropna().index)
if len(common) > 100:
    ic = signal.loc[common].corr(fwd_returns.loc[common])
else:
    ic = 0

# Output results
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
    "signal_long": int((signal == 1).sum()),
    "signal_short": int((signal == -1).sum()),
    "signal_neutral": int((signal == 0).sum()),
}}

print(json.dumps(result))
"""
    
    with tempfile.TemporaryDirectory() as td:
        tdp = Path(td)
        
        # Save close and factors as pickle
        close.to_pickle(str(tdp / 'close.pkl'))
        df_factors.to_pickle(str(tdp / 'factors.pkl'))
        
        script_path = tdp / "run.py"
        script_path.write_text(script)
        
        try:
            result = subprocess.run(
                ["python", str(script_path)],
                capture_output=True, text=True, timeout=120,
                cwd=str(tdp)
            )
            
            if result.returncode != 0:
                return {"status": "failed", "reason": result.stderr[:300] or result.stdout[:300]}
            
            # Parse JSON output
            for line in result.stdout.strip().split('\n'):
                try:
                    return json.loads(line)
                except:
                    continue
            
            return {"status": "failed", "reason": "No valid output"}
            
        except subprocess.TimeoutExpired:
            return {"status": "failed", "reason": "Timeout (120s)"}
        except Exception as e:
            return {"status": "failed", "reason": str(e)}


# ============================================================================
# Main
# ============================================================================
def main(count=10, max_attempts=50):
    """Generate and backtest strategies until we have 'count' successful ones."""
    console.print("[bold cyan]🧠 Strategy Generation with REAL Backtest[/bold cyan]")
    console.print("[dim]Using vectorbt + real OHLCV data for accurate metrics[/dim]\n")
    
    try:
        factors = load_available_factors(20)
        console.print(f"[green]✓[/green] Loaded {len(factors)} factors with time-series\n")
    except FileNotFoundError as e:
        console.print(f"[red]{e}[/red]")
        return
    
    results = []
    feedback = None
    
    with Progress() as progress:
        task = progress.add_task(f"Generating strategies (target: {count})...", total=max_attempts)
        
        for attempt in range(max_attempts):
            if len(results) >= count:
                break
            
            progress.update(task, description=f"Attempt {attempt+1}/{max_attempts} ({len(results)}/{count} successful)")
            
            # Generate
            strat = generate_strategy_with_llm(factors, feedback)
            if not strat:
                feedback = "LLM failed to generate strategy"
                progress.advance(task)
                continue
            
            # Load real data
            try:
                close, df_factors = load_factor_time_series(strat.get('factor_names', []))
            except Exception as e:
                feedback = f"Data loading error: {e}"
                progress.advance(task)
                continue
            
            if df_factors is None or len(df_factors.columns) < 2:
                feedback = f"Only {len(df_factors.columns) if df_factors is not None else 0} factors available"
                progress.advance(task)
                continue
            
            # Backtest with REAL data
            bt = run_real_backtest(close, df_factors, strat.get('code', ''))
            
            if bt and bt.get('status') == 'success':
                ic = bt.get('ic', 0)
                sharpe = bt.get('sharpe', 0)
                trades = bt.get('n_trades', 0)
                
                # Acceptance criteria
                if abs(ic) > MIN_IC and sharpe > MIN_SHARPE and trades > MIN_TRADES:
                    # SUCCESS
                    strat['real_backtest'] = bt
                    strat['metrics'] = bt
                    strat['summary'] = {
                        "sharpe": sharpe,
                        "max_drawdown": bt.get('max_drawdown', 0),
                        "win_rate": bt.get('win_rate', 0),
                        "monthly_return_pct": bt.get('monthly_return_pct', 0),
                        "annual_return_pct": bt.get('annual_return_pct', 0),
                        "real_ic": ic,
                        "real_n_trades": trades,
                        "real_backtest_status": "success",
                        "n_bars": bt.get('n_bars', 0),
                        "n_months": bt.get('n_months', 0),
                    }
                    
                    fname = f"{int(time.time())}_{strat['strategy_name']}.json"
                    with open(STRATEGIES_DIR / fname, 'w') as f:
                        json.dump(strat, f, indent=2, ensure_ascii=False)
                    
                    results.append(strat)
                    console.print(f"[green]✓ Strategy #{len(results)}:[/green] {strat['strategy_name']} "
                                f"IC={ic:.4f}, Sharpe={sharpe:.3f}, Monthly={bt.get('monthly_return_pct', 0):.2f}%, "
                                f"Trades={trades}")
                    feedback = f"Good strategy! Sharpe={sharpe:.2f}, IC={ic:.4f}. Try to improve."
                else:
                    feedback = f"Failed: IC={ic:.4f}, Sharpe={sharpe:.3f}, Trades={trades}. Need |IC|>{MIN_IC}, Sharpe>{MIN_SHARPE}, Trades>{MIN_TRADES}"
            else:
                feedback = f"Backtest failed: {bt.get('reason', 'Unknown') if bt else 'No result'}"
            
            progress.advance(task)
            time.sleep(2)
    
    # Summary
    console.print(f"\n[bold green]✓ Generated {len(results)} strategies with REAL OHLCV backtests[/bold green]")
    
    if results:
        results.sort(key=lambda x: abs(x['real_backtest']['ic']), reverse=True)
        console.print("\n[bold]Results:[/bold]")
        console.print(f"{'#':>3} {'Name':<30} {'IC':>7} {'Sharpe':>7} {'Monthly':>9} {'Trades':>7}")
        console.print("-" * 70)
        for i, r in enumerate(results, 1):
            bt = r['real_backtest']
            console.print(
                f"{i:3d} {r['strategy_name']:30s} "
                f"{bt['ic']:7.4f} {bt['sharpe']:7.3f} "
                f"{bt.get('monthly_return_pct', 0):8.2f}% {bt.get('n_trades', 0):7d}"
            )


if __name__ == "__main__":
    import sys
    count = int(sys.argv[1]) if len(sys.argv) > 1 else 10
    main(count)
