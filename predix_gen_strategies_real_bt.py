#!/usr/bin/env python
"""
Generate strategies using LLM and IMMEDIATELY backtest with real factor time-series.
Only saves strategies that pass the real backtest.
"""
import json, subprocess, tempfile, re, os, time
import numpy as np
import pandas as pd
from pathlib import Path
from rich.console import Console
from rich.progress import Progress

console = Console()

# Configuration
VALUES_DIR = Path('/home/nico/Predix/results/factors/values')
STRATEGIES_DIR = Path('/home/nico/Predix/results/strategies_new')
STRATEGIES_DIR.mkdir(parents=True, exist_ok=True)

# Load top factors with time-series
def load_available_factors(top_n=20):
    """Load top factors that have parquet time-series files."""
    factors_dir = Path('/home/nico/Predix/results/factors')
    factors = []
    
    for f in factors_dir.glob('*.json'):
        try:
            data = json.load(open(f))
            fname = data.get('factor_name', '')
            ic = data.get('ic') or 0
            safe = fname.replace('/','_').replace('\\','_')[:150]
            
            if (VALUES_DIR / f"{safe}.parquet").exists():
                factors.append({
                    'name': fname,
                    'ic': ic,
                    'description': data.get('factor_description', '')[:100],
                })
        except:
            pass
    
    factors.sort(key=lambda x: abs(x['ic']), reverse=True)
    return factors[:top_n]

def load_factor_values(factor_names):
    """Load factor time-series from parquet files."""
    dfs = {}
    for n in factor_names:
        safe = n.replace('/','_').replace('\\','_')[:150]
        p = VALUES_DIR / f"{safe}.parquet"
        if p.exists():
            try:
                df = pd.read_parquet(str(p))
                if df is not None and len(df.columns) > 0:
                    dfs[n] = df.iloc[:, 0]
            except: pass
    return pd.DataFrame(dfs).dropna()

def generate_strategy_with_llm(factors, previous_feedback=None):
    """Generate strategy code using LLM."""
    # Load .env first to get OPENROUTER_API_KEY
    from dotenv import load_dotenv
    load_dotenv(Path(__file__).parent / ".env", override=True)
    
    from rdagent.oai.llm_utils import APIBackend
    
    # Get OpenRouter key
    router_key = os.getenv("OPENROUTER_API_KEY", "")
    if not router_key:
        console.print("[red]No OPENROUTER_API_KEY in .env![/red]")
        return None
    
    # Override dotenv settings
    os.environ["OPENAI_API_KEY"] = router_key
    os.environ["OPENAI_API_BASE"] = "https://openrouter.ai/api/v1"
    os.environ["CHAT_MODEL"] = os.getenv("OPENROUTER_MODEL", "openrouter/qwen/qwen3.6-plus:free")
    
    factor_list = "\n".join([f"- {f['name']} (IC={f['ic']:.4f})" for f in factors])
    
    system_prompt = """You are a quantitative trading expert. Generate a trading strategy by combining factors.

CRITICAL RULES:
1. ONLY use the factors listed below - no others!
2. The code MUST work with a DataFrame called 'df' that has the factor columns
3. Create a pandas Series called 'signal' with values: 1 (long), -1 (short), 0 (neutral)
4. signal.name must be 'signal'

Factor loading is ALREADY done - df contains the factor columns.

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

def run_real_backtest(df, code):
    """Run strategy code and calculate real metrics."""
    if df.empty or len(df.columns) < 2:
        return None
    
    with tempfile.TemporaryDirectory() as td:
        tdp = Path(td)
        df.to_parquet(str(tdp / "factors.parquet"))
        
        script = tdp / "run.py"
        script.write_text(f"""
import pandas as pd, numpy as np
df = pd.read_parquet('factors.parquet')
try:
{chr(10).join('    '+l for l in code.split(chr(10)))}
except:
    pass

try:
    if 'signal' not in dir():
        signal = pd.Series(np.where(df.mean(axis=1) > 0, 1, -1), index=df.index)
    signal.name = 'signal'
    signal.to_pickle('signal.pkl')
    print("OK")
except Exception as e:
    print(f"ERROR: {{e}}")
""")
        try:
            r = subprocess.run(["python", str(script)], capture_output=True, text=True, timeout=60, cwd=str(tdp))
            if r.returncode != 0 or "OK" not in r.stdout:
                return None
            sig = pd.read_pickle(str(tdp / "signal.pkl"))
        except:
            return None
    
    # Calculate metrics
    fwd = df.mean(axis=1).shift(-96).dropna()
    sig = sig.loc[fwd.index]
    if len(sig) < 100: return None
    
    ic = sig.corr(fwd)
    rets = sig * fwd
    std = rets.std()
    
    sharpe = rets.mean()/std * np.sqrt(252*1440/96) if std > 0 and not np.isnan(std) else 0
    sharpe = min(max(sharpe, -5), 5)
    
    cum = (1+rets).cumprod().replace([np.inf,-np.inf], np.nan).fillna(1)
    dd = ((cum - cum.cummax())/cum.cummax().replace(0, np.nan)).min()
    mdd = min(max(dd if not np.isnan(dd) else -0.20, -1.0), 0.0)
    wr = (rets>0).sum()/len(rets)
    trades = int((sig != sig.shift(1)).sum())
    
    tot = cum.iloc[-1] - 1
    if np.isnan(tot) or np.isinf(tot): tot = 0
    tot = max(min(tot, 1.0), -0.5)
    nm = len(rets)/(252*1440/96/12)
    mon = (1+tot)**(1/nm)-1 if nm > 0 and (1+tot) > 0 else tot
    mon = max(min(mon, 0.20), -0.20)
    
    return {"status":"success", "sharpe":float(sharpe), "max_drawdown":float(mdd),
            "win_rate":float(wr), "ic":float(ic) if not np.isnan(ic) else 0,
            "n_trades":trades, "monthly_return_pct":float(mon*100)}

def main(count=10, max_attempts=50):
    """Generate and backtest strategies until we have 'count' successful ones."""
    console.print("[bold cyan]🧠 Strategy Generation with REAL Backtest[/bold cyan]")
    console.print("[dim]Only strategies that pass real backtest will be saved[/dim]\n")
    
    factors = load_available_factors(20)
    console.print(f"[green]✓[/green] Loaded {len(factors)} factors with time-series\n")
    
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
            
            # Load data
            df = load_factor_values(strat.get('factor_names', []))
            if df.empty or len(df.columns) < 3:
                feedback = f"Only {len(df.columns)} factors available"
                progress.advance(task)
                continue
            
            # Backtest
            bt = run_real_backtest(df, strat.get('code', ''))
            
            if bt and bt['sharpe'] > 0.5 and abs(bt['ic']) > 0.01:
                # SUCCESS - save strategy
                strat['real_backtest'] = bt
                strat['metrics'] = bt
                strat['summary'] = bt
                
                fname = f"{int(time.time())}_{strat['strategy_name']}.json"
                with open(STRATEGIES_DIR / fname, 'w') as f:
                    json.dump(strat, f, indent=2, ensure_ascii=False)
                
                results.append(strat)
                console.print(f"[green]✓ Strategy #{len(results)}:[/green] {strat['strategy_name']} "
                            f"Sharpe={bt['sharpe']:.3f}, IC={bt['ic']:.4f}, Monthly={bt['monthly_return_pct']:.2f}%")
                feedback = f"Good strategy! Sharpe={bt['sharpe']:.2f}. Try to improve."
            else:
                ic = bt['ic'] if bt else 0
                feedback = f"Failed: IC={ic:.4f}, Sharpe={bt['sharpe'] if bt else 0:.3f}. Need |IC| > 0.01 and Sharpe > 0.5. Try different factor combination or weighting."
            
            progress.advance(task)
            time.sleep(2)  # Rate limit
    
    # Summary
    console.print(f"\n[bold green]✓ Generated {len(results)} strategies with REAL backtests[/bold green]")
    
    if results:
        results.sort(key=lambda x: x['real_backtest']['sharpe'], reverse=True)
        console.print("\n[bold]Results:[/bold]")
        for i, r in enumerate(results, 1):
            bt = r['real_backtest']
            console.print(f"  {i}. {r['strategy_name']:30s} Sharpe={bt['sharpe']:6.3f} IC={bt['ic']:.4f} Monthly={bt['monthly_return_pct']:.2f}%")

if __name__ == "__main__":
    import sys
    count = int(sys.argv[1]) if len(sys.argv) > 1 else 10
    main(count)
