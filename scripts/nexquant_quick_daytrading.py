#!/usr/bin/env python
"""
Quick Daytrading Strategy Generator with CORRECT factor alignment.

Uses forward-fill to align daily factors to 1-min frequency,
then runs fast backtests without LLM calls.

Usage:
    python nexquant_quick_daytrading.py 5
    python nexquant_quick_daytrading.py 10
"""
import json, time, subprocess, tempfile  # nosec
from pathlib import Path
import numpy as np
import pandas as pd
from rich.console import Console

console = Console()

STRATEGIES_DIR = Path('results/strategies_new')
STRATEGIES_DIR.mkdir(parents=True, exist_ok=True)

FACTOR_FILES = Path('results/factors')
VALUE_FILES = FACTOR_FILES / 'values'
OHLCV_PATH = Path('git_ignore_folder/factor_implementation_source_data/intraday_pv.h5')

# Best daytrading strategies (12-min horizon, optimized for FTMO)
DAYTRADING_COMBOS = [
    {
        'name': 'MomentumDivergence12min',
        'factors': ['daily_close_return_96', 'daily_session_momentum_divergence_1d'],
        'code': '''mom = factors['daily_close_return_96']
div = factors['daily_session_momentum_divergence_1d']

w = 20
mom_z = (mom - mom.rolling(w).mean()) / (mom.rolling(w).std() + 1e-8)
div_z = (div - div.rolling(w).mean()) / (div.rolling(w).std() + 1e-8)

composite = (mom_z - div_z).fillna(0)
signal = pd.Series(0, index=close.index, name='signal')
signal[composite > 0.3] = 1
signal[composite < -0.3] = -1
signal = signal.fillna(0).astype(int)''',
    },
    {
        'name': 'LondonSessionScalp',
        'factors': ['london_mom', 'daily_session_momentum_divergence_1d'],
        'code': '''mom = factors['london_mom']
div = factors['daily_session_momentum_divergence_1d']

w = 15
mom_z = (mom - mom.rolling(w).mean()) / (mom.rolling(w).std() + 1e-8)
div_z = (div - div.rolling(w).mean()) / (div.rolling(w).std() + 1e-8)

composite = (mom_z - div_z).fillna(0)
signal = pd.Series(0, index=close.index, name='signal')
signal[composite > 0.25] = 1
signal[composite < -0.25] = -1
signal = signal.fillna(0).astype(int)''',
    },
    {
        'name': 'TrendReversionScalp',
        'factors': ['daily_ols_slope_96', 'daily_session_momentum_divergence_1d', 'DailyTrendStrength_Raw'],
        'code': '''slope = factors['daily_ols_slope_96']
div = factors['daily_session_momentum_divergence_1d']
trend = factors['DailyTrendStrength_Raw']

w = 20
slope_z = (slope - slope.rolling(w).mean()) / (slope.rolling(w).std() + 1e-8)
div_z = (div - div.rolling(w).mean()) / (div.rolling(w).std() + 1e-8)
trend_z = (trend - trend.rolling(w).mean()) / (trend.rolling(w).std() + 1e-8)

composite = (0.5 * slope_z - 0.3 * div_z + 0.2 * trend_z).fillna(0)
signal = pd.Series(0, index=close.index, name='signal')
signal[composite > 0.3] = 1
signal[composite < -0.3] = -1
signal = signal.fillna(0).astype(int)''',
    },
    {
        'name': 'VolAdjMomentum12',
        'factors': ['daily_ret_vol_adj_1d', 'daily_session_momentum_divergence_1d', 'DCP'],
        'code': '''vol = factors['daily_ret_vol_adj_1d']
div = factors['daily_session_momentum_divergence_1d']
dcp = factors['DCP']

w = 20
vol_z = (vol - vol.rolling(w).mean()) / (vol.rolling(w).std() + 1e-8)
div_z = (div - div.rolling(w).mean()) / (div.rolling(w).std() + 1e-8)
dcp_z = (dcp - dcp.rolling(w).mean()) / (dcp.rolling(w).std() + 1e-8)

composite = (0.5 * vol_z - 0.3 * div_z + 0.2 * dcp_z).fillna(0)
signal = pd.Series(0, index=close.index, name='signal')
signal[composite > 0.35] = 1
signal[composite < -0.35] = -1
signal = signal.fillna(0).astype(int)''',
    },
    {
        'name': 'SessionMeanReversion',
        'factors': ['session_momentum_diff', 'daily_norm_body', 'daily_c2c_return'],
        'code': '''session = factors['session_momentum_diff']
body = factors['daily_norm_body']
c2c = factors['daily_c2c_return']

w = 15
sess_z = (session - session.rolling(w).mean()) / (session.rolling(w).std() + 1e-8)
body_z = (body - body.rolling(w).mean()) / (body.rolling(w).std() + 1e-8)
c2c_z = (c2c - c2c.rolling(w).mean()) / (c2c.rolling(w).std() + 1e-8)

composite = (0.5 * sess_z + 0.3 * body_z + 0.2 * c2c_z).fillna(0)
signal = pd.Series(0, index=close.index, name='signal')
signal[composite > 0.4] = 1
signal[composite < -0.4] = -1
signal = signal.fillna(0).astype(int)''',
    },
    {
        'name': 'MomentumContinuation',
        'factors': ['daily_mom', 'daily_ret_1d', 'momentum_1d'],
        'code': '''mom = factors['daily_mom']
ret = factors['daily_ret_1d']
mom2 = factors['momentum_1d']

w = 12
mom_z = (mom - mom.rolling(w).mean()) / (mom.rolling(w).std() + 1e-8)
ret_z = (ret - ret.rolling(w).mean()) / (ret.rolling(w).std() + 1e-8)
mom2_z = (mom2 - mom2.rolling(w).mean()) / (mom2.rolling(w).std() + 1e-8)

composite = (0.4 * mom_z + 0.3 * ret_z + 0.3 * mom2_z).fillna(0)
signal = pd.Series(0, index=close.index, name='signal')
signal[composite > 0.2] = 1
signal[composite < -0.2] = -1
signal = signal.fillna(0).astype(int)''',
    },
    {
        'name': 'HighFreqScalper',
        'factors': ['daily_close_return_96', 'DCP', 'london_mom'],
        'code': '''close_ret = factors['daily_close_return_96']
dcp = factors['DCP']
london = factors['london_mom']

w = 10
cr_z = (close_ret - close_ret.rolling(w).mean()) / (close_ret.rolling(w).std() + 1e-8)
dcp_z = (dcp - dcp.rolling(w).mean()) / (dcp.rolling(w).std() + 1e-8)
lon_z = (london - london.rolling(w).mean()) / (london.rolling(w).std() + 1e-8)

composite = (0.4 * cr_z + 0.3 * dcp_z + 0.3 * lon_z).fillna(0)
signal = pd.Series(0, index=close.index, name='signal')
signal[composite > 0.25] = 1
signal[composite < -0.25] = -1
signal = signal.fillna(0).astype(int)''',
    },
    {
        'name': 'AdaptiveMomentumMR',
        'factors': ['daily_close_return_96', 'daily_session_momentum_divergence_1d', 'daily_ols_slope_96'],
        'code': '''mom = factors['daily_close_return_96']
div = factors['daily_session_momentum_divergence_1d']
slope = factors['daily_ols_slope_96']

w = 20
mom_z = (mom - mom.rolling(w).mean()) / (mom.rolling(w).std() + 1e-8)
div_z = (div - div.rolling(w).mean()) / (div.rolling(w).std() + 1e-8)
slope_z = (slope - slope.rolling(w).mean()) / (slope.rolling(w).std() + 1e-8)

# Regime detection: high momentum = trend, low = mean reversion
regime = (mom_z.abs() > 1.0).astype(float)
composite = (regime * mom_z + (1 - regime) * (-div_z) + 0.3 * slope_z).fillna(0)
signal = pd.Series(0, index=close.index, name='signal')
signal[composite > 0.4] = 1
signal[composite < -0.4] = -1
signal = signal.fillna(0).astype(int)''',
    },
    {
        'name': 'TrendPullbackScalp',
        'factors': ['daily_close_return_96', 'daily_session_momentum_divergence_1d', 'daily_norm_body'],
        'code': '''mom = factors['daily_close_return_96']
div = factors['daily_session_momentum_divergence_1d']
body = factors['daily_norm_body']

w = 15
mom_z = (mom - mom.rolling(w).mean()) / (mom.rolling(w).std() + 1e-8)
div_z = (div - div.rolling(w).mean()) / (div.rolling(w).std() + 1e-8)
body_z = (body - body.rolling(w).mean()) / (body.rolling(w).std() + 1e-8)

# Enter on pullbacks (divergence against trend)
composite = (mom_z - 0.5 * div_z * mom_z.sign() + 0.2 * body_z).fillna(0)
signal = pd.Series(0, index=close.index, name='signal')
signal[composite > 0.35] = 1
signal[composite < -0.35] = -1
signal = signal.fillna(0).astype(int)''',
    },
    {
        'name': 'IntradayMomentumBlend',
        'factors': ['daily_close_return_96', 'london_mom', 'daily_session_momentum_divergence_1d', 'DCP'],
        'code': '''mom = factors['daily_close_return_96']
lon = factors['london_mom']
div = factors['daily_session_momentum_divergence_1d']
dcp = factors['DCP']

w = 20
mom_z = (mom - mom.rolling(w).mean()) / (mom.rolling(w).std() + 1e-8)
lon_z = (lon - lon.rolling(w).mean()) / (lon.rolling(w).std() + 1e-8)
div_z = (div - div.rolling(w).mean()) / (div.rolling(w).std() + 1e-8)
dcp_z = (dcp - dcp.rolling(w).mean()) / (dcp.rolling(w).std() + 1e-8)

composite = (0.3 * mom_z + 0.3 * lon_z - 0.2 * div_z + 0.2 * dcp_z).fillna(0)
signal = pd.Series(0, index=close.index, name='signal')
signal[composite > 0.3] = 1
signal[composite < -0.3] = -1
signal = signal.fillna(0).astype(int)''',
    },
]

def load_factor_series(name):
    """Load factor parquet and return as Series with correct index."""
    safe = name.replace('/','_').replace('\\','_')[:150]
    pf = VALUE_FILES / f"{safe}.parquet"
    if not pf.exists():
        return None
    
    df = pd.read_parquet(str(pf))
    
    # Extract EURUSD
    if df.index.names == ['datetime', 'instrument']:
        df_reset = df.reset_index()
        if 'instrument' in df_reset.columns:
            df_eur = df_reset[df_reset['instrument'] == 'EURUSD'].copy()
            df_eur = df_eur.set_index('datetime')
            series = df_eur.iloc[:, -1]  # Last column is the factor value
            series.name = name
            return series
    
    # If single index, just return first column
    series = df.iloc[:, 0]
    series.name = name
    return series

def main(n_strategies=5):
    console.print("[bold cyan]🎯 Daytrading Strategy Generator (Quick Mode)[/bold cyan]\n")
    console.print("   Style: 12-minute forward returns")
    console.print("   Target: FTMO compliant (IC>0.02, Sharpe>0.5, Trades>20, DD>-10%)\n")
    
    # Load OHLCV data
    if not OHLCV_PATH.exists():
        console.print(f"[red]✗ OHLCV data not found: {OHLCV_PATH}[/red]")
        return
    
    ohlcv = pd.read_hdf(str(OHLCV_PATH), key='data')
    
    # Extract close prices with datetime-only index (not MultiIndex)
    if '$close' in ohlcv.columns:
        close = ohlcv['$close'].dropna()
    elif 'close' in ohlcv.columns:
        close = ohlcv['close'].dropna()
    else:
        close = ohlcv.select_dtypes(include=[np.number]).iloc[:, 0].dropna()
    
    # Extract datetime from MultiIndex if present
    if isinstance(close.index, pd.MultiIndex):
        close_dt_idx = close.index.get_level_values('datetime')
        close_series = pd.Series(close.values, index=close_dt_idx, name='close')
    else:
        close_series = close
    
    close_series = close_series.dropna()
    console.print(f"[green]✓[/green] Loaded {len(close_series):,} OHLCV bars")
    
    # Load all factor series and align to close index
    all_factor_series = {}
    for combo in DAYTRADING_COMBOS:
        for factor_name in combo['factors']:
            if factor_name in all_factor_series:
                continue
            
            series = load_factor_series(factor_name)
            if series is not None:
                # Forward fill to match close frequency
                series_ff = series.reindex(close_series.index).ffill()
                all_factor_series[factor_name] = series_ff
    
    # Create factors DataFrame
    df_factors = pd.DataFrame(all_factor_series)
    df_factors = df_factors.dropna(how='all')
    
    console.print(f"[green]✓[/green] Loaded {len(df_factors.columns)} factor series")
    console.print(f"[green]✓[/green] Aligned to {len(df_factors):,} bars\n")
    
    accepted = []
    
    for i, combo in enumerate(DAYTRADING_COMBOS[:n_strategies]):
        console.print(f"[{i+1}/{n_strategies}] Testing {combo['name']}...")
        
        # Build factor dataframe
        valid_factors = [f for f in combo['factors'] if f in df_factors.columns]
        if len(valid_factors) < 2:
            console.print(f"  ✗ Not enough valid factors")
            continue
        
        strat_factors = df_factors[valid_factors].dropna()
        
        if len(strat_factors) < 1000:
            console.print(f"  ✗ Not enough data: {len(strat_factors)} bars")
            continue
        
        # Build backtest script
        forward_bars = 12
        strategy_code = combo['code']
        
        script = f"""
import pandas as pd
import numpy as np
import json

close = pd.read_pickle('close.pkl')  # nosec
factors = pd.read_pickle('factors.pkl')  # nosec

# Execute strategy
try:
{chr(10).join('    ' + l for l in strategy_code.split(chr(10)))}
except Exception as e:
    print(f"ERROR: {{e}}")
    exit(1)

if 'signal' not in dir():
    print("ERROR: No signal generated")
    exit(1)

signal = signal.fillna(0)

# Align
common_idx = close.index.intersection(signal.index)
close = close.loc[common_idx]
signal = signal.loc[common_idx]

# Forward returns (12-min horizon for daytrading)
FORWARD_BARS = {forward_bars}
returns_fwd = close.pct_change(FORWARD_BARS).shift(-FORWARD_BARS)
signal_aligned = signal.loc[returns_fwd.dropna().index]
fwd_returns = returns_fwd.loc[signal_aligned.index]

if len(signal_aligned) < 100 or len(fwd_returns) < 100:
    print("ERROR: Not enough data")
    exit(1)

# Metrics
ic = signal_aligned.corr(fwd_returns)
strategy_returns = signal_aligned * fwd_returns
sharpe = strategy_returns.mean() / strategy_returns.std() * np.sqrt(252 * 1440 / {forward_bars}) if strategy_returns.std() > 0 else 0

cum = (1 + strategy_returns).cumprod()
running_max = cum.expanding().max()
drawdown = (cum - running_max) / running_max.replace(0, np.nan)
max_dd = drawdown.min() if len(drawdown) > 0 else 0

win_rate = (strategy_returns > 0).sum() / len(strategy_returns) if len(strategy_returns) > 0 else 0
n_trades = int((signal_aligned != signal_aligned.shift(1)).sum())
total_return = cum.iloc[-1] - 1
n_bars = len(strategy_returns)
n_months = n_bars / (252 * 1440 / {forward_bars} / 12) if n_bars > 0 else 1
monthly_return = (1 + total_return) ** (1 / n_months) - 1 if n_months > 0 and (1 + total_return) > 0 else total_return

result = {{
    "status": "success",
    "sharpe": float(sharpe),
    "max_drawdown": float(max_dd) if not np.isnan(max_dd) else -0.20,
    "win_rate": float(win_rate),
    "ic": float(ic) if not np.isnan(ic) else 0,
    "n_trades": n_trades,
    "total_return": float(total_return),
    "monthly_return_pct": float(monthly_return * 100),
    "n_bars": int(n_bars),
    "n_months": float(n_months),
    "signal_long": int((signal_aligned == 1).sum()),
    "signal_short": int((signal_aligned == -1).sum()),
    "signal_neutral": int((signal_aligned == 0).sum()),
}}

print(json.dumps(result))
"""
        
        # Run backtest
        import tempfile
        with tempfile.TemporaryDirectory() as td:
            tdp = Path(td)
            strat_close = close_series.loc[strat_factors.index]
            strat_close.to_pickle(str(tdp / 'close.pkl'))  # nosec
            strat_factors.to_pickle(str(tdp / 'factors.pkl'))  # nosec
            
            script_path = tdp / 'run.py'
            script_path.write_text(script)
            
            try:
                result_proc = subprocess.run( # nosec B603
                    [sys.executable, str(script_path)],
                    capture_output=True, text=True, timeout=60,
                    cwd=str(tdp)
                )
                
                if result_proc.returncode != 0:
                    console.print(f"  ✗ Failed: {result_proc.stderr[:200]}")
                    continue
                
                result = None
                for line in result_proc.stdout.strip().split('\n'):
                    try:
                        result = json.loads(line)
                        break
                    except:
                        continue
                
                if not result or result.get('status') != 'success':
                    console.print(f"  ✗ Invalid result")
                    continue
                
            except subprocess.TimeoutExpired:  # nosec
                console.print(f"  ✗ Timeout")
                continue
            except Exception as e:
                console.print(f"  ✗ Error: {e}")
                continue
        
        ic = result.get('ic', 0)
        sharpe = result.get('sharpe', 0)
        trades = result.get('n_trades', 0)
        dd = result.get('max_drawdown', 0)
        
        # FTMO criteria
        if abs(ic) > 0.02 and sharpe > 0.5 and trades > 20 and dd > -0.10:
            strategy = {
                'strategy_name': combo['name'],
                'factor_names': combo['factors'],
                'description': f"Daytrading strategy combining {', '.join(combo['factors'])}",
                'code': combo['code'],
                'real_backtest': result,
                'metrics': result,
                'summary': {
                    'sharpe': sharpe,
                    'max_drawdown': dd,
                    'win_rate': result.get('win_rate', 0),
                    'monthly_return_pct': result.get('monthly_return_pct', 0),
                    'real_ic': ic,
                    'real_n_trades': trades,
                    'forward_bars': 12,
                    'trading_style': 'daytrading',
                }
            }
            
            fname = f"{int(time.time())}_{combo['name']}.json"
            with open(STRATEGIES_DIR / fname, 'w') as f:
                json.dump(strategy, f, indent=2, ensure_ascii=False)
            
            accepted.append(strategy)
            console.print(f"  ✓ [green]ACCEPT[/green]: IC={ic:.4f}, Sharpe={sharpe:.2f}, Trades={trades}, DD={dd:.1%}")
        else:
            console.print(f"  ✗ [red]REJECT[/red]: IC={ic:.4f}, Sharpe={sharpe:.2f}, Trades={trades}, DD={dd:.1%}")
    
    console.print(f"\n[bold green]✓ {len(accepted)}/{n_strategies} strategies accepted[/bold green]\n")
    
    if accepted:
        console.print("[bold]Results:[/bold]")
        for s in accepted:
            bt = s['real_backtest']
            console.print(f"  • {s['strategy_name']:30s} IC={bt['ic']:.4f} Sharpe={bt['sharpe']:.2f} "
                          f"Monthly={bt['monthly_return_pct']:.2f}% Trades={bt['n_trades']}")

if __name__ == '__main__':
    import sys
    n = int(sys.argv[1]) if len(sys.argv) > 1 else 5
    main(n)
