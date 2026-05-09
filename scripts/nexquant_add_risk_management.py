#!/usr/bin/env python
"""
Add FTMO-compliant risk management to existing strategies.

For each accepted strategy, add:
- Stop Loss: 2%
- Take Profit: 4% (2x SL)
- Trailing Stop: 1.5% after 2% profit
- Re-evaluate with risk management
- Generate Live Trading report

Usage:
    python nexquant_add_risk_management.py
    python nexquant_add_risk_management.py --live  # Mark as live-ready
"""
import os, sys, json, time
from pathlib import Path
from datetime import datetime
import numpy as np
import pandas as pd
from rich.console import Console
from rich.table import Table
from rich.progress import Progress, SpinnerColumn, TextColumn, BarColumn

console = Console()

STRATEGIES_DIR = Path('results/strategies_new')
OHLCV_PATH = Path('git_ignore_folder/factor_implementation_source_data/intraday_pv.h5')

# FTMO Risk Parameters
STOP_LOSS = 0.02      # 2% hard stop
TAKE_PROFIT = 0.04    # 4% target (2x SL)
TRAILING_STOP = 0.015 # 1.5% trail after 2% profit
MAX_DAILY_LOSS = 0.05 # 5% FTMO daily limit

def load_ohlcv():
    """Load OHLCV close prices."""
    ohlcv = pd.read_hdf(str(OHLCV_PATH), key='data')
    if '$close' in ohlcv.columns:
        close = ohlcv['$close'].dropna()
    elif 'close' in ohlcv.columns:
        close = ohlcv['close'].dropna()
    else:
        close = ohlcv.select_dtypes(include=[np.number]).iloc[:, 0].dropna()
    
    if isinstance(close.index, pd.MultiIndex):
        close_dt_idx = close.index.get_level_values('datetime')
        close = pd.Series(close.values, index=close_dt_idx, name='close')
    
    return close.dropna()

def apply_risk_management(signal, close, sl=0.02, tp=0.04, trailing=0.015):
    """
    Apply Stop Loss, Take Profit, and Trailing Stop to strategy.
    
    Returns strategy returns after risk management.
    """
    FORWARD_BARS = 12  # 12-min forward returns for daytrading
    returns_fwd = close.pct_change(FORWARD_BARS).shift(-FORWARD_BARS)
    
    signal_aligned = signal.loc[returns_fwd.dropna().index]
    fwd_returns = returns_fwd.loc[signal_aligned.index]
    
    if len(signal_aligned) < 100:
        return None, None
    
    strategy_returns = pd.Series(0.0, index=fwd_returns.index)
    position = 0
    entry_price = 0
    peak_pnl = 0
    
    for i in range(len(fwd_returns)):
        sig = signal_aligned.iloc[i]
        ret = fwd_returns.iloc[i]
        
        if position != 0:
            # Calculate PnL
            pnl = position * ret
            
            # Check Stop Loss
            if pnl <= -sl:
                strategy_returns.iloc[i] = -sl
                position = 0
                peak_pnl = 0
                continue
            
            # Check Take Profit
            if pnl >= tp:
                strategy_returns.iloc[i] = tp
                position = 0
                peak_pnl = 0
                continue
            
            # Check Trailing Stop
            if pnl > 0.02:  # After 2% profit
                peak_pnl = max(peak_pnl, pnl)
                if pnl < peak_pnl - trailing:
                    strategy_returns.iloc[i] = peak_pnl - trailing
                    position = 0
                    peak_pnl = 0
                    continue
            
            strategy_returns.iloc[i] = pnl
            peak_pnl = max(peak_pnl, pnl)
        
        elif sig != 0:
            # Enter position
            position = sig
            entry_price = close.iloc[i] if i < len(close) else 1.0
    
    return strategy_returns, signal_aligned

def evaluate_strategy(strategy_returns, signal_aligned):
    """Calculate comprehensive metrics."""
    if strategy_returns is None or len(strategy_returns) < 100:
        return None
    
    ic = signal_aligned.corr(strategy_returns / (strategy_returns.std() + 1e-8)) if strategy_returns.std() > 0 else 0
    
    sharpe = strategy_returns.mean() / (strategy_returns.std() + 1e-8) * np.sqrt(252 * 1440 / 12)
    
    cum = (1 + strategy_returns).cumprod()
    running_max = cum.expanding().max()
    drawdown = (cum - running_max) / running_max.replace(0, np.nan)
    max_dd = drawdown.min() if len(drawdown) > 0 else 0
    
    win_rate = (strategy_returns > 0).sum() / len(strategy_returns)
    n_trades = int((signal_aligned != signal_aligned.shift(1)).sum())
    total_return = cum.iloc[-1] - 1
    n_bars = len(strategy_returns)
    n_months = n_bars / (252 * 1440 / 12 / 12) if n_bars > 0 else 1
    
    monthly_return = (1 + total_return) ** (1 / n_months) - 1 if n_months > 0 and (1 + total_return) > 0 else total_return
    
    # Daily loss check
    daily_returns = strategy_returns.groupby(strategy_returns.index.date if hasattr(strategy_returns.index[0], 'date') else strategy_returns.index).sum()
    max_daily_loss = abs(daily_returns.min()) if len(daily_returns) > 0 else 0
    
    return {
        'ic': float(ic) if not np.isnan(ic) else 0,
        'sharpe': float(sharpe),
        'max_drawdown': float(max_dd) if not np.isnan(max_dd) else 0,
        'win_rate': float(win_rate),
        'n_trades': n_trades,
        'total_return': float(total_return),
        'monthly_return_pct': float(monthly_return * 100),
        'n_bars': int(n_bars),
        'n_months': float(n_months),
        'max_daily_loss': float(max_daily_loss),
        'ftmo_compliant': max_daily_loss <= MAX_DAILY_LOSS and max_dd > -0.10,
    }

def main():
    console.print("[bold cyan]🔒 Adding FTMO Risk Management to Existing Strategies[/bold cyan]\n")
    
    # Load OHLCV
    console.print("📊 Loading OHLCV data...")
    close = load_ohlcv()
    console.print(f"   ✓ Loaded {len(close):,} bars\n")
    
    # Load strategies
    strategies = []
    for f in sorted(STRATEGIES_DIR.glob('*.json')):
        try:
            data = json.load(open(f))
            bt = data.get('real_backtest', {})
            if bt.get('status') == 'success':
                strategies.append((f, data))
        except:
            pass
    
    console.print(f"📁 Found {len(strategies)} accepted strategies\n")
    
    # Process each strategy
    results = []
    
    with Progress(
        SpinnerColumn(),
        TextColumn("[bold blue]{task.description}"),
        BarColumn(),
        TextColumn("[bold green]{task.completed}/{task.total}"),
    ) as progress:
        task = progress.add_task("Processing...", total=len(strategies))
        
        for fpath, data in strategies:
            name = data.get('strategy_name', 'Unknown')
            progress.update(task, description=f"Processing {name}...")
            
            # Load factors
            factor_names = data.get('factor_names', [])
            
            # Load factor parquet files
            factors_data = {}
            for fname in factor_names:
                safe = fname.replace('/', '_').replace('\\', '_')[:150]
                pf = Path('results/factors/values') / f"{safe}.parquet"
                if pf.exists():
                    try:
                        df = pd.read_parquet(str(pf))
                        if df.index.names == ['datetime', 'instrument']:
                            df_reset = df.reset_index()
                            if 'instrument' in df_reset.columns:
                                df_eur = df_reset[df_reset['instrument'] == 'EURUSD'].copy()
                                df_eur = df_eur.set_index('datetime')
                                factors_data[fname] = df_eur.iloc[:, -1]
                    except:
                        pass
            
            if len(factors_data) < 2:
                progress.update(task, advance=1)
                continue
            
            # Build factors DataFrame
            df_factors = pd.DataFrame(factors_data)
            common_idx = close.index.intersection(df_factors.dropna(how='all').index)
            close_aligned = close.loc[common_idx]
            df_aligned = df_factors.loc[common_idx]
            
            # Execute strategy code
            try:
                local_vars = {'factors': df_aligned, 'close': close_aligned}
                exec(data.get('code', ''), {}, local_vars)
                signal = local_vars.get('signal', pd.Series(0, index=close_aligned.index))
            except:
                progress.update(task, advance=1)
                continue
            
            # Apply risk management
            strat_returns, sig_aligned = apply_risk_management(
                signal, close_aligned,
                sl=STOP_LOSS, tp=TAKE_PROFIT, trailing=TRAILING_STOP
            )
            
            if strat_returns is None:
                progress.update(task, advance=1)
                continue
            
            # Evaluate
            metrics = evaluate_strategy(strat_returns, sig_aligned)
            if metrics is None:
                progress.update(task, advance=1)
                continue
            
            # Store result
            result = {
                'name': name,
                'file': fpath.name,
                'original_ic': data.get('real_backtest', {}).get('ic', 0),
                'original_sharpe': data.get('real_backtest', {}).get('sharpe', 0),
                'new_ic': metrics['ic'],
                'new_sharpe': metrics['sharpe'],
                'new_max_dd': metrics['max_drawdown'],
                'new_win_rate': metrics['win_rate'],
                'new_trades': metrics['n_trades'],
                'new_monthly_ret': metrics['monthly_return_pct'],
                'max_daily_loss': metrics['max_daily_loss'],
                'ftmo_compliant': bool(metrics['ftmo_compliant']),
            }
            results.append(result)
            
            # Update strategy JSON
            data['risk_management'] = {
                'stop_loss': STOP_LOSS,
                'take_profit': TAKE_PROFIT,
                'trailing_stop': TRAILING_STOP,
                'trailing_trigger': 0.02,
                'max_daily_loss': MAX_DAILY_LOSS,
                'ftmo_compliant': bool(metrics['ftmo_compliant']),
            }
            data['evaluated_with_risk_mgmt'] = metrics
            data['summary'] = {
                'sharpe': metrics['sharpe'],
                'max_drawdown': metrics['max_drawdown'],
                'win_rate': metrics['win_rate'],
                'monthly_return_pct': metrics['monthly_return_pct'],
                'real_ic': metrics['ic'],
                'real_n_trades': metrics['n_trades'],
                'ftmo_compliant': bool(metrics['ftmo_compliant']),
                'forward_bars': 12,
                'trading_style': 'daytrading',
            }
            
            with open(fpath, 'w') as f:
                # Convert numpy types for JSON
                def sanitize(obj):
                    if hasattr(obj, 'item'): return obj.item()
                    if isinstance(obj, dict): return {k: sanitize(v) for k, v in obj.items()}
                    if isinstance(obj, list): return [sanitize(v) for v in obj]
                    if isinstance(obj, (np.bool_, bool)): return bool(obj)
                    return obj
                
                json.dump(sanitize(data), f, indent=2, ensure_ascii=False)
            
            progress.update(task, advance=1)
    
    # Display results
    console.print("\n[bold green]✓ All strategies processed![/bold green]\n")
    
    table = Table(title="📊 FTMO Risk Management Results")
    table.add_column("#", justify="right")
    table.add_column("Strategy", style="cyan")
    table.add_column("IC", justify="right")
    table.add_column("Sharpe", justify="right")
    table.add_column("Trades", justify="right")
    table.add_column("Monthly %", justify="right")
    table.add_column("Max DD", justify="right")
    table.add_column("FTMO", justify="center")
    
    results.sort(key=lambda x: x['new_sharpe'], reverse=True)
    for i, r in enumerate(results, 1):
        ftmo = "✅" if r['ftmo_compliant'] else "❌"
        table.add_row(
            str(i), r['name'],
            f"{r['new_ic']:.4f}",
            f"{r['new_sharpe']:.2f}",
            str(r['new_trades']),
            f"{r['new_monthly_ret']:.2f}%",
            f"{r['new_max_dd']:.1%}",
            ftmo
        )
    
    console.print(table)
    
    # Summary
    ftmo_count = sum(1 for r in results if r['ftmo_compliant'])
    console.print(f"\n[bold]FTMO-Compliant:[/bold] {ftmo_count}/{len(results)} strategies")
    
    if results:
        best = results[0]
        console.print(f"\n[bold green]🏆 Best Strategy: {best['name']}[/bold green]")
        console.print(f"   Sharpe: {best['new_sharpe']:.2f}")
        console.print(f"   Monthly Return: {best['new_monthly_ret']:.2f}%")
        console.print(f"   Max Drawdown: {best['new_max_dd']:.1%}")
        console.print(f"   FTMO Compliant: {'✅' if best['ftmo_compliant'] else '❌'}")

if __name__ == '__main__':
    main()
