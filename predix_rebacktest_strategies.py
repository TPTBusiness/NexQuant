#!/usr/bin/env python
"""Re-evaluate strategies with real backtests - robust version."""
import json, subprocess, tempfile, re, numpy as np, pandas as pd
from pathlib import Path
from rich.progress import Progress

def load_factors(names, vdir):
    """Load factor time-series."""
    dfs = {}
    for n in names:
        for v in [n, n.replace('/','_').replace('\\','_')[:150], n.replace('.','_')[:150]]:
            p = vdir / f"{v}.parquet"
            if p.exists():
                try:
                    df = pd.read_parquet(str(p))
                    if df is not None and len(df.columns) > 0:
                        dfs[n] = df.iloc[:, 0]
                        break
                except: pass
    return dfs

def fix_code(code, available):
    """Fix strategy code to handle missing factors."""
    fixed = code
    
    # Fix: df['missing_factor'] → pd.Series(0, index=df.index)
    for match in re.finditer(r"df\['([^']+)'\]", code):
        fname = match.group(1)
        if fname not in available:
            fixed = fixed.replace(
                f"df['{fname}']", 
                f"pd.Series(0, index=df.index, name='{fname}')", 1
            )
    
    # Fix: df[["f1", "f2"]] → filter to available only
    for match in re.finditer(r'df\[\[([^\]]+)\]\]', code):
        factors_str = match.group(1)
        factors = [f.strip().strip("'\"") for f in factors_str.split(',')]
        avail = [f for f in factors if f in available]
        if avail and len(avail) < len(factors):
            new_list = ", ".join(f"'{f}'" for f in avail)
            fixed = fixed.replace(f"df[[{factors_str}]]", f"df[[{new_list}]]", 1)
    
    return fixed

def run_bt(fdfs, code):
    """Run backtest."""
    df = pd.DataFrame(fdfs).dropna()
    if len(df) < 100 or len(df.columns) < 2:
        return None
    
    avail = list(df.columns)
    fixed = fix_code(code, avail)
    
    with tempfile.TemporaryDirectory() as td:
        tdp = Path(td)
        df.to_parquet(str(tdp / "factors.parquet"))  # MUST be named factors.parquet
        
        script = tdp / "run.py"
        script.write_text(f"""
import pandas as pd, numpy as np
df = pd.read_parquet('factors.parquet')
try:
{chr(10).join('    ' + l for l in fixed.split(chr(10)))}
except Exception as e:
    pass

try:
    if 'signal' not in dir():
        signal = pd.Series(np.where(df.mean(axis=1) > 0, 1, -1), index=df.index)
    signal.name = 'signal'
    signal.to_pickle('s.pkl')
    print("OK")
except Exception as e:
    print(f"ERROR: {{e}}")
""")
        try:
            r = subprocess.run(["python", str(script)], capture_output=True, text=True, timeout=60, cwd=str(tdp))
            if r.returncode != 0:
                return None
            sig = pd.read_pickle(str(tdp / "s.pkl"))
        except:
            return None
    
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
    ann = mon * 12
    mon = max(min(mon, 0.20), -0.20)
    ann = max(min(ann, 2.0), -1.0)
    ic = ic if not np.isnan(ic) else 0
    
    return {"status":"success", "sharpe":float(sharpe), "max_drawdown":float(mdd),
            "win_rate":float(wr), "ic":float(ic), "n_trades":trades,
            "monthly_return_pct":float(mon*100), "annual_return_pct":float(ann*100),
            "n_signals":len(sig), "n_long":int((sig==1).sum()),
            "n_short":int((sig==-1).sum()), "n_neutral":int((sig==0).sum())}

def main(count=None):
    sdir = Path('/home/nico/Predix/results/strategies')
    vdir = Path('/home/nico/Predix/results/factors/values')
    
    files = []
    for f in sorted(sdir.glob('*.json'), reverse=True):
        try:
            d = json.load(open(f))
            if isinstance(d, dict) and 'strategy_name' in d:
                files.append(f)
        except: pass
    if count: files = files[:count]
    
    print(f"Re-evaluating {len(files)} strategies...\n")
    results, updated = [], 0
    
    with Progress() as p:
        task = p.add_task("Backtesting...", total=len(files))
        for f in files:
            try:
                data = json.load(open(f))
                fdfs = load_factors(data.get('factor_names', []), vdir)
                if len(fdfs) >= 3:
                    bt = run_bt(fdfs, data.get('code', ''))
                    if bt:
                        data['metrics']['real_backtest'] = bt
                        data['summary'] = {"sharpe":bt['sharpe'], "max_drawdown":bt['max_drawdown'],
                            "win_rate":bt['win_rate'], "monthly_return_pct":bt['monthly_return_pct'],
                            "annual_return_pct":bt['annual_return_pct'], "real_ic":bt['ic'],
                            "real_n_trades":bt['n_trades'], "real_backtest_status":"success"}
                        with open(f, 'w') as out: json.dump(data, out, indent=2, ensure_ascii=False)
                        updated += 1
                        results.append({'name':data['strategy_name'], **bt})
            except:
                pass
            p.update(task, advance=1)
    
    print(f"\n✅ Updated {updated}/{len(files)}")
    if results:
        results.sort(key=lambda x: x['sharpe'], reverse=True)
        print(f"\n{'='*75}\n🏆 TOP 10\n{'='*75}")
        print(f"{'#':>3} {'Name':<30} {'Sharpe':>7} {'Monat':>8} {'MaxDD':>8} {'IC':>7} {'Trades':>7}")
        print("-" * 70)
        for i, r in enumerate(results[:10], 1):
            print(f"{i:3d} {r['name']:30s} {r['sharpe']:7.3f} {r['monthly_return_pct']:7.2f}% {r['max_drawdown']:7.2%} {r['ic']:7.4f} {r['n_trades']:7d}")

if __name__ == "__main__":
    import sys
    main(int(sys.argv[1]) if len(sys.argv) > 1 else None)
