import json, numpy as np, pandas as pd
from pathlib import Path
from rdagent.components.backtesting.vbt_backtest import backtest_signal_ftmo

close = pd.read_hdf("git_ignore_folder/factor_implementation_source_data/intraday_pv.h5", key="data")["$close"]
close = close.droplevel(-1).sort_index().dropna().resample("1h").last().dropna()
print(f"1h bars: {len(close):,}")

FACTORS_DIR = Path("results/factors"); VALS = FACTORS_DIR / "values"
factors = []
for f in sorted(FACTORS_DIR.glob("*.json")):
    try: d = json.loads(f.read_text())
    except: continue
    if d.get("status") != "success" or d.get("ic") is None: continue
    name = d.get("factor_name", f.stem)
    safe = name.replace("/", "_")[:150]
    if (VALS / f"{safe}.parquet").exists():
        factors.append({"name": name, "ic": d["ic"], "safe": safe})

factors.sort(key=lambda x: abs(x["ic"]), reverse=True)
print(f"Testing top-100 factors by |IC|...")

results = []
is_session = (close.index.hour >= 7) & (close.index.hour < 17)

for i, f in enumerate(factors[:100]):
    try:
        s = pd.read_parquet(VALS / f"{f['safe']}.parquet").iloc[:, 0]
        if isinstance(s.index, pd.MultiIndex): s = s.droplevel(-1)
        fac = s.resample("1h").last().reindex(close.index).ffill()
    except: continue
    
    for dr, label in [(1, "STD"), (-1, "INV")]:
        sig = pd.Series(dr * np.sign(fac).fillna(0), index=close.index)
        sig[~is_session] = 0
        if sig.abs().sum() < 20: continue
        r = backtest_signal_ftmo(close, sig.fillna(0), txn_cost_bps=2.14)
        oos = r.get("wf_oos_sharpe_mean") or r.get("oos_sharpe", -999)
        oos_m = r.get("oos_monthly_return_pct", 0) or 0
        results.append((f"{f['name']}_{label}", oos, oos_m, r.get("oos_n_trades",0)))

    if i % 25 == 0:
        bests = sorted(results, key=lambda x: x[1], reverse=True)[:3]
        print(f"  {i}/100... best: {bests[0][0][:35]} OOS={bests[0][1]:+.1f}")

results.sort(key=lambda x: x[1], reverse=True)
print(f"\nTop 15 — 1h Factor Signals (Session-Filtered):")
for i, (name, oos, mon, t) in enumerate(results[:15]):
    s = "✅" if mon > 0 else ""
    print(f"  {i+1:2d}. {name[:50]:50s} OOS={oos:+8.1f} Mon={mon:+7.3f}% T={t:5d} {s}")

# Combine best
top = [r for r in results if r[2] > 0][:8]
if top:
    all_sig = {}
    for name, oos, mon, t in top:
        fn = name.rsplit("_", 1)[0]; dr = 1 if name.endswith("_STD") else -1
        safe = fn.replace("/", "_")[:150]
        try:
            s = pd.read_parquet(VALS/f"{safe}.parquet").iloc[:, 0]
            if isinstance(s.index, pd.MultiIndex): s = s.droplevel(-1)
            fac = s.resample("1h").last().reindex(close.index).ffill()
            sig = pd.Series(dr * np.sign(fac).fillna(0), index=close.index)
            sig[~is_session] = 0; all_sig[name] = sig
        except: pass
    
    df = pd.DataFrame(all_sig, index=close.index).fillna(0)
    for n in [3, 5, 8]:
        combo = df[list(df.columns)[:n]].mean(axis=1)
        r = backtest_signal_ftmo(close, combo.fillna(0), txn_cost_bps=2.14, wf_rolling=True)
        oos_m = r.get("oos_monthly_return_pct",0) or 0
        dd = (r.get("oos_max_drawdown",0) or 0)*100
        ann = ((1+oos_m/100)**12-1)*100
        print(f"  Top-{n} combo: Mon={oos_m:+.3f}% Ann={ann:+.1f}% DD={dd:+.1f}% T={r.get('oos_n_trades',0)}")

print("\nDone")
