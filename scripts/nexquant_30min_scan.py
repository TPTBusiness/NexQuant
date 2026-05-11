#!/usr/bin/env python
"""30min Full Factor Scan — find all profitable signals."""
import json, numpy as np, pandas as pd
from pathlib import Path
from rdagent.components.backtesting.vbt_backtest import backtest_signal_ftmo

c = pd.read_hdf("git_ignore_folder/factor_implementation_source_data/intraday_pv.h5", key="data")["$close"]
c = c.droplevel(-1).sort_index().dropna().resample("30min").last().dropna()
is_s = (c.index.hour >= 7) & (c.index.hour < 17)
F = Path("results/factors"); V = F / "values"

factors = []
for f in sorted(F.glob("*.json")):
    try: d = json.loads(f.read_text())
    except: continue
    if d.get("status") != "success" or d.get("ic") is None: continue
    name = d.get("factor_name", f.stem)
    safe = name.replace("/", "_")[:150]
    if (V / f"{safe}.parquet").exists():
        factors.append({"name": name, "ic": d["ic"], "safe": safe})
factors.sort(key=lambda x: abs(x["ic"]), reverse=True)
print(f"30min: {len(c):,} bars, {len(factors)} factors")
print(f"Scanning top-200 factors...")

results = []
for i, f in enumerate(factors[:200]):
    try:
        s = pd.read_parquet(V / f"{f['safe']}.parquet").iloc[:, 0]
        if isinstance(s.index, pd.MultiIndex): s = s.droplevel(-1)
        fac = s.resample("30min").last().reindex(c.index).ffill()
    except: continue
    for dr in [1, -1]:
        sig = pd.Series(dr * np.sign(fac).fillna(0), index=c.index)
        sig[~is_s] = 0
        if sig.abs().sum() < 20: continue
        r = backtest_signal_ftmo(c, sig.fillna(0), txn_cost_bps=2.14)
        oos = r.get("wf_oos_sharpe_mean") or r.get("oos_sharpe", -999)
        oos_m = r.get("oos_monthly_return_pct", 0) or 0
        if oos_m > 0.2:
            results.append((f"{f['name']}_{dr}", oos, oos_m, r.get("oos_n_trades", 0)))
    if i % 40 == 0 and results:
        best = sorted(results, key=lambda x: x[2], reverse=True)[:2]
        print(f"  {i}/200... best: {best[0][0][:40]} Mon={best[0][2]:+.2f}%")

results.sort(key=lambda x: x[2], reverse=True)
print(f"\nProfitable (>0.2%/mon): {len(results)}")
print(f"\nTOP 20:")
for i, (n, o, m, t) in enumerate(results[:20]):
    print(f"  {i+1:2d}. {n[:52]:52s} OOS={o:+8.1f} Mon={m:+7.2f}% T={t:5d}")

# Save top signals for combo testing
if results:
    top = results[:15]
    all_sig = {}
    for name, oos, mon, t in top:
        fn = name.rsplit("_", 1)[0]
        dr = -1 if name.endswith("_-1") else 1
        if dr == -1: dr = -1
        safe = fn.replace("/", "_")[:150]
        try:
            s = pd.read_parquet(V / f"{safe}.parquet").iloc[:, 0]
            if isinstance(s.index, pd.MultiIndex): s = s.droplevel(-1)
            fac = s.resample("30min").last().reindex(c.index).ffill()
            sig = pd.Series(dr * np.sign(fac).fillna(0), index=c.index)
            sig[~is_s] = 0
            all_sig[name] = sig
        except: pass
    
    if all_sig:
        df = pd.DataFrame(all_sig, index=c.index).fillna(0)
        cols = list(df.columns)
        print(f"\n=== COMBO TESTS ===")
        for n in [2, 3, 5, 8, len(cols)]:
            combo = df[cols[:n]].mean(axis=1)
            r = backtest_signal_ftmo(c, combo.fillna(0), txn_cost_bps=2.14, wf_rolling=True)
            m = r.get("oos_monthly_return_pct", 0) or 0
            dd = (r.get("oos_max_drawdown", 0) or 0) * 100
            t = r.get("oos_n_trades", 0)
            hit = "🎯" if m >= 4 else "✅" if m > 0 else ""
            print(f"  {n:2d} sig: Mon={m:+.2f}% DD={dd:+.1f}% T={t} {hit}")

print("\nDone!")
