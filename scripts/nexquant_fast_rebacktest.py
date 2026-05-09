#!/usr/bin/env python
"""Fast rebacktest: only strategies with factor parquets, skip already-done."""
import json, sys, pandas as pd, subprocess, tempfile, numpy as np
from pathlib import Path
from datetime import datetime

sys.path.insert(0, str(Path(__file__).resolve().parent))
from rdagent.components.backtesting.vbt_backtest import backtest_signal

OHLCV = Path("git_ignore_folder/factor_implementation_source_data/intraday_pv.h5")
FACTORS_DIR = Path("results/factors/values")
STRAT_DIR = Path("results/strategies_new")

# Pre-build factor name → path map
fmap = {p.stem: str(p) for p in FACTORS_DIR.glob("*.parquet")}

# Load close once
print("Loading OHLCV...")
ohlcv = pd.read_hdf(str(OHLCV), key="data")
close = ohlcv["$close"].dropna()
if isinstance(close.index, pd.MultiIndex):
    close = close.droplevel(-1)
close = close.astype(float).sort_index()
print(f"{len(close):,} bars")

# Build work list
work = []
for f in sorted(STRAT_DIR.glob("*.json")):
    try:
        d = json.loads(f.read_text())
    except Exception:
        continue
    if d.get("reevaluation_status") == "verified_v2":
        continue
    names = d.get("factor_names", [])
    code = d.get("code", "")
    if not names or not code:
        continue
    paths = []
    for n in names:
        p = fmap.get(n) or fmap.get(n.replace("/", "_")[:150])
        if p:
            paths.append((n, p))
    if len(paths) >= 2:
        work.append((f, d, paths))

print(f"{len(work)} strategies to process")

if not work:
    print("All done!")
    sys.exit(0)

ok = skip = fail = 0
start = datetime.now()

for i, (f, data, factor_paths) in enumerate(work):
    name = data.get("strategy_name", f.stem)[:45]
    code = data.get("code", "")

    # Load factor series
    series = {}
    for fn, fp in factor_paths:
        try:
            s = pd.read_parquet(fp).iloc[:, 0]
            series[fn] = s
        except Exception:
            pass

    if len(series) < 2:
        skip += 1
        continue

    df = pd.DataFrame(series).sort_index()
    if isinstance(df.index, pd.MultiIndex):
        df = df.droplevel(-1)

    try:
        df_1m = df.reindex(close.index).ffill()
    except Exception:
        skip += 1
        continue

    valid = df_1m.notna().any(axis=1)
    if valid.sum() < 1000:
        skip += 1
        continue

    ca = close.loc[valid]
    fa = df_1m.loc[valid]

    # Execute strategy code
    try:
        with tempfile.TemporaryDirectory() as td:
            tdp = Path(td)
            fa.to_parquet(str(tdp / "factors.parquet"))
            ca.to_pickle(str(tdp / "close.pkl"))

            exec_script = (
                "import pandas as pd, numpy as np\n"
                "factors = pd.read_parquet('factors.parquet')\n"
                "close = pd.read_pickle('close.pkl')\n"
                "df = factors\n"
                + code +
                "\nif 'signal' not in dir():\n"
                "    raise SystemExit(1)\n"
                "pd.Series(signal).fillna(0).to_pickle('signal.pkl')\n"
            )
            (tdp / "run.py").write_text(exec_script)
            r = subprocess.run(
                ["python", "run.py"],
                capture_output=True, text=True, timeout=60, cwd=str(tdp),
            )
            if r.returncode != 0:
                fail += 1
                continue
            sig = pd.read_pickle(tdp / "signal.pkl")
    except Exception:
        fail += 1
        continue

    try:
        sig = sig.reindex(ca.index).ffill().fillna(0)
        result = backtest_signal(ca, sig, txn_cost_bps=2.14)
    except Exception:
        fail += 1
        continue

    # Write back
    data["reevaluation_status"] = "verified_v2"
    data["sharpe_ratio"] = result.get("sharpe")
    data["max_drawdown"] = result.get("max_drawdown")
    data["win_rate"] = result.get("win_rate")
    data["total_return"] = result.get("total_return")
    data["summary"] = {
        **data.get("summary", {}),
        "sharpe": result.get("sharpe"),
        "max_drawdown": result.get("max_drawdown"),
        "win_rate": result.get("win_rate"),
        "monthly_return_pct": result.get("monthly_return_pct"),
        "real_n_trades": result.get("n_trades"),
        "total_return": result.get("total_return"),
        "annualized_return": result.get("annualized_return"),
        "engine": "verified_v2",
        "txn_cost_bps": 2.14,
    }
    f.write_text(json.dumps(data, indent=2, ensure_ascii=False))
    ok += 1

    elapsed = (datetime.now() - start).total_seconds()
    rate = ok / elapsed * 60 if elapsed > 0 else 0
    print(f"  [{ok:4d}/{len(work)}] {rate:5.0f}/min  {name:45s}  "
          f"S={result['sharpe']:6.1f} DD={result['max_drawdown']:7.2%} "
          f"WR={result['win_rate']:5.1%} T={result['n_trades']:4d}")

elapsed = (datetime.now() - start).total_seconds()
print(f"\nDONE: ok={ok} skip={skip} fail={fail} in {elapsed:.0f}s")
