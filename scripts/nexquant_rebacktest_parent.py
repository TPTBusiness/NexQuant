#!/usr/bin/env python
"""Parent orchestrator: calls nexquant_rebacktest_one.py for each strategy."""
import json, subprocess, sys
from pathlib import Path
from datetime import datetime

STRAT_DIR = Path("results/strategies_new")

# Build work list
work = []
for f in sorted(STRAT_DIR.glob("*.json")):
    if "verified_v2" in f.read_text():
        continue
    try:
        d = json.loads(f.read_text())
    except Exception:
        continue
    if d.get("factor_names") and d.get("code"):
        work.append(f)

print(f"{len(work)} strategies to re-backtest", flush=True)

ok = skip = fail = 0
start = datetime.now()

for i, f in enumerate(work):
    name = f.stem[:45]
    print(f"[{i+1}/{len(work)}] {name} ...", end=" ", flush=True)
    try:
        r = subprocess.run(
            ["timeout", "-s", "KILL", "90", "python", "scripts/nexquant_rebacktest_one.py", str(f)],
            capture_output=True, text=True, timeout=120,
            stdin=subprocess.DEVNULL,
        )
        result = json.loads(r.stdout.strip() or "{}")
    except subprocess.TimeoutExpired:
        print("TIMEOUT", flush=True)
        fail += 1
        continue
    except Exception as e:
        print(f"ERROR: {e}", flush=True)
        fail += 1
        continue

    if result.get("status") == "ok":
        data = json.loads(f.read_text())
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
        print(f"S={result['sharpe']:.1f} DD={result['max_drawdown']:.2%} WR={result['win_rate']:.1%} T={result['n_trades']}", flush=True)
    elif result.get("status") == "skipped":
        skip += 1
        print(f"SKIP: {result.get('reason', '?')}", flush=True)
    else:
        fail += 1
        print(f"FAIL: {result.get('stderr', result.get('error', '?'))[:100]}", flush=True)

elapsed = (datetime.now() - start).total_seconds()
print(f"\nDONE: ok={ok} skip={skip} fail={fail} in {elapsed:.0f}s", flush=True)
