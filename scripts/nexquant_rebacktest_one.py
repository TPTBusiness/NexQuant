#!/usr/bin/env python
"""One strategy runner — standalone, called from parent script."""
import json, sys, pandas as pd, subprocess, tempfile, numpy as np
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
from rdagent.components.backtesting.vbt_backtest import backtest_signal

if len(sys.argv) < 2:
    print("Usage: python nexquant_rebacktest_one.py <strategy_json_path>")
    sys.exit(1)

strat_path = Path(sys.argv[1])
data = json.loads(strat_path.read_text())

OHLCV = Path("git_ignore_folder/factor_implementation_source_data/intraday_pv.h5")
FACTORS_DIR = Path("results/factors/values")

fmap = {p.stem: str(p) for p in FACTORS_DIR.glob("*.parquet")}

names = data.get("factor_names", [])
code = data.get("code", "")
name = data.get("strategy_name", strat_path.stem)

if not names or not code:
    print(json.dumps({"status": "skipped", "reason": "no factors/code"}))
    sys.exit(0)

# Load close
ohlcv = pd.read_hdf(str(OHLCV), key="data")
close = ohlcv["$close"].dropna()
if isinstance(close.index, pd.MultiIndex):
    close = close.droplevel(-1)
close = close.astype(float).sort_index()

# Load factors
series = {}
for fn in names:
    fp = fmap.get(fn) or fmap.get(fn.replace("/", "_")[:150])
    if fp:
        try:
            s = pd.read_parquet(fp).iloc[:, 0]
            series[fn] = s
        except Exception:
            pass

if len(series) < 2:
    print(json.dumps({"status": "skipped", "reason": f"only {len(series)} factors loaded"}))
    sys.exit(0)

df = pd.DataFrame(series).sort_index()
if isinstance(df.index, pd.MultiIndex):
    df = df.droplevel(-1)

df_1m = df.reindex(close.index).ffill()
valid = df_1m.notna().any(axis=1)
if valid.sum() < 1000:
    print(json.dumps({"status": "skipped", "reason": f"only {valid.sum()} valid bars"}))
    sys.exit(0)

ca = close.loc[valid]
fa = df_1m.loc[valid]

# Execute
try:
    with tempfile.TemporaryDirectory() as td:
        tdp = Path(td)
        fa.to_parquet(str(tdp / "factors.parquet"))
        ca.to_pickle(str(tdp / "close.pkl"))
        exec_script = (
            "import sys, os\n"
            "sys.stdout = open(os.devnull, 'w')\n"
            "sys.stderr = open(os.devnull, 'w')\n"
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
            stdin=subprocess.DEVNULL,
        )
        if r.returncode != 0:
            print(json.dumps({"status": "code_failed", "stderr": r.stderr[:500]}))
            sys.exit(1)
        sig = pd.read_pickle(tdp / "signal.pkl")
except Exception as e:
    print(json.dumps({"status": "code_failed", "error": str(e)[:500]}))
    sys.exit(1)

sig = sig.reindex(ca.index).ffill().fillna(0)
result = backtest_signal(ca, sig, txn_cost_bps=2.14)

# Return result as JSON
output = {
    "status": "ok",
    "sharpe": result.get("sharpe"),
    "max_drawdown": result.get("max_drawdown"),
    "win_rate": result.get("win_rate"),
    "n_trades": result.get("n_trades"),
    "total_return": result.get("total_return"),
    "monthly_return_pct": result.get("monthly_return_pct"),
    "annualized_return": result.get("annualized_return"),
}
print(json.dumps(output))
