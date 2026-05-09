#!/usr/bin/env python
"""
Re-run existing strategies through the unified backtest engine.

For every strategy JSON in results/strategies_new (or a user-supplied dir):
  1. Load the factor values it references.
  2. Execute its ``code`` in a sandboxed subprocess to produce the signal.
  3. Run the signal through ``backtest_signal`` on REAL 1-min EUR/USD close.
  4. Print old-vs-new sharpe / DD / trades / total-return so the impact of
     the unified engine (no return clipping, proper 1-min annualization,
     trade-epoch win rate) is visible.

Does NOT mutate the strategy JSON files — read-only comparison.

Usage:
    python scripts/nexquant_rebacktest_unified.py              # all strategies
    python scripts/nexquant_rebacktest_unified.py 50           # first 50
    python scripts/nexquant_rebacktest_unified.py 50 --csv report.csv
"""
from __future__ import annotations

import argparse
import csv
import json
import logging
import subprocess
import sys
import tempfile
from datetime import datetime
from pathlib import Path
from typing import Any, Dict, List, Optional

import numpy as np
import pandas as pd
from rich.console import Console
from rich.progress import BarColumn, Progress, SpinnerColumn, TextColumn, TimeElapsedColumn

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
from rdagent.components.backtesting.vbt_backtest import backtest_signal_ftmo  # noqa: E402

OHLCV_PATH = Path("/home/nico/NexQuant/git_ignore_folder/factor_implementation_source_data/intraday_pv.h5")
FACTORS_VALUES_DIR = Path("/home/nico/NexQuant/results/factors/values")
STRATEGIES_DIR = Path("/home/nico/NexQuant/results/strategies_new")

# ── Logging setup: everything printed goes to log file + stdout ───────────────
_LOG_DIR = Path(__file__).resolve().parent.parent / "git_ignore_folder" / "logs"
_LOG_DIR.mkdir(parents=True, exist_ok=True)
_log_file_path = _LOG_DIR / f"rebacktest_{datetime.now().strftime('%Y%m%d_%H%M%S')}.log"
_log_file = open(_log_file_path, "w", encoding="utf-8", buffering=1)

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(message)s",
    handlers=[
        logging.StreamHandler(sys.stdout),
        logging.FileHandler(_log_file_path, encoding="utf-8"),
    ],
)

class _TeeFile:
    """Writes to both stdout and log file — used as Rich Console file."""
    def __init__(self, *files):
        self._files = files
    def write(self, data):
        for f in self._files:
            f.write(data)
    def flush(self):
        for f in self._files:
            f.flush()
    def fileno(self):
        return self._files[0].fileno()

console = Console(file=_TeeFile(sys.stdout, _log_file), highlight=False)


def load_close() -> pd.Series:
    ohlcv = pd.read_hdf(str(OHLCV_PATH), key="data")
    col = "$close" if "$close" in ohlcv.columns else "close"
    close = ohlcv[col].dropna()
    # Drop the "EURUSD" instrument level if present — the strategies work
    # on a single series indexed by timestamp.
    if isinstance(close.index, pd.MultiIndex):
        close = close.droplevel(-1)
    return close.astype(float).sort_index()


def load_factor_series(names: List[str]) -> Dict[str, pd.Series]:
    out: Dict[str, pd.Series] = {}
    for name in names:
        for variant in (name, name.replace("/", "_").replace("\\", "_")[:150]):
            path = FACTORS_VALUES_DIR / f"{variant}.parquet"
            if path.exists():
                try:
                    df = pd.read_parquet(str(path))
                    if df is not None and len(df.columns) > 0:
                        out[name] = df.iloc[:, 0]
                        break
                except Exception:
                    pass
    return out


def execute_strategy(
    factors_df: pd.DataFrame,
    close: pd.Series,
    strategy_code: str,
    timeout: int = 45,
) -> Optional[pd.Series]:
    """Run untrusted LLM code in a subprocess and return the resulting signal."""
    script = f"""
import pandas as pd, numpy as np
factors = pd.read_pickle('factors.pkl')
close = pd.read_pickle('close.pkl')
df = factors  # some strategies reference 'df', others 'factors'

try:
{chr(10).join('    ' + line for line in strategy_code.split(chr(10)))}
except Exception as e:
    print(f"ERROR: {{e}}")
    raise SystemExit(1)

if 'signal' not in dir():
    print("ERROR: no signal")
    raise SystemExit(1)

pd.Series(signal).fillna(0).to_pickle('signal.pkl')
"""
    with tempfile.TemporaryDirectory() as td:
        tdp = Path(td)
        factors_df.to_pickle(str(tdp / "factors.pkl"))
        close.to_pickle(str(tdp / "close.pkl"))
        (tdp / "run.py").write_text(script)

        try:
            result = subprocess.run(
                ["python", "run.py"],
                capture_output=True,
                text=True,
                timeout=timeout,
                cwd=str(tdp),
            )
            if result.returncode != 0:
                return None
            signal = pd.read_pickle(tdp / "signal.pkl")
            return signal
        except (subprocess.TimeoutExpired, Exception):
            return None


def rebacktest_one(
    strategy_data: Dict[str, Any],
    close: pd.Series,
    txn_cost_bps: float,
) -> Dict[str, Any]:
    factor_names = strategy_data.get("factor_names") or strategy_data.get("factors_used") or []
    code = strategy_data.get("code", "")
    if not factor_names or not code:
        return {"status": "skipped", "reason": "missing factors or code"}

    factor_series = load_factor_series(factor_names)
    if len(factor_series) < 2:
        return {"status": "skipped", "reason": f"only {len(factor_series)} factor files found"}

    factors_df = pd.DataFrame(factor_series).dropna(how="all")
    if isinstance(factors_df.index, pd.MultiIndex):
        factors_df = factors_df.droplevel(-1)
    factors_df = factors_df.sort_index()

    # Factors are typically daily-timestamped; close is 1-min.
    # Direct index intersection would be near-zero → reindex and ffill first,
    # matching exactly what the orchestrator's evaluate_strategy does.
    factors_1min = factors_df.reindex(close.index).ffill()
    valid_rows = factors_1min.notna().any(axis=1)
    if valid_rows.sum() < 1000:
        return {"status": "skipped", "reason": f"only {valid_rows.sum()} valid rows after ffill"}

    close_a = close.loc[valid_rows]
    factors_a = factors_1min.loc[valid_rows]

    signal = execute_strategy(factors_a, close_a, code)
    if signal is None:
        return {"status": "code_failed"}

    # Signal can arrive on either the factor index or the close index.
    signal = signal.reindex(close_a.index).ffill().fillna(0)

    result = backtest_signal_ftmo(
        close=close_a,
        signal=signal,
        txn_cost_bps=txn_cost_bps,
        wf_rolling=True,
        mc_n_permutations=200,
    )
    result["status_detail"] = result.pop("status")
    result["status"] = "ok"
    return result


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("count", type=int, nargs="?", default=None,
                        help="Limit to first N strategies (default: all)")
    parser.add_argument("--dir", type=Path, default=STRATEGIES_DIR,
                        help="Strategy directory to re-backtest")
    parser.add_argument("--csv", type=Path, default=None,
                        help="Write a CSV report to this path")
    parser.add_argument("--txn-cost-bps", type=float, default=2.14,
                        help="Transaction cost bps (default 2.14 ≈ 2.35 pip EUR/USD)")
    parser.add_argument("--write-back", action="store_true",
                        help="Overwrite summary field in strategy JSON files with new results")
    args = parser.parse_args()

    console.print(f"[dim]Log: {_log_file_path}[/dim]")
    console.print(f"[cyan]Loading OHLCV close...[/cyan]")
    close = load_close()
    console.print(f"[green]✓[/green] {len(close):,} 1-min bars "
                  f"({close.index[0]} → {close.index[-1]})\n")

    files = sorted(args.dir.glob("*.json"))
    if args.count:
        files = files[:args.count]
    console.print(f"[cyan]Re-backtesting {len(files)} strategies with unified engine...[/cyan]\n")

    rows: List[Dict[str, Any]] = []
    with Progress(
        SpinnerColumn(),
        TextColumn("[bold blue]{task.description}"),
        BarColumn(),
        TextColumn("[bold green]{task.completed}/{task.total}"),
        TimeElapsedColumn(),
    ) as progress:
        task = progress.add_task("Backtesting", total=len(files))
        for f in files:
            try:
                data = json.load(open(f))
            except Exception:
                progress.update(task, advance=1)
                continue

            old = data.get("summary", {})
            name = data.get("strategy_name", f.stem)[:38]

            bt = rebacktest_one(data, close, args.txn_cost_bps)

            if args.write_back and bt.get("status") == "ok":
                data["summary"] = {
                    "sharpe": bt.get("sharpe"),
                    "max_drawdown": bt.get("max_drawdown"),
                    "win_rate": bt.get("win_rate"),
                    "monthly_return_pct": bt.get("monthly_return_pct"),
                    "real_ic": data.get("summary", {}).get("real_ic"),
                    "real_n_trades": bt.get("n_trades"),
                    "total_return": bt.get("total_return"),
                    "annualized_return": bt.get("annualized_return"),
                    "ftmo_daily_loss_hit": bt.get("ftmo_daily_loss_hit"),
                    "ftmo_total_loss_hit": bt.get("ftmo_total_loss_hit"),
                    "trading_style": data.get("summary", {}).get("trading_style"),
                    "engine": "ftmo_v2",
                    "txn_cost_bps": args.txn_cost_bps,
                    # Walk-forward OOS
                    "is_sharpe": bt.get("is_sharpe"),
                    "is_monthly_return_pct": bt.get("is_monthly_return_pct"),
                    "oos_sharpe": bt.get("oos_sharpe"),
                    "oos_monthly_return_pct": bt.get("oos_monthly_return_pct"),
                    "oos_max_drawdown": bt.get("oos_max_drawdown"),
                    "oos_win_rate": bt.get("oos_win_rate"),
                    "oos_n_trades": bt.get("oos_n_trades"),
                    "oos_start": bt.get("oos_start"),
                    # Rolling walk-forward
                    "wf_n_windows": bt.get("wf_n_windows"),
                    "wf_oos_sharpe_mean": bt.get("wf_oos_sharpe_mean"),
                    "wf_oos_sharpe_std": bt.get("wf_oos_sharpe_std"),
                    "wf_oos_monthly_return_mean": bt.get("wf_oos_monthly_return_mean"),
                    "wf_oos_consistency": bt.get("wf_oos_consistency"),
                    # Monte Carlo significance
                    "mc_pvalue": bt.get("mc_pvalue"),
                    "mc_n_permutations": bt.get("mc_n_permutations"),
                }
                data["sharpe_ratio"] = bt.get("sharpe")
                data["max_drawdown"] = bt.get("max_drawdown")
                data["win_rate"] = bt.get("win_rate")
                data["total_return"] = bt.get("total_return")
                data["reevaluation_status"] = "ftmo_v2"
                try:
                    import json as _json
                    f.write_text(_json.dumps(data, indent=2, ensure_ascii=False))
                except Exception as _e:
                    logging.warning(f"write-back failed for {f.name}: {_e}")

            row = {
                "file": f.name,
                "name": name,
                "status": bt.get("status"),
                "reason": bt.get("reason", bt.get("status_detail", "")),
                "old_sharpe": old.get("sharpe"),
                "old_dd": old.get("max_drawdown"),
                "old_trades": old.get("real_n_trades"),
                "old_monthly_pct": old.get("monthly_return_pct"),
                "new_sharpe": bt.get("sharpe"),
                "new_dd": bt.get("max_drawdown"),
                "new_trades": bt.get("n_trades"),
                "new_total_return": bt.get("total_return"),
                "new_monthly_pct": bt.get("monthly_return_pct"),
                "new_annual_return_cagr": None,
                "data_quality": bt.get("data_quality_flag"),
                # OOS walk-forward
                "is_sharpe": bt.get("is_sharpe"),
                "is_monthly_pct": bt.get("is_monthly_return_pct"),
                "oos_sharpe": bt.get("oos_sharpe"),
                "oos_monthly_pct": bt.get("oos_monthly_return_pct"),
                "oos_dd": bt.get("oos_max_drawdown"),
                "oos_trades": bt.get("oos_n_trades"),
                # Rolling walk-forward
                "wf_n_windows": bt.get("wf_n_windows"),
                "wf_oos_sharpe_mean": bt.get("wf_oos_sharpe_mean"),
                "wf_oos_consistency": bt.get("wf_oos_consistency"),
                # Monte Carlo
                "mc_pvalue": bt.get("mc_pvalue"),
            }
            if "annualized_return" in bt:
                row["new_annual_return_cagr"] = bt["annualized_return"]
            rows.append(row)
            progress.update(task, advance=1)

    # Summary
    ok_rows = [r for r in rows if r["status"] == "ok"]
    console.print(f"\n[bold]{len(ok_rows)}/{len(rows)} strategies successfully re-backtested[/bold]\n")

    status_counts: Dict[str, int] = {}
    for r in rows:
        status_counts[r["status"]] = status_counts.get(r["status"], 0) + 1
    for status, n in sorted(status_counts.items(), key=lambda kv: -kv[1]):
        console.print(f"  {status}: {n}")

    if ok_rows:
        # Compare old vs new where both exist
        comparable = [r for r in ok_rows if r["old_sharpe"] is not None]
        if comparable:
            old_sharpe = np.array([r["old_sharpe"] for r in comparable], dtype=float)
            new_sharpe = np.array([r["new_sharpe"] for r in comparable], dtype=float)
            console.print(f"\n[bold]Sharpe drift ({len(comparable)} strategies with old metrics):[/bold]")
            console.print(f"  old  mean={old_sharpe.mean():+.3f}  median={np.median(old_sharpe):+.3f}  max={old_sharpe.max():+.3f}")
            console.print(f"  new  mean={new_sharpe.mean():+.3f}  median={np.median(new_sharpe):+.3f}  max={new_sharpe.max():+.3f}")
            diff = new_sharpe - old_sharpe
            console.print(f"  Δ    mean={diff.mean():+.3f}  median={np.median(diff):+.3f}")
            agree_sign = int(((np.sign(old_sharpe) == np.sign(new_sharpe)) | (np.abs(new_sharpe) < 0.1)).sum())
            console.print(f"  sign-agreement: {agree_sign}/{len(comparable)} "
                          f"({agree_sign/len(comparable):.0%})")

        ok_rows.sort(key=lambda r: r["new_sharpe"] if r["new_sharpe"] is not None else -1e9, reverse=True)
        console.print(f"\n[bold]Top 15 by new Sharpe:[/bold]")
        console.print(f"  {'name':<38} {'old_sh':>7} {'new_sh':>7} {'new_dd':>8} {'new_trd':>7} {'new_ret':>9}")
        for r in ok_rows[:15]:
            osh = f"{r['old_sharpe']:+.2f}" if r["old_sharpe"] is not None else "   —"
            ddv = f"{r['new_dd']:.2%}" if r["new_dd"] is not None else "—"
            rtv = f"{r['new_total_return']:+.2%}" if r["new_total_return"] is not None else "—"
            console.print(f"  {r['name']:<38} {osh:>7} {r['new_sharpe']:>+7.2f} {ddv:>8} {r['new_trades'] or 0:>7} {rtv:>9}")

        flagged = [r for r in ok_rows if r["data_quality"]]
        if flagged:
            console.print(f"\n[yellow]⚠ {len(flagged)} strategies flagged with extreme bars "
                          f"(would have been hidden by old ±10% clipping)[/yellow]")

    if args.csv:
        with open(args.csv, "w", newline="") as fh:
            w = csv.DictWriter(fh, fieldnames=list(rows[0].keys()))
            w.writeheader()
            w.writerows(rows)
        console.print(f"\n[green]✓[/green] CSV report written to {args.csv}")


if __name__ == "__main__":
    main()
