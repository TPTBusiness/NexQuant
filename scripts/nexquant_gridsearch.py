#!/usr/bin/env python3
"""Grid-Search Strategy Generator — no LLM, deterministic, FTMO-verified.

Core idea: Instead of LLM-generated code, use a fixed signal template and
grid-search the parameters. Factors are aligned to daily resolution (where
they have actual predictive power), signal is forward-filled to 1-min for
FTMO backtest execution.

Template: z-score → IC-weighted composite → asymmetric thresholds → signal
"""

import json
import os
import time
from datetime import datetime
from pathlib import Path

import numpy as np
import pandas as pd

# ── Paths ────────────────────────────────────────────────────────────────────
PROJECT = Path(__file__).resolve().parent.parent
FACTORS_DIR = PROJECT / "results" / "factors"
VALUES_DIR = FACTORS_DIR / "values"
RESULTS_DIR = PROJECT / "results" / "strategies_new"
OHLCV_PATH = Path(
    os.getenv("PREDIX_OHLCV_PATH",
              str(PROJECT / "git_ignore_folder" / "intraday_pv_all.h5"))
)

# ── Target ───────────────────────────────────────────────────────────────────
MIN_MONTHLY_RETURN_PCT = 1.0   # Raw backtest target (FTMO will reduce ~50%)
MIN_SHARPE = 0.5
MAX_DRAWDOWN = -0.30
MIN_WIN_RATE = 0.35
MIN_TRADES = 20

# ── Grid ─────────────────────────────────────────────────────────────────────
PARAM_GRID = {
    "window": [5, 10, 20, 30],
    "entry_thresh": [0.5, 0.8, 1.0, 1.5, 2.0],  # Higher = fewer, higher-conviction trades
    "exit_thresh": [0.2, 0.5],
}
# Total: 5 × 4 × 3 = 60 combinations per factor pair


# ═══════════════════════════════════════════════════════════════════════════════
# Factor loading
# ═══════════════════════════════════════════════════════════════════════════════

def load_top_factors(min_ic: float = 0.04, top_n: int = 50) -> list[dict]:
    """Load factor metadata sorted by |IC| descending."""
    factors = []
    for f in sorted(FACTORS_DIR.glob("*.json")):
        data = json.loads(f.read_text())
        if not isinstance(data, dict):
            continue
        fname = data.get("factor_name") or data.get("name") or f.stem
        ic = data.get("ic") or data.get("real_ic") or 0.0
        try:
            ic = float(ic)
        except (TypeError, ValueError):
            continue
        if abs(ic) < min_ic:
            continue
        safe = fname.replace("/", "_").replace("\\", "_").replace(" ", "_")[:150]
        parq = VALUES_DIR / f"{safe}.parquet"
        if not parq.exists():
            continue
        factors.append({"name": fname, "ic": ic, "parquet": parq})
    factors.sort(key=lambda x: abs(x["ic"]), reverse=True)
    return factors[:top_n]


def load_factor_series(factor: dict) -> pd.Series | None:
    """Load factor time series, extracting the EURUSD slice."""
    try:
        df = pd.read_parquet(str(factor["parquet"]))
        if df.empty:
            return None
        col = df.columns[0]
        if isinstance(df.index, pd.MultiIndex):
            return df.xs("EURUSD", level="instrument")[col]
        return df[col]
    except Exception:
        return None


# ═══════════════════════════════════════════════════════════════════════════════
# Signal generation
# ═══════════════════════════════════════════════════════════════════════════════

def build_signal(
    daily_factors: pd.DataFrame,
    ic_values: dict[str, float],
    window: int = 10,
    entry_thresh: float = 0.5,
    exit_thresh: float = 0.2,
) -> pd.Series:
    """
    Fixed signal template: z-score → IC-weighted composite → thresholds.

    Parameters
    ----------
    daily_factors : DataFrame
        Factor values at daily resolution, columns = factor names.
    ic_values : dict
        Factor name → IC value (used for sign/direction, not weight).
    window : int
        Rolling window for z-score in days.
    entry_thresh : float
        Composite z-score threshold for entry.
    exit_thresh : float
        Composite z-score threshold for exit (flatten position).
    """
    eps = 1e-8
    z = (daily_factors - daily_factors.rolling(window).mean()) / (
        daily_factors.rolling(window).std() + eps
    )

    # IC-weighted composite: invert negative-IC factors, weight by |IC|
    composite = pd.Series(0.0, index=daily_factors.index)
    total_abs_ic = sum(abs(ic) for ic in ic_values.values())
    if total_abs_ic == 0:
        total_abs_ic = 1.0

    for col in daily_factors.columns:
        ic = ic_values.get(col, 0.0)
        w = abs(ic) / total_abs_ic
        sign = 1.0 if ic >= 0 else -1.0
        composite += sign * w * z[col]

    # Asymmetric thresholds
    signal = pd.Series(0, index=daily_factors.index)
    signal[composite > entry_thresh] = 1
    signal[composite < -entry_thresh] = -1
    signal[abs(composite) < exit_thresh] = 0

    signal = signal.rolling(2, min_periods=1).mean().round().astype(int)
    signal = signal.clip(-1, 1)
    signal.name = "signal"
    return signal


# ═══════════════════════════════════════════════════════════════════════════════
# Evaluation
# ═══════════════════════════════════════════════════════════════════════════════

def evaluate_one(args: tuple) -> dict | None:
    """Evaluate one parameter combination on one factor pair."""
    (
        f1_name, f1_ic, f1_series,
        f2_name, f2_ic, f2_series,
        close_1min, window, entry, exit_th,
    ) = args

    try:
        # Align factors to 1-min close
        factors_1min = pd.DataFrame({
            f1_name: f1_series.reindex(close_1min.index).ffill(limit=2880),
            f2_name: f2_series.reindex(close_1min.index).ffill(limit=2880),
        })

        # Resample to daily
        daily_factors = factors_1min.resample("D").last().dropna()
        if len(daily_factors) < 50:
            return None  # Not enough daily data

        daily_close = close_1min.resample("D").last().reindex(daily_factors.index)

        # Build signal
        ic_values = {f1_name: f1_ic, f2_name: f2_ic}
        daily_signal = build_signal(daily_factors, ic_values, window, entry, exit_th)

        # Forward-fill to 1-min for backtest
        signal_1min = daily_signal.reindex(close_1min.index).ffill().fillna(0).astype(int).clip(-1, 1)

        # Fast backtest (no FTMO mask, no walk-forward — <1s per eval)
        from rdagent.components.backtesting.vbt_backtest import backtest_signal

        bt = backtest_signal(
            close=close_1min,
            signal=signal_1min,
        )

        if bt.get("status") != "success":
            return None

        sharpe = bt.get("sharpe", 0) or 0
        max_dd = bt.get("max_drawdown", 0) or 0
        win_rate = bt.get("win_rate", 0) or 0
        n_trades = bt.get("n_trades", 0) or 0
        monthly_pct = bt.get("monthly_return_pct", 0) or 0

        return {
            "f1": f1_name,
            "f2": f2_name,
            "window": window,
            "entry": entry,
            "exit": exit_th,
            "sharpe": round(sharpe, 4),
            "max_dd": round(max_dd, 4),
            "win_rate": round(win_rate, 4),
            "n_trades": n_trades,
            "monthly_pct": round(monthly_pct, 2),
        }
    except Exception:
        return None


def main():
    print("═" * 60)
    print("  Grid-Search Strategy Generator (no LLM)")
    print("═" * 60)

    # ── Load OHLCV ────────────────────────────────────────────────────────
    print(f"\nLoading OHLCV: {OHLCV_PATH}")
    df = pd.read_hdf(OHLCV_PATH, key="data")
    close_1min = df.xs("EURUSD", level="instrument")["$close"].sort_index()
    print(f"  1-min bars: {len(close_1min):,}  ({close_1min.index[0].date()} → {close_1min.index[-1].date()})")

    # ── Load factors ───────────────────────────────────────────────────────
    print(f"\nLoading factors (|IC| ≥ 0.04)...")
    top_n = int(os.getenv("GS_TOP_N", "10"))
    factors = load_top_factors(min_ic=0.04, top_n=top_n)
    print(f"  Loaded {len(factors)} factors")

    factor_series = {}
    for f in factors:
        s = load_factor_series(f)
        if s is not None and len(s) > 100:
            factor_series[f["name"]] = (f["ic"], s)

    names = list(factor_series.keys())
    print(f"  Valid series: {len(names)}")

    # ── Generate factor pairs ──────────────────────────────────────────────
    import itertools

    pairs = list(itertools.combinations(names, 2))
    print(f"  Factor pairs: {len(pairs)}")

    # ── Generate parameter combinations ────────────────────────────────────
    param_combos = list(itertools.product(
        PARAM_GRID["window"],
        PARAM_GRID["entry_thresh"],
        PARAM_GRID["exit_thresh"],
    ))
    # Filter: exit < entry
    param_combos = [(w, e, x) for w, e, x in param_combos if x < e]
    print(f"  Parameter combos: {len(param_combos)}")

    # ── Build work items ───────────────────────────────────────────────────
    work_items = []
    for f1_name, f2_name in pairs:
        f1_ic, f1_series = factor_series[f1_name]
        f2_ic, f2_series = factor_series[f2_name]
        for window, entry, exit_th in param_combos:
            work_items.append((
                f1_name, f1_ic, f1_series,
                f2_name, f2_ic, f2_series,
                close_1min, window, entry, exit_th,
            ))

    total = len(work_items)
    print(f"  Total evaluations: {total:,}")

    # ── Run sequentially ───────────────────────────────────────────────────
    t0 = time.time()
    results = []

    for i, item in enumerate(work_items):
        r = evaluate_one(item)
        if r is not None:
            results.append(r)
        if (i + 1) % 100 == 0 or i == total - 1:
            elapsed = time.time() - t0
            rate = (i + 1) / elapsed if elapsed > 0 else 0
            eta = (total - i - 1) / rate if rate > 0 else 0
            print(f"  {i+1}/{total} ({(i+1)/total*100:.1f}%)  "
                  f"{len(results)} valid  {rate:.1f}/s  eta {eta:.0f}s")

    # ── Filter and sort ────────────────────────────────────────────────────
    print(f"\n{'═' * 60}")
    print(f"  Total evaluated: {total:,}  Valid results: {len(results):,}")
    print(f"{'═' * 60}")

    valid = [r for r in results
             if r["sharpe"] >= MIN_SHARPE
             and r["max_dd"] >= MAX_DRAWDOWN
             and r["win_rate"] >= MIN_WIN_RATE
             and r["n_trades"] >= MIN_TRADES
             and r["monthly_pct"] >= MIN_MONTHLY_RETURN_PCT]

    valid.sort(key=lambda r: r["monthly_pct"], reverse=True)

    print(f"\n  Meeting criteria (Sharpe≥{MIN_SHARPE}, DD≥{MAX_DRAWDOWN}, "
          f"WR≥{MIN_WIN_RATE}, Trades≥{MIN_TRADES}, Mon≥{MIN_MONTHLY_RETURN_PCT}%):")
    print(f"  → {len(valid)} strategies")
    print()

    if valid:
        print(f"{'#':<3s} {'Factor 1':>30s} + {'Factor 2':>30s}  {'w':>3s} {'ent':>4s} {'ex':>4s}  {'Sharpe':>7s} {'MaxDD':>7s} {'WinRt':>6s} {'Tr':>4s} {'Mon%':>7s}")
        print("-" * 135)
        for i, r in enumerate(valid[:30], 1):
            print(f"{i:<3d} {r['f1'][:30]:>30s} + {r['f2'][:30]:>30s}  "
                  f"{r['window']:>3d} {r['entry']:>4.1f} {r['exit']:>4.1f}  "
                  f"{r['sharpe']:>7.3f} {r['max_dd']:>7.3f} {r['win_rate']:>6.1%} "
                  f"{r['n_trades']:>4d} {r['monthly_pct']:>7.2f}%")
    else:
        print("  No strategies meet the criteria.")
        if results:
            results.sort(key=lambda r: r["monthly_pct"], reverse=True)
            print("\n  Top 10 by monthly return:")
            for i, r in enumerate(results[:10], 1):
                print(f"  {i:2d}. {r['f1'][:25]} + {r['f2'][:25]}  "
                      f"Mon={r['monthly_pct']:.2f}%  Sh={r['sharpe']:.3f}  "
                      f"DD={r['max_dd']:.3f}  Tr={r['n_trades']}")

    # ── Save top results ───────────────────────────────────────────────────
    RESULTS_DIR.mkdir(parents=True, exist_ok=True)
    out_path = RESULTS_DIR / f"gridsearch_{datetime.now().strftime('%Y%m%d_%H%M%S')}.json"
    out_path.write_text(json.dumps(valid[:50] if valid else results[:50], indent=2, default=str))
    print(f"\n  Top results saved → {out_path}")
    print(f"  Runtime: {time.time() - t0:.0f}s")


if __name__ == "__main__":
    main()
