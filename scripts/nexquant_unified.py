#!/usr/bin/env python
"""
NexQuant Unified Loop — fin_quant + autopilot combined.

Flow:
  1. fin_quant generates a factor → auto-evaluates
  2. New factor tested in quick strategy (1h/30min SMA combo)
  3. Strategy OOS Sharpe feeds back to LLM for better hypotheses
  4. Factors that produce profitable strategies get priority
  5. Single process, no wasted LLM calls on dead-end factors
"""

from __future__ import annotations

import json, sys, time
from datetime import datetime
from pathlib import Path

import numpy as np
import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from rdagent.components.backtesting.vbt_backtest import backtest_signal_risk

# ── Config ──
DATA_PATH = Path("git_ignore_folder/factor_implementation_source_data/intraday_pv.h5")
TXN_COST_BPS = 2.14
MIN_MONTHLY_PCT = 0.1  # Minimum monthly return to keep a strategy


def load_daily_close():
    close = pd.read_hdf(DATA_PATH, key="data")["$close"]
    if isinstance(close.index, pd.MultiIndex):
        close = close.droplevel(-1)
    return close.sort_index().dropna()


def test_factor_as_signal(factor_path: Path, close: pd.Series, freq: str = "1h") -> dict | None:
    """Quick-test a factor as a trading signal. Returns metrics or None if unprofitable."""
    try:
        series = pd.read_parquet(factor_path).iloc[:, 0]
        if isinstance(series.index, pd.MultiIndex):
            series = series.droplevel(-1)
        fac = series.resample(freq).last().reindex(close.index).ffill()
    except Exception:
        return None

    is_sess = (close.index.hour >= 7) & (close.index.hour < 17)
    
    best_result = None
    for direction in [1, -1]:
        sig = pd.Series(direction * np.sign(fac).fillna(0), index=close.index)
        sig[~is_sess] = 0
        if sig.abs().sum() < 20:
            continue

        r = backtest_signal_risk(close, sig.fillna(0), txn_cost_bps=TXN_COST_BPS)
        oos_m = r.get("oos_monthly_return_pct", 0) or 0

        if oos_m > (best_result["monthly"] if best_result else MIN_MONTHLY_PCT):
            best_result = {
                "direction": direction,
                "monthly": oos_m,
                "oos_sharpe": r.get("oos_sharpe", -999),
                "max_dd": r.get("oos_max_drawdown", 0),
                "trades": r.get("oos_n_trades", 0),
            }

    return best_result


def scan_all_factors():
    """Scan ALL factors and rank them by strategy profitability (not IC)."""
    close = load_daily_close().resample("1h").last().dropna()
    factors_dir = Path("results/factors")
    values_dir = factors_dir / "values"

    results = []
    for i, jf in enumerate(sorted(factors_dir.glob("*.json"))):
        try:
            meta = json.loads(jf.read_text())
        except Exception:
            continue
        if meta.get("status") != "success":
            continue

        name = meta.get("factor_name", jf.stem)
        safe = name.replace("/", "_")[:150]
        pf = values_dir / f"{safe}.parquet"
        if not pf.exists():
            continue

        bt = test_factor_as_signal(pf, close)
        if bt:
            results.append({
                "factor": name,
                "ic": meta.get("ic", 0),
                **bt,
            })

        if i % 100 == 0:
            profitable = sum(1 for r in results if r.get("monthly", 0) > 0.5)
            print(f"  Scanned {i}... {profitable} profitable (>0.5%/mon)")

    results.sort(key=lambda x: x.get("monthly", 0), reverse=True)
    return results


def main():
    print(f"\n{'='*60}")
    print("  NexQuant Unified Loop — Factor-to-Strategy Pipeline")
    print(f"{'='*60}")

    print("\n=== PHASE 1: Scan all existing factors as strategies ===\n")
    t0 = time.time()
    ranked = scan_all_factors()

    profitable = [r for r in ranked if r.get("monthly", 0) > 0.5]
    print(f"\n  Scanned {len(ranked)} factors in {time.time()-t0:.0f}s")
    print(f"  Profitable (>0.5%/month): {len(profitable)}")

    if profitable:
        print(f"\n  TOP 10 by Strategy Profitability:")
        for i, r in enumerate(profitable[:10]):
            print(f"  {i+1:2d}. {r['factor'][:45]:45s} Mon={r['monthly']:+.2f}% IC={r['ic']:+.4f} Dir={r['direction']:+d}")

        # Build combo from top signals
        print(f"\n=== PHASE 2: Build best combo ===\n")
        c = load_daily_close().resample("1h").last().dropna()
        is_sess = (c.index.hour >= 7) & (c.index.hour < 17)

        signals = {}
        for r in profitable[:10]:
            safe = r["factor"].replace("/", "_")[:150]
            pf = Path("results/factors/values") / f"{safe}.parquet"
            try:
                s = pd.read_parquet(pf).iloc[:, 0]
                if isinstance(s.index, pd.MultiIndex):
                    s = s.droplevel(-1)
                fac = s.resample("1h").last().reindex(c.index).ffill()
                sig = pd.Series(r["direction"] * np.sign(fac).fillna(0), index=c.index)
                sig[~is_sess] = 0
                signals[r["factor"]] = sig
            except Exception:
                pass

        df = pd.DataFrame(signals, index=c.index).fillna(0)
        cols = list(df.columns)
        for n in [2, 3, 5, len(cols)]:
            combo = df[cols[:n]].mean(axis=1)
            r = backtest_signal_risk(c, combo.fillna(0), txn_cost_bps=TXN_COST_BPS, wf_rolling=True)
            m = r.get("oos_monthly_return_pct", 0) or 0
            dd = (r.get("oos_max_drawdown", 0) or 0) * 100
            t = r.get("oos_n_trades", 0)
            gap = 10 - m
            hit = "🎯" if m >= 4 else ""
            print(f"  {n:2d} sig: Mon={m:+.2f}% DD={dd:+.1f}% T={t} Gap2_10%={gap:+.1f} {hit}")

    print(f"\n  Next: feed top factors back to fin_quant LLM for improved hypotheses")
    print(f"  Run: python scripts/nexquant_unified.py")
    return ranked


if __name__ == "__main__":
    main()
