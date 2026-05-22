#!/usr/bin/env python
"""
NexQuant Systematic Strategy Generator — kein LLM, nur Mathematik.

Grid-searched threshold strategies with IC-weighted z-score composites.
Optionally trains LightGBM directional classifier.

Approaches:
  A) IC-weighted z-score composite (always used as base)
  B) Grid-search entry/exit thresholds (primary)
  C) LightGBM directional classifier (optional, if factors ≥ 5)
  D) Factor-ranking top/bottom deciles (fast baseline)

Output: Best strategy by OOS Walk-Forward Sharpe, saved to results/strategies_systematic/
"""

from __future__ import annotations

import json
import sys
import time
from datetime import datetime
from pathlib import Path
from typing import Optional

import numpy as np
import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

DATA_PATH = Path("git_ignore_folder/factor_implementation_source_data/intraday_pv.h5")
FACTORS_DIR = Path("results/factors")
OUT_DIR = Path("results/strategies_systematic")
OUT_DIR.mkdir(parents=True, exist_ok=True)

TXN_COST_BPS = 2.14
OOS_START = "2024-01-01"
WF_WINDOWS = 4


def load_data() -> tuple:
    """Load OHLCV close prices and top factors."""
    ohlcv = pd.read_hdf(DATA_PATH, key="data")
    close = ohlcv["$close"]
    if isinstance(close.index, pd.MultiIndex):
        close = close.droplevel(-1)
    close = close.sort_index().dropna()

    factors = []
    for f in sorted(FACTORS_DIR.glob("*.json")):
        try:
            d = json.loads(f.read_text())
        except Exception:
            continue
        if d.get("status") != "success" or d.get("ic") is None:
            continue
        name = d.get("factor_name", f.stem)
        safe = name.replace("/", "_").replace("\\", "_")[:150]
        pf = FACTORS_DIR / "values" / f"{safe}.parquet"
        if pf.exists():
            factors.append({"name": name, "ic": d["ic"]})

    factors.sort(key=lambda x: abs(x["ic"]), reverse=True)
    return close, factors


def load_factor_values(factor_names: list, close: pd.Series) -> pd.DataFrame:
    """Load and align factor time series."""
    data = {}
    for name in factor_names:
        safe = name.replace("/", "_").replace("\\", "_")[:150]
        pf = FACTORS_DIR / "values" / f"{safe}.parquet"
        if not pf.exists():
            continue
        series = pd.read_parquet(pf).iloc[:, 0]
        if isinstance(series.index, pd.MultiIndex):
            series = series.droplevel(-1)
        data[name] = series

    df = pd.DataFrame(data)
    common = close.index.intersection(df.dropna(how="all").index)
    return df.loc[common].ffill(), close.loc[common]


def compute_ic_weighted_composite(factors_df: pd.DataFrame, ics: dict[str, float]) -> pd.Series:
    """Compute z-score normalized, IC-weighted composite signal."""
    composite = pd.Series(0.0, index=factors_df.index)
    total_abs_ic = 0.0

    for col in factors_df.columns:
        if col not in ics:
            continue
        ic = ics[col]
        if abs(ic) < 0.001:
            continue
        z = (factors_df[col] - factors_df[col].rolling(20).mean()) / (
            factors_df[col].rolling(20).std() + 1e-8
        )
        weight = ic  # Keep sign: if IC < 0, invert factor
        composite += weight * z
        total_abs_ic += abs(ic)

    if total_abs_ic > 0:
        composite /= total_abs_ic
    return composite


def generate_signal_threshold(composite: pd.Series, entry: float, exit_thresh: float) -> pd.Series:
    """Generate signal from composite with entry/exit thresholds (vectorized)."""
    signal = pd.Series(0, index=composite.index, dtype=float)
    signal[composite > entry] = 1
    signal[composite < -entry] = -1
    # Simple: no hysteresis for speed. Entry = exit.
    return signal


def generate_signal_ranking(factors_df: pd.DataFrame, ics: dict, top_pct: float = 0.10) -> pd.Series:
    """Factor-ranking: top/bottom deciles = long/short, daily rebalanced."""
    composite = compute_ic_weighted_composite(factors_df, ics)
    signal = pd.Series(0, index=composite.index)

    for date, group in composite.groupby(composite.index.normalize()):
        n = len(group)
        k = max(1, int(n * top_pct))
        ranked = group.abs().sort_values(ascending=False)
        top_idx = ranked.index[:k]
        bot_idx = ranked.index[-k:]
        signal.loc[top_idx] = np.sign(composite.loc[top_idx])
        signal.loc[bot_idx] = np.sign(composite.loc[bot_idx]) * -1

    return signal


def grid_search(close: pd.Series, composite: pd.Series, style: str = "swing") -> dict:
    """Grid-search optimal entry thresholds."""
    from rdagent.components.backtesting.vbt_backtest import backtest_signal_risk

    best = None
    best_sharpe = -999

    entries = np.arange(0.3, 2.1, 0.3)

    for entry in entries:
        sig = generate_signal_threshold(composite, entry, 0.0)
        r = backtest_signal_risk(close, sig, txn_cost_bps=TXN_COST_BPS, wf_rolling=True)

        wf_sharpe = r.get("wf_oos_sharpe_mean", -999) or -999
        if wf_sharpe > best_sharpe:
            best_sharpe = wf_sharpe
            best = {
                "entry": entry,
                "wf_sharpe": wf_sharpe,
                "oos_sharpe": r.get("oos_sharpe", -999),
                "oos_monthly": r.get("oos_monthly_return_pct", 0),
                "oos_dd": r.get("oos_max_drawdown", 0),
                "oos_trades": r.get("oos_n_trades", 0),
                "oos_wr": r.get("oos_win_rate", 0),
                "is_sharpe": r.get("is_sharpe", -999),
                "consistency": r.get("wf_oos_consistency", 0),
                "mc_pvalue": r.get("mc_pvalue", 1),
                "full_result": r,
            }
        print(f"  entry={entry:.1f} → WF={wf_sharpe:.3f} OOS_S={r.get('oos_sharpe',0):.3f} OOS_M={r.get('oos_monthly_return_pct',0):.2f}%")

    return best


def train_lightgbm(factors_df: pd.DataFrame, close: pd.Series, forward_bars: int = 96) -> Optional[dict]:
    """Train LightGBM directional classifier (approach C)."""
    try:
        import lightgbm as lgb
    except ImportError:
        print("  LightGBM not available — skipping")
        return None

    from rdagent.components.backtesting.vbt_backtest import backtest_signal_risk

    print("  Training LightGBM directional classifier...")
    fwd_ret = close.pct_change(forward_bars).shift(-forward_bars)
    common = factors_df.index.intersection(fwd_ret.dropna().index)
    X = factors_df.loc[common].ffill().values
    y = np.sign(fwd_ret.loc[common].values)

    split = int(len(X) * 0.7)
    X_train, X_test = X[:split], X[split:]
    y_train, y_test = y[:split], y[split:]

    model = lgb.LGBMClassifier(n_estimators=200, max_depth=6, num_leaves=31,
                                learning_rate=0.05, random_state=42, verbose=-1)
    model.fit(X_train, y_train)
    preds = model.predict(X_test)
    signal = pd.Series(preds, index=common[split:])

    r = backtest_signal_risk(close.loc[common[split:]], signal,
                             txn_cost_bps=TXN_COST_BPS, wf_rolling=True)
    wf = r.get("wf_oos_sharpe_mean", -999) or -999
    print(f"  LightGBM: WF_Sharpe={wf:.3f}")
    return {
        "method": "LightGBM",
        "wf_sharpe": wf,
        "oos_sharpe": r.get("oos_sharpe", -999),
        "oos_monthly": r.get("oos_monthly_return_pct", 0),
        "oos_dd": r.get("oos_max_drawdown", 0),
        "oos_trades": r.get("oos_n_trades", 0),
        "full_result": r,
    }


def main():
    print(f"\n{'='*60}")
    print("  NexQuant Systematic Strategy Generator")
    print(f"  Cost: {TXN_COST_BPS} bps | OOS: {OOS_START} | WF: {WF_WINDOWS} windows")
    print(f"{'='*60}\n")

    close, factors = load_data()
    print(f"Loaded: {len(close):,} bars, {len(factors)} factors")

    # Take top-10 diverse factors
    top_names = [f["name"] for f in factors[:10]]
    ics = {f["name"]: f["ic"] for f in factors[:10]}
    factors_df, close_a = load_factor_values(top_names, close)
    print(f"Aligned: {len(factors_df.columns)} factors, {len(close_a):,} bars\n")

    results = []

    # ---- Approach A+B: IC-weighted z-score + grid-search thresholds ----
    print("=== A+B: IC-Weighted Z-Score + Grid-Search Thresholds ===")
    t0 = time.time()
    composite = compute_ic_weighted_composite(factors_df, ics)
    best_thresh = grid_search(close_a, composite)
    if best_thresh:
        best_thresh["method"] = "IC-weighted + thresholds"
        best_thresh["composite_style"] = "zscore"
        best_thresh["factors_used"] = top_names[:5]
        results.append(best_thresh)
        print(f"  Best: entry={best_thresh['entry']:.1f} exit={best_thresh['exit']:.1f} "
              f"WF_Sharpe={best_thresh['wf_sharpe']:.3f} ({time.time()-t0:.0f}s)\n")

    # ---- Approach D: Factor-Ranking Top/Bottom ----
    print("=== D: Factor-Ranking Top/Bottom Deciles ===")
    t0 = time.time()
    sig_rank = generate_signal_ranking(factors_df, ics, top_pct=0.10)
    from rdagent.components.backtesting.vbt_backtest import backtest_signal_risk
    r_rank = backtest_signal_risk(close_a, sig_rank, txn_cost_bps=TXN_COST_BPS, wf_rolling=True)
    wf_rank = r_rank.get("wf_oos_sharpe_mean", -999) or -999
    results.append({
        "method": "Factor-Ranking D",
        "wf_sharpe": wf_rank,
        "oos_sharpe": r_rank.get("oos_sharpe", -999),
        "oos_monthly": r_rank.get("oos_monthly_return_pct", 0),
        "oos_dd": r_rank.get("oos_max_drawdown", 0),
        "oos_trades": r_rank.get("oos_n_trades", 0),
        "full_result": r_rank,
    })
    print(f"  Factor-Ranking: WF_Sharpe={wf_rank:.3f} ({time.time()-t0:.0f}s)\n")

    # ---- Approach C: LightGBM (if enough factors) ----
    if len(factors_df.columns) >= 5:
        print("=== C: LightGBM Directional Classifier ===")
        t0 = time.time()
        lgb_result = train_lightgbm(factors_df, close_a)
        if lgb_result:
            lgb_result["factors_used"] = top_names[:10]
            results.append(lgb_result)
        print(f"  ({time.time()-t0:.0f}s)\n")

    # ---- Report ----
    results.sort(key=lambda x: x.get("wf_sharpe", -999) or -999, reverse=True)

    print(f"\n{'='*60}")
    print("  RESULTS (sorted by Walk-Forward OOS Sharpe)")
    print(f"{'='*60}")
    print(f"{'Method':<30} {'WF Sharpe':>10} {'OOS Sharpe':>10} {'OOS Mon%':>8} {'OOS DD%':>8}")
    print("-" * 70)

    for r in results:
        wf = r.get("wf_sharpe", -999) or -999
        oos_s = r.get("oos_sharpe", -999)
        oos_m = (r.get("oos_monthly", 0) or 0)
        oos_d = (r.get("oos_dd", 0) or 0) * 100
        print(f"{r['method']:<30} {wf:>10.3f} {oos_s:>10.3f} {oos_m:>8.2f}% {oos_d:>7.1f}%")

    # Save best result
    if results:
        best = results[0]
        best["generated_at"] = datetime.now().isoformat()
        best["n_factors"] = len(factors_df.columns)
        best["n_bars"] = len(close_a)
        best["cost_bps"] = TXN_COST_BPS

        fname = f"systematic_{datetime.now().strftime('%Y%m%d_%H%M%S')}_{best['method'].replace(' ','_')[:40]}.json"
        with open(OUT_DIR / fname, "w") as f:
            json.dump({k: v for k, v in best.items() if k != "full_result"}, f, indent=2, default=str)
        print(f"\nBest strategy saved: {fname}")

    print()


if __name__ == "__main__":
    main()
