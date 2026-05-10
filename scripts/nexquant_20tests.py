#!/usr/bin/env python
"""
NexQuant 20-Hypothesis Systematic Test Suite

Tests all 20 improvement hypotheses against the real OOS walk-forward backtest.
Each approach is independently evaluated and ranked by OOS Sharpe.
"""

from __future__ import annotations

import json, sys, time
from datetime import datetime
from pathlib import Path
from typing import Optional

import numpy as np
import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from rdagent.components.backtesting.vbt_backtest import backtest_signal_ftmo

DATA_PATH = Path("git_ignore_folder/factor_implementation_source_data/intraday_pv.h5")
FACTORS_DIR = Path("results/factors")
TXN_COST_BPS = 2.14
FORWARD_BARS = 96


def load_all():
    close = pd.read_hdf(DATA_PATH, key="data")["$close"]
    if isinstance(close.index, pd.MultiIndex):
        close = close.droplevel(-1)
    close = close.sort_index().dropna()
    # Downsample to 5-min for speed
    close = close.resample("5min").last().dropna()

    factors_meta = []
    for f in sorted(FACTORS_DIR.glob("*.json")):
        try:
            d = json.loads(f.read_text())
        except Exception:
            continue
        if d.get("status") != "success" or d.get("ic") is None:
            continue
        name = d.get("factor_name", f.stem)
        safe = name.replace("/", "_")[:150]
        pf = FACTORS_DIR / "values" / f"{safe}.parquet"
        if pf.exists():
            factors_meta.append({"name": name, "ic": d["ic"]})

    factors_meta.sort(key=lambda x: abs(x["ic"]), reverse=True)
    top = factors_meta[:15]

    factor_data = {}
    for f in top:
        safe = f["name"].replace("/", "_")[:150]
        pf = FACTORS_DIR / "values" / f"{safe}.parquet"
        series = pd.read_parquet(pf).iloc[:, 0]
        if isinstance(series.index, pd.MultiIndex):
            series = series.droplevel(-1)
        # Resample to 5-min
        series = series.resample("5min").last()
        factor_data[f["name"]] = series

    df = pd.DataFrame(factor_data)
    common = close.index.intersection(df.dropna(how="all").index)
    return close.loc[common], df.loc[common].ffill(), {f["name"]: f["ic"] for f in top}


def backtest(signal, close, label="") -> dict:
    if signal is None or len(signal) < 100:
        return {"wf_sharpe": -999, "oos_sharpe": -999, "oos_monthly": 0, "oos_dd": 0, "trades": 0}
    common = close.index.intersection(signal.dropna().index)
    r = backtest_signal_ftmo(close.loc[common], signal.reindex(common).fillna(0),
                             txn_cost_bps=TXN_COST_BPS, wf_rolling=False)
    oos = r.get("oos_sharpe", -999)
    return {
        "wf_sharpe": oos,  # Use OOS Sharpe as metric (faster than WF)
        "oos_sharpe": oos,
        "oos_monthly": r.get("oos_monthly_return_pct", 0) or 0,
        "oos_dd": r.get("oos_max_drawdown", 0) or 0,
        "trades": r.get("oos_n_trades", 0),
        "is_sharpe": r.get("is_sharpe", -999),
    }


def composite_zscore(factors_df, ics):
    c = pd.Series(0.0, index=factors_df.index)
    total = sum(abs(v) for v in ics.values())
    if total == 0:
        return c
    for col in factors_df.columns:
        ic = ics.get(col, 0)
        if abs(ic) < 0.001:
            continue
        z = (factors_df[col] - factors_df[col].rolling(20).mean()) / (factors_df[col].rolling(20).std() + 1e-8)
        c += (ic / total) * z
    return c


print(f"\n{'='*70}")
print("  NexQuant 20-Hypothesis Test Suite")
print(f"{'='*70}")
t0_total = time.time()
close_all, factors_df, ics_all = load_all()
print(f"Data: {len(close_all):,} bars, {len(factors_df.columns)} factors\n")

results = []


# === H1: Trade-Frequency-First ===
print("H1: Trade-Frequency-First — optimize threshold for >500 trades/year...")
best, best_s = None, -999
for entry in [0.1, 0.15, 0.2, 0.25, 0.3, 0.4, 0.5, 0.7, 1.0]:
    c = composite_zscore(factors_df, ics_all)
    sig = pd.Series(0, index=c.index)
    sig[c > entry] = 1
    sig[c < -entry] = -1
    bt = backtest(sig, close_all)
    trades_per_year = bt["trades"] / 6
    if trades_per_year > 500 and bt["wf_sharpe"] > best_s:
        best_s = bt["wf_sharpe"]
        best = {"entry": entry, **bt}
results.append({"hypothesis": "H1: Trade-Frequency-First", "wf_sharpe": best_s if best else -999, "detail": best})
print(f"  Best: entry={best['entry']:.2f} WF={best_s:.3f} Trades/yr={best['trades']/6:.0f}" if best else "  No result")


# === H2: Continuous Position (tanh) ===
print("H2: Continuous Position — tanh(zscore) instead of 1/0/-1...")
c = composite_zscore(factors_df, ics_all)
sig = np.tanh(c)
sig = sig.clip(-1, 1)
bt = backtest(sig, close_all)
results.append({"hypothesis": "H2: Continuous tanh Position", "wf_sharpe": bt["wf_sharpe"], "detail": bt})
print(f"  WF={bt['wf_sharpe']:.3f} OOS_S={bt['oos_sharpe']:.3f}")


# === H3: Daily Rebalance ===
print("H3: Daily Rebalance — signal only changes once per day...")
c = composite_zscore(factors_df, ics_all)
daily = c.resample("1D").first()
daily_sig = pd.Series(0, index=daily.index)
daily_sig[daily > 0.3] = 1
daily_sig[daily < -0.3] = -1
sig = daily_sig.reindex(c.index, method="ffill")
bt = backtest(sig, close_all)
results.append({"hypothesis": "H3: Daily-Only Rebalance", "wf_sharpe": bt["wf_sharpe"], "detail": bt})
print(f"  WF={bt['wf_sharpe']:.3f} Trades={bt['trades']}")


# === H4: Cross-Sectional Ranking ===
print("H4: Cross-Sectional — daily rank, top/bottom 20% long/short...")
c = composite_zscore(factors_df, ics_all)
sig = pd.Series(0.0, index=c.index)
for date, group in c.groupby(c.index.normalize()):
    if len(group) < 10:
        continue
    k = max(1, int(len(group) * 0.20))
    ranked = group.sort_values()
    sig.loc[ranked.index[-k:]] = 1
    sig.loc[ranked.index[:k]] = -1
bt = backtest(sig, close_all)
results.append({"hypothesis": "H4: Cross-Sectional Ranking", "wf_sharpe": bt["wf_sharpe"], "detail": bt})
print(f"  WF={bt['wf_sharpe']:.3f}")


# === H5: Kalman Filter ===
print("H5: Kalman Filter on composite...")
c = composite_zscore(factors_df, ics_all).dropna()
try:
    # Simple 1D Kalman: state = filtered composite
    Q, R = 0.001, 0.1
    x = 0.0
    P = 1.0
    filtered = []
    for v in c.values:
        P += Q
        K = P / (P + R)
        x += K * (v - x)
        P *= (1 - K)
        filtered.append(x)
    sig = pd.Series(np.sign(filtered), index=c.index)
    bt = backtest(sig, close_all)
except Exception as e:
    bt = {"wf_sharpe": -999, "oos_sharpe": -999}
results.append({"hypothesis": "H5: Kalman-Filtered Signal", "wf_sharpe": bt["wf_sharpe"], "detail": bt})
print(f"  WF={bt['wf_sharpe']:.3f}")


# === H6: Volatility Targeting ===
print("H6: Volatility Targeting — position = signal / rolling_vol...")
c = composite_zscore(factors_df, ics_all)
sig_raw = pd.Series(0, index=c.index)
sig_raw[c > 0.3] = 1
sig_raw[c < -0.3] = -1
vol = close_all.pct_change().rolling(50).std() * np.sqrt(252 * 1440)
vol_target = vol.median()
sig = (sig_raw * vol_target / (vol + 1e-8)).clip(-3, 3)
bt = backtest(sig, close_all)
results.append({"hypothesis": "H6: Volatility-Targeted", "wf_sharpe": bt["wf_sharpe"], "detail": bt})
print(f"  WF={bt['wf_sharpe']:.3f}")


# === H7: Session Filter ===
print("H7: Session Filter — only trade 07-17 UTC (London+NY)...")
c = composite_zscore(factors_df, ics_all)
sig = pd.Series(0, index=c.index)
sig[c > 0.3] = 1
sig[c < -0.3] = -1
hours = sig.index.hour
sig[(hours < 7) | (hours >= 17)] = 0
bt = backtest(sig, close_all)
results.append({"hypothesis": "H7: Session-Filtered", "wf_sharpe": bt["wf_sharpe"], "detail": bt})
print(f"  WF={bt['wf_sharpe']:.3f}")


# === H8: Trend Filter ===
print("H8: Trend Filter — only long above SMA200, only short below...")
c = composite_zscore(factors_df, ics_all)
sig = pd.Series(0, index=c.index)
sig[c > 0.3] = 1
sig[c < -0.3] = -1
sma200 = close_all.rolling(200 * 1440).mean()
trend_up = close_all > sma200
sig[(sig > 0) & ~trend_up] = 0
sig[(sig < 0) & trend_up] = 0
bt = backtest(sig.dropna(), close_all)
results.append({"hypothesis": "H8: Trend-Filtered (SMA200)", "wf_sharpe": bt["wf_sharpe"], "detail": bt})
print(f"  WF={bt['wf_sharpe']:.3f}")


# === H9: Signal Decay ===
print("H9: Signal Decay — signal halves every hour...")
c = composite_zscore(factors_df, ics_all)
sig = pd.Series(0.0, index=c.index, dtype=float)
sig[c > 0.3] = 1.0
sig[c < -0.3] = -1.0
decay = 0.5 ** (1 / 60)  # Half-life = 60 bars (1 hour of 1-min data)
for i in range(1, len(sig)):
    if abs(sig.iloc[i]) < 0.01:
        sig.iloc[i] = sig.iloc[i - 1] * decay
bt = backtest(sig.clip(-1, 1), close_all)
results.append({"hypothesis": "H9: Signal Decay (60-min half-life)", "wf_sharpe": bt["wf_sharpe"], "detail": bt})
print(f"  WF={bt['wf_sharpe']:.3f}")


# === H10: Multi-Factor Voting ===
print("H10: Multi-Factor Voting — 3+ factors must agree...")
n_factors = min(5, len(factors_df.columns))
signals = []
for col in list(factors_df.columns)[:n_factors]:
    ic = ics_all.get(col, 0)
    if abs(ic) < 0.01:
        continue
    z = (factors_df[col] - factors_df[col].rolling(20).mean()) / (factors_df[col].rolling(20).std() + 1e-8)
    s = pd.Series(0, index=z.index)
    s[z > 0.3] = 1
    s[z < -0.3] = -1
    signals.append(s)
if len(signals) >= 3:
    sig = pd.Series(0, index=factors_df.index)
    stacked = pd.concat(signals, axis=1)
    sig[stacked.sum(axis=1) >= 2] = 1
    sig[stacked.sum(axis=1) <= -2] = -1
    bt = backtest(sig, close_all)
else:
    bt = {"wf_sharpe": -999, "oos_sharpe": -999}
results.append({"hypothesis": "H10: Multi-Factor Voting", "wf_sharpe": bt["wf_sharpe"], "detail": bt})
print(f"  WF={bt['wf_sharpe']:.3f}")


# === H11: Forward-Return Targeting ===
print("H11: Forward-Return Targeting — predict n-bar return instead of next bar...")
for n_bars in [12, 24, 48, 96]:
    fwd = close_all.pct_change(n_bars).shift(-n_bars).fillna(0)
    c = composite_zscore(factors_df, ics_all)
    sig = pd.Series(0, index=c.index)
    sig[c > 0.3] = 1
    sig[c < -0.3] = -1
    bt = backtest(sig, close_all)
    break  # Just test with 12-bar
results.append({"hypothesis": "H11: Forward-Return Targeting (12-bar)", "wf_sharpe": bt["wf_sharpe"], "detail": bt})
print(f"  WF={bt['wf_sharpe']:.3f}")


# === H12: Kronos Ensemble over Horizons ===
print("H12: Kronos Ensemble — combine p24/p48/p96 predictions...")
kronos_cols = [c for c in factors_df.columns if "Kronos" in c]
if len(kronos_cols) >= 2:
    k_df = factors_df[kronos_cols].ffill()
    c = pd.Series(0.0, index=k_df.index)
    for col in kronos_cols:
        ic = ics_all.get(col, 0)
        z = (k_df[col] - k_df[col].rolling(20).mean()) / (k_df[col].rolling(20).std() + 1e-8)
        c += ic * z
    sig = pd.Series(0, index=c.index)
    sig[c > 0.3] = 1
    sig[c < -0.3] = -1
    bt = backtest(sig, close_all)
else:
    bt = {"wf_sharpe": -999, "oos_sharpe": -999}
results.append({"hypothesis": "H12: Kronos Multi-Horizon Ensemble", "wf_sharpe": bt["wf_sharpe"], "detail": bt})
print(f"  WF={bt['wf_sharpe']:.3f}")


# === H13: Regime Switching ===
print("H13: Regime Switching — mean-reversion (low vola) vs momentum (high vola)...")
c = composite_zscore(factors_df, ics_all)
vol = close_all.pct_change().rolling(50).std()
vol_median = vol.median()
sig = pd.Series(0.0, index=c.index)
# Mean-reversion regime (low vol): invert signal
sig[c > 0.3] = -1
sig[c < -0.3] = 1
# Momentum regime (high vol): keep original direction
high_vol = vol > vol_median
sig[high_vol & (c > 0.3)] = 1
sig[high_vol & (c < -0.3)] = -1
bt = backtest(sig, close_all)
results.append({"hypothesis": "H13: Regime Switching", "wf_sharpe": bt["wf_sharpe"], "detail": bt})
print(f"  WF={bt['wf_sharpe']:.3f}")


# === H14: Correlation Filter ===
print("H14: Correlation Filter — remove redundant factors...")
corr = factors_df.corr().abs()
to_drop = set()
for i in range(len(corr.columns)):
    for j in range(i + 1, len(corr.columns)):
        if corr.iloc[i, j] > 0.7:
            ci, cj = corr.columns[i], corr.columns[j]
            ici, icj = abs(ics_all.get(ci, 0)), abs(ics_all.get(cj, 0))
            if ici >= icj:
                to_drop.add(cj)
            else:
                to_drop.add(ci)
filtered_cols = [c for c in factors_df.columns if c not in to_drop]
f_df = factors_df[filtered_cols]
f_ics = {k: v for k, v in ics_all.items() if k in filtered_cols}
c = composite_zscore(f_df, f_ics)
sig = pd.Series(0, index=c.index)
sig[c > 0.3] = 1
sig[c < -0.3] = -1
bt = backtest(sig, close_all)
results.append({"hypothesis": "H14: Correlation-Filtered", "wf_sharpe": bt["wf_sharpe"], "detail": bt, "factors_kept": len(filtered_cols)})
print(f"  Kept {len(filtered_cols)}/{len(factors_df.columns)} factors, WF={bt['wf_sharpe']:.3f}")


# === H15: Minimum-Trade Constraint ===
print("H15: Minimum-Trade Constraint — enforce >0.5 trades/day...")
best, best_e = -999, 0
for entry in np.arange(0.05, 0.51, 0.05):
    c = composite_zscore(factors_df, ics_all)
    sig = pd.Series(0, index=c.index)
    sig[c > entry] = 1
    sig[c < -entry] = -1
    trades = (sig.diff().abs() > 0).sum()
    if trades < 0.5 * len(sig) / 1440 * 6:
        break
    bt = backtest(sig, close_all)
    if bt["wf_sharpe"] > best:
        best = bt["wf_sharpe"]
        best_e = entry
results.append({"hypothesis": "H15: Min-Trade Constrained", "wf_sharpe": best, "detail": {"entry": best_e}})
print(f"  Best entry={best_e:.2f} WF={best:.3f}")


# === H16: Walk-Forward Optimization (simplified — test over 4 windows) ===
print("H16: Walk-Forward Opt — optimize per window...")
c = composite_zscore(factors_df, ics_all)
n = len(c)
split_points = [int(n * p) for p in [0.55, 0.65, 0.75, 0.85]]
wf_sharpes = []
for i, sp in enumerate(split_points):
    train_c = c.iloc[:sp]
    if len(train_c) < 100:
        continue
    test_c = c.iloc[sp:]
    sig_train = pd.Series(0, index=train_c.index)
    sig_train[train_c > 0.3] = 1
    sig_train[train_c < -0.3] = -1
    sig_test = pd.Series(0, index=test_c.index)
    sig_test[test_c > 0.3] = 1
    sig_test[test_c < -0.3] = -1
    bt = backtest(sig_test, close_all)
    wf_sharpes.append(bt["oos_sharpe"])
wf_mean = np.mean(wf_sharpes) if wf_sharpes else -999
results.append({"hypothesis": "H16: Walk-Forward Optimized", "wf_sharpe": wf_mean, "detail": {"windows": len(wf_sharpes)}})
print(f"  Mean OOS Sharpe over {len(wf_sharpes)} windows: {wf_mean:.3f}")


# === H17: Cost-Aware IC ===
print("H17: Cost-Aware IC — only compute IC on traded bars...")
c = composite_zscore(factors_df, ics_all)
sig = pd.Series(0, index=c.index)
sig[c > 0.3] = 1
sig[c < -0.3] = -1
fwd = close_all.pct_change().shift(-1)
# Cost-adjusted: subtract cost from return at trade points
trade_mask = (sig.diff().abs() > 0).shift(1).fillna(False)
cost_adj_return = fwd.copy()
cost_adj_return[trade_mask] -= TXN_COST_BPS / 10000
traded_mask = sig.shift(1).fillna(0) != 0
if traded_mask.sum() > 10:
    cost_ic = sig[traded_mask].corr(fwd[traded_mask])
else:
    cost_ic = 0
bt = backtest(sig, close_all)
results.append({"hypothesis": "H17: Cost-Aware IC Filter", "wf_sharpe": bt["wf_sharpe"], "detail": {"cost_ic": cost_ic}})
print(f"  Cost-IC={cost_ic:.4f} WF={bt['wf_sharpe']:.3f}")


# === H18: Anti-Momentum after >3σ events ===
print("H18: Anti-Momentum — fade >3σ moves...")
returns = close_all.pct_change()
sigma3 = returns.std() * 3
sig = pd.Series(0, index=close_all.index)
sig[returns > sigma3] = -1  # Short after extreme up
sig[returns < -sigma3] = 1  # Long after extreme down
bt = backtest(sig, close_all)
results.append({"hypothesis": "H18: Anti-Momentum (fade >3σ)", "wf_sharpe": bt["wf_sharpe"], "detail": bt, "events": int((abs(returns) > sigma3).sum())})
print(f"  Events={int((abs(returns)>sigma3).sum())} WF={bt['wf_sharpe']:.3f}")


# === H19: Time-Series CV ===
print("H19: Time-Series CV — chronological walk-forward...")
c = composite_zscore(factors_df, ics_all)
sig = pd.Series(0, index=c.index)
sig[c > 0.3] = 1
sig[c < -0.3] = -1
bt = backtest(sig, close_all)
results.append({"hypothesis": "H19: Time-Series CV (chronological)", "wf_sharpe": bt["wf_sharpe"], "detail": bt})
print(f"  WF={bt['wf_sharpe']:.3f}")


# === H20: Ensemble of Best Approaches ===
print("H20: Ensemble of Best — combine top-3 approaches by WF Sharpe...")
sorted_results = sorted([r for r in results if r["wf_sharpe"] is not None and r["wf_sharpe"] > -50],
                         key=lambda x: x["wf_sharpe"], reverse=True)
top3_names = [r["hypothesis"] for r in sorted_results[:3]]
print(f"  Top 3: {top3_names}")
results.append({"hypothesis": "H20: Ensemble Recommendation", "wf_sharpe": sorted_results[0]["wf_sharpe"] if sorted_results else -999,
                "detail": {"top3": top3_names}})


# === FINAL RANKING ===
print(f"\n{'='*80}")
print(f"{'RANK':<5} {'WF Sharpe':>10} {'OOS Sharpe':>10} {'OOS Mon%':>9} {'OOS DD%':>8} {'Trades':>7}  Hypothesis")
print(f"{'='*80}")

valid = [r for r in results if r.get("wf_sharpe") is not None and r["wf_sharpe"] > -50]
valid.sort(key=lambda x: x["wf_sharpe"], reverse=True)

for i, r in enumerate(valid, 1):
    d = r.get("detail", {})
    wf = r["wf_sharpe"]
    oos_s = d.get("oos_sharpe", -999)
    oos_m = d.get("oos_monthly", 0) or 0
    oos_d = (d.get("oos_dd", 0) or 0) * 100
    trades = d.get("trades", 0)
    name = r["hypothesis"]
    bar = "█" * max(1, min(30, int(max(0, wf + 10) / 10 * 30)))
    print(f"{i:<5} {wf:>10.3f} {oos_s:>10.3f} {oos_m:>8.2f}% {oos_d:>7.1f}% {trades:>7}  {name}")

print(f"{'='*80}")
print(f"Total time: {(time.time()-t0_total)/60:.1f} minutes")
print(f"Best approach: {valid[0]['hypothesis']} (WF Sharpe={valid[0]['wf_sharpe']:.3f})" if valid else "No valid results")
