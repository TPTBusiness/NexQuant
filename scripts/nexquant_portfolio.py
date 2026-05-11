#!/usr/bin/env python
"""
NexQuant Multi-Asset Portfolio Generator — Target: 10%/month.
Combines best strategies per asset, optimizes position sizing, adds leverage.
"""

from __future__ import annotations

import json, sys
from pathlib import Path

import numpy as np
import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from rdagent.components.backtesting.vbt_backtest import backtest_signal_ftmo

DATA = Path("git_ignore_folder/factor_implementation_source_data/multi_asset_daily.h5")


def load_all():
    df = pd.read_hdf(DATA, key="data")
    close_dict = {}
    for col in df.columns:
        c = df[col].dropna()
        if len(c) > 500:
            close_dict[col] = c
    return close_dict


def rsi_signal(c, period, lo, hi):
    d = c.diff(); g = d.clip(lower=0); l = -d.clip(upper=0)
    rsi = 100 - (100 / (1 + g.rolling(period).mean() / (l.rolling(period).mean() + 1e-8)))
    sig = pd.Series(0.0, index=c.index)
    sig[rsi < lo] = 1; sig[rsi > hi] = -1
    return sig


def sma_signal(c, fast, slow):
    f = c.rolling(fast).mean(); s = c.rolling(slow).mean()
    sig = pd.Series(0.0, index=c.index)
    sig[f > s] = 1; sig[f < s] = -1
    return sig


def mr_signal(c, n):
    ret = c.pct_change(n)
    return pd.Series(-np.sign(ret).fillna(0), index=c.index)


def mom_signal(c, n):
    mom = c.pct_change(n)
    return pd.Series(np.sign(mom).fillna(0), index=c.index)


# Best strategy per asset (from our grid search)
STRATEGIES = {
    "OIL": lambda c: mr_signal(c, 50),
    "DXY": lambda c: sma_signal(c, 5, 25),
    "SPX": lambda c: mom_signal(c, 100),
    "EURUSD": lambda c: rsi_signal(c, 21, 25, 75),
    "USDJPY": lambda c: sma_signal(c, 50, 200),
    "GOLD": lambda c: rsi_signal(c, 21, 25, 75),
    "GBPUSD": lambda c: rsi_signal(c, 21, 25, 75),
}


def main():
    print(f"\n{'='*65}")
    print("  NexQuant Multi-Asset Portfolio — 10%/month Target")
    print(f"{'='*65}")

    closes = load_all()
    assets = sorted(closes.keys())
    print(f"Assets: {len(assets)} | Total bars: {max(len(c) for c in closes.values()):,}\n")

    aligned_signals = {}
    all_returns = []

    # Step 1: Generate signals per asset
    print("=== Individual Asset Performance ===")
    for name in assets:
        c = closes[name]
        sig_func = STRATEGIES.get(name, lambda c: rsi_signal(c, 21, 25, 75))
        sig = sig_func(c).fillna(0)

        r = backtest_signal_ftmo(c, sig, txn_cost_bps=2.14, wf_rolling=True)
        oos = r.get("wf_oos_sharpe_mean") or r.get("oos_sharpe", -999)
        oos_m = r.get("oos_monthly_return_pct", 0) or 0
        status = "✅" if oos > 0 else "  "
        print(f"  {name:<10} OOS={oos:+8.2f} Mon={oos_m:+7.3f}% {status}")

        aligned_signals[name] = sig
        # Monthly returns for this asset
        ret = c.pct_change() * sig.shift(1)
        ret.name = name
        all_returns.append(ret)

    # Step 2: Build equal-weight portfolio returns
    returns_df = pd.concat(all_returns, axis=1).dropna(how="all")
    common = returns_df.dropna().index
    returns_df = returns_df.loc[common].fillna(0)
    port_ret_equal = returns_df.mean(axis=1)

    print(f"\n=== Equal-Weight Portfolio ({len(returns_df.columns)} assets) ===")
    # Monthly returns
    monthly_eq = port_ret_equal.resample("M").apply(lambda x: (1 + x).prod() - 1) * 100
    months = len(monthly_eq.dropna())
    print(f"  Mean monthly: {monthly_eq.mean():+.3f}%")
    print(f"  Median monthly: {monthly_eq.median():+.3f}%")
    print(f"  Positive months: {(monthly_eq > 0).mean()*100:.1f}%")
    print(f"  Months: {months}")
    # Annualized
    ann_ret = (1 + port_ret_equal).prod() ** (252 / len(port_ret_equal)) - 1
    ann_vol = port_ret_equal.std() * np.sqrt(252)
    ann_sharpe = ann_ret / ann_vol if ann_vol > 0 else 0
    print(f"  Annual return: {ann_ret*100:.1f}%")
    print(f"  Annual vol: {ann_vol*100:.1f}%")
    print(f"  Annual Sharpe: {ann_sharpe:.3f}")

    # Step 3: Risk-parity weighting
    vols = returns_df.std()
    inv_vols = 1.0 / (vols + 1e-8)
    rp_weights = inv_vols / inv_vols.sum()
    port_ret_rp = (returns_df * rp_weights).sum(axis=1)

    monthly_rp = port_ret_rp.resample("M").apply(lambda x: (1 + x).prod() - 1) * 100
    print(f"\n=== Risk-Parity Portfolio ===")
    print(f"  Weights: {dict(zip(returns_df.columns, rp_weights.round(3)))}")
    print(f"  Mean monthly: {monthly_rp.mean():+.3f}%")
    print(f"  Positive months: {(monthly_rp > 0).mean()*100:.1f}%")
    ann_rp = (1 + port_ret_rp).prod() ** (252 / len(port_ret_rp)) - 1
    print(f"  Annual return: {ann_rp*100:.1f}%")

    # Step 4: With leverage
    print(f"\n=== With Leverage (2x, 3x, 5x) ===")
    for lev in [2, 3, 5]:
        port_lev = port_ret_rp * lev
        monthly_lev = port_lev.resample("M").apply(lambda x: (1 + x).prod() - 1) * 100
        ann_lev = (1 + port_lev).prod() ** (252 / len(port_lev)) - 1
        max_dd = (port_lev.cumsum().cummax() - port_lev.cumsum()).max()
        print(f"  {lev}x: Ann={ann_lev*100:+.1f}% Mon={monthly_lev.mean():+.2f}% MaxDD={max_dd*100:.1f}%")

    # Step 5: Check if 10% is reachable
    target_monthly = 10.0
    needed_lev = target_monthly / monthly_rp.mean() if monthly_rp.mean() > 0 else float("inf")
    print(f"\n=== Target: {target_monthly}%/month ===")
    print(f"  Current (risk-parity): {monthly_rp.mean():+.2f}%/month")
    print(f"  Leverage needed: {needed_lev:.1f}x")
    if needed_lev < 10:
        print(f"  ✅ Achievable with {needed_lev:.1f}x leverage")
    else:
        print(f"  ❌ Not achievable — need {needed_lev:.1f}x leverage")


if __name__ == "__main__":
    main()
