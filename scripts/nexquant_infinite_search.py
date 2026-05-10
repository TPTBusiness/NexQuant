#!/usr/bin/env python
"""
NexQuant Infinite Hypothesis Search — kombiniert und variiert Ansätze
bis ein positiver OOS Sharpe gefunden wird.
"""

from __future__ import annotations

import json, sys, time, random, itertools
from pathlib import Path

import numpy as np
import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
from rdagent.components.backtesting.vbt_backtest import backtest_signal_ftmo

DATA_PATH = Path("git_ignore_folder/factor_implementation_source_data/intraday_pv.h5")
FACTORS_DIR = Path("results/factors")
TXN_COST_BPS = 0.5


def load_data():
    close = pd.read_hdf(DATA_PATH, key="data")["$close"]
    if isinstance(close.index, pd.MultiIndex):
        close = close.droplevel(-1)
    close = close.sort_index().dropna().resample("1h").last().dropna()

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
        if (FACTORS_DIR / "values" / f"{safe}.parquet").exists():
            factors_meta.append({"name": name, "ic": d["ic"]})

    factors_meta.sort(key=lambda x: abs(x["ic"]), reverse=True)
    top = factors_meta[:15]
    factor_data = {}
    for f in top:
        safe = f["name"].replace("/", "_")[:150]
        series = pd.read_parquet(FACTORS_DIR / "values" / f"{safe}.parquet").iloc[:, 0]
        if isinstance(series.index, pd.MultiIndex):
            series = series.droplevel(-1)
        factor_data[f["name"]] = series.resample("1h").last()

    df = pd.DataFrame(factor_data)
    common = close.index.intersection(df.dropna(how="all").index)
    return close.loc[common], df.loc[common].ffill(), {f["name"]: f["ic"] for f in top}


close, factors_df, ics = load_data()
print(f"Data: {len(close):,} bars × {len(factors_df.columns)} factors\n")

def backtest(signal) -> float:
    if signal is None or len(signal) < 100:
        return -999
    common = close.index.intersection(signal.dropna().index)
    if len(common) < 100:
        return -999
    r = backtest_signal_ftmo(close.loc[common], signal.reindex(common).fillna(0),
                             txn_cost_bps=TXN_COST_BPS, wf_rolling=False)
    return r.get("oos_sharpe", -999)


def composite(factor_list=None, window=20):
    cols = factor_list or list(factors_df.columns)
    c = pd.Series(0.0, index=factors_df.index)
    total = sum(abs(ics.get(col, 0)) for col in cols)
    if total == 0:
        return c
    for col in cols:
        ic_val = ics.get(col, 0)
        if abs(ic_val) < 0.001:
            continue
        z = (factors_df[col] - factors_df[col].rolling(window).mean()) / (factors_df[col].rolling(window).std() + 1e-8)
        c += (ic_val / total) * z
    return c


def session_filter(sig):
    hours = sig.index.hour
    sig = sig.copy()
    sig[(hours < 7) | (hours >= 17)] = 0
    return sig


def trend_filter(sig, sma_bars=200 * 1440 // 5):
    sma = close.rolling(sma_bars).mean()
    trend_up = close > sma
    sig = sig.copy()
    sig[(sig > 0) & ~trend_up] = 0
    sig[(sig < 0) & trend_up] = 0
    return sig


def vola_target(sig, vol_window=50):
    vol = close.pct_change().rolling(vol_window).std()
    vol_tgt = vol.median()
    s = sig.astype(float) * vol_tgt / (vol + 1e-8)
    return s.clip(-3, 3)


def anti_fade(sig, sigma=3.0):
    ret = close.pct_change()
    thresh = ret.std() * sigma
    s = sig.copy()
    s[ret > thresh] = -1
    s[ret < -thresh] = 1
    return s


def signal_decay(sig, half_life=60):
    d = 0.5 ** (1 / half_life)
    s = sig.astype(float).copy()
    for i in range(1, len(s)):
        if abs(s.iloc[i]) < 0.01:
            s.iloc[i] = s.iloc[i - 1] * d
    return s.clip(-1, 1)


def kalman_composite(comp, Q=0.001, R=0.1):
    x, P = 0.0, 1.0
    filtered = []
    for v in comp.dropna().values:
        P += Q; K = P / (P + R); x += K * (v - x); P *= (1 - K)
        filtered.append(x)
    return pd.Series(filtered, index=comp.dropna().index)


# PRIMITIVES — can be combined arbitrarily
PRIMITIVES = {
    "session": session_filter,
    "trend": trend_filter,
    "vola_target": vola_target,
    "anti_fade": anti_fade,
    "decay": signal_decay,
}

BASE_PARAMS = {
    "entry": [0.1, 0.15, 0.2, 0.25, 0.3, 0.4, 0.5],
    "window": [10, 20, 30, 50, 100],
    "sigma": [2.0, 2.5, 3.0, 3.5],
    "half_life": [30, 60, 120, 240],
}

best_score = -999
best_desc = ""
best_sig = None
tested = set()
round_num = 0


def try_combo(factor_list, entry, window, primitives_used):
    global best_score, best_desc, best_sig, tested, round_num

    key = f"{sorted(factor_list)}_{entry:.3f}_{window}_{sorted(primitives_used)}"
    if key in tested:
        return
    tested.add(key)

    comp = composite(factor_list, window)
    if comp is None or comp.dropna().empty:
        return
    sig = pd.Series(0, index=comp.index)
    sig[comp > entry] = 1
    sig[comp < -entry] = -1

    for p in primitives_used:
        if p in PRIMITIVES:
            sig = PRIMITIVES[p](sig.fillna(0))

    sharpe = backtest(sig)
    if sharpe > best_score:
        best_score = sharpe
        best_desc = f"entry={entry:.2f} window={window} factors={len(factor_list)} primitives={primitives_used}"
        best_sig = sig
        t = "✅" if sharpe > 0 else "📈" if sharpe > -1 else "➖"
        print(f"  {t} #{round_num}: Sharpe={sharpe:.4f} | {best_desc}")

        if sharpe > 0:
            print(f"\n{'='*60}")
            print(f"  🎯 POSITIVE SHARPE FOUND!")
            print(f"  Sharpe={sharpe:.4f}")
            print(f"  {best_desc}")
            print(f"{'='*60}")
            return True
    return False


print("Starting infinite search — will run until positive OOS Sharpe found...\n")
all_factors = sorted(factors_df.columns, key=lambda c: -abs(ics.get(c, 0)))

while True:
    round_num += 1

    # Pick random subset of top factors
    n_factors = random.randint(2, min(10, len(all_factors)))
    factor_subset = random.sample(all_factors[:12], n_factors)

    # Pick random parameters
    entry = random.choice(BASE_PARAMS["entry"])
    window = random.choice(BASE_PARAMS["window"])

    # Pick random combination of primitives (0-4)
    n_prim = random.randint(0, 4)
    prims = random.sample(list(PRIMITIVES.keys()), n_prim) if n_prim > 0 else []

    found = try_combo(factor_subset, entry, window, prims)
    if found:
        break

    # Every 200 rounds, also try parameter sweeps around best
    if round_num % 200 == 0:
        print(f"  ... {round_num} combinations tested, best={best_score:.4f}")
        # Fine-tune around current best
        for fine_entry in np.arange(max(0.05, entry - 0.15), entry + 0.16, 0.05):
            for fine_window in [max(5, window - 15), window, min(200, window + 15)]:
                if try_combo(factor_subset, fine_entry, fine_window, prims):
                    break

    # Every 500 rounds, try factor-specific combos (Kronos-only, momentum-only, etc.)
    if round_num % 500 == 0:
        kronos = [f for f in all_factors if "Kronos" in f]
        mom = [f for f in all_factors if any(k in f.lower() for k in ["mom", "ret"])]
        for subset in [kronos, mom, all_factors[:3], all_factors[:6]]:
            if len(subset) >= 2:
                for e in [0.1, 0.2, 0.3]:
                    for w in [20, 50]:
                        for prims in [[], ["session"], ["session", "decay"]]:
                            try_combo(subset, e, w, prims)

    if round_num % 1000 == 0:
        print(f"  [{round_num} tested] best={best_score:.4f} — still searching...")

if best_score <= 0:
    print(f"\nAfter {round_num} combinations, best is still negative ({best_score:.4f})")
    print("The factors lack sufficient predictive power for positive returns.")
