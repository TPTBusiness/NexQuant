#!/usr/bin/env python3
"""Daily Strategy Generator — Kronos factors at daily resolution.

Daily timeframe eliminates 1-min noise and transaction cost overhead.
Factors with daily IC translate directly to daily trading edge.
"""

import json
import os
import time
from datetime import datetime
from pathlib import Path

import numpy as np
import pandas as pd

PROJECT = Path(__file__).resolve().parent.parent
FACTORS_DIR = PROJECT / "results" / "factors"
VALUES_DIR = FACTORS_DIR / "values"
RESULTS_DIR = PROJECT / "results" / "strategies_new"
OHLCV_PATH = Path(os.getenv("PREDIX_OHLCV_PATH",
    str(PROJECT / "git_ignore_folder" / "intraday_pv_all.h5")))

MIN_MONTHLY = 5.0       # Raw backtest target (conservative for daily)
MIN_SHARPE  = 1.0
MAX_DD      = -0.20
MIN_TRADES  = 30


def load_kronos(name: str) -> pd.Series:
    s = pd.read_parquet(VALUES_DIR / f"{name}.parquet")
    col = s.columns[0]
    return s.xs("EURUSD", level="instrument")[col]


def load_factor_ic(name: str) -> float:
    jf = FACTORS_DIR / f"{name}.json"
    if jf.exists():
        return float(json.loads(jf.read_text()).get("ic", 0))
    return 0.0


def daily_backtest(close_daily: pd.Series, signal_daily: pd.Series) -> dict:
    """Simple daily backtest — no intraday noise, no 1-min costs."""
    common = close_daily.index.intersection(signal_daily.index)
    c = close_daily.loc[common]
    s = signal_daily.loc[common].clip(-1, 1)

    rets = c.pct_change().shift(-1)  # Next day's return
    strat_rets = s.shift(1) * rets  # Today's signal × tomorrow's return
    strat_rets = strat_rets.dropna()

    if len(strat_rets) < 10:
        return {"sharpe": 0, "monthly_pct": 0, "max_dd": 0, "n_trades": 0, "win_rate": 0}

    # Trade-level stats
    trades = []
    in_trade = False
    trade_ret = 0.0
    wins = 0
    for r, sig in zip(strat_rets, s.loc[strat_rets.index]):
        if sig != 0:
            if not in_trade:
                in_trade = True
                trade_ret = r
            else:
                trade_ret += r
        elif in_trade:
            in_trade = False
            trades.append(trade_ret)
            if trade_ret > 0:
                wins += 1
            trade_ret = 0.0
    if in_trade:
        trades.append(trade_ret)
        if trade_ret > 0:
            wins += 1

    n_trades = len(trades)
    if n_trades < 5:
        return {"sharpe": 0, "monthly_pct": 0, "max_dd": 0, "n_trades": n_trades, "win_rate": 0}

    t_arr = np.array(trades)
    sharpe = float(t_arr.mean() / t_arr.std() * np.sqrt(n_trades)) if t_arr.std() > 0 else 0.0
    win_rate = wins / n_trades

    # Equity curve
    eq = (1 + pd.Series(trades)).cumprod()
    peak = eq.cummax()
    dd = float(((eq - peak) / peak).min())

    total_ret = eq.iloc[-1] - 1 if len(eq) > 0 else 0.0
    n_days = (close_daily.index[-1] - close_daily.index[0]).days
    n_months = n_days / 30.44
    monthly = float((1 + total_ret) ** (1 / max(n_months, 1)) - 1)

    return {
        "sharpe": sharpe, "monthly_pct": monthly * 100,
        "max_dd": dd, "n_trades": n_trades, "win_rate": win_rate,
        "total_return": total_ret, "n_months": n_months,
    }


def build_signal(daily_factor: pd.Series, ic: float, threshold_sigma: float,
                 session: str = "all") -> pd.Series:
    """Build daily signal from a single factor."""
    sigma = daily_factor.std()
    thresh = threshold_sigma * sigma

    # Invert if IC is negative
    sign = -1 if ic < 0 else 1

    signal = pd.Series(0, index=daily_factor.index, dtype=int)
    signal[daily_factor > thresh] = sign
    signal[daily_factor < -thresh] = -sign

    # Smooth: keep signal for min_hold days to avoid whipsaw
    signal = signal.replace(0, np.nan).ffill(limit=1).fillna(0).astype(int)

    return signal


def combine_signals(s1: pd.Series, s2: pd.Series, mode: str = "confirm") -> pd.Series:
    """Combine two daily signals."""
    common = s1.index.intersection(s2.index)
    s1c = s1.loc[common]
    s2c = s2.loc[common]

    if mode == "confirm":
        result = pd.Series(0, index=common, dtype=int)
        result[(s1c == s2c) & (s1c != 0)] = s1c
        return result
    elif mode == "any":
        result = s1c.copy()
        result[(result == 0) & (s2c != 0)] = s2c
        return result
    else:
        return s1c


def main():
    print("=" * 60)
    print("  Daily Strategy Generator")
    print("=" * 60)

    # Load OHLCV → daily
    print("\nLoading OHLCV...")
    df = pd.read_hdf(OHLCV_PATH, key="data")
    close = df.xs("EURUSD", level="instrument")["$close"].sort_index()
    close_daily = close.resample("D").last().dropna()
    print(f"  Daily bars: {len(close_daily)} ({close_daily.index[0].date()} → {close_daily.index[-1].date()})")

    # Load Kronos factors → daily
    print("\nLoading Kronos factors...")
    kronos = {}
    for name in ["KronosPredReturn_p96", "KronosPredReturn_p24", "KronosPredReturn_p48"]:
        series = load_kronos(name)
        ic = load_factor_ic(name)
        daily = series.resample("D").last().dropna()
        # Align to close_daily
        daily = daily.reindex(close_daily.index)
        kronos[name] = {"series": daily, "ic": ic, "std": daily.std()}
        print(f"  {name}: IC={ic:+.4f} daily_rows={daily.dropna().sum()}")

    # Load top daily factors
    print("\nLoading top daily factors...")
    daily_factors = {}
    for f in sorted(FACTORS_DIR.glob("*.json")):
        d = json.loads(f.read_text())
        if not isinstance(d, dict):
            continue
        ic = float(d.get("ic") or 0)
        if abs(ic) < 0.06:
            continue
        fname = d.get("factor_name") or d.get("name") or f.stem
        safe = fname.replace("/", "_").replace("\\", "_")[:150]
        parq = VALUES_DIR / f"{safe}.parquet"
        if not parq.exists():
            continue
        series = pd.read_parquet(str(parq))
        if isinstance(series.index, pd.MultiIndex):
            series = series.xs("EURUSD", level="instrument")[series.columns[0]]
        daily = series.resample("D").last().dropna().reindex(close_daily.index)
        daily_factors[fname] = {"series": daily, "ic": ic, "std": daily.std()}

    names = list(daily_factors.keys())
    print(f"  Loaded {len(names)} factors (IC ≥ 0.06)")

    # Grid search
    thresholds = [1.0, 1.5, 2.0, 2.5, 3.0]
    results = []
    t0 = time.time()

    # A) Kronos single-factor
    print("\n--- Kronos single-factor grid ---")
    for kname, kdata in kronos.items():
        ks = kdata["series"]
        for thresh in thresholds:
            signal = build_signal(ks, kdata["ic"], thresh)
            bt = daily_backtest(close_daily, signal)
            bt["strategy"] = f"{kname} t={thresh}σ"
            bt["factors"] = [kname]
            bt["threshold"] = thresh
            results.append(bt)

    # B) Kronos + daily factor (confirmation)
    print("--- Kronos + daily factor combinations ---")
    for kname, kdata in kronos.items():
        ks = kdata["series"]
        for fname, fdata in daily_factors.items():
            for thresh_k in [1.5, 2.0]:
                for thresh_f in [1.0, 1.5, 2.0]:
                    s1 = build_signal(ks, kdata["ic"], thresh_k)
                    s2 = build_signal(fdata["series"], fdata["ic"], thresh_f)
                    signal = combine_signals(s1, s2, "confirm")
                    bt = daily_backtest(close_daily, signal)
                    bt["strategy"] = f"{kname}(t={thresh_k}) + {fname}(t={thresh_f})"
                    bt["factors"] = [kname, fname]
                    bt["threshold"] = f"{thresh_k}/{thresh_f}"
                    results.append(bt)

    # C) Two daily factors (no Kronos)
    print("--- Daily factor pairs ---")
    name_list = list(daily_factors.keys())
    for i in range(min(len(name_list), 10)):
        for j in range(i + 1, min(len(name_list), 10)):
            f1, f2 = name_list[i], name_list[j]
            for t1 in [1.0, 1.5, 2.0]:
                for t2 in [1.0, 1.5, 2.0]:
                    s1 = build_signal(daily_factors[f1]["series"], daily_factors[f1]["ic"], t1)
                    s2 = build_signal(daily_factors[f2]["series"], daily_factors[f2]["ic"], t2)
                    signal = combine_signals(s1, s2, "confirm")
                    bt = daily_backtest(close_daily, signal)
                    bt["strategy"] = f"{f1[:20]}(t={t1}) + {f2[:20]}(t={t2})"
                    bt["factors"] = [f1, f2]
                    bt["threshold"] = f"{t1}/{t2}"
                    results.append(bt)

    # Filter & sort
    print(f"\n{'=' * 60}")
    print(f"  Total evaluations: {len(results)}  Time: {time.time()-t0:.0f}s")
    print(f"{'=' * 60}")

    valid = [r for r in results
             if r["sharpe"] >= MIN_SHARPE
             and r["max_dd"] >= MAX_DD
             and r["n_trades"] >= MIN_TRADES
             and r["monthly_pct"] >= MIN_MONTHLY]

    valid.sort(key=lambda r: r["monthly_pct"], reverse=True)

    print(f"\n  Meeting: Sharpe≥{MIN_SHARPE} DD≥{MAX_DD} Tr≥{MIN_TRADES} Mon≥{MIN_MONTHLY}%")
    print(f"  → {len(valid)} strategies\n")

    fmt = "{:3s} {:55s} {:>7s} {:>7s} {:>7s} {:>5s} {:>6s}"
    print(fmt.format("#", "Strategy", "Sharpe", "Mon%", "MaxDD", "Tr", "WinRt"))
    print("-" * 90)
    for i, r in enumerate(valid[:30], 1):
        print(fmt.format(str(i), r["strategy"][:55],
              f'{r["sharpe"]:.2f}', f'{r["monthly_pct"]:.1f}%',
              f'{r["max_dd"]:.3f}', str(r["n_trades"]),
              f'{r["win_rate"]:.1%}'))

    if not valid:
        results.sort(key=lambda r: r["monthly_pct"], reverse=True)
        print("\n  Top 10 by monthly return:")
        for i, r in enumerate(results[:10], 1):
            print(f"  {i:2d}. {r['strategy'][:50]} Mon={r['monthly_pct']:.1f}% Sh={r['sharpe']:.2f} Tr={r['n_trades']}")

    # Save
    RESULTS_DIR.mkdir(parents=True, exist_ok=True)
    out = RESULTS_DIR / f"daily_{datetime.now().strftime('%Y%m%d_%H%M%S')}.json"
    out.write_text(json.dumps(valid[:50] if valid else results[:50], indent=2, default=str))
    print(f"\n  Saved → {out}")


if __name__ == "__main__":
    main()
