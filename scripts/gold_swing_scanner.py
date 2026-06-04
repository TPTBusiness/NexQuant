#!/usr/bin/env python3
"""Gold Swing Scanner — Daily strategies for position/swing trading.

Unlike the 1-min grid search, this targets multi-day holds on daily Gold data.
Tests: Trend-following, momentum, mean-reversion, breakout on 1-20 day horizons.
"""
import json, os, sys, time, itertools
from datetime import datetime
from pathlib import Path
import numpy as np, pandas as pd

PROJECT = Path(__file__).resolve().parent.parent
OUTPUT_DIR = PROJECT / "results" / "gold_swing"
OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

sys.path.insert(0, str(PROJECT / "scripts"))
from nexquant_rd_loop import _backtest_numba

def build_daily_signal(close, indicator, params):
    """Build signal on raw daily close (no resampling)."""
    import talib
    c = close.values.astype(np.float64)
    s = np.zeros(len(c), dtype=np.int32)

    if indicator == 'MACD':
        mc, sc, _ = talib.MACD(c, fastperiod=params.get('fast',12),
                               slowperiod=params.get('slow',26),
                               signalperiod=params.get('sig',9))
        s[mc > sc] = 1; s[mc < sc] = -1
    elif indicator == 'SMA':
        fa = pd.Series(c).rolling(params.get('fast',20)).mean().values
        sl = pd.Series(c).rolling(params.get('slow',50)).mean().values
        s[fa > sl] = 1; s[fa < sl] = -1
    elif indicator == 'EMA':
        fa = pd.Series(c).ewm(span=params.get('fast',12)).mean().values
        sl = pd.Series(c).ewm(span=params.get('slow',26)).mean().values
        s[fa > sl] = 1; s[fa < sl] = -1
    elif indicator == 'ROC':
        v = talib.ROC(c, timeperiod=params.get('period',20))
        th = params.get('threshold',2.0)
        s[v > th] = 1; s[v < -th] = -1
    elif indicator == 'MOM':
        v = talib.MOM(c, timeperiod=params.get('period',20))
        s[v > 0] = 1; s[v < 0] = -1
    elif indicator == 'RSI_OBOS':
        v = talib.RSI(c, timeperiod=params.get('period',14))
        s[v < params.get('oversold',30)] = 1; s[v > params.get('overbought',70)] = -1
    elif indicator == 'Donchian':
        hi = pd.Series(c).rolling(params.get('period',20)).max().shift(1).values
        lo = pd.Series(c).rolling(params.get('period',20)).min().shift(1).values
        s[c > hi] = 1; s[c < lo] = -1
        # Hold until reverse
        hold = params.get('hold',5)
        if hold > 0:
            last = 0; cnt = 0
            for i in range(len(s)):
                if s[i] != 0: last = s[i]; cnt = hold
                elif cnt > 0: s[i] = last; cnt -= 1
    elif indicator == 'BB':
        up, mi, lo = talib.BBANDS(c, timeperiod=params.get('period',20),
                                  nbdevup=params.get('std',2), nbdevdn=params.get('std',2))
        s[c < lo] = 1; s[c > up] = -1

    return pd.Series(s, index=close.index).fillna(0).astype(int).clip(-1,1)


# ── Grid Definition ──
INDICATOR_GRIDS = {
    'MACD': {
        'fast': [3,5,8,12,21],
        'slow': [10,15,21,26,34,50],
        'sig': [3,5,9,13],
    },
    'SMA': {
        'fast': [10,20,50,100],
        'slow': [20,50,100,200],
    },
    'EMA': {
        'fast': [5,8,12,21],
        'slow': [13,21,34,55],
    },
    'ROC': {
        'period': [5,10,20,50,100],
        'threshold': [0.5,1.0,2.0,3.0,5.0],
    },
    'MOM': {
        'period': [10,20,50,100],
    },
    'RSI_OBOS': {
        'period': [7,14,21],
        'oversold': [20,25,30,35],
        'overbought': [65,70,75,80],
    },
    'Donchian': {
        'period': [5,10,20,50,100],
        'hold': [0,1,3,5,10],
    },
    'BB': {
        'period': [10,20,50],
        'std': [1.5,2.0,2.5,3.0],
    },
}


def load_gold_daily():
    """Load daily Gold data."""
    path = PROJECT / "git_ignore_folder" / "xau_daily.h5"
    if path.exists():
        return pd.read_hdf(path, key="data")
    return None


def main():
    print("=" * 60)
    print("  Gold Swing Scanner — Daily Position Strategies")
    print("=" * 60)

    close = load_gold_daily()
    if close is None:
        print("  XAUUSD daily data not found! Run download first."); return
    print(f"  XAUUSD daily: {len(close)} bars, {close.index[0].date()} -> {close.index[-1].date()}")

    all_results = []
    total = 0
    for ind_name, grid in INDICATOR_GRIDS.items():
        keys = list(grid.keys())
        values = list(grid.values())
        for combo in itertools.product(*values):
            total += 1
            params = dict(zip(keys, combo))
            try:
                sig = build_daily_signal(close, ind_name, params)
                if sig is None or sig.nunique() <= 1: continue
            except: continue

            n = len(close); is_n = int(n * 0.8)
            if is_n < 10: continue  # too little data
            p = close.values.astype(float); s = sig.values.astype(np.int32)

            if np.sum(np.abs(s)) < 10: continue

            p_is = close.iloc[:is_n].values.astype(float); s_is = sig.iloc[:is_n].values.astype(np.int32)
            p_oos = close.iloc[is_n:].values.astype(float); s_oos = sig.iloc[is_n:].values.astype(np.int32)

            _, dd, tr, w, ret, sh, _ = _backtest_numba(p, s)
            _, _, tr_o, _, ret_o, sh_o, _ = _backtest_numba(p_oos, s_oos)

            nd = (close.index[-1] - close.index[0]).days
            if nd <= 0: continue
            mon = ((1+ret)**(1/(nd/30.44))-1)*100 if ret > -1 else 0
            nd_o = (close.index[is_n:][-1] - close.index[is_n:][0]).days
            if nd_o <= 0: nd_o = 1
            mon_o = ((1+ret_o)**(1/(nd_o/30.44))-1)*100 if ret_o > -1 else 0

            all_results.append({
                'indicator': ind_name, 'params': params,
                'sharpe': float(sh), 'sharpe_oos': float(sh_o),
                'monthly_pct': float(mon), 'monthly_oos': float(mon_o),
                'n_trades': int(tr), 'n_trades_oos': int(tr_o),
                'win_rate': float(w/tr) if tr>0 else 0,
                'max_dd': float(-dd),
            })

    all_results.sort(key=lambda r: r['sharpe_oos'], reverse=True)

    print(f"  {len(all_results)}/{total} strategies with trades\n")

    print(f"  TOP 20 by OOS Sharpe:")
    print(f"  {'Rank':>4s} {'Indicator':<15s} {'Sh IS':>6s} {'Sh OOS':>7s} {'Mon IS':>7s} {'Mon OOS':>7s} {'DD':>6s} {'Tr':>5s}")
    for i, r in enumerate(all_results[:20], 1):
        print(f"  {i:4d} {r['indicator']:<15s} {r['sharpe']:+6.1f} {r['sharpe_oos']:+7.1f} "
              f"{r['monthly_pct']:+6.1f}% {r['monthly_oos']:+6.1f}% "
              f"{r['max_dd']:.4f} {r['n_trades']:5d}")

    # Save
    ts = datetime.now().strftime("%Y%m%d_%H%M%S")
    out = OUTPUT_DIR / f"gold_swing_{ts}.json"
    out.write_text(json.dumps(all_results, indent=2, default=str))
    print(f"\n  Saved: {out}")

    # Indicator summary
    from collections import Counter
    print(f"\n  Indicator Performance:")
    for ind in INDICATOR_GRIDS.keys():
        r = [r for r in all_results if r['indicator'] == ind]
        if r:
            print(f"  {ind:<15s}: max Sh={max(x['sharpe'] for x in r):+.1f} "
                  f"OOS={max(x['sharpe_oos'] for x in r):+.1f} "
                  f"({len(r)} combos)")


if __name__ == "__main__":
    main()
