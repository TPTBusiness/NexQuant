#!/usr/bin/env python3
"""Price-Action Strategy Generator — no LLM, no factors, pure technical analysis.

Uses Donchian channels, moving averages, RSI, Bollinger Bands, and MACD
on daily resolution. Grid-searches parameters, validates via backtest_signal.
"""

import json
import os
import time
from datetime import datetime
from pathlib import Path

import numpy as np
import pandas as pd

PROJECT = Path(__file__).resolve().parent.parent
OHLCV_PATH = Path(os.getenv("PREDIX_OHLCV_PATH",
    str(PROJECT / "git_ignore_folder" / "intraday_pv_all.h5")))
RESULTS_DIR = PROJECT / "results" / "strategies_new"

MIN_MONTHLY = 1.0
MIN_SHARPE  = 1.0
MAX_DD      = -0.15
MIN_TRADES  = 30


def load_data():
    df = pd.read_hdf(OHLCV_PATH, key="data")
    close = df.xs("EURUSD", level="instrument")["$close"].sort_index()
    daily = close.resample("D").last().dropna()
    return close, daily


def to_1min(daily_signal: pd.Series, close_1min: pd.Series) -> pd.Series:
    return daily_signal.reindex(close_1min.index).ffill().fillna(0).astype(int).clip(-1, 1)


# ═══════════════════════════════════════════════════════════════════════════════
# Strategy templates
# ═══════════════════════════════════════════════════════════════════════════════

def donchian(close: pd.Series, period: int, hold: int) -> pd.Series:
    """Donchian channel breakout."""
    high = close.rolling(period).max()
    low = close.rolling(period).min()
    s = pd.Series(0, index=close.index)
    s[close > high.shift(1)] = 1
    s[close < low.shift(1)] = -1
    s = s.replace(0, np.nan).ffill(limit=hold).fillna(0).astype(int).clip(-1, 1)
    return s


def sma_cross(close: pd.Series, fast: int, slow: int) -> pd.Series:
    """SMA crossover."""
    s = pd.Series(0, index=close.index)
    s[close.rolling(fast).mean() > close.rolling(slow).mean()] = 1
    s[close.rolling(fast).mean() < close.rolling(slow).mean()] = -1
    return s.fillna(0).astype(int).clip(-1, 1)


def rsi_mr(close: pd.Series, period: int, oversold: int, overbought: int) -> pd.Series:
    """RSI mean-reversion."""
    delta = close.diff()
    gain = delta.clip(lower=0).rolling(period).mean()
    loss = (-delta.clip(upper=0)).rolling(period).mean()
    rs = gain / (loss + 1e-8)
    rsi = 100 - 100 / (1 + rs)
    s = pd.Series(0, index=close.index)
    s[rsi < oversold] = 1
    s[rsi > overbought] = -1
    return s.fillna(0).astype(int).clip(-1, 1)


def bollinger_mr(close: pd.Series, period: int, std: float) -> pd.Series:
    """Bollinger Band mean-reversion."""
    ma = close.rolling(period).mean()
    st = close.rolling(period).std()
    s = pd.Series(0, index=close.index)
    s[close < ma - std * st] = 1
    s[close > ma + std * st] = -1
    return s.fillna(0).astype(int).clip(-1, 1)


def macd(close: pd.Series, fast: int, slow: int, signal_p: int) -> pd.Series:
    """MACD crossover."""
    ema_fast = close.ewm(span=fast, adjust=False).mean()
    ema_slow = close.ewm(span=slow, adjust=False).mean()
    macd_line = ema_fast - ema_slow
    sig_line = macd_line.ewm(span=signal_p, adjust=False).mean()
    s = pd.Series(0, index=close.index)
    s[macd_line > sig_line] = 1
    s[macd_line < sig_line] = -1
    return s.fillna(0).astype(int).clip(-1, 1)


def ma_envelope(close: pd.Series, period: int, pct: float) -> pd.Series:
    """Moving average envelope mean-reversion."""
    ma = close.rolling(period).mean()
    s = pd.Series(0, index=close.index)
    s[close < ma * (1 - pct)] = 1
    s[close > ma * (1 + pct)] = -1
    return s.replace(0, np.nan).ffill(limit=3).fillna(0).astype(int).clip(-1, 1)


def atr_breakout(close: pd.Series, period: int, mult: float) -> pd.Series:
    """ATR-based volatility breakout (simplified, using close-only)."""
    atr = (close.diff().abs()).rolling(period).mean()
    ma = close.rolling(period).mean()
    s = pd.Series(0, index=close.index)
    s[close > ma + mult * atr] = 1
    s[close < ma - mult * atr] = -1
    return s.replace(0, np.nan).ffill(limit=2).fillna(0).astype(int).clip(-1, 1)


# ═══════════════════════════════════════════════════════════════════════════════
# Main
# ═══════════════════════════════════════════════════════════════════════════════

def main():
    print("=" * 60)
    print("  Price-Action Strategy Generator (No LLM, No Factors)")
    print("=" * 60)

    from rdagent.components.backtesting.vbt_backtest import backtest_signal

    close, daily = load_data()
    print(f"\nDaily data: {len(daily)} bars ({daily.index[0].date()} → {daily.index[-1].date()})")

    import itertools

    grid = [
        ("Donchian", donchian, [
            (p, h) for p in [5, 7, 10, 12, 15, 20, 25, 30, 40, 60]
            for h in [1, 2, 3, 5]
        ]),
        ("SMA_Crossover", sma_cross, [
            (f, s) for f in [5, 10, 20]
            for s in [20, 50, 100, 200] if s > f
        ]),
        ("RSI_MR", rsi_mr, [
            (p, lo, hi) for p in [7, 14, 21]
            for lo, hi in [(30, 70), (25, 75), (20, 80)]
        ]),
        ("Bollinger_MR", bollinger_mr, [
            (p, s) for p in [10, 20, 40]
            for s in [1.5, 2.0, 2.5]
        ]),
        ("MACD", macd, [
            (f, s, sig) for f, s, sig in [(8, 21, 5), (12, 26, 9), (5, 20, 3)]
        ]),
        ("MA_Envelope", ma_envelope, [
            (p, pct) for p in [20, 50, 100]
            for pct in [0.01, 0.02, 0.03]
        ]),
        ("ATR_Breakout", atr_breakout, [
            (p, m) for p in [10, 20, 40]
            for m in [1.0, 1.5, 2.0]
        ]),
    ]

    results = []
    t0 = time.time()
    total = sum(len(params) for _, _, params in grid)
    done = 0

    print(f"\nTesting {total} parameter combinations...\n")

    for name, fn, params_list in grid:
        for params in params_list:
            done += 1
            daily_signal = fn(daily, *params)
            signal_1min = to_1min(daily_signal, close)
            bt = backtest_signal(close=close, signal=signal_1min)
            bt["strategy"] = name
            bt["params"] = params
            bt["name"] = f"{name}{params}"
            bt["monthly_pct"] = bt.get("monthly_return_pct", 0)
            bt["max_dd"] = bt.get("max_drawdown", 0)
            results.append(bt)
            if done % 50 == 0 or done == total:
                elapsed = time.time() - t0
                rate = done / elapsed if elapsed > 0 else 0
                eta = (total - done) / rate if rate > 0 else 0
                print(f"  {done}/{total} ({done/total*100:.0f}%)  {rate:.0f}/s  eta {eta:.0f}s")

    elapsed = time.time() - t0
    print(f"\n{'=' * 60}")
    print(f"  Evaluated: {total} in {elapsed:.0f}s")
    print(f"{'=' * 60}")

    valid = [r for r in results
             if r.get("sharpe", 0) >= MIN_SHARPE
             and r.get("max_dd", 0) >= MAX_DD
             and r.get("n_trades", 0) >= MIN_TRADES
             and r.get("monthly_pct", 0) >= MIN_MONTHLY]
    valid.sort(key=lambda r: r.get("monthly_pct", 0), reverse=True)

    print(f"\n  Meeting: Sharpe≥{MIN_SHARPE} DD≥{MAX_DD} Tr≥{MIN_TRADES} Mon≥{MIN_MONTHLY}%")
    print(f"  → {len(valid)} strategies\n")

    hdr = "{:>3s} {:20s} {:20s} {:>7s} {:>7s} {:>7s} {:>5s} {:>6s}"
    print(hdr.format("#", "Strategy", "Params", "Sharpe", "Mon%", "MaxDD", "Tr", "WinRt"))
    print("-" * 85)
    for i, r in enumerate(valid[:30], 1):
        ps = str(r["params"]).replace(" ", "")[:18]
        print(hdr.format(str(i), r["strategy"][:20], ps,
              f'{r.get("sharpe",0):.2f}', f'{r.get("monthly_pct",0):.1f}%',
              f'{r.get("max_dd",0):.3f}', str(r.get("n_trades",0)),
              f'{r.get("win_rate",0):.1%}'))

    print(f"\n  Best by category:")
    seen = set()
    for r in valid:
        if r["strategy"] not in seen:
            seen.add(r["strategy"])
            print(f"    {r['strategy']:20s} {r['name'][:30]:30s} "
                  f"Sh={r.get('sharpe',0):.2f} Mon={r.get('monthly_pct',0):.1f}% "
                  f"DD={r.get('max_dd',0):.3f} Tr={r.get('n_trades',0)}")

    RESULTS_DIR.mkdir(parents=True, exist_ok=True)
    out = RESULTS_DIR / f"priceaction_{datetime.now().strftime('%Y%m%d_%H%M%S')}.json"
    out.write_text(json.dumps(valid[:100] if valid else results[:100], indent=2, default=str))
    print(f"\n  Saved → {out}")


if __name__ == "__main__":
    main()
