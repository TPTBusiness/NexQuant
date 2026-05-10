#!/usr/bin/env python
"""
NexQuant Daily Strategy Generator — systematisch, kein LLM.

Grid-search für SMA/EMA/RSI/MACD/Momentum/Mean-Reversion auf Tagesdaten.
Speichert Top-Strategien als JSON für den Live-Trading-Workflow.

Usage:
    python scripts/nexquant_daily_strategies.py
    python scripts/nexquant_daily_strategies.py --top 10 --cost 2.14
"""

from __future__ import annotations

import json, sys, time
from datetime import datetime
from pathlib import Path

import numpy as np
import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from rdagent.components.backtesting.vbt_backtest import backtest_signal_ftmo

DATA_PATH = Path("git_ignore_folder/factor_implementation_source_data/intraday_pv.h5")
OUT_DIR = Path("results/strategies_daily")
OUT_DIR.mkdir(parents=True, exist_ok=True)

TXN_COST_BPS = 2.14
MIN_TRADES_OOS = 5


def load_daily_data():
    close = pd.read_hdf(DATA_PATH, key="data")["$close"]
    if isinstance(close.index, pd.MultiIndex):
        close = close.droplevel(-1)
    return close.sort_index().dropna().resample("1D").last().dropna()


def backtest(signal: pd.Series, close: pd.Series) -> dict:
    if signal is None or len(signal) < 10:
        return {}
    sig = signal.fillna(0).replace([np.inf, -np.inf], 0)
    r = backtest_signal_ftmo(close, sig, txn_cost_bps=TXN_COST_BPS, wf_rolling=True)
    return {
        "is_sharpe": r.get("is_sharpe", None),
        "is_monthly_pct": r.get("is_monthly_return_pct", None),
        "is_trades": r.get("is_n_trades", 0),
        "oos_sharpe": r.get("oos_sharpe", None),
        "oos_monthly_pct": r.get("oos_monthly_return_pct", None),
        "oos_max_dd": r.get("oos_max_drawdown", None),
        "oos_win_rate": r.get("oos_win_rate", None),
        "oos_trades": r.get("oos_n_trades", 0),
        "wf_sharpe": r.get("wf_oos_sharpe_mean", None),
        "wf_monthly_pct": r.get("wf_oos_monthly_return_mean", None),
        "wf_consistency": r.get("wf_oos_consistency", None),
        "mc_pvalue": r.get("mc_pvalue", None),
        "full_metrics": r,
    }


def make_sma_signal(close, fast, slow):
    f = close.rolling(fast).mean()
    s = close.rolling(slow).mean()
    sig = pd.Series(0.0, index=close.index)
    sig[f > s] = 1
    sig[f < s] = -1
    return sig


def make_ema_signal(close, fast, slow):
    f = close.ewm(span=fast).mean()
    s = close.ewm(span=slow).mean()
    sig = pd.Series(0.0, index=close.index)
    sig[f > s] = 1
    sig[f < s] = -1
    return sig


def make_rsi_signal(close, period, oversold, overbought):
    delta = close.diff()
    gain = delta.clip(lower=0)
    loss = -delta.clip(upper=0)
    rsi = 100 - (100 / (1 + gain.rolling(period).mean() / (loss.rolling(period).mean() + 1e-8)))
    sig = pd.Series(0.0, index=close.index)
    sig[rsi < oversold] = 1
    sig[rsi > overbought] = -1
    return sig


def make_macd_signal(close, fast, slow, signal_period):
    ema_fast = close.ewm(span=fast).mean()
    ema_slow = close.ewm(span=slow).mean()
    macd = ema_fast - ema_slow
    sig_line = macd.ewm(span=signal_period).mean()
    sig = pd.Series(0.0, index=close.index)
    sig[macd > sig_line] = 1
    sig[macd < sig_line] = -1
    return sig


def make_momentum_signal(close, n):
    mom = close.pct_change(n)
    return pd.Series(np.sign(mom).fillna(0), index=close.index)


def make_meanrev_signal(close, n):
    ret = close.pct_change(n)
    return pd.Series(-np.sign(ret).fillna(0), index=close.index)


def make_bollinger_signal(close, period, std_dev):
    ma = close.rolling(period).mean()
    std = close.rolling(period).std()
    sig = pd.Series(0.0, index=close.index)
    sig[close < ma - std_dev * std] = 1
    sig[close > ma + std_dev * std] = -1
    return sig


def main(top_n=15, cost_bps=2.14):
    global TXN_COST_BPS
    TXN_COST_BPS = cost_bps

    print(f"\n{'='*60}")
    print(f"  NexQuant Daily Strategy Generator")
    print(f"  Cost: {cost_bps} bps | Saving top {top_n}")
    print(f"{'='*60}")

    close = load_daily_data()
    print(f"Data: {len(close):,} daily bars ({close.index[0].date()} - {close.index[-1].date()})\n")

    results = []

    # SMA Crossovers
    print("SMA crossovers...")
    for fast in [5, 10, 15, 20, 30]:
        for slow in [fast * 2, fast * 3, fast * 4, fast * 5]:
            if slow > 250: continue
            sig = make_sma_signal(close, fast, slow)
            bt = backtest(sig, close)
            if bt.get("oos_trades", 0) >= MIN_TRADES_OOS:
                score = bt.get("oos_sharpe") or -999
                results.append(("SMA", f"SMA{fast}/{slow}", fast, slow, score, bt))

    # EMA Crossovers
    print("EMA crossovers...")
    for fast in [5, 10, 15, 20, 30]:
        for slow in [fast * 2, fast * 3, fast * 4, fast * 5]:
            if slow > 250: continue
            sig = make_ema_signal(close, fast, slow)
            bt = backtest(sig, close)
            if bt.get("oos_trades", 0) >= MIN_TRADES_OOS:
                score = bt.get("oos_sharpe") or -999
                results.append(("EMA", f"EMA{fast}/{slow}", fast, slow, score, bt))

    # RSI
    print("RSI strategies...")
    for period in [7, 10, 14, 21]:
        for oversold, overbought in [(20, 80), (25, 75), (30, 70), (35, 65)]:
            sig = make_rsi_signal(close, period, oversold, overbought)
            bt = backtest(sig, close)
            if bt.get("oos_trades", 0) >= MIN_TRADES_OOS:
                score = bt.get("oos_sharpe") or -999
                results.append(("RSI", f"RSI{period}({oversold}/{overbought})", period, 0, score, bt))

    # MACD
    print("MACD...")
    for fast, slow, sig_p in [(8, 17, 9), (12, 26, 9), (5, 35, 5), (10, 20, 7)]:
        s = make_macd_signal(close, fast, slow, sig_p)
        bt = backtest(s, close)
        if bt.get("oos_trades", 0) >= MIN_TRADES_OOS:
            score = bt.get("oos_sharpe") or -999
            results.append(("MACD", f"MACD{fast}/{slow}/{sig_p}", fast, slow, score, bt))

    # Momentum
    print("Momentum...")
    for n in [5, 10, 20, 30, 50, 60, 90, 100, 120, 150, 200]:
        sig = make_momentum_signal(close, n)
        bt = backtest(sig, close)
        if bt.get("oos_trades", 0) >= MIN_TRADES_OOS:
            score = bt.get("oos_sharpe") or -999
            results.append(("Mom", f"Mom{n}d", n, 0, score, bt))

    # Mean Reversion
    print("Mean reversion...")
    for n in [3, 5, 7, 10, 15, 20, 30, 50]:
        sig = make_meanrev_signal(close, n)
        bt = backtest(sig, close)
        if bt.get("oos_trades", 0) >= MIN_TRADES_OOS:
            score = bt.get("oos_sharpe") or -999
            results.append(("MR", f"MR{n}d", n, 0, score, bt))

    # Bollinger Bands
    print("Bollinger...")
    for period in [10, 20, 50]:
        for std_dev in [1.5, 2.0, 2.5]:
            sig = make_bollinger_signal(close, period, std_dev)
            bt = backtest(sig, close)
            if bt.get("oos_trades", 0) >= MIN_TRADES_OOS:
                score = bt.get("oos_sharpe") or -999
                results.append(("BB", f"BB{period}/{std_dev}", period, std_dev, score, bt))

    # Sort by OOS Sharpe
    results.sort(key=lambda x: x[4] if x[4] is not None else -999, reverse=True)

    print(f"\n{'='*70}")
    print(f"  TOP {top_n} DAILY STRATEGIES (Cost: {cost_bps} bps)")
    print(f"{'='*70}")
    print(f"  {'#':<3} {'Type':<6} {'Name':<22} {'OOS S':>8} {'Mon%':>7} {'DD%':>6} {'WF S':>8} {'Trades':>6}")
    print(f"  {'-'*68}")

    saved = []
    for i, (stype, name, p1, p2, score, bt) in enumerate(results[:top_n]):
        oos_m = (bt.get("oos_monthly_pct") or 0)
        oos_dd = (bt.get("oos_max_dd") or 0) * 100
        wf_s = bt.get("wf_sharpe") or 0
        trades = bt.get("oos_trades", 0)
        status = "✅" if score > 0 else "  "
        print(f"  {i+1:<3} {stype:<6} {name:<22} {score:>+8.2f} {oos_m:>+6.2f}% {oos_dd:>+5.1f}% {wf_s:>+8.2f} {trades:>6}  {status}")

        entry = {
            "strategy_name": name,
            "type": stype,
            "param1": p1,
            "param2": p2,
            "cost_bps": cost_bps,
            "frequency": "daily",
            "generated_at": datetime.now().isoformat(),
            "metrics": {k: v for k, v in bt.items() if k != "full_metrics"},
        }
        saved.append(entry)

        # Save individual strategy
        safe_name = name.replace("(", "").replace(")", "").replace("/", "-")
        fname = OUT_DIR / f"daily_{safe_name}_{datetime.now().strftime('%Y%m%d_%H%M%S')}.json"
        with open(fname, "w") as f:
            json.dump(entry, f, indent=2)

    # Save summary
    summary = {
        "generated_at": datetime.now().isoformat(),
        "cost_bps": cost_bps,
        "frequency": "daily",
        "n_bars": len(close),
        "date_range": [str(close.index[0].date()), str(close.index[-1].date())],
        "top_strategies": [
            {"name": s["strategy_name"], "oos_sharpe": s["metrics"].get("oos_sharpe"),
             "oos_monthly_pct": s["metrics"].get("oos_monthly_pct")}
            for s in saved[:10]
        ],
    }
    with open(OUT_DIR / "daily_summary.json", "w") as f:
        json.dump(summary, f, indent=2)

    profit_count = sum(1 for r in results if r[4] and r[4] > 0)
    print(f"\n{profit_count}/{len(results)} strategies profitable ({profit_count/len(results)*100:.0f}%)")
    print(f"Saved to {OUT_DIR}/")
    return saved


if __name__ == "__main__":
    import argparse
    parser = argparse.ArgumentParser()
    parser.add_argument("--top", type=int, default=15)
    parser.add_argument("--cost", type=float, default=2.14)
    args = parser.parse_args()
    main(top_n=args.top, cost_bps=args.cost)
