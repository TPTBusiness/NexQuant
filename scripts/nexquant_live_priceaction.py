#!/usr/bin/env python3
"""Live Price-Action Strategy Pipeline — No LLM, No Factors.

Generates daily signals from Donchian + MACD portfolio, executes via risk
backtest, and optionally sends signals to live trading.

Usage:
    python scripts/nexquant_live_priceaction.py           # Generate today's signal
    python scripts/nexquant_live_priceaction.py --daemon  # Run continuously
    python scripts/nexquant_live_priceaction.py --backfill  # Full historical backtest
"""

import json
import os
import sys
import time
from datetime import datetime, timedelta
from pathlib import Path

import numpy as np
import pandas as pd

PROJECT = Path(__file__).resolve().parent.parent
OHLCV_PATH = Path(os.getenv("PREDIX_OHLCV_PATH",
    str(PROJECT / "git_ignore_folder" / "intraday_pv_all.h5")))
SIGNAL_PATH = PROJECT / "git_ignore_folder" / "priceaction_signal.json"
RESULTS_DIR = PROJECT / "results" / "reports"

# Portfolio config
STRATEGIES = [
    {"name": "Donchian(30,1)", "type": "donchian", "period": 30, "hold": 1},
    {"name": "MACD(3,15,3)", "type": "macd", "fast": 3, "slow": 15, "signal_period": 3},
]

VOTE_THRESHOLD = 0.25


def load_close() -> tuple[pd.Series, pd.Series]:
    """Load 1-min and daily close prices."""
    df = pd.read_hdf(OHLCV_PATH, key="data")
    close = df.xs("EURUSD", level="instrument")["$close"].sort_index()
    daily = close.resample("D").last().dropna()
    return close, daily


def donchian_signal(daily: pd.Series, period: int, hold: int) -> pd.Series:
    """Donchian channel breakout signal (daily)."""
    high = daily.rolling(period).max()
    low = daily.rolling(period).min()
    s = pd.Series(0, index=daily.index)
    s[daily > high.shift(1)] = 1
    s[daily < low.shift(1)] = -1
    return s.replace(0, np.nan).ffill(limit=hold).fillna(0).astype(int).clip(-1, 1)


def macd_signal(daily: pd.Series, fast: int, slow: int, signal_period: int) -> pd.Series:
    """MACD crossover signal (daily)."""
    ema_fast = daily.ewm(span=fast, adjust=False).mean()
    ema_slow = daily.ewm(span=slow, adjust=False).mean()
    macd_line = ema_fast - ema_slow
    sig_line = macd_line.ewm(span=signal_period, adjust=False).mean()
    s = pd.Series(0, index=daily.index)
    s[macd_line > sig_line] = 1
    s[macd_line < sig_line] = -1
    return s.fillna(0).astype(int).clip(-1, 1)


def compute_portfolio_signal(daily: pd.Series) -> pd.Series:
    """Compute majority-vote portfolio signal."""
    signals = []
    for cfg in STRATEGIES:
        if cfg["type"] == "donchian":
            sig = donchian_signal(daily, cfg["period"], cfg["hold"])
        elif cfg["type"] == "macd":
            sig = macd_signal(daily, cfg["fast"], cfg["slow"], cfg["signal_period"])
        else:
            continue
        signals.append(sig)

    if not signals:
        return pd.Series(0, index=daily.index)

    port = pd.DataFrame({f"s{i}": s for i, s in enumerate(signals)}).dropna()
    vote = port.mean(axis=1)
    result = pd.Series(0, index=vote.index)
    result[vote > VOTE_THRESHOLD] = 1
    result[vote < -VOTE_THRESHOLD] = -1
    result.name = "signal"
    return result


def get_todays_signal() -> dict:
    """Generate today's trading signal."""
    close, daily = load_close()
    portfolio_signal = compute_portfolio_signal(daily)

    # Latest signal
    latest = portfolio_signal.iloc[-1]
    direction = {1: "LONG", -1: "SHORT", 0: "NEUTRAL"}[int(latest)]

    # Last signal change
    changes = portfolio_signal.diff().abs()
    last_change_idx = changes[changes > 0].index[-1] if (changes > 0).any() else None
    days_in_position = (daily.index[-1] - last_change_idx).days if last_change_idx is not None else 0

    result = {
        "timestamp": datetime.now().isoformat(),
        "date": str(daily.index[-1].date()),
        "signal": int(latest),
        "direction": direction,
        "days_in_position": days_in_position,
        "strategies": {cfg["name"]: int(
            donchian_signal(daily, cfg["period"], cfg["hold"]).iloc[-1] if cfg["type"] == "donchian"
            else macd_signal(daily, cfg["fast"], cfg["slow"], cfg["signal_period"]).iloc[-1]
        ) for cfg in STRATEGIES},
    }

    SIGNAL_PATH.parent.mkdir(parents=True, exist_ok=True)
    SIGNAL_PATH.write_text(json.dumps(result, indent=2))

    return result


def run_backfill():
    """Run full historical backtest and save report."""
    print("Running full historical backtest...")
    close, daily = load_close()
    signal = compute_portfolio_signal(daily)

    # ffill to 1-min
    sig_1min = signal.reindex(close.index).ffill().fillna(0).astype(int).clip(-1, 1)

    from rdagent.components.backtesting.vbt_backtest import backtest_signal, backtest_signal_risk

    bt = backtest_signal(close=close, signal=sig_1min)
    bt_risk = backtest_signal_risk(close=close, signal=sig_1min, risk_pct=0.0035, oos_start=None, wf_rolling=True)

    report = {
        "strategy": "Donchian(30,1) + MACD(3,15,3) Majority-Vote",
        "timestamp": datetime.now().isoformat(),
        "backtest": {
            "sharpe": round(bt["sharpe"], 2),
            "monthly_return_pct": round(bt["monthly_return_pct"], 2),
            "max_drawdown": round(bt["max_drawdown"], 4),
            "n_trades": bt["n_trades"],
            "win_rate": round(bt["win_rate"], 4),
        },
        "risk_backtest": {
            "sharpe": round(bt_risk.get("sharpe", 0), 2),
            "monthly_pct": round(bt_risk.get("monthly_return_pct", 0), 2),
            "max_dd": round(bt_risk.get("max_drawdown", 0), 4),
            "wf_consistency": round(bt_risk.get("wf_oos_consistency", 0), 4),
        },
    }

    RESULTS_DIR.mkdir(parents=True, exist_ok=True)
    path = RESULTS_DIR / f"backfill_{datetime.now().strftime('%Y%m%d_%H%M%S')}.json"
    path.write_text(json.dumps(report, indent=2))

    print(f"\n{'='*50}")
    print(f"  Sharpe: {bt['sharpe']:.2f}")
    print(f"  Monthly: {bt['monthly_return_pct']:.2f}%")
    print(f"  Max DD: {bt['max_drawdown']:.4f}")
    print(f"  Trades: {bt['n_trades']}")
    print(f"  Win Rate: {bt['win_rate']:.1%}")
    print(f"  Report saved: {path}")
    print(f"{'='*50}")


def main():
    if "--backfill" in sys.argv:
        run_backfill()
    elif "--daemon" in sys.argv:
        print("Daemon mode — generating signals every 5 minutes...")
        while True:
            result = get_todays_signal()
            print(f"  [{result['timestamp']}] {result['direction']:>8s} ({result['days_in_position']}d in position)")
            time.sleep(300)
    else:
        result = get_todays_signal()
        print(json.dumps(result, indent=2))


if __name__ == "__main__":
    main()
