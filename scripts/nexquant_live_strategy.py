#!/usr/bin/env python
"""
NexQuant Live Strategy — Multi-mode trading signal generator.

Modes:
  - price: SMA10/30 crossover on 1h bars (proven +0.40%/month)
  - factors: London momentum factors (proven +3.29%/month, needs factor data)
  
For FTMO live trading. Reads 1-min bar from file, computes 1h signal.
"""

from __future__ import annotations

import json, sys
from datetime import datetime
from pathlib import Path

import numpy as np
import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

OHLCV_PATH = Path("git_ignore_folder/factor_implementation_source_data/intraday_pv.h5")
CONFIG_PATH = Path("results/strategies_live/live_config.json")


def load_config():
    with open(CONFIG_PATH) as f:
        return json.load(f)


def get_latest_close():
    """Get the most recent 1-min close price."""
    close = pd.read_hdf(OHLCV_PATH, key="data")["$close"]
    if isinstance(close.index, pd.MultiIndex):
        close = close.droplevel(-1)
    return close.sort_index().dropna()


class LiveSignal:
    def __init__(self, mode="price"):
        self.mode = mode
        self.close = get_latest_close()
        self.config = load_config()
        self.session = self.config["session_hours"]  # [7, 17]

    def get_signal(self) -> dict:
        """Compute current trading signal."""
        now = pd.Timestamp.now(tz="UTC").floor("1h")
        hour = now.hour
        is_session = self.session[0] <= hour < self.session[1]

        if not is_session:
            return {"signal": 0, "active": False, "reason": "Outside session", "timestamp": now}

        if self.mode == "price":
            return self._price_mode(now)
        else:
            return self._factor_mode(now)

    def _price_mode(self, now) -> dict:
        """SMA10/30 crossover on 1h bars."""
        c = self.close.resample("1h").last()
        if now not in c.index:
            c.loc[now] = c.iloc[-1]
        
        # Compute SMAs
        sma10 = c.rolling(10).mean()
        sma30 = c.rolling(30).mean()
        
        if len(sma10.dropna()) < 30:
            return {"signal": 0, "active": True, "reason": "Not enough bars", "timestamp": now}
        
        current_sma10 = sma10.iloc[-1]
        current_sma30 = sma30.iloc[-1]
        prev_sma10 = sma10.iloc[-2]
        prev_sma30 = sma30.iloc[-2]
        
        # Signal
        if current_sma10 > current_sma30:
            signal = 1
            reason = "SMA10 > SMA30 (uptrend)"
        elif current_sma10 < current_sma30:
            signal = -1
            reason = "SMA10 < SMA30 (downtrend)"
        else:
            signal = 0
            reason = "SMA10 == SMA30 (flat)"

        # Cross detection
        crossed = (prev_sma10 - prev_sma30) * (current_sma10 - current_sma30) < 0
        if crossed:
            reason += " ⚡ CROSSOVER!"

        return {
            "signal": signal,
            "active": True,
            "mode": "price",
            "sma10": round(float(current_sma10), 6),
            "sma30": round(float(current_sma30), 6),
            "crossed": crossed,
            "price": round(float(c.iloc[-1]), 6),
            "reason": reason,
            "timestamp": now,
        }

    def _factor_mode(self, now) -> dict:
        return {"signal": 0, "active": True, "mode": "factors",
                "reason": "Factor data not available for live trading", "timestamp": now}


def main():
    signal = LiveSignal(mode="price")
    result = signal.get_signal()
    print(json.dumps(result, indent=2, default=str))


if __name__ == "__main__":
    main()
