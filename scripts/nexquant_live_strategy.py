#!/usr/bin/env python
"""
NexQuant Live Strategy — 1h London Session Momentum.

Generates real-time trading signals for FTMO live trading.
Reads current 1h bar, computes factor value, outputs signal (LONG/SHORT/FLAT).
"""

from __future__ import annotations

import json, sys
from pathlib import Path
from typing import Optional

import numpy as np
import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))


class LiveStrategy:
    """1h London Session Momentum — pull latest bar, compute signal."""

    def __init__(self):
        self.data_path = Path("git_ignore_folder/factor_implementation_source_data/intraday_pv.h5")
        self.factors_dir = Path("results/factors")
        self.values_dir = self.factors_dir / "values"
        self.factors = {
            "london_session_momentum": 1,
            "london_session_drift": 1,
        }
        self._factor_cache = {}

    def _load_factor(self, name: str) -> pd.Series:
        if name in self._factor_cache:
            return self._factor_cache[name]
        safe = name.replace("/", "_")[:150]
        pf = self.values_dir / f"{safe}.parquet"
        s = pd.read_parquet(pf).iloc[:, 0]
        if isinstance(s.index, pd.MultiIndex):
            s = s.droplevel(-1)
        self._factor_cache[name] = s.sort_index()
        return self._factor_cache[name]

    def get_signal(self, current_time: pd.Timestamp = None) -> dict:
        """
        Compute trading signal for the current 1h bar.
        
        Returns dict with:
            signal: 1 (long), -1 (short), 0 (flat)
            strength: 0.0-1.0 (confidence)
            factors: dict of individual factor signals
            active: bool (is London/NY session?)
            timestamp: current bar time
        """
        if current_time is None:
            current_time = pd.Timestamp.now(tz="UTC").floor("1h")

        hour = current_time.hour
        is_session = 7 <= hour < 17

        if not is_session:
            return {
                "signal": 0, "strength": 0.0,
                "factors": {}, "active": False,
                "timestamp": current_time,
                "reason": "Outside trading session (07-17 UTC)"
            }

        signals = {}
        for name, direction in self.factors.items():
            try:
                series = self._load_factor(name)
                fac_1h = series.resample("1h").last()
                if current_time in fac_1h.index:
                    val = fac_1h.loc[current_time]
                else:
                    val = fac_1h.asof(current_time)
                if pd.isna(val):
                    signals[name] = 0
                else:
                    signals[name] = direction * int(np.sign(val))
            except Exception:
                signals[name] = 0

        # Combine: average of individual signals
        values = list(signals.values())
        combo = np.mean(values) if values else 0
        
        # Round to nearest direction
        if combo > 0.3:
            signal = 1
        elif combo < -0.3:
            signal = -1
        else:
            signal = 0

        strength = abs(combo)
        agreeing = sum(1 for v in values if v == signal)

        return {
            "signal": signal,
            "strength": round(strength, 3),
            "factors": signals,
            "active": True,
            "timestamp": current_time,
            "agreeing_factors": f"{agreeing}/{len(values)}",
            "reason": f"{'LONG' if signal == 1 else 'SHORT' if signal == -1 else 'FLAT'} ({agreeing}/{len(values)} factors agree)"
        }


def main():
    strat = LiveStrategy()
    result = strat.get_signal()
    print(json.dumps(result, indent=2, default=str))


if __name__ == "__main__":
    main()
