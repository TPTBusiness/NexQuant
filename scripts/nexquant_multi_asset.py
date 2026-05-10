#!/usr/bin/env python
"""
NexQuant Multi-Asset Data Pipeline — Download + Test on expanded universe.
Downloads DXY, Gold, S&P 500, Bund, EUR/USD extended history via yfinance.
"""

from __future__ import annotations

import json, sys, time
from pathlib import Path

import numpy as np
import pandas as pd
import yfinance as yf

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

DATA_DIR = Path("git_ignore_folder/factor_implementation_source_data")
DATA_DIR.mkdir(parents=True, exist_ok=True)

# Multi-asset tickers (free via Yahoo Finance)
ASSETS = {
    "EURUSD": "EURUSD=X",
    "DXY": "DX-Y.NYB",        # US Dollar Index
    "GOLD": "GC=F",           # Gold Futures
    "SPX": "^GSPC",           # S&P 500
    "BUND": "BUN24-EUX",      # German Bund (approximate)
    "GBPUSD": "GBPUSD=X",
    "USDJPY": "USDJPY=X",
    "OIL": "CL=F",            # Crude Oil
}

def download_asset(name: str, ticker: str, period: str = "max") -> pd.DataFrame:
    print(f"  Downloading {name} ({ticker})...")
    try:
        data = yf.download(ticker, period=period, progress=False, auto_adjust=True)
        if data.empty:
            print(f"    Empty — skipping")
            return None
        close = data["Close"]
        if isinstance(close, pd.DataFrame):
            close = close.iloc[:, 0]
        close.name = name
        print(f"    {len(close):,} bars ({close.index[0].date()} - {close.index[-1].date()})")
        return close
    except Exception as e:
        print(f"    Failed: {e}")
        return None

def main():
    print(f"\n{'='*60}")
    print("  NexQuant Multi-Asset Data Download")
    print(f"{'='*60}\n")

    all_data = {}
    for name, ticker in ASSETS.items():
        series = download_asset(name, ticker)
        if series is not None and len(series) > 100:
            all_data[name] = series

    if not all_data:
        print("No data downloaded!")
        return

    # Build combined DataFrame
    df = pd.DataFrame(all_data).dropna(how="all")
    print(f"\nCombined data: {len(df):,} daily bars, {len(df.columns)} assets")
    print(f"Date range: {df.index[0].date()} - {df.index[-1].date()}")

    # Save to HDF5
    h5_path = DATA_DIR / "multi_asset_daily.h5"
    df.to_hdf(h5_path, key="data", mode="w")
    print(f"Saved to {h5_path}")

    # Quick strategy test
    print(f"\n{'='*60}")
    print("  Quick Daily Strategy Test on Multi-Asset")
    print(f"{'='*60}")

    from rdagent.components.backtesting.vbt_backtest import backtest_signal_ftmo

    for asset in df.columns:
        c = df[asset].dropna()
        if len(c) < 500:
            continue

        # SMA 10/30
        f = c.rolling(10).mean()
        s = c.rolling(30).mean()
        sig = pd.Series(0.0, index=c.index)
        sig[f > s] = 1
        sig[f < s] = -1

        r = backtest_signal_ftmo(c, sig.fillna(0), txn_cost_bps=2.14, wf_rolling=True)
        oos = r.get("wf_oos_sharpe_mean") or r.get("oos_sharpe", -999)
        oos_m = r.get("oos_monthly_return_pct", 0) or 0
        status = "✅" if oos > 0 else "  "
        print(f"  {asset:<10} SMA10/30: OOS={oos:+8.2f}  Mon={oos_m:+6.2f}%  {status}")

    # Also test extended EUR/USD
    eurusd = df["EURUSD"].dropna()
    print(f"\n  Extended EUR/USD: {len(eurusd):,} bars")
    c = eurusd
    f = c.rolling(10).mean()
    s = c.rolling(30).mean()
    sig = pd.Series(0.0, index=c.index)
    sig[f > s] = 1
    sig[f < s] = -1
    r = backtest_signal_ftmo(c, sig.fillna(0), txn_cost_bps=2.14, wf_rolling=True)
    oos = r.get("wf_oos_sharpe_mean") or r.get("oos_sharpe", -999)
    print(f"  SMA10/30 extended: OOS={oos:+8.2f} Mon={r.get('oos_monthly_return_pct',0):+.2f}%")


if __name__ == "__main__":
    main()
