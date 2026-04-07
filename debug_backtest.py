#!/usr/bin/env python
"""Debug backtest logic: check alignment, signal quality, and IC calculation."""
import json
import numpy as np
import pandas as pd
from pathlib import Path

OHLCV_PATH = Path('/home/nico/Predix/git_ignore_folder/factor_implementation_source_data/intraday_pv.h5')
FACTORS_DIR = Path('/home/nico/Predix/results/factors')
VALUES_DIR = FACTORS_DIR / 'values'

print("=" * 70)
print("🔍 BACKTEST DEBUG - Alignment & IC Check")
print("=" * 70)

# 1. Load OHLCV close prices
print("\n1️⃣ Loading OHLCV close prices...")
ohlcv = pd.read_hdf(str(OHLCV_PATH), key='data')
if '$close' in ohlcv.columns:
    close = ohlcv['$close']
elif 'close' in ohlcv.columns:
    close = ohlcv['close']
else:
    close = ohlcv.select_dtypes(include=[np.number]).iloc[:, 0]

close = close.dropna()
print(f"   Close prices: {len(close):,} bars")
print(f"   Date range: {close.index.min()} → {close.index.max()}")
print(f"   Sample: {close.head(3).values}")

# Calculate forward returns (what we predict)
returns = close.pct_change().dropna()
print(f"   Returns: {len(returns):,} values")
print(f"   Return stats: mean={returns.mean():.8f}, std={returns.std():.8f}")

# 2. Load factor parquet files
print("\n2️⃣ Loading factor time-series...")
factor_files = sorted(VALUES_DIR.glob('*.parquet'))[:5]  # Top 5
for ff in factor_files:
    df_factor = pd.read_parquet(str(ff))
    if len(df_factor.columns) > 0:
        col = df_factor.iloc[:, 0]
        # Align with close
        common = close.index.intersection(col.dropna().index)
        if len(common) > 0:
            c = close.loc[common]
            f = col.loc[common]
            r = returns.loc[common] if len(returns) > 0 else pd.Series()
            
            # Simple test: factor value vs next return
            # IC = correlation(factor, forward_return)
            fwd_r = r.shift(-1)  # Next bar return
            common2 = f.dropna().index.intersection(fwd_r.dropna().index)
            
            if len(common2) > 100:
                ic = f.loc[common2].corr(fwd_r.loc[common2])
                print(f"   {ff.stem:40s} len={len(f):>8,} IC_vs_next_return={ic:.6f}")
            else:
                print(f"   {ff.stem:40s} len={len(f):>8,} (not enough common data)")

# 3. Test simple signals
print("\n3️⃣ Testing SIMPLE signals (what should work)...")

# Load metadata for best factors
factors_meta = []
for f in FACTORS_DIR.glob('*.json'):
    try:
        d = json.load(open(f))
        ic = d.get('ic', 0) or 0
        fname = d.get('factor_name', '')
        if abs(ic) > 0.1:
            factors_meta.append({'name': fname, 'ic': ic})
    except:
        pass

factors_meta.sort(key=lambda x: abs(x['ic']), reverse=True)
top = factors_meta[:3]
print(f"   Top factors: {[(f['name'], f['ic']) for f in top]}")

# For each top factor, test simple signal
for fm in top:
    fname = fm['name']
    safe = fname.replace('/', '_').replace('\\', '_')[:150]
    pf = VALUES_DIR / f"{safe}.parquet"
    if not pf.exists():
        print(f"   ❌ {fname}: parquet not found")
        continue
    
    factor_series = pd.read_parquet(str(pf)).iloc[:, 0]
    factor_ic = fm['ic']
    
    # Align
    common = close.index.intersection(factor_series.dropna().index)
    if len(common) < 1000:
        print(f"   ❌ {fname}: not enough common data ({len(common)})")
        continue
    
    f = factor_series.loc[common]
    r = returns.loc[common]
    
    # Test A: Raw factor value vs next return
    fwd = r.shift(-1)
    common2 = f.index.intersection(fwd.dropna().index)
    ic_raw = f.loc[common2].corr(fwd.loc[common2])
    
    # Test B: Factor sign as signal (positive → LONG)
    signal_a = (f > 0).astype(int).replace(0, -1)  # +1 or -1
    strat_ret_a = signal_a.shift(1) * r
    sharpe_a = strat_ret_a.mean() / strat_ret_a.std() * np.sqrt(252*1440/96) if strat_ret_a.std() > 0 else 0
    ic_signal = signal_a.corr(fwd)
    
    # Test C: Factor percentile as signal
    pct = f.rank(pct=True)
    signal_c = (pct > 0.6).astype(int).replace(0, -1)
    strat_ret_c = signal_c.shift(1) * r
    sharpe_c = strat_ret_c.mean() / strat_ret_c.std() * np.sqrt(252*1440/96) if strat_ret_c.std() > 0 else 0
    
    # Test D: What the LLM strategies actually compute (factor z-score → threshold)
    w = 60
    z = (f - f.rolling(w).mean()) / f.rolling(w).std()
    z = z.fillna(0)
    signal_d = (z > 0.5).astype(int).replace(0, -1)
    strat_ret_d = signal_d.shift(1) * r
    sharpe_d = strat_ret_d.mean() / strat_ret_d.std() * np.sqrt(252*1440/96) if strat_ret_d.std() > 0 else 0
    
    print(f"\n   📊 {fname} (factor IC={factor_ic:.4f}):")
    print(f"      Raw factor IC vs fwd return: {ic_raw:.6f}")
    print(f"      Signal (sign) IC={ic_signal:.6f}  Sharpe={sharpe_a:.4f}")
    print(f"      Signal (percentile) Sharpe={sharpe_c:.4f}")
    print(f"      Signal (z-score thresh) Sharpe={sharpe_d:.4f}")

# 4. Key finding
print("\n" + "=" * 70)
print("4️⃣ KEY INSIGHT:")
print("=" * 70)

# Check if factor values are already aligned with returns or are predictions
# Load one factor and check timing
if top:
    fname = top[0]['name']
    safe = fname.replace('/', '_').replace('\\', '_')[:150]
    pf = VALUES_DIR / f"{safe}.parquet"
    f = pd.read_parquet(str(pf)).iloc[:, 0]
    
    # What does a HIGH factor value mean?
    # If factor IC is positive (0.25), high values should predict positive returns
    # Let's check: when factor is in top 10%, what's the average NEXT return?
    common = close.index.intersection(f.dropna().index)
    fv = f.loc[common]
    rv = returns.loc[common]
    fwd = rv.shift(-1)
    
    common2 = fv.index.intersection(fwd.dropna().index)
    fv2 = fv.loc[common2]
    fwd2 = fwd.loc[common2]
    
    top_decile = fv2 > fv2.quantile(0.9)
    bot_decile = fv2 < fv2.quantile(0.1)
    
    avg_ret_when_high = fwd2[top_decile].mean()
    avg_ret_when_low = fwd2[bot_decile].mean()
    
    print(f"\n   Factor: {fname} (IC={top[0]['ic']:.4f})")
    print(f"   Avg NEXT return when factor in TOP 10%: {avg_ret_when_high*100:.6f}%")
    print(f"   Avg NEXT return when factor in BOT 10%: {avg_ret_when_low*100:.6f}%")
    print(f"   Difference: {(avg_ret_when_high - avg_ret_when_low)*100:.6f}%")
    print(f"\n   → If difference > 0, factor has predictive power")
    print(f"   → If difference ≈ 0, factor has NO predictive power for next-bar returns")
    print(f"   → If difference < 0, factor is INVERTED (use negative)")
