#!/usr/bin/env python
"""
NexQuant Optuna ML Strategy Pipeline — automatic hyperparameter tuning for daily strategies.
Target: 8%/month through ML-driven signal generation on daily OHLCV data.
"""

from __future__ import annotations

import json, sys, time, warnings
from datetime import datetime
from pathlib import Path
from typing import Optional

import numpy as np
import pandas as pd

warnings.filterwarnings("ignore")
sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

import optuna
from sklearn.ensemble import RandomForestClassifier

from rdagent.components.backtesting.vbt_backtest import backtest_signal_ftmo

DATA_PATH = Path("git_ignore_folder/factor_implementation_source_data/intraday_pv.h5")
TXN_COST_BPS = 2.14
N_TRIALS = 50


def load_data():
    c = pd.read_hdf(DATA_PATH, key="data")["$close"]
    if isinstance(c.index, pd.MultiIndex):
        c = c.droplevel(-1)
    return c.sort_index().dropna().resample("1D").last().dropna()


def make_features(c: pd.Series) -> pd.DataFrame:
    df = pd.DataFrame(index=c.index)
    # Returns
    for n in [1, 3, 5, 10, 20]:
        df[f"r{n}"] = c.pct_change(n)
        df[f"lr{n}"] = np.log(c / c.shift(n))
    # MAs
    for n in [5, 10, 20, 50, 100, 200]:
        df[f"sma{n}"] = c.rolling(n).mean() / c - 1
        df[f"ema{n}"] = c.ewm(span=n).mean() / c - 1
    # Crossovers
    df["sma5_20"] = c.rolling(5).mean() / c.rolling(20).mean() - 1
    df["sma10_50"] = c.rolling(10).mean() / c.rolling(50).mean() - 1
    df["sma20_100"] = c.rolling(20).mean() / c.rolling(100).mean() - 1
    df["ema12_26"] = c.ewm(span=12).mean() / c.ewm(span=26).mean() - 1
    # Volatility
    for n in [5, 10, 20, 50]:
        df[f"vol{n}"] = c.pct_change().rolling(n).std()
    df["vol_ratio"] = df["vol10"] / (df["vol20"] + 1e-8)
    # RSI
    for p in [7, 14, 21]:
        d = c.diff(); g = d.clip(lower=0); l = -d.clip(upper=0)
        df[f"rsi{p}"] = 100 - (100 / (1 + g.rolling(p).mean() / (l.rolling(p).mean() + 1e-8)))
    # MACD
    e12 = c.ewm(span=12).mean(); e26 = c.ewm(span=26).mean()
    macd = e12 - e26; sig = macd.ewm(span=9).mean()
    df["macd"] = macd / c; df["macd_hist"] = (macd - sig) / c
    # Bollinger
    for p, k in [(20, 2)]:
        ma = c.rolling(p).mean(); sd = c.rolling(p).std()
        df[f"bb"] = (c - ma) / (k * sd + 1e-8)
        df[f"bbw"] = (2 * k * sd) / (ma + 1e-8)
    # ATR
    tr = pd.concat([c.diff().abs(), (c-c.shift(1)).abs(), (c.shift(1)-c).abs()], axis=1).max(axis=1)
    df["atr14"] = tr.rolling(14).mean() / c
    # ADX
    a14 = tr.rolling(14).mean()
    pdi = 100 * (c.diff().clip(lower=0).ewm(span=14).mean() / (a14 + 1e-8))
    mdi = 100 * (-c.diff().clip(upper=0).ewm(span=14).mean() / (a14 + 1e-8))
    df["adx"] = (100 * abs(pdi - mdi) / (pdi + mdi + 1e-8)).ewm(span=14).mean()
    # Donchian
    for n in [20, 50]:
        hi = c.rolling(n).max(); lo = c.rolling(n).min()
        df[f"don{n}"] = (c - lo) / (hi - lo + 1e-8)
    # Calendar
    df["dow"] = c.index.dayofweek
    df["dom"] = c.index.day
    df["mon"] = c.index.month
    df["mon_sin"] = np.sin(2 * np.pi * c.index.month / 12)
    df["mon_cos"] = np.cos(2 * np.pi * c.index.month / 12)
    # Distance from extremes
    for n in [10, 50]:
        df[f"dhi{n}"] = (c.rolling(n).max() - c) / (c + 1e-8)
        df[f"dlo{n}"] = (c - c.rolling(n).min()) / (c + 1e-8)
    # Momentum divergence
    df["md5_20"] = c.pct_change(5) - c.pct_change(20)
    df["md10_50"] = c.pct_change(10) - c.pct_change(50)
    # Price acceleration
    df["acc5"] = c.pct_change(5).diff(5)
    return df


def make_target(c: pd.Series, horizon: int = 20) -> np.ndarray:
    fwd = c.shift(-horizon); ret = (fwd / c - 1).fillna(0)
    # Only strong moves
    t = ret.std() * 0.5
    y = np.zeros(len(c))
    y[ret > t] = 1; y[ret < -t] = -1
    return y


def wf_backtest_metric(c: pd.Series, y_pred: np.ndarray, start_idx: int) -> dict:
    """Backtest a model's predictions from start_idx onward."""
    test_c = c.iloc[start_idx:]
    test_y = y_pred[start_idx:]
    test_y = test_y[:len(test_c)]
    sig = pd.Series(test_y, index=test_c.index[:len(test_y)])
    r = backtest_signal_ftmo(test_c.iloc[:len(test_y)], sig.astype(float), txn_cost_bps=TXN_COST_BPS)
    return {
        "oos_sharpe": r.get("oos_sharpe", -999) or -999,
        "oos_monthly": r.get("oos_monthly_return_pct", 0) or 0,
        "oos_dd": r.get("oos_max_drawdown", 0) or 0,
        "oos_trades": r.get("oos_n_trades", 0),
        "is_sharpe": r.get("is_sharpe", -999),
    }


def objective(trial, X, y, c, split_idx):
    """Optuna objective: maximize OOS Sharpe on validation fold."""
    n_est = trial.suggest_int("n_estimators", 100, 500, step=50)
    max_d = trial.suggest_int("max_depth", 3, 20)
    min_split = trial.suggest_int("min_samples_split", 2, 20)
    min_leaf = trial.suggest_int("min_samples_leaf", 1, 10)
    max_feat = trial.suggest_float("max_features", 0.3, 1.0)

    X_train, y_train = X[:split_idx], y[:split_idx]

    model = RandomForestClassifier(
        n_estimators=n_est, max_depth=max_d,
        min_samples_split=min_split, min_samples_leaf=min_leaf,
        max_features=max_feat, random_state=42, n_jobs=-1,
    )
    model.fit(X_train, y_train)
    y_pred = model.predict(X)

    metrics = wf_backtest_metric(c, y_pred, split_idx)
    return metrics["oos_sharpe"]


def main():
    print(f"\n{'='*65}")
    print("  NexQuant Optuna ML Pipeline — Daily Strategies")
    print(f"  Target: 8%/month  |  Trials: {N_TRIALS}  |  Cost: {TXN_COST_BPS} bps")
    print(f"{'='*65}")

    c = load_data()
    X_df = make_features(c).dropna()
    common = c.index.intersection(X_df.index)
    c = c.loc[common]; X_df = X_df.loc[common]
    print(f"Data: {len(c)} daily bars  |  Features: {len(X_df.columns)}\n")

    # Test multiple horizons
    all_results = []

    for horizon in [5, 10, 20]:
        print(f"{'─'*50}")
        print(f"  HORIZON: {horizon}d forward return")
        print(f"{'─'*50}")

        y = make_target(c, horizon)
        mask = ~np.isnan(y) & ~np.isinf(np.abs(y))
        X = X_df.loc[mask].values.astype(np.float32)
        y_vals = y[mask].astype(int)
        split_idx = int(len(X) * 0.75)

        # Only train if we have enough OOS data
        if len(X) - split_idx < 20:
            print("  Not enough OOS data, skipping\n")
            continue

        # Run Optuna
        print(f"  Training: {split_idx} bars  |  OOS: {len(X) - split_idx} bars")
        study = optuna.create_study(
            direction="maximize",
            sampler=optuna.samplers.TPESampler(seed=42),
            pruner=optuna.pruners.HyperbandPruner(),
        )

        def obj(trial):
            return objective(trial, X, y_vals, c, split_idx)

        study.optimize(obj, n_trials=N_TRIALS, show_progress_bar=False)

        best = study.best_params
        best_val = study.best_value

        # Train best model
        best_model = RandomForestClassifier(
            n_estimators=best["n_estimators"], max_depth=best["max_depth"],
            min_samples_split=best["min_samples_split"],
            min_samples_leaf=best["min_samples_leaf"],
            max_features=best["max_features"], random_state=42, n_jobs=-1,
        )
        best_model.fit(X[:split_idx], y_vals[:split_idx])
        y_pred = best_model.predict(X)

        metrics = wf_backtest_metric(c, y_pred, split_idx)
        print(f"  Best params: n={best['n_estimators']} d={best['max_depth']}"
              f" split={best['min_samples_split']} leaf={best['min_samples_leaf']}"
              f" feat={best['max_features']:.2f}")
        print(f"  Best OOS Sharpe: {metrics['oos_sharpe']:+.2f}")
        print(f"  OOS Monthly:     {metrics['oos_monthly']:+.3f}%")
        print(f"  OOS Max DD:      {metrics['oos_dd']*100:+.1f}%")
        print(f"  OOS Trades:      {metrics['oos_trades']}")
        print()

        all_results.append({
            "horizon": horizon, **best, **metrics,
            "features": len(X_df.columns),
        })

    # Summary
    print(f"{'='*65}")
    print(f"  RESULTS")
    print(f"{'='*65}")
    print(f"  {'Horiz':<7} {'OOS S':>8} {'Mon%':>8} {'DD%':>7} {'Trades':>7}  {'n_est':>6} {'depth':>6} {'split':>6}")
    print(f"  {'─'*57}")
    for r in sorted(all_results, key=lambda x: x["oos_monthly"], reverse=True):
        print(f"  {r['horizon']:>4}d   {r['oos_sharpe']:>+8.2f} {r['oos_monthly']:>+7.3f}% {r['oos_dd']*100:>+6.1f}% {r['oos_trades']:>7}  {r['n_estimators']:>6} {r['max_depth']:>6} {r['min_samples_split']:>6}")

    best_r = max(all_results, key=lambda x: x["oos_monthly"])
    print(f"\n  Best: {best_r['horizon']}d horizon → {best_r['oos_monthly']:+.3f}%/month")
    print(f"  Target 8%/month: {'✅ ACHIEVED!' if best_r['oos_monthly'] >= 8 else 'Still working...'}")

    # Feature importance
    print(f"\n  Top 10 Features:")
    imps = best_model.feature_importances_
    cols = X_df.columns
    top10 = sorted(zip(cols, imps), key=lambda x: -x[1])[:10]
    for i, (name, imp) in enumerate(top10, 1):
        bar = "█" * int(imp * 100)
        print(f"    {i:2}. {name:<20s} {imp:.3f}  {bar}")


if __name__ == "__main__":
    main()
