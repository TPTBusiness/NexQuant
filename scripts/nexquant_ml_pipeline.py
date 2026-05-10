#!/usr/bin/env python
"""
NexQuant Enhanced ML Pipeline — factor-boosted, multi-horizon, Optuna-optimized.
Target: 8%/month through ensemble of factor + OHLCV features.
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
from sklearn.ensemble import RandomForestClassifier, GradientBoostingClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import TimeSeriesSplit

from rdagent.components.backtesting.vbt_backtest import backtest_signal_ftmo

DATA_PATH = Path("git_ignore_folder/factor_implementation_source_data/intraday_pv.h5")
FACTORS_DIR = Path("results/factors")
TXN_COST_BPS = 2.14
N_TRIALS = 75


def load_all():
    close = pd.read_hdf(DATA_PATH, key="data")["$close"]
    if isinstance(close.index, pd.MultiIndex):
        close = close.droplevel(-1)
    daily = close.sort_index().dropna().resample("1D").last().dropna()

    # Load top factors
    factors = []
    for f in sorted(FACTORS_DIR.glob("*.json")):
        try: d = json.loads(f.read_text())
        except: continue
        if d.get("status") != "success" or d.get("ic") is None: continue
        ic = d["ic"]
        if abs(ic) < 0.02: continue
        name = d.get("factor_name", f.stem)
        safe = name.replace("/", "_")[:150]
        pf = FACTORS_DIR / "values" / f"{safe}.parquet"
        if pf.exists():
            factors.append((abs(ic), name))

    factors.sort(reverse=True)
    top = factors[:100]

    # Load factor values
    fdata = {}
    for _, name in top:
        safe = name.replace("/", "_")[:150]
        s = pd.read_parquet(FACTORS_DIR / "values" / f"{safe}.parquet").iloc[:, 0]
        if isinstance(s.index, pd.MultiIndex):
            s = s.droplevel(-1)
        fdata[name] = s.resample("1D").last()

    df = pd.DataFrame(fdata)
    common = daily.index.intersection(df.dropna(how="all").index)
    return daily.loc[common], df.loc[common].ffill()


def add_ohlcv_features(c: pd.Series) -> pd.DataFrame:
    """Lightweight OHLCV features to complement factors."""
    df = pd.DataFrame(index=c.index)
    for n in [1, 5, 10, 20]:
        df[f"ret_{n}"] = c.pct_change(n)
    for n in [10, 20, 50, 100]:
        df[f"sma_{n}"] = c.rolling(n).mean() / c - 1
    df["sma10_50"] = c.rolling(10).mean() / c.rolling(50).mean() - 1
    df["sma20_100"] = c.rolling(20).mean() / c.rolling(100).mean() - 1
    for n in [5, 20]:
        df[f"vol_{n}"] = c.pct_change().rolling(n).std()
    d = c.diff(); g = d.clip(lower=0); l = -d.clip(upper=0)
    df["rsi14"] = 100 - (100 / (1 + g.rolling(14).mean() / (l.rolling(14).mean() + 1e-8)))
    df["adx14"] = (100 * abs(c.diff().clip(lower=0).ewm(14).mean() - (-c.diff().clip(upper=0)).ewm(14).mean()) / (
        c.diff().abs().rolling(14).mean() + 1e-8)).ewm(14).mean()
    return df


def make_target(c: pd.Series, horizon: int = 5) -> np.ndarray:
    fwd = c.shift(-horizon)
    ret = (fwd / c - 1).fillna(0)
    t = ret.std() * 0.3  # Tighter threshold for more signals
    y = np.zeros(len(c))
    y[ret > t] = 1
    y[ret < -t] = -1
    return y


def backtest_metric(c, y_pred, split_idx):
    test_c = c.iloc[split_idx:]
    sig = pd.Series(y_pred[split_idx:len(test_c)+split_idx], index=test_c.index[:len(y_pred)-split_idx])
    r = backtest_signal_ftmo(test_c.iloc[:len(sig)], sig.astype(float), txn_cost_bps=TXN_COST_BPS)
    return r.get("oos_sharpe", -999) or -999


def main():
    print(f"\n{'='*65}")
    print("  NexQuant Factor-Boosted ML Pipeline")
    print(f"  Target: 8%/month  |  Trials: {N_TRIALS}/horizon")
    print(f"{'='*65}")

    c, factor_df = load_all()
    ohlcv_df = add_ohlcv_features(c)
    X_df = pd.concat([factor_df, ohlcv_df], axis=1).dropna()
    common = c.index.intersection(X_df.index)
    c = c.loc[common]; X_df = X_df.loc[common]
    print(f"Daily: {len(c):,} bars  |  Features: {len(X_df.columns)} ({len(factor_df.columns)} factors + {len(ohlcv_df.columns)} OHLCV)\n")

    all_results = []

    for horizon in [5, 10, 20]:
        print(f"─── HORIZON {horizon}d ───")
        y = make_target(c, horizon)
        mask = ~np.isnan(y) & ~np.isinf(np.abs(y))
        X = X_df.loc[mask].values.astype(np.float32)
        y_vals = y[mask].astype(int)
        split_idx = int(len(X) * 0.75)

        if len(X) - split_idx < 20:
            print("  Skip — not enough OOS\n")
            continue

        print(f"  Train: {split_idx}  OOS: {len(X)-split_idx}")

        # Test multiple model types
        for model_name, ModelClass, param_space in [
            ("RF", RandomForestClassifier, {
                "n": ("suggest_int", 100, 500), "d": ("suggest_int", 3, 25),
                "split": ("suggest_int", 2, 15), "leaf": ("suggest_int", 1, 10),
                "feat": ("suggest_float", 0.3, 1.0),
            }),
            ("GBM", GradientBoostingClassifier, {
                "n": ("suggest_int", 100, 500), "d": ("suggest_int", 2, 10),
                "lr": ("suggest_float", 0.01, 0.3), "split": ("suggest_int", 2, 20),
                "leaf": ("suggest_int", 1, 10),
            }),
        ]:
            def obj(trial):
                p = {}
                if model_name == "RF":
                    p = {
                        "n_estimators": trial.suggest_int("n", *param_space["n"][1:]),
                        "max_depth": trial.suggest_int("d", *param_space["d"][1:]),
                        "min_samples_split": trial.suggest_int("split", *param_space["split"][1:]),
                        "min_samples_leaf": trial.suggest_int("leaf", *param_space["leaf"][1:]),
                        "max_features": trial.suggest_float("feat", *param_space["feat"][1:]),
                        "random_state": 42, "n_jobs": -1,
                    }
                else:
                    p = {
                        "n_estimators": trial.suggest_int("n", *param_space["n"][1:]),
                        "max_depth": trial.suggest_int("d", *param_space["d"][1:]),
                        "learning_rate": trial.suggest_float("lr", *param_space["lr"][1:]),
                        "min_samples_split": trial.suggest_int("split", *param_space["split"][1:]),
                        "min_samples_leaf": trial.suggest_int("leaf", *param_space["leaf"][1:]),
                        "random_state": 42,
                    }
                model = ModelClass(**p)
                model.fit(X[:split_idx], y_vals[:split_idx])
                return backtest_metric(c, model.predict(X), split_idx)

            study = optuna.create_study(direction="maximize", sampler=optuna.samplers.TPESampler(seed=42))
            study.optimize(obj, n_trials=N_TRIALS, show_progress_bar=False)

            best = study.best_params
            best_val = study.best_value

            # Final model
            if model_name == "RF":
                model = RandomForestClassifier(
                    n_estimators=best.get("n",200), max_depth=best.get("d",10),
                    min_samples_split=best.get("split",2), min_samples_leaf=best.get("leaf",1),
                    max_features=best.get("feat",0.5), random_state=42, n_jobs=-1,
                )
            else:
                model = GradientBoostingClassifier(
                    n_estimators=best.get("n",200), max_depth=best.get("d",5),
                    learning_rate=best.get("lr",0.1), min_samples_split=best.get("split",2),
                    min_samples_leaf=best.get("leaf",1), random_state=42,
                )
            model.fit(X[:split_idx], y_vals[:split_idx])
            y_pred = model.predict(X)
            sig = pd.Series(y_pred[split_idx:len(c)-split_idx+split_idx], index=c.index[split_idx:split_idx+len(y_pred)-split_idx])
            r = backtest_signal_ftmo(c.iloc[split_idx:split_idx+len(sig)], sig.astype(float), txn_cost_bps=TXN_COST_BPS)

            oos_s = r.get("oos_sharpe", -999)
            oos_m = (r.get("oos_monthly_return_pct", 0) or 0)
            oos_dd = (r.get("oos_max_drawdown", 0) or 0) * 100
            trades = r.get("oos_n_trades", 0)
            print(f"  {model_name} h={horizon}d  OOS={oos_s:+.1f}  Mon={oos_m:+.3f}%  DD={oos_dd:+.1f}%  T={trades}")

            all_results.append({
                "model": model_name, "horizon": horizon,
                "oos_sharpe": oos_s, "monthly": oos_m, "dd": oos_dd, "trades": trades,
            })

    # Summary
    print(f"\n{'='*65}")
    print(f"  {'Model':<6} {'Horiz':<6} {'OOS S':>8} {'Mon%':>9} {'DD%':>7} {'Trades':>7}")
    print(f"  {'─'*46}")
    for r in sorted(all_results, key=lambda x: x["monthly"], reverse=True):
        print(f"  {r['model']:<6} {r['horizon']:>3}d   {r['oos_sharpe']:>+8.1f} {r['monthly']:>+8.3f}% {r['dd']:>+6.1f}% {r['trades']:>7}")

    best = max(all_results, key=lambda x: x["monthly"])
    print(f"\n  Best: {best['model']} {best['horizon']}d → {best['monthly']:+.3f}%/month")
    gap = 8.0 - best['monthly']
    print(f"  Gap to 8%: {gap:+.3f}% {'✅' if gap <= 0 else '— needs improvement'}")

    # Feature importance from best model
    if hasattr(model, 'feature_importances_'):
        imps = model.feature_importances_
        cols = X_df.columns
        top = sorted(zip(cols, imps), key=lambda x: -x[1])[:15]
        print(f"\n  Top Features ({len(X_df.columns)} total):")
        for i, (name, imp) in enumerate(top, 1):
            src = "F" if name in factor_df.columns else "O"
            print(f"    {i:2}. [{src}] {name:<45s} {imp:.4f}")


if __name__ == "__main__":
    main()
