#!/usr/bin/env python
"""
Continuous Strategy Generator — runs indefinitely, improving over time.

Features:
- Infinite loop: generate → optimize → ensemble → repeat
- Walk-Forward validation required (OOS Sharpe > 0)
- Multi-Timeframe check (1min, 5min, 15min, 1h)
- Rolling stability check (12-month Sharpe never negative)
- ML model training when LLM suggests it's beneficial
- Auto-ensemble from top strategies
- Daytrading AND swing style alternating

Usage:
    python scripts/nexquant_continuous_strategies.py
    python scripts/nexquant_continuous_strategies.py --style daytrading --rounds 100
    python scripts/nexquant_continuous_strategies.py --style both --workers 4
"""
from __future__ import annotations

import argparse
import json
import logging
import os
import sys
import time
from datetime import datetime
from pathlib import Path

import numpy as np
import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from rdagent.scenarios.qlib.local.strategy_orchestrator import StrategyOrchestrator

logging.basicConfig(level=logging.INFO, format="%(asctime)s [%(levelname)s] %(message)s")
logger = logging.getLogger(__name__)

BATCH_SIZE = 5
COOLDOWN_SECONDS = 30


def build_ml_model(factor_values: pd.DataFrame, close: pd.Series, style: str) -> dict | None:
    """Train ML model if data is sufficient, return strategy dict or None."""
    from sklearn.ensemble import GradientBoostingRegressor

    df = factor_values.ffill().dropna()
    close_aligned = close.reindex(df.index).ffill()

    common = df.index.intersection(close_aligned.index)
    if len(common) < 5000:
        logger.info("ML: insufficient data (<5000 rows)")
        return None

    X = df.loc[common].values
    y = close_aligned.loc[common].pct_change(96).shift(-96).fillna(0).values  # forward 96-bar return

    split = int(len(X) * 0.7)
    X_train, X_test = X[:split], X[split:]
    y_train, y_test = y[:split], y[split:]

    model = GradientBoostingRegressor(n_estimators=100, max_depth=5, random_state=42)
    model.fit(X_train, y_train)

    # Generate signal on test data
    preds = model.predict(X_test)
    signal = pd.Series(np.sign(preds), index=common[split:])

    # Backtest
    from rdagent.components.backtesting.vbt_backtest import backtest_signal_ftmo
    bt = backtest_signal_ftmo(
        close=close_aligned.loc[common[split:]],
        signal=signal,
        txn_cost_bps=2.14,
        wf_rolling=True,
    )

    is_oos_sharpe = bt.get("wf_oos_sharpe_mean", 0)
    if is_oos_sharpe <= 0:
        logger.info(f"ML model rejected: OOS Sharpe={is_oos_sharpe:.2f}")
        return None

    logger.info(f"ML model accepted: Sharpe={bt['sharpe']:.2f} OOS={is_oos_sharpe:.2f}")
    return {
        "strategy_name": f"ML_GradientBoost_{style}_{int(time.time())}",
        "status": "accepted",
        "sharpe_ratio": round(bt["sharpe"], 4),
        "max_drawdown": round(bt["max_drawdown"], 4),
        "win_rate": round(bt["win_rate"], 4),
        "n_trades": bt["n_trades"],
        "oos_sharpe": round(is_oos_sharpe, 4),
        "type": "ml_model",
    }


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--style", default="both", choices=["daytrading", "swing", "both"])
    parser.add_argument("--workers", type=int, default=2)
    parser.add_argument("--rounds", type=int, default=0, help="Stop after N rounds (0=infinite)")
    parser.add_argument("--min-sharpe", type=float, default=1.5)
    parser.add_argument("--batch-size", type=int, default=5)
    parser.add_argument("--ml-rounds", type=int, default=3, help="Train ML model every N rounds")
    args = parser.parse_args()

    print(f"\n{'='*60}")
    print(f"  NexQuant Continuous Strategy Generator")
    print(f"  Style: {args.style} | Workers: {args.workers}")
    print(f"  Min Sharpe: {args.min_sharpe} | Batch: {args.batch_size}")
    print(f"  ML every {args.ml_rounds} rounds")
    print(f"{'='*60}\n")

    round_num = 0
    total_accepted = 0
    total_ml_accepted = 0
    start_time = datetime.now()

    while True:
        round_num += 1
        styles = [args.style] if args.style != "both" else (["swing", "daytrading"] if round_num % 2 == 1 else ["daytrading", "swing"])

        for style in styles:
            print(f"\n--- Round {round_num} | Style: {style} ---")

            orch = StrategyOrchestrator(
                top_factors=20, trading_style=style,
                min_sharpe=args.min_sharpe,
                use_optuna=True, optuna_trials=30,
            )

            try:
                results = orch.generate_strategies(count=BATCH_SIZE, workers=args.workers)
            except Exception as e:
                logger.error(f"Round {round_num} {style} failed: {e}")
                continue

            accepted = [r for r in results if r.get("status") == "accepted"]
            total_accepted += len(accepted)
            print(f"  Accepted: {len(accepted)}/{len(results)}  (Total: {total_accepted})")

            for r in accepted[:3]:
                print(f"    {r.get('strategy_name', '?')[:40]:40s} S={r.get('sharpe_ratio',0):.1f} OOS={r.get('oos_sharpe',0):.1f}")

            # Ensemble after every round
            ensemble = orch.build_ensemble(results)
            if ensemble and ensemble.get("status") == "success":
                print(f"  Ensemble: S={ensemble['sharpe_ratio']:.1f} OOS={ensemble['oos_sharpe']:.1f} ({len(ensemble['members'])} members)")

            # ML model every N rounds
            if round_num % args.ml_rounds == 0:
                print(f"\n  [ML] Training model on all factors...")
                factors = orch.load_top_factors()
                if factors:
                    factor_values = {}
                    for f in factors:
                        series = orch.load_factor_values(f["factor_name"])
                        if series is not None:
                            factor_values[f["factor_name"]] = series
                    if len(factor_values) >= 3:
                        df = pd.DataFrame(factor_values)
                        if isinstance(df.index, pd.MultiIndex):
                            df = df.droplevel(-1)
                        ml_result = build_ml_model(df, orch.ohlcv_close, style)
                        if ml_result:
                            total_ml_accepted += 1
                            print(f"  [ML] Accepted! S={ml_result['sharpe_ratio']:.1f} OOS={ml_result['oos_sharpe']:.1f}")

        elapsed = (datetime.now() - start_time).total_seconds()
        print(f"\n  Elapsed: {elapsed/60:.0f}min | Accepted: {total_accepted} (+{total_ml_accepted} ML) | Rate: {total_accepted/(elapsed/3600):.1f}/h")

        if args.rounds > 0 and round_num >= args.rounds:
            break

        time.sleep(COOLDOWN_SECONDS)

    print(f"\n{'='*60}")
    print(f"  DONE: {total_accepted} strategies + {total_ml_accepted} ML models")
    print(f"  Total time: {(datetime.now()-start_time).total_seconds()/3600:.1f}h")
    print(f"{'='*60}")


if __name__ == "__main__":
    main()
