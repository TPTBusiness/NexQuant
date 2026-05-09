#!/usr/bin/env python
"""
NexQuant Auto-Pilot — vollautomatischer Strategie-Generator.

Läuft unbegrenzt, kein menschlicher Eingriff nötig.
Jede Runde: Factors laden → LLM Code → Pre-Flight → Backtest → Optuna → Ensemble
Bei Crash: auto-restart nach 30s.

Usage:
    python scripts/nexquant_autopilot.py
"""
from __future__ import annotations

import json, logging, os, sys, time, traceback
from datetime import datetime
from pathlib import Path

import numpy as np, pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

# Load .env before any rdagent imports (required for pydantic-settings)
try:
    from dotenv import load_dotenv
    _env_path = Path(__file__).resolve().parent.parent / ".env"
    load_dotenv(_env_path)
except ImportError:
    pass

logging.basicConfig(level=logging.INFO, format="%(asctime)s [%(levelname)s] %(message)s")
logger = logging.getLogger("autopilot")

LOG_FILE = Path(__file__).resolve().parent.parent / "git_ignore_folder" / "logs" / f"autopilot_{datetime.now().strftime('%Y%m%d_%H%M%S')}.log"
LOG_FILE.parent.mkdir(parents=True, exist_ok=True)
fh = logging.FileHandler(str(LOG_FILE))
fh.setFormatter(logging.Formatter("%(asctime)s [%(levelname)s] %(message)s"))
logger.addHandler(fh)

BATCH_SIZE = 2
OPTUNA_TRIALS = 10
COOLDOWN = 30
MAX_CONSECUTIVE_FAILS = 5

def main_round(style: str, round_num: int) -> int:
    """Run one round. Returns number of accepted strategies."""
    from rdagent.scenarios.qlib.local.strategy_orchestrator import StrategyOrchestrator

    accepted_count = 0
    try:
        orch = StrategyOrchestrator(
            top_factors=20, trading_style=style,
            min_sharpe=0.1, use_optuna=True, optuna_trials=OPTUNA_TRIALS,
        )
    except Exception as e:
        logger.error(f"Orchestrator init failed: {e}")
        return 0

    try:
        results = orch.generate_strategies(count=BATCH_SIZE, workers=1)
    except Exception as e:
        logger.error(f"generate_strategies failed: {e}")
        return 0

    for r in results:
        status = r.get("status", "?")
        if status == "accepted":
            accepted_count += 1
            logger.info(f"  ✓ {r.get('strategy_name','?')[:40]:40s} S={r.get('sharpe_ratio',0):.1f} OOS={r.get('oos_sharpe',0):.1f}")
        else:
            reason = r.get("reason", "?")[:80]
            logger.debug(f"  ✗ {r.get('strategy_name','?')[:40]:40s} {reason}")

    if accepted_count >= 2:
        try:
            ensemble = orch.build_ensemble(results)
            if ensemble and ensemble.get("status") == "success":
                logger.info(f"  Ensemble: S={ensemble['sharpe_ratio']:.1f} OOS={ensemble['oos_sharpe']:.1f} ({len(ensemble['members'])} members)")
        except Exception:
            pass

    return accepted_count


def main():
    print(f"\n{'='*50}")
    print(f"  NexQuant Auto-Pilot")
    print(f"  Log: {LOG_FILE}")
    print(f"  Batch: {BATCH_SIZE} | Optuna: {OPTUNA_TRIALS} trials")
    print(f"{'='*50}\n")

    round_num = 0
    total_accepted = 0
    consecutive_fails = 0
    start_time = datetime.now()
    styles = ["swing", "daytrading"]

    while True:
        round_num += 1
        style = styles[round_num % 2]
        print(f"\n[Round {round_num}] {style} | {datetime.now().strftime('%H:%M:%S')}", flush=True)

        try:
            accepted = main_round(style, round_num)
            total_accepted += accepted

            if accepted == 0:
                consecutive_fails += 1
            else:
                consecutive_fails = 0

            elapsed = (datetime.now() - start_time).total_seconds()
            rate = total_accepted / (elapsed / 3600) if elapsed > 0 else 0
            print(f"  Accepted: {accepted} | Total: {total_accepted} | Rate: {rate:.1f}/h | Fails: {consecutive_fails}", flush=True)

            if consecutive_fails >= MAX_CONSECUTIVE_FAILS:
                logger.warning(f"{consecutive_fails} consecutive failures — cooling down {COOLDOWN*2}s")
                time.sleep(COOLDOWN * 2)
                consecutive_fails = 0

        except KeyboardInterrupt:
            print(f"\n\nStopped after {round_num} rounds. Total accepted: {total_accepted}")
            break
        except Exception as e:
            logger.error(f"Round {round_num} crashed: {e}\n{traceback.format_exc()[-500:]}")
            consecutive_fails += 1
            time.sleep(COOLDOWN)

        time.sleep(COOLDOWN)


if __name__ == "__main__":
    main()
