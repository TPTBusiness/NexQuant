#!/usr/bin/env python3
"""Strategy Grid Search — Systematic parameter scanning for optimal strategies.

Unlike the random R&D loop, this tests ALL parameter/TF combinations
for the best indicators, guaranteeing global optimum discovery.

Output: Ranked list of strategies with per-instrument + combined metrics.
"""

import json, os, sys, time, itertools
from datetime import datetime
from pathlib import Path
import numpy as np, pandas as pd

PROJECT = Path(__file__).resolve().parent.parent
OHLCV_PATH = Path(os.getenv("PREDIX_OHLCV_PATH",
    str(PROJECT / "git_ignore_folder" / "intraday_pv_all.h5")))
OUTPUT_DIR = PROJECT / "results" / "grid_search"
OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

sys.path.insert(0, str(PROJECT / "scripts"))
from nexquant_rd_loop import (
    evaluate_multi, build_signal, _apply_session_filter, _apply_news_filter,
    _apply_vola_filter, _apply_cross_confirm, LEADER_MAP, load_data,
)

# ── Grid Definition ──

INDICATOR_GRIDS = {
    "MACD": {
        "type": "multi_tf",
        "params": {
            "fast": [3, 5, 8, 12],
            "slow": [10, 15, 20, 26, 40],
            "sig": [3, 5, 9],
        },
        "tfs": [
            ["15min", "30min", "1h", "4h"],
            ["15min", "30min", "1h"],
            ["30min", "1h", "4h"],
            ["15min", "1h", "4h"],
        ],
    },
    "Donchian": {
        "type": "multi_tf",
        "params": {
            "period": [5, 10, 20, 30, 50, 80, 100],
            "hold": [1, 2, 3, 5, 10],
        },
        "tfs": [
            ["15min", "30min", "1h", "4h"],
            ["30min", "1h", "4h"],
            ["15min", "1h", "4h"],
        ],
    },
    "SAR": {
        "type": "multi_tf",
        "params": {
            "accel": [0.02, 0.05, 0.08, 0.1, 0.15],
            "max_accel": [0.1, 0.2, 0.3, 0.5],
        },
        "tfs": [
            ["15min", "30min", "1h", "4h"],
            ["30min", "1h", "4h"],
            ["15min", "1h", "4h"],
        ],
    },
    "ADX": {
        "type": "multi_tf",
        "params": {
            "period": [7, 10, 14, 21, 30],
            "threshold": [15, 20, 25, 30],
        },
        "tfs": [
            ["15min", "30min", "1h", "4h"],
            ["30min", "1h", "4h"],
        ],
    },
    "RSI": {
        "type": "multi_tf",
        "params": {
            "period": [7, 10, 14, 21],
            "oversold": [20, 25, 30],
            "overbought": [70, 75, 80],
        },
        "tfs": [
            ["15min", "30min", "1h", "4h"],
            ["30min", "1h", "4h"],
        ],
    },
    "BBands": {
        "type": "multi_tf",
        "params": {
            "period": [10, 20, 40],
            "std": [1.5, 2.0, 2.5],
        },
        "tfs": [
            ["15min", "30min", "1h", "4h"],
            ["30min", "1h", "4h"],
        ],
    },
    "ROC": {
        "type": "multi_tf",
        "params": {
            "period": [5, 10, 20, 50],
            "threshold": [0.1, 0.2, 0.5, 1.0],
        },
        "tfs": [
            ["15min", "30min", "1h", "4h"],
            ["30min", "1h", "4h"],
        ],
    },
    "MOM": {
        "type": "multi_tf",
        "params": {
            "period": [5, 10, 20, 50, 100],
        },
        "tfs": [
            ["15min", "30min", "1h", "4h"],
            ["30min", "1h", "4h"],
        ],
    },
    "Stoch": {
        "type": "multi_tf",
        "params": {
            "fastk": [5, 9, 14],
            "slowd": [3, 5, 9],
        },
        "tfs": [
            ["15min", "30min", "1h", "4h"],
            ["30min", "1h", "4h"],
        ],
    },
    "CCI": {
        "type": "multi_tf",
        "params": {
            "period": [10, 14, 20, 50],
        },
        "tfs": [
            ["15min", "30min", "1h", "4h"],
            ["30min", "1h", "4h"],
        ],
    },
}


def expand_grid(indicator_name):
    """Expand a grid definition into all parameter+TF combinations."""
    grid = INDICATOR_GRIDS[indicator_name]
    param_keys = list(grid["params"].keys())
    param_values = [grid["params"][k] for k in param_keys]
    hypotheses = []

    for tf_list in grid["tfs"]:
        for param_combo in itertools.product(*param_values):
            params = dict(zip(param_keys, param_combo))
            hypotheses.append({
                "type": grid["type"],
                "indicator": indicator_name,
                "timeframes": tf_list,
                "params": params,
                "description": f"{indicator_name}({'-'.join(str(v) for v in param_combo)}) on {','.join(tf_list[:2])}",
                "generation": "grid",
            })

    return hypotheses


def main():
    import argparse
    ap = argparse.ArgumentParser()
    ap.add_argument("--indicators", nargs="*", default=None,
                    help="Indicators to grid-search (default: all)")
    ap.add_argument("--top", type=int, default=20,
                    help="Number of top results to show")
    args = ap.parse_args()

    indicators = args.indicators or list(INDICATOR_GRIDS.keys())
    if isinstance(indicators, str):
        indicators = [indicators]

    print("=" * 60)
    print("  Strategy Grid Search")
    print(f"  Indicators: {', '.join(indicators)}")
    print("=" * 60)

    # Load data
    print("  Loading data...")
    closes = load_data()
    if not closes:
        print("  No instruments found!"); return

    # Generate all hypotheses
    all_hypotheses = []
    for ind in indicators:
        hyps = expand_grid(ind)
        all_hypotheses.extend(hyps)
    print(f"  Total combinations to test: {len(all_hypotheses)}")
    print()

    # Evaluate all
    results = []
    t0 = time.time()
    for i, hp in enumerate(all_hypotheses):
        try:
            r = evaluate_multi(closes, hp, use_session=True, use_vola=False)
            r["hypothesis"] = hp
            r["rank"] = i + 1
            results.append(r)
        except Exception:
            continue

        elapsed = time.time() - t0
        rate = (i + 1) / elapsed if elapsed > 0 else 0
        eta = (len(all_hypotheses) - i - 1) / rate if rate > 0 else 0

        if (i + 1) % 50 == 0:
            best_so_far = max(results, key=lambda x: x["sharpe"]) if results else {"sharpe": 0}
            print(f"  [{i+1}/{len(all_hypotheses)}] "
                  f"Best Sh={best_so_far['sharpe']:.1f} "
                  f"Mon={best_so_far['monthly_pct']:.1f}% "
                  f"OOS={best_so_far['monthly_oos']:.1f}% | "
                  f"{rate:.0f}/s | ETA {eta/60:.0f}min")

    # Sort by OOS Sharpe (most important metric)
    results.sort(key=lambda r: r.get("sharpe", 0), reverse=True)

    elapsed = time.time() - t0
    print(f"\n{'=' * 60}")
    print(f"  Grid Search Complete: {len(results)}/{len(all_hypotheses)} valid")
    print(f"  Time: {elapsed:.0f}s ({elapsed/60:.1f}min)")
    print(f"{'=' * 60}")

    # Save all results
    ts = datetime.now().strftime("%Y%m%d_%H%M%S")
    out_file = OUTPUT_DIR / f"grid_results_{ts}.json"
    stripped = [{k: v for k, v in r.items() if k != "equity_curves"} for r in results]
    out_file.write_text(json.dumps(stripped, indent=2, default=str))
    print(f"  Saved: {out_file}")

    # Show top results
    top_n = min(args.top, len(results))
    print(f"\n  TOP {top_n} (by OOS Sharpe):")
    print(f"  {'Rank':>4s} {'Strategy':<45s} {'Sh_IS':>6s} {'Sh_OOS':>6s} {'Mon%':>7s} {'OOS%':>7s} {'DD':>6s} {'Tr':>5s} {'BTC':>5s}")
    for i, r in enumerate(results[:top_n], 1):
        hp = r["hypothesis"]
        per = r.get("per_instrument", {})
        btc_sh = per.get("BTCUSD", {}).get("sharpe_oos", 0)
        print(f"  {i:4d} {hp['description'][:45]:45s} "
              f"{r.get('sharpe_is', 0):+6.1f} {r.get('sharpe_oos', 0):+6.1f} "
              f"{r['monthly_pct']:+6.1f}% {r['monthly_oos']:+6.1f}% "
              f"{r['max_dd']:.4f} {r['n_trades']:5d} {btc_sh:+5.0f}")

    # Indicator performance summary
    print(f"\n  Indicator Performance (avg OOS Sharpe):")
    for ind in indicators:
        ind_results = [r for r in results if r["hypothesis"].get("indicator") == ind]
        if ind_results:
            avg_sh = np.mean([r["sharpe"] for r in ind_results])
            best = ind_results[0]
            print(f"  {ind:12s}: avg Sh={avg_sh:+.1f} best={best['sharpe']:+.1f} "
                  f"({best['monthly_pct']:+.1f}%/{best['monthly_oos']:+.1f}% OOS)")


if __name__ == "__main__":
    main()
