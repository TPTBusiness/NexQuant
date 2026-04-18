#!/usr/bin/env python
"""
Option B: Evaluate Kronos-mini as a model alongside LightGBM.

Computes IC (Information Coefficient) and hit rate for Kronos predictions
vs actual realized returns. Results are printed for comparison with LightGBM.

Usage:
    conda activate predix
    python scripts/kronos_model_eval.py
    python scripts/kronos_model_eval.py --pred 30 --context 512 --device cuda
"""

import argparse
import json
from pathlib import Path

from dotenv import load_dotenv

load_dotenv(Path(__file__).parent.parent / ".env")

import torch

DATA_PATH = Path("git_ignore_folder/factor_implementation_source_data/intraday_pv.h5")
OUTPUT_DIR = Path("results/kronos")


def main():
    parser = argparse.ArgumentParser(description="Evaluate Kronos as model (alongside LightGBM)")
    parser.add_argument("--context", type=int, default=512, help="Context window in bars")
    parser.add_argument("--pred", type=int, default=30, help="Prediction horizon in bars")
    parser.add_argument("--stride", type=int, default=None, help="Stride between evaluations (default: pred)")
    parser.add_argument("--device", type=str, default="cuda" if torch.cuda.is_available() else "cpu")
    args = parser.parse_args()

    stride = args.stride or args.pred

    print(f"Kronos Model Evaluator (alongside LightGBM)")
    print(f"  Context: {args.context} bars  |  Pred: {args.pred} bars  |  Device: {args.device}")
    print()

    if not DATA_PATH.exists():
        print(f"ERROR: Data not found at {DATA_PATH}")
        raise SystemExit(1)

    from rdagent.components.coder.kronos_adapter import evaluate_kronos_model

    print("Running evaluation (this may take several minutes)...")
    metrics = evaluate_kronos_model(
        hdf5_path=DATA_PATH,
        context_bars=args.context,
        pred_bars=args.pred,
        stride_bars=stride,
        device=args.device,
    )

    print("\n" + "=" * 50)
    print("Kronos-mini Model Evaluation Results")
    print("=" * 50)
    print(f"  Predictions:  {metrics['n_predictions']}")
    print(f"  IC (mean):    {metrics['IC_mean']:.4f}")
    print(f"  IC (std):     {metrics['IC_std']:.4f}")
    print(f"  IC IR:        {metrics['IC_IR']:.4f}  (>0.5 = good)")
    print(f"  Hit Rate:     {metrics['hit_rate']:.2%}  (>50% = directionally useful)")
    print("=" * 50)
    print()
    print("Reference: LightGBM baseline IC typically 0.01–0.05 on 1-min EUR/USD")

    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
    out = OUTPUT_DIR / f"kronos_eval_ctx{args.context}_pred{args.pred}.json"
    with open(out, "w") as f:
        json.dump({**metrics, "context_bars": args.context, "pred_bars": args.pred}, f, indent=2)
    print(f"\nResults saved to: {out}")


if __name__ == "__main__":
    main()
