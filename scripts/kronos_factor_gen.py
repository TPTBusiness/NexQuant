#!/usr/bin/env python
"""
Option A: Generate Kronos predicted-return factor from EUR/USD 1-min data.

Runs Kronos-mini inference in daily strides (96 bars/day) over all available
OHLCV data and saves the resulting factor for use in NexQuant's factor pipeline.

Usage:
    conda activate nexquant
    python scripts/kronos_factor_gen.py
    python scripts/kronos_factor_gen.py --context 512 --pred 96 --device cuda
    python scripts/kronos_factor_gen.py --device cpu  # slower but no GPU needed
"""

import argparse
import json
from datetime import datetime
from pathlib import Path

from dotenv import load_dotenv

load_dotenv(Path(__file__).parent.parent / ".env")

import pandas as pd
import torch

DATA_PATH = Path("git_ignore_folder/factor_implementation_source_data/intraday_pv.h5")
OUTPUT_DIR = Path("results/factors")


def main():
    parser = argparse.ArgumentParser(description="Generate Kronos predicted-return factor")
    parser.add_argument("--context", type=int, default=512, help="Context window in bars (max 512 for Kronos-mini)")
    parser.add_argument("--pred", type=int, default=96, help="Prediction horizon in bars (default: 96 = 1 trading day)")
    parser.add_argument("--stride", type=int, default=None, help="Stride between windows (default: same as --pred)")
    parser.add_argument("--device", type=str, default="cuda" if torch.cuda.is_available() else "cpu")
    parser.add_argument("--output", type=str, default=None, help="Output parquet path (default: auto)")
    args = parser.parse_args()

    stride = args.stride or args.pred

    print(f"Kronos Factor Generator")
    print(f"  Data:    {DATA_PATH}")
    print(f"  Context: {args.context} bars")
    print(f"  Pred:    {args.pred} bars ({args.pred} min = {args.pred/96:.1f} trading days)")
    print(f"  Stride:  {stride} bars")
    print(f"  Device:  {args.device}")
    print()

    if not DATA_PATH.exists():
        print(f"ERROR: Data not found at {DATA_PATH}")
        print("Run data conversion first — see README Data Setup section.")
        raise SystemExit(1)

    from rdagent.components.coder.kronos_adapter import build_kronos_factor

    factor_df = build_kronos_factor(
        hdf5_path=DATA_PATH,
        context_bars=args.context,
        pred_bars=args.pred,
        stride_bars=stride,
        device=args.device,
    )

    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
    out_path = args.output or OUTPUT_DIR / f"kronos_pred_return_p{args.pred}.parquet"
    factor_df.to_parquet(out_path)
    print(f"\nFactor saved to: {out_path}")
    print(f"Shape: {factor_df.shape}")
    print(f"Non-NaN: {factor_df['KronosPredReturn'].notna().sum()}")
    print(f"\nSample (first 5):")
    print(factor_df.head())

    # Save metadata for nexquant.py top / best integration
    meta = {
        "factor_name": f"KronosPredReturn_p{args.pred}",
        "description": f"Kronos-mini predicted return, {args.pred}-bar horizon",
        "model": "NeoQuasar/Kronos-mini",
        "context_bars": args.context,
        "pred_bars": args.pred,
        "stride_bars": stride,
        "device": args.device,
        "generated_at": datetime.now().isoformat(),
        "n_bars": len(factor_df),
        "n_non_nan": int(factor_df["KronosPredReturn"].notna().sum()),
        "parquet_path": str(out_path),
    }
    meta_path = out_path.with_suffix(".json")
    with open(meta_path, "w") as f:
        json.dump(meta, f, indent=2)
    print(f"Metadata saved to: {meta_path}")


if __name__ == "__main__":
    main()
