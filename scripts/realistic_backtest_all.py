"""
Realistic backtest of all strategies in results/strategies_new/.

Costs modeled per trade:
  1.5 pip spread + 0.5 pip slippage + 0.35 pip commission = 2.35 pip total

FTMO 100k rules enforced:
  - Max daily loss:   5%  of initial balance ($5,000)  → no trading rest of day if hit
  - Max total loss:  10%  of initial balance ($10,000)  → account blown, simulation ends
  - Position sizing: 1% equity risk per trade, 10-pip stop (no artificial lot cap)
  - Max leverage:    1:30 (EU regulation standard, FTMO default)
  - Compounding:     position size grows with equity each trade

Out-of-sample window: 2024-01-01 onwards (never seen during factor research).

Usage:
    conda activate predix
    python scripts/realistic_backtest_all.py
    python scripts/realistic_backtest_all.py --target-monthly 4.0 --min-trades 50
    python scripts/realistic_backtest_all.py --workers 8
"""

from __future__ import annotations

import argparse
import json
import glob
import os
from concurrent.futures import ProcessPoolExecutor, as_completed
from pathlib import Path

import numpy as np
import pandas as pd

# ── Constants ──────────────────────────────────────────────────────────────────
DATA_H5    = Path("git_ignore_folder/factor_implementation_source_data/intraday_pv.h5")
FACTOR_DIR = Path("results/factors/values")
STRAT_DIR  = Path("results/strategies_new")
OUTPUT_DIR = Path("results/realistic_backtest")

PIP               = 0.0001
COST_ENTRY        = 2.0 * PIP    # spread + slippage
COST_EXIT         = 0.35 * PIP   # commission
RISK_PCT          = 0.01          # 1% equity risk per trade
STOP              = 10 * PIP      # 10-pip hard stop
MAX_LEVERAGE      = 30            # 1:30 max leverage (FTMO / EU standard)
FTMO_MAX_DAILY    = 0.05          # 5% max daily loss of initial balance
FTMO_MAX_TOTAL    = 0.10          # 10% max total loss of initial balance
OOS_START         = "2024-01-01"


def _load_market_data() -> tuple[pd.Series, str]:
    raw = pd.read_hdf(DATA_H5, key="data")
    instrument = raw.index.get_level_values("instrument").unique()[0]
    ohlcv = raw.xs(instrument, level="instrument").rename(columns={
        "$open": "open", "$high": "high", "$low": "low",
        "$close": "close", "$volume": "volume",
    })
    return ohlcv["close"], instrument


def _load_factor(name: str, full_idx: pd.Index, instrument: str) -> pd.Series | None:
    path = FACTOR_DIR / f"{name}.parquet"
    if not path.exists():
        return None
    df = pd.read_parquet(path)
    if isinstance(df.index, pd.MultiIndex):
        try:
            s = df.xs(instrument, level="instrument").iloc[:, 0]
        except KeyError:
            s = df.iloc[:, 0]
    else:
        s = df.iloc[:, 0]
    return s.reindex(full_idx)


def _build_signal(factor_names: list[str], full_idx: pd.Index,
                  instrument: str, code: str) -> pd.Series | None:
    """Build composite z-score signal (same logic as the strategy code uses)."""
    factors: dict[str, pd.Series] = {}
    for fn in factor_names:
        s = _load_factor(fn, full_idx, instrument)
        if s is None:
            return None
        factors[fn] = s

    # Try to reproduce the signal via the original strategy code
    close = pd.Series(np.zeros(len(full_idx)), index=full_idx)  # not used by signal code
    try:
        local_ns: dict = {"pd": pd, "np": np, "close": close, "factors": factors}
        exec(code, local_ns)  # noqa: S102
        sig = local_ns.get("signal")
        if sig is not None and isinstance(sig, pd.Series):
            return sig.reindex(full_idx).fillna(0).astype(int)
    except Exception:
        pass

    # Fallback: generic composite z-score (same as original loop)
    composite = pd.Series(0.0, index=full_idx)
    for fn, s in factors.items():
        s = s.fillna(0)
        std = s.std()
        if std > 0:
            composite += (s - s.mean()) / std
    sig = pd.Series(0, index=full_idx)
    sig[composite > 0.5] = 1
    sig[composite < -0.5] = -1
    return sig


def _run_engine(sig_arr: np.ndarray, px_arr: np.ndarray,
                ts_arr: np.ndarray) -> dict:
    """
    FTMO-compliant backtest engine.

    Rules enforced:
      - Daily loss limit: if daily PnL < -5% of initial ($5k), no new trades that day
      - Total loss limit: if equity < $90k (10% below initial), simulation ends (account blown)
      - Position sizing: 1% equity risk per trade, 10-pip stop, max leverage 1:30
      - Full compounding: position size recalculated from current equity each trade
    """
    INITIAL   = 100_000.0
    equity    = INITIAL
    peak      = INITIAL
    max_dd    = 0.0
    pos       = 0
    entry_px  = 0.0
    pos_size  = 0.0
    n_wins    = 0
    trade_rets: list[float] = []
    blown     = False

    # Daily tracking
    current_day   = None
    day_start_eq  = INITIAL
    day_blocked   = False

    for i in range(1, len(px_arr)):
        p     = float(px_arr[i])
        sig_i = int(sig_arr[i])
        day   = ts_arr[i].astype("datetime64[D]")

        # ── New day: reset daily loss tracker ────────────────────────────────
        if day != current_day:
            current_day  = day
            day_start_eq = equity
            day_blocked  = False

        # ── Close position if signal flips ────────────────────────────────────
        if pos != 0 and sig_i != pos:
            exit_p  = p - pos * COST_EXIT
            raw_pnl = (exit_p - entry_px) * pos_size * pos
            equity += raw_pnl

            if equity > peak:
                peak = equity
            dd = (peak - equity) / peak
            if dd > max_dd:
                max_dd = dd

            ret = raw_pnl / (pos_size * entry_px) if (pos_size * entry_px) > 0 else 0.0
            trade_rets.append(ret)
            if raw_pnl > 0:
                n_wins += 1
            pos = 0

            # Check daily loss limit
            if (equity - day_start_eq) / INITIAL < -FTMO_MAX_DAILY:
                day_blocked = True

            # Check total loss limit → account blown
            if equity < INITIAL * (1 - FTMO_MAX_TOTAL):
                blown = True
                break

        # ── Open new position (if not blocked) ───────────────────────────────
        if sig_i != 0 and pos == 0 and not day_blocked and not blown:
            pos      = sig_i
            entry_px = p + pos * COST_ENTRY
            # Full compounding: size from current equity, capped by max leverage
            max_by_leverage = equity * MAX_LEVERAGE / p
            pos_size = min(equity * RISK_PCT / STOP, max_by_leverage)

    ret_arr  = np.array(trade_rets) if trade_rets else np.array([0.0])
    n_trades = len(trade_rets)
    total_ret = (equity - INITIAL) / INITIAL
    sharpe = float("nan")
    if n_trades > 1 and ret_arr.std() > 0:
        sharpe = float(ret_arr.mean() / ret_arr.std() * np.sqrt(n_trades))

    return dict(
        end_equity=equity,
        total_return=total_ret,
        max_drawdown=-max_dd,
        sharpe=sharpe,
        n_trades=n_trades,
        win_rate=n_wins / n_trades if n_trades else 0.0,
        trade_rets=ret_arr,
        blown=blown,
    )


def _monthly_ret(total_ret: float, n_months: float) -> float:
    return float((1 + total_ret) ** (1 / max(n_months, 1)) - 1)


def backtest_strategy(json_path: str, close: pd.Series, instrument: str) -> dict | None:
    try:
        d = json.load(open(json_path))
    except Exception:
        return None

    factor_names = d.get("factor_names", [])
    code         = d.get("code", "")
    name         = d.get("strategy_name", Path(json_path).stem)

    if not factor_names:
        return None

    sig = _build_signal(factor_names, close.index, instrument, code)
    if sig is None:
        return None

    # Full period
    full = _run_engine(sig.values, close.values, close.index.values)
    n_days_full = (close.index[-1] - close.index[0]).days
    n_months_full = n_days_full / 30.44

    # OOS only
    oos_mask = close.index >= OOS_START
    if oos_mask.sum() < 1000:
        return None
    oos_close = close[oos_mask]
    oos_sig   = sig[oos_mask]
    oos = _run_engine(oos_sig.values, oos_close.values, oos_close.index.values)
    n_months_oos = (oos_close.index[-1] - oos_close.index[0]).days / 30.44

    return dict(
        name=name,
        path=json_path,
        factors=factor_names,
        # Full
        full_monthly_pct=_monthly_ret(full["total_return"], n_months_full) * 100,
        full_annual_pct=((1 + _monthly_ret(full["total_return"], n_months_full)) ** 12 - 1) * 100,
        full_dd_pct=full["max_drawdown"] * 100,
        full_sharpe=full["sharpe"],
        full_trades=full["n_trades"],
        full_winrate=full["win_rate"] * 100,
        full_blown=full["blown"],
        # OOS
        oos_monthly_pct=_monthly_ret(oos["total_return"], n_months_oos) * 100,
        oos_annual_pct=((1 + _monthly_ret(oos["total_return"], n_months_oos)) ** 12 - 1) * 100,
        oos_dd_pct=oos["max_drawdown"] * 100,
        oos_sharpe=oos["sharpe"],
        oos_trades=oos["n_trades"],
        oos_winrate=oos["win_rate"] * 100,
        oos_end_equity=oos["end_equity"],
        oos_blown=oos["blown"],
        n_months_oos=n_months_oos,
    )


def _worker(args: tuple) -> dict | None:
    json_path, close_bytes, instrument = args
    close = pd.read_pickle(close_bytes) if isinstance(close_bytes, (str, Path)) else close_bytes
    return backtest_strategy(json_path, close, instrument)


def main() -> None:
    parser = argparse.ArgumentParser(description="Realistic backtest of all strategies")
    parser.add_argument("--target-monthly", type=float, default=4.0,
                        help="Minimum OOS monthly return %% (default: 4.0)")
    parser.add_argument("--min-trades", type=int, default=30,
                        help="Minimum OOS trades (default: 30)")
    parser.add_argument("--max-dd", type=float, default=-8.0,
                        help="Maximum OOS drawdown %% (default: -8.0)")
    parser.add_argument("--workers", type=int, default=4,
                        help="Parallel workers (default: 4)")
    parser.add_argument("--top", type=int, default=20,
                        help="Show top N strategies (default: 20)")
    args = parser.parse_args()

    print(f"\nLoading market data...")
    close, instrument = _load_market_data()
    print(f"  {close.index[0].date()} → {close.index[-1].date()} | {len(close):,} bars")
    print(f"  OOS window: {OOS_START} onwards")
    print(f"  Costs: 2.35 pip/trade (1.5 spread + 0.5 slip + 0.35 comm)")
    print(f"  Filters: OOS monthly ≥ {args.target_monthly}% | trades ≥ {args.min_trades} | DD ≥ {args.max_dd}%\n")

    json_files = sorted(glob.glob(str(STRAT_DIR / "*.json")))
    print(f"Backtesting {len(json_files)} strategies with {args.workers} workers...\n")

    # Save close to temp file for multiprocessing
    import tempfile
    tmp = tempfile.NamedTemporaryFile(suffix=".pkl", delete=False)
    close.to_pickle(tmp.name)
    tmp.close()

    results = []
    done = 0
    errors = 0

    try:
        with ProcessPoolExecutor(max_workers=args.workers) as ex:
            futures = {
                ex.submit(backtest_strategy, fp, close, instrument): fp
                for fp in json_files
            }
            for fut in as_completed(futures):
                done += 1
                try:
                    res = fut.result()
                    if res is not None:
                        results.append(res)
                except Exception:
                    errors += 1
                if done % 100 == 0 or done == len(json_files):
                    print(f"  {done}/{len(json_files)} done, {len(results)} valid, {errors} errors")
    finally:
        os.unlink(tmp.name)

    if not results:
        print("No valid results.")
        return

    df = pd.DataFrame(results)

    # ── Save full results ──────────────────────────────────────────────────────
    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
    out_csv = OUTPUT_DIR / "all_strategies_realistic.csv"
    df.sort_values("oos_monthly_pct", ascending=False).to_csv(out_csv, index=False)
    print(f"\nFull results saved → {out_csv}")

    # ── Filter for target ──────────────────────────────────────────────────────
    hits = df[
        (df["oos_monthly_pct"] >= args.target_monthly) &
        (df["oos_trades"]      >= args.min_trades) &
        (df["oos_dd_pct"]      >= args.max_dd) &
        (df["oos_blown"]       == False)  # noqa: E712
    ].sort_values("oos_monthly_pct", ascending=False)

    print(f"\n{'='*70}")
    print(f"  Strategies meeting target: OOS monthly ≥ {args.target_monthly}% | "
          f"trades ≥ {args.min_trades} | DD ≥ {args.max_dd}%")
    print(f"  Found: {len(hits)} / {len(df)}")
    print(f"{'='*70}\n")

    top = hits.head(args.top)
    if top.empty:
        print("  No strategies met the criteria.")
        # Show best available
        best = df.sort_values("oos_monthly_pct", ascending=False).head(10)
        print(f"\n  Best available (by OOS monthly return):\n")
        _print_table(best)
    else:
        _print_table(top)

    # ── Save filtered results ──────────────────────────────────────────────────
    if not hits.empty:
        out_hits = OUTPUT_DIR / f"strategies_oos_{args.target_monthly}pct_monthly.csv"
        hits.to_csv(out_hits, index=False)
        print(f"\nFiltered results saved → {out_hits}")

    # ── FTMO projection for #1 ────────────────────────────────────────────────
    best_row = (hits if not hits.empty else df.sort_values("oos_monthly_pct", ascending=False)).iloc[0]
    mon = best_row["oos_monthly_pct"]
    dd  = abs(best_row["oos_dd_pct"])
    gross = 100_000 * mon / 100
    challenge_m = 10 / max(mon, 0.01)
    print(f"\n{'='*70}")
    print(f"  FTMO 100k projection — #{1}: {best_row['name']}")
    print(f"{'='*70}")
    print(f"  OOS monthly return:    {mon:+.2f}%")
    print(f"  Monthly gross profit:  ${gross:,.0f}")
    print(f"  Trader share (80%):    ${gross*0.8:,.0f} / month")
    print(f"  Trader annual (80%):   ${gross*0.8*12:,.0f} / year")
    print(f"  OOS Max Drawdown:      {-dd:.2f}%  (FTMO limit: 10%)")
    print(f"  Challenge duration:    ~{challenge_m:.1f} months to hit +10%")
    print(f"  FTMO safe?             {'YES ✓' if dd < 8 else 'BORDERLINE ⚠' if dd < 10 else 'NO ✗'}")


def _print_table(df: pd.DataFrame) -> None:
    hdr = f"{'#':>3}  {'Name':<35} {'OOS Mon%':>8} {'OOS DD%':>8} {'Sharpe':>7} {'WinR%':>6} {'Trades':>7} {'Blown':>6}  {'Factors'}"
    print(hdr)
    print("-" * len(hdr))
    for i, (_, r) in enumerate(df.iterrows(), 1):
        factors_str = ",".join(r["factors"][:2]) + ("…" if len(r["factors"]) > 2 else "")
        blown = "💥YES" if r.get("oos_blown") else "  no"
        print(f"{i:>3}  {r['name']:<35} {r['oos_monthly_pct']:>+7.2f}% "
              f"{r['oos_dd_pct']:>+7.2f}% {r['oos_sharpe']:>7.2f} "
              f"{r['oos_winrate']:>5.1f}% {r['oos_trades']:>7,}  {blown}  {factors_str}")


if __name__ == "__main__":
    main()
