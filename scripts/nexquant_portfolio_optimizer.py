#!/usr/bin/env python3
"""Portfolio Optimizer — combine uncorrelated strategies for 15% monthly target.

Given N strategies with daily returns, find the optimal combination that:
- Maximizes monthly return
- Keeps max drawdown within FTMO limits (10% total, 5% daily)
- Diversifies across uncorrelated strategies
"""

import json
import os
from pathlib import Path

import numpy as np
import pandas as pd

PROJECT = Path(__file__).resolve().parent.parent
RESULTS_DIR = PROJECT / "results" / "strategies_new"
STRATEGIES_DIR = PROJECT / "results" / "strategies"
FACTORS_DIR = PROJECT / "results" / "factors"
VALUES_DIR = FACTORS_DIR / "values"
OHLCV_PATH = Path(os.getenv("PREDIX_OHLCV_PATH",
    str(PROJECT / "git_ignore_folder" / "intraday_pv_all.h5")))

TARGET_MONTHLY = 15.0
MAX_DD = 0.10       # FTMO: 10% max total drawdown
MAX_DAILY_DD = 0.05  # FTMO: 5% max daily drawdown
MIN_TRADES = 30
MIN_SHARPE = 0.5


def load_strategies() -> list[dict]:
    """Load all strategy JSONs with real (non-fabricated) verified metrics."""
    strategies = []
    seen = set()
    for d in (STRATEGIES_DIR, RESULTS_DIR):
        if not d.exists():
            continue
        for p in d.glob("*.json"):
            try:
                r = json.loads(p.read_text())
            except Exception:
                continue
            if not isinstance(r, dict):
                continue
            name = r.get("strategy_name", p.stem)
            if name in seen:
                continue
            seen.add(name)

            s = r.get("summary", {})
            if not isinstance(s, dict):
                s = {}
            m = r.get("metrics", {})
            if not isinstance(m, dict):
                m = {}

            # Extract metrics (prefer summary, fallback to metrics)
            sharpe = float(s.get("sharpe") or m.get("sharpe") or 0)
            mon_pct = float(s.get("monthly_return_pct") or s.get("oos_monthly_return_pct")
                         or m.get("monthly_return_pct") or 0)
            max_dd = float(s.get("max_drawdown") or s.get("oos_max_drawdown")
                        or m.get("max_drawdown") or 0)
            win_rate = float(s.get("win_rate") or s.get("oos_win_rate")
                          or m.get("win_rate") or 0)
            n_trades = int(s.get("n_trades") or s.get("oos_n_trades")
                        or s.get("real_n_trades") or m.get("n_trades") or 0)
            total_ret = float(s.get("total_return") or m.get("total_return") or 0)

            # Filter fabricated
            if mon_pct == 200 and sharpe == 3.0 and abs(max_dd + 0.167) < 0.01:
                continue
            if mon_pct == -20 and max_dd == -1.0:
                continue
            if sharpe == 200:
                continue

            # Filter quality
            if n_trades < MIN_TRADES or sharpe < MIN_SHARPE:
                continue
            if mon_pct <= 0:
                continue

            strategies.append({
                "name": name,
                "file": str(p),
                "sharpe": sharpe,
                "monthly_pct": mon_pct,
                "max_dd": max_dd,
                "win_rate": win_rate,
                "n_trades": n_trades,
                "total_return": total_ret,
                "factors": r.get("factor_names") or r.get("factors_used") or [],
                "code": r.get("code", ""),
            })

    return strategies


def load_strategy_returns(strategy: dict, close_daily: pd.Series) -> pd.Series | None:
    """Reconstruct daily strategy returns from code and factor data."""
    code = strategy.get("code", "")
    if not code:
        return None

    factors_list = strategy.get("factors", [])
    if not factors_list:
        return None

    # Load factor values
    factor_series = {}
    for fname in factors_list:
        safe = str(fname).replace("/", "_").replace("\\", "_").replace(" ", "_")[:150]
        parq = VALUES_DIR / f"{safe}.parquet"
        if not parq.exists():
            continue
        try:
            s = pd.read_parquet(str(parq))
            if isinstance(s.index, pd.MultiIndex):
                s = s.xs("EURUSD", level="instrument")[s.columns[0]]
            # Align to close_daily index
            s = s.resample("D").last().reindex(close_daily.index).ffill(limit=5)
            factor_series[fname] = s
        except Exception:
            continue

    if len(factor_series) < 2:
        return None

    df_factors = pd.DataFrame(factor_series).dropna()
    if len(df_factors) < 100:
        return None

    # Execute strategy code on daily data
    local_vars = {"factors": df_factors, "close": close_daily.reindex(df_factors.index)}
    try:
        exec(code, {"np": np, "pd": pd, "numpy": np}, local_vars)
    except Exception:
        # Can't execute — use simple IC-weighted signal as fallback
        return None

    signal = local_vars.get("signal")
    if signal is None or not isinstance(signal, pd.Series):
        return None

    # Compute daily returns from signal
    common = close_daily.index.intersection(signal.index)
    c = close_daily.loc[common]
    s = signal.loc[common].clip(-1, 1).fillna(0)

    fwd_ret = c.pct_change().shift(-1)
    strat_ret = s.shift(1) * fwd_ret
    strat_ret = strat_ret.dropna()

    if len(strat_ret) < 30:
        return None

    return strat_ret


def build_simple_signal(factors_list: list[str], close_daily: pd.Series) -> tuple[pd.Series, pd.Series]:
    """Build simple IC-weighted daily signal (fallback when code fails)."""
    import json as _json

    factor_series = {}
    ic_values = {}
    for fname in factors_list:
        safe = str(fname).replace("/", "_").replace("\\", "_").replace(" ", "_")[:150]
        parq = VALUES_DIR / f"{safe}.parquet"
        jf = FACTORS_DIR / f"{safe}.json"
        if not parq.exists():
            continue
        ic = 0.0
        if jf.exists():
            ic = float(_json.loads(jf.read_text()).get("ic", 0))
        try:
            s = pd.read_parquet(str(parq))
            if isinstance(s.index, pd.MultiIndex):
                s = s.xs("EURUSD", level="instrument")[s.columns[0]]
            s = s.resample("D").last().reindex(close_daily.index).ffill(limit=5)
            factor_series[fname] = s
            ic_values[fname] = ic
        except Exception:
            continue

    df = pd.DataFrame(factor_series).dropna()
    if len(df) < 50:
        return pd.Series(), pd.Series()

    # z-score composite
    window = 20
    z = (df - df.rolling(window).mean()) / (df.rolling(window).std() + 1e-8)

    composite = pd.Series(0.0, index=df.index)
    total_ic = sum(abs(v) for v in ic_values.values())
    if total_ic == 0:
        total_ic = 1.0
    for col in df.columns:
        ic = ic_values.get(col, 0)
        w = abs(ic) / total_ic
        sign = -1 if ic < 0 else 1
        composite += sign * w * z[col]

    signal = pd.Series(0, index=df.index)
    signal[composite > 0.5] = 1
    signal[composite < -0.5] = -1

    # Compute returns
    common = close_daily.index.intersection(signal.index)
    c = close_daily.loc[common]
    s = signal.loc[common].clip(-1, 1).fillna(0)
    fwd_ret = c.pct_change().shift(-1)
    strat_ret = s.shift(1) * fwd_ret
    return signal, strat_ret.dropna()


def compute_portfolio_metrics(returns: list[pd.Series], weights: list[float],
                              close_daily: pd.Series) -> dict:
    """Compute portfolio-level metrics from weighted strategy returns."""
    if not returns:
        return {"monthly_pct": 0, "max_dd": 0, "sharpe": 0}

    # Align all return series
    common_idx = returns[0].index
    for r in returns[1:]:
        common_idx = common_idx.intersection(r.index)
    if len(common_idx) < 50:
        return {"monthly_pct": 0, "max_dd": 0, "sharpe": 0}

    aligned = pd.DataFrame({i: r.loc[common_idx] for i, r in enumerate(returns)}).dropna()
    if len(aligned) < 30:
        return {"monthly_pct": 0, "max_dd": 0, "sharpe": 0}

    # Weighted portfolio return
    port_ret = pd.Series(0.0, index=aligned.index)
    for i in range(len(returns)):
        port_ret += weights[i] * aligned[i]

    # Equity curve
    eq = (1 + port_ret).cumprod()
    peak = eq.cummax()
    max_dd = float(((eq - peak) / peak).min())

    total_ret = float(eq.iloc[-1] - 1)
    n_days = (port_ret.index[-1] - port_ret.index[0]).days
    n_months = max(n_days / 30.44, 1)
    monthly = float((1 + total_ret) ** (1 / n_months) - 1)

    sharpe = float(port_ret.mean() / port_ret.std() * np.sqrt(252)) if port_ret.std() > 0 else 0
    daily_dd = float(port_ret.min())  # Worst daily return

    return {
        "monthly_pct": monthly * 100,
        "max_dd": max_dd,
        "sharpe": sharpe,
        "daily_worst": daily_dd,
        "n_days": len(port_ret),
        "n_months": n_months,
    }


def main():
    print("=" * 60)
    print("  Portfolio Optimizer — 15% Monthly Target")
    print("=" * 60)

    # Load OHLCV daily
    print("\nLoading data...")
    df = pd.read_hdf(OHLCV_PATH, key="data")
    close = df.xs("EURUSD", level="instrument")["$close"].sort_index()
    close_daily = close.resample("D").last().dropna()
    print(f"  Daily bars: {len(close_daily)}")

    # Load strategies
    strategies = load_strategies()
    print(f"  Real strategies: {len(strategies)}")

    # Build daily returns for each strategy
    print("\nBuilding strategy returns...")
    strat_returns = []
    strat_names = []
    for s in strategies[:50]:  # Limit to top 50 for speed
        rets = load_strategy_returns(s, close_daily)
        if rets is None or len(rets) < 30:
            # Use simple signal as fallback
            _, rets = build_simple_signal(s["factors"], close_daily)
        if rets is not None and len(rets) >= 30:
            strat_returns.append(rets)
            strat_names.append(s["name"])
            print(f"  [{len(strat_returns)}] {s['name'][:40]:40s} "
                  f"Sh={s['sharpe']:.1f} Mon={s['monthly_pct']:.1f}% Tr={s['n_trades']}")

    if len(strat_returns) < 2:
        print("\n  Not enough valid strategies.")
        return

    print(f"\n  Valid return series: {len(strat_returns)}")

    # Find best portfolio via greedy selection (low correlation, high return)
    print("\n--- Greedy Portfolio Selection ---")
    print(f"  Target: {TARGET_MONTHLY}% monthly | Max DD: {MAX_DD:.0%} | Max Daily DD: {MAX_DAILY_DD:.0%}")
    print()

    # Compute individual metrics
    individual = []
    for i, (rets, name) in enumerate(zip(strat_returns, strat_names)):
        eq = (1 + rets).cumprod()
        dd = float(((eq - eq.cummax()) / eq.cummax()).min())
        total = float(eq.iloc[-1] - 1)
        n = max((rets.index[-1] - rets.index[0]).days / 30.44, 1)
        mon = float((1 + total) ** (1 / n) - 1) * 100
        individual.append({"idx": i, "name": name, "monthly": mon, "dd": dd, "n": len(rets)})

    individual.sort(key=lambda x: x["monthly"], reverse=True)

    # Greedy: add strategies one by one if they don't increase correlation too much
    selected = []
    selected_rets = []

    for s in individual:
        if len(selected) >= 8:
            break
        # Check correlation with existing portfolio
        new_ret = strat_returns[s["idx"]]
        if selected_rets:
            common = new_ret.index
            for r in selected_rets:
                common = common.intersection(r.index)
            if len(common) < 30:
                continue
            cors = []
            for r in selected_rets:
                aligned_new = new_ret.loc[common]
                aligned_r = r.loc[common]
                if len(aligned_new) >= 30:
                    cors.append(abs(aligned_new.corr(aligned_r)))
            if cors and max(cors) > 0.5:
                print(f"  SKIP {s['name'][:40]} (max_corr={max(cors):.2f})")
                continue

        selected.append(s)
        selected_rets.append(new_ret)
        print(f"  ADD  {s['name'][:40]:40s} Mon={s['monthly']:+.1f}% DD={s['dd']:.3f} corr<0.5")

    # Evaluate portfolio
    if len(selected) >= 2:
        print(f"\n  Portfolio: {len(selected)} strategies")
        weights = [1.0 / len(selected)] * len(selected)
        rets = [strat_returns[s["idx"]] for s in selected]
        pm = compute_portfolio_metrics(rets, weights, close_daily)

        print(f"  Equal-weight metrics:")
        print(f"    Monthly return: {pm['monthly_pct']:.2f}%")
        print(f"    Max drawdown:   {pm['max_dd']:.3f}")
        print(f"    Sharpe:         {pm['sharpe']:.2f}")
        print(f"    Worst day:      {pm['daily_worst']:.3%}")
        print(f"    Period:         {pm['n_months']:.1f} months ({pm['n_days']} days)")

        # Leverage scaling
        max_safe_lev = min(
            MAX_DD / abs(pm["max_dd"]) if pm["max_dd"] != 0 else 30,
            MAX_DAILY_DD / abs(pm["daily_worst"]) if pm["daily_worst"] != 0 else 30,
            30,
        )
        leveraged_monthly = pm["monthly_pct"] * max_safe_lev
        print(f"\n  Max safe leverage: {max_safe_lev:.1f}× (limited by max DD {MAX_DD:.0%})")
        print(f"  Leveraged monthly:  {leveraged_monthly:.1f}%")

        if leveraged_monthly >= TARGET_MONTHLY:
            print(f"\n  ✓ MEETS TARGET! {leveraged_monthly:.1f}% ≥ {TARGET_MONTHLY}%")
        else:
            gap = TARGET_MONTHLY - leveraged_monthly
            needed_strategies = int(np.ceil(len(selected) * TARGET_MONTHLY / max(leveraged_monthly, 0.1)))
            print(f"\n  ✗ Below target. Need ~{needed_strategies} strategies or {TARGET_MONTHLY/max(pm['monthly_pct'],0.01):.1f}× better monthly.")

    # Save portfolio config
    out = {
        "target_monthly": TARGET_MONTHLY,
        "selected": [{"name": s["name"], "monthly": s["monthly"], "dd": s["dd"]} for s in selected],
        "portfolio": pm if len(selected) >= 2 else {},
    }
    out_path = RESULTS_DIR / "portfolio_config.json"
    out_path.write_text(json.dumps(out, indent=2, default=str))
    print(f"\n  Saved → {out_path}")


if __name__ == "__main__":
    main()
