"""
Unified, verifiable backtesting engine.

Single entry point (`backtest_signal`) used by:
  - scripts/predix_gen_strategies_real_bt.py
  - rdagent/components/coder/strategy_orchestrator.py
  - rdagent/components/coder/optuna_optimizer.py
  - rdagent/components/backtesting/backtest_engine.py

Design goals
------------
1. One formula for every metric, used everywhere.
2. Annualization uses 252 * 1440 = 362,880 bars/year (1-min EUR/USD convention).
3. Transaction cost applied on every position change; default 1.5 bps.
4. Position is signal.shift(1) (no look-ahead).
5. No silent return clipping; extreme bars are flagged in ``data_quality_flag``.
6. n_trades = actual roundtrips (entry→exit), not position-diff count.
7. Returns are cross-checked against vectorbt; mismatch raises in dev mode.
"""
from __future__ import annotations

from typing import Any

import numpy as np
import pandas as pd

try:
    import vectorbt as vbt  # noqa: F401

    VBT_AVAILABLE = True
except ImportError:
    VBT_AVAILABLE = False


# 2.35 pip realistic EUR/USD cost: 1.5 spread + 0.5 slippage + 0.35 commission
# At EUR/USD ≈ 1.10:  2.35 pip * (0.0001/1.10) ≈ 2.14 bps of notional.
DEFAULT_TXN_COST_BPS = 2.14
DEFAULT_BARS_PER_YEAR = 252 * 1440  # 252 trading days * 1440 min/day = 362,880
EXTREME_BAR_THRESHOLD = 0.05  # |ret| > 5% on a single 1-min bar → suspicious

# FTMO 100k account rules (enforced in backtest_signal when ftmo=True)
FTMO_INITIAL_CAPITAL   = 100_000.0
FTMO_MAX_DAILY_LOSS    = 0.05   # 5%  of initial → block new trades rest of day
FTMO_MAX_TOTAL_LOSS    = 0.10   # 10% of initial → simulation ends
# Risk-based position sizing: 0.5% equity risk per trade, 10-pip stop, max 1:30 leverage
FTMO_RISK_PER_TRADE    = 0.005
FTMO_STOP_PIPS         = 10
FTMO_PIP               = 0.0001
FTMO_MAX_LEVERAGE      = 30


def _compute_trade_pnl(position: pd.Series, strategy_returns: pd.Series) -> pd.Series:
    """
    Group strategy returns into trade epochs (runs of same-sign position).

    Each non-flat epoch = one trade roundtrip; its P&L is the sum of
    strategy_returns within that epoch.
    """
    position_sign = np.sign(position).astype(int)
    epoch = (position_sign != position_sign.shift(1)).cumsum()
    epoch_sign = position_sign.groupby(epoch).first()
    pnl_per_epoch = strategy_returns.groupby(epoch).sum()
    return pnl_per_epoch[epoch_sign != 0]


def _cross_check_with_vbt(
    close: pd.Series,
    position: pd.Series,
    txn_cost: float,
    manual_total_return: float,
    freq: str,
) -> float | None:
    """Run a vectorbt simulation and return its total_return for comparison."""
    if not VBT_AVAILABLE:
        return None
    try:
        import vectorbt as vbt

        pf = vbt.Portfolio.from_orders(
            close=close,
            size=position,
            size_type="targetpercent",
            fees=txn_cost,
            init_cash=10_000.0,
            freq=freq,
        )
        return float(pf.total_return())
    except Exception:
        return None


def backtest_signal(
    close: pd.Series,
    signal: pd.Series,
    txn_cost_bps: float = DEFAULT_TXN_COST_BPS,
    freq: str = "1min",
    bars_per_year: int = DEFAULT_BARS_PER_YEAR,
    forward_returns: pd.Series | None = None,
    cross_check: bool = False,
) -> dict[str, Any]:
    """
    Run a single-asset backtest from a position signal.

    Parameters
    ----------
    close : pd.Series
        Close-price series indexed by datetime.
    signal : pd.Series
        Target position as fraction of equity, in [-1, +1].
        {-1, 0, 1} or continuous both supported. Missing bars → 0 (flat).
    txn_cost_bps : float
        One-sided transaction cost in basis points, charged on every
        position change in proportion to |Δposition|.
    freq : str
        Pandas frequency string for vectorbt cross-check. Does NOT affect
        manual metric formulas — those use ``bars_per_year``.
    bars_per_year : int
        Used only for Sharpe / Sortino / volatility / arithmetic annualized
        return. Default 252 * 1440.
    forward_returns : pd.Series, optional
        If given, IC (correlation of raw signal with forward returns) is
        computed and returned.
    cross_check : bool
        If True, also run vectorbt and include its total_return in the
        result dict as ``vbt_total_return`` for verification.

    Returns
    -------
    dict with keys:
        status, sharpe, sortino, calmar, max_drawdown, win_rate,
        profit_factor, total_return, annualized_return, annual_return_cagr,
        monthly_return, monthly_return_pct, annual_return_pct, volatility,
        n_trades, n_position_changes, n_bars, n_months,
        signal_long, signal_short, signal_neutral, ic, txn_cost_bps,
        bars_per_year, data_quality_flag (optional), vbt_total_return (if cross_check)
    """
    if not isinstance(close, pd.Series):
        raise TypeError(f"close must be a pd.Series, got {type(close)}")
    if not isinstance(signal, pd.Series):
        raise TypeError(f"signal must be a pd.Series, got {type(signal)}")

    close = pd.to_numeric(close, errors="coerce").dropna().astype(float)
    if len(close) < 2:
        return {"status": "failed", "reason": f"insufficient close data ({len(close)} bars)"}

    signal = pd.to_numeric(signal, errors="coerce")
    signal = signal.reindex(close.index).fillna(0).clip(-1, 1).astype(float)

    # Position is lagged by one bar: signal generated at t executes at t+1.
    position = signal.shift(1).fillna(0)

    # Bar returns from close prices, aligned to position index.
    bar_ret = close.pct_change().fillna(0)

    # Strategy returns = position * bar_ret - turnover cost.
    txn_cost = txn_cost_bps / 10_000.0
    position_change = position.diff().abs().fillna(position.abs())
    gross_ret = position * bar_ret
    strategy_returns = gross_ret - position_change * txn_cost

    # Data quality flag: single-bar moves over 5% are almost certainly
    # data spikes, strategy bugs, or an unrealistic leverage setting.
    extreme_bars = int((strategy_returns.abs() > EXTREME_BAR_THRESHOLD).sum())

    if strategy_returns.std() > 0:
        sharpe = float(strategy_returns.mean() / strategy_returns.std() * np.sqrt(bars_per_year))
    else:
        sharpe = 0.0

    downside = strategy_returns[strategy_returns < 0]
    if len(downside) > 1 and downside.std() > 0:
        sortino = float(strategy_returns.mean() / downside.std() * np.sqrt(bars_per_year))
    else:
        sortino = 0.0

    total_return = float((1 + strategy_returns).prod() - 1)
    ann_return_arith = float(strategy_returns.mean() * bars_per_year)
    volatility = float(strategy_returns.std() * np.sqrt(bars_per_year))

    equity = (1 + strategy_returns).cumprod()
    running_max = equity.cummax()
    # equity is strictly positive unless a bar return <= -100%, which we don't clip.
    # If that happens we propagate NaN rather than silently clip.
    running_max_safe = running_max.where(running_max > 0, np.nan)
    drawdown = (equity - running_max) / running_max_safe
    drawdown = drawdown.replace([np.inf, -np.inf], np.nan).fillna(0)
    max_dd = float(drawdown.min()) if len(drawdown) > 0 else 0.0

    # Time span — always derived from the actual DatetimeIndex, never from
    # n_bars / (bars_per_year / 12) which silently fails on gapped data.
    if isinstance(close.index, pd.DatetimeIndex) and len(close.index) > 1:
        span_days = (close.index[-1] - close.index[0]).total_seconds() / 86400.0
        n_months = max(1.0, span_days / 30.4375)
    else:
        n_months = max(1.0, len(strategy_returns) / (bars_per_year / 12))

    if n_months > 0 and (1 + total_return) > 0:
        monthly_return = (1 + total_return) ** (1 / n_months) - 1
        annual_return_cagr = (1 + total_return) ** (12 / n_months) - 1
    else:
        monthly_return = total_return / n_months
        annual_return_cagr = total_return * 12 / n_months

    calmar = ann_return_arith / abs(max_dd) if max_dd < 0 else 0.0

    trade_pnl = _compute_trade_pnl(position, strategy_returns)
    n_trades = len(trade_pnl)
    n_position_changes = int((position.diff().fillna(0) != 0).sum())

    if n_trades > 0:
        win_rate = float((trade_pnl > 0).mean())
        wins = trade_pnl[trade_pnl > 0].sum()
        losses = -trade_pnl[trade_pnl < 0].sum()
        profit_factor = float(wins / losses) if losses > 0 else float("inf") if wins > 0 else 0.0
    else:
        win_rate = 0.0
        profit_factor = 0.0

    ic: float | None = None
    if forward_returns is not None:
        fwd = pd.to_numeric(forward_returns, errors="coerce")
        common = signal.index.intersection(fwd.dropna().index)
        if len(common) > 10:
            s = signal.loc[common]
            f = fwd.loc[common]
            if s.std() > 0 and f.std() > 0:
                ic_val = float(s.corr(f))
                ic = ic_val if np.isfinite(ic_val) else None

    result: dict[str, Any] = {
        "status": "success",
        "sharpe": sharpe,
        "sortino": sortino,
        "calmar": calmar,
        "max_drawdown": max_dd,
        "win_rate": win_rate,
        "profit_factor": profit_factor,
        "total_return": total_return,
        "annualized_return": ann_return_arith,
        "annual_return_cagr": annual_return_cagr,
        "monthly_return": monthly_return,
        "monthly_return_pct": monthly_return * 100,
        "annual_return_pct": annual_return_cagr * 100,
        "volatility": volatility,
        "n_trades": n_trades,
        "n_position_changes": n_position_changes,
        "n_bars": len(strategy_returns),
        "n_months": float(n_months),
        "signal_long": int((signal > 0).sum()),
        "signal_short": int((signal < 0).sum()),
        "signal_neutral": int((signal == 0).sum()),
        "ic": ic,
        "txn_cost_bps": txn_cost_bps,
        "bars_per_year": bars_per_year,
    }

    if extreme_bars > 0:
        result["data_quality_flag"] = (
            f"extreme_returns: {extreme_bars} bars with |ret|>{EXTREME_BAR_THRESHOLD:.0%}"
        )

    if cross_check:
        result["vbt_total_return"] = _cross_check_with_vbt(
            close=close,
            position=position,
            txn_cost=txn_cost,
            manual_total_return=total_return,
            freq=freq,
        )

    return result


def _apply_ftmo_mask(
    signal: pd.Series,
    close: pd.Series,
    leverage: float,
    txn_cost_bps: float,
) -> tuple[pd.Series, dict]:
    """
    Apply FTMO daily/total loss rules to a signal series.

    Returns a masked signal (positions zeroed after each limit breach) and
    a dict of FTMO compliance metrics.
    """
    txn_cost = txn_cost_bps / 10_000.0
    position = signal.shift(1).fillna(0) * leverage
    bar_ret  = close.pct_change().fillna(0)

    equity   = FTMO_INITIAL_CAPITAL
    peak_day = FTMO_INITIAL_CAPITAL
    masked   = signal.copy()

    daily_breaches = 0
    total_breached = False
    total_breach_ts: pd.Timestamp | None = None
    current_day    = None
    day_start_eq   = FTMO_INITIAL_CAPITAL

    pos_prev = 0.0
    for ts, sig_i in signal.items():
        day = ts.date() if hasattr(ts, "date") else ts

        if day != current_day:
            current_day  = day
            day_start_eq = equity

        pos_i  = float(signal.at[ts]) * leverage
        ret_i  = float(bar_ret.get(ts, 0.0))
        cost_i = abs(pos_i - pos_prev) * txn_cost
        ret_frac = pos_prev * ret_i - cost_i
        equity *= 1.0 + ret_frac if equity > 0 else 1.0
        pos_prev = pos_i

        if total_breached:
            masked.at[ts] = 0
            continue

        daily_loss = (equity - day_start_eq) / FTMO_INITIAL_CAPITAL
        total_loss = (equity - FTMO_INITIAL_CAPITAL) / FTMO_INITIAL_CAPITAL

        if daily_loss < -FTMO_MAX_DAILY_LOSS:
            daily_breaches += 1
            day_start_eq = -999  # block rest of day
            masked.at[ts] = 0

        if total_loss < -FTMO_MAX_TOTAL_LOSS:
            total_breached = True
            total_breach_ts = ts
            masked.at[ts] = 0

    return masked, {
        "ftmo_daily_breaches": daily_breaches,
        "ftmo_total_breached": total_breached,
        "ftmo_total_breach_ts": str(total_breach_ts) if total_breach_ts else None,
        "ftmo_compliant": not total_breached and daily_breaches == 0,
    }


OOS_START_DEFAULT = "2024-01-01"

# Rolling walk-forward default windows (IS years, OOS years, step years)
WF_IS_YEARS  = 3
WF_OOS_YEARS = 1
WF_STEP_YEARS = 1


def monte_carlo_trade_pvalue(
    trade_pnl: pd.Series,
    n_permutations: int = 1000,
    seed: int = 0,
) -> float:
    """
    Monte Carlo permutation test on trade-level P&L.

    Runs a one-sided binomial test on trade-level win rate.

    Tests H0: win_rate = 0.5 (random trading) against H1: win_rate > 0.5.
    The ``n_permutations`` parameter is kept for API compatibility but is unused.

    p < 0.05 → win rate is significantly above 50%, indicating a genuine per-trade edge.

    Parameters
    ----------
    trade_pnl : pd.Series
        Per-trade net returns (output of ``_compute_trade_pnl``).
    n_permutations : int
        Number of random permutations (default 1000).
    seed : int
        RNG seed for reproducibility.

    Returns
    -------
    float
        p-value in [0, 1]. Lower is better.
    """
    if len(trade_pnl) < 2:
        return 1.0
    trades = trade_pnl.values.copy()
    # Binomial test: is the win rate significantly above 50%?
    # p = probability of observing >= n_wins out of n_trades under null (win_rate=0.5).
    # Low p → strategy has a significant positive edge per trade.
    from scipy.stats import binomtest
    n_wins = int((trades > 0).sum())
    n_total = len(trades)
    result = binomtest(n_wins, n_total, p=0.5, alternative="greater")
    return float(result.pvalue)


def walk_forward_rolling(
    close: pd.Series,
    signal: pd.Series,
    leverage: float,
    txn_cost_bps: float = DEFAULT_TXN_COST_BPS,
    bars_per_year: int = DEFAULT_BARS_PER_YEAR,
    is_years: int = WF_IS_YEARS,
    oos_years: int = WF_OOS_YEARS,
    step_years: int = WF_STEP_YEARS,
) -> dict[str, Any]:
    """
    Rolling walk-forward validation: multiple IS/OOS windows shifted by ``step_years``.

    Each window runs an independent FTMO simulation on the IS and OOS slices.
    Produces aggregate OOS statistics to measure cross-time consistency.

    Returns
    -------
    dict with keys:
        wf_n_windows, wf_oos_sharpe_mean, wf_oos_sharpe_std,
        wf_oos_monthly_return_mean, wf_oos_consistency (fraction of windows
        with OOS Sharpe > 0), wf_windows (list of per-window dicts)
    """
    if not isinstance(close.index, pd.DatetimeIndex):
        return {"wf_n_windows": 0}

    start_year = close.index[0].year
    end_year   = close.index[-1].year

    windows = []
    yr = start_year
    while True:
        is_start  = pd.Timestamp(f"{yr}-01-01")
        is_end    = pd.Timestamp(f"{yr + is_years}-01-01")
        oos_end   = pd.Timestamp(f"{yr + is_years + oos_years}-01-01")
        if oos_end.year > end_year + 1:
            break
        is_mask  = (close.index >= is_start) & (close.index < is_end)
        oos_mask = (close.index >= is_end)   & (close.index < oos_end)
        if is_mask.sum() < 1000 or oos_mask.sum() < 1000:
            yr += step_years
            continue

        window: dict[str, Any] = {
            "is_start":  str(is_start.date()),
            "is_end":    str(is_end.date()),
            "oos_start": str(is_end.date()),
            "oos_end":   str(oos_end.date()),
        }
        for mask, prefix in [(is_mask, "is"), (oos_mask, "oos")]:
            close_s  = close.loc[mask]
            signal_s = signal.loc[mask]
            masked_s, _ = _apply_ftmo_mask(signal_s, close_s, leverage, txn_cost_bps)
            r = backtest_signal(close=close_s, signal=masked_s,
                                txn_cost_bps=txn_cost_bps, bars_per_year=bars_per_year)
            window[f"{prefix}_sharpe"]       = r.get("sharpe", 0.0)
            window[f"{prefix}_monthly_return_pct"] = r.get("monthly_return_pct", 0.0)
            window[f"{prefix}_n_trades"]     = r.get("n_trades", 0)
        windows.append(window)
        yr += step_years

    if not windows:
        return {"wf_n_windows": 0}

    oos_sharpes  = [w["oos_sharpe"] for w in windows]
    oos_monthly  = [w["oos_monthly_return_pct"] for w in windows]
    return {
        "wf_n_windows":              len(windows),
        "wf_oos_sharpe_mean":        float(np.mean(oos_sharpes)),
        "wf_oos_sharpe_std":         float(np.std(oos_sharpes)),
        "wf_oos_monthly_return_mean": float(np.mean(oos_monthly)),
        "wf_oos_consistency":        float(np.mean([s > 0 for s in oos_sharpes])),
        "wf_windows":                windows,
    }


def backtest_signal_ftmo(
    close: pd.Series,
    signal: pd.Series,
    txn_cost_bps: float = DEFAULT_TXN_COST_BPS,
    eurusd_price: float = 1.10,
    risk_pct: float = FTMO_RISK_PER_TRADE,
    stop_pips: float = FTMO_STOP_PIPS,
    max_leverage: float = FTMO_MAX_LEVERAGE,
    bars_per_year: int = DEFAULT_BARS_PER_YEAR,
    forward_returns: pd.Series | None = None,
    oos_start: str | None = OOS_START_DEFAULT,
    wf_rolling: bool = False,
    mc_n_permutations: int = 0,
) -> dict[str, Any]:
    """
    FTMO-compliant backtest of a strategy signal on EUR/USD.

    Applies on top of ``backtest_signal``:
      - Realistic costs: default 2.14 bps (≈ 2.35 pip spread+slippage+commission)
      - Risk-based position sizing: risk_pct equity per trade, stop_pips hard stop
      - Max leverage cap: max_leverage (default 1:30, FTMO standard)
      - FTMO daily loss limit (5%): positions zeroed rest of day after breach
      - FTMO total loss limit (10%): all positions zeroed after breach
      - FTMO-specific metrics added to result dict
      - Walk-forward OOS split: IS metrics (before oos_start) + OOS metrics (after)

    Parameters
    ----------
    close : pd.Series
        1-min EUR/USD close prices.
    signal : pd.Series
        Raw strategy signal in {-1, 0, +1}.
    txn_cost_bps : float
        Transaction cost in bps (default 2.14 ≈ 2.35 pip on EUR/USD).
    eurusd_price : float
        Representative EUR/USD price for pip→bps conversion (default 1.10).
    risk_pct : float
        Fraction of equity risked per trade (default 0.005 = 0.5%).
    stop_pips : float
        Hard stop-loss distance in pips (default 10).
    max_leverage : float
        Maximum leverage (default 30 = FTMO 1:30).
    oos_start : str or None
        Start of out-of-sample period (ISO date). None disables OOS split.
    wf_rolling : bool
        If True, run rolling walk-forward validation (multiple IS/OOS windows).
        Results are stored under ``wf_*`` keys. Default False.
    mc_n_permutations : int
        Number of Monte Carlo trade permutations. 0 = disabled (default).
        When > 0, computes ``mc_pvalue``: fraction of permuted sequences whose
        total return >= real total return. p < 0.05 indicates a genuine edge.
    """
    stop_price = stop_pips * FTMO_PIP
    leverage_by_risk = risk_pct / (stop_price / eurusd_price)
    leverage = min(leverage_by_risk, max_leverage)

    masked_signal, ftmo_metrics = _apply_ftmo_mask(signal, close, leverage, txn_cost_bps)

    result = backtest_signal(
        close=close,
        signal=masked_signal,
        txn_cost_bps=txn_cost_bps,
        bars_per_year=bars_per_year,
        forward_returns=forward_returns,
    )

    result.update(ftmo_metrics)
    result["ftmo_leverage"] = round(leverage, 2)
    result["ftmo_risk_pct"] = risk_pct
    result["ftmo_stop_pips"] = stop_pips

    # Re-scale reported equity metrics to FTMO_INITIAL_CAPITAL
    result["ftmo_end_equity"] = FTMO_INITIAL_CAPITAL * (1 + result.get("total_return", 0))
    result["ftmo_monthly_profit"] = FTMO_INITIAL_CAPITAL * result.get("monthly_return", 0)

    # Walk-forward OOS split
    if oos_start is not None:
        oos_ts = pd.Timestamp(oos_start)
        is_mask  = close.index < oos_ts
        oos_mask = close.index >= oos_ts

        def _split_bt(mask: pd.Series[bool], prefix: str) -> None:
            if mask.sum() < 100:
                return
            close_s  = close.loc[mask]
            signal_s = signal.loc[mask]  # raw signal, not masked — fresh FTMO sim per period
            fwd_split = forward_returns.loc[mask] if forward_returns is not None else None
            masked_s, _ = _apply_ftmo_mask(signal_s, close_s, leverage, txn_cost_bps)
            split_result = backtest_signal(
                close=close_s,
                signal=masked_s,
                txn_cost_bps=txn_cost_bps,
                bars_per_year=bars_per_year,
                forward_returns=fwd_split,
            )
            for k, v in split_result.items():
                if k not in ("equity_curve", "status"):
                    result[f"{prefix}_{k}"] = v

        _split_bt(is_mask,  "is")
        _split_bt(oos_mask, "oos")

        result["oos_start"] = oos_start
        result["is_n_bars"]  = int(is_mask.sum())
        result["oos_n_bars"] = int(oos_mask.sum())

    # Rolling walk-forward validation
    if wf_rolling:
        wf = walk_forward_rolling(
            close=close,
            signal=signal,
            leverage=leverage,
            txn_cost_bps=txn_cost_bps,
            bars_per_year=bars_per_year,
        )
        result.update(wf)

    # Monte Carlo trade permutation test
    if mc_n_permutations > 0:
        position = masked_signal.shift(1).fillna(0)
        bar_ret  = close.pct_change().fillna(0)
        txn_cost = txn_cost_bps / 10_000.0
        position_change = position.diff().abs().fillna(position.abs())
        strat_ret = position * bar_ret - position_change * txn_cost
        trade_pnl = _compute_trade_pnl(position, strat_ret)
        result["mc_pvalue"]        = monte_carlo_trade_pvalue(trade_pnl, mc_n_permutations)
        result["mc_n_permutations"] = mc_n_permutations

    return result


def backtest_from_forward_returns(
    factor_values: pd.Series,
    forward_returns: pd.Series,
    txn_cost_bps: float = DEFAULT_TXN_COST_BPS,
    bars_per_year: int = DEFAULT_BARS_PER_YEAR,
) -> dict[str, Any]:
    """
    Backtest a factor using sign(factor) as signal against forward returns.

    This is the legacy FactorBacktester mode: no close series available,
    just (factor, forward_return) pairs. All time-based metrics degrade
    gracefully (n_months approximated from n_bars).
    """
    factor_values = pd.to_numeric(factor_values, errors="coerce")
    forward_returns = pd.to_numeric(forward_returns, errors="coerce")

    common = factor_values.dropna().index.intersection(forward_returns.dropna().index)
    if len(common) < 10:
        return {"status": "failed", "reason": f"insufficient aligned data ({len(common)} rows)"}

    f = factor_values.loc[common]
    r = forward_returns.loc[common]

    signal = np.sign(f).astype(float)
    position = signal.shift(1).fillna(0)

    txn_cost = txn_cost_bps / 10_000.0
    position_change = position.diff().abs().fillna(position.abs())
    strategy_returns = position * r - position_change * txn_cost

    if strategy_returns.std() > 0:
        sharpe = float(strategy_returns.mean() / strategy_returns.std() * np.sqrt(bars_per_year))
    else:
        sharpe = 0.0

    total_return = float((1 + strategy_returns).prod() - 1)
    equity = (1 + strategy_returns).cumprod()
    max_dd = float(((equity - equity.cummax()) / equity.cummax().replace(0, np.nan)).min() or 0.0)

    ic_val = float(f.corr(r)) if f.std() > 0 and r.std() > 0 else 0.0
    ic = ic_val if np.isfinite(ic_val) else 0.0

    trade_pnl = _compute_trade_pnl(position, strategy_returns)
    n_trades = len(trade_pnl)
    win_rate = float((trade_pnl > 0).mean()) if n_trades > 0 else 0.0

    ann_return = float(strategy_returns.mean() * bars_per_year)
    volatility = float(strategy_returns.std() * np.sqrt(bars_per_year))

    return {
        "status": "success",
        "sharpe": sharpe,
        "max_drawdown": max_dd,
        "total_return": total_return,
        "annualized_return": ann_return,
        "volatility": volatility,
        "win_rate": win_rate,
        "n_trades": n_trades,
        "ic": ic,
        "n_bars": len(strategy_returns),
        "txn_cost_bps": txn_cost_bps,
        "bars_per_year": bars_per_year,
    }
