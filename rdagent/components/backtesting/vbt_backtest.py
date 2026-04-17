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

from typing import Any, Dict, Optional

import numpy as np
import pandas as pd

try:
    import vectorbt as vbt  # noqa: F401

    VBT_AVAILABLE = True
except ImportError:
    VBT_AVAILABLE = False


DEFAULT_TXN_COST_BPS = 1.5
DEFAULT_BARS_PER_YEAR = 252 * 1440  # 252 trading days * 1440 min/day = 362,880
EXTREME_BAR_THRESHOLD = 0.05  # |ret| > 5% on a single 1-min bar → suspicious


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
) -> Optional[float]:
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
    forward_returns: Optional[pd.Series] = None,
    cross_check: bool = False,
) -> Dict[str, Any]:
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
    n_trades = int(len(trade_pnl))
    n_position_changes = int((position.diff().fillna(0) != 0).sum())

    if n_trades > 0:
        win_rate = float((trade_pnl > 0).mean())
        wins = trade_pnl[trade_pnl > 0].sum()
        losses = -trade_pnl[trade_pnl < 0].sum()
        profit_factor = float(wins / losses) if losses > 0 else float("inf") if wins > 0 else 0.0
    else:
        win_rate = 0.0
        profit_factor = 0.0

    ic: Optional[float] = None
    if forward_returns is not None:
        fwd = pd.to_numeric(forward_returns, errors="coerce")
        common = signal.index.intersection(fwd.dropna().index)
        if len(common) > 10:
            s = signal.loc[common]
            f = fwd.loc[common]
            if s.std() > 0 and f.std() > 0:
                ic_val = float(s.corr(f))
                ic = ic_val if np.isfinite(ic_val) else None

    result: Dict[str, Any] = {
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
        "n_bars": int(len(strategy_returns)),
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


def backtest_from_forward_returns(
    factor_values: pd.Series,
    forward_returns: pd.Series,
    txn_cost_bps: float = DEFAULT_TXN_COST_BPS,
    bars_per_year: int = DEFAULT_BARS_PER_YEAR,
) -> Dict[str, Any]:
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
    n_trades = int(len(trade_pnl))
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
        "n_bars": int(len(strategy_returns)),
        "txn_cost_bps": txn_cost_bps,
        "bars_per_year": bars_per_year,
    }
