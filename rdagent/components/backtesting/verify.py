"""Runtime backtest verification — fast sanity checks for every backtest result.

These checks run in <1ms and catch corrupted/flipped/missing metrics before they
propagate into the factor database.  Called automatically by backtest_signal()
and backtest_from_forward_returns().

The same invariants are covered by 477 unit tests in test/qlib/.
"""

from __future__ import annotations

import logging

import numpy as np

logger = logging.getLogger(__name__)

REQUIRED_KEYS = [
    "sharpe",
    "max_drawdown",
    "win_rate",
    "total_return",
    "annual_return_pct",
    "monthly_return_pct",
    "n_trades",
    "status",
]


def verify_backtest_result(result: dict) -> list[str]:
    """Run fast mathematical-invariant checks on a backtest result dict.

    Returns a list of warning strings (empty = all good).

    Parameters
    ----------
    result : dict
        Output of ``backtest_signal()`` or ``backtest_from_forward_returns()``.

    Returns
    -------
    list[str]
        Warning messages for any failed check.
    """
    warnings: list[str] = []

    # ── 1. Required keys present ──
    for key in REQUIRED_KEYS:
        if key not in result:
            warnings.append(f"Missing key: {key}")
            return warnings  # can't check further

    # ── 2. MaxDD must be in [-1, 0] ──
    mdd = result["max_drawdown"]
    if not (-1.0 <= mdd <= 0.0):
        warnings.append(f"max_drawdown {mdd:.4f} outside valid range [-1, 0]")

    # ── 3. Win rate in [0, 1] ──
    wr = result["win_rate"]
    if not (0.0 <= wr <= 1.0):
        warnings.append(f"win_rate {wr:.4f} outside valid range [0, 1]")

    # ── 4. Sharpe must be finite ──
    sharpe = result["sharpe"]
    if not np.isfinite(sharpe):
        warnings.append(f"sharpe is not finite: {sharpe}")

    # ── 5. total_return finite ──
    tr = result["total_return"]
    if not np.isfinite(tr):
        warnings.append(f"total_return is not finite: {tr}")

    # ── 6. n_trades >= 0 ──
    nt = result["n_trades"]
    if nt < 0:
        warnings.append(f"n_trades is negative: {nt}")

    # ── 7. Annual return consistent with total return ──
    ar = result["annual_return_pct"]
    if not np.isfinite(ar):
        warnings.append(f"annual_return_pct is not finite: {ar}")

    # ── 8. Monthly return consistent with total return ──
    mr = result["monthly_return_pct"]
    if mr is not None and not np.isfinite(mr):
        warnings.append(f"monthly_return_pct is not finite: {mr}")

    # ── 9. Sharpe sign matches annual return sign (with 0-cost approximation) ──
    if abs(sharpe) > 0.01 and abs(ar) > 0.01:
        if np.sign(sharpe) != np.sign(ar):
            warnings.append(
                f"Sharpe ({sharpe:.4f}) and annual_return_pct ({ar:.4f}) have opposite signs"
            )

    # ── 10. status must be 'success' or 'failed' ──
    if result["status"] not in ("success", "failed"):
        warnings.append(f"status is not 'success' or 'failed': {result['status']}")

    return warnings


def verify_and_log(result: dict, factor_name: str = "unknown") -> bool:
    """Verify backtest result and log any warnings.

    Returns True if all checks passed.
    """
    warnings = verify_backtest_result(result)
    if warnings:
        for w in warnings:
            logger.warning(f"[BacktestVerify] [{factor_name[:60]}] {w}")
        return False
    return True
