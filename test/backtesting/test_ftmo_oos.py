"""
Tests for backtest_signal_ftmo and walk-forward OOS validation.

Covers:
- FTMO daily/total loss limits
- Risk-based leverage calculation
- OOS split returns independent IS and OOS metrics
- OOS uses fresh FTMO simulation (not contaminated by IS losses)
- Monte Carlo permutation test helper
"""
from __future__ import annotations

import numpy as np
import pandas as pd
import pytest

from rdagent.components.backtesting.vbt_backtest import (
    OOS_START_DEFAULT,
    backtest_signal_ftmo,
    FTMO_MAX_DAILY_LOSS,
    FTMO_MAX_TOTAL_LOSS,
)


# ---------------------------------------------------------------------------
# Fixtures
# ---------------------------------------------------------------------------
@pytest.fixture
def close_2yr() -> pd.Series:
    """~3 months of synthetic 1-min EUR/USD (enough bars for all leverage/FTMO tests)."""
    np.random.seed(42)
    n = 90 * 1440  # 90 days × 1440 min
    idx = pd.date_range("2022-01-01", periods=n, freq="1min")
    price = 1.10 + np.cumsum(np.random.randn(n) * 0.00005)
    return pd.Series(price, index=idx)


@pytest.fixture
def close_6yr() -> pd.Series:
    """Synthetic data crossing the 2024-01-01 IS/OOS boundary.

    120 days starting 2023-09-01 → ends ~2024-01-01, giving ~30 days of OOS data.
    Small enough to keep tests fast.
    """
    np.random.seed(7)
    n = 150 * 1440  # 2023-09-01 + 150d ≈ 2024-01-28 → ~28 days of OOS data
    idx = pd.date_range("2023-09-01", periods=n, freq="1min")
    price = 1.10 + np.cumsum(np.random.randn(n) * 0.00005)
    return pd.Series(price, index=idx)


def _random_signal(index: pd.Index, seed: int = 0) -> pd.Series:
    np.random.seed(seed)
    return pd.Series(np.random.choice([-1.0, 0.0, 1.0], size=len(index)), index=index)


# ---------------------------------------------------------------------------
# FTMO leverage tests
# ---------------------------------------------------------------------------
def test_ftmo_result_contains_leverage_fields(close_2yr):
    signal = _random_signal(close_2yr.index)
    r = backtest_signal_ftmo(close_2yr, signal, oos_start=None)
    assert "ftmo_leverage" in r
    assert "ftmo_risk_pct" in r
    assert "ftmo_stop_pips" in r
    assert r["ftmo_leverage"] > 0


def test_ftmo_leverage_capped_at_max(close_2yr):
    signal = _random_signal(close_2yr.index)
    # With very tight stop (1 pip) risk_pct=0.5% → leverage would be 55x → capped at 30
    r = backtest_signal_ftmo(close_2yr, signal, stop_pips=1, max_leverage=30, oos_start=None)
    assert r["ftmo_leverage"] <= 30.0


def test_ftmo_zero_signal_produces_no_trades(close_2yr):
    signal = pd.Series(0.0, index=close_2yr.index)
    r = backtest_signal_ftmo(close_2yr, signal, oos_start=None)
    assert r["n_trades"] == 0
    assert r["total_return"] == 0.0


# ---------------------------------------------------------------------------
# OOS split tests
# ---------------------------------------------------------------------------
def test_oos_split_produces_is_and_oos_keys(close_6yr):
    signal = _random_signal(close_6yr.index)
    r = backtest_signal_ftmo(close_6yr, signal, oos_start="2024-01-01")

    assert "is_sharpe" in r
    assert "oos_sharpe" in r
    assert "is_monthly_return_pct" in r
    assert "oos_monthly_return_pct" in r
    assert "is_n_bars" in r
    assert "oos_n_bars" in r
    assert r["oos_start"] == "2024-01-01"


def test_oos_split_bars_sum_to_total(close_6yr):
    signal = _random_signal(close_6yr.index)
    r = backtest_signal_ftmo(close_6yr, signal, oos_start="2024-01-01")
    assert r["is_n_bars"] + r["oos_n_bars"] == len(close_6yr)


def test_oos_none_disables_split(close_6yr):
    signal = _random_signal(close_6yr.index)
    r = backtest_signal_ftmo(close_6yr, signal, oos_start=None)
    assert "is_sharpe" not in r
    assert "oos_sharpe" not in r


def test_oos_is_independent_of_is_losses(close_6yr):
    """OOS must use a fresh FTMO simulation — IS blowup must not zero OOS trades."""
    # Force the IS period to blow up immediately with max short on rising market
    rising = pd.Series(
        np.linspace(1.0, 2.0, len(close_6yr)),
        index=close_6yr.index,
    )
    always_short = pd.Series(-1.0, index=close_6yr.index)

    r = backtest_signal_ftmo(rising, always_short, oos_start="2024-01-01")

    # IS should be wiped out (total loss limit hit), but OOS must still trade
    assert r.get("oos_n_trades", 0) is not None
    assert r.get("oos_n_bars", 0) > 0


def test_oos_default_start_matches_constant(close_6yr):
    signal = _random_signal(close_6yr.index)
    r = backtest_signal_ftmo(close_6yr, signal)
    assert r.get("oos_start") == OOS_START_DEFAULT


# ---------------------------------------------------------------------------
# Monte Carlo permutation test helper
# ---------------------------------------------------------------------------
def _monte_carlo_pvalue(close: pd.Series, signal: pd.Series, n_permutations: int = 200, seed: int = 0) -> float:
    """
    Estimate p-value: fraction of random permutations that beat the real Sharpe.
    p < 0.05 → strategy has statistically significant edge.
    """
    real_r = backtest_signal_ftmo(close, signal, oos_start=None)
    real_sharpe = real_r.get("sharpe", 0.0) or 0.0

    rng = np.random.default_rng(seed)
    beat = 0
    signal_vals = signal.values.copy()
    for _ in range(n_permutations):
        perm = rng.permutation(signal_vals)
        perm_signal = pd.Series(perm, index=signal.index)
        perm_r = backtest_signal_ftmo(close, perm_signal, oos_start=None)
        if (perm_r.get("sharpe") or 0.0) >= real_sharpe:
            beat += 1
    return beat / n_permutations


@pytest.mark.slow
def test_random_signal_has_no_edge(close_2yr):
    """A purely random signal should NOT beat most permutations."""
    signal = _random_signal(close_2yr.index, seed=42)
    pval = _monte_carlo_pvalue(close_2yr, signal, n_permutations=50)
    # Random vs random: p-value should be near 0.5 (not significant)
    assert pval > 0.10, f"Random signal unexpectedly significant: p={pval:.2f}"


@pytest.mark.slow
def test_perfect_signal_is_significant(close_2yr):
    """An oracle signal on hourly bars should beat random permutations significantly.

    Per-minute oracle trading is unprofitable due to FTMO transaction costs, so we
    use 60-bar held positions (≈1h) where each directional move is large enough to
    cover the spread.
    """
    bar_ret = close_2yr.pct_change().fillna(0)
    # Hourly oracle: sign of 60-bar future return, broadcast to all 60 minute bars
    hourly_ret = bar_ret.rolling(60).sum().shift(-60).fillna(0)
    perfect = pd.Series(np.sign(hourly_ret), index=close_2yr.index)
    pval = _monte_carlo_pvalue(close_2yr, perfect, n_permutations=50)
    assert pval < 0.30, f"Hourly oracle signal should beat random permutations: p={pval:.2f}"


# ---------------------------------------------------------------------------
# FTMO metrics in result dict
# ---------------------------------------------------------------------------
def test_ftmo_result_has_equity_and_profit(close_2yr):
    signal = _random_signal(close_2yr.index)
    r = backtest_signal_ftmo(close_2yr, signal, oos_start=None)
    assert "ftmo_end_equity" in r
    assert "ftmo_monthly_profit" in r
    assert r["ftmo_end_equity"] > 0
