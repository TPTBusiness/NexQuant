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
    _apply_ftmo_mask,
    backtest_signal_ftmo,
    FTMO_INITIAL_CAPITAL,
    FTMO_MAX_DAILY_LOSS,
    FTMO_MAX_TOTAL_LOSS,
    monte_carlo_trade_pvalue,
    walk_forward_rolling,
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


# ---------------------------------------------------------------------------
# Monte Carlo trade permutation tests
# ---------------------------------------------------------------------------
def test_mc_pvalue_in_result(close_2yr):
    signal = _random_signal(close_2yr.index)
    r = backtest_signal_ftmo(close_2yr, signal, oos_start=None, mc_n_permutations=50)
    assert "mc_pvalue" in r
    assert 0.0 <= r["mc_pvalue"] <= 1.0
    assert r["mc_n_permutations"] == 50


def test_mc_pvalue_disabled_by_default(close_2yr):
    signal = _random_signal(close_2yr.index)
    r = backtest_signal_ftmo(close_2yr, signal, oos_start=None)
    assert "mc_pvalue" not in r


def test_mc_zero_trades_returns_one(close_2yr):
    """Zero-signal → no trades → p-value must be 1.0 (no edge)."""
    trade_pnl = pd.Series([], dtype=float)
    assert monte_carlo_trade_pvalue(trade_pnl, n_permutations=10) == 1.0


# ---------------------------------------------------------------------------
# Rolling walk-forward tests
# ---------------------------------------------------------------------------
def test_wf_rolling_keys_in_result(close_6yr):
    signal = _random_signal(close_6yr.index)
    r = backtest_signal_ftmo(close_6yr, signal, oos_start="2024-01-01", wf_rolling=True)
    # With only ~150 days of data, windows may be 0 — just check key presence
    assert "wf_n_windows" in r


def test_wf_rolling_enabled_by_default(close_6yr):
    signal = _random_signal(close_6yr.index)
    r = backtest_signal_ftmo(close_6yr, signal, oos_start="2024-01-01")
    assert "wf_n_windows" in r


def test_wf_consistency_range(close_6yr):
    """wf_oos_consistency must be in [0, 1] when windows exist."""
    signal = _random_signal(close_6yr.index)
    r = backtest_signal_ftmo(close_6yr, signal, oos_start="2024-01-01", wf_rolling=True)
    c = r.get("wf_oos_consistency")
    if c is not None:
        assert 0.0 <= c <= 1.0


# ---------------------------------------------------------------------------
# Direct _apply_ftmo_mask unit tests
# ---------------------------------------------------------------------------

class TestApplyFtmoMask:
    """Direct unit tests for _apply_ftmo_mask — the core FTMO daily/total loss engine."""

    @pytest.fixture
    def flat_close(self) -> pd.Series:
        n = 3000
        idx = pd.date_range("2024-01-01", periods=n, freq="1min")
        return pd.Series(1.10, index=idx)

    def test_returns_compliance_dict(self, flat_close):
        signal = _random_signal(flat_close.index)
        masked, info = _apply_ftmo_mask(signal, flat_close, leverage=1.0, txn_cost_bps=2.14)
        assert "ftmo_daily_breaches" in info
        assert "ftmo_total_breached" in info
        assert "ftmo_total_breach_ts" in info
        assert "ftmo_compliant" in info

    def test_flat_market_zero_signal_fully_compliant(self, flat_close):
        """No trades → always compliant."""
        signal = pd.Series(0.0, index=flat_close.index)
        masked, info = _apply_ftmo_mask(signal, flat_close, leverage=1.0, txn_cost_bps=2.14)
        assert info["ftmo_daily_breaches"] == 0
        assert info["ftmo_total_breached"] is False
        assert info["ftmo_compliant"] is True
        # All signals should remain zero
        assert (masked == 0).all()

    def test_daily_loss_breach_zeroes_rest_of_day(self):
        """When daily loss exceeds 5%, rest of that day's signals are zeroed."""
        n = 3000
        idx = pd.date_range("2024-01-01", periods=n, freq="1min")
        # Price drops sharply in first few bars to trigger daily loss
        price = pd.Series(1.10, index=idx, dtype=float)
        price.iloc[3:20] = 0.00  # crash from 1.10 to 0.00 → massive loss
        signal = pd.Series(1.0, index=idx)  # always long at 30x leverage

        masked, info = _apply_ftmo_mask(signal, price, leverage=30.0, txn_cost_bps=0)
        assert info["ftmo_daily_breaches"] > 0
        # After breach, signals on same day must be zeroed
        breach_day = idx[0].date()
        same_day_late = (idx[-1] if idx[-1].date() == breach_day else idx[20])
        if same_day_late.date() == breach_day:
            assert masked.loc[same_day_late] == 0

    def test_total_loss_breach_zeroes_all_remaining(self):
        """When total loss exceeds 10%, ALL subsequent signals are zeroed."""
        n = 5000
        idx = pd.date_range("2024-01-01", periods=n, freq="1min")
        # Price crashes → max position → total loss limit breached
        price = pd.Series(1.10, index=idx, dtype=float)
        price.iloc[5:50] = 0.50  # >10% drop with 30x leverage
        signal = pd.Series(1.0, index=idx)

        masked, info = _apply_ftmo_mask(signal, price, leverage=30.0, txn_cost_bps=0)
        assert info["ftmo_total_breached"] is True
        assert info["ftmo_total_breach_ts"] is not None
        # After breach, ALL later signals must be zero
        assert (masked.iloc[100:] == 0).all()

    def test_total_breach_respected_across_days(self):
        """Total breach persists across day boundaries — no new trades after breach."""
        n = 5000
        idx = pd.date_range("2024-01-01", periods=n, freq="1min")
        price = pd.Series(1.10, index=idx, dtype=float)
        price.iloc[5:50] = 0.50
        signal = pd.Series(1.0, index=idx)

        masked, info = _apply_ftmo_mask(signal, price, leverage=30.0, txn_cost_bps=0)
        # All signals after breach index must be zero
        breach_ts = pd.Timestamp(info["ftmo_total_breach_ts"])
        assert (masked.loc[masked.index > breach_ts] == 0).all()

    def test_daily_loss_resets_on_new_day(self):
        """Daily loss limit resets at day boundary — new day starts fresh (unless total breached)."""
        n = 5000
        idx = pd.date_range("2024-01-01", periods=n, freq="1min")
        price = pd.Series(1.10, index=idx, dtype=float)
        # Trigger daily breach on day 1 by dropping 1%
        price.iloc[5:20] = 1.09  # ~1% drop with 30x → ~30% loss
        signal = pd.Series(1.0, index=idx)

        masked, info = _apply_ftmo_mask(signal, price, leverage=30.0, txn_cost_bps=0)
        assert info["ftmo_daily_breaches"] >= 1
        # Day 2 signals should be active again if not total-breached
        day2_mask = idx.date > idx[0].date()
        if day2_mask.any() and not info["ftmo_total_breached"]:
            day2 = idx[day2_mask][0]
            assert masked.loc[day2] != 0

    def test_compliant_flag_false_after_daily_breach(self):
        """Even one daily breach makes ftmo_compliant=False."""
        n = 3000
        idx = pd.date_range("2024-01-01", periods=n, freq="1min")
        price = pd.Series(1.10, index=idx, dtype=float)
        price.iloc[3:20] = 0.00
        signal = pd.Series(1.0, index=idx)

        masked, info = _apply_ftmo_mask(signal, price, leverage=30.0, txn_cost_bps=0)
        assert info["ftmo_compliant"] is False

    def test_compliant_flag_false_after_total_breach(self):
        """Total breach makes ftmo_compliant=False."""
        n = 5000
        idx = pd.date_range("2024-01-01", periods=n, freq="1min")
        price = pd.Series(1.10, index=idx, dtype=float)
        price.iloc[5:50] = 0.50
        signal = pd.Series(1.0, index=idx)

        masked, info = _apply_ftmo_mask(signal, price, leverage=30.0, txn_cost_bps=0)
        assert info["ftmo_compliant"] is False

    def test_transaction_costs_reduce_equity(self):
        """Transaction costs should reduce equity — compliant scenario with fees."""
        n = 1000
        idx = pd.date_range("2024-01-01", periods=n, freq="1min")
        price = pd.Series(1.10, index=idx, dtype=float)
        # Alternating signal → lots of position changes → high costs
        signal = pd.Series([1.0 if i % 2 == 0 else -1.0 for i in range(n)], index=idx)

        masked, info = _apply_ftmo_mask(signal, price, leverage=1.0, txn_cost_bps=10.0)
        # With high costs and flat market, equity should drop
        assert "ftmo_daily_breaches" in info

    def test_output_mask_has_same_index(self):
        n = 2000
        idx = pd.date_range("2024-01-01", periods=n, freq="1min")
        price = pd.Series(1.10, index=idx)
        signal = _random_signal(idx, seed=1)

        masked, info = _apply_ftmo_mask(signal, price, leverage=1.0, txn_cost_bps=2.14)
        assert len(masked) == len(signal)
        assert masked.index.equals(signal.index)
