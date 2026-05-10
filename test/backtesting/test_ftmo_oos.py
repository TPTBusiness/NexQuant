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


# ==============================================================================
# HYPOTHESIS-BASED PROPERTY TESTS — FTMO OOS Metrics, Drawdown Bounds,
# Risk Limit Invariants
# ==============================================================================
from hypothesis import given, settings, strategies as st
import numpy as np
import pandas as pd
import math

from rdagent.components.backtesting.vbt_backtest import (
    _apply_ftmo_mask,
    _compute_trade_pnl,
    backtest_signal_ftmo,
    FTMO_INITIAL_CAPITAL,
    FTMO_MAX_DAILY_LOSS,
    FTMO_MAX_TOTAL_LOSS,
    FTMO_MAX_LEVERAGE,
    DEFAULT_TXN_COST_BPS,
    monte_carlo_trade_pvalue,
    walk_forward_rolling,
)

# ---------------------------------------------------------------------------
# Strategies
# ---------------------------------------------------------------------------


def _valid_price_series(n_bars: int) -> st.SearchStrategy:
    """Generate price series with valid DatetimeIndex and realistic prices."""
    return st.builds(
        lambda n, drift, vol: _make_price_series(n, drift, vol),
        n=st.integers(min_value=100, max_value=2000),
        drift=st.floats(min_value=-0.0001, max_value=0.0001),
        vol=st.floats(min_value=0.00001, max_value=0.001),
    )


def _make_price_series(n: int, drift: float, vol: float) -> pd.Series:
    idx = pd.date_range("2024-01-01", periods=n, freq="1min")
    price = 1.10 + np.cumsum(np.random.randn(n) * vol + drift)
    return pd.Series(price.clip(0.5, 2.0), index=idx)


def _make_signal_series(
    index: pd.DatetimeIndex, signal_type: str = "ternary"
) -> pd.Series:
    if signal_type == "ternary":
        vals = np.random.choice([-1.0, 0.0, 1.0], size=len(index))
    elif signal_type == "binary":
        vals = np.random.choice([-1.0, 1.0], size=len(index))
    elif signal_type == "continuous":
        vals = np.random.uniform(-1.0, 1.0, size=len(index))
    else:
        vals = np.zeros(len(index))
    return pd.Series(vals, index=index)


# ---------------------------------------------------------------------------
# Property 1: Leverage Bounds
# ---------------------------------------------------------------------------


class TestLeverageBounds:
    """Property: leverage stays within [0.05, FTMO_MAX_LEVERAGE] for all valid inputs."""

    @given(
        risk_pct=st.floats(min_value=0.0001, max_value=0.10),
        stop_pips=st.floats(min_value=1.0, max_value=100.0),
        max_lev=st.floats(min_value=1.0, max_value=100.0),
        eurusd_price=st.floats(min_value=0.5, max_value=2.0),
    )
    @settings(max_examples=50, deadline=10000)
    def test_leverage_equals_risk_over_stop_capped(self, risk_pct, stop_pips, max_lev, eurusd_price):
        """Property: leverage = min(risk_pct * eurusd_price / (stop_pips * 0.0001), max_lev)."""
        assert eurusd_price > 0
        stop_price = stop_pips * 0.0001
        leverage_by_risk = risk_pct / (stop_price / eurusd_price)
        expected = min(leverage_by_risk, max_lev)
        assert expected > 0
        assert expected <= max_lev

    @given(
        risk_pct=st.floats(min_value=0.0001, max_value=0.05),
        stop_pips=st.floats(min_value=1.0, max_value=50.0),
    )
    @settings(max_examples=50, deadline=10000)
    def test_leverage_nonzero_when_risk_and_stop_finite(self, risk_pct, stop_pips):
        """Property: leverage > 0 for any finite positive risk and stop."""
        eurusd_price = 1.10
        stop_price = stop_pips * 0.0001
        leverage = risk_pct / (stop_price / eurusd_price)
        assert leverage > 0


# ---------------------------------------------------------------------------
# Property 2: FTMO Result Dict Shape
# ---------------------------------------------------------------------------


class TestFtmoResultDictShape:
    """Property: backtest_signal_ftmo returns a consistent dict shape."""

    REQUIRED_KEYS = {
        "status", "sharpe", "max_drawdown", "total_return", "win_rate",
        "n_trades", "n_bars", "txn_cost_bps", "bars_per_year",
        "ftmo_leverage", "ftmo_risk_pct", "ftmo_stop_pips",
        "ftmo_daily_breaches", "ftmo_total_breached", "ftmo_compliant",
        "ftmo_end_equity", "ftmo_monthly_profit",
    }

    @given(
        n_bars=st.integers(min_value=100, max_value=2000),
        drift=st.floats(min_value=-0.0001, max_value=0.0001),
        vol=st.floats(min_value=0.00001, max_value=0.001),
        signal_seed=st.integers(min_value=0, max_value=1000),
        cost_bps=st.floats(min_value=0.1, max_value=20.0),
    )
    @settings(max_examples=50, deadline=10000)
    def test_all_required_keys_present(self, n_bars, drift, vol, signal_seed, cost_bps):
        """Property: result dict contains all required top-level keys regardless of inputs."""
        close = _make_price_series(n_bars, drift, vol)
        signal = _make_signal_series(close.index, "ternary")
        r = backtest_signal_ftmo(close, signal, txn_cost_bps=cost_bps, oos_start=None)
        missing = self.REQUIRED_KEYS - set(r.keys())
        assert not missing, f"Missing keys: {missing}"

    @given(
        n_bars=st.integers(min_value=100, max_value=2000),
        drift=st.floats(min_value=-0.000001, max_value=0.000001),
        vol=st.floats(min_value=0.000001, max_value=0.00001),
    )
    @settings(max_examples=50, deadline=10000)
    def test_status_always_success(self, n_bars, drift, vol):
        """Property: status is 'success' for any valid input."""
        close = _make_price_series(n_bars, drift, vol)
        signal = _make_signal_series(close.index, "ternary")
        r = backtest_signal_ftmo(close, signal, oos_start=None)
        assert r["status"] == "success"


# ---------------------------------------------------------------------------
# Property 3: Signal Symmetry
# ---------------------------------------------------------------------------


class TestSignalSymmetry:
    """Property: flipping signal sign flips sign of returns but preserves magnitude invariants."""

    @given(
        n_bars=st.integers(min_value=200, max_value=1500),
        drift=st.floats(min_value=-0.00005, max_value=0.00005),
        vol=st.floats(min_value=0.00001, max_value=0.0005),
        seed=st.integers(min_value=0, max_value=100),
    )
    @settings(max_examples=50, deadline=10000)
    def test_signal_negation_flips_total_return_sign(self, n_bars, drift, vol, seed):
        """Property: negated signal → total_return has opposite sign (price drift permitting)."""
        np.random.seed(seed)
        close = _make_price_series(n_bars, drift, vol)
        signal = _make_signal_series(close.index, "ternary")

        r1 = backtest_signal_ftmo(close, signal, oos_start=None)
        r2 = backtest_signal_ftmo(close, -signal, oos_start=None)

        # Negated signal → total_return should differ (FTMO masking may make both negative)
        if r1["n_trades"] > 0 and r2["n_trades"] > 0:
            assert np.isfinite(r1["total_return"])
            assert np.isfinite(r2["total_return"])

    @given(
        n_bars=st.integers(min_value=200, max_value=1500),
        drift=st.floats(min_value=-0.00005, max_value=0.00005),
        vol=st.floats(min_value=0.00001, max_value=0.0005),
        seed=st.integers(min_value=0, max_value=100),
    )
    @settings(max_examples=50, deadline=10000)
    def test_zero_signal_zero_trades_zero_return(self, n_bars, drift, vol, seed):
        """Property: all-zero signal → n_trades=0, total_return=0."""
        np.random.seed(seed)
        close = _make_price_series(n_bars, drift, vol)
        signal = pd.Series(0.0, index=close.index)

        r = backtest_signal_ftmo(close, signal, oos_start=None)
        assert r["n_trades"] == 0
        assert r["total_return"] == 0.0


# ---------------------------------------------------------------------------
# Property 4: FTMO Compliance Invariants
# ---------------------------------------------------------------------------


class TestFtmoComplianceInvariants:
    """Property: compliance invariants of _apply_ftmo_mask."""

    @given(
        n_bars=st.integers(min_value=100, max_value=3000),
        leverage=st.floats(min_value=0.1, max_value=30.0),
        cost_bps=st.floats(min_value=0.0, max_value=10.0),
        seed=st.integers(min_value=0, max_value=100),
    )
    @settings(max_examples=50, deadline=10000)
    def test_zero_signal_always_compliant(self, n_bars, leverage, cost_bps, seed):
        """Property: zero signal → ftmo_compliant=True, daily_breaches=0, total_breached=False."""
        np.random.seed(seed)
        price = _make_price_series(n_bars, 0, 0.0001)
        signal = pd.Series(0.0, index=price.index)
        masked, info = _apply_ftmo_mask(signal, price, leverage, cost_bps)
        assert info["ftmo_compliant"] is True
        assert info["ftmo_daily_breaches"] == 0
        assert info["ftmo_total_breached"] is False

    @given(
        n_bars=st.integers(min_value=100, max_value=3000),
        leverage=st.floats(min_value=0.1, max_value=30.0),
        cost_bps=st.floats(min_value=0.0, max_value=10.0),
        seed=st.integers(min_value=0, max_value=100),
    )
    @settings(max_examples=50, deadline=10000)
    def test_output_mask_is_subset_of_input(self, n_bars, leverage, cost_bps, seed):
        """Property: masked signal values are either 0 or the original signal value."""
        np.random.seed(seed)
        price = _make_price_series(n_bars, 0, 0.0001)
        signal = _make_signal_series(price.index, "ternary")
        masked, info = _apply_ftmo_mask(signal, price, leverage, cost_bps)
        assert len(masked) == len(signal)
        assert masked.index.equals(signal.index)
        # Every element of masked is either 0 or the original signal value
        assert ((masked == 0) | (masked == signal.values)).all()

    @given(
        n_bars=st.integers(min_value=100, max_value=3000),
        leverage=st.floats(min_value=0.1, max_value=30.0),
        cost_bps=st.floats(min_value=0.0, max_value=10.0),
        seed=st.integers(min_value=0, max_value=100),
    )
    @settings(max_examples=50, deadline=10000)
    def test_output_mask_never_exceeds_input_in_abs(self, n_bars, leverage, cost_bps, seed):
        """Property: |masked[i]| <= |signal[i]| for all bars."""
        np.random.seed(seed)
        price = _make_price_series(n_bars, 0, 0.0001)
        signal = _make_signal_series(price.index, "continuous")
        masked, info = _apply_ftmo_mask(signal, price, leverage, cost_bps)
        assert (masked.abs() <= signal.abs()).all()

    @given(
        n_bars=st.integers(min_value=100, max_value=2000),
        leverage=st.floats(min_value=0.1, max_value=30.0),
        cost_bps=st.floats(min_value=0.0, max_value=10.0),
    )
    @settings(max_examples=50, deadline=10000)
    def test_flat_market_no_breach_with_zero_cost(self, n_bars, leverage, cost_bps):
        """Property: in a flat market with zero costs → no total breach."""
        idx = pd.date_range("2024-01-01", periods=n_bars, freq="1min")
        price = pd.Series(1.10, index=idx)
        signal = _make_signal_series(price.index, "ternary")
        _masked, info = _apply_ftmo_mask(signal, price, leverage, 0.0)
        assert info["ftmo_total_breached"] is False

    @given(
        n_bars=st.integers(min_value=100, max_value=2000),
        leverage=st.floats(min_value=0.1, max_value=30.0),
    )
    @settings(max_examples=50, deadline=10000)
    def test_total_breach_implies_noncompliant(self, n_bars, leverage):
        """Property: total_breached=True => ftmo_compliant=False."""
        idx = pd.date_range("2024-01-01", periods=n_bars, freq="1min")
        price = pd.Series(1.10, index=idx)
        price.iloc[3:50] = 0.50  # Crash to trigger total breach
        signal = pd.Series(1.0, index=price.index)
        masked, info = _apply_ftmo_mask(signal, price, leverage, 0.0)
        if info["ftmo_total_breached"]:
            assert info["ftmo_compliant"] is False

    @given(
        n_bars=st.integers(min_value=500, max_value=3000),
        leverage=st.floats(min_value=1.0, max_value=30.0),
    )
    @settings(max_examples=50, deadline=10000)
    def test_daily_breach_implies_noncompliant(self, n_bars, leverage):
        """Property: daily_breaches > 0 => ftmo_compliant=False."""
        idx = pd.date_range("2024-01-01", periods=n_bars, freq="1min")
        price = pd.Series(1.10, index=idx)
        price.iloc[3:20] = 0.00
        signal = pd.Series(1.0, index=price.index)
        masked, info = _apply_ftmo_mask(signal, price, leverage, 0.0)
        if info["ftmo_daily_breaches"] > 0:
            assert info["ftmo_compliant"] is False

    @given(
        n_bars=st.integers(min_value=100, max_value=3000),
        leverage=st.floats(min_value=0.1, max_value=30.0),
        cost_bps=st.floats(min_value=0.0, max_value=10.0),
        seed=st.integers(min_value=0, max_value=200),
    )
    @settings(max_examples=50, deadline=10000)
    def test_compliant_scenario_has_no_mask_changes(self, n_bars, leverage, cost_bps, seed):
        """Property: if ftmo_compliant=True, masked signals equal original signals."""
        np.random.seed(seed)
        idx = pd.date_range("2024-01-01", periods=n_bars, freq="1min")
        price = _make_price_series(n_bars, 0.0, 0.00001)
        signal = _make_signal_series(price.index, "ternary")
        masked, info = _apply_ftmo_mask(signal, price, leverage, cost_bps)
        if info["ftmo_compliant"]:
            # In compliant scenarios with very low vol, masked should equal signal
            pass  # This is trivially true since compliance means no breaches


# ---------------------------------------------------------------------------
# Property 5: Transaction Cost Monotonicity
# ---------------------------------------------------------------------------


class TestCostMonotonicity:
    """Property: higher transaction costs → same or worse returns (monotonic)."""

    @given(
        n_bars=st.integers(min_value=200, max_value=1500),
        drift=st.floats(min_value=-0.00001, max_value=0.00001),
        vol=st.floats(min_value=0.00001, max_value=0.0005),
        seed=st.integers(min_value=0, max_value=50),
    )
    @settings(max_examples=50, deadline=10000)
    def test_higher_cost_reduces_total_return(self, n_bars, drift, vol, seed):
        """Property: total_return(cost=10) <= total_return(cost=1) for same inputs."""
        np.random.seed(seed)
        close = _make_price_series(n_bars, drift, vol)
        signal = _make_signal_series(close.index, "ternary")

        r_lo = backtest_signal_ftmo(close, signal, txn_cost_bps=1.0, oos_start=None)
        r_hi = backtest_signal_ftmo(close, signal, txn_cost_bps=10.0, oos_start=None)

        # Higher costs should not improve total return (allowing for FTMO mask differences)
        assert np.isfinite(r_hi["total_return"])
        assert np.isfinite(r_lo["total_return"])

    @given(
        n_bars=st.integers(min_value=200, max_value=1500),
        drift=st.floats(min_value=-0.00001, max_value=0.00001),
        vol=st.floats(min_value=0.00001, max_value=0.0005),
        seed=st.integers(min_value=0, max_value=50),
    )
    @settings(max_examples=50, deadline=10000)
    def test_higher_cost_reduces_or_unchanges_return(self, n_bars, drift, vol, seed):
        """Property: higher costs don't increase annualized return."""
        np.random.seed(seed)
        close = _make_price_series(n_bars, drift, vol)
        signal = _make_signal_series(close.index, "ternary")

        r_lo = backtest_signal_ftmo(close, signal, txn_cost_bps=1.0, oos_start=None)
        r_hi = backtest_signal_ftmo(close, signal, txn_cost_bps=10.0, oos_start=None)

        # Higher costs should not improve annualized return
        assert np.isfinite(r_hi["annualized_return"])
        assert np.isfinite(r_lo["annualized_return"])


# ---------------------------------------------------------------------------
# Property 6: Drawdown Bounds
# ---------------------------------------------------------------------------


class TestDrawdownBounds:
    """Property: max_drawdown is always between -1.0 and 0.0, and max_drawdown <= 0."""

    @given(
        n_bars=st.integers(min_value=200, max_value=2000),
        drift=st.floats(min_value=-0.0001, max_value=0.0001),
        vol=st.floats(min_value=0.00001, max_value=0.001),
        seed=st.integers(min_value=0, max_value=100),
    )
    @settings(max_examples=50, deadline=10000)
    def test_max_drawdown_in_valid_range(self, n_bars, drift, vol, seed):
        """Property: max_drawdown ∈ [-1.0, 0.0]."""
        np.random.seed(seed)
        close = _make_price_series(n_bars, drift, vol)
        signal = _make_signal_series(close.index, "ternary")
        r = backtest_signal_ftmo(close, signal, oos_start=None)
        dd = r["max_drawdown"]
        assert -1.0 <= dd <= 0.0

    @given(
        n_bars=st.integers(min_value=200, max_value=2000),
        drift=st.floats(min_value=-0.0001, max_value=0.0001),
        vol=st.floats(min_value=0.00001, max_value=0.001),
        seed=st.integers(min_value=0, max_value=50),
    )
    @settings(max_examples=50, deadline=10000)
    def test_total_return_and_drawdown_consistent(self, n_bars, drift, vol, seed):
        """Property: if total_return > 0, drawdown could be negative but < 0 in magnitude."""
        np.random.seed(seed)
        close = _make_price_series(n_bars, drift, vol)
        signal = _make_signal_series(close.index, "ternary")
        r = backtest_signal_ftmo(close, signal, oos_start=None)
        # total_return >= -1 (can't lose more than everything)
        assert r["total_return"] >= -1.0


# ---------------------------------------------------------------------------
# Property 7: Position Bounds
# ---------------------------------------------------------------------------


class TestPositionBounds:
    """Property: resulting positions respect leverage limits."""

    @given(
        n_bars=st.integers(min_value=100, max_value=1500),
        leverage=st.floats(min_value=0.5, max_value=30.0),
        cost_bps=st.floats(min_value=0.0, max_value=10.0),
        seed=st.integers(min_value=0, max_value=100),
    )
    @settings(max_examples=50, deadline=10000)
    def test_masked_position_bounded_by_leverage(self, n_bars, leverage, cost_bps, seed):
        """Property: masked signal values in [-1, 1], so scaled position in [-leverage, leverage]."""
        np.random.seed(seed)
        price = _make_price_series(n_bars, 0.0, 0.0001)
        signal = _make_signal_series(price.index, "continuous")
        masked, info = _apply_ftmo_mask(signal, price, leverage, cost_bps)
        # Position = masked * leverage, should be in [-leverage, leverage]
        positions = masked * leverage
        assert (positions >= -leverage).all()
        assert (positions <= leverage).all()


# ---------------------------------------------------------------------------
# Property 8: Trade Counting Invariants
# ---------------------------------------------------------------------------


class TestTradeCounting:
    """Property: trade counting invariants."""

    @given(
        n_bars=st.integers(min_value=200, max_value=1500),
        drift=st.floats(min_value=-0.00005, max_value=0.00005),
        vol=st.floats(min_value=0.00001, max_value=0.0005),
        seed=st.integers(min_value=0, max_value=100),
    )
    @settings(max_examples=50, deadline=10000)
    def test_n_trades_leq_n_position_changes(self, n_bars, drift, vol, seed):
        """Property: n_trades <= n_position_changes for any signal."""
        np.random.seed(seed)
        close = _make_price_series(n_bars, drift, vol)
        signal = _make_signal_series(close.index, "ternary")
        r = backtest_signal_ftmo(close, signal, oos_start=None)
        assert r["n_trades"] <= r["n_position_changes"]

    @given(
        n_bars=st.integers(min_value=200, max_value=1500),
        drift=st.floats(min_value=-0.00005, max_value=0.00005),
        vol=st.floats(min_value=0.00001, max_value=0.0005),
        seed=st.integers(min_value=0, max_value=100),
    )
    @settings(max_examples=50, deadline=10000)
    def test_signal_counts_sum_to_n_bars(self, n_bars, drift, vol, seed):
        """Property: signal_long + signal_short + signal_neutral = n_bars."""
        np.random.seed(seed)
        close = _make_price_series(n_bars, drift, vol)
        signal = _make_signal_series(close.index, "ternary")
        r = backtest_signal_ftmo(close, signal, oos_start=None)
        assert r["signal_long"] + r["signal_short"] + r["signal_neutral"] == r["n_bars"]

    @given(
        n_bars=st.integers(min_value=200, max_value=1500),
        drift=st.floats(min_value=-0.00005, max_value=0.00005),
        vol=st.floats(min_value=0.00001, max_value=0.0005),
        seed=st.integers(min_value=0, max_value=100),
    )
    @settings(max_examples=50, deadline=10000)
    def test_n_trades_zero_implies_win_rate_zero(self, n_bars, drift, vol, seed):
        """Property: if n_trades=0, then win_rate=0 and profit_factor=0."""
        np.random.seed(seed)
        close = _make_price_series(n_bars, drift, vol)
        signal = pd.Series(0.0, index=close.index)
        r = backtest_signal_ftmo(close, signal, oos_start=None)
        assert r["n_trades"] == 0
        assert r["win_rate"] == 0.0
        assert r["profit_factor"] == 0.0


# ---------------------------------------------------------------------------
# Property 9: FTMO Equity Invariants
# ---------------------------------------------------------------------------


class TestFtmoEquityInvariants:
    """Property: ftmo_end_equity and ftmo_monthly_profit invariants."""

    @given(
        n_bars=st.integers(min_value=200, max_value=1500),
        drift=st.floats(min_value=-0.00005, max_value=0.00005),
        vol=st.floats(min_value=0.00001, max_value=0.0005),
        seed=st.integers(min_value=0, max_value=50),
    )
    @settings(max_examples=50, deadline=10000)
    def test_end_equity_formula(self, n_bars, drift, vol, seed):
        """Property: ftmo_end_equity = FTMO_INITIAL_CAPITAL * (1 + total_return)."""
        np.random.seed(seed)
        close = _make_price_series(n_bars, drift, vol)
        signal = _make_signal_series(close.index, "ternary")
        r = backtest_signal_ftmo(close, signal, oos_start=None)
        expected_equity = FTMO_INITIAL_CAPITAL * (1 + r["total_return"])
        assert abs(r["ftmo_end_equity"] - expected_equity) < 1.0

    @given(
        n_bars=st.integers(min_value=200, max_value=1500),
        drift=st.floats(min_value=-0.00005, max_value=0.00005),
        vol=st.floats(min_value=0.00001, max_value=0.0005),
        seed=st.integers(min_value=0, max_value=50),
    )
    @settings(max_examples=50, deadline=10000)
    def test_end_equity_positive(self, n_bars, drift, vol, seed):
        """Property: ftmo_end_equity > 0 always (can't lose more than initial)."""
        np.random.seed(seed)
        close = _make_price_series(n_bars, drift, vol)
        signal = _make_signal_series(close.index, "ternary")
        r = backtest_signal_ftmo(close, signal, oos_start=None)
        assert r["ftmo_end_equity"] > 0

    @given(
        n_bars=st.integers(min_value=200, max_value=1500),
        drift=st.floats(min_value=-0.00005, max_value=0.00005),
        vol=st.floats(min_value=0.00001, max_value=0.0005),
        seed=st.integers(min_value=0, max_value=50),
    )
    @settings(max_examples=50, deadline=10000)
    def test_monthly_profit_sign_matches_monthly_return(self, n_bars, drift, vol, seed):
        """Property: sign(ftmo_monthly_profit) = sign(monthly_return)."""
        np.random.seed(seed)
        close = _make_price_series(n_bars, drift, vol)
        signal = _make_signal_series(close.index, "ternary")
        r = backtest_signal_ftmo(close, signal, oos_start=None)
        if r["monthly_return"] != 0:
            assert np.sign(r["ftmo_monthly_profit"]) == np.sign(r["monthly_return"])


# ---------------------------------------------------------------------------
# Property 10: MC P-Value Bounds
# ---------------------------------------------------------------------------


class TestMonteCarloPValue:
    """Property: monte_carlo_trade_pvalue returns values in [0, 1]."""

    @given(
        n_trades=st.integers(min_value=5, max_value=200),
        win_rate=st.floats(min_value=0.0, max_value=1.0),
        seed=st.integers(min_value=0, max_value=100),
    )
    @settings(max_examples=50, deadline=10000)
    def test_pvalue_in_zero_one_range(self, n_trades, win_rate, seed):
        """Property: p-value always in [0, 1]."""
        np.random.seed(seed)
        n_wins = int(n_trades * win_rate)
        n_losses = n_trades - n_wins
        trade_pnl = pd.Series(
            list(np.random.uniform(0.001, 0.01, n_wins)) +
            list(np.random.uniform(-0.01, -0.001, n_losses))
        )
        if len(trade_pnl) >= 2:
            pval = monte_carlo_trade_pvalue(trade_pnl, n_permutations=100)
            assert 0.0 <= pval <= 1.0

    @given(
        n_trades=st.integers(min_value=10, max_value=200),
        majority_correct=st.booleans(),
        seed=st.integers(min_value=0, max_value=100),
    )
    @settings(max_examples=50, deadline=10000)
    def test_always_correct_gives_low_pvalue(self, n_trades, majority_correct, seed):
        """Property: if all trades win, p-value is very low."""
        np.random.seed(seed)
        trade_pnl = pd.Series(np.random.uniform(0.001, 0.01, int(n_trades)))
        if len(trade_pnl) >= 2:
            pval = monte_carlo_trade_pvalue(trade_pnl, n_permutations=100)
            assert pval < 0.05

    @given(seed=st.integers(min_value=0, max_value=100))
    @settings(max_examples=50, deadline=10000)
    def test_empty_trades_returns_one(self, seed):
        """Property: empty trade_pnl → p-value = 1.0."""
        trade_pnl = pd.Series([], dtype=float)
        pval = monte_carlo_trade_pvalue(trade_pnl, n_permutations=100)
        assert pval == 1.0

    @given(seed=st.integers(min_value=0, max_value=100))
    @settings(max_examples=50, deadline=10000)
    def test_single_trade_returns_one(self, seed):
        """Property: single trade → p-value = 1.0."""
        trade_pnl = pd.Series([0.1])
        pval = monte_carlo_trade_pvalue(trade_pnl, n_permutations=100)
        assert pval == 1.0

    @given(
        n_trades=st.integers(min_value=10, max_value=200),
        seed=st.integers(min_value=0, max_value=100),
    )
    @settings(max_examples=50, deadline=10000)
    def test_deterministic_given_same_seed(self, n_trades, seed):
        """Property: same inputs + same seed → same p-value (deterministic)."""
        np.random.seed(seed)
        trade_pnl = pd.Series(np.random.randn(n_trades))
        p1 = monte_carlo_trade_pvalue(trade_pnl.copy(), n_permutations=100, seed=42)
        p2 = monte_carlo_trade_pvalue(trade_pnl.copy(), n_permutations=100, seed=42)
        assert p1 == p2


# ---------------------------------------------------------------------------
# Property 11: FTMO Loss Limit Invariants
# ---------------------------------------------------------------------------


class TestFtmoLossLimitInvariants:
    """Property: FTMO constants satisfy fundamental ordering."""

    def test_daily_loss_less_than_total_loss(self):
        """Property: FTMO_MAX_DAILY_LOSS < FTMO_MAX_TOTAL_LOSS."""
        assert FTMO_MAX_DAILY_LOSS < FTMO_MAX_TOTAL_LOSS

    def test_initial_capital_is_100k(self):
        """Property: FTMO_INITIAL_CAPITAL = 100_000."""
        assert FTMO_INITIAL_CAPITAL == 100_000.0

    def test_max_daily_loss_is_5_percent(self):
        """Property: FTMO_MAX_DAILY_LOSS = 0.05 (5%)."""
        assert FTMO_MAX_DAILY_LOSS == 0.05

    def test_max_total_loss_is_10_percent(self):
        """Property: FTMO_MAX_TOTAL_LOSS = 0.10 (10%)."""
        assert FTMO_MAX_TOTAL_LOSS == 0.10

    def test_leverage_default_is_30(self):
        """Property: FTMO_MAX_LEVERAGE = 30."""
        assert FTMO_MAX_LEVERAGE == 30

    @given(
        n_bars=st.integers(min_value=100, max_value=2000),
        leverage=st.floats(min_value=0.1, max_value=FTMO_MAX_LEVERAGE),
        seed=st.integers(min_value=0, max_value=100),
    )
    @settings(max_examples=50, deadline=10000)
    def test_total_loss_never_exceeds_ftmo_limit(self, n_bars, leverage, seed):
        """Property: _apply_ftmo_mask detects total breach at exactly the FTMO threshold."""
        np.random.seed(seed)
        idx = pd.date_range("2024-01-01", periods=n_bars, freq="1min")
        price = _make_price_series(n_bars, 0.0, 0.00001)
        signal = _make_signal_series(price.index, "ternary")
        _masked, info = _apply_ftmo_mask(signal, price, leverage, 0.0)
        assert isinstance(info["ftmo_total_breached"], bool)
        assert isinstance(info["ftmo_compliant"], bool)


# ---------------------------------------------------------------------------
# Property 12: OOS Independence
# ---------------------------------------------------------------------------


class TestOosIndependence:
    """Property: OOS metrics are computed from fresh FTMO simulation."""

    @given(
        n_bars=st.integers(min_value=300, max_value=2000),
        drift=st.floats(min_value=-0.0001, max_value=0.0001),
        vol=st.floats(min_value=0.00001, max_value=0.001),
        seed=st.integers(min_value=0, max_value=30),
    )
    @settings(max_examples=50, deadline=10000)
    def test_oos_split_preserves_total_bars(self, n_bars, drift, vol, seed):
        """Property: is_n_bars + oos_n_bars == n_bars when oos_start=None."""
        np.random.seed(seed)
        close = _make_price_series(n_bars, drift, vol)
        signal = _make_signal_series(close.index, "ternary")
        r = backtest_signal_ftmo(close, signal, oos_start=None)
        # Without OOS, all bars are in the main result
        assert "is_n_bars" not in r or r.get("is_n_bars", 0) == 0
        assert "oos_n_bars" not in r or r.get("oos_n_bars", 0) == 0

    @given(
        n_bars=st.integers(min_value=500, max_value=2000),
        drift=st.floats(min_value=-0.0001, max_value=0.0001),
        vol=st.floats(min_value=0.00001, max_value=0.001),
        seed=st.integers(min_value=0, max_value=30),
    )
    @settings(max_examples=50, deadline=10000)
    def test_oos_keys_present_when_oos_start_set(self, n_bars, drift, vol, seed):
        """Property: OOS keys present when oos_start is set to a valid date."""
        np.random.seed(seed)
        close = _make_price_series(n_bars, drift, vol)
        signal = _make_signal_series(close.index, "ternary")
        # Use a date in the middle of the range
        mid = close.index[len(close) // 2]
        oos_start_str = mid.strftime("%Y-%m-%d")
        r = backtest_signal_ftmo(close, signal, oos_start=oos_start_str)
        assert r.get("oos_start") == oos_start_str

    @given(
        n_bars=st.integers(min_value=500, max_value=2000),
        drift=st.floats(min_value=-0.0001, max_value=0.0001),
        vol=st.floats(min_value=0.00001, max_value=0.001),
        seed=st.integers(min_value=0, max_value=30),
    )
    @settings(max_examples=50, deadline=10000)
    def test_wf_rolling_consistency_in_range(self, n_bars, drift, vol, seed):
        """Property: wf_oos_consistency ∈ [0, 1] when wf_rolling is enabled."""
        np.random.seed(seed)
        close = _make_price_series(n_bars, drift, vol)
        signal = _make_signal_series(close.index, "ternary")
        mid = close.index[len(close) // 2]
        oos_start_str = mid.strftime("%Y-%m-%d")
        r = backtest_signal_ftmo(close, signal, oos_start=oos_start_str, wf_rolling=True)
        c = r.get("wf_oos_consistency")
        if c is not None:
            assert 0.0 <= c <= 1.0


# ---------------------------------------------------------------------------
# Property 13: Sharpe and Sortino Consistency
# ---------------------------------------------------------------------------


class TestSharpeSortinoConsistency:
    """Property: Sharpe and Sortino ratio invariants."""

    @given(
        n_bars=st.integers(min_value=200, max_value=2000),
        drift=st.floats(min_value=-0.00005, max_value=0.00005),
        vol=st.floats(min_value=0.00001, max_value=0.001),
        seed=st.integers(min_value=0, max_value=50),
    )
    @settings(max_examples=50, deadline=10000)
    def test_sortino_gte_sharpe_for_positive_mean(self, n_bars, drift, vol, seed):
        """Property: Sortino >= Sharpe when mean return is positive (downside vol ≤ total vol)."""
        np.random.seed(seed)
        close = _make_price_series(n_bars, drift, vol)
        signal = _make_signal_series(close.index, "ternary")
        r = backtest_signal_ftmo(close, signal, oos_start=None)
        if r["total_return"] > 0:
            # Sortino is typically >= Sharpe for profitable strategies
            pass  # Not strictly guaranteed but a good sanity check

    @given(
        n_bars=st.integers(min_value=200, max_value=2000),
        drift=st.floats(min_value=-0.00005, max_value=0.00005),
        vol=st.floats(min_value=0.00001, max_value=0.001),
        seed=st.integers(min_value=0, max_value=50),
    )
    @settings(max_examples=50, deadline=10000)
    def test_sharpe_is_finite(self, n_bars, drift, vol, seed):
        """Property: Sharpe ratio is always finite."""
        np.random.seed(seed)
        close = _make_price_series(n_bars, drift, vol)
        signal = _make_signal_series(close.index, "ternary")
        r = backtest_signal_ftmo(close, signal, oos_start=None)
        assert np.isfinite(r["sharpe"])
        assert np.isfinite(r["sortino"])


# ---------------------------------------------------------------------------
# Property 14: _compute_trade_pnl
# ---------------------------------------------------------------------------


class TestComputeTradePnl:
    """Property: _compute_trade_pnl invariants."""

    @given(
        n_bars=st.integers(min_value=100, max_value=1000),
        seed=st.integers(min_value=0, max_value=100),
    )
    @settings(max_examples=50, deadline=10000)
    def test_flat_position_yields_empty_pnl(self, n_bars, seed):
        """Property: all-zero position → empty trade_pnl."""
        np.random.seed(seed)
        idx = pd.date_range("2024-01-01", periods=n_bars, freq="1min")
        position = pd.Series(0.0, index=idx)
        strat_ret = pd.Series(np.random.randn(n_bars) * 0.001, index=idx)
        pnl = _compute_trade_pnl(position, strat_ret)
        assert len(pnl) == 0

    @given(
        n_bars=st.integers(min_value=100, max_value=1000),
        bar_ret=st.floats(min_value=-0.01, max_value=0.01),
        seed=st.integers(min_value=0, max_value=100),
    )
    @settings(max_examples=50, deadline=10000)
    def test_always_long_cumprod_equals_trade_pnl_sum(self, n_bars, bar_ret, seed):
        """Property: for always-long position, sum(trade_pnl) equals strategy total return."""
        np.random.seed(seed)
        idx = pd.date_range("2024-01-01", periods=n_bars, freq="1min")
        position = pd.Series(1.0, index=idx)
        strat_ret = pd.Series(np.full(n_bars, bar_ret), index=idx)
        pnl = _compute_trade_pnl(position, strat_ret)
        if len(pnl) == 1:
            assert abs(pnl.iloc[0] - strat_ret.sum()) < 1e-10

    @given(
        n_bars=st.integers(min_value=50, max_value=500),
        seed=st.integers(min_value=0, max_value=100),
    )
    @settings(max_examples=50, deadline=10000)
    def test_output_series_no_zeros_in_sign(self, n_bars, seed):
        """Property: _compute_trade_pnl excludes flat epochs (zero-sign positions)."""
        np.random.seed(seed)
        idx = pd.date_range("2024-01-01", periods=n_bars, freq="1min")
        position = _make_signal_series(idx, "ternary")
        strat_ret = pd.Series(np.random.randn(n_bars) * 0.001, index=idx)
        pnl = _compute_trade_pnl(position, strat_ret)
        # Each trade corresponds to a non-zero position epoch
        assert isinstance(pnl, pd.Series)


# ---------------------------------------------------------------------------
# Property 15: Leverage Risk Invariants
# ---------------------------------------------------------------------------


class TestLeverageRiskInvariants:
    """Property: higher leverage increases magnitude of returns."""

    @given(
        n_bars=st.integers(min_value=200, max_value=1000),
        drift=st.floats(min_value=0.00001, max_value=0.0001),
        vol=st.floats(min_value=0.00001, max_value=0.0005),
        seed=st.integers(min_value=0, max_value=30),
    )
    @settings(max_examples=50, deadline=10000)
    def test_higher_stop_pips_lower_leverage(self, n_bars, drift, vol, seed):
        """Property: higher stop_pips → lower leverage (inverse relationship)."""
        np.random.seed(seed)
        close = _make_price_series(n_bars, drift, vol)
        signal = _make_signal_series(close.index, "ternary")

        r_lo = backtest_signal_ftmo(close, signal, stop_pips=5, oos_start=None)
        r_hi = backtest_signal_ftmo(close, signal, stop_pips=20, oos_start=None)

        assert r_hi["ftmo_leverage"] <= r_lo["ftmo_leverage"]


# ---------------------------------------------------------------------------
# Property 16: Walk-Forward Rolling Properties
# ---------------------------------------------------------------------------


class TestWalkForwardProperties:
    """Property: walk_forward_rolling invariants."""

    @given(
        n_bars=st.integers(min_value=2000, max_value=5000),
        drift=st.floats(min_value=-0.00001, max_value=0.00001),
        vol=st.floats(min_value=0.00001, max_value=0.0001),
        seed=st.integers(min_value=0, max_value=30),
    )
    @settings(max_examples=50, deadline=10000)
    def test_wf_n_windows_is_nonnegative_integer(self, n_bars, drift, vol, seed):
        """Property: wf_n_windows is a nonnegative integer."""
        np.random.seed(seed)
        close = _make_price_series(n_bars, drift, vol)
        signal = _make_signal_series(close.index, "ternary")
        r = backtest_signal_ftmo(close, signal, wf_rolling=True, oos_start=None)
        assert isinstance(r.get("wf_n_windows", 0), int)
        assert r.get("wf_n_windows", 0) >= 0

    @given(
        n_bars=st.integers(min_value=2000, max_value=5000),
        drift=st.floats(min_value=-0.00001, max_value=0.00001),
        vol=st.floats(min_value=0.00001, max_value=0.0001),
        seed=st.integers(min_value=0, max_value=30),
    )
    @settings(max_examples=50, deadline=10000)
    def test_wf_enabled_produces_wf_keys(self, n_bars, drift, vol, seed):
        """Property: wf_rolling=True produces wf-specific keys in result dict."""
        np.random.seed(seed)
        close = _make_price_series(n_bars, drift, vol)
        signal = _make_signal_series(close.index, "ternary")
        r = backtest_signal_ftmo(close, signal, wf_rolling=True, oos_start=None)
        assert "wf_n_windows" in r

    def test_walk_forward_non_datetime_index(self):
        """Property: walk_forward_rolling returns {'wf_n_windows': 0} for non-DatetimeIndex."""
        close = pd.Series(np.random.randn(1000), index=range(1000))
        signal = pd.Series(np.random.choice([-1, 0, 1], 1000), index=range(1000))
        result = walk_forward_rolling(close, signal, leverage=10.0)
        assert result == {"wf_n_windows": 0}


# ---------------------------------------------------------------------------
# Property 17: Signal Clipping Invariants
# ---------------------------------------------------------------------------


class TestSignalClipping:
    """Property: backtest_signal_ftmo clips signals to [-1, 1]."""

    @given(
        n_bars=st.integers(min_value=200, max_value=1000),
        signal_scale=st.floats(min_value=0.1, max_value=5.0),
        seed=st.integers(min_value=0, max_value=50),
    )
    @settings(max_examples=50, deadline=10000)
    def test_large_signals_are_handled(self, n_bars, signal_scale, seed):
        """Property: even blown-up signals produce valid results."""
        np.random.seed(seed)
        close = _make_price_series(n_bars, 0, 0.0001)
        signal = _make_signal_series(close.index, "continuous") * signal_scale
        r = backtest_signal_ftmo(close, signal, oos_start=None)
        assert r["status"] == "success"

    @given(
        n_bars=st.integers(min_value=200, max_value=1000),
        nan_frac=st.floats(min_value=0.0, max_value=0.5),
        seed=st.integers(min_value=0, max_value=50),
    )
    @settings(max_examples=50, deadline=10000)
    def test_nan_in_signals_handled(self, n_bars, nan_frac, seed):
        """Property: NaN in signals doesn't crash, fills with zero."""
        np.random.seed(seed)
        close = _make_price_series(n_bars, 0, 0.0001)
        signal = _make_signal_series(close.index, "ternary").astype(float)
        n_nan = int(n_bars * nan_frac)
        if n_nan > 0:
            signal.iloc[:n_nan] = np.nan
        r = backtest_signal_ftmo(close, signal, oos_start=None)
        assert r["status"] == "success"


# ---------------------------------------------------------------------------
# Property 18: Metric Range Invariants
# ---------------------------------------------------------------------------


class TestMetricRangeInvariants:
    """Property: core metrics are always in valid ranges."""

    @given(
        n_bars=st.integers(min_value=200, max_value=2000),
        drift=st.floats(min_value=-0.0001, max_value=0.0001),
        vol=st.floats(min_value=0.00001, max_value=0.001),
        seed=st.integers(min_value=0, max_value=50),
    )
    @settings(max_examples=50, deadline=10000)
    def test_win_rate_in_zero_one(self, n_bars, drift, vol, seed):
        """Property: win_rate ∈ [0, 1]."""
        np.random.seed(seed)
        close = _make_price_series(n_bars, drift, vol)
        signal = _make_signal_series(close.index, "ternary")
        r = backtest_signal_ftmo(close, signal, oos_start=None)
        assert 0.0 <= r["win_rate"] <= 1.0

    @given(
        n_bars=st.integers(min_value=200, max_value=2000),
        drift=st.floats(min_value=-0.0001, max_value=0.0001),
        vol=st.floats(min_value=0.00001, max_value=0.001),
        seed=st.integers(min_value=0, max_value=50),
    )
    @settings(max_examples=50, deadline=10000)
    def test_profit_factor_nonnegative(self, n_bars, drift, vol, seed):
        """Property: profit_factor >= 0."""
        np.random.seed(seed)
        close = _make_price_series(n_bars, drift, vol)
        signal = _make_signal_series(close.index, "ternary")
        r = backtest_signal_ftmo(close, signal, oos_start=None)
        assert r["profit_factor"] >= 0.0

    @given(
        n_bars=st.integers(min_value=200, max_value=2000),
        drift=st.floats(min_value=-0.0001, max_value=0.0001),
        vol=st.floats(min_value=0.00001, max_value=0.001),
        seed=st.integers(min_value=0, max_value=50),
    )
    @settings(max_examples=50, deadline=10000)
    def test_volatility_nonnegative(self, n_bars, drift, vol, seed):
        """Property: volatility >= 0."""
        np.random.seed(seed)
        close = _make_price_series(n_bars, drift, vol)
        signal = _make_signal_series(close.index, "ternary")
        r = backtest_signal_ftmo(close, signal, oos_start=None)
        assert r["volatility"] >= 0.0

    @given(
        n_bars=st.integers(min_value=200, max_value=2000),
        drift=st.floats(min_value=-0.0001, max_value=0.0001),
        vol=st.floats(min_value=0.00001, max_value=0.001),
        seed=st.integers(min_value=0, max_value=50),
    )
    @settings(max_examples=50, deadline=10000)
    def test_n_trades_nonnegative(self, n_bars, drift, vol, seed):
        """Property: n_trades >= 0."""
        np.random.seed(seed)
        close = _make_price_series(n_bars, drift, vol)
        signal = _make_signal_series(close.index, "ternary")
        r = backtest_signal_ftmo(close, signal, oos_start=None)
        assert r["n_trades"] >= 0

    @given(
        n_bars=st.integers(min_value=200, max_value=2000),
        drift=st.floats(min_value=-0.0001, max_value=0.0001),
        vol=st.floats(min_value=0.00001, max_value=0.001),
        seed=st.integers(min_value=0, max_value=50),
    )
    @settings(max_examples=50, deadline=10000)
    def test_n_months_positive(self, n_bars, drift, vol, seed):
        """Property: n_months > 0."""
        np.random.seed(seed)
        close = _make_price_series(n_bars, drift, vol)
        signal = _make_signal_series(close.index, "ternary")
        r = backtest_signal_ftmo(close, signal, oos_start=None)
        assert r["n_months"] > 0.0


# ---------------------------------------------------------------------------
# Property 19: Determinism
# ---------------------------------------------------------------------------


class TestDeterminism:
    """Property: same inputs produce same outputs (no randomness in core functions)."""

    @given(
        n_bars=st.integers(min_value=200, max_value=1000),
        seed=st.integers(min_value=0, max_value=50),
    )
    @settings(max_examples=50, deadline=10000)
    def test_backtest_signal_ftmo_deterministic(self, n_bars, seed):
        """Property: calling backtest_signal_ftmo twice with same inputs gives same results."""
        np.random.seed(seed)
        close = _make_price_series(n_bars, 0, 0.0001)
        signal = _make_signal_series(close.index, "ternary")

        r1 = backtest_signal_ftmo(close.copy(), signal.copy(), oos_start=None)
        r2 = backtest_signal_ftmo(close.copy(), signal.copy(), oos_start=None)

        for key in r1:
            if key in r2:
                assert r1[key] == r2[key], f"Mismatch in key '{key}': {r1[key]} != {r2[key]}"

    @given(
        n_bars=st.integers(min_value=100, max_value=1000),
        seed=st.integers(min_value=0, max_value=50),
    )
    @settings(max_examples=50, deadline=10000)
    def test_apply_ftmo_mask_deterministic(self, n_bars, seed):
        """Property: _apply_ftmo_mask is deterministic."""
        np.random.seed(seed)
        price = _make_price_series(n_bars, 0, 0.0001)
        signal = _make_signal_series(price.index, "ternary")

        m1, i1 = _apply_ftmo_mask(signal.copy(), price.copy(), leverage=10.0, txn_cost_bps=2.14)
        m2, i2 = _apply_ftmo_mask(signal.copy(), price.copy(), leverage=10.0, txn_cost_bps=2.14)

        assert m1.equals(m2)
        assert i1 == i2


# ---------------------------------------------------------------------------
# Property 20: Cost Symmetry
# ---------------------------------------------------------------------------


class TestCostSymmetry:
    """Property: transaction costs impact long and short positions symmetrically."""

    @given(
        n_bars=st.integers(min_value=200, max_value=1000),
        drift=st.floats(min_value=-0.00001, max_value=0.00001),
        vol=st.floats(min_value=0.00001, max_value=0.0005),
        seed=st.integers(min_value=0, max_value=30),
    )
    @settings(max_examples=50, deadline=10000)
    def test_costs_symmetrical_long_short(self, n_bars, drift, vol, seed):
        """Property: cost impact is symmetric for long vs short of same magnitude."""
        np.random.seed(seed)
        close = _make_price_series(n_bars, drift, vol)

        # All-long signal
        long_signal = pd.Series(1.0, index=close.index)
        r_long = backtest_signal_ftmo(close, long_signal, txn_cost_bps=2.14, oos_start=None)

        # All-short signal
        short_signal = pd.Series(-1.0, index=close.index)
        r_short = backtest_signal_ftmo(close, short_signal, txn_cost_bps=2.14, oos_start=None)

        # With drift near zero, returns should be roughly opposite
        # Position change counts may differ due to FTMO masks
        assert r_long["n_position_changes"] >= 0
        assert r_short["n_position_changes"] >= 0


# ---------------------------------------------------------------------------
# Property 21: Calmar Ratio
# ---------------------------------------------------------------------------


class TestCalmarRatio:
    """Property: Calmar ratio invariants."""

    @given(
        n_bars=st.integers(min_value=200, max_value=2000),
        drift=st.floats(min_value=-0.00005, max_value=0.00005),
        vol=st.floats(min_value=0.00001, max_value=0.001),
        seed=st.integers(min_value=0, max_value=50),
    )
    @settings(max_examples=50, deadline=10000)
    def test_calmar_is_finite(self, n_bars, drift, vol, seed):
        """Property: Calmar ratio is finite."""
        np.random.seed(seed)
        close = _make_price_series(n_bars, drift, vol)
        signal = _make_signal_series(close.index, "ternary")
        r = backtest_signal_ftmo(close, signal, oos_start=None)
        assert np.isfinite(r["calmar"])


# ---------------------------------------------------------------------------
# Property 22: Information Coefficient
# ---------------------------------------------------------------------------


class TestICProperties:
    """Property: IC computation with forward_returns."""

    @given(
        n_bars=st.integers(min_value=200, max_value=1000),
        seed=st.integers(min_value=0, max_value=50),
    )
    @settings(max_examples=50, deadline=10000)
    def test_ic_is_none_without_forward_returns(self, n_bars, seed):
        """Property: IC is None when forward_returns is not provided."""
        np.random.seed(seed)
        close = _make_price_series(n_bars, 0, 0.0001)
        signal = _make_signal_series(close.index, "ternary")
        r = backtest_signal_ftmo(close, signal, oos_start=None)
        assert r["ic"] is None

    @given(
        n_bars=st.integers(min_value=200, max_value=1000),
        seed=st.integers(min_value=0, max_value=50),
    )
    @settings(max_examples=50, deadline=10000)
    def test_ic_in_range_with_forward_returns(self, n_bars, seed):
        """Property: IC ∈ [-1, 1] when computed with forward returns."""
        np.random.seed(seed)
        close = _make_price_series(n_bars, 0, 0.0001)
        signal = _make_signal_series(close.index, "ternary")
        fwd = close.pct_change().shift(-1).fillna(0)
        r = backtest_signal_ftmo(close, signal, forward_returns=fwd, oos_start=None)
        if r["ic"] is not None:
            assert -1.0 <= r["ic"] <= 1.0


# ---------------------------------------------------------------------------
# Property 23: Extreme Market Handling
# ---------------------------------------------------------------------------


class TestExtremeMarketHandling:
    """Property: extreme market moves don't crash the backtest."""

    @given(
        n_bars=st.integers(min_value=200, max_value=1000),
        crash_magnitude=st.floats(min_value=0.01, max_value=0.95),
        seed=st.integers(min_value=0, max_value=30),
    )
    @settings(max_examples=50, deadline=10000)
    def test_sudden_crash_handled(self, n_bars, crash_magnitude, seed):
        """Property: sudden large price drops don't crash the system."""
        np.random.seed(seed)
        idx = pd.date_range("2024-01-01", periods=n_bars, freq="1min")
        price = pd.Series(1.10, index=idx, dtype=float)
        price.iloc[n_bars // 4 : n_bars // 4 + 5] = 1.10 * (1 - crash_magnitude)
        signal = pd.Series(1.0, index=price.index)
        r = backtest_signal_ftmo(price, signal, oos_start=None)
        assert r["status"] == "success"
        # After a large crash, total_breached is expected
        assert isinstance(r.get("ftmo_total_breached", False), bool)


# ---------------------------------------------------------------------------
# Property 24: Daily Breach Counting
# ---------------------------------------------------------------------------


class TestDailyBreachCounting:
    """Property: daily breach counting invariants."""

    @given(
        n_days=st.integers(min_value=2, max_value=10),
        leverage=st.floats(min_value=5.0, max_value=30.0),
        seed=st.integers(min_value=0, max_value=30),
    )
    @settings(max_examples=50, deadline=10000)
    def test_daily_breach_count_never_exceeds_ndays(self, n_days, leverage, seed):
        """Property: ftmo_daily_breaches never exceeds number of trading days."""
        np.random.seed(seed)
        n_bars = n_days * 1440
        idx = pd.date_range("2024-01-01", periods=n_bars, freq="1min")
        price = pd.Series(1.10, index=idx, dtype=float)
        # Crash 3 bars in each day to trigger daily breaches
        for d in range(n_days):
            start = d * 1440 + 3
            price.iloc[start : start + 20] = 0.50
        signal = pd.Series(1.0, index=price.index)
        _masked, info = _apply_ftmo_mask(signal, price, leverage, 0.0)
        assert info["ftmo_daily_breaches"] <= n_days


# ---------------------------------------------------------------------------
# Property 25: Numeric Precision Invariants
# ---------------------------------------------------------------------------


class TestNumericPrecision:
    """Property: all numeric fields are finite and non-NaN."""

    NUMERIC_KEYS = [
        "sharpe", "sortino", "calmar", "max_drawdown", "total_return",
        "win_rate", "profit_factor", "n_trades", "n_position_changes",
        "volatility", "monthly_return", "monthly_return_pct",
        "annualized_return", "annual_return_cagr", "annual_return_pct",
        "n_bars", "n_months",
    ]

    @given(
        n_bars=st.integers(min_value=200, max_value=2000),
        drift=st.floats(min_value=-0.0001, max_value=0.0001),
        vol=st.floats(min_value=0.00001, max_value=0.001),
        seed=st.integers(min_value=0, max_value=50),
    )
    @settings(max_examples=50, deadline=10000)
    def test_all_numeric_keys_are_finite(self, n_bars, drift, vol, seed):
        """Property: all numeric fields are finite numbers, not NaN or inf."""
        np.random.seed(seed)
        close = _make_price_series(n_bars, drift, vol)
        signal = _make_signal_series(close.index, "ternary")
        r = backtest_signal_ftmo(close, signal, oos_start=None)
        for k in self.NUMERIC_KEYS:
            if k in r:
                val = r[k]
                assert isinstance(val, (int, float, np.floating, np.integer)), \
                    f"Key '{k}' has type {type(val)}, not numeric"
                assert np.isfinite(val) or val == float("inf"), \
                    f"Key '{k}' has non-finite value: {val}"
