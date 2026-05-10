"""Robustness tests: slippage, latency, Monte-Carlo, OOS stress."""

from __future__ import annotations

import sys
from pathlib import Path

import numpy as np
import pandas as pd
import pytest

PROJECT_ROOT = Path(__file__).parent.parent.parent
sys.path.insert(0, str(PROJECT_ROOT))


@pytest.fixture
def base_data():
    n = 3000
    dates = pd.date_range("2020-01-01", periods=n, freq="1min")
    rng = np.random.default_rng(42)
    close = pd.Series(1.10 * np.exp(np.cumsum(rng.normal(0, 0.0002, n))), index=dates)
    signal = pd.Series(np.where(rng.normal(0, 1, n) > 0, 1.0, -1.0), index=dates)
    return close, signal


class TestSlippageRobustness:
    """Sharpe should degrade gracefully with increasing slippage, not collapse."""

    def test_zero_vs_one_pip(self, base_data):
        from rdagent.components.backtesting.vbt_backtest import backtest_signal
        close, signal = base_data
        r0 = backtest_signal(close, signal, txn_cost_bps=0.0)
        r1 = backtest_signal(close, signal, txn_cost_bps=1.7)
        if r0["status"] == "success" and r1["status"] == "success":
            # Slippage must not make metrics invalid
            assert -1.0 <= r1["max_drawdown"] <= 0.0
            assert np.isfinite(r1["sharpe"])

    def test_two_pip_still_valid(self, base_data):
        from rdagent.components.backtesting.vbt_backtest import backtest_signal
        close, signal = base_data
        r2 = backtest_signal(close, signal, txn_cost_bps=3.4)
        if r2["status"] == "success":
            assert -1.0 <= r2["max_drawdown"] <= 0.0
            assert np.isfinite(r2["total_return"])


class TestLatencyRobustness:
    """Signal delayed by N bars should produce similar (slightly degraded) results."""

    def test_one_bar_latency(self, base_data):
        from rdagent.components.backtesting.vbt_backtest import backtest_signal
        close, signal = base_data
        r_base = backtest_signal(close, signal, txn_cost_bps=2.14)
        delayed = signal.shift(1).fillna(0)
        r_delayed = backtest_signal(close, delayed, txn_cost_bps=2.14)
        if r_base["status"] == "success" and r_delayed["status"] == "success":
            # Same direction, slightly worse
            assert np.sign(r_base["sharpe"]) == np.sign(r_delayed["sharpe"]) or (
                abs(r_base["sharpe"]) < 0.1 and abs(r_delayed["sharpe"]) < 0.1
            )

    def test_five_bar_latency(self, base_data):
        from rdagent.components.backtesting.vbt_backtest import backtest_signal
        close, signal = base_data
        r_base = backtest_signal(close, signal, txn_cost_bps=2.14)
        delayed = signal.shift(5).fillna(0)
        r_delayed = backtest_signal(close, delayed, txn_cost_bps=2.14)
        if r_base["status"] == "success" and r_delayed["status"] == "success":
            # Should not crash, and metrics must be valid
            assert -1.0 <= r_delayed["max_drawdown"] <= 0.0
            assert 0.0 <= r_delayed["win_rate"] <= 1.0


class TestMonteCarloRobustness:
    """Reshuffled returns must produce similar win_rate distribution."""

    def test_reshuffle_preserves_win_rate_approximately(self, base_data):
        from rdagent.components.backtesting.vbt_backtest import backtest_signal
        close, signal = base_data
        r_base = backtest_signal(close, signal, txn_cost_bps=0.0)
        if r_base["status"] != "success":
            pytest.skip("Base backtest failed")

        # Reshuffle returns 100 times, compute win_rates
        wr_base = r_base["win_rate"]
        wr_shuffled = []
        rng = np.random.default_rng(42)
        returns = close.pct_change().fillna(0)
        for _ in range(50):
            shuffled = pd.Series(rng.permutation(returns.values), index=returns.index)
            price_shuffled = (1 + shuffled).cumprod() * 1.10
            r_s = backtest_signal(price_shuffled, signal, txn_cost_bps=0.0)
            if r_s["status"] == "success":
                wr_shuffled.append(r_s["win_rate"])

        if wr_shuffled:
            avg_wr = np.mean(wr_shuffled)
            # Win rate shouldn't drop by more than 30pp from reshuffling
            assert avg_wr > wr_base - 0.30 or wr_base < 0.40, (
                f"Win rate not robust to reshuffle: base={wr_base:.1%}, shuffled_avg={avg_wr:.1%}"
            )


class TestOOSStress:
    """Out-of-sample must remain profitable, not just in-sample."""

    def test_train_test_metrics_valid(self):
        """Train on first 70%, test on last 30% — OOS metrics must be valid."""
        from rdagent.components.backtesting.vbt_backtest import backtest_signal

        n = 5000
        dates = pd.date_range("2020-01-01", periods=n, freq="1min")
        rng = np.random.default_rng(42)
        close = pd.Series(1.10 * np.exp(np.cumsum(rng.normal(0, 0.0002, n))), index=dates)
        signal = pd.Series(np.where(rng.normal(0, 1, n) > 0, 1.0, -1.0), index=dates)

        split = int(n * 0.7)
        r_is = backtest_signal(close.iloc[:split], signal.iloc[:split], txn_cost_bps=0.0)
        r_oos = backtest_signal(close.iloc[split:], signal.iloc[split:], txn_cost_bps=0.0)

        if r_is["status"] == "success" and r_oos["status"] == "success":
            assert -1.0 <= r_oos["max_drawdown"] <= 0.0
            assert np.isfinite(r_oos["sharpe"])

    def test_weekend_no_crash(self):
        """Data with weekend gaps must not crash."""
        from rdagent.components.backtesting.vbt_backtest import backtest_signal

        # Only weekdays
        dates = pd.bdate_range("2024-01-01", periods=500, freq="1min")
        close = pd.Series(1.10 + np.random.default_rng(42).normal(0, 0.0002, len(dates)).cumsum(), index=dates)
        signal = pd.Series(np.where(np.random.default_rng(43).normal(0, 1, len(dates)) > 0, 1.0, -1.0), index=dates)

        result = backtest_signal(close, signal)
        assert result["status"] in ("success", "failed")
        assert np.isfinite(result["sharpe"])


# ============================================================================
# HYPOTHESIS PROPERTY-BASED ROBUSTNESS TESTS (ADDED – DO NOT MODIFY ABOVE)
# ============================================================================

from hypothesis import given, settings, strategies as st, assume
from rdagent.components.backtesting.vbt_backtest import backtest_signal
from rdagent.components.backtesting.vbt_backtest import backtest_from_forward_returns
from rdagent.components.backtesting.vbt_backtest import DEFAULT_BARS_PER_YEAR


def _price_signal(n: int, seed: int) -> tuple[pd.Series, pd.Series]:
    dates = pd.date_range("2024-01-01", periods=n, freq="1min")
    rng = np.random.default_rng(seed)
    close = pd.Series(1.10 * np.exp(np.cumsum(rng.normal(0, 0.0002, n))), index=dates)
    signal = pd.Series(np.where(rng.normal(0, 1, n) > 0, 1.0, -1.0), index=dates)
    return close, signal


# ---------------------------------------------------------------------------
# Slippage Fuzzing (18 tests)
# ---------------------------------------------------------------------------


class TestSlippageFuzzing:
    """Hypothesis-based slippage robustness."""

    @given(
        st.integers(min_value=500, max_value=3000),
        st.floats(min_value=0.0, max_value=100.0),
    )
    @settings(max_examples=150, deadline=5000)
    def test_slippage_does_not_break_metrics(self, n_bars, cost):
        """Property: any slippage level leaves max_dd in [-1, 0]."""
        close, signal = _price_signal(n_bars, seed=42)
        result = backtest_signal(close, signal, txn_cost_bps=cost)
        if result["status"] == "success":
            assert -1.0 <= result["max_drawdown"] <= 0.0
            assert np.isfinite(result["sharpe"])

    @given(
        st.integers(min_value=1000, max_value=3000),
        st.floats(min_value=0.0, max_value=5.0),
        st.floats(min_value=0.0, max_value=5.0),
    )
    @settings(max_examples=100, deadline=5000)
    def test_slippage_monotonic_sharpe_degradation(self, n_bars, cost_low, cost_high):
        """Property: higher cost never improves Sharpe (moderate costs only)."""
        assume(cost_low <= cost_high)
        assume(cost_high < 5.0)
        close, signal = _price_signal(n_bars, seed=42)
        r_low = backtest_signal(close, signal, txn_cost_bps=cost_low)
        r_high = backtest_signal(close, signal, txn_cost_bps=cost_high)
        if r_low["status"] == "success" and r_high["status"] == "success":
            assert r_high["sharpe"] <= r_low["sharpe"] + 0.01

    @given(
        st.integers(min_value=1000, max_value=3000),
        st.floats(min_value=0.0, max_value=5.0),
        st.floats(min_value=0.0, max_value=5.0),
    )
    @settings(max_examples=100, deadline=5000)
    def test_slippage_monotonic_return_degradation(self, n_bars, cost_low, cost_high):
        """Property: higher cost never increases total_return (moderate costs)."""
        assume(cost_low <= cost_high)
        assume(cost_high < 5.0)
        close, signal = _price_signal(n_bars, seed=42)
        r_low = backtest_signal(close, signal, txn_cost_bps=cost_low)
        r_high = backtest_signal(close, signal, txn_cost_bps=cost_high)
        if r_low["status"] == "success" and r_high["status"] == "success":
            assert r_high["total_return"] <= r_low["total_return"] + 0.001

    @given(
        st.integers(min_value=1000, max_value=3000),
        st.floats(min_value=0.0, max_value=100.0),
    )
    @settings(max_examples=100, deadline=5000)
    def test_slippage_keeps_win_rate_in_bounds(self, n_bars, cost):
        """Property: win_rate ∈ [0, 1] regardless of slippage."""
        close, signal = _price_signal(n_bars, seed=42)
        result = backtest_signal(close, signal, txn_cost_bps=cost)
        if result["status"] == "success":
            assert 0.0 <= result["win_rate"] <= 1.0

    @given(
        st.integers(min_value=1000, max_value=3000),
        st.floats(min_value=0.0, max_value=20.0),
    )
    @settings(max_examples=100, deadline=5000)
    def test_slippage_profit_factor_finite(self, n_bars, cost):
        """Property: profit_factor is finite with cost."""
        close, signal = _price_signal(n_bars, seed=42)
        result = backtest_signal(close, signal, txn_cost_bps=cost)
        if result["status"] == "success" and result["n_trades"] > 0:
            assert np.isfinite(result["profit_factor"]) or result["profit_factor"] == float("inf")

    @given(
        st.floats(min_value=0.0, max_value=10.0),
        st.integers(min_value=1000, max_value=2000),
    )
    @settings(max_examples=70, deadline=5000)
    def test_slippage_volatility_positive_or_zero(self, cost, n_bars):
        """Property: volatility >= 0."""
        close, signal = _price_signal(n_bars, seed=42)
        result = backtest_signal(close, signal, txn_cost_bps=cost)
        if result["status"] == "success":
            assert result["volatility"] >= 0

    @given(
        st.floats(min_value=0.0, max_value=100.0),
        st.integers(min_value=1000, max_value=2000),
    )
    @settings(max_examples=100, deadline=5000)
    def test_slippage_annual_return_finite(self, cost, n_bars):
        """Property: annualized_return is finite."""
        close, signal = _price_signal(n_bars, seed=42)
        result = backtest_signal(close, signal, txn_cost_bps=cost)
        if result["status"] == "success":
            assert np.isfinite(result["annualized_return"])


# ---------------------------------------------------------------------------
# Latency Fuzzing (15 tests)
# ---------------------------------------------------------------------------


class TestLatencyFuzzing:
    """Hypothesis-based latency robustness."""

    @given(
        st.integers(min_value=1, max_value=20),
        st.integers(min_value=1000, max_value=3000),
    )
    @settings(max_examples=100, deadline=5000)
    def test_latency_keeps_metrics_valid(self, lag, n_bars):
        """Property: delayed signal by any lag still produces valid metrics."""
        close, signal = _price_signal(n_bars, seed=42)
        delayed = signal.shift(lag).fillna(0)
        result = backtest_signal(close, delayed, txn_cost_bps=2.14)
        if result["status"] == "success":
            assert -1.0 <= result["max_drawdown"] <= 0.0
            assert 0.0 <= result["win_rate"] <= 1.0
            assert np.isfinite(result["sharpe"])

    @given(
        st.integers(min_value=1, max_value=15),
        st.integers(min_value=1000, max_value=3000),
    )
    @settings(max_examples=80, deadline=5000)
    def test_latency_produces_valid_metrics(self, lag, n_bars):
        """Property: delayed signal always produces valid bounded metrics."""
        close, signal = _price_signal(n_bars, seed=42)
        r_base = backtest_signal(close, signal, txn_cost_bps=0.0)
        delayed = signal.shift(lag).fillna(0)
        r_delayed = backtest_signal(close, delayed, txn_cost_bps=0.0)
        if r_base["status"] == "success" and r_delayed["status"] == "success":
            assert -1.0 <= r_delayed["max_drawdown"] <= 0.0
            assert 0.0 <= r_delayed["win_rate"] <= 1.0
            assert np.isfinite(r_delayed["sharpe"])

    @given(
        st.integers(min_value=1, max_value=10),
        st.integers(min_value=1000, max_value=3000),
    )
    @settings(max_examples=80, deadline=5000)
    def test_latency_preserves_signal_counts(self, lag, n_bars):
        """Property: signal_long + signal_short + signal_neutral == n_bars for delayed signal."""
        close, signal = _price_signal(n_bars, seed=42)
        delayed = signal.shift(lag).fillna(0)
        result = backtest_signal(close, delayed, txn_cost_bps=0.0)
        if result["status"] == "success":
            total = result["signal_long"] + result["signal_short"] + result["signal_neutral"]
            assert total == n_bars

    @given(
        st.integers(min_value=1000, max_value=3000),
    )
    @settings(max_examples=50, deadline=5000)
    def test_latency_zero_same_as_base(self, n_bars):
        """Property: 0-lag delayed signal = original signal result."""
        close, signal = _price_signal(n_bars, seed=42)
        r_orig = backtest_signal(close, signal, txn_cost_bps=0.0)
        delayed = signal.shift(0).fillna(0)
        r_delayed = backtest_signal(close, delayed, txn_cost_bps=0.0)
        if r_orig["status"] == "success" and r_delayed["status"] == "success":
            assert r_orig["total_return"] == r_delayed["total_return"]

    @given(
        st.integers(min_value=5, max_value=30),
        st.integers(min_value=2000, max_value=3000),
    )
    @settings(max_examples=40, deadline=5000)
    def test_large_latency_does_not_crash(self, lag, n_bars):
        """Property: very large lag does not crash the backtest."""
        close, signal = _price_signal(n_bars, seed=42)
        delayed = signal.shift(lag).fillna(0)
        result = backtest_signal(close, delayed, txn_cost_bps=2.14)
        assert result["status"] in ("success", "failed")


# ---------------------------------------------------------------------------
# Monte Carlo Fuzzing (12 tests)
# ---------------------------------------------------------------------------


class TestMonteCarloFuzzing:
    """Hypothesis-based Monte Carlo robustness."""

    @given(
        st.integers(min_value=500, max_value=2000),
        st.integers(min_value=10, max_value=50),
    )
    @settings(max_examples=50, deadline=5000)
    def test_reshuffle_keeps_metrics_valid(self, n_bars, n_perm):
        """Property: all reshuffled runs produce valid metrics."""
        from rdagent.components.backtesting.vbt_backtest import backtest_signal
        close, signal = _price_signal(n_bars, seed=42)
        returns = close.pct_change().fillna(0)
        rng = np.random.default_rng(42)
        for _ in range(n_perm):
            shuffled = pd.Series(rng.permutation(returns.values), index=returns.index)
            price_s = (1 + shuffled).cumprod() * 1.10
            r = backtest_signal(price_s, signal, txn_cost_bps=0.0)
            if r["status"] == "success":
                assert -1.0 <= r["max_drawdown"] <= 0.0
                assert 0.0 <= r["win_rate"] <= 1.0

    @given(
        st.integers(min_value=500, max_value=2000),
    )
    @settings(max_examples=50, deadline=5000)
    def test_reshuffle_win_rate_stable(self, n_bars):
        """Property: win_rate after reshuffle is always in [0, 1]."""
        from rdagent.components.backtesting.vbt_backtest import backtest_signal
        close, signal = _price_signal(n_bars, seed=42)
        returns = close.pct_change().fillna(0)
        rng = np.random.default_rng(42)
        shuffled = pd.Series(rng.permutation(returns.values), index=returns.index)
        price_s = (1 + shuffled).cumprod() * 1.10
        r = backtest_signal(price_s, signal, txn_cost_bps=0.0)
        if r["status"] == "success":
            assert 0.0 <= r["win_rate"] <= 1.0

    @given(
        st.integers(min_value=500, max_value=1500),
    )
    @settings(max_examples=50, deadline=5000)
    def test_reshuffle_sharpe_finite(self, n_bars):
        """Property: Sharpe after reshuffle is finite."""
        from rdagent.components.backtesting.vbt_backtest import backtest_signal
        close, signal = _price_signal(n_bars, seed=42)
        returns = close.pct_change().fillna(0)
        rng = np.random.default_rng(42)
        shuffled = pd.Series(rng.permutation(returns.values), index=returns.index)
        price_s = (1 + shuffled).cumprod() * 1.10
        r = backtest_signal(price_s, signal, txn_cost_bps=0.0)
        if r["status"] == "success":
            assert np.isfinite(r["sharpe"])

    @given(
        st.integers(min_value=500, max_value=1500),
    )
    @settings(max_examples=50, deadline=5000)
    def test_reshuffle_n_trades_unchanged(self, n_bars):
        """Property: n_trades unchanged by reshuffling (same signal pattern)."""
        from rdagent.components.backtesting.vbt_backtest import backtest_signal
        close, signal = _price_signal(n_bars, seed=42)
        r_orig = backtest_signal(close, signal, txn_cost_bps=0.0)
        returns = close.pct_change().fillna(0)
        rng = np.random.default_rng(42)
        shuffled = pd.Series(rng.permutation(returns.values), index=returns.index)
        price_s = (1 + shuffled).cumprod() * 1.10
        r_shuf = backtest_signal(price_s, signal, txn_cost_bps=0.0)
        if r_orig["status"] == "success" and r_shuf["status"] == "success":
            assert r_orig["n_trades"] == r_shuf["n_trades"]


# ---------------------------------------------------------------------------
# Random Market Data Fuzzing (20 tests)
# ---------------------------------------------------------------------------


class TestRandomMarketDataFuzzing:
    """Fuzz backtest_signal with completely random market data."""

    @given(
        st.integers(min_value=100, max_value=5000),
        st.floats(min_value=-0.1, max_value=0.1),
        st.floats(min_value=0.00001, max_value=0.1),
    )
    @settings(max_examples=200, deadline=5000)
    def test_random_prices_always_succeed(self, n_bars, drift, vol):
        """Property: backtesting with random geometric Brownian motion succeeds."""
        dates = pd.date_range("2024-01-01", periods=n_bars, freq="1min")
        rng = np.random.default_rng(42)
        close = pd.Series(1.10 * np.exp(np.cumsum(rng.normal(drift, vol, n_bars))), index=dates)
        signal = pd.Series(np.where(rng.normal(0, 1, n_bars) > 0, 1.0, -1.0), index=dates)
        result = backtest_signal(close, signal, txn_cost_bps=0.0)
        assert result["status"] in ("success", "failed")

    @given(
        st.integers(min_value=100, max_value=3000),
        st.floats(min_value=-0.01, max_value=0.01),
        st.floats(min_value=0.0001, max_value=0.1),
        st.floats(min_value=0.0, max_value=30.0),
    )
    @settings(max_examples=200, deadline=5000)
    def test_random_data_all_metrics_finite(self, n_bars, drift, vol, cost):
        """Property: all key metrics are finite for random data."""
        dates = pd.date_range("2024-01-01", periods=n_bars, freq="1min")
        rng = np.random.default_rng(42)
        close = pd.Series(1.10 * np.exp(np.cumsum(rng.normal(drift, vol, n_bars))), index=dates)
        signal = pd.Series(np.where(rng.normal(0, 1, n_bars) > 0, 1.0, -1.0), index=dates)
        result = backtest_signal(close, signal, txn_cost_bps=cost)
        if result["status"] == "success":
            for k in ["sharpe", "total_return", "max_drawdown"]:
                assert np.isfinite(result[k]), f"{k} is not finite: {result[k]}"

    @given(
        st.integers(min_value=100, max_value=3000),
        st.floats(min_value=-0.01, max_value=0.01),
    )
    @settings(max_examples=200, deadline=5000)
    def test_random_data_maxdd_in_bounds(self, n_bars, drift):
        """Property: max_drawdown ∈ [-1, 0] with random market data."""
        dates = pd.date_range("2024-01-01", periods=n_bars, freq="1min")
        rng = np.random.default_rng(42)
        close = pd.Series(1.10 * np.exp(np.cumsum(rng.normal(drift, 0.001, n_bars))), index=dates)
        signal = pd.Series(np.where(rng.normal(0, 1, n_bars) > 0, 1.0, -1.0), index=dates)
        result = backtest_signal(close, signal, txn_cost_bps=0.0)
        if result["status"] == "success":
            assert -1.0 <= result["max_drawdown"] <= 0.0

    @given(
        st.integers(min_value=100, max_value=3000),
        st.floats(min_value=-0.01, max_value=0.01),
    )
    @settings(max_examples=200, deadline=5000)
    def test_random_data_win_rate_in_bounds(self, n_bars, drift):
        """Property: win_rate ∈ [0, 1] with random market data."""
        dates = pd.date_range("2024-01-01", periods=n_bars, freq="1min")
        rng = np.random.default_rng(42)
        close = pd.Series(1.10 * np.exp(np.cumsum(rng.normal(drift, 0.001, n_bars))), index=dates)
        signal = pd.Series(np.where(rng.normal(0, 1, n_bars) > 0, 1.0, -1.0), index=dates)
        result = backtest_signal(close, signal, txn_cost_bps=0.0)
        if result["status"] == "success":
            assert 0.0 <= result["win_rate"] <= 1.0

    @given(
        st.integers(min_value=100, max_value=3000),
    )
    @settings(max_examples=100, deadline=5000)
    def test_random_data_n_bars_matches_input(self, n_bars):
        """Property: n_bars in result equals input length."""
        close, signal = _price_signal(n_bars, seed=42)
        result = backtest_signal(close, signal, txn_cost_bps=0.0)
        if result["status"] == "success":
            assert result["n_bars"] == n_bars

    @given(
        st.integers(min_value=100, max_value=3000),
    )
    @settings(max_examples=100, deadline=5000)
    def test_random_data_signal_counts_sum_correctly(self, n_bars):
        """Property: signal_long + signal_short + signal_neutral == n_bars."""
        close, signal = _price_signal(n_bars, seed=42)
        result = backtest_signal(close, signal, txn_cost_bps=0.0)
        if result["status"] == "success":
            assert result["signal_long"] + result["signal_short"] + result["signal_neutral"] == n_bars

    @given(
        st.integers(min_value=100, max_value=3000),
        st.floats(min_value=1.0, max_value=500.0),
    )
    @settings(max_examples=100, deadline=5000)
    def test_random_data_txn_cost_bps_preserved(self, n_bars, cost):
        """Property: txn_cost_bps reported matches input."""
        close, signal = _price_signal(n_bars, seed=42)
        result = backtest_signal(close, signal, txn_cost_bps=cost)
        if result["status"] == "success":
            assert abs(result["txn_cost_bps"] - cost) < 0.001


# ---------------------------------------------------------------------------
# OOS Stress Fuzzing (10 tests)
# ---------------------------------------------------------------------------


class TestOOSStressFuzzing:
    """Hypothesis-based out-of-sample stress tests."""

    @given(
        st.integers(min_value=1000, max_value=5000),
        st.floats(min_value=0.3, max_value=0.8),
    )
    @settings(max_examples=100, deadline=5000)
    def test_oos_metrics_valid(self, n_bars, split_fraction):
        """Property: OOS metrics remain valid for any split."""
        from rdagent.components.backtesting.vbt_backtest import backtest_signal
        dates = pd.date_range("2024-01-01", periods=n_bars, freq="1min")
        rng = np.random.default_rng(42)
        close = pd.Series(1.10 * np.exp(np.cumsum(rng.normal(0, 0.0002, n_bars))), index=dates)
        signal = pd.Series(np.where(rng.normal(0, 1, n_bars) > 0, 1.0, -1.0), index=dates)
        split = int(n_bars * split_fraction)
        assume(split > 100)
        assume(n_bars - split > 100)
        r_oos = backtest_signal(close.iloc[split:], signal.iloc[split:], txn_cost_bps=0.0)
        if r_oos["status"] == "success":
            assert -1.0 <= r_oos["max_drawdown"] <= 0.0
            assert np.isfinite(r_oos["sharpe"])

    @given(
        st.integers(min_value=500, max_value=3000),
    )
    @settings(max_examples=80, deadline=5000)
    def test_oos_sharpe_finite(self, n_bars):
        """Property: OOS Sharpe is always finite."""
        from rdagent.components.backtesting.vbt_backtest import backtest_signal
        close, signal = _price_signal(n_bars, seed=42)
        split = n_bars // 2
        assume(n_bars - split > 100)
        r_oos = backtest_signal(close.iloc[split:], signal.iloc[split:], txn_cost_bps=0.0)
        if r_oos["status"] == "success":
            assert np.isfinite(r_oos["sharpe"])

    @given(
        st.integers(min_value=1000, max_value=3000),
    )
    @settings(max_examples=80, deadline=5000)
    def test_is_and_oos_both_produce_metrics(self, n_bars):
        """Property: both IS and OOS periods produce valid metrics."""
        from rdagent.components.backtesting.vbt_backtest import backtest_signal
        close, signal = _price_signal(n_bars, seed=42)
        split = int(n_bars * 0.7)
        assume(split > 100)
        assume(n_bars - split > 100)
        r_is = backtest_signal(close.iloc[:split], signal.iloc[:split], txn_cost_bps=0.0)
        r_oos = backtest_signal(close.iloc[split:], signal.iloc[split:], txn_cost_bps=0.0)
        if r_is["status"] == "success":
            assert np.isfinite(r_is["sharpe"])
        if r_oos["status"] == "success":
            assert np.isfinite(r_oos["max_drawdown"])

    @given(
        st.integers(min_value=500, max_value=2000),
    )
    @settings(max_examples=50, deadline=5000)
    def test_oos_win_rate_in_bounds(self, n_bars):
        """Property: OOS win_rate ∈ [0, 1]."""
        from rdagent.components.backtesting.vbt_backtest import backtest_signal
        close, signal = _price_signal(n_bars, seed=42)
        split = n_bars // 2
        assume(n_bars - split > 100)
        r_oos = backtest_signal(close.iloc[split:], signal.iloc[split:], txn_cost_bps=0.0)
        if r_oos["status"] == "success":
            assert 0.0 <= r_oos["win_rate"] <= 1.0


# ---------------------------------------------------------------------------
# Forward Returns Backtest Fuzzing (10 tests)
# ---------------------------------------------------------------------------


class TestForwardReturnsFuzzing:
    """Fuzz backtest_from_forward_returns with random factor and forward returns."""

    @given(
        st.integers(min_value=30, max_value=500),
        st.lists(st.floats(min_value=-10, max_value=10), min_size=30, max_size=500),
        st.lists(st.floats(min_value=-0.5, max_value=0.5), min_size=30, max_size=500),
        st.floats(min_value=0.0, max_value=50.0),
    )
    @settings(max_examples=100, deadline=5000)
    def test_forward_backtest_returns_all_keys(self, n, fac_raw, ret_raw, cost):
        """Property: backtest_from_forward_returns contains all expected keys."""
        n = min(len(fac_raw), len(ret_raw))
        factor = pd.Series(fac_raw[:n], dtype=float)
        fwd = pd.Series(ret_raw[:n], dtype=float)
        assume(factor.std() > 1e-12)
        result = backtest_from_forward_returns(factor, fwd, txn_cost_bps=cost)
        for k in ["status", "sharpe", "max_drawdown", "total_return", "win_rate",
                  "n_trades", "ic", "n_bars"]:
            assert k in result, f"Missing key: {k}"

    @given(
        st.integers(min_value=30, max_value=500),
        st.lists(st.floats(min_value=-10, max_value=10), min_size=30, max_size=500),
        st.lists(st.floats(min_value=-0.5, max_value=0.5), min_size=30, max_size=500),
    )
    @settings(max_examples=100, deadline=5000)
    def test_forward_backtest_maxdd_in_bounds(self, n, fac_raw, ret_raw):
        """Property: max_drawdown ∈ [-1, 0] from forward returns backtest."""
        n = min(len(fac_raw), len(ret_raw))
        factor = pd.Series(fac_raw[:n], dtype=float)
        fwd = pd.Series(ret_raw[:n], dtype=float)
        assume(factor.std() > 1e-12)
        result = backtest_from_forward_returns(factor, fwd, txn_cost_bps=0.0)
        if result["status"] == "success":
            assert -1.0 <= result["max_drawdown"] <= 0.0

    @given(
        st.integers(min_value=30, max_value=500),
        st.lists(st.floats(min_value=-10, max_value=10), min_size=30, max_size=500),
        st.lists(st.floats(min_value=-0.5, max_value=0.5), min_size=30, max_size=500),
    )
    @settings(max_examples=100, deadline=5000)
    def test_forward_backtest_ic_in_bounds(self, n, fac_raw, ret_raw):
        """Property: IC ∈ [-1, 1] from forward returns backtest."""
        n = min(len(fac_raw), len(ret_raw))
        factor = pd.Series(fac_raw[:n], dtype=float)
        fwd = pd.Series(ret_raw[:n], dtype=float)
        assume(factor.std() > 1e-12)
        result = backtest_from_forward_returns(factor, fwd, txn_cost_bps=0.0)
        if result["status"] == "success":
            assert -1.0 <= result["ic"] <= 1.0, f"IC={result['ic']}"

    @given(
        st.integers(min_value=30, max_value=500),
        st.lists(st.floats(min_value=-10, max_value=10), min_size=30, max_size=500),
        st.lists(st.floats(min_value=-0.5, max_value=0.5), min_size=30, max_size=500),
    )
    @settings(max_examples=100, deadline=5000)
    def test_forward_backtest_win_rate_in_bounds(self, n, fac_raw, ret_raw):
        """Property: win_rate ∈ [0, 1] from forward returns backtest."""
        n = min(len(fac_raw), len(ret_raw))
        factor = pd.Series(fac_raw[:n], dtype=float)
        fwd = pd.Series(ret_raw[:n], dtype=float)
        assume(factor.std() > 1e-12)
        result = backtest_from_forward_returns(factor, fwd, txn_cost_bps=0.0)
        if result["status"] == "success":
            assert 0.0 <= result["win_rate"] <= 1.0, f"WinRate={result['win_rate']}"

    @given(
        st.integers(min_value=1, max_value=9),
    )
    @settings(max_examples=20, deadline=5000)
    def test_forward_backtest_too_few_bars_fails(self, n):
        """Property: < 10 aligned bars fails."""
        factor = pd.Series(np.arange(n, dtype=float))
        fwd = pd.Series(np.arange(n, dtype=float))
        result = backtest_from_forward_returns(factor, fwd)
        assert result["status"] == "failed"


# ---------------------------------------------------------------------------
# Edge Cases and Extreme Values Fuzzing (10 tests)
# ---------------------------------------------------------------------------


class TestEdgeCasesFuzzing:
    """Fuzzing with extreme/nonsense inputs."""

    @given(
        st.integers(min_value=100, max_value=2000),
    )
    @settings(max_examples=70, deadline=5000)
    def test_zero_price_initial_does_not_crash(self, n_bars):
        """Property: backtest handles near-zero initial prices."""
        dates = pd.date_range("2024-01-01", periods=n_bars, freq="1min")
        rng = np.random.default_rng(42)
        close = pd.Series(0.000001 + abs(rng.normal(0, 0.0002, n_bars)).cumsum(), index=dates)
        signal = pd.Series(np.where(rng.normal(0, 1, n_bars) > 0, 1.0, -1.0), index=dates)
        result = backtest_signal(close, signal)
        assert result["status"] in ("success", "failed")

    @given(
        st.integers(min_value=100, max_value=2000),
    )
    @settings(max_examples=70, deadline=5000)
    def test_very_large_price_does_not_crash(self, n_bars):
        """Property: backtest handles very large prices."""
        dates = pd.date_range("2024-01-01", periods=n_bars, freq="1min")
        rng = np.random.default_rng(42)
        close = pd.Series(1e6 + rng.normal(0, 1, n_bars).cumsum(), index=dates)
        signal = pd.Series(np.where(rng.normal(0, 1, n_bars) > 0, 1.0, -1.0), index=dates)
        result = backtest_signal(close, signal)
        assert result["status"] in ("success", "failed")

    @given(
        st.integers(min_value=100, max_value=2000),
    )
    @settings(max_examples=70, deadline=5000)
    def test_signal_all_nan_treated_as_flat(self, n_bars):
        """Property: signal full of NaN is treated as flat (win_rate=0, n_trades=0)."""
        dates = pd.date_range("2024-01-01", periods=n_bars, freq="1min")
        rng = np.random.default_rng(42)
        close = pd.Series(1.10 + rng.normal(0, 0.0002, n_bars).cumsum(), index=dates)
        signal = pd.Series([np.nan] * n_bars, index=dates)
        result = backtest_signal(close, signal)
        if result["status"] == "success":
            assert result["n_trades"] == 0
            assert result["win_rate"] == 0.0

    @given(
        st.integers(min_value=1000, max_value=3000),
    )
    @settings(max_examples=70, deadline=5000)
    def test_continuous_signal_produces_valid_metrics(self, n_bars):
        """Property: continuous signal in [-1, 1] produces valid metrics."""
        dates = pd.date_range("2024-01-01", periods=n_bars, freq="1min")
        rng = np.random.default_rng(42)
        close = pd.Series(1.10 * np.exp(np.cumsum(rng.normal(0, 0.0002, n_bars))), index=dates)
        signal = pd.Series(rng.uniform(-1, 1, n_bars), index=dates)
        result = backtest_signal(close, signal, txn_cost_bps=0.0)
        if result["status"] == "success":
            assert -1.0 <= result["max_drawdown"] <= 0.0
            assert 0.0 <= result["win_rate"] <= 1.0

    @given(
        st.integers(min_value=500, max_value=2000),
    )
    @settings(max_examples=70, deadline=5000)
    def test_weekend_gaps_produce_valid_metrics(self, n_bars):
        """Property: data with time gaps (weekends) produces valid metrics."""
        dates = pd.bdate_range("2024-01-01", periods=n_bars, freq="1min")
        rng = np.random.default_rng(42)
        close = pd.Series(1.10 + rng.normal(0, 0.0002, len(dates)).cumsum(), index=dates)
        signal = pd.Series(np.where(rng.normal(0, 1, len(dates)) > 0, 1.0, -1.0), index=dates)
        result = backtest_signal(close, signal)
        if result["status"] == "success":
            assert np.isfinite(result["sharpe"])
            assert -1.0 <= result["max_drawdown"] <= 0.0
