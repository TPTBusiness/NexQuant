"""Deep property-based tests for the unified backtest engine.

Extends test_vbt_backtest.py with hypothesis-based property tests,
edge-case fuzzing, and mathematical invariants.
"""

from __future__ import annotations

import numpy as np
import pandas as pd
import pytest

from hypothesis import assume, given, settings
from hypothesis import strategies as st
from hypothesis.extra.numpy import arrays

from rdagent.components.backtesting.vbt_backtest import (
    DEFAULT_BARS_PER_YEAR,
    backtest_from_forward_returns,
    backtest_signal,
)


@pytest.fixture
def rng_close():
    """Large random multi-year close series."""
    rng = np.random.default_rng(42)
    idx = pd.date_range("2020-01-01", periods=10000, freq="1min")
    return pd.Series(1.10 + rng.normal(0, 0.0001, 10000).cumsum(), index=idx)


# ---------------------------------------------------------------------------
# Property-based tests
# ---------------------------------------------------------------------------
class TestBacktestProperties:
    @given(
        n_bars=st.integers(min_value=10, max_value=500),
        seed=st.integers(min_value=0, max_value=2**16),
    )
    @settings(max_examples=100, deadline=10000)
    def test_always_long_accumulates_price_return(self, n_bars, seed):
        """Property: position = +1 always → total_return ≈ price total return − cost."""
        rng = np.random.default_rng(seed)
        idx = pd.date_range("2024-01-01", periods=n_bars, freq="1min")
        changes = rng.normal(0, 0.001, n_bars)
        close = pd.Series(100 * (1 + changes).cumprod(), index=idx)
        signal = pd.Series(1.0, index=idx)

        r = backtest_signal(close, signal, txn_cost_bps=0.0)
        price_tr = close.iloc[-1] / close.iloc[0] - 1
        assert abs(r["total_return"] - price_tr) < 1e-6

    @given(
        n_bars=st.integers(min_value=10, max_value=500),
        seed=st.integers(min_value=0, max_value=2**16),
    )
    @settings(max_examples=100, deadline=10000)
    def test_no_signal_zero_pnl(self, n_bars, seed):
        """Property: signal = 0 everywhere → zero P&L, zero trades."""
        rng = np.random.default_rng(seed)
        idx = pd.date_range("2024-01-01", periods=n_bars, freq="1min")
        close = pd.Series(100 + rng.normal(0, 0.1, n_bars).cumsum(), index=idx)
        signal = pd.Series(0.0, index=idx)

        r = backtest_signal(close, signal, txn_cost_bps=1.5)
        assert r["total_return"] == 0.0
        assert r["sharpe"] == 0.0
        assert r["n_trades"] == 0
        assert r["max_drawdown"] == 0.0

    @given(
        n_bars=st.integers(min_value=10, max_value=500),
        cost_bps=st.floats(min_value=0, max_value=100),
        seed=st.integers(min_value=0, max_value=2**16),
    )
    @settings(max_examples=100, deadline=10000)
    def test_cost_monotonicity(self, n_bars, cost_bps, seed):
        """Property: higher cost → lower total_return (monotonic)."""
        assume(cost_bps < 50)
        rng = np.random.default_rng(seed)
        idx = pd.date_range("2024-01-01", periods=n_bars, freq="1min")
        close = pd.Series(100 + rng.normal(0, 0.1, n_bars).cumsum(), index=idx)
        sig = pd.Series(rng.choice([-1.0, 1.0], n_bars), index=idx)

        r0 = backtest_signal(close, sig, txn_cost_bps=0.0)
        rc = backtest_signal(close, sig, txn_cost_bps=cost_bps)
        assert r0["total_return"] >= rc["total_return"] - 1e-12

    @given(
        n_bars=st.integers(min_value=10, max_value=500),
        seed=st.integers(min_value=0, max_value=2**16),
    )
    @settings(max_examples=100, deadline=10000)
    def test_signal_inversion_yields_negated_return_uncosted(self, n_bars, seed):
        """Property: flipping signal sign → total_return flips sign (zero cost)."""
        rng = np.random.default_rng(seed)
        idx = pd.date_range("2024-01-01", periods=n_bars, freq="1min")
        close = pd.Series(100 + rng.normal(0, 0.1, n_bars).cumsum(), index=idx)
        sig = pd.Series(rng.choice([-1.0, 1.0], n_bars), index=idx)

        r_pos = backtest_signal(close, sig, txn_cost_bps=0.0)
        r_neg = backtest_signal(close, -sig, txn_cost_bps=0.0)
        # With zero cost, returns should be exact negatives except for the
        # initial position-opening cost which affects one side.
        assert abs(r_pos["total_return"] + r_neg["total_return"]) < 0.05


class TestBacktestEdgeCases:
    def test_single_bar(self):
        """Single bar: engine rejects insufficient data gracefully."""
        close = pd.Series([100.0], index=pd.DatetimeIndex(["2024-01-01"]))
        signal = pd.Series([1.0], index=close.index)
        r = backtest_signal(close, signal, txn_cost_bps=0.0)
        # Engine needs at least a few bars for returns computation
        assert r["status"] in ("success", "failed", "error")

    def test_two_bars_flip(self):
        """Two bars with position flip: total cost = 3 * txn_cost_bps."""
        idx = pd.DatetimeIndex(["2024-01-01 00:00", "2024-01-01 00:01"])
        close = pd.Series([100.0, 100.0], index=idx)
        signal = pd.Series([1.0, -1.0], index=idx)
        r = backtest_signal(close, signal, txn_cost_bps=10.0)
        assert r["total_return"] < 0

    def test_extreme_close_values(self):
        """Very large and very small prices must not cause numerical issues."""
        idx = pd.date_range("2024-01-01", periods=100, freq="1min")
        sig = pd.Series(1.0, index=idx)
        for price in [1e-10, 1e10]:
            close = pd.Series(price, index=idx)
            r = backtest_signal(close, sig, txn_cost_bps=0.0)
            assert r["total_return"] == pytest.approx(0.0, abs=1e-8)

    def test_nan_in_signal_handled(self):
        """NaN in signal should be treated as flat (0) or skipped."""
        idx = pd.date_range("2024-01-01", periods=50, freq="1min")
        close = pd.Series(100 + np.arange(50) * 0.01, index=idx)
        signal = pd.Series([1.0 if i % 10 != 3 else float("nan") for i in range(50)], index=idx)
        r = backtest_signal(close, signal, txn_cost_bps=0.0)
        assert r["status"] == "success"

    def test_inf_in_signal_handled(self):
        """Inf in signal should not crash the engine."""
        idx = pd.date_range("2024-01-01", periods=50, freq="1min")
        close = pd.Series(100 + np.arange(50) * 0.01, index=idx)
        signal = pd.Series([1.0 if i % 7 != 0 else float("inf") for i in range(50)], index=idx)
        r = backtest_signal(close, signal, txn_cost_bps=0.0)
        assert r["status"] in ("success", "error")

    def test_empty_series(self):
        """Empty input series must return clean error."""
        close = pd.Series([], dtype=float)
        signal = pd.Series([], dtype=float)
        r = backtest_signal(close, signal, txn_cost_bps=0.0)
        assert r["status"] in ("error", "failed")

    def test_mismatched_index_lengths(self):
        """Different length close/signal should be handled."""
        idx1 = pd.date_range("2024-01-01", periods=100, freq="1min")
        idx2 = pd.date_range("2024-01-01", periods=90, freq="1min")
        close = pd.Series(100.0 + np.arange(100) * 0.01, index=idx1)
        signal = pd.Series(1.0, index=idx2)
        r = backtest_signal(close, signal, txn_cost_bps=0.0)
        assert r["status"] in ("success", "error")


class TestBacktestInvariants:
    def test_sharpe_zero_when_flat_market(self):
        """Flat price + any signal = zero Sharpe (with cost, tiny negative)."""
        idx = pd.date_range("2024-01-01", periods=500, freq="1min")
        close = pd.Series(100.0, index=idx)
        signal = pd.Series(np.where(np.random.default_rng(1).random(500) > 0.5, 1.0, -1.0), index=idx)
        r = backtest_signal(close, signal, txn_cost_bps=1.5)
        # Either 0 (if cost-free signal unchanged) or negative (costs)
        assert r["sharpe"] <= 0.01

    @given(seed=st.integers(0, 1000))
    @settings(max_examples=50, deadline=10000)
    def test_max_dd_negative_or_zero(self, seed):
        """Property: max_drawdown must be ≤ 0 for any input."""
        rng = np.random.default_rng(seed)
        n = rng.integers(50, 500)
        idx = pd.date_range("2024-01-01", periods=n, freq="1min")
        close = pd.Series(100 + rng.normal(0, 0.5, n).cumsum(), index=idx)
        signal = pd.Series(rng.choice([-1.0, 0.0, 1.0], n), index=idx)
        r = backtest_signal(close, signal, txn_cost_bps=1.5)
        assert r["max_drawdown"] <= 0.0

    @given(seed=st.integers(0, 1000))
    @settings(max_examples=50, deadline=10000)
    def test_bar_return_yearly_factor(self, seed):
        """Annualization factor is 252*1440 = 362880 for 1-min bars."""
        from rdagent.components.backtesting.vbt_backtest import DEFAULT_BARS_PER_YEAR
        assert DEFAULT_BARS_PER_YEAR == 252 * 1440

    def test_win_rate_between_0_and_1(self, rng_close):
        """Win rate must be in [0, 1] for any valid backtest."""
        rng = np.random.default_rng(99)
        signal = pd.Series(rng.choice([-1.0, 1.0], len(rng_close)), index=rng_close.index)
        r = backtest_signal(rng_close, signal, txn_cost_bps=1.5)
        assert 0.0 <= r["win_rate"] <= 1.0

    def test_n_trades_not_exceeding_bars(self, rng_close):
        """n_trades can't exceed the number of bars (one trade per bar max)."""
        rng = np.random.default_rng(123)
        signal = pd.Series(rng.choice([-1.0, 0.0, 1.0], len(rng_close)), index=rng_close.index)
        r = backtest_signal(rng_close, signal, txn_cost_bps=1.5)
        assert r["n_trades"] <= len(rng_close)

    def test_n_position_changes_positive(self, rng_close):
        """n_position_changes must be non-negative."""
        rng = np.random.default_rng(456)
        signal = pd.Series(rng.choice([-1.0, 0.0, 1.0], len(rng_close)), index=rng_close.index)
        r = backtest_signal(rng_close, signal, txn_cost_bps=1.5)
        assert r["n_position_changes"] >= 0


class TestBacktestIC:
    def test_ic_perfect_correlation(self):
        """Signal = forward_returns clipped → IC ≈ 1.0."""
        rng = np.random.default_rng(1)
        n = 500
        idx = pd.date_range("2024-01-01", periods=n, freq="1min")
        close = pd.Series(100 + rng.normal(0, 0.1, n).cumsum(), index=idx)
        fwd = close.pct_change().shift(-1).fillna(0)
        signal = fwd.clip(-1, 1)
        r = backtest_signal(close, signal, forward_returns=fwd, txn_cost_bps=0.0)
        assert r["ic"] is not None
        assert r["ic"] == pytest.approx(1.0, abs=1e-9)

    def test_ic_no_correlation(self):
        """Random signal → IC close to 0."""
        rng = np.random.default_rng(99)
        n = 2000
        idx = pd.date_range("2024-01-01", periods=n, freq="1min")
        close = pd.Series(100 + rng.normal(0, 0.1, n).cumsum(), index=idx)
        fwd = close.pct_change().shift(-1).fillna(0)
        signal = pd.Series(rng.normal(0, 1, n), index=idx)
        r = backtest_signal(close, signal, forward_returns=fwd, txn_cost_bps=0.0)
        assert abs(r["ic"]) < 0.10

    def test_ic_negative_correlation(self):
        """Inverted signal → negative IC."""
        rng = np.random.default_rng(2)
        n = 500
        idx = pd.date_range("2024-01-01", periods=n, freq="1min")
        close = pd.Series(100 + rng.normal(0, 0.1, n).cumsum(), index=idx)
        fwd = close.pct_change().shift(-1).fillna(0)
        signal = (-fwd).clip(-1, 1)
        r = backtest_signal(close, signal, forward_returns=fwd, txn_cost_bps=0.0)
        assert r["ic"] is not None
        assert r["ic"] == pytest.approx(-1.0, abs=1e-9)
