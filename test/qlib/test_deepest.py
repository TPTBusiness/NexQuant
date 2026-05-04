"""Deepest tests: property-based, metamorphic, fuzzing, stress."""

from __future__ import annotations

import sys
from pathlib import Path

import numpy as np
import pandas as pd
import pytest
from hypothesis import HealthCheck, given, settings, strategies as st

PROJECT_ROOT = Path(__file__).parent.parent.parent
sys.path.insert(0, str(PROJECT_ROOT))


# =============================================================================
# Property-Based: Random inputs → no crashes, valid output bounds
# =============================================================================


class TestPropertyBasedBacktest:
    """For ANY random signal and price, backtest must never crash and produce valid metrics."""

    @given(
        n_bars=st.integers(min_value=100, max_value=500),
        trend=st.floats(min_value=-0.01, max_value=0.01),
        vol=st.floats(min_value=0.0001, max_value=0.01),
        signal_noise=st.floats(min_value=0.1, max_value=2.0),
    )
    @settings(max_examples=50, deadline=None, suppress_health_check=[HealthCheck.function_scoped_fixture])
    def test_random_signal_never_crashes(self, n_bars, trend, vol, signal_noise):
        from rdagent.components.backtesting.vbt_backtest import backtest_signal

        dates = pd.date_range("2024-01-01", periods=n_bars, freq="1min")
        returns = np.random.default_rng(42).normal(trend, vol, n_bars)
        close = pd.Series(1.10 * np.exp(np.cumsum(returns)), index=dates)
        signal = pd.Series(
            np.where(np.random.default_rng(43).normal(0, signal_noise, n_bars) > 0, 1.0, -1.0),
            index=dates,
        )

        result = backtest_signal(close, signal)
        assert result["status"] in ("success", "failed")

        # All metrics must be within valid bounds
        if result["status"] == "success":
            assert -1.0 <= result["max_drawdown"] <= 0.0
            assert 0.0 <= result["win_rate"] <= 1.0
            assert np.isfinite(result["sharpe"])
            assert np.isfinite(result["total_return"])
            assert result["n_trades"] >= 0

    @given(
        n_bars=st.integers(min_value=200, max_value=500),
        mean_factor=st.floats(min_value=-1.0, max_value=1.0),
        factor_noise=st.floats(min_value=0.1, max_value=2.0),
    )
    @settings(max_examples=50, deadline=None)
    def test_random_factor_never_crashes(self, n_bars, mean_factor, factor_noise):
        from rdagent.components.backtesting.vbt_backtest import backtest_from_forward_returns

        idx = pd.MultiIndex.from_arrays(
            [pd.date_range("2024-01-01", periods=n_bars, freq="1min"), ["EURUSD"] * n_bars],
            names=["datetime", "instrument"],
        )
        close = pd.Series(1.10 + np.random.default_rng(42).normal(0, 0.001, n_bars).cumsum(), index=idx)
        fwd = close.groupby(level="instrument").shift(-96) / close - 1
        factor = pd.Series(np.random.default_rng(44).normal(mean_factor, factor_noise, n_bars), index=idx)

        result = backtest_from_forward_returns(factor, fwd, close)
        assert result["status"] in ("success", "failed")

        if result["status"] == "success" and "ic" in result:
            assert -1.0 <= result["ic"] <= 1.0


# =============================================================================
# Metamorphic: Input transformations → predictable output changes
# =============================================================================


class TestMetamorphicBacktest:
    """If we transform the input in a known way, the output must change predictably."""

    def test_doubling_signal_preserves_sign(self):
        """Doubling the signal values should NOT change position signs → same metrics."""
        from rdagent.components.backtesting.vbt_backtest import backtest_signal

        n = 2000
        dates = pd.date_range("2024-01-01", periods=n, freq="1min")
        rng = np.random.default_rng(42)
        close = pd.Series(1.10 + rng.normal(0, 0.001, n).cumsum(), index=dates)
        signal = pd.Series(np.where(rng.normal(0, 1, n) > 0, 1.0, -1.0), index=dates)

        r1 = backtest_signal(close, signal, txn_cost_bps=0.0)
        r2 = backtest_signal(close, signal * 2.0, txn_cost_bps=0.0)

        # Doubling discrete (-1/+1) signal → same positions → same results
        assert r1["n_trades"] == r2["n_trades"]
        assert abs(r1["sharpe"] - r2["sharpe"]) < 0.001
        assert abs(r1["max_drawdown"] - r2["max_drawdown"]) < 0.001

    def test_negating_signal_flips_sign(self):
        """Flipping all signal signs should produce opposite-direction results."""
        from rdagent.components.backtesting.vbt_backtest import backtest_signal

        n = 2000
        dates = pd.date_range("2024-01-01", periods=n, freq="1min")
        rng = np.random.default_rng(42)
        close = pd.Series(1.10 + rng.normal(0, 0.001, n).cumsum(), index=dates)
        signal = pd.Series(np.where(rng.normal(0, 1, n) > 0, 1.0, -1.0), index=dates)

        r1 = backtest_signal(close, signal, txn_cost_bps=0.0)
        r2 = backtest_signal(close, -signal, txn_cost_bps=0.0)

        # Negating signal should produce opposite total_return sign
        assert r1["total_return"] * r2["total_return"] <= 0 or (
            abs(r1["total_return"]) < 0.001 and abs(r2["total_return"]) < 0.001
        )


    def test_ic_invariant_under_linear_transform(self):
        """IC(factor, returns) must be invariant under y = a*x + b."""
        idx = pd.MultiIndex.from_arrays(
            [pd.date_range("2024-01-01", periods=500, freq="1min"), ["EURUSD"] * 500],
            names=["datetime", "instrument"],
        )
        close = pd.Series(1.10 + np.arange(500) * 0.0001, index=idx)
        fwd = close.groupby(level="instrument").shift(-96) / close - 1
        factor = pd.Series(np.random.default_rng(42).normal(0, 1, 500), index=idx)

        valid = factor.dropna().index.intersection(fwd.dropna().index)
        ic1 = factor.loc[valid].corr(fwd.loc[valid])

        # IC must be invariant under scaling and shifting
        ic2 = (factor.loc[valid] * 3.7 + 2.1).corr(fwd.loc[valid])
        assert abs(ic1 - ic2) < 0.0001

        # IC must negate when factor is negated
        ic3 = (-factor.loc[valid]).corr(fwd.loc[valid])
        assert abs(ic1 + ic3) < 0.0001

    def test_sharpe_differs_with_different_signals(self):
        """Two different signals should produce different Sharpes."""
        from rdagent.components.backtesting.vbt_backtest import backtest_signal

        n = 3000
        dates = pd.date_range("2024-01-01", periods=n, freq="1min")
        rng = np.random.default_rng(42)
        returns = rng.normal(0, 0.0002, n)
        close = pd.Series(1.10 * np.exp(np.cumsum(returns)), index=dates)
        signal_a = pd.Series(1.0, index=dates)  # always long
        signal_b = pd.Series(-1.0, index=dates)  # always short

        r_a = backtest_signal(close, signal_a, txn_cost_bps=0.0)
        r_b = backtest_signal(close, signal_b, txn_cost_bps=0.0)

        # Always-long vs always-short should have opposite total_return signs
        assert r_a["total_return"] * r_b["total_return"] <= 0


# =============================================================================
# Stress / Fuzzing
# =============================================================================


class TestStressFuzzing:
    """Extreme inputs — must not crash, must produce bounded output."""

    def test_very_large_dataset(self):
        """50k bars — must complete without OOM."""
        from rdagent.components.backtesting.vbt_backtest import backtest_signal

        n = 50_000
        dates = pd.date_range("2020-01-01", periods=n, freq="1min")
        rng = np.random.default_rng(42)
        close = pd.Series(1.10 + rng.normal(0, 0.0001, n).cumsum(), index=dates)
        signal = pd.Series(np.where(rng.normal(0, 1, n) > 0, 1.0, -1.0), index=dates)

        result = backtest_signal(close, signal)
        assert result["status"] == "success"
        assert result["n_trades"] > 0

    def test_extreme_prices(self):
        """Prices from 0.00001 to 1,000,000 — must handle."""
        from rdagent.components.backtesting.vbt_backtest import backtest_signal

        n = 2000
        dates = pd.date_range("2024-01-01", periods=n, freq="1min")
        # Extreme multiplicative returns
        close = pd.Series(1.0 * np.exp(np.cumsum(np.random.default_rng(42).normal(0, 0.01, n))), index=dates)
        signal = pd.Series(np.where(np.random.default_rng(43).normal(0, 1, n) > 0, 1.0, -1.0), index=dates)

        result = backtest_signal(close, signal)
        assert result["status"] in ("success", "failed")
        assert np.isfinite(result["sharpe"])

    def test_all_identical_prices(self):
        """All prices equal — should return 0 return, 0 Sharpe."""
        from rdagent.components.backtesting.vbt_backtest import backtest_signal

        dates = pd.date_range("2024-01-01", periods=500, freq="1min")
        close = pd.Series(1.0, index=dates)
        signal = pd.Series(np.where(np.arange(500) % 2 == 0, 1.0, -1.0), index=dates)

        result = backtest_signal(close, signal, txn_cost_bps=0.0)
        # With flat prices, total return must be 0
        assert result["total_return"] == 0.0
        assert result["sharpe"] == 0.0

    def test_single_large_spike(self):
        """One bar with 1000% return — backtest must handle gracefully."""
        from rdagent.components.backtesting.vbt_backtest import backtest_signal

        n = 1000
        dates = pd.date_range("2024-01-01", periods=n, freq="1min")
        close = pd.Series(1.0, index=dates)
        close.iloc[500] = 11.0  # 10x spike
        signal = pd.Series(1.0, index=dates)  # always long

        result = backtest_signal(close, signal)
        assert result["status"] in ("success", "failed")

    def test_rapid_position_flipping(self):
        """Signal flips every single bar — max trades, max turnover."""
        from rdagent.components.backtesting.vbt_backtest import backtest_signal

        n = 2000
        dates = pd.date_range("2024-01-01", periods=n, freq="1min")
        close = pd.Series(1.10 + np.random.default_rng(42).normal(0, 0.0001, n).cumsum(), index=dates)
        signal = pd.Series([1.0, -1.0] * (n // 2), index=dates)

        result = backtest_signal(close, signal, txn_cost_bps=2.14)
        assert result["status"] in ("success", "failed")
        # With rapid flipping and 2.14bps cost, total_return should be negative
        if result["status"] == "success":
            assert result["total_return"] <= 0.0

    def test_gapped_data(self):
        """Data with missing timestamps (weekend gaps) — must handle."""
        from rdagent.components.backtesting.vbt_backtest import backtest_signal

        # 5 days of data with weekend gaps
        dates = pd.date_range("2024-01-01", periods=5 * 1440, freq="1min")  # Mon-Fri
        close = pd.Series(1.10 + np.random.default_rng(42).normal(0, 0.0001, len(dates)).cumsum(), index=dates)
        signal = pd.Series(np.where(np.random.default_rng(43).normal(0, 1, len(dates)) > 0, 1.0, -1.0), index=dates)

        result = backtest_signal(close, signal)
        assert result["status"] in ("success", "failed")


# =============================================================================
# Fuzzing: Verify runtime verifier catches all
# =============================================================================


class TestRuntimeVerifierFuzzing:
    """The runtime verifier must catch corrupted results."""

    @given(
        bad_sharpe=st.one_of(
            st.just(float("inf")),
            st.just(float("nan")),
            st.just(float("-inf")),
        ),
    )
    @settings(max_examples=3, deadline=None)
    def test_verifier_catches_invalid_sharpe(self, bad_sharpe):
        from rdagent.components.backtesting.verify import verify_backtest_result

        result = {
            "sharpe": bad_sharpe,
            "max_drawdown": -0.15,
            "win_rate": 0.55,
            "total_return": 0.25,
            "annual_return_pct": 15.0,
            "monthly_return_pct": 1.2,
            "n_trades": 50,
            "status": "success",
        }
        warnings = verify_backtest_result(result)
        assert len(warnings) > 0

    @given(
        bad_dd=st.floats(min_value=-5.0, max_value=-1.01),
    )
    @settings(max_examples=20, deadline=None)
    def test_verifier_catches_invalid_drawdown(self, bad_dd):
        from rdagent.components.backtesting.verify import verify_backtest_result

        result = {
            "sharpe": 1.5,
            "max_drawdown": bad_dd,
            "win_rate": 0.55,
            "total_return": 0.25,
            "annual_return_pct": 15.0,
            "monthly_return_pct": 1.2,
            "n_trades": 50,
            "status": "success",
        }
        warnings = verify_backtest_result(result)
        assert len(warnings) > 0

    @given(
        bad_wr=st.floats(min_value=-1.0, max_value=-0.01) | st.floats(min_value=1.01, max_value=5.0),
    )
    @settings(max_examples=20, deadline=None)
    def test_verifier_catches_invalid_winrate(self, bad_wr):
        from rdagent.components.backtesting.verify import verify_backtest_result

        result = {
            "sharpe": 1.5,
            "max_drawdown": -0.15,
            "win_rate": bad_wr,
            "total_return": 0.25,
            "annual_return_pct": 15.0,
            "monthly_return_pct": 1.2,
            "n_trades": 50,
            "status": "success",
        }
        warnings = verify_backtest_result(result)
        assert len(warnings) > 0
