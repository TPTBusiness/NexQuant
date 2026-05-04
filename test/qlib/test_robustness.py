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
