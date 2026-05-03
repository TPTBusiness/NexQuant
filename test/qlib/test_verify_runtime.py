"""Tests for runtime backtest verification."""

from __future__ import annotations

import sys
from pathlib import Path

import numpy as np
import pytest

PROJECT_ROOT = Path(__file__).parent.parent.parent
sys.path.insert(0, str(PROJECT_ROOT))


GOOD_RESULT = {
    "sharpe": 1.5,
    "max_drawdown": -0.15,
    "win_rate": 0.55,
    "total_return": 0.25,
    "annual_return_pct": 15.0,
    "monthly_return_pct": 1.2,
    "n_trades": 50,
    "status": "success",
}


class TestVerifyBacktestResult:
    def test_good_result_passes(self):
        from rdagent.components.backtesting.verify import verify_backtest_result
        assert verify_backtest_result(GOOD_RESULT) == []

    def test_missing_key_detected(self):
        from rdagent.components.backtesting.verify import verify_backtest_result
        bad = {**GOOD_RESULT}
        del bad["sharpe"]
        w = verify_backtest_result(bad)
        assert len(w) > 0
        assert any("Missing" in x for x in w)

    def test_max_dd_out_of_bounds(self):
        from rdagent.components.backtesting.verify import verify_backtest_result
        for val in [-1.5, 0.5]:
            bad = {**GOOD_RESULT, "max_drawdown": val}
            assert len(verify_backtest_result(bad)) > 0

    def test_win_rate_out_of_bounds(self):
        from rdagent.components.backtesting.verify import verify_backtest_result
        for val in [-0.1, 1.5]:
            bad = {**GOOD_RESULT, "win_rate": val}
            assert len(verify_backtest_result(bad)) > 0

    def test_infinite_sharpe(self):
        from rdagent.components.backtesting.verify import verify_backtest_result
        bad = {**GOOD_RESULT, "sharpe": float("inf")}
        assert len(verify_backtest_result(bad)) > 0

    def test_nan_total_return(self):
        from rdagent.components.backtesting.verify import verify_backtest_result
        bad = {**GOOD_RESULT, "total_return": float("nan")}
        assert len(verify_backtest_result(bad)) > 0

    def test_negative_trades(self):
        from rdagent.components.backtesting.verify import verify_backtest_result
        bad = {**GOOD_RESULT, "n_trades": -5}
        assert len(verify_backtest_result(bad)) > 0

    def test_opposite_signs(self):
        from rdagent.components.backtesting.verify import verify_backtest_result
        bad = {**GOOD_RESULT, "sharpe": 2.0, "annual_return_pct": -10.0}
        assert len(verify_backtest_result(bad)) > 0

    def test_invalid_status(self):
        from rdagent.components.backtesting.verify import verify_backtest_result
        bad = {**GOOD_RESULT, "status": "unknown"}
        assert len(verify_backtest_result(bad)) > 0

    def test_verify_and_log_returns_false_on_bad(self):
        from rdagent.components.backtesting.verify import verify_and_log
        assert verify_and_log({**GOOD_RESULT, "n_trades": -1}) is False

    def test_verify_and_log_returns_true_on_good(self):
        from rdagent.components.backtesting.verify import verify_and_log
        assert verify_and_log(GOOD_RESULT) is True


class TestRuntimeVerification:
    """Verify that backtest_signal automatically calls the verifier."""

    def test_backtest_signal_produces_verified_output(self):
        from rdagent.components.backtesting.vbt_backtest import backtest_signal
        import pandas as pd

        dates = pd.date_range("2024-01-01", periods=500, freq="1min")
        close = pd.Series(1.10 + np.random.default_rng(42).normal(0, 0.0001, 500).cumsum(), index=dates)
        signal = pd.Series(np.where(np.random.default_rng(99).normal(0, 1, 500) > 0, 1.0, -1.0), index=dates)

        result = backtest_signal(close, signal)
        # All fields should pass verification
        from rdagent.components.backtesting.verify import verify_backtest_result
        assert verify_backtest_result(result) == []
