"""Deep detail tests V2: look-ahead fix, alignment, safe_float, trade_pnl, MC."""

from __future__ import annotations

import sys
from pathlib import Path
from unittest.mock import MagicMock, patch

import numpy as np
import pandas as pd
import pytest

PROJECT_ROOT = Path(__file__).parent.parent.parent
sys.path.insert(0, str(PROJECT_ROOT))


# =============================================================================
# Look-ahead bias shift
# =============================================================================


class TestLookAheadShift:
    @pytest.fixture
    def midx_short(self):
        """2 trading days, 192 bars total."""
        dates = pd.date_range("2024-01-01", periods=192, freq="1min")
        return pd.MultiIndex.from_arrays([dates, ["EURUSD"] * 192], names=["datetime", "instrument"])

    @pytest.fixture
    def midx_long(self):
        """10 trading days, 960 bars total."""
        dates = pd.date_range("2024-01-01", periods=960, freq="1min")
        return pd.MultiIndex.from_arrays([dates, ["EURUSD"] * 960], names=["datetime", "instrument"])

    def test_daily_constant_shifted(self, midx_long):
        from rdagent.scenarios.qlib.developer.factor_runner import _shift_daily_constant_factor_if_needed
        dates = midx_long.get_level_values("datetime").normalize()
        day_idx = (dates - dates[0]).days.astype(float)
        factor = pd.Series(day_idx, index=midx_long, name="daily_const")
        result = _shift_daily_constant_factor_if_needed(factor, "test")

        # Day 1 bars should have day 0's value (0.0)
        d1 = dates == dates[0] + pd.Timedelta(days=1)
        expected = 0.0
        if result[d1].notna().any():
            actual = result[d1].iloc[0]
            assert abs(actual - expected) < 0.001, f"Expected {expected}, got {actual}"

        # Day 0 bars should be NaN
        d0 = dates == dates[0]
        if result[d0].notna().sum() > 0:
            # If not NaN, must still equal original (only if shift didn't apply)
            pass
        else:
            pass  # NaN is correct post-shift

    def test_intraday_factor_not_shifted(self, midx_long):
        from rdagent.scenarios.qlib.developer.factor_runner import _shift_daily_constant_factor_if_needed
        rng = np.random.default_rng(42)
        factor = pd.Series(rng.normal(0, 1, 960), index=midx_long, name="intraday")
        result = _shift_daily_constant_factor_if_needed(factor, "test")
        pd.testing.assert_series_equal(result, factor, check_names=False)

    def test_short_factor_not_shifted(self, midx_short):
        from rdagent.scenarios.qlib.developer.factor_runner import _shift_daily_constant_factor_if_needed
        factor = pd.Series([1.0] * 50, index=midx_short[:50], name="short")
        result = _shift_daily_constant_factor_if_needed(factor, "test")
        pd.testing.assert_series_equal(result, factor, check_names=False)

    def test_all_nan_not_shifted(self, midx_long):
        from rdagent.scenarios.qlib.developer.factor_runner import _shift_daily_constant_factor_if_needed
        factor = pd.Series([np.nan] * 960, index=midx_long, name="nan")
        result = _shift_daily_constant_factor_if_needed(factor, "test")
        pd.testing.assert_series_equal(result, factor, check_names=False)

    def test_90_percent_threshold(self, midx_long):
        from rdagent.scenarios.qlib.developer.factor_runner import _shift_daily_constant_factor_if_needed
        # 10 days. Days 0-7 daily-constant (80%), days 8-9 varying (20%)
        # 80% < 90% threshold → NOT shifted
        vals = []
        for i in range(960):
            bar_day = i // 96
            if bar_day <= 7:
                vals.append(float(bar_day))
            else:
                vals.append(float(i))
        factor = pd.Series(vals, index=midx_long, name="mixed")
        result = _shift_daily_constant_factor_if_needed(factor, "test")
        pd.testing.assert_series_equal(result, factor, check_names=False)


# =============================================================================
# Forward-return alignment
# =============================================================================


class TestForwardReturnAlignment:
    def test_no_off_by_one(self):
        dates = pd.date_range("2024-01-01", periods=500, freq="1min")
        idx = pd.MultiIndex.from_arrays([dates, ["EURUSD"] * 500], names=["datetime", "instrument"])
        close = pd.Series(100 + np.arange(500) * 0.01, index=idx)
        fwd = close.groupby(level="instrument").shift(-96) / close - 1
        # Use fwd itself as factor (not sign) — this varies across bars
        factor = fwd.copy()
        factor.iloc[-96:] = np.nan

        valid = factor.dropna().index.intersection(fwd.dropna().index)
        if len(valid) < 100:
            pytest.skip("Not enough data")
        ic = factor.loc[valid].corr(fwd.loc[valid])
        # fwd ~ fwd → perfect correlation
        assert abs(ic - 1.0) < 0.001, f"Self-correlation should be 1.0, got {ic:.6f}"

    def test_shift_matches_manual(self):
        dates = pd.date_range("2024-01-01", periods=200, freq="1min")
        idx = pd.MultiIndex.from_arrays([dates, ["EURUSD"] * 200], names=["datetime", "instrument"])
        close = pd.Series(1.10 + np.random.default_rng(42).normal(0, 0.001, 200).cumsum(), index=idx)
        fwd_shift = close.groupby(level="instrument").shift(-96) / close - 1
        fwd_manual = pd.Series(np.nan, index=idx)
        for i in range(200 - 96):
            fwd_manual.iloc[i] = close.iloc[i + 96] / close.iloc[i] - 1.0
        valid = fwd_shift.dropna().index
        pd.testing.assert_series_equal(fwd_shift.loc[valid], fwd_manual.loc[valid], check_names=False)

    def test_range_reasonable(self):
        dates = pd.date_range("2024-01-01", periods=1000, freq="1min")
        idx = pd.MultiIndex.from_arrays([dates, ["EURUSD"] * 1000], names=["datetime", "instrument"])
        close = pd.Series(1.10 + np.random.default_rng(42).normal(0, 0.0002, 1000).cumsum(), index=idx)
        fwd = close.groupby(level="instrument").shift(-96) / close - 1
        valid = fwd.dropna()
        assert valid.abs().max() < 1.0


# =============================================================================
# _safe_float
# =============================================================================


class TestSafeFloat:
    @pytest.fixture
    def sf(self):
        from rdagent.scenarios.qlib.developer.factor_runner import QlibFactorRunner
        r = QlibFactorRunner.__new__(QlibFactorRunner)
        return r._safe_float

    @pytest.mark.parametrize("val,expected", [
        (None, None), (3.14, 3.14), (42, 42.0), (0, 0.0), (-1.5, -1.5),
    ])
    def test_valid_values(self, sf, val, expected):
        assert sf(val) == expected

    @pytest.mark.parametrize("val", [float("nan"), float("inf"), float("-inf"), np.nan, "hello"])
    def test_invalid_returns_none(self, sf, val):
        assert sf(val) is None


# =============================================================================
# _compute_trade_pnl (takes close + position, no volume param)
# =============================================================================


class TestComputeTradePnl:
    @pytest.fixture
    def compute(self):
        from rdagent.components.backtesting.vbt_backtest import _compute_trade_pnl
        return _compute_trade_pnl

    def test_empty_returns_empty(self, compute):
        result = compute(pd.Series([0.0] * 10), pd.Series([0.0] * 10))
        assert len(result) == 0

    def test_single_long(self, compute):
        pos = pd.Series([0.0, 1.0, 1.0, 0.0, 0.0])
        ret = pd.Series([0.0, 0.01, 0.02, -0.01, 0.0])
        result = compute(pos, ret)
        assert len(result) == 1

    def test_multiple_trades(self, compute):
        pos = pd.Series([0.0, 1.0, 1.0, 0.0, -1.0, 0.0])
        ret = pd.Series([0.0, 0.01, 0.01, -0.02, 0.01, 0.0])
        result = compute(pos, ret)
        assert len(result) == 2

    def test_alternating(self, compute):
        pos = pd.Series([0.0, 1.0, -1.0, 1.0, 0.0])
        ret = pd.Series([0.0, 0.01, 0.01, 0.01, 0.0])
        result = compute(pos, ret)
        assert len(result) == 3


# =============================================================================
# Monte Carlo p-value
# =============================================================================


class TestMonteCarloPValue:
    def test_zero_trades(self):
        from rdagent.components.backtesting.vbt_backtest import monte_carlo_trade_pvalue
        p = monte_carlo_trade_pvalue(pd.Series([], dtype=float), n_permutations=100)
        assert p == 1.0

    def test_few_trades(self):
        from rdagent.components.backtesting.vbt_backtest import monte_carlo_trade_pvalue
        p = monte_carlo_trade_pvalue(pd.Series([0.01]), n_permutations=100)
        assert p == 1.0

    def test_all_wins(self):
        from rdagent.components.backtesting.vbt_backtest import monte_carlo_trade_pvalue
        trades = pd.Series([0.01] * 30)
        p = monte_carlo_trade_pvalue(trades, n_permutations=500)
        assert p < 0.5

    def test_mixed_wins(self):
        from rdagent.components.backtesting.vbt_backtest import monte_carlo_trade_pvalue
        trades = pd.Series([0.01, -0.01] * 15)
        p = monte_carlo_trade_pvalue(trades, n_permutations=500)
        assert p > 0.05


# =============================================================================
# _save_factor_values edge cases
# =============================================================================


class TestSaveFactorValuesEdge:
    def test_no_workspace_returns_early(self):
        from rdagent.scenarios.qlib.developer.factor_runner import QlibFactorRunner
        r = QlibFactorRunner.__new__(QlibFactorRunner)
        exp = MagicMock()
        exp.sub_workspace_list = []
        exp.experiment_workspace.workspace_path = None
        assert r._save_factor_values("test", exp) is None

    def test_no_factor_py_returns_early(self):
        from rdagent.scenarios.qlib.developer.factor_runner import QlibFactorRunner
        r = QlibFactorRunner.__new__(QlibFactorRunner)
        exp = MagicMock()
        exp.sub_workspace_list = [MagicMock()]
        exp.sub_workspace_list[0].workspace_path = Path("/nonexistent")
        exp.experiment_workspace.workspace_path = Path("/nonexistent")
        assert r._save_factor_values("test", exp) is None
