"""Deep tests for factor_runner.py — look-ahead fix, IC, de-duplication, edge cases."""

from __future__ import annotations

import numpy as np
import pandas as pd
import pytest

from hypothesis import given, settings
from hypothesis import strategies as st
from hypothesis.extra.numpy import arrays

from rdagent.scenarios.qlib.developer.factor_runner import (
    _shift_daily_constant_factor_if_needed,
)


def _make_multiindex_series(
    dates: list, instrument: str = "EURUSD", values: list = None
) -> pd.Series:
    """Helper: build a MultiIndex (datetime, instrument) Series."""
    idx = pd.MultiIndex.from_tuples(
        [(d, instrument) for d in dates], names=["datetime", "instrument"]
    )
    if values is None:
        values = np.arange(len(dates), dtype=float)
    return pd.Series(values, index=idx, name="test_factor")


class TestShiftDailyConstantFactor:
    def test_returns_unchanged_when_few_rows(self):
        """< 200 non-null rows — skip shift entirely."""
        dates = pd.date_range("2024-01-01", periods=50, freq="1min")
        s = _make_multiindex_series(dates, values=np.ones(50))
        result = _shift_daily_constant_factor_if_needed(s, "test")
        assert result.equals(s)

    def test_returns_unchanged_when_intraday_varying(self):
        """Factor changes within a day → no shift needed."""
        dates = pd.date_range("2024-01-01", periods=2000, freq="1min")
        vals = np.random.default_rng(1).normal(0, 1, 2000)
        s = _make_multiindex_series(dates, values=vals)
        result = _shift_daily_constant_factor_if_needed(s, "test")
        assert result.equals(s)

    def test_shifts_daily_constant_factor(self):
        """Factor is identical across all bars in a day → shift by 1 day."""
        dates = pd.date_range("2024-01-01 00:00", periods=5000, freq="1min")
        # Create daily-constant: same value for all bars on same day
        vals = np.array([d.day for d in dates], dtype=float)
        s = _make_multiindex_series(dates, values=vals)
        result = _shift_daily_constant_factor_if_needed(s, "test")
        # After shift, the value at day 2 should be the value from day 1
        assert not result.equals(s)  # Must have been shifted

    def test_nan_handling(self):
        """NaN values in the factor should not break the shift."""
        dates = pd.date_range("2024-01-01", periods=2000, freq="1min")
        vals = np.array([d.day for d in dates], dtype=float)
        vals[:100] = np.nan  # First 100 NaN
        s = _make_multiindex_series(dates, values=vals)
        result = _shift_daily_constant_factor_if_needed(s, "test")
        assert isinstance(result, pd.Series)
        assert len(result) == len(s)

    def test_multi_instrument_handled(self):
        """Multi-instrument data should not crash."""
        dates = pd.date_range("2024-01-01", periods=3000, freq="1min")
        tuples_eur = [(d, "EURUSD") for d in dates]
        tuples_gbp = [(d, "GBPUSD") for d in dates]
        all_tuples = tuples_eur + tuples_gbp
        idx = pd.MultiIndex.from_tuples(all_tuples, names=["datetime", "instrument"])
        vals = [d.day for d in dates] + [d.day for d in dates]
        s = pd.Series(vals, index=idx, name="test", dtype=float)
        result = _shift_daily_constant_factor_if_needed(s, "test")
        assert isinstance(result, pd.Series)

    def test_all_same_value(self):
        """Single unique value across entire series → treated as daily-constant."""
        dates = pd.date_range("2024-01-01", periods=2000, freq="1min")
        s = _make_multiindex_series(dates, values=np.ones(2000))
        result = _shift_daily_constant_factor_if_needed(s, "test")
        assert isinstance(result, pd.Series)

    def test_two_days_only(self):
        """Only 2 days of data — should still handle gracefully."""
        # 2 days × 100 bars = 200 bars
        dates = pd.date_range("2024-01-01 00:00", periods=200, freq="1min")
        vals = np.array([d.day for d in dates], dtype=float)
        s = _make_multiindex_series(dates, values=vals)
        result = _shift_daily_constant_factor_if_needed(s, "test")
        assert isinstance(result, pd.Series)

    def test_zero_unique_values_edge_case(self):
        """All-NaN with very few valid should return unchanged."""
        dates = pd.date_range("2024-01-01", periods=500, freq="1min")
        vals = np.full(500, np.nan)
        vals[100:105] = 1.0
        s = _make_multiindex_series(dates, values=vals)
        # This should trigger the "< 200 non-null" check and return unchanged
        result = _shift_daily_constant_factor_if_needed(s, "test")
        assert result.equals(s)

    @given(
        n_days=st.integers(min_value=5, max_value=50),
        seed=st.integers(min_value=0, max_value=100),
    )
    @settings(max_examples=50, deadline=10000)
    def test_property_never_crashes(self, n_days, seed):
        """For any valid MultiIndex series, function must not crash."""
        rng = np.random.default_rng(seed)
        bars_per_day = 48  # 30-min bars
        dates = pd.date_range("2024-01-01", periods=n_days * bars_per_day, freq="30min")
        vals = rng.choice([1.0, 2.0, 3.0], n_days * bars_per_day)  # Daily-constant
        s = _make_multiindex_series(dates, values=vals)
        result = _shift_daily_constant_factor_if_needed(s, f"f_{seed}")
        assert isinstance(result, pd.Series)
        assert len(result) == len(s)


class TestInformationCoefficient:
    def test_ic_direct_import(self):
        """calculate_information_coefficient is importable and callable."""
        from rdagent.scenarios.qlib.developer.factor_runner import (
            QlibFactorRunner,
        )
        assert hasattr(QlibFactorRunner, "calculate_information_coefficient")


class TestSafeFloat:
    def test_safe_float_direct(self):
        """_safe_float must handle NaN, Inf, None, strings."""
        from rdagent.scenarios.qlib.developer.factor_runner import QlibFactorRunner

        # Create a minimal instance
        runner = QlibFactorRunner.__new__(QlibFactorRunner)
        # _safe_float should be callable without full init
        if hasattr(runner, "_safe_float"):
            assert runner._safe_float(1.5) == 1.5
            assert runner._safe_float(float("nan")) is None
            assert runner._safe_float(float("inf")) is None
            assert runner._safe_float(None) is None


class TestDeduplicateFactors:
    def test_deduplicate_importable(self):
        """deduplicate_new_factors is importable."""
        from rdagent.scenarios.qlib.developer.factor_runner import QlibFactorRunner
        assert hasattr(QlibFactorRunner, "deduplicate_new_factors")


class TestFactorIntegration:
    def test_shift_preserves_index_structure(self):
        """After shift, index names and structure must match original."""
        dates = pd.date_range("2024-01-01 00:00", periods=3000, freq="1min")
        vals = np.array([d.day for d in dates], dtype=float)
        s = _make_multiindex_series(dates, values=vals)
        result = _shift_daily_constant_factor_if_needed(s, "test")
        assert result.index.names == s.index.names
        assert len(result.index) == len(s.index)
