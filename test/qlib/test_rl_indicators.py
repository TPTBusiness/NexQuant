"""Tests for rl/indicators.py — pure technical indicator functions."""

from __future__ import annotations

import sys
from pathlib import Path

import numpy as np
import pandas as pd
import pytest

PROJECT_ROOT = Path(__file__).parent.parent.parent
sys.path.insert(0, str(PROJECT_ROOT))


def _load_indicators():
    import importlib.util
    spec = importlib.util.spec_from_file_location(
        "indicators",
        PROJECT_ROOT / "rdagent/components/coder/rl/indicators.py",
    )
    mod = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mod)
    return mod


@pytest.fixture(scope="module")
def indicators():
    return _load_indicators()


@pytest.fixture
def prices():
    rng = np.random.default_rng(42)
    return pd.Series(100 + rng.normal(0, 1, 200).cumsum())


class TestRSI:
    def test_returns_series(self, indicators, prices):
        rsi = indicators.calculate_rsi(prices, period=14)
        assert isinstance(rsi, pd.Series)
        assert len(rsi) == len(prices)

    def test_first_period_is_nan(self, indicators, prices):
        rsi = indicators.calculate_rsi(prices, period=14)
        assert rsi.iloc[:13].isna().all()
        assert not np.isnan(rsi.iloc[14])

    def test_range_between_0_and_100(self, indicators, prices):
        rsi = indicators.calculate_rsi(prices, period=14)
        valid = rsi.dropna()
        assert (valid >= 0).all()
        assert (valid <= 100).all()

    def test_constant_prices_gives_neutral_rsi(self, indicators):
        const = pd.Series([100.0] * 50)
        rsi = indicators.calculate_rsi(const, period=14)
        # With no change, gain=loss=0 → RSI = NaN (division by zero)
        valid = rsi.dropna()
        assert len(valid) == 0  # all NaN when no movement


class TestMACD:
    def test_returns_dataframe(self, indicators, prices):
        macd = indicators.calculate_macd(prices)
        assert isinstance(macd, pd.DataFrame)
        assert list(macd.columns) == ["macd", "signal", "histogram"]

    def test_histogram_is_macd_minus_signal(self, indicators, prices):
        macd = indicators.calculate_macd(prices)
        computed = macd["macd"] - macd["signal"]
        pd.testing.assert_series_equal(macd["histogram"], computed, check_names=False)


class TestBollinger:
    def test_returns_dataframe(self, indicators, prices):
        bb = indicators.calculate_bollinger_bands(prices, period=20)
        assert isinstance(bb, pd.DataFrame)
        assert list(bb.columns) == ["upper", "middle", "lower"]

    def test_upper_above_middle_lower_below(self, indicators, prices):
        bb = indicators.calculate_bollinger_bands(prices, period=20)
        valid = bb.dropna()
        assert (valid["upper"] > valid["middle"]).all()
        assert (valid["lower"] < valid["middle"]).all()


class TestATR:
    def test_returns_series(self, indicators, prices):
        high = prices * 1.01
        low = prices * 0.99
        close = prices
        atr = indicators.calculate_atr(high, low, close, period=14)
        assert isinstance(atr, pd.Series)
        assert len(atr) == len(prices)

    def test_non_negative(self, indicators, prices):
        high = prices * 1.01
        low = prices * 0.99
        atr = indicators.calculate_atr(high, low, prices, period=14)
        valid = atr.dropna()
        assert (valid >= 0).all()


class TestCCI:
    def test_returns_series(self, indicators, prices):
        cci = indicators.calculate_cci(prices, prices * 1.01, prices * 0.99, period=20)
        assert isinstance(cci, pd.Series)


class TestPrepareFeatures:
    def test_returns_dataframe_with_columns(self, indicators, prices):
        df_input = pd.DataFrame({"close": prices})
        df = indicators.prepare_features(df_input, ["rsi", "macd"])
        assert isinstance(df, pd.DataFrame)
        assert len(df.columns) >= 3  # close + at least rsi + macd columns
