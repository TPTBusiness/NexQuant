"""
Tests for Technical Indicators.

Covers:
- RSI calculation
- MACD calculation
- Bollinger Bands calculation
- CCI calculation
- ATR calculation
- prepare_features integration
- Edge cases (NaN handling, short series)
"""

import numpy as np
import pandas as pd
import pytest

from rdagent.components.coder.rl.indicators import (
    calculate_atr,
    calculate_bollinger_bands,
    calculate_cci,
    calculate_macd,
    calculate_rsi,
    prepare_features,
)


# =============================================================================
# FIXTURES
# =============================================================================


@pytest.fixture
def price_data() -> pd.DataFrame:
    """Generate 100 bars of realistic price data."""
    np.random.seed(42)
    n = 100
    close = 100.0 + np.cumsum(np.random.randn(n) * 0.5)
    high = close + np.abs(np.random.randn(n) * 0.3)
    low = close - np.abs(np.random.randn(n) * 0.3)
    volume = np.random.randint(1000, 10000, n)

    return pd.DataFrame(
        {"close": close, "high": high, "low": low, "volume": volume},
        index=pd.date_range("2024-01-01", periods=n, freq="B"),
    )


# =============================================================================
# RSI
# =============================================================================


class TestRSI:
    """Test RSI calculation."""

    def test_rsi_values_in_range(self, price_data: pd.DataFrame) -> None:
        """RSI should be between 0 and 100."""
        rsi = calculate_rsi(price_data["close"], period=14)
        # Skip NaN values at the beginning
        valid_rsi = rsi.dropna()
        assert (valid_rsi >= 0).all()
        assert (valid_rsi <= 100).all()

    def test_rsi_default_period(self, price_data: pd.DataFrame) -> None:
        """Default RSI period should be 14."""
        rsi = calculate_rsi(price_data["close"])
        assert len(rsi) == len(price_data)

    def test_rsi_custom_period(self, price_data: pd.DataFrame) -> None:
        """Custom period should be respected."""
        rsi_7 = calculate_rsi(price_data["close"], period=7)
        rsi_21 = calculate_rsi(price_data["close"], period=21)

        # Shorter period = more valid values at start
        assert rsi_7.dropna().iloc[0] >= 0
        assert rsi_7.dropna().iloc[0] <= 100

    def test_rsi_nan_at_start(self, price_data: pd.DataFrame) -> None:
        """RSI should have NaN values at the beginning (period-1)."""
        rsi = calculate_rsi(price_data["close"], period=14)
        # First 13 values should be NaN (need 14 for rolling mean)
        assert rsi.iloc[:13].isna().all()
        # Value at index 13 should be valid
        assert not np.isnan(rsi.iloc[13])

    def test_rsi_short_series(self) -> None:
        """RSI should handle short series gracefully."""
        prices = pd.Series([100.0, 101.0, 102.0])
        rsi = calculate_rsi(prices, period=14)
        # All should be NaN (not enough data)
        assert rsi.isna().all()


# =============================================================================
# MACD
# =============================================================================


class TestMACD:
    """Test MACD calculation."""

    def test_macd_output_columns(self, price_data: pd.DataFrame) -> None:
        """MACD should return DataFrame with macd, signal, histogram."""
        macd_df = calculate_macd(price_data["close"])
        assert "macd" in macd_df.columns
        assert "signal" in macd_df.columns
        assert "histogram" in macd_df.columns

    def test_macd_histogram_consistency(self, price_data: pd.DataFrame) -> None:
        """Histogram should equal MACD - Signal."""
        macd_df = calculate_macd(price_data["close"])
        valid = macd_df.dropna()
        expected_histogram = valid["macd"] - valid["signal"]
        np.testing.assert_array_almost_equal(
            valid["histogram"].values,
            expected_histogram.values,
            decimal=10,
        )

    def test_macd_custom_parameters(self, price_data: pd.DataFrame) -> None:
        """Custom MACD parameters should be respected."""
        macd_df = calculate_macd(price_data["close"], fast=6, slow=13, signal=4)
        assert len(macd_df) == len(price_data)


# =============================================================================
# BOLLINGER BANDS
# =============================================================================


class TestBollingerBands:
    """Test Bollinger Bands calculation."""

    def test_bb_output_columns(self, price_data: pd.DataFrame) -> None:
        """Bollinger Bands should return upper, middle, lower."""
        bb_df = calculate_bollinger_bands(price_data["close"])
        assert "upper" in bb_df.columns
        assert "middle" in bb_df.columns
        assert "lower" in bb_df.columns

    def test_bb_ordering(self, price_data: pd.DataFrame) -> None:
        """Upper >= Middle >= Lower for valid data."""
        bb_df = calculate_bollinger_bands(price_data["close"]).dropna()
        assert (bb_df["upper"] >= bb_df["middle"]).all()
        assert (bb_df["middle"] >= bb_df["lower"]).all()

    def test_bb_middle_is_sma(self, price_data: pd.DataFrame) -> None:
        """Middle band should equal SMA."""
        bb_df = calculate_bollinger_bands(price_data["close"], period=20)
        sma = price_data["close"].rolling(window=20).mean()
        valid = bb_df.dropna()
        np.testing.assert_array_almost_equal(
            valid["middle"].values,
            sma.dropna().values,
            decimal=10,
        )

    def test_bb_custom_std_dev(self, price_data: pd.DataFrame) -> None:
        """Custom std_dev should affect band width."""
        bb_1 = calculate_bollinger_bands(price_data["close"], std_dev=1.0).dropna()
        bb_2 = calculate_bollinger_bands(price_data["close"], std_dev=3.0).dropna()

        # Higher std_dev = wider bands
        width_1 = (bb_1["upper"] - bb_1["lower"]).mean()
        width_2 = (bb_2["upper"] - bb_2["lower"]).mean()
        assert width_2 > width_1


# =============================================================================
# CCI
# =============================================================================


class TestCCI:
    """Test CCI calculation."""

    def test_cci_values(self, price_data: pd.DataFrame) -> None:
        """CCI should produce finite values after warmup."""
        cci = calculate_cci(
            price_data["close"], price_data["high"], price_data["low"], period=20
        )
        valid_cci = cci.dropna()
        assert len(valid_cci) > 0
        assert np.all(np.isfinite(valid_cci))

    def test_cci_nan_at_start(self, price_data: pd.DataFrame) -> None:
        """CCI should have NaN at the beginning."""
        cci = calculate_cci(
            price_data["close"], price_data["high"], price_data["low"], period=20
        )
        # First ~19 values should be NaN
        assert cci.iloc[:19].isna().any()


# =============================================================================
# ATR
# =============================================================================


class TestATR:
    """Test ATR calculation."""

    def test_atr_positive(self, price_data: pd.DataFrame) -> None:
        """ATR should be positive (for non-zero price changes)."""
        atr = calculate_atr(
            price_data["high"], price_data["low"], price_data["close"], period=14
        )
        valid_atr = atr.dropna()
        assert (valid_atr > 0).all()

    def test_atr_nan_at_start(self, price_data: pd.DataFrame) -> None:
        """ATR should have NaN at the beginning."""
        atr = calculate_atr(
            price_data["high"], price_data["low"], price_data["close"], period=14
        )
        # First value should be NaN (no previous close)
        assert np.isnan(atr.iloc[0])


# =============================================================================
# PREPARE FEATURES
# =============================================================================


class TestPrepareFeatures:
    """Test feature preparation integration."""

    def test_default_indicators(self, price_data: pd.DataFrame) -> None:
        """Default should include rsi, macd, bollinger, sma."""
        features = prepare_features(price_data)

        assert "rsi" in features.columns
        assert "macd" in features.columns
        assert "signal" in features.columns
        assert "histogram" in features.columns
        assert "upper" in features.columns
        assert "middle" in features.columns
        assert "lower" in features.columns
        assert "sma_20" in features.columns
        assert "sma_50" in features.columns

    def test_custom_indicator_list(self, price_data: pd.DataFrame) -> None:
        """Only requested indicators should be included."""
        features = prepare_features(price_data, indicator_list=["rsi"])

        assert "rsi" in features.columns
        # MACD and Bollinger should NOT be present
        assert "macd" not in features.columns
        assert "upper" not in features.columns

    def test_no_nan_in_output(self, price_data: pd.DataFrame) -> None:
        """Output should have no NaN values."""
        features = prepare_features(price_data)
        assert not features.isna().any().any()

    def test_preserves_original_columns(self, price_data: pd.DataFrame) -> None:
        """Original price columns should be preserved."""
        features = prepare_features(price_data)
        for col in price_data.columns:
            assert col in features.columns

    def test_empty_indicator_list(self, price_data: pd.DataFrame) -> None:
        """Empty list should return only original data."""
        features = prepare_features(price_data, indicator_list=[])
        assert list(features.columns) == list(price_data.columns)

    def test_close_only_dataframe(self) -> None:
        """Should work with only 'close' column."""
        np.random.seed(42)
        prices = pd.DataFrame(
            {"close": 100.0 + np.cumsum(np.random.randn(100) * 0.5)}
        )
        features = prepare_features(prices, indicator_list=["rsi"])
        assert "rsi" in features.columns
        assert not features.isna().any().any()
