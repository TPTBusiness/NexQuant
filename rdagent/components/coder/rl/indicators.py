"""
Technical Indicators for RL Trading.

Common technical indicators used as features for RL agents.
All functions operate on pandas Series/DataFrames and return the same.
"""

from typing import List, Optional

import numpy as np
import pandas as pd


def calculate_rsi(prices: pd.Series, period: int = 14) -> pd.Series:
    """
    Relative Strength Index (RSI).

    Momentum oscillator measuring speed and change of price movements.
    Values range from 0 to 100. Above 70 = overbought, below 30 = oversold.

    Parameters
    ----------
    prices : pd.Series
        Close price series
    period : int
        RSI calculation period

    Returns
    -------
    pd.Series
        RSI values
    """
    delta = prices.diff()
    gain = delta.where(delta > 0, 0.0)
    loss = -delta.where(delta < 0, 0.0)

    avg_gain = gain.rolling(window=period, min_periods=period).mean()
    avg_loss = loss.rolling(window=period, min_periods=period).mean()

    rs = avg_gain / avg_loss
    return 100.0 - (100.0 / (1.0 + rs))


def calculate_macd(
    prices: pd.Series,
    fast: int = 12,
    slow: int = 26,
    signal: int = 9,
) -> pd.DataFrame:
    """
    Moving Average Convergence Divergence (MACD).

    Trend-following momentum indicator showing the relationship between
    two exponential moving averages.

    Parameters
    ----------
    prices : pd.Series
        Close price series
    fast : int
        Fast EMA period
    slow : int
        Slow EMA period
    signal : int
        Signal line EMA period

    Returns
    -------
    pd.DataFrame
        DataFrame with columns: macd, signal, histogram
    """
    ema_fast = prices.ewm(span=fast, adjust=False).mean()
    ema_slow = prices.ewm(span=slow, adjust=False).mean()

    macd_line = ema_fast - ema_slow
    signal_line = macd_line.ewm(span=signal, adjust=False).mean()
    histogram = macd_line - signal_line

    return pd.DataFrame(
        {"macd": macd_line, "signal": signal_line, "histogram": histogram}
    )


def calculate_bollinger_bands(
    prices: pd.Series,
    period: int = 20,
    std_dev: float = 2.0,
) -> pd.DataFrame:
    """
    Bollinger Bands.

    Volatility bands placed above and below a moving average.
    Band width expands/contracts with volatility.

    Parameters
    ----------
    prices : pd.Series
        Close price series
    period : int
        Moving average period
    std_dev : float
        Number of standard deviations for bands

    Returns
    -------
    pd.DataFrame
        DataFrame with columns: upper, middle, lower
    """
    sma = prices.rolling(window=period).mean()
    std = prices.rolling(window=period).std()

    upper = sma + (std * std_dev)
    lower = sma - (std * std_dev)

    return pd.DataFrame({"upper": upper, "middle": sma, "lower": lower})


def calculate_cci(prices: pd.Series, high: pd.Series, low: pd.Series, period: int = 20) -> pd.Series:
    """
    Commodity Channel Index (CCI).

    Momentum-based oscillator used to determine when an asset is
    overbought or oversold.

    Parameters
    ----------
    prices : pd.Series
        Close price series
    high : pd.Series
        High price series
    low : pd.Series
        Low price series
    period : int
        CCI calculation period

    Returns
    -------
    pd.Series
        CCI values
    """
    typical_price = (high + low + prices) / 3.0
    sma_tp = typical_price.rolling(window=period).mean()
    mad = typical_price.rolling(window=period).apply(
        lambda x: np.abs(x - x.mean()).mean(), raw=False
    )

    cci = (typical_price - sma_tp) / (0.015 * mad)
    return cci


def calculate_atr(high: pd.Series, low: pd.Series, close: pd.Series, period: int = 14) -> pd.Series:
    """
    Average True Range (ATR).

    Volatility indicator measuring market volatility.

    Parameters
    ----------
    high : pd.Series
        High price series
    low : pd.Series
        Low price series
    close : pd.Series
        Close price series
    period : int
        ATR calculation period

    Returns
    -------
    pd.Series
        ATR values
    """
    prev_close = close.shift(1)

    tr1 = high - low
    tr2 = (high - prev_close).abs()
    tr3 = (low - prev_close).abs()

    true_range = pd.concat([tr1, tr2, tr3], axis=1).max(axis=1)
    return true_range.rolling(window=period).mean()


def prepare_features(
    prices: pd.DataFrame,
    indicator_list: Optional[List[str]] = None,
) -> pd.DataFrame:
    """
    Prepare features for RL agent from price data.

    Calculates requested technical indicators and concatenates
    them into a single features DataFrame.

    Parameters
    ----------
    prices : pd.DataFrame
        Price data with at least 'close' column.
        Optionally: 'high', 'low', 'volume'
    indicator_list : list, optional
        List of indicator names to calculate.
        Default: ['rsi', 'macd', 'bollinger', 'sma']

    Returns
    -------
    pd.DataFrame
        Features DataFrame with original prices + indicators.
        NaN values are filled with 0.

    Examples
    --------
    >>> df = pd.DataFrame({'close': [100, 101, 102, ...]})
    >>> features = prepare_features(df, ['rsi', 'macd'])
    """
    if indicator_list is None:
        indicator_list = ["rsi", "macd", "bollinger", "sma"]

    features = prices.copy()

    if "rsi" in indicator_list:
        features["rsi"] = calculate_rsi(prices["close"])

    if "macd" in indicator_list:
        macd_df = calculate_macd(prices["close"])
        features = pd.concat([features, macd_df], axis=1)

    if "bollinger" in indicator_list:
        bb_df = calculate_bollinger_bands(prices["close"])
        features = pd.concat([features, bb_df], axis=1)

    if "sma" in indicator_list:
        features["sma_20"] = prices["close"].rolling(window=20).mean()
        features["sma_50"] = prices["close"].rolling(window=50).mean()

    if "cci" in indicator_list and "high" in prices.columns and "low" in prices.columns:
        features["cci"] = calculate_cci(prices["close"], prices["high"], prices["low"])

    if "atr" in indicator_list and "high" in prices.columns and "low" in prices.columns:
        features["atr"] = calculate_atr(prices["high"], prices["low"], prices["close"])

    # Fill NaN values (from rolling calculations) with 0
    features = features.fillna(0.0)

    return features
