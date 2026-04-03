"""
Qlib Factor Coder - Generates trading factors using LLM.

Integrates with technical indicators module to provide
available indicator functions for factor implementation.
"""

from rdagent.components.coder.factor_coder import FactorCoSTEER
from rdagent.core.scenario import Scenario


# Technical indicators documentation string for LLM prompts
TECHNICAL_INDICATORS_DOCSTRING = """
## Available Technical Indicator Functions

You can use these pre-implemented technical indicators in your factor implementations:

```python
from rdagent.components.coder.rl.indicators import (
    calculate_rsi,              # Relative Strength Index (0-100, overbought/oversold)
    calculate_macd,             # MACD (Moving Average Convergence Divergence)
    calculate_bollinger_bands,  # Bollinger Bands (upper, middle, lower)
    calculate_cci,              # Commodity Channel Index
    calculate_atr,              # Average True Range (volatility)
    prepare_features            # Combine all indicators
)

# Example usage:
rsi = calculate_rsi(df['close'], period=14)
macd_df = calculate_macd(df['close'])
bb_df = calculate_bollinger_bands(df['close'], period=20, std_dev=2.0)
cci = calculate_cci(df['close'], df['high'], df['low'], period=20)
atr = calculate_atr(df['high'], df['low'], df['close'], period=14)
```

All functions return pandas Series or DataFrames ready to be used as factor values.

### Indicator Descriptions

- **RSI (Relative Strength Index)**: Momentum oscillator, range 0-100.
  - Above 70 = overbought (potential reversal down)
  - Below 30 = oversold (potential reversal up)

- **MACD (Moving Average Convergence Divergence)**: Trend-following momentum.
  - Returns DataFrame with 'macd', 'signal', 'histogram' columns
  - Crossovers indicate potential trend changes

- **Bollinger Bands**: Volatility bands around moving average.
  - Returns DataFrame with 'upper', 'middle', 'lower' columns
  - Price near upper band = potentially overbought
  - Price near lower band = potentially oversold

- **CCI (Commodity Channel Index)**: Momentum oscillator.
  - Above +100 = overbought
  - Below -100 = oversold

- **ATR (Average True Range)**: Volatility measure.
  - Higher values = more volatile market
  - Useful for dynamic stop-loss placement
"""


class QlibFactorCoSTEER(FactorCoSTEER):
    """
    Qlib-specific Factor Coder that includes technical indicators documentation.

    Enhances the scenario with available technical indicator functions
    so the LLM knows what tools it can use for factor generation.
    """

    def __init__(self, scen: Scenario, *args, **kwargs) -> None:
        # Add technical indicators documentation to scenario
        if hasattr(scen, "factor_knowledge"):
            scen.factor_knowledge += TECHNICAL_INDICATORS_DOCSTRING
        elif hasattr(scen, "__dict__"):
            scen.technical_indicators_doc = TECHNICAL_INDICATORS_DOCSTRING

        super().__init__(scen, *args, **kwargs)


# Keep the alias for backward compatibility
QlibFactorCoSTEER = QlibFactorCoSTEER
