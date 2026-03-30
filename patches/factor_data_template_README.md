# How to read files.
For example, if you want to read `filename.h5`
```Python
import pandas as pd
df = pd.read_hdf("filename.h5", key="data")
```
NOTE: **key is always "data" for all hdf5 files **.

# Here is a short description about the data
| Filename       | Description                                                      |
| -------------- | -----------------------------------------------------------------|
| "daily_pv.h5"  | EURUSD 1-minute OHLCV intraday data (2020-2026).                 |

# For different data, We have some basic knowledge for them

## EURUSD 1min intraday data
$open: open price of EURUSD at the start of the 1min bar.
$close: close price of EURUSD at the end of the 1min bar.
$high: highest price of EURUSD during the 1min bar.
$low: lowest price of EURUSD during the 1min bar.
$volume: traded volume during the 1min bar (tick volume for FX).

**IMPORTANT: There is NO $factor column. Use only $open, $close, $high, $low, $volume.**

## Market sessions (UTC)
- Asian session: 00:00 - 08:00 (mean reversion tendencies)
- London session: 08:00 - 16:00 (trending, momentum works)
- NY session: 13:00 - 21:00 (high volatility)
- London-NY overlap: 13:00 - 16:00 (highest volume)

## Lookback reference for 1min data
- 4 bars = 4 minutes
- 8 bars = 8 minutes
- 16 bars = 16 minutes
- 32 bars = 32 minutes
- 96 bars = 1.6 hours
- 1440 bars = 1 day (24 hours)

## Data range
- Start: 2020-01-01 17:00:00 UTC
- End: 2026-03-20 15:58:00 UTC
- Total bars: ~2.26 million
