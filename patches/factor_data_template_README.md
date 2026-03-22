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
| "daily_pv.h5"  | EURUSD 15-minute OHLCV intraday data.                            |

# For different data, We have some basic knowledge for them

## EURUSD 15min intraday data
$open: open price of EURUSD at the start of the 15min bar.
$close: close price of EURUSD at the end of the 15min bar.
$high: highest price of EURUSD during the 15min bar.
$low: lowest price of EURUSD during the 15min bar.
$volume: traded volume during the 15min bar.

**IMPORTANT: There is NO $factor column. Use only $open, $close, $high, $low, $volume.**

## Market sessions (UTC)
- Asian session: 00:00 - 08:00 (mean reversion tendencies)
- London session: 08:00 - 16:00 (trending, momentum works)
- NY session: 13:00 - 21:00 (high volatility)
- London-NY overlap: 13:00 - 16:00 (highest volume)

## Lookback reference
- 4 bars = 1 hour
- 8 bars = 2 hours  
- 16 bars = 4 hours
- 32 bars = 8 hours
- 96 bars = 1 day
