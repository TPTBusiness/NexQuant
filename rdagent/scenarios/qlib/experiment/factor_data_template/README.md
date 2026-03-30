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
| "daily_pv.h5"  | EURUSD 1-minute price and volume data (2020-2026).               |


# For different data, We have some basic knowledge for them

## 1-Minute Price and Volume data (EURUSD)
$open: open price at 1-minute bar.
$close: close price at 1-minute bar.
$high: high price at 1-minute bar.
$low: low price at 1-minute bar.
$volume: volume at 1-minute bar (tick volume for FX).

## Important Notes for 1min Data
- 96 bars = 1 trading day (24 hours for FX)
- 16 bars = 16 minutes
- 4 bars = 4 minutes
- 1 bar = 1 minute
- Data range: 2020-01-01 to 2026-03-20
- Instrument: EURUSD
- Timezone: UTC

## Session Times (UTC)
- Asian: 00:00-08:00 UTC (low volatility)
- London: 08:00-16:00 UTC (high volatility)
- NY: 13:00-21:00 UTC (high volatility)
- Overlap: 13:00-16:00 UTC (highest volatility)
