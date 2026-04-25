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
| "intraday_pv.h5"  | EURUSD 1-minute OHLCV intraday data (2020-2026).               |


# For different data, We have some basic knowledge for them

## 1-Minute Price and Volume data (EURUSD)
$open: open price at 1-minute bar.
$close: close price at 1-minute bar.
$high: high price at 1-minute bar.
$low: low price at 1-minute bar.
$volume: volume at 1-minute bar (tick volume for FX).

## Important Notes for 1min Data
- 1 bar = 1 minute (confirmed)
- 16 bars = 16 minutes
- 60 bars = 1 hour
- ~1440 bars = 1 full trading day (FX trades nearly 24h, Mon 00:00 - Fri 22:00 UTC approx.)
- Typical bars per calendar day: ~1200-1440 (varies by weekday, holidays have fewer)
- Do NOT assume 96 bars/day — the actual count depends on the date
- Data range: 2020-01-01 to 2026-03-20
- Instrument: EURUSD
- Timezone: UTC

## IMPORTANT: Bars per Day Correction
The dataset has approximately 1440 bars per full trading day (1 bar = 1 minute, ~24h of FX trading).
Some older documentation incorrectly stated "96 bars = 1 day" — this is WRONG. Always use:
- 60 bars = 1 hour
- 480 bars = 8 hours (London session 08:00-16:00 UTC)
- 180 bars = 3 hours (London/NY overlap 13:00-16:00 UTC)
Use datetime hour filtering (e.g., `df[df.index.get_level_values('datetime').hour.between(8, 15)]`)
to select session bars — do NOT use bar-count offsets to define sessions.

## Session Times (UTC)
- Asian: 00:00-08:00 UTC (low volatility)
- London: 08:00-16:00 UTC (high volatility)
- NY: 13:00-21:00 UTC (high volatility)
- Overlap: 13:00-16:00 UTC (highest volatility)
