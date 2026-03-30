import qlib

# EURUSD 1-Minuten Daten verwenden
qlib.init(provider_uri="~/.qlib/qlib_data/eurusd_1min_data")

from qlib.data import D

instruments = D.instruments()
fields = ["$open", "$close", "$high", "$low", "$volume"]

# 1min Daten für EURUSD
# Start: 2020-01-01, End: 2026-03-20
data = D.features(instruments, fields, freq="1min").swaplevel().sort_index()

data.to_hdf("./daily_pv_all.h5", key="data")


# Debug-Daten: Nur letzte ~100 Instrumente für schnelleres Testing
fields = ["$open", "$close", "$high", "$low", "$volume"]
data_debug = (
    D.features(instruments, fields, start_time="2024-01-01", end_time="2024-12-31", freq="1min")
    .swaplevel()
    .sort_index()
)

# Nimm erste 100 unique instruments
unique_inst = data_debug.reset_index()["instrument"].unique()[:100]
data_debug = (
    data_debug.swaplevel()
    .loc[unique_inst]
    .swaplevel()
    .sort_index()
)

data_debug.to_hdf("./daily_pv_debug.h5", key="data")

print(f"Generated daily_pv_all.h5 with {len(data)} rows")
print(f"Generated daily_pv_debug.h5 with {len(data_debug)} rows")
print(f"Date range: {data.index.min()} to {data.index.max()}")
