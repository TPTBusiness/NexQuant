import subprocess  # nosec B404
import sys
import os

# Qlib läuft in rdagent4qlib environment
result = subprocess.run( # nosec B603
    ["/home/nico/miniconda3/envs/rdagent4qlib/bin/python3", "-c", """
import qlib
from qlib.data import D
qlib.init(provider_uri="~/.qlib/qlib_data/eurusd_1min_data")
fields = ["$open", "$close", "$high", "$low", "$volume"]
data = (D.features(["EURUSD"], fields, start_time="2022-03-14", end_time="2026-03-20", freq="1min")
        .swaplevel().sort_index())
data.to_hdf("./intraday_pv_all.h5", key="data")
data_debug = (D.features(["EURUSD"], fields, start_time="2024-01-01", end_time="2026-03-20", freq="1min")
              .swaplevel().sort_index())
data_debug.to_hdf("./intraday_pv_debug.h5", key="data")
print(f"Done: {data.shape[0]} rows")
"""],
    capture_output=False
)
