import json
import pandas as pd
import numpy as np

# Strategy parameters
factors_used = ["daily_ret", "daily_close_return_96", "daily_cc_return", "momentum_1d", "london_mom"]
strategy_name = "ActiveDayMultiFactorScalper"
description = "Daytrading-Strategie mit 5 niedrig-korrelierten Faktoren und niedrigen Schwellenwerten für 50+ Trades"

# Python code for signal generation
code = '''import numpy as np
import pandas as pd

# Rolling Z-Scores mit kurzen Fenstern für schnelle Signale
z_daily_ret = (factors["daily_ret"] - factors["daily_ret"].rolling(15).mean()) / factors["daily_ret"].rolling(15).std()
z_close_ret = (factors["daily_close_return_96"] - factors["daily_close_return_96"].rolling(20).mean()) / factors["daily_close_return_96"].rolling(20).std()
z_cc_ret = (factors["daily_cc_return"] - factors["daily_cc_return"].rolling(15).mean()) / factors["daily_cc_return"].rolling(15).std()
z_mom = (factors["momentum_1d"] - factors["momentum_1d"].rolling(25).mean()) / factors["momentum_1d"].rolling(25).std()
z_london = (factors["london_mom"] - factors["london_mom"].rolling(30).mean()) / factors["london_mom"].rolling(30).std()

# Kombiniere alle Z-Scores mit Gewichtung
composite_signal = (
    0.25 * z_close_ret +  # Höchste IC (0.255) - stärkstes Gewicht
    0.20 * z_london +     # Zweithöchste IC (0.1857)
    0.20 * z_daily_ret +  # IC 0.1291
    0.20 * z_cc_ret +     # IC 0.1291
    0.15 * z_mom          # IC 0.1291
)

# Niedrige Schwellenwerte für häufigere Signale (0.2-0.3)
threshold_long = 0.25
threshold_short = -0.25

# Signal generieren
signal = pd.Series(0, index=close.index, name="signal")
signal[composite_signal > threshold_long] = 1
signal[composite_signal < threshold_short] = -1

# NaN behandeln (am Anfang durch rolling window)
signal = signal.fillna(0).astype(int)
'''

# Create strategy dict
strategy = {
    "strategy_name": strategy_name,
    "factor_names": factors_used,
    "description": description,
    "code": code
}

# Save to JSON
output_file = f"{strategy_name}_strategy.json"
with open(output_file, "w") as f:
    json.dump(strategy, f, indent=2)

print(f"✅ Strategie gespeichert: {output_file}")
print(f"📊 Faktoren: {', '.join(factors_used)}")
print(f"🎯 Ziel: 50+ Trades mit niedrigen Schwellenwerten (±0.25)")
