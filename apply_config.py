#!/usr/bin/env python3
"""
Liest data_config.yaml und schreibt alle Werte in:
- .env (Zeiträume, Pfade)
- generate.py (Qlib Datengenerierung)
"""
import yaml
import re
from pathlib import Path

CONFIG = Path(__file__).parent / "data_config.yaml"
ENV = Path(__file__).parent / ".env"
GENERATE = Path("/home/nico/miniconda3/envs/rdagent/lib/python3.10/site-packages/rdagent/scenarios/qlib/experiment/factor_data_template/generate.py")

with open(CONFIG) as f:
    cfg = yaml.safe_load(f)

# --- .env updaten ---
env_text = ENV.read_text()

replacements = {
    r"QLIB_DATA_DIR=.*": f"QLIB_DATA_DIR={cfg['data_path'].replace('~', str(Path.home()))}",
    r"QLIB_FREQ=.*": f"QLIB_FREQ={cfg['frequency']}",
    r"QLIB_FACTOR_TRAIN_START=.*": f"QLIB_FACTOR_TRAIN_START={cfg['train_start']}",
    r"QLIB_FACTOR_TRAIN_END=.*": f"QLIB_FACTOR_TRAIN_END={cfg['train_end']}",
    r"QLIB_FACTOR_VALID_START=.*": f"QLIB_FACTOR_VALID_START={cfg['valid_start']}",
    r"QLIB_FACTOR_VALID_END=.*": f"QLIB_FACTOR_VALID_END={cfg['valid_end']}",
    r"QLIB_FACTOR_TEST_START=.*": f"QLIB_FACTOR_TEST_START={cfg['test_start']}",
    r"QLIB_FACTOR_TEST_END=.*": f"QLIB_FACTOR_TEST_END={cfg['test_end']}",
    r"QLIB_MODEL_TRAIN_START=.*": f"QLIB_MODEL_TRAIN_START={cfg['train_start']}",
    r"QLIB_MODEL_TRAIN_END=.*": f"QLIB_MODEL_TRAIN_END={cfg['train_end']}",
    r"QLIB_MODEL_VALID_START=.*": f"QLIB_MODEL_VALID_START={cfg['valid_start']}",
    r"QLIB_MODEL_VALID_END=.*": f"QLIB_MODEL_VALID_END={cfg['valid_end']}",
    r"QLIB_MODEL_TEST_START=.*": f"QLIB_MODEL_TEST_START={cfg['test_start']}",
    r"QLIB_MODEL_TEST_END=.*": f"QLIB_MODEL_TEST_END={cfg['test_end']}",
    r"QLIB_QUANT_TRAIN_START=.*": f"QLIB_QUANT_TRAIN_START={cfg['train_start']}",
    r"QLIB_QUANT_TRAIN_END=.*": f"QLIB_QUANT_TRAIN_END={cfg['train_end']}",
    r"QLIB_QUANT_VALID_START=.*": f"QLIB_QUANT_VALID_START={cfg['valid_start']}",
    r"QLIB_QUANT_VALID_END=.*": f"QLIB_QUANT_VALID_END={cfg['valid_end']}",
    r"QLIB_QUANT_TEST_START=.*": f"QLIB_QUANT_TEST_START={cfg['test_start']}",
    r"QLIB_QUANT_TEST_END=.*": f"QLIB_QUANT_TEST_END={cfg['test_end']}",
}

for pattern, replacement in replacements.items():
    env_text = re.sub(pattern, replacement, env_text)

ENV.write_text(env_text)
print("✓ .env aktualisiert")

# --- generate.py updaten ---
data_path = cfg['data_path']
freq = cfg['frequency']
train_start = cfg['train_start']
test_end = cfg['test_end']
valid_start = cfg['valid_start']
cols = str(cfg['columns'])

generate_text = f'''import qlib
import pandas as pd

qlib.init(provider_uri="{data_path}", freq="{freq}")

from qlib.data import D

instruments = D.instruments(market="all")
fields = {cols}

data = (
    D.features(instruments, fields, freq="{freq}")
    .swaplevel()
    .sort_index()
    .loc["{train_start}":]
    .sort_index()
)
data.to_hdf("./daily_pv_all.h5", key="data")

data_debug = (
    D.features(instruments, fields, start_time="{valid_start}", end_time="{test_end}", freq="{freq}")
    .swaplevel()
    .sort_index()
)
data_debug.to_hdf("./daily_pv_debug.h5", key="data")
'''

GENERATE.write_text(generate_text)
print("✓ generate.py aktualisiert")
print(f"\nKonfiguration angewendet:")
print(f"  Instrument: {cfg['instrument']}")
print(f"  Frequenz:   {cfg['frequency']}")
print(f"  Train:      {cfg['train_start']} → {cfg['train_end']}")
print(f"  Test:       {cfg['test_start']} → {cfg['test_end']}")
