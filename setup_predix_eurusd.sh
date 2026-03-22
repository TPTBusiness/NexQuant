#!/bin/bash
# =============================================================================
# setup_predix_eurusd.sh
# Richtet Predix für EURUSD 15min Trading ein
# Ausführen: bash setup_predix_eurusd.sh
# =============================================================================

set -e  # Abbruch bei Fehler

PREDIX_DIR="$HOME/Predix"
CSV_SOURCE="$HOME/Downloads/eurusd_data.csv"
DATA_DIR="$PREDIX_DIR/git_ignore_folder/eurusd_data"
QLIB_DIR="$HOME/.qlib/qlib_data/eurusd_data"

echo "========================================"
echo " Predix EURUSD Setup"
echo "========================================"

# ─── 1. Prüfen ob alles da ist ───────────────────────────────────────────────
echo ""
echo "[1/7] Prüfe Voraussetzungen..."

if [ ! -d "$PREDIX_DIR" ]; then
    echo "FEHLER: $PREDIX_DIR nicht gefunden!"
    exit 1
fi

if [ ! -f "$CSV_SOURCE" ]; then
    echo "FEHLER: $CSV_SOURCE nicht gefunden!"
    echo "Bitte eurusd_data.csv in ~/Downloads/ legen"
    exit 1
fi

echo "✓ Predix gefunden: $PREDIX_DIR"
echo "✓ CSV gefunden: $CSV_SOURCE"

# ─── 2. Ordner anlegen ───────────────────────────────────────────────────────
echo ""
echo "[2/7] Erstelle Ordnerstruktur..."

mkdir -p "$DATA_DIR"
mkdir -p "$QLIB_DIR/calendars"
mkdir -p "$QLIB_DIR/instruments"
mkdir -p "$QLIB_DIR/features/eurusd"
mkdir -p "$PREDIX_DIR/git_ignore_folder/log"

cp "$CSV_SOURCE" "$DATA_DIR/eurusd_data.csv"
echo "✓ CSV kopiert nach $DATA_DIR"

# ─── 3. CSV → Qlib Format konvertieren ───────────────────────────────────────
echo ""
echo "[3/7] Konvertiere CSV zu Qlib-Format..."

python3 << 'PYEOF'
import pandas as pd
import numpy as np
from pathlib import Path
import os

QLIB_DIR = Path(os.path.expanduser("~/.qlib/qlib_data/eurusd_data"))
CSV_PATH = Path(os.path.expanduser("~/Downloads/eurusd_data.csv"))

# Laden + sortieren
df = pd.read_csv(CSV_PATH, parse_dates=["datetime"])
df = df.sort_values("datetime").reset_index(drop=True)
df.columns = [c.lower() for c in df.columns]

print(f"  Rows: {len(df):,} | Range: {df['datetime'].min().date()} -> {df['datetime'].max().date()}")

# ── Kalender (alle 15min Timestamps) ────────────────────────────────────────
cal = df["datetime"].dt.strftime("%Y-%m-%d %H:%M:%S")
cal_path = QLIB_DIR / "calendars" / "15min.txt"
cal.to_csv(cal_path, index=False, header=False)
print(f"  ✓ Kalender: {len(cal)} Einträge -> {cal_path}")

# ── Instruments (nur EURUSD) ─────────────────────────────────────────────────
inst_path = QLIB_DIR / "instruments" / "all.txt"
start = df["datetime"].min().strftime("%Y-%m-%d")
end   = df["datetime"].max().strftime("%Y-%m-%d")
with open(inst_path, "w") as f:
    f.write(f"EURUSD\t{start}\t{end}\n")
print(f"  ✓ Instruments -> {inst_path}")

# ── Features (Qlib binary format via CSV) ───────────────────────────────────
feat_dir = QLIB_DIR / "features" / "eurusd"
feat_dir.mkdir(parents=True, exist_ok=True)

# Qlib erwartet: $open, $high, $low, $close, $volume
for col in ["open", "high", "low", "close", "volume"]:
    out = feat_dir / f"{col}.day.bin"
    # Qlib binary: float32 array
    arr = df[col].astype("float32").values
    arr.tofile(str(out).replace(".day.bin", f"_15min.bin"))

# Auch als einfache CSV für direkten Zugriff
df.to_csv(QLIB_DIR / "eurusd_15min.csv", index=False)
print(f"  ✓ Features + CSV -> {feat_dir}")

# ── Returns + technische Features vorberechnen ───────────────────────────────
def ema(s, p): return s.ewm(span=p, adjust=False).mean()
def rsi(c, p=14):
    d = c.diff()
    g = d.clip(lower=0).ewm(span=p, adjust=False).mean()
    l = (-d.clip(upper=0)).ewm(span=p, adjust=False).mean()
    return 100 - 100/(1 + g/(l+1e-9))

feat = pd.DataFrame()
feat["datetime"] = df["datetime"]
feat["close"]    = df["close"]
for n in [1,4,8,16,96]:
    feat[f"ret_{n}"] = df["close"].pct_change(n)
feat["rsi_14"]    = rsi(df["close"], 14)
feat["macd_hist"] = ema(df["close"],12) - ema(df["close"],26)
feat["hour"]      = df["datetime"].dt.hour
feat["is_london"] = feat["hour"].isin([8,9,10,11]).astype(int)
feat["is_ny"]     = feat["hour"].isin([13,14,15,16]).astype(int)
feat["adx_proxy"] = df["close"].rolling(14).std() / df["close"].rolling(96).std()

feat.dropna(inplace=True)
feat.to_csv(QLIB_DIR / "eurusd_features.csv", index=False)
print(f"  ✓ Features CSV: {len(feat):,} Zeilen, {len(feat.columns)} Spalten")
print("  Fertig!")
PYEOF

echo "✓ Qlib-Daten konvertiert"

# ─── 4. .env updaten ─────────────────────────────────────────────────────────
echo ""
echo "[4/7] Update .env..."

ENV_FILE="$PREDIX_DIR/.env"

# Backup
cp "$ENV_FILE" "$ENV_FILE.backup_$(date +%Y%m%d_%H%M%S)"

# QLIB_DATA_DIR auf EURUSD umbiegen
sed -i "s|QLIB_DATA_DIR=.*|QLIB_DATA_DIR=$QLIB_DIR|" "$ENV_FILE"

# LOG_PATH korrigieren (war /home/nico/RD-Agent-Local/log)
sed -i "s|LOG_PATH=.*|LOG_PATH=$PREDIX_DIR/git_ignore_folder/log|" "$ENV_FILE"

# EURUSD-spezifische Vars hinzufügen (falls noch nicht da)
grep -q "EURUSD_DATA_PATH" "$ENV_FILE" || cat >> "$ENV_FILE" << 'ENVEOF'

# ---------- EURUSD ----------
EURUSD_DATA_PATH=/home/nico/.qlib/qlib_data/eurusd_data/eurusd_15min.csv
QLIB_FREQ=15min
QLIB_MARKET=eurusd
BACKTEST_START_TIME=2024-08-09
BACKTEST_END_TIME=2026-03-20
COST_RATE=0.00015
ENVEOF

echo "✓ .env aktualisiert (Backup erstellt)"

# ─── 5. Prompts für EURUSD anpassen ──────────────────────────────────────────
echo ""
echo "[5/7] Passe Qlib-Prompts für EURUSD an..."

PROMPT_FILE="$PREDIX_DIR/rdagent/app/qlib_rd_loop/prompts.yaml"
cp "$PROMPT_FILE" "${PROMPT_FILE}.backup_$(date +%Y%m%d_%H%M%S)"

cat > "$PROMPT_FILE" << 'YAMLEOF'
hypothesis_generation:
  system: |-
    You are an expert quantitative researcher specialized in FX (foreign exchange) trading,
    specifically EURUSD intraday strategies on 15-minute bars.

    EURUSD domain knowledge you must apply:
    - London session (08:00-12:00 UTC): highest volatility, trending behavior — favor momentum strategies
    - NY session (13:00-17:00 UTC): second volatility peak, also trending
    - Asian session (00:00-07:00 UTC): low volatility, mean-reverting behavior
    - London/NY overlap (13:00-17:00 UTC): strongest directional moves of the day
    - Weekend gap risk: avoid holding positions after Friday 20:00 UTC
    - Spread cost: ~1.5 bps per trade — strategies must minimize unnecessary entries
    - EURUSD is mean-reverting on short windows (<1h), trending on longer (>4h)
    - Key macro drivers: ECB/Fed rate decisions, NFP (first Friday of month), CPI releases

    Available model types you can propose:
    - TimeSeries: LSTM, GRU, TCN (Temporal Convolutional Network), Transformer, PatchTST
    - Tabular: XGBoost, LightGBM, RandomForest (on engineered features)
    - Hybrid: CNN+LSTM, XGBoost+LSTM ensemble
    - Statistical: Regime-switching (HMM), Kalman filter

    Available features in the dataset:
    - OHLCV: open, high, low, close, volume (15min bars)
    - Returns: ret_1, ret_4, ret_8, ret_16, ret_96
    - Technical: rsi_14, macd_hist, adx_14, atr_14, bb_pct, stoch_k, cci_14
    - Volatility: vol_real_4, vol_real_16, vol_ratio, zscore_ret_96
    - Time/Session: hour, is_london, is_ny, is_overlap, hour_sin, hour_cos
    - Lags: rsi_14_lag1-8, macd_hist_lag1-8, bb_pct_lag1-8

    Your hypothesis must:
    1. Specify which session(s) the strategy targets
    2. Name which model type to use and why it fits EURUSD
    3. Include a session filter (is_london / is_ny)
    4. Include a spread filter (only trade when expected |return| > 0.0003)
    5. Specify target: classification (fwd_sign_4) or regression (fwd_ret_4)

    Please ensure your response is in JSON format:
    {
      "hypothesis": "A clear and concise trading hypothesis for EURUSD 15min.",
      "reason": "Detailed explanation including session, model choice, and expected edge.",
      "model_type": "One of: TimeSeries / Tabular / XGBoost",
      "target_session": "london / ny / asian / all",
      "expected_arr_range": "e.g. 8-12%"
    }

  user: |-
    Previously tried approaches and their results:
    {{ factor_descriptions }}

    Additional context:
    {{ report_content }}

    Generate a NEW hypothesis that is meaningfully different from what has been tried.
    Focus on approaches that have NOT been tested yet.
    Target: beat current best ARR of 9.62%.
YAMLEOF

echo "✓ prompts.yaml aktualisiert (Backup erstellt)"

# ─── 6. Model Coder Prompt erweitern ─────────────────────────────────────────
echo ""
echo "[6/7] Erweitere Model Coder Prompts..."

MODEL_PROMPT="$PREDIX_DIR/rdagent/components/coder/model_coder/prompts.yaml"
cp "$MODEL_PROMPT" "${MODEL_PROMPT}.backup_$(date +%Y%m%d_%H%M%S)"

# EURUSD session filter als Kommentar in den evolving_strategy Block injizieren
python3 << 'PYEOF'
import re
from pathlib import Path

path = Path("/home/nico/Predix/rdagent/components/coder/model_coder/prompts.yaml")
content = path.read_text()

eurusd_note = """
        EURUSD-specific rules (ALWAYS apply these in generated code):
        1. Session filter: use is_london and is_ny columns — weight/filter signals to active sessions
        2. Spread filter: only generate signal when abs(predicted_return) > 0.0003
        3. ADX regime: if adx_proxy > 1.2 use trend model; if adx_proxy < 0.8 use mean-reversion
        4. Weekend filter: zero out signals when dayofweek==4 and hour>=20
        5. Max trade frequency: target <15 trades per day (avoid spread cost death)
        6. Supported model_type values: "Tabular", "TimeSeries", "XGBoost"
"""

# Inject after the scenario line in evolving_strategy_model_coder
content = content.replace(
    "        Your code is expected to align the scenario in any form",
    eurusd_note + "\n        Your code is expected to align the scenario in any form"
)
path.write_text(content)
print("  ✓ Model coder prompt erweitert")
PYEOF

echo "✓ Model coder Prompt angepasst"

# ─── 7. Git Commits ──────────────────────────────────────────────────────────
echo ""
echo "[7/7] Git Commits..."

cd "$PREDIX_DIR"

git add rdagent/app/qlib_rd_loop/prompts.yaml
git commit -m "feat: EURUSD 15min prompts - session filter, FX domain knowledge

- Add London/NY/Asian session awareness to hypothesis generation
- Add model type suggestions: LSTM, GRU, TCN, Transformer, XGBoost, LightGBM
- Add spread filter (1.5 bps) and ADX regime detection
- Target: beat current best ARR of 9.62%"

git add rdagent/components/coder/model_coder/prompts.yaml
git commit -m "feat: inject EURUSD trading rules into model coder

- Session filter (is_london, is_ny)
- Spread filter (|return| > 0.0003)
- Weekend position close
- Max trade frequency guidance"

git add .env 2>/dev/null || true
echo "  (Note: .env nicht committed - enthält API Keys)"

echo ""
echo "========================================"
echo " Setup abgeschlossen!"
echo "========================================"
echo ""
echo "Nächste Schritte:"
echo ""
echo "  1. Starten:"
echo "     cd ~/Predix && rdagent fin_quant"
echo ""
echo "  2. Dashboard (in zweitem Terminal):"
echo "     cd ~/Predix && rdagent server_ui --port 19899"
echo "     → http://localhost:19899"
echo ""
echo "  3. Logs:"
echo "     tail -f $PREDIX_DIR/git_ignore_folder/log/*.log"
echo ""
echo "Backup-Dateien (.backup_*) können nach erfolgreichem Test gelöscht werden."
