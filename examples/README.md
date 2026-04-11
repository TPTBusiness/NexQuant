# PREDIX Examples

Willkommen zu den PREDIX Trading Platform Beispielen! Dieser Ordner enthält vollständi  ge, lauffä  hige Beispiele, die dir den Einstieg in algorithmisches Trading mit EUR/USD erleichtern.

## 📚 Beispiele im Überblick

| Nr. | Beispiel | Beschreibung | Dauer | Schwierigkeit |
|-----|----------|--------------|-------|---------------|
| 01 | [`factor_discovery.py`](01_factor_discovery.py) | Automatische Generierung neuer Trading-Faktoren | ~10 Min | ⭐ Anfänger |
| 02 | [`factor_evolution.py`](02_factor_evolution.py) | Optimierung bestehender Faktoren | ~15 Min | ⭐⭐ Mittel |
| 03 | [`strategy_generation.py`](03_strategy_generation.py) | Kombination von Faktoren zu Strategien | ~5 Min | ⭐ Anfänger |
| 04 | [`backtest_simple.py`](04_backtest_simple.py) | Backtest einer Trading-Strategie | ~3 Min | ⭐ Anfänger |
| 05 | [`model_training.py`](05_model_training.py) | ML-Modell-Training (LSTM/XGBoost) | ~30 Min | ⭐⭐⭐ Fortgeschritten |
| 06 | [`rl_trading_agent.py`](06_rl_trading_agent.py) | Reinforcement Learning Agent | ~60 Min | ⭐⭐⭐ Fortgeschritten |

## 🚀 Schnellstart

### Voraussetzungen

```bash
# Installation
pip install -e ".[all]"

# Daten herunterladen (falls noch nicht geschehen)
rdagent download-data
```

### Beispiel ausführen

```bash
# Faktor-Generierung (3 Loops)
python examples/01_factor_discovery.py --loop-n 3

# Backtest durchführen
python examples/04_backtest_simple.py --strategy momentum
```

## 📖 Detaillierte Anleitungen

### Beispiel 01: Factor Discovery

**Ziel:** Automatisch neue Trading-Faktoren mit LLM generieren lassen

```bash
python examples/01_factor_discovery.py --loop-n 5 --llm local
```

**Output:**
- Generierte Faktoren in `RD-Agent_workspace/`
- Performance-Metriken (ARR, Sharpe, IC)
- Faktor-Implementierungen als Python-Code

**Nächste Schritte:**
→ Siehe `02_factor_evolution.py` um Faktoren zu optimieren

### Beispiel 02: Factor Evolution

**Ziel:** Bestehende Faktoren mit Session/Regime Filters verbessern

```bash
python examples/02_factor_evolution.py --factor momentum_16 --improve session_filter
```

**Output:**
- Verbesserte Faktoren mit Before/After-Vergleich
- Metrik-Verbesserungen (ARR +X%, Sharpe +X.X)

### Beispiel 03: Strategy Generation

**Ziel:** Mehrere Faktoren zu einer robusten Strategie kombinieren

```bash
python examples/03_strategy_generation.py --factors momentum_16,reversal,session_alpha
```

**Output:**
- IC-weighted Faktor-Kombination
- Signal-Verteilung (Long/Short/Neutral)

### Beispiel 04: Backtest

**Ziel:** Backtest einer Trading-Strategie auf historischen Daten

```bash
python examples/04_backtest_simple.py --strategy momentum --start 2020-01-01 --end 2025-12-31
```

**Output:**
- Key-Metriken: ARR, Sharpe, MaxDD, WinRate
- Equity Curve (optional als Plot)

### Beispiel 05: Model Training

**Ziel:** ML-Modell (LSTM/XGBoost) auf Faktor-Daten trainieren

```bash
python examples/05_model_training.py --model lstm --features momentum_16,reversal
```

**Output:**
- Trainiertes Modell in `models/`
- Train/Val/Test Split Ergebnisse
- Feature Importance (bei XGBoost)

### Beispiel 06: RL Trading Agent

**Ziel:** Reinforcement Learning Agent für Trading trainieren

```bash
python examples/06_rl_trading_agent.py --algo ppo --episodes 1000
```

**Output:**
- Trainierter RL-Agent in `models/rl_agent/`
- Learning Curve
- Trading-Statistiken

## 📓 Jupyter Notebook

Für eine interaktive Einführung siehe:

```bash
jupyter notebook examples/notebooks/quickstart.ipynb
```

## 🐛 Probleme?

- **Dokumentation:** `docs/` oder [README.md](../README.md)
- **CLI Hilfe:** `rdagent COMMAND --help`
- **Issues:** [GitHub Issues](https://github.com/nico/Predix/issues)
- **Community:** [Discussions](https://github.com/nico/Predix/discussions)

## ⚠️ Wichtige Hinweise

- **Keine Closed-Source Assets:** Commite niemals `git_ignore_folder/`, `results/`, `.env`, `models/local/`, `prompts/local/`
- **Daten-Pfade:** Passe ggf. Datenpfade in den Beispielen an deine Installation an
- **Laufzeit:** ML/RL-Beispiele benötigen ggf. GPU für akzeptable Laufzeiten
