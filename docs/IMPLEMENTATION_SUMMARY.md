# 🎯 PREDIX: Vollständige Integration in fin_quant Loop

## ✅ Implementierte Features

### 1. Realistisches Backtesting
- **Echte OHLCV-Daten** aus `intraday_pv.h5` (2.26M Bars, 2020-2026)
- **Forward-Fill** täglicher Faktoren auf 1-Min-Frequenz
- **Spread-Kosten**: 1.5 bps pro Trade
- **Korrekte Annualisierung**: sqrt(252*1440) für 1-Min-Daten

### 2. Verbesserter LLM-Prompt
- **IC-geführte Faktorwahl**: |IC| > 0.10 PRIORITIZE, |IC| > 0.05 USE
- **IC-gewichtete Kombinationen**: Höhere IC = höheres Gewicht
- **Bessere Beispiele** mit IC-Gewichten im Prompt
- **Verfügbarkeit von 'close' Series** für zusätzliche Berechnungen

### 3. Optuna-Optimierung
- **20 Trials pro Strategie** (konfigurierbar)
- **TPESampler** mit MedianPruner
- **Optimiert**: entry_threshold, rolling_window, SL, TP, Trailing Stop
- **Auto-Update** wenn Optuna Sharpe verbessert

### 4. Automatische Strategiegenerierung
- **Trigger**: Alle 500 Faktoren (konfigurierbar)
- **3 Strategien pro Zyklus** mit zufälligen Faktor-Kombinationen
- **Graceful Degradation**: Bricht Hauptloop nicht bei Fehlern

## 🚀 Benutzung

### Automatisch (im fin_quant Loop)
```bash
# Standard: Alle 500 Faktoren
rdagent fin_quant --auto-strategies

# Custom threshold
rdagent fin_quant --auto-strategies --auto-strategies-threshold 1000

# Mit OpenRouter
rdagent fin_quant -m openrouter --auto-strategies
```

### Manuell
```bash
# 5 Strategien mit Optuna
rdagent generate_strategies --count 5 --optuna --optuna-trials 20

# Ohne Optuna (schneller)
rdagent generate_strategies --count 5 --no-optuna
```

## 📊 Testergebnisse

### MomentumDivergenceZScore (vorher vs. nachher)

| Metrik | Vorher | Nachher |
|--------|--------|---------|
| **Datenpunkte** | 259 (4.3h) | 823,450 (2.27 Jahre) |
| **Sharpe** | 3.59 | 6.04 |
| **Max DD** | -0.22% | -1.57% |
| **Win Rate** | 49.46% | 49.19% |
| **Ann Return** | 543% (falsch) | 21.88% ✅ |

## 🔧 Architecture

```
fin_quant Loop
    │
    ├─ Factor Generation (LLM → Docker → Evaluation)
    │   └─ Every 500 factors → Trigger Strategy Generation
    │
    └─ StrategyOrchestrator (auto-strategies)
        │
        ├─ Load Top 50 Factors (by IC)
        ├─ For each strategy (3x):
        │   ├─ Select random 2-5 factors
        │   ├─ LLM generates code (improved prompt)
        │   ├─ Evaluate with real OHLCV
        │   ├─ Optuna optimize (20 trials)
        │   └─ Save if accepted
        │
        └─ Log results
```

## 📝 Nächste Schritte

1. **Live Trading**: Bestehende Strategien für Paper Trading nutzen
2. **Mehr Faktoren**: Weiterhin Faktoren generieren für bessere Strategien
3. **Dashboard**: Live-Statistiken im Web/CLI Dashboard anzeigen

## ⚠️ Wichtige Hinweise

- **Forward-Fill** kann zu Daten-Leakage führen (tägliche Werte werden auf Minuten aufgefüllt)
- **Optuna** benötigt 20-30 Sekunden pro Strategie
- **Auto-Strategies** nur wenn ≥10 Faktoren verfügbar
- **LLM** muss verfügbar sein (local oder openrouter)
