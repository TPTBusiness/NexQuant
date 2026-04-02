# Predix Results Documentation

Dieser Ordner enthält alle Backtesting-Ergebnisse, Faktor-Analysen und Performance-Daten.

## ⚠️ WICHTIG

**Dieser Ordner ist in `.gitignore` aufgenommen!**

- Ergebnisse werden **NICHT** zu Git hinzugefügt
- Jeder Entwickler hat lokale Ergebnisse
- Sensible Performance-Daten bleiben privat

---

## 📁 Ordner-Struktur

```
results/
├── backtests/          # Einzelne Backtest-Ergebnisse (JSON, CSV)
│   ├── FactorName_20240402_120000.json
│   ├── FactorName_20240402_120000_returns.csv
│   └── FactorName_20240402_120000_equity.csv
│
├── factors/            # Faktor-spezifische Analysen
│   ├── factor_performance.json
│   └── ic_history.csv
│
├── runs/               # Komplette Run-Ergebnisse
│   ├── risk_report_20240402_120000.json
│   └── portfolio_weights_20240402.json
│
├── logs/               # Backtesting-Logs
│   └── backtest_20240402.log
│
└── db/                 # SQLite-Datenbank
    ├── backtest_results.db
    └── test_export.json
```

---

## 📊 Gespeicherte Daten

### Backtests (`backtests/`)

Für jeden Faktor werden gespeichert:

| Datei | Inhalt |
|-------|--------|
| `{Factor}_{Timestamp}.json` | Alle Metriken (IC, Sharpe, Drawdown, etc.) |
| `{Factor}_{Timestamp}_returns.csv` | Tägliche Returns |
| `{Factor}_{Timestamp}_equity.csv` | Equity Curve |

**Metriken pro Faktor:**
- IC (Information Coefficient)
- ICIR (IC Information Ratio)
- Sharpe Ratio
- Sortino Ratio
- Calmar Ratio
- Annualized Return
- Max Drawdown
- Win Rate
- Total Trades

---

### Datenbank (`db/backtest_results.db`)

**Tabellen:**

| Tabelle | Inhalt |
|---------|--------|
| `factors` | Alle generierten Faktoren |
| `backtest_runs` | Backtest-Durchläufe |
| `backtest_metrics` | Performance-Metriken pro Run |
| `daily_returns` | Tägliche Returns pro Run |
| `loop_results` | Loop-Zusammenfassungen |
| `factor_correlations` | Korrelationen zwischen Faktoren |

**Abfragen:**

```python
from rdagent.components.backtesting import ResultsDatabase

db = ResultsDatabase()

# Top 20 Faktoren nach Sharpe Ratio
top_factors = db.get_top_factors('sharpe_ratio', limit=20)

# Performance-Historie für Faktor
perf = db.get_factor_performance('Momentum_8Bar')

# Loop-Zusammenfassung
loops = db.get_loop_summary()

# Aggregierte Statistiken
stats = db.get_aggregate_stats()
```

---

### Risk Reports (`runs/`)

**Inhalt:**
- Portfolio-Volatilität
- Sharpe Ratio
- Diversifikations-Ratio
- Max Drawdown
- Limit-Checks (Position Size, Leverage, Drawdown)
- Korrelationsmatrix

---

## 🔧 Verwendung

### 1. Backtest durchführen

```python
from rdagent.components.backtesting import FactorBacktester, ResultsDatabase

# Backtester initialisieren
backtester = FactorBacktester()
db = ResultsDatabase()

# Faktor-Daten laden
factor_values = pd.Series(...)  # Faktorwerte
forward_returns = pd.Series(...)  # Forward Returns

# Backtest durchführen
metrics = backtester.run_backtest(
    factor_values=factor_values,
    forward_returns=forward_returns,
    factor_name="MyFactor"
)

# In Datenbank speichern
db.add_backtest_run(
    factor_name="MyFactor",
    metrics=metrics,
    returns=...,
    equity_curve=...
)
```

### 2. Portfolio-Optimierung

```python
from rdagent.components.backtesting import PortfolioOptimizer, CorrelationAnalyzer

# Korrelationsmatrix
corr_analyzer = CorrelationAnalyzer()
corr_matrix = corr_analyzer.calculate_correlation_matrix(factor_returns)

# Optimierung
optimizer = PortfolioOptimizer()
weights = optimizer.mean_variance_optimization(
    expected_returns=expected_returns,
    cov_matrix=cov_matrix
)

# Speichern
optimizer.save_optimization_results(weights, factor_names, 'mean_variance')
```

### 3. Risiko-Bericht

```python
from rdagent.components.backtesting import AdvancedRiskManager

risk_manager = AdvancedRiskManager()

report = risk_manager.generate_risk_report(
    factor_returns=factor_returns,
    portfolio_weights=weights
)

print(f"Sharpe: {report['sharpe_ratio']:.2f}")
print(f"Alle Limits OK: {report['all_limits_ok']}")
```

---

## 📈 Export

### JSON Export

```python
db.export_to_json("results/db/full_export.json")
```

**Inhalt:**
- Aggregierte Statistiken
- Top-Faktoren
- Loop-Zusammenfassung
- Export-Datum

---

## 🎯 Ziel-Metriken

| Metrik | Ziel | Minimum |
|--------|------|---------|
| **IC** | > 0.05 | > 0.02 |
| **ICIR** | > 2.0 | > 1.0 |
| **Sharpe Ratio** | > 2.0 | > 1.0 |
| **Max Drawdown** | < 15% | < 25% |
| **Win Rate** | > 55% | > 45% |
| **Annualized Return** | > 10% | > 5% |

---

## 📝 Dokumentation

Jeder Backtest wird automatisch dokumentiert mit:
- Timestamp
- Faktor-Name
- Alle Metriken
- Returns & Equity Curve
- Konfigurierte Parameter (Transaction Costs, etc.)

**Manuelle Notizen:**
- Erstelle `results/logs/notes_YYYYMMDD.md` für manuelle Notizen
- Dokumentiere besondere Ereignisse (Markt-Crashes, etc.)

---

## 🔒 Datenschutz

- Ergebnisse sind **lokal** (nicht in Git)
- Datenbank ist **lokal** (SQLite)
- Bei Team-Nutzung: Ergebnisse manuell teilen oder zentrale DB verwenden

---

## 🚀 Nächste Schritte

1. **Backtesting für alle 110 Faktoren durchführen**
2. **Top-20 Faktoren nach IC/Sharpe auswählen**
3. **Portfolio-Optimierung durchführen**
4. **4 Wochen Paper-Trading**
5. **Live-Performance dokumentieren**

---

**Stand:** April 2026
**Version:** 1.0
