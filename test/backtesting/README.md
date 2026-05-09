# PREDIX Backtesting Tests

Umfassende Test-Suite für das PREDIX Backtesting-Modul.

## Verzeichnisstruktur

```
test/backtesting/
├── __init__.py              # Package-Initialisierung
├── conftest.py              # Pytest Fixtures und Test-Daten
├── test_backtest_engine.py  # Tests für BacktestMetrics & FactorBacktester
├── test_results_db.py       # Tests für ResultsDatabase (SQLite)
└── test_risk_management.py  # Tests für Risk Management Komponenten
```

## Voraussetzungen

```bash
pip install pytest pytest-cov
```

Die Pakete sind in `requirements.txt` enthalten.

## Tests ausführen

### Alle Tests ausführen

```bash
cd /home/nico/NexQuant
pytest test/backtesting/
```

### Tests mit Coverage-Bericht

```bash
pytest test/backtesting/ --cov=rdagent/components/backtesting --cov-report=term-missing
```

### HTML Coverage-Bericht generieren

```bash
pytest test/backtesting/ --cov=rdagent/components/backtesting --cov-report=html
# Öffne htmlcov/index.html im Browser
```

### Spezifische Test-Datei ausführen

```bash
# Nur Backtest Engine Tests
pytest test/backtesting/test_backtest_engine.py -v

# Nur Database Tests
pytest test/backtesting/test_results_db.py -v

# Nur Risk Management Tests
pytest test/backtesting/test_risk_management.py -v
```

### Spezifischen Test ausführen

```bash
# Einzelnen Test nach Name
pytest test/backtesting/test_backtest_engine.py::TestBacktestMetricsCalculateIC::test_calculate_ic_normal_data -v

# Alle Tests einer Klasse
pytest test/backtesting/test_backtest_engine.py::TestBacktestMetricsCalculateIC -v
```

### Tests mit Filtern

```bash
# Nur Unit Tests (wenn markiert)
pytest -m unit

# Langsame Tests überspringen
pytest -m "not slow"

# Tests mit bestimmtem Keyword
pytest -k "ic"  # Alle Tests mit "ic" im Namen
```

## Test-Abdeckung (Coverage)

Das Ziel ist **>80% Code-Coverage** für alle Backtesting-Komponenten.

### Coverage-Ziele pro Modul

| Modul | Ziel-Coverage |
|-------|---------------|
| backtest_engine.py | >80% |
| results_db.py | >80% |
| risk_management.py | >80% |

### Coverage-Berichte

**Terminal-Bericht:**
```bash
pytest --cov=rdagent/components/backtesting --cov-report=term-missing
```

**HTML-Bericht:**
```bash
pytest --cov=rdagent/components/backtesting --cov-report=html
# Öffne: htmlcov/index.html
```

**XML-Bericht (für CI/CD):**
```bash
pytest --cov=rdagent/components/backtesting --cov-report=xml
# Output: coverage.xml
```

## Getestete Komponenten

### 1. BacktestMetrics (`test_backtest_engine.py`)

| Methode | Test-Fälle |
|---------|------------|
| `calculate_ic()` | Normale Daten, perfekte Korrelation, leere Daten, NaN, insufficient data |
| `calculate_sharpe()` | Normale Daten, annualisiert vs. raw, leere Daten, zero variance |
| `calculate_max_drawdown()` | Normale Daten, monotonic increasing, significant drop, empty |
| `calculate_all()` | Complete metrics, without factor data, total return, win rate |

### 2. FactorBacktester (`test_backtest_engine.py`)

| Methode | Test-Fälle |
|---------|------------|
| `run_backtest()` | Complete output, JSON save, transaction costs, NaN values, empty data |

### 3. ResultsDatabase (`test_results_db.py`)

| Methode | Test-Fälle |
|---------|------------|
| `__init__()` | Default path, creates tables, parent directories, multiple instances |
| `add_factor()` | New factor, duplicate, special characters, empty name, many factors |
| `add_backtest()` | Basic, creates factor, missing metrics, NaN values, multiple runs |
| `add_loop()` | Basic, success rate calculation, zero total, multiple loops |
| `get_top_factors()` | By sharpe, by ic, limit, empty db, all columns |
| `get_aggregate_stats()` | Populated, empty, after additions |

### 4. CorrelationAnalyzer (`test_risk_management.py`)

| Methode | Test-Fälle |
|---------|------------|
| `calculate_matrix()` | Normal data, perfect correlation, empty data, NaN, single asset |
| `find_uncorrelated()` | Identifies uncorrelated, all correlated, custom threshold, empty |

### 5. PortfolioOptimizer (`test_risk_management.py`)

| Methode | Test-Fälle |
|---------|------------|
| `mean_variance()` | Basic, higher expected return, singular covariance, zero covariance |
| `risk_parity()` | Basic, equal volatility, different volatility, convergence, single asset |

### 6. AdvancedRiskManager (`test_risk_management.py`)

| Methode | Test-Fälle |
|---------|------------|
| `check_limits()` | All pass, position exceeded, leverage exceeded, drawdown exceeded, boundary |

## Fixtures (conftest.py)

Wiederverwendbare Test-Fixtures:

| Fixture | Beschreibung |
|---------|--------------|
| `sample_factor_data` | Normale Faktor-Daten (252 Tage) |
| `sample_returns_data` | Returns und Equity-Daten |
| `backtest_metrics` | BacktestMetrics Instanz |
| `empty_data` | Leere Daten für Edge-Cases |
| `nan_data` | Daten mit vielen NaN-Werten |
| `insufficient_data` | Zu wenig Daten (<10 Punkte) |
| `extreme_values_data` | Daten mit Extremwerten |
| `constant_data` | Konstante Daten (Std=0) |
| `temp_db_path` | Temporäre Datenbank-Pfad |
| `results_database` | ResultsDatabase mit temp DB |
| `populated_database` | Befüllte ResultsDatabase |
| `sample_returns_matrix` | Returns-Matrix für Korrelation |
| `correlation_analyzer` | CorrelationAnalyzer Instanz |
| `portfolio_optimizer` | PortfolioOptimizer Instanz |
| `sample_expected_returns` | Erwartete Returns |
| `sample_covariance_matrix` | Kovarianz-Matrix |
| `risk_manager` | AdvancedRiskManager Instanz |
| `sample_weights` | Test-Gewichtungen |
| `factor_backtester` | FactorBacktester Instanz |
| `realistic_market_data` | Realistischere Markt-Daten |
| `zero_variance_returns` | Returns mit Varianz=0 |
| `negative_equity_data` | Equity mit Drawdowns |

## Edge Cases

Die Tests decken folgende Edge Cases ab:

- **Leere Daten**: Empty Series, DataFrames
- **NaN-Werte**: Teilweise oder komplett NaN
- **Zu wenig Daten**: Weniger als 10 Datenpunkte
- **Extremwerte**: Sehr große/kleine Zahlen
- **Konstante Daten**: Varianz = 0
- **Singuläre Matrizen**: Nicht invertierbare Kovarianz
- **Grenzwerte**: Genau an den Limits
- **Negative Werte**: Negative Returns, Gewichte, Drawdowns

## CI/CD Integration

Für GitHub Actions oder andere CI/CD-Systeme:

```yaml
# Beispiel GitHub Actions
- name: Run Tests
  run: |
    pip install -r requirements.txt
    pytest test/backtesting/ --cov=rdagent/components/backtesting --cov-report=xml --cov-fail-under=80
```

## Qualitätsstandards

- ✅ Jeder Test hat eine klare Assertion
- ✅ Test-Namen beschreiben das getestete Verhalten
- ✅ Tests sind unabhängig und reproduzierbar
- ✅ Externe Dependencies werden gemockt wo angemessen
- ✅ Keine Tests werden übersprungen

## Fehlerbehebung

### Tests schlagen fehl wegen Import-Fehlern

```bash
# Stelle sicher dass du im Projekt-Verzeichnis bist
cd /home/nico/NexQuant
export PYTHONPATH=/home/nico/NexQuant:$PYTHONPATH
pytest test/backtesting/
```

### Coverage ist zu niedrig

```bash
# Siehe welche Zeilen nicht getestet sind
pytest --cov=rdagent/components/backtesting --cov-report=term-missing

# Öffne HTML-Bericht für detaillierte Analyse
pytest --cov=rdagent/components/backtesting --cov-report=html
# Öffne htmlcov/index.html
```

### Datenbank-Tests schlagen fehl

```bash
# Temporäre Dateien bereinigen
rm -rf /tmp/test_*.db
pytest test/backtesting/test_results_db.py
```

## Kontakt & Support

Bei Fragen oder Problemen mit den Tests:
- Siehe die Test-Dateien für Beispiele
- Prüfe die Fixture-Definitionen in conftest.py
- Konsultiere die pytest-Dokumentation: https://docs.pytest.org/
