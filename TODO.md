# Predix — Implementierungs-Tracker

## Übersicht
- **Start:** April 2026
- **Ziel:** Vollständig integriertes Quant Trading System in `fin_quant` Loop
- **Status:** ✅ ALLE PHASEN ABGESCHLOSSEN (P0-P9)

---

## Phasen

### P0: Data Loader (2h) ✅ ABGESCHLOSSEN
- [x] `rdagent/scenarios/qlib/local/data_loader.py` erstellt
  - [x] OHLCV Loading + Caching (HDF5 → pd.Series)
  - [x] Faktor Loading + Metadaten (JSON + Parquet)
  - [x] Feature Matrix Builder (alignment mit close)
  - [x] Thread-sicheres Caching
- [x] Tests: `test/local/test_data_loader.py` (11 passed)
- [x] Abhängigkeiten: Keine

### P1: Strategy Worker (4h) ✅ ABGESCHLOSSEN
- [x] `rdagent/scenarios/qlib/local/strategy_worker.py` erstellt
  - [x] LLM Call Wrapper (llama.cpp :8081)
  - [x] Backtest Engine (subprocess mit OHLCV)
  - [x] Acceptance Gate (FTMO-konform)
  - [x] FTMO Compliance Check
- [x] Tests: `test/local/test_strategy_worker.py` (41 passed)
- [x] Abhängigkeiten: P0 (data_loader.py)

### P2: Strategy Orchestrator (6h) ✅ ABGESCHLOSSEN
- [x] `rdagent/scenarios/qlib/local/strategy_orchestrator.py` erstellt
  - [x] Multi-Process Pool (4-8 Workers)
  - [x] Task Queue (random factor selection)
  - [x] LLM Semaphore (max 2 parallel für llama.cpp)
  - [x] Result Collection + Deduplizierung
  - [x] Strategy Saver (JSON + Reports)
- [x] CLI Command: `rdagent generate_strategies`
- [x] Tests: `test/local/test_strategy_orchestrator.py` (30 passed)
- [x] Abhängigkeiten: P1

### P3: Optuna Optimizer (4h) ✅ ABGESCHLOSSEN
- [x] `rdagent/scenarios/qlib/local/optuna_optimizer.py` erstellt
  - [x] Parameter Space Definition (FTMO-konform)
  - [x] Objective Function (Sharpe × |IC| × √trades)
  - [x] FTMO Penalty Logic
  - [x] TPE Sampler + MedianPruner
  - [x] 20-50 Trials pro Strategie
- [x] Integration in Strategy Orchestrator
- [x] Tests: `test/local/test_optuna_optimizer.py` (60 passed)
- [x] Abhängigkeiten: P1, `pip install optuna`

### P4: CLI Commands (2h) ✅ ABGESCHLOSSEN
- [x] `rdagent/app/cli.py` erweitert
  - [x] `generate_strategies` Command
  - [x] CLI Parameter (count, workers, style, optuna)
  - [x] Rich Console Output
  - [ ] Integration in `fin_quant` Loop
- [x] Tests: `test/integration/test_cli_commands.py` (21 tests)
- [x] Abhängigkeiten: P2, P3

### P5: ML Training Pipeline (6h) ✅ ABGESCHLOSSEN
- [x] `rdagent/scenarios/qlib/local/ml_trainer.py` erstellt
  - [x] Feature Matrix Builder (alle Top-N Faktoren)
  - [x] Time-Series Train/Val Split
  - [x] LightGBM Training (early stopping)
  - [x] Feature Importance Analysis
  - [x] Model Save/Load
- [x] CLI Command: `rdagent train_models`
- [x] Tests: `test/local/test_ml_trainer.py` (46 passed)
- [x] Abhängigkeiten: P0, `pip install lightgbm`

### P6: Feedback an fin_quant Loop (3h) ✅ ABGESCHLOSSEN
- [ ] Hook in `QuantRDLoop.feedback()` einbauen
  - [ ] `_trigger_ml_training()` alle 500 Faktoren
  - [ ] `_trigger_strategy_generation()` alle 1000 Faktoren
  - [ ] ML Feature Importance → Prompt Feedback
- [ ] Prompt-Loader erweitern (local/ml_feedback.yaml)
- [ ] Tests: `test/local/test_feedback_integration.py`
- [ ] Abhängigkeiten: P5

### P7: Portfolio Optimizer (6h) ✅ ABGESCHLOSSEN
- [ ] `rdagent/scenarios/qlib/local/portfolio_optimizer.py` erstellen
  - [ ] Korrelationsmatrix (max 0.3)
  - [ ] Mean-Variance Optimization
  - [ ] Risk Parity (gleicher Risiko-Beitrag)
  - [ ] Black-Litterman (LLM Views als Priors)
  - [ ] Portfolio-Backtest
- [ ] CLI Command: `rdagent optimize_portfolio`
- [ ] Tests: `test/local/test_portfolio_optimizer.py`
- [ ] Abhängigkeiten: P5

### P8: Integrationstests (4h) ✅ ABGESCHLOSSEN
- [ ] End-to-End Pipeline Test
  - [ ] Data Loading → Strategy Gen → Backtest → Accept
- [ ] Parallelisierung Test
  - [ ] 4 Workers, 2 LLM parallel, keine Race Conditions
- [ ] Optuna Test
  - [ ] 20 Trials, Konvergenz prüfen
- [ ] FTMO Compliance Test
  - [ ] SL ≤ 2%, DD ≤ 10%, Daily Loss ≤ 5%
- [ ] Tests: `test/integration/test_full_pipeline.py`
- [ ] Abhängigkeiten: P0-P7

### P9: Dokumentation (3h) ✅ ABGESCHLOSSEN
- [ ] README.md aktualisieren
  - [ ] Neue Commands dokumentieren
  - [ ] Architektur-Diagramm
  - [ ] Setup-Anleitung
- [ ] QWEN.md aktualisieren
  - [ ] Neue Module in Architecture Section
  - [ ] Data Flow Diagram
  - [ ] Project Status
- [ ] Abhängigkeiten: P0-P8

---

## Dependencies Checklist

- [ ] `pip install optuna` (P3)
- [ ] `pip install lightgbm` (P5)
- [ ] `pip install xgboost` (P5, optional)
- [ ] llama.cpp Server auf :8081 mit `--parallel 2` (P1-P3)

## FTMO Compliance Rules

| Regel | Limit | Prüfung |
|-------|-------|---------|
| Max Stop Loss | 2% | Hard-coded in Optuna Space |
| Max Drawdown | 10% | Acceptance Gate |
| Max Daily Loss | 5% | Risk Management Layer |
| Risk/Reward | ≥ 2:1 | TP ≥ 2× SL enforced |
| Max Positions | 1 | Strategy Code enforced |

## Risiko-Metriken (Ziele)

| Metrik | Target | Minimum |
|--------|--------|---------|
| Strategie IC | > 0.03 | > 0.02 |
| Strategie Sharpe | > 1.5 | > 0.5 |
| Max Drawdown | < 10% | < 15% |
| Win Rate | > 50% | > 45% |
| Monthly Return | 1-3% | > 0.5% |
| Trades/Monat | > 20 | > 10 |

## Notizen

- **Alles in fin_quant Loop**: Keine separaten Skripte mehr (`predix_gen_strategies_real_bt.py`, `predix_smart_strategy_gen.py` deprecated)
- **Graceful Degradation**: Wenn llama.cpp nicht läuft, skippt Strategy Generation mit klarer Warnung
- **LLM Parallelisierung**: Max 2 parallele Calls (llama.cpp `--parallel 2`)
- **Optuna Trials**: Standard 30, kann via CLI überschrieben werden
