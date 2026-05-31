# NexQuant

<p align="center">
  <img src="https://img.shields.io/badge/Python-3.10%20|%203.11-blue?style=for-the-badge&logo=python" alt="Python">
  <img src="https://img.shields.io/badge/Platform-Linux-lightgrey?style=for-the-badge&logo=linux" alt="Platform">
  <img src="https://img.shields.io/badge/Numba-0.59+-00A3E0?style=for-the-badge&logo=numba" alt="Numba">
  <img src="https://img.shields.io/badge/Optuna-4.8+-009B77?style=for-the-badge&logo=optuna" alt="Optuna">
</p>

<p align="center">
  <img src="https://img.shields.io/badge/TA--Lib-0.6+-green?style=for-the-badge" alt="TA-Lib">
  <img src="https://img.shields.io/badge/LightGBM-4.6+-00A1E0?style=for-the-badge" alt="LightGBM">
  <img src="https://img.shields.io/badge/Pandas-2.0+-150458?style=for-the-badge&logo=pandas" alt="Pandas">
  <img src="https://img.shields.io/badge/cTrader-OpenAPI-FF6B6B?style=for-the-badge" alt="cTrader">
</p>

<h4 align="center">
  <strong>High-Speed Strategy Discovery Framework</strong>
</h4>

<p align="center">
  <a href="#quick-start">Quick Start</a> •
  <a href="#strategy-discovery">Strategy Discovery</a> •
  <a href="#live-trading">Live Trading</a> •
  <a href="#features">Features</a>
</p>

<p align="center">
  <a href="https://github.com/TPTBusiness/NexQuant/actions/workflows/ci.yml">
    <img src="https://img.shields.io/github/actions/workflow/status/TPTBusiness/NexQuant/ci.yml?branch=master&label=CI&logo=github&style=flat-square" alt="CI Status">
  </a>
  <a href="https://github.com/TPTBusiness/NexQuant/actions/workflows/codacy.yml">
    <img src="https://img.shields.io/github/actions/workflow/status/TPTBusiness/NexQuant/codacy.yml?branch=master&label=Security&logo=shield&style=flat-square" alt="Security Scan">
  </a>
  <a href="https://github.com/TPTBusiness/NexQuant/blob/master/LICENSE">
    <img src="https://img.shields.io/github/license/TPTBusiness/NexQuant?style=flat-square" alt="License">
  </a>
  <a href="https://github.com/astral-sh/ruff">
    <img src="https://img.shields.io/endpoint?url=https://raw.githubusercontent.com/astral-sh/ruff/main/assets/badge/v2.json&style=flat-square" alt="Ruff">
  </a>
  <a href="https://github.com/TPTBusiness/NexQuant/commits/master">
    <img src="https://img.shields.io/github/last-commit/TPTBusiness/NexQuant?style=flat-square" alt="Last Commit">
  </a>
</p>

---

## Overview

**NexQuant** discovers profitable trading strategies through high-speed search — no LLM required. Core engine: Numba JIT-compiled backtest at **735 million bars/second** (245× faster than pandas). Four discovery methods run in a continuous loop:

| Method | Frequency | Description |
|--------|-----------|-------------|
| **Explore** | 30% of iterations | Random strategies from 17 TA-Lib indicators across timeframes |
| **Exploit** | 70% of iterations | Mutate the best-known strategy (change params, indicator, or timeframe) |
| **Optuna** | Every 500 iterations | 20-trial hyperparameter optimization on the current best |
| **LightGBM** | Every 2000 iterations | ML classifier trained on SOTA indicator signals to predict direction |

**Current best strategy**: MACD(3,10,3) 4-TF with 2/4 vote majority — **+32.0%/month** (Numba), **+24.3%/month** (verified independent backtest), 0/75 negative months.

> **This repository contains the research framework.** Trading strategies, broker integrations, and live trading infrastructure are available as separate closed-source modules (`git_ignore_folder/`).

---

## Quick Start

```bash
# Prerequisites
conda create -n nexquant python=3.10 -y && conda activate nexquant
pip install -e .
# Ensure OHLCV data exists: git_ignore_folder/intraday_pv_all.h5

# Strategy Discovery Loop (10,000 iterations, ~1 hour)
python scripts/nexquant_rd_loop.py --iterations 10000

# Price-Action Indicator Loop (grid search all TA-Lib indicators)
python scripts/nexquant_priceaction_loop.py

# Top strategies report
python nexquant.py best -n 20 -m monthly_return --min-trades 30
```

---

## Strategy Discovery

### R&D Loop (`scripts/nexquant_rd_loop.py`)

```
 ┌──────────┐     ┌──────────┐     ┌──────────┐     ┌──────────┐
 │ Explore  │ ──→ │ Exploit  │ ──→ │ Optuna   │ ──→ │ LightGBM │
 │ (Random) │     │ (Mutate) │     │ (Tuning) │     │   (ML)   │
 └──────────┘     └──────────┘     └──────────┘     └──────────┘
       30%              70%           /500 iter        /2000 iter
```

**17 TA-Lib indicators**: MACD, RSI, Donchian, SAR, ADX, BBANDS, CCI, WCLPRICE, MFI, OBV, STOCH, ROC, AROON, AROONOSC, MOM, ULTOSC, WILLR

**4 timeframes**: 15min, 30min, 1h, 4h

**3 strategy types**: Single-TF, Multi-TF (vote majority), Portfolio (indicator ensemble)

**Discovery example** (50,000 iterations):
```
random → SAR(+65) → MACD(+73) → MACD-mutated(+102.75, +32%/month)
                                        ↓
                                  Optuna tuned params
                                        ↓
                                  LightGBM ensemble
```

### Grid Search (`scripts/nexquant_priceaction_loop.py`)

Deterministic parameter grid over all 17 indicators. Finds MACD(3,10,3) as optimal.

### Portfolio Optimizer (`scripts/nexquant_portfolio_optimizer.py`)

Greedy correlation-aware selection from discovered strategies.

---

## Live Trading

Closed-source module at `git_ignore_folder/nexquant_live_trader.py`. Architecture:

```
MACD(3,10,3) Signal → cTrader OpenAPI → Live Account
 4-TF 2/4 Votes      (WebSocket+Protobuf)        ↓
                                             Paper Mode
```

Integration: cTrader WebSocket `live.ctraderapi.com:5035`, OAuth2 authentication, Protobuf message encoding, FIX protocol.

---

## Features

### ⚡ Numba Backtest
- 735M bars/second (0.003s for 2.26M bars)
- JIT-compiled profit/drawdown/sharpe computation
- Signal construction via pandas resample + TA-Lib (~0.4s) is the bottleneck

### 🔍 Four Discovery Methods
- **Explore**: Random indicator + timeframe + parameters
- **Exploit**: Mutation of top-5 SOTA strategies (parameter tweak, indicator swap, timeframe change)
- **Optuna**: 20-trial TPE hyperparameter optimization on best strategy
- **LightGBM**: ML classifier on SOTA indicator signals (80/20 train/test split)

### 📊 TA-Lib Integration
- 17 indicators with full parameter ranges
- Auto-guard against bad parameters (negative/zero values that crash TA-Lib)
- Multi-timeframe voting with configurable threshold

### 🔒 Security & Quality
- 0 Dependabot alerts, 0 CodeScan alerts
- No proprietary terms in git history
- Closed-source detection CI

---

## Project Structure

```
nexquant/
├── scripts/                     # Strategy discovery & trading
│   ├── nexquant_rd_loop.py              # High-speed R&D loop (Numba + Optuna + ML)
│   ├── nexquant_priceaction_loop.py     # TA-Lib grid search loop
│   ├── nexquant_portfolio_optimizer.py  # Correlation-aware portfolio selection
│   ├── nexquant_gridsearch.py           # Deterministic parameter grid search
│   ├── nexquant_daily_strategies.py     # Daily Kronos + factor combinations
│   ├── nexquant_gen_strategies_real_bt.py  # LLM-based strategy generation
│   ├── nexquant_autopilot.py            # 24/7 continuous generator
│   └── nexquant_parallel.py             # Multi-instance parallel runs
├── rdagent/                     # Core framework (LLM-based, see note below)
│   ├── app/                     # CLI and scenario apps
│   ├── components/              # Backtest engine, protections, coders
│   ├── core/                    # Core abstractions
│   ├── scenarios/               # Domain-specific scenarios
│   └── utils/                   # Utilities
├── git_ignore_folder/           # Closed-source (never committed)
│   ├── nexquant_live_trader.py          # cTrader live trading
│   ├── nexquant_fix_trader.py           # FIX protocol trader
│   ├── intraday_pv_all.h5               # OHLCV data
│   ├── gbpusdt_1min.h5                  # GBP/USD data
│   └── btc_1min.h5                      # BTC data
├── test/                        # 1,125+ collected tests
├── data_config.yaml             # Walk-forward split configuration
├── requirements.txt             # Dependencies
└── AGENTS.md                    # Agent configuration & workflow guide
```

> **Note on `rdagent/`**: The LLM-based R&D framework (`rdagent fin_quant`) is part of the codebase but the Qlib/CoSTEER pipeline currently produces zero factors. The primary strategy discovery path is the Numba-based loop in `scripts/`.

---

## Installation

### Prerequisites
- **Conda** (Miniconda or Anaconda)
- **TA-Lib** system library (`apt install ta-lib` or `brew install ta-lib`)
- **Linux** (Ubuntu 22.04+)

### Install

```bash
git clone https://github.com/TPTBusiness/NexQuant && cd NexQuant
conda create -n nexquant python=3.10 -y && conda activate nexquant
pip install -e .
```

### Data
Place OHLCV HDF5 data at `git_ignore_folder/intraday_pv_all.h5`:
```python
# Format: MultiIndex (datetime, instrument), columns: $open $close $high $low $volume
df.to_hdf('git_ignore_folder/intraday_pv_all.h5', key='data')
```

---

## License

**GNU Affero General Public License v3.0 (AGPL-3.0)**. See [`LICENSE`](LICENSE).

---

## Disclaimer

NexQuant is provided for **research and educational purposes only**. Past performance does not guarantee future results. Users assume all liability.
