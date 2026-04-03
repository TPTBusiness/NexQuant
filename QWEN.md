# Predix - QWEN.md Context File

## Project Overview

**Predix** is an autonomous AI-powered quantitative trading agent for EUR/USD forex markets. Built on the RD-Agent framework, it automates the full research and development cycle for trading strategies.

### Core Purpose
- Generate trading factors (signals) autonomously using LLMs
- Backtest and validate factors on 1-minute EUR/USD data
- Optimize portfolios using modern portfolio theory
- Target: 1-3% monthly returns with Sharpe > 2.0

### Key Technologies
- **Python 3.10/3.11** - Primary language
- **PyTorch** - Deep learning models
- **Qlib** - Backtesting engine
- **LLM (Qwen3.5-35B)** - Factor generation via local llama.cpp
- **Flask** - Web dashboard API
- **SQLite** - Results database
- **Rich/Typer** - CLI interface

### Architecture

```
Predix/
├── rdagent/                    # Core agent framework
│   ├── app/
│   │   └── cli.py              # Main CLI entry point (rdagent command)
│   ├── components/
│   │   ├── backtesting/        # Backtest engine, metrics, database
│   │   ├── coder/
│   │   │   ├── factor_coder/   # Factor generation & EURUSD-specific modules
│   │   │   └── rl/             # RL Trading Agent (NEW)
│   │   │       ├── env.py      # Gym-compatible trading environment
│   │   │       ├── agent.py    # Stable Baselines3 wrapper (PPO/A2C/SAC)
│   │   │       ├── costeer.py  # RL trading controller + LLM code generation
│   │   │       └── indicators.py # Technical indicators (RSI, MACD, BB, CCI, ATR)
│   │   ├── loader.py           # Prompt loader (auto-loads local prompts)
│   │   └── model_loader.py     # Model loader (auto-loads local models)
│   └── scenarios/
│       └── qlib/               # Qlib integration for FX trading
├── prompts/                    # LLM Prompts
│   ├── standard_prompts.yaml   # Standard prompts (in Git)
│   └── local/                  # Your improved prompts (NOT in Git!)
│       ├── factor_discovery_v2.yaml
│       ├── factor_evolution_v2.yaml
│       └── model_coder_v2.yaml
├── models/                     # ML Models
│   ├── standard/               # Standard models (in Git)
│   │   ├── xgboost_factor.py
│   │   └── lightgbm_factor.py
│   └── local/                  # Your improved models (NOT in Git!)
│       ├── transformer_factor.py
│       ├── tcn_factor.py
│       ├── patchtst_factor.py
│       └── cnn_lstm_hybrid.py
├── results/                    # Backtest results (NOT in git)
│   ├── backtests/              # Individual factor backtests (JSON/CSV)
│   ├── db/                     # SQLite database
│   ├── factors/                # Factor analysis
│   ├── runs/                   # Run results & risk reports
│   └── logs/                   # Backtest logs
├── web/                        # Dashboard frontend
│   ├── dashboard_api.py        # Flask API backend
│   └── dashboard.html          # Web UI
├── .env                        # Environment config (API keys, etc.)
├── data_config.yaml            # EURUSD data configuration
└── requirements.txt            # Python dependencies
```

### Open Source vs. Closed Source

**🟢 OPEN SOURCE (Public on GitHub - FULLY WORKING):**
- `rdagent/` - Core framework (ALL components)
- `models/standard/` - Base models (XGBoost, LightGBM)
- `prompts/standard_prompts.yaml` - Base prompts
- `web/` - Dashboards
- `test/` - ALL tests (integration, unit, security)
- `rdagent/components/coder/rl/` - RL Trading System (with fallback)
- `rdagent/components/backtesting/protections/` - Trading Protection System
- `scripts/` - Utility scripts

**GitHub users get:**
✅ Full working trading system
✅ RL Trading with graceful fallback (no stable-baselines3 needed)
✅ Protection Manager (drawdown, cooldown, stoploss guard)
✅ Backtesting Engine with RL support
✅ CLI commands (`fin_quant`, `rl_trading`, etc.)
✅ Web and CLI dashboards
✅ All 200+ integration tests

**🔒 CLOSED SOURCE (Local Only - NOT on GitHub):**
- `models/local/` - Your improved models (Transformer, TCN, PatchTST, CNN+LSTM)
- `prompts/local/` - Your improved prompts (v2.0 optimized)
- `.env` - API keys
- `results/` - Backtest results
- `git_ignore_folder/` - Trading data
- `QWEN.md`, `TODO.md` - Internal docs

**Protection:**
- `.gitignore` excludes all `local/` directories
- Your competitive edge (alpha) stays private
- Framework is open, but your best models/prompts are closed

### Open Source Fallback Strategy

**For users without stable-baselines3:**
The RL system provides graceful degradation:
- ❌ No stable-baselines3 → Uses simple momentum-based fallback
- ✅ Still fully functional: CLI, backtesting, protections work
- ✅ No errors or broken features
- ✅ Clear warning message with installation instructions

**For users without LLM (llama.cpp):**
- Factor evolution degrades gracefully
- System still works with standard models
- Clear error messages for missing LLM

**PRINCIPLE:** Every GitHub user MUST be able to run the full system. Missing optional components should never break the project.

## Building and Running

### Installation

```bash
# Clone repository
git clone https://github.com/PredixAI/predix
cd predix

# Create conda environment
conda create -n predix python=3.10
conda activate predix

# Install in editable mode
pip install -e .[test,lint]
```

### Configuration

1. **Create `.env` file:**
```bash
# Local LLM (llama.cpp)
OPENAI_API_KEY=local
OPENAI_API_BASE=http://localhost:8081/v1
CHAT_MODEL=qwen3.5-35b

# Embedding (Ollama)
LITELLM_PROXY_API_KEY=local
LITELLM_PROXY_API_BASE=http://localhost:11434/v1
EMBEDDING_MODEL=nomic-embed-text

# Paths
QLIB_DATA_DIR=~/.qlib/qlib_data/eurusd_1min_data
```

2. **Start LLM server (llama.cpp):**
```bash
~/llama.cpp/build/bin/llama-server \
  --model ~/models/qwen3.5/Qwen3.5-35B-A3B-Q3_K_M.gguf \
  --n-gpu-layers 36 \
  --ctx-size 80000 \
  --port 8081
```

### Running the Trading Loop

```bash
# Start trading loop (24/7)
./start_loop.sh

# Or single run
rdagent fin_quant

# With dashboard
rdagent fin_quant --with-dashboard

# With CLI dashboard
rdagent fin_quant --cli-dashboard
```

### Running the Dashboard

```bash
# Web dashboard (runs with fin_quant --with-dashboard)
# Access at: http://localhost:5000/dashboard.html

# Or standalone
python web/dashboard_api.py
```

### Testing

#### Integration Test Suite (ALL Features)

**Comprehensive test system that validates ALL 13 implemented features:**

```bash
# Run ALL integration tests (60 tests, ~7.5 seconds)
pytest test/integration/test_all_features.py -v

# Run with coverage report
pytest test/integration/test_all_features.py --cov=rdagent.components.backtesting -v

# Run via test runner script
./scripts/run_all_tests.sh

# Test specific features only
pytest test/integration/test_all_features.py -k "backtest or database" -v

# Skip slow tests
pytest test/integration/test_all_features.py -m "not slow" -v
```

**Tested Features (60 Tests, ALL MUST PASS):**

| # | Feature | Tests | Status |
|---|---------|-------|--------|
| 1 | Factor Evolution | 5 | ✅ LLM generates trading factors autonomously |
| 2 | Model Evolution | 5 | ✅ ML models auto-improved |
| 3 | Quant Loop (fin_quant) | 4 | ✅ Main 24/7 trading loop |
| 4 | Backtesting Engine | 5 | ✅ IC, Sharpe, Drawdown, Win Rate |
| 5 | Results Database | 5 | ✅ SQLite with queries |
| 6 | Risk Management | 6 | ✅ Correlation, Portfolio Optimization |
| 7 | CLI Dashboard | 4 | ✅ Rich live-progress display |
| 8 | Web Dashboard | 4 | ✅ Flask API + HTML |
| 9 | Health Check | 4 | ✅ Environment validation |
| 10 | Streamlit UI | 3 | ✅ Alternative dashboard |
| 11 | LLM Integration | 5 | ✅ llama.cpp (Qwen3.5-35B) |
| 12 | Embedding | 3 | ✅ Ollama (nomic-embed-text) |
| 13 | Security Scanning | 5 | ✅ Bandit pre-commit hook |

**⚠️ MANDATORY: These tests run BEFORE every commit and MUST pass!**

#### Unit Tests

```bash
# Run all unit tests
pytest test/

# Run with coverage
pytest --cov=rdagent --cov-report=html

# Test backtesting module
python rdagent/components/backtesting/backtest_engine.py
python rdagent/components/backtesting/results_db.py
python rdagent/components/backtesting/risk_management.py
```

### Code Quality

```bash
# Linting
ruff check rdagent/

# Type checking
mypy rdagent/

# Format
black rdagent/

# Pre-commit (install first)
pre-commit install
pre-commit run --all-files
```

## Development Conventions

### Language Policy

**ALL code comments and documentation MUST be in English.**

❌ **Wrong (German):**
```python
# Inspiriert von: TradingAgents
# Berechnet den Sharpe Ratio
# Achtung: Division durch Null möglich!
# Hinweis: Diese Funktion ist experimentell
```

✅ **Correct (English):**
```python
# Inspired by: TradingAgents
# Calculates the Sharpe ratio
# Warning: Division by zero possible!
# Note: This function is experimental
```

**Rationale:**
- International collaboration
- Better searchability
- Professional codebase
- Consistent with commit messages (also English-only)

**Enforcement:**
- All new code must have English comments
- Existing German comments should be translated when modified
- PRs with German comments will be rejected

### Code Style

- **Line length:** 120 characters (configured in pyproject.toml)
- **Type hints:** Required for all public functions
- **Docstrings:** Google style for public APIs
- **Imports:** Sorted automatically with isort

### Testing Practices
- Unit tests in `test/` directory
- Test files named `test_*.py`
- Use pytest fixtures for common setup
- Mock external APIs (LLM, yfinance)
- Minimum 80% coverage target

### Commit Conventions
```bash
git commit --author="TPTBusiness <tpt.requests@pm.me>" -m "type: description"

# Types:
# - feat: New feature
# - fix: Bug fix
# - docs: Documentation
# - style: Formatting
# - refactor: Code restructuring
# - test: Tests
# - chore: Maintenance
```

### Module Structure
```python
"""
Module Name - Brief description

Longer description if needed.
"""

import numpy as np
import pandas as pd
from typing import Dict, List, Optional
from datetime import datetime

class ClassName:
    """Class docstring."""
    
    def __init__(self, param: type) -> None:
        """Initialize."""
        pass
    
    def method(self, param: type) -> ReturnType:
        """
        Method docstring.
        
        Parameters
        ----------
        param : type
            Description
        
        Returns
        -------
        ReturnType
            Description
        """
        pass
```

### Backtesting Module Usage

```python
from rdagent.components.backtesting import (
    FactorBacktester,
    ResultsDatabase,
    PortfolioOptimizer,
    AdvancedRiskManager
)

# Run backtest
backtester = FactorBacktester()
metrics = backtester.run_backtest(
    factor_values=factor_series,
    forward_returns=forward_returns,
    factor_name="MyFactor"
)

# Save to database
db = ResultsDatabase()
db.add_backtest("MyFactor", metrics)

# Query top factors
top = db.get_top_factors('sharpe_ratio', limit=20)

# Portfolio optimization
optimizer = PortfolioOptimizer()
weights = optimizer.mean_variance(expected_returns, cov_matrix)

# Risk management
risk_manager = AdvancedRiskManager()
report = risk_manager.generate_risk_report(returns, weights)
```

### Key Metrics

| Metric | Target | Minimum |
|--------|--------|---------|
| IC (Information Coefficient) | > 0.05 | > 0.02 |
| Sharpe Ratio | > 2.0 | > 1.0 |
| Max Drawdown | < 15% | < 25% |
| Win Rate | > 55% | > 45% |
| Annualized Return | > 10% | > 5% |

### Important Files

- `rdagent/app/cli.py` - Main CLI entry point
- `rdagent/components/backtesting/` - Backtest engine
- `rdagent/components/coder/factor_coder/` - Factor generation
- `results/README.md` - Results documentation
- `data_config.yaml` - EURUSD configuration
- `web/dashboard_api.py` - Dashboard API
- `requirements.txt` - Dependencies

### External Dependencies

- **llama.cpp** - Local LLM inference (Qwen3.5-35B)
- **Ollama** - Embedding models
- **Qlib** - Backtesting engine
- **yfinance** - Live market data

### Common Issues

1. **LLM Connection Errors:** Ensure llama.cpp server is running on port 8081
2. **Embedding Errors:** Check Ollama is running with nomic-embed-text loaded
3. **Database Lock:** Close all connections before running multiple processes
4. **Memory Issues:** Reduce batch size or context length for LLM

### Project Status

- ✅ Factor Generation (110+ factors created)
- ✅ Backtesting Engine (IC, Sharpe, Drawdown, RL support)
- ✅ Results Database (SQLite with queries)
- ✅ Risk Management (Correlation, Portfolio Optimization)
- ✅ Trading Protection System (Drawdown, Cooldown, Stoploss Guard, Low Performance)
- ✅ RL Trading Agent (PPO/A2C/SAC with Gymnasium environment + fallback)
- ✅ Dashboards (Web + CLI)
- ✅ CLI Commands (`fin_quant`, `rl_trading`, `health_check`, etc.)
- ✅ Integration Tests (200+ tests, run before EVERY commit)
- ✅ Security Scanning (Bandit pre-commit hook)
- ⏳ Live Trading (Paper trading - in development)

### Next Steps

1. ✅ Connect RL with Protection Manager (DONE)
2. ✅ Connect RL with Backtesting Engine (DONE)
3. ✅ Add CLI command for RL Trading (DONE)
4. ✅ Ensure GitHub users can run full system (DONE - fallback system)
5. Backtest all 110 factors
6. Select top 20 by IC/Sharpe
7. Portfolio optimization
8. 4 weeks paper trading
9. Live trading with small capital

---

## Git Commit Guidelines

### Language Policy

**ALL commit messages MUST be in English.**

❌ **Wrong (German):**
```bash
git commit -m "feat: Neue Funktion hinzugefügt"
git commit -m "fix: Fehler behoben"
git commit -m "chore: QWEN.md zu .gitignore hinzugefügt"
```

✅ **Correct (English):**
```bash
git commit -m "feat: Add new feature"
git commit -m "fix: Fix bug"
git commit -m "chore: Add QWEN.md to .gitignore"
```

### Pre-Commit Checklist

**BEFORE every commit, you MUST:**

1. **Run `git status`** and verify:
   - Only intended files are staged
   - No generated files (.qwen/, results/, *.db, etc.)
   - No sensitive data (.env, API keys, etc.)

2. **Check .gitignore** is working:
   ```bash
   git status
   # Verify .qwen/, results/, *.db are NOT shown
   ```

3. **Review staged changes:**
   ```bash
   git diff --staged
   # Review what will be committed
   ```

4. **Run tests** (if applicable):
   ```bash
   pytest test/backtesting/ -v
   # Ensure all tests pass
   ```

### Commit Message Format

Use [Conventional Commits](https://www.conventionalcommits.org/):

```
<type>: <description in English>

[optional body]
```

**Types:**
- `feat:` - New feature
- `fix:` - Bug fix
- `test:` - Tests
- `docs:` - Documentation
- `chore:` - Maintenance
- `style:` - Formatting
- `refactor:` - Code restructuring

**Examples:**
```bash
feat: Add backtesting tests with 98% coverage
fix: Remove .qwen/ from Git tracking
test: Add unit tests for ResultsDatabase
docs: Update QWEN.md with commit guidelines
chore: Add pytest to requirements.txt
```

### Protected Files (NEVER commit)

These files/directories MUST NEVER be committed:

```
.qwen/              # AI agent files (generated)
results/            # Backtest results (sensitive data)
*.db                # SQLite databases
.env                # Environment variables (API keys!)
git_ignore_folder/  # Generated data
*.log               # Log files
```

If you accidentally commit any of these:

```bash
# Remove from last commit (keeps files locally)
git reset HEAD~1

# Or remove from tracking
git rm -r --cached .qwen/
git commit -m "chore: Remove .qwen/ from tracking"
```

### Fixing Past Commits

**To fix the last 3-5 commits:**

```bash
# For last 5 commits
git rebase -i HEAD~5

# In the editor, change 'pick' to 'reword' for commits to rename
# Save and close
# Write new English message for each commit
```

**To fix older commits (advanced):**

```bash
# Find the commit hash
git log --oneline

# Start rebase from that commit
git rebase -i <commit-hash>^

# Follow same process as above
```

**Current German commits to fix (as of April 2026):**
```
73140b68 test: Backtesting Tests mit 98.77% Coverage
     → test: Add backtesting tests with 98.77% coverage

5148d17d chore: QWEN.md zu .gitignore hinzugefügt
     → chore: Add QWEN.md to .gitignore

df93e162 feat: Intelligent Embedding Chunking statt Kürzung
     → feat: Intelligent embedding chunking instead of truncation

01aa183a fix: CLI Dashboard in separatem Terminal-Fenster
     → fix: CLI dashboard in separate terminal window

df356978 feat: predix.py Wrapper für Dashboard-Support
     → feat: predix.py wrapper for dashboard support

89d01f5d feat: Beautiful CLI Dashboard + korrigierter Start-Befehl
     → feat: Beautiful CLI dashboard + corrected start command

48e4f44e feat: Auto-Start Dashboard für fin_quant
     → feat: Auto-start dashboard for fin_quant

59122a19 feat: Dashboard + Live-Daten Integration (Phase 4)
     → feat: Dashboard + live data integration (Phase 4)

a0f414ed feat: EURUSD Trading-Verbesserungen (Phase 2 & 3)
     → feat: EURUSD trading improvements (Phase 2 & 3)

e8b962b5 feat: EURUSD Trading-Verbesserungen implementiert (Phase 1)
     → feat: Implement EURUSD trading improvements (Phase 1)
```

**⚠️ Warning:** Rewriting history changes commit hashes. If you've already pushed:

```bash
# After rebasing locally
git push --force-with-lease origin master

# Tell team members to re-clone:
git clone <repo-url>
```

### Push Policy

**BEFORE pushing:**

1. Verify commit messages are in English
2. Verify no protected files are included
3. Run tests one final time

```bash
git status
git log -3 --oneline  # Verify last 3 commits
pytest test/backtesting/ -v  # Quick test
git push origin master
```

### Enforcement

- All PRs will be rejected if commit messages are not in English
- Protected files in commits will be rejected
- Tests must pass before merging

**Remember:** Consistent English commit messages ensure:
- International collaboration
- Better searchability
- Professional project history

---

## Implementation Guide: Prompts & Models

### Using the Prompt Loader

**Auto-Load Prompts (Local First):**

```python
from rdagent.components.loader import load_prompt

# Load factor discovery prompt
# Automatically loads from prompts/local/ if exists!
prompt = load_prompt("factor_discovery")

# Load specific section
system_prompt = load_prompt("factor_discovery", section="system")
user_prompt = load_prompt("factor_discovery", section="user")

# Force local only (raise error if not found)
prompt = load_prompt("factor_discovery", local_only=True)

# List available prompts
from rdagent.components.loader import list_available_prompts
available = list_available_prompts()
print(f"Standard: {available['standard']}")
print(f"Local: {available['local']}")
```

**Priority:**
1. `prompts/local/factor_discovery_v2.yaml` (loaded first if exists)
2. `prompts/local/factor_discovery.yaml`
3. `prompts/standard_prompts.yaml` (fallback)

---

### Using the Model Loader

**Auto-Load Models (Local First):**

```python
from rdagent.components.model_loader import load_model

# Load XGBoost model
# Automatically loads from models/local/ if exists!
model_factory = load_model("xgboost_factor")

# Create model instance
model = model_factory(max_depth=8, learning_rate=0.03)

# Train
model.fit(X_train, y_train, epochs=50, batch_size=64)

# Predict
predictions = model.predict(X_test)

# Save/Load
model.save("models/my_model.pth")
model.load("models/my_model.pth")
```

**Available Models:**

| Model | Location | Use Case |
|-------|----------|----------|
| `xgboost_factor` | `models/standard/` | Tabular data, fast training |
| `lightgbm_factor` | `models/standard/` | Large datasets, faster than XGBoost |
| `transformer_factor` | `models/local/` | Time-series, long-range dependencies |
| `tcn_factor` | `models/local/` | Multi-scale patterns |
| `patchtst_factor` | `models/local/` | **SOTA** for time-series forecasting |
| `cnn_lstm_hybrid` | `models/local/` | Complex pattern recognition |

**Priority:**
1. `models/local/{name}_v2.py` (loaded first if exists)
2. `models/local/{name}.py`
3. `models/standard/{name}.py` (fallback)

---

### Creating Your Improved Prompts

**Step 1: Create Local Prompt**

```bash
mkdir -p prompts/local
nano prompts/local/factor_discovery_v3.yaml
```

**Step 2: Add Your Improvements**

```yaml
# prompts/local/factor_discovery_v3.yaml

factor_discovery:
  system: |-
    YOUR IMPROVED SYSTEM PROMPT HERE
    
    Add your proprietary insights:
    - Specific EURUSD patterns you've discovered
    - Your unique factor formulas
    - Custom session filters
    - Proprietary risk management rules
    
  user: |-
    YOUR IMPROVED USER PROMPT HERE
```

**Step 3: Test**

```python
from rdagent.components.loader import load_prompt

# Auto-loads your v3!
prompt = load_prompt("factor_discovery")
```

---

### Creating Your Improved Models

**Step 1: Create Local Model**

```bash
mkdir -p models/local
nano models/local/my_optimized_model.py
```

**Step 2: Implement Model**

```python
# models/local/my_optimized_model.py
"""
My Optimized Model v1.0
Better than standard with custom improvements.
"""

import torch
import torch.nn as nn

class MyOptimizedModel(nn.Module):
    def __init__(self, **params):
        super().__init__()
        # Your custom architecture
        pass
    
    def forward(self, x):
        # Your custom forward pass
        pass

def create_my_optimized_model(**params):
    """Factory function."""
    return MyOptimizedModel(**params)
```

**Step 3: Test**

```python
from rdagent.components.model_loader import load_model

# Auto-loads your optimized model!
model_factory = load_model("my_optimized_model")
model = model_factory()
```

---

### Backup Your Private Assets

**Backup Prompts & Models to Private Repo:**

```bash
# Create private repo on GitHub: predix-private-assets

# Clone private repo
cd ~/Dev
git clone git@github.com:TPTBusiness/predix-private-assets.git

# Copy local assets
cp -r ~/Predix/prompts/local/* ~/predix-private-assets/prompts/
cp -r ~/Predix/models/local/* ~/predix-private-assets/models/

# Commit to private repo
cd ~/predix-private-assets
git add .
git commit -m "Backup: prompts v2, models (Transformer, TCN, PatchTST, CNN+LSTM)"
git push
```

**Auto-Sync Script:**

```bash
# ~/Predix/sync_private.sh
#!/bin/bash
echo "Syncing private assets..."
rsync -av prompts/local/ ~/predix-private-assets/prompts/
rsync -av models/local/ ~/predix-private-assets/models/
cd ~/predix-private-assets && git add . && git commit -m "Auto-sync $(date)" && git push
echo "Done!"
```

---

### Security Best Practices

**What to Keep Private:**

✅ Your proprietary model architectures
✅ Optimized prompt templates
✅ Best-performing factors
✅ Evolution weights
✅ Trade secrets & alpha-generating logic

**What NOT to Commit:**

❌ Anything in `prompts/local/`
❌ Anything in `models/local/`
❌ `.env` (API keys)
❌ `results/` (backtest performance)
❌ `git_ignore_folder/` (trading data)

**Verify Before Committing:**

```bash
# Check what will be committed
git status
git diff --staged

# Verify .gitignore is working
git status
# Should NOT show prompts/local/, models/local/, .env, results/
```

---

## Development Guidelines for AI Assistant

### 🌍 CRITICAL: Open Source Compatibility

**BEFORE implementing ANY feature, ask yourself:**

1. **Can a GitHub user run this without our local files?**
   - ✅ YES → Good, proceed
   - ❌ NO → Add fallback or graceful degradation

2. **Does this break if optional dependencies are missing?**
   - Example: `stable-baselines3`, `llama.cpp`, `Ollama`
   - Solution: Try/except with clear warning messages

3. **Is this feature documented for external users?**
   - Update README.md with usage instructions
   - Ensure installation guide covers all dependencies

**PRINCIPLE:** The project on GitHub MUST be fully functional for users. Our closed-source assets (`models/local/`, `prompts/local/`, `.env`) are ENHANCEMENTS, not requirements.

### ⚠️ MANDATORY Rules for ALL Development

**When implementing NEW features or making SIGNIFICANT changes, you MUST:**

#### 1. 📝 Update QWEN.md

**When:** Every time you add a new feature, module, or change existing architecture.

**What to update:**
- Architecture section (if structure changes)
- Important Files section
- Testing section
- Key Metrics (if targets change)
- Project Status
- Next Steps

**Example:**
```markdown
### Architecture
├── rdagent/
│   └── components/
│       └── backtesting/
│           └── protections/          # NEW: Trading protection system
│               ├── base.py
│               ├── max_drawdown.py
│               └── protection_manager.py
```

#### 2. 📖 Update README.md

**When:** Every user-facing feature change or major update.

**What to update:**
- Features list
- Installation instructions
- Usage examples
- Configuration examples

**Keep it user-focused:**
```markdown
## Features
- ✅ Trading Protection System (NEW)
  * Automatic drawdown protection
  * Cooldown periods after losses
  * Stoploss cluster detection
```

#### 3. 📦 Update requirements.txt

**When:** Adding new dependencies or removing unused ones.

**What to update:**
- `requirements.txt` (main dependencies)
- `requirements/lint.txt` (dev dependencies)
- `requirements/test.txt` (test dependencies)

**Example:**
```bash
# If you add a new library
echo "new-library==1.0.0" >> requirements.txt

# If you add a new test dependency
echo "pytest-mock" >> requirements/test.txt
```

#### 4. ✅ Extend Tests

**When:** EVERY time you add new code.

**Rule:** New features MUST have tests with >80% coverage.

**What to create:**
- Unit tests in `test/` directory
- Integration tests in `test/integration/`
- Update existing tests if behavior changed

**Test structure:**
```python
# test/feature_type/test_new_feature.py
"""Tests for New Feature"""

class TestNewFeature:
    """Test new feature thoroughly."""
    
    def test_basic_functionality(self): ...
    def test_edge_cases(self): ...
    def test_error_handling(self): ...
    def test_integration_with_existing(self): ...
```

**Update integration tests:**
```python
# Add to test/integration/test_all_features.py
class TestNewFeature:
    """Test new feature integration."""
    
    def test_imports(self): ...
    def test_initialization(self): ...
    def test_full_workflow(self): ...
```

#### 5. 🔄 Pre-Commit Checklist

**BEFORE every commit with new features:**

```bash
# 1. Run ALL tests
pytest test/ -v

# 2. Run integration tests
pytest test/integration/test_all_features.py -v

# 3. Check test coverage
pytest --cov=rdagent.components.new_module -v

# 4. Run security scan
bandit -r rdagent/ -c .bandit.yml

# 5. Verify tests updated
git status
# Should show test files modified
```

### Documentation Priority Order

1. **QWEN.md** - Internal AI assistant context (UPDATE ALWAYS)
2. **Test files** - Code documentation through tests (MANDATORY)
3. **README.md** - User-facing documentation (UPDATE for user-visible changes)
4. **requirements.txt** - Dependencies (UPDATE when adding libraries)
5. **Inline code comments** - English only (ALWAYS)

### Example Workflow: Adding New Feature

```
1. Plan feature
   ↓
2. Implement code
   ↓
3. Write unit tests (test/...)
   ↓
4. Write integration tests (test/integration/...)
   ↓
5. Run ALL tests → Must pass
   ↓
6. Update QWEN.md ← MANDATORY
   ↓
7. Update README.md (if user-visible)
   ↓
8. Update requirements.txt (if new deps)
   ↓
9. Commit with clear message
   ↓
10. Pre-commit hooks run automatically
    ↓
11. Push to remote
```

### Penalties for Not Following Rules

**If you forget to update:**
- ❌ Missing tests → Code cannot be committed (pre-commit blocks)
- ❌ Missing QWEN.md update → Next AI assistant will work with outdated context
- ❌ Missing README update → Users won't understand new features
- ❌ Missing requirements.txt → Installation will fail

**Remember:** These rules ensure:
1. Code quality through tests
2. AI assistant has current context
3. Users understand changes
4. Dependencies are tracked

---
