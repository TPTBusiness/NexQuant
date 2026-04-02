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
│   │   │   └── factor_coder/   # Factor generation & EURUSD-specific modules
│   │   └── ...
│   └── scenarios/
│       └── qlib/               # Qlib integration for FX trading
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

```bash
# Run all tests
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
- ✅ Backtesting Engine (IC, Sharpe, Drawdown)
- ✅ Results Database (SQLite with queries)
- ✅ Risk Management (Correlation, Portfolio Optimization)
- ✅ Dashboards (Web + CLI)
- ⏳ Live Trading (Paper trading pending)

### Next Steps

1. Backtest all 110 factors
2. Select top 20 by IC/Sharpe
3. Portfolio optimization
4. 4 weeks paper trading
5. Live trading with small capital

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

If you need to fix commit messages in history:

```bash
# For last 3 commits
git rebase -i HEAD~3

# Change 'pick' to 'reword' for commits to rename
# Editor opens - write new message in English
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
