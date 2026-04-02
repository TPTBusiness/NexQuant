# Changelog

All notable changes to Predix will be documented in this file.

The format is based on [Keep a Changelog](https://keepachangelog.com/en/1.0.0/),
and this project adheres to [Semantic Versioning](https://semver.org/spec/v2.0.0.html).

## [1.0.0] - 2026-04-02

### ✨ Added
- **Autonomous Factor Generation**
  - 110+ EURUSD factors generated autonomously using LLMs
  - Multi-agent debate system (Bull/Bear/Neutral analysts)
  - Stanley Druckenmiller-style macro analysis agent
  - Market regime detection using Hurst Exponent

- **Backtesting Engine**
  - IC (Information Coefficient) calculation
  - Sharpe Ratio, Sortino Ratio, Calmar Ratio
  - Max Drawdown with start/end dates
  - Win Rate, Total Trades tracking
  - Transaction cost modeling

- **Results Database**
  - SQLite database for tracking all backtest results
  - Factors, backtest runs, loop results tables
  - Queries for top factors by Sharpe/IC
  - Aggregate statistics

- **Risk Management**
  - Correlation matrix between factors
  - Portfolio optimization (Mean-Variance, Risk Parity)
  - Position sizing with volatility adjustment
  - Risk limits (position size, leverage, drawdown)

- **Dashboards & UI**
  - Web Dashboard (Flask + HTML) with live progress
  - CLI Dashboard (Rich library) for terminal
  - Real-time macro data display
  - Session-aware analysis (Asian/London/NY)

- **Testing Infrastructure**
  - 97 unit tests with 98.77% code coverage
  - Edge case testing for all metrics
  - Integration tests for full workflows
  - pytest configuration

- **Documentation**
  - Comprehensive QWEN.md (development guide)
  - ATTRIBUTION.md (usage guidelines)
  - README.md (installation, quick start)
  - All code comments in English

- **Developer Experience**
  - English-only commit messages policy
  - Clean git history
  - .gitignore for sensitive files
  - Makefile for common tasks

### 🔧 Changed
- Rebranded from RD-Agent to Predix for EUR/USD quantitative trading
- Updated project metadata for PredixAI organization
- All code comments translated to English
- Removed 'Inspired by' comments, added comprehensive Acknowledgments
- Enhanced .gitignore for better file management

### 🛡️ Fixed
- Removed all Chinese stock references, replaced with EUR/USD 1min FX data
- Migrated to 1min EURUSD data (2020-2026)
- Injected MultiIndex warning into factor interface prompt
- Fixed Embedding Context Length errors with intelligent chunking
- Removed test configuration files from root directory
- Cleaned up log files and test artifacts from git history

### 📦 Dependencies
- Python 3.10/3.11
- PyTorch for deep learning
- Qlib for backtesting
- Flask for web dashboard
- Rich/Typer for CLI
- pytest for testing (98.77% coverage)

### 🙏 Acknowledgments
- Built on Microsoft RD-Agent framework (MIT License)
- Inspired by TradingAgents (Apache 2.0 License)
- Concepts from ai-hedge-fund

---

## [Unreleased]

## Historical Changes (from RD-Agent upstream)

For earlier changes inherited from the RD-Agent project, see the [upstream changelog](https://github.com/microsoft/RD-Agent/blob/main/CHANGELOG.md).
