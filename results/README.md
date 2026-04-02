# Predix Results Directory

This directory stores all backtesting results, databases, and reports.

**⚠️ IMPORTANT:** This directory is in `.gitignore` and will NOT be committed to GitHub.

---

## 📁 Directory Structure

```
results/
├── backtests/              # Individual factor backtest results (JSON, CSV)
│   ├── FactorName_20260402_120000.json
│   ├── FactorName_20260402_120000_returns.csv
│   └── FactorName_20260402_120000_equity.csv
│
├── db/                     # SQLite database for all results
│   └── backtest_results.db
│
├── factors/                # Factor-specific analysis
│   ├── factor_performance.json
│   └── ic_history.csv
│
├── runs/                   # Complete run results & risk reports
│   ├── risk_report_20260402_120000.json
│   └── portfolio_weights_20260402.json
│
└── logs/                   # Backtest logs
    └── backtest_20260402.log
```

---

## 📊 What Gets Stored

### Backtests (`backtests/`)

For each factor backtest:
- **JSON file**: All metrics (IC, Sharpe, Drawdown, Win Rate, etc.)
- **Returns CSV**: Daily returns time series
- **Equity CSV**: Equity curve

**Example JSON:**
```json
{
  "factor_name": "Momentum_8Bar",
  "ic": 0.045,
  "sharpe_ratio": 1.85,
  "max_drawdown": -0.08,
  "win_rate": 0.58,
  "total_trades": 252,
  "timestamp": "2026-04-02T12:00:00"
}
```

### Database (`db/backtest_results.db`)

SQLite database with tables:
- `factors` - All generated factors
- `backtest_runs` - Backtest results with metrics
- `backtest_metrics` - Detailed metrics per run
- `daily_returns` - Daily returns time series
- `loop_results` - Loop execution summaries

### Risk Reports (`runs/`)

- Portfolio volatility
- Sharpe Ratio
- Diversification Ratio
- Max Drawdown
- Limit Checks (Position Size, Leverage, Drawdown)
- Correlation Matrix

---

## 🔍 Querying Results

### Python Example

```python
from rdagent.components.backtesting import ResultsDatabase

# Connect to database
db = ResultsDatabase()

# Get top 20 factors by Sharpe Ratio
top_factors = db.get_top_factors('sharpe_ratio', limit=20)
print(top_factors)

# Get aggregate statistics
stats = db.get_aggregate_stats()
print(f"Total factors: {stats['total_factors']}")
print(f"Average IC: {stats['avg_ic']}")
print(f"Max Sharpe: {stats['max_sharpe']}")

# Close connection
db.close()
```

### SQL Example

```bash
# Open database
sqlite3 results/db/backtest_results.db

# Query top factors
SELECT factor_name, sharpe, ic, win_rate 
FROM backtest_runs 
ORDER BY sharpe DESC 
LIMIT 10;

# Get aggregate stats
SELECT COUNT(*) as total_factors,
       AVG(ic) as avg_ic,
       MAX(sharpe) as max_sharpe
FROM backtest_runs;
```

---

## 🧹 Cleanup

To clean up old results:

```bash
# Remove all results
rm -rf results/*

# Remove only backtests
rm -rf results/backtests/*

# Remove database
rm -f results/db/backtest_results.db

# Keep logs but remove everything else
find results/ -type f ! -path "*/logs/*" -delete
```

---

## 📝 Notes

- Results are stored locally and never committed to Git
- Database is automatically created on first run
- JSON files are human-readable for quick inspection
- Use SQLite database for programmatic access
- Logs are stored separately for debugging

---

**For detailed usage guidelines, see [README.md](../README.md)**
