"""
Predix Results Database - SQLite für Backtest-Ergebnisse

Stores backtest metrics from Qlib/MLflow runs for querying and dashboard display.
"""
import sqlite3
import json
import pandas as pd
from pathlib import Path
from datetime import datetime
from typing import Dict, Optional


class ResultsDatabase:
    def __init__(self, db_path: Optional[str] = None):
        if db_path is None:
            db_path = Path(__file__).parent.parent.parent / "results" / "db" / "backtest_results.db"
        self.db_path = db_path
        Path(db_path).parent.mkdir(parents=True, exist_ok=True)
        self.conn = sqlite3.connect(db_path)
        self._create_tables()

    def _create_tables(self):
        """Create database tables if they don't exist and migrate schema if needed."""
        c = self.conn.cursor()

        # Factors table - stores factor metadata
        c.execute("""CREATE TABLE IF NOT EXISTS factors (
            id INTEGER PRIMARY KEY,
            factor_name TEXT UNIQUE,
            factor_type TEXT,
            created_at TIMESTAMP
        )""")

        # Backtest runs table - stores individual backtest metrics
        c.execute("""CREATE TABLE IF NOT EXISTS backtest_runs (
            id INTEGER PRIMARY KEY,
            factor_id INTEGER,
            run_name TEXT,
            run_date TIMESTAMP,
            ic REAL,
            sharpe REAL,
            annual_return REAL,
            max_drawdown REAL,
            win_rate REAL,
            FOREIGN KEY (factor_id) REFERENCES factors(id)
        )""")

        # Loop results table - stores overall loop statistics
        c.execute("""CREATE TABLE IF NOT EXISTS loop_results (
            id INTEGER PRIMARY KEY,
            loop_index INTEGER,
            factors_success INTEGER,
            factors_fail INTEGER,
            success_rate REAL,
            best_ic REAL,
            status TEXT
        )""")

        # Migrate schema: add new columns if they don't exist
        self._add_column_if_not_exists('backtest_runs', 'information_ratio', 'REAL')
        self._add_column_if_not_exists('backtest_runs', 'volatility', 'REAL')
        self._add_column_if_not_exists('backtest_runs', 'raw_metrics', 'TEXT')

        # Create indexes for common queries
        c.execute("""CREATE INDEX IF NOT EXISTS idx_backtest_ic ON backtest_runs(ic)""")
        c.execute("""CREATE INDEX IF NOT EXISTS idx_backtest_sharpe ON backtest_runs(sharpe)""")
        c.execute("""CREATE INDEX IF NOT EXISTS idx_backtest_date ON backtest_runs(run_date)""")

        self.conn.commit()

    def _add_column_if_not_exists(self, table: str, column: str, col_type: str) -> None:
        """
        Add a column to a table if it doesn't already exist.

        Parameters
        ----------
        table : str
            Table name
        column : str
            Column name to add
        col_type : str
            SQL column type (e.g., 'REAL', 'TEXT')
        """
        c = self.conn.cursor()
        try:
            # Try to query the column - if it fails, it doesn't exist
            # nosec B608: Internal schema migration, column names are controlled
            c.execute(f"SELECT {column} FROM {table} LIMIT 1")  # nosec B608
        except sqlite3.OperationalError:
            # Column doesn't exist, add it
            c.execute(f"ALTER TABLE {table} ADD COLUMN {column} {col_type}")  # nosec B608
    
    def add_factor(self, name: str, type: str = "unknown") -> int:
        c = self.conn.cursor()
        c.execute("INSERT OR IGNORE INTO factors (factor_name, factor_type, created_at) VALUES (?, ?, ?)", 
                  (name, type, datetime.now()))
        c.execute("SELECT id FROM factors WHERE factor_name = ?", (name,))
        self.conn.commit()
        result = c.fetchone()
        return result[0] if result else -1
    
    def add_backtest(self, factor_name: str, metrics: Dict) -> int:
        """
        Add a backtest result to the database.

        Parameters
        ----------
        factor_name : str
            Name of the factor (max 100 chars)
        metrics : Dict
            Dictionary containing metrics like ic, sharpe_ratio, etc.

        Returns
        -------
        int
            The ID of the inserted backtest run
        """
        factor_id = self.add_factor(factor_name)
        if factor_id <= 0:
            return -1

        c = self.conn.cursor()

        # Serialize raw_metrics if present
        raw_metrics_json = None
        if "raw_metrics" in metrics:
            try:
                # Convert all values to native Python types for JSON serialization
                raw = metrics["raw_metrics"]
                raw_metrics_json = json.dumps({
                    k: (float(v) if v is not None else None)
                    for k, v in raw.items()
                })
            except Exception:
                raw_metrics_json = None

        c.execute(
            """INSERT INTO backtest_runs
            (factor_id, run_name, run_date, ic, sharpe, annual_return, max_drawdown,
             win_rate, information_ratio, volatility, raw_metrics)
            VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)""",
            (
                factor_id,
                f"{factor_name}_{datetime.now().strftime('%Y%m%d_%H%M%S')}",
                datetime.now(),
                metrics.get('ic'),
                metrics.get('sharpe_ratio'),
                metrics.get('annualized_return'),
                metrics.get('max_drawdown'),
                metrics.get('win_rate'),
                metrics.get('information_ratio'),
                metrics.get('volatility'),
                raw_metrics_json,
            )
        )
        self.conn.commit()
        return c.lastrowid
    
    def add_loop(self, loop_idx: int, success: int, fail: int, best_ic: float = None, status: str = "completed") -> int:
        c = self.conn.cursor()
        rate = success / (success + fail) if (success + fail) > 0 else 0
        c.execute("""INSERT INTO loop_results (loop_index, factors_success, factors_fail, success_rate, best_ic, status)
            VALUES (?, ?, ?, ?, ?, ?)""", (loop_idx, success, fail, rate, best_ic, status))
        self.conn.commit()
        return c.lastrowid
    
    def get_top_factors(self, metric: str = 'sharpe', limit: int = 20) -> pd.DataFrame:
        """
        Get top performing factors sorted by specified metric.

        Parameters
        ----------
        metric : str
            Metric to sort by ('sharpe', 'ic', 'annual_return', 'max_drawdown')
        limit : int
            Number of top factors to return

        Returns
        -------
        pd.DataFrame
            DataFrame with factor names and metrics
        """
        # Map shorthand to full column name
        metric_map = {
            'sharpe': 'sharpe',
            'ic': 'ic',
            'return': 'annual_return',
            'drawdown': 'max_drawdown',
            'win_rate': 'win_rate',
            'information_ratio': 'information_ratio',
        }
        col = metric_map.get(metric, metric)

        return pd.read_sql_query(
            f"""SELECT factor_name, ic, sharpe, annual_return, max_drawdown,
                       win_rate, information_ratio, volatility, run_date
                FROM backtest_runs
                JOIN factors ON factor_id = factors.id
                WHERE {col} IS NOT NULL
                ORDER BY {col} DESC
                LIMIT ?""",
            self.conn,
            params=[limit]
        )

    def get_factor_history(self, factor_name: str) -> pd.DataFrame:
        """
        Get all backtest runs for a specific factor.

        Parameters
        ----------
        factor_name : str
            Name of the factor

        Returns
        -------
        pd.DataFrame
            DataFrame with all runs for the factor
        """
        return pd.read_sql_query(
            """SELECT factor_name, ic, sharpe, annual_return, max_drawdown,
                      win_rate, information_ratio, run_date, run_name
               FROM backtest_runs
               JOIN factors ON factor_id = factors.id
               WHERE factor_name = ?
               ORDER BY run_date DESC""",
            self.conn,
            params=[factor_name]
        )

    def get_aggregate_stats(self) -> Dict:
        """
        Get aggregate statistics across all factors.

        Returns
        -------
        Dict
            Dictionary with total_factors, avg_ic, max_sharpe, avg_return
        """
        c = self.conn.cursor()
        c.execute(
            """SELECT COUNT(DISTINCT factor_name), AVG(ic), MAX(sharpe), AVG(annual_return)
               FROM backtest_runs
               JOIN factors ON factor_id = factors.id"""
        )
        r = c.fetchone()
        return {
            'total_factors': r[0],
            'avg_ic': r[1],
            'max_sharpe': r[2],
            'avg_return': r[3]
        }

    def get_all_results(self) -> pd.DataFrame:
        """
        Get all backtest results with factor details.

        Returns
        -------
        pd.DataFrame
            DataFrame with all backtest results
        """
        return pd.read_sql_query(
            """SELECT f.id as factor_id, f.factor_name, f.factor_type, f.created_at,
                      b.id as run_id, b.run_name, b.run_date,
                      b.ic, b.sharpe, b.annual_return, b.max_drawdown,
                      b.win_rate, b.information_ratio, b.volatility,
                      b.raw_metrics
               FROM factors f
               LEFT JOIN backtest_runs b ON f.id = b.factor_id
               ORDER BY b.run_date DESC""",
            self.conn
        )
    
    def close(self):
        self.conn.close()

if __name__ == "__main__":
    print("=== DB Test ===")
    db = ResultsDatabase()
    db.add_factor("TestFactor", "Momentum")
    db.add_backtest("TestFactor", {'ic': 0.05, 'sharpe_ratio': 1.5, 'annualized_return': 0.15, 'max_drawdown': -0.08, 'win_rate': 0.55})
    db.add_loop(1, 4, 6, 0.05, "completed")
    
    print("Top Faktoren:")
    print(db.get_top_factors())
    print("\nAggregate Stats:")
    print(db.get_aggregate_stats())
    db.close()
    print("✅ Test bestanden!")
