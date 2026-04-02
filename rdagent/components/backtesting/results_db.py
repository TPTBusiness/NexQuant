"""
Predix Results Database - SQLite für Backtest-Ergebnisse
"""
import sqlite3
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
        c = self.conn.cursor()
        c.execute("""CREATE TABLE IF NOT EXISTS factors (
            id INTEGER PRIMARY KEY, factor_name TEXT UNIQUE, factor_type TEXT, created_at TIMESTAMP)""")
        c.execute("""CREATE TABLE IF NOT EXISTS backtest_runs (
            id INTEGER PRIMARY KEY, factor_id INTEGER, run_name TEXT, run_date TIMESTAMP,
            ic REAL, sharpe REAL, annual_return REAL, max_drawdown REAL, win_rate REAL)""")
        c.execute("""CREATE TABLE IF NOT EXISTS loop_results (
            id INTEGER PRIMARY KEY, loop_index INTEGER, factors_success INTEGER, 
            factors_fail INTEGER, success_rate REAL, best_ic REAL, status TEXT)""")
        self.conn.commit()
    
    def add_factor(self, name: str, type: str = "unknown") -> int:
        c = self.conn.cursor()
        c.execute("INSERT OR IGNORE INTO factors (factor_name, factor_type, created_at) VALUES (?, ?, ?)", 
                  (name, type, datetime.now()))
        c.execute("SELECT id FROM factors WHERE factor_name = ?", (name,))
        self.conn.commit()
        result = c.fetchone()
        return result[0] if result else -1
    
    def add_backtest(self, factor_name: str, metrics: Dict) -> int:
        factor_id = self.add_factor(factor_name)
        c = self.conn.cursor()
        c.execute("""INSERT INTO backtest_runs 
            (factor_id, run_name, run_date, ic, sharpe, annual_return, max_drawdown, win_rate)
            VALUES (?, ?, ?, ?, ?, ?, ?, ?)""",
            (factor_id, f"{factor_name}_{datetime.now().strftime('%Y%m%d_%H%M%S')}", 
             datetime.now(), metrics.get('ic'), metrics.get('sharpe_ratio'),
             metrics.get('annualized_return'), metrics.get('max_drawdown'), metrics.get('win_rate')))
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
        return pd.read_sql_query(f"""SELECT factor_name, {metric}, ic, annual_return, max_drawdown 
            FROM backtest_runs JOIN factors ON factor_id = factors.id 
            WHERE {metric} IS NOT NULL ORDER BY {metric} DESC LIMIT ?""", 
            self.conn, params=[limit])
    
    def get_aggregate_stats(self) -> Dict:
        c = self.conn.cursor()
        c.execute("""SELECT COUNT(DISTINCT factor_name), AVG(ic), MAX(sharpe), AVG(annual_return) 
            FROM backtest_runs JOIN factors ON factor_id = factors.id""")
        r = c.fetchone()
        return {'total_factors': r[0], 'avg_ic': r[1], 'max_sharpe': r[2], 'avg_return': r[3]}
    
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
