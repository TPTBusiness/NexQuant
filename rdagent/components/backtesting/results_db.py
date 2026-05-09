"""
NexQuant Results Database - SQLite für Backtest-Ergebnisse

Stores backtest metrics from Qlib/MLflow runs for querying and dashboard display.
"""
import sqlite3
import json
import pandas as pd
from pathlib import Path
from datetime import datetime
from typing import Dict, Optional, List


class ResultsDatabase:
    def __init__(self, db_path: Optional[str] = None):
        if db_path is None:
            # Go up from rdagent/components/backtesting/ to project root (4 levels)
            project_root = Path(__file__).parent.parent.parent.parent
            db_path = str(project_root / "results" / "db" / "backtest_results.db")
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

    _ALLOWED_TABLES = frozenset({"factors", "backtest_runs", "loop_results"})
    _ALLOWED_COL_TYPES = frozenset({"REAL", "TEXT", "INTEGER", "BLOB"})

    def _add_column_if_not_exists(self, table: str, column: str, col_type: str) -> None:
        """
        Add a column to a table if it doesn't already exist.

        Parameters
        ----------
        table : str
            Table name (must be in _ALLOWED_TABLES)
        column : str
            Column name to add (alphanumeric + underscore only)
        col_type : str
            SQL column type (must be in _ALLOWED_COL_TYPES)
        """
        if table not in self._ALLOWED_TABLES:
            raise ValueError(f"Unknown table: {table!r}")
        if not column.replace("_", "").isalnum():
            raise ValueError(f"Invalid column name: {column!r}")
        if col_type not in self._ALLOWED_COL_TYPES:
            raise ValueError(f"Invalid column type: {col_type!r}")

        c = self.conn.cursor()
        c.execute("SELECT name FROM pragma_table_info(?)", (table,))
        existing = {row[0] for row in c.fetchall()}
        if column not in existing:
            c.execute(f"ALTER TABLE {table} ADD COLUMN {column} {col_type}")
    
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
    
    def add_loop(self, loop_idx: int, success: int, fail: int, best_ic: float | None = None, status: str = "completed") -> int:
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
        _ALLOWED_METRICS = frozenset({
            'sharpe', 'ic', 'annual_return', 'max_drawdown',
            'win_rate', 'information_ratio', 'volatility',
        })
        metric_map = {
            'sharpe': 'sharpe', 'ic': 'ic', 'return': 'annual_return',
            'drawdown': 'max_drawdown', 'win_rate': 'win_rate',
            'information_ratio': 'information_ratio',
        }
        col = metric_map.get(metric, metric)
        if col not in _ALLOWED_METRICS:
            raise ValueError(f"Unknown metric: {metric!r}")

        return pd.read_sql_query(
            f"""SELECT factor_name, ic, sharpe, annual_return, max_drawdown,
                       win_rate, information_ratio, volatility, run_date
                FROM backtest_runs
                JOIN factors ON factor_id = factors.id
                WHERE {col} IS NOT NULL
                ORDER BY {col} DESC
                LIMIT ?""",  # nosec B608 — col is validated against _ALLOWED_METRICS above
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
    
    def generate_results_summary(self, output_path: Optional[str] = None,
                                  print_to_console: bool = True) -> Dict:
        """
        Generate a comprehensive results summary report.

        Scans the database and factors directory to produce a summary report
        with total runs, successful/failed counts, best metrics, and statistics.

        Parameters
        ----------
        output_path : str, optional
            Path to write the summary as Markdown file.
            Defaults to results/RESULTS_SUMMARY.md
        print_to_console : bool
            If True, print the summary to console

        Returns
        -------
        Dict
            Summary statistics dictionary
        """
        # Database stats
        db_stats = self.get_aggregate_stats()

        # Get all results
        all_results = self.get_all_results()

        # Top factors
        top_sharpe = self.get_top_factors('sharpe', limit=10)
        top_ic = self.get_top_factors('ic', limit=10)

        # Count successful vs failed runs
        total_runs = len(all_results)
        runs_with_ic = int(all_results['ic'].notna().sum()) if total_runs > 0 else 0
        runs_with_sharpe = int(all_results['sharpe'].notna().sum()) if total_runs > 0 else 0

        # Count unique factors
        unique_factors = all_results['factor_name'].nunique() if total_runs > 0 else 0

        # Best metrics
        best_ic = all_results['ic'].max() if total_runs > 0 and all_results['ic'].notna().any() else None
        best_sharpe = all_results['sharpe'].max() if total_runs > 0 and all_results['sharpe'].notna().any() else None
        best_return = all_results['annual_return'].max() if total_runs > 0 and all_results['annual_return'].notna().any() else None
        worst_drawdown = all_results['max_drawdown'].min() if total_runs > 0 and all_results['max_drawdown'].notna().any() else None

        # Scan factors directory for JSON files
        factors_dir = Path(__file__).parent.parent.parent.parent / "results" / "factors"
        json_factor_files = 0
        if factors_dir.exists():
            json_factor_files = len(list(factors_dir.glob("*.json")))

        # Scan failed runs
        failed_dir = Path(__file__).parent.parent.parent.parent / "results" / "failed_runs"
        failed_runs_file = failed_dir / "failed_runs.json"
        failed_runs_count = 0
        failed_runs_data = []
        if failed_runs_file.exists():
            try:
                failed_runs_data = json.loads(failed_runs_file.read_text(encoding="utf-8"))
                if isinstance(failed_runs_data, list):
                    failed_runs_count = len(failed_runs_data)
                elif isinstance(failed_runs_data, dict):
                    failed_runs_count = 1
            except (json.JSONDecodeError, Exception):
                failed_runs_count = 0

        # Build summary
        summary = {
            "generated_at": datetime.now().isoformat(),
            "database_path": str(self.db_path),
            "overview": {
                "total_runs": int(total_runs),
                "unique_factors": int(unique_factors),
                "runs_with_ic": int(runs_with_ic),
                "runs_with_sharpe": int(runs_with_sharpe),
                "json_factor_files": json_factor_files,
                "failed_runs": failed_runs_count,
            },
            "best_metrics": {
                "best_ic": float(best_ic) if best_ic is not None else None,
                "best_sharpe": float(best_sharpe) if best_sharpe is not None else None,
                "best_annual_return": float(best_return) if best_return is not None else None,
                "worst_drawdown": float(worst_drawdown) if worst_drawdown is not None else None,
            },
            "aggregate_stats": db_stats,
            "top_10_by_sharpe": top_sharpe.to_dict(orient='records') if len(top_sharpe) > 0 else [],
            "top_10_by_ic": top_ic.to_dict(orient='records') if len(top_ic) > 0 else [],
        }

        # Write to Markdown
        if output_path is None:
            # Go up from rdagent/components/backtesting/ to project root (4 levels)
            project_root = Path(__file__).parent.parent.parent.parent
            output_path = str(project_root / "results" / "RESULTS_SUMMARY.md")

        self._write_summary_markdown(summary, output_path)

        if print_to_console:
            self._print_summary_console(summary)

        return summary

    def _fmt_float(self, value, fmt: str = ".4f") -> str:
        """Format float value, returning 'N/A' for None."""
        if value is None:
            return "N/A"
        return f"{value:{fmt}}"

    def _write_summary_markdown(self, summary: Dict, output_path: str) -> None:
        """Write summary as Markdown file."""
        path = Path(output_path)
        path.parent.mkdir(parents=True, exist_ok=True)

        overview = summary['overview']
        best = summary['best_metrics']
        top_sharpe = summary['top_10_by_sharpe']
        top_ic = summary['top_10_by_ic']

        # Pre-format best metrics for display
        best_ic_str = self._fmt_float(best['best_ic'], ".6f")
        best_sharpe_str = self._fmt_float(best['best_sharpe'], ".4f")
        best_return_str = self._fmt_float(best['best_annual_return'], ".4f")
        worst_dd_str = self._fmt_float(best['worst_drawdown'], ".4f")

        md_lines = [
            "# NexQuant Results Summary",
            "",
            f"**Generated:** {summary['generated_at']}",
            f"**Database:** `{summary['database_path']}`",
            "",
            "## Overview",
            "",
            f"| Metric | Value |",
            f"|--------|-------|",
            f"| Total Runs | {overview['total_runs']} |",
            f"| Unique Factors | {overview['unique_factors']} |",
            f"| Runs with IC | {overview['runs_with_ic']} |",
            f"| Runs with Sharpe | {overview['runs_with_sharpe']} |",
            f"| JSON Factor Files | {overview['json_factor_files']} |",
            f"| Failed Runs | {overview['failed_runs']} |",
            "",
            "## Best Metrics",
            "",
            f"| Metric | Value |",
            f"|--------|-------|",
            f"| Best IC | {best_ic_str} |",
            f"| Best Sharpe | {best_sharpe_str} |",
            f"| Best Annual Return | {best_return_str} |",
            f"| Worst Drawdown | {worst_dd_str} |",
            "",
        ]

        if top_sharpe:
            md_lines.extend([
                "## Top 10 by Sharpe Ratio",
                "",
                "| # | Factor | Sharpe | IC | Annual Return | Max Drawdown |",
                "|---|--------|--------|-----|---------------|--------------|",
            ])
            for i, row in enumerate(top_sharpe[:10], 1):
                md_lines.append(
                    f"| {i} | {row.get('factor_name', 'N/A')[:50]} | "
                    f"{self._fmt_float(row.get('sharpe'), '.4f')} | "
                    f"{self._fmt_float(row.get('ic'), '.6f')} | "
                    f"{self._fmt_float(row.get('annual_return'), '.4f')} | "
                    f"{self._fmt_float(row.get('max_drawdown'), '.4f')} |"
                )
            md_lines.append("")

        if top_ic:
            md_lines.extend([
                "## Top 10 by IC",
                "",
                "| # | Factor | IC | Sharpe | Annual Return |",
                "|---|--------|-----|--------|---------------|",
            ])
            for i, row in enumerate(top_ic[:10], 1):
                md_lines.append(
                    f"| {i} | {row.get('factor_name', 'N/A')[:50]} | "
                    f"{self._fmt_float(row.get('ic'), '.6f')} | "
                    f"{self._fmt_float(row.get('sharpe'), '.4f')} | "
                    f"{self._fmt_float(row.get('annual_return'), '.4f')} |"
                )
            md_lines.append("")

        md_lines.extend([
            "## Failed Runs",
            "",
            f"Total failed runs tracked: **{overview['failed_runs']}**",
            "",
            "Failed runs are stored in `results/failed_runs/failed_runs.json`.",
            "",
        ])

        path.write_text("\n".join(md_lines), encoding="utf-8")

    def _print_summary_console(self, summary: Dict) -> None:
        """Print summary to console."""
        overview = summary['overview']
        best = summary['best_metrics']

        print("\n" + "=" * 70)
        print("  PREDIX RESULTS SUMMARY")
        print("=" * 70)
        print(f"  Generated:    {summary['generated_at']}")
        print(f"  Database:     {summary['database_path']}")
        print("-" * 70)
        print(f"  Total Runs:        {overview['total_runs']}")
        print(f"  Unique Factors:    {overview['unique_factors']}")
        print(f"  Runs with IC:      {overview['runs_with_ic']}")
        print(f"  Runs with Sharpe:  {overview['runs_with_sharpe']}")
        print(f"  JSON Factor Files: {overview['json_factor_files']}")
        print(f"  Failed Runs:       {overview['failed_runs']}")
        print("-" * 70)
        print(f"  Best IC:           {self._fmt_float(best['best_ic'], '.6f')}")
        print(f"  Best Sharpe:       {self._fmt_float(best['best_sharpe'], '.4f')}")
        print(f"  Best Ann. Return:  {self._fmt_float(best['best_annual_return'], '.4f')}")
        print(f"  Worst Drawdown:    {self._fmt_float(best['worst_drawdown'], '.4f')}")
        print("=" * 70 + "\n")

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
