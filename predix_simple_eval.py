"""
Predix Simple Factor Evaluator - Direct IC/Sharpe computation.

Evaluates existing factor results by computing IC and Sharpe directly
from factor values and forward returns, without Qlib infrastructure.

Usage:
    python predix_simple_eval.py --top 100    # Evaluate top 100 factors
    python predix_simple_eval.py --all        # Evaluate all
    python predix_simple_eval.py --parallel 4 # 4 parallel workers
"""

import json
import os
import sys
import time
from concurrent.futures import ProcessPoolExecutor, as_completed
from dataclasses import dataclass, field
from datetime import datetime
from pathlib import Path
from typing import List, Optional

import numpy as np
import pandas as pd
from rich.console import Console
from rich.progress import (
    Progress,
    SpinnerColumn,
    TextColumn,
    BarColumn,
    TaskProgressColumn,
    TimeElapsedColumn,
)
from rich.table import Table
from rich.panel import Panel

console = Console()

# ---------------------------------------------------------------------------
# Configuration
# ---------------------------------------------------------------------------
PROJECT_ROOT = Path(__file__).resolve().parent
WORKSPACE_DIR = PROJECT_ROOT / "git_ignore_folder" / "RD-Agent_workspace"
RESULTS_DIR = PROJECT_ROOT / "results"
BACKTESTS_DIR = RESULTS_DIR / "backtests"
DB_DIR = RESULTS_DIR / "db"
DB_PATH = DB_DIR / "backtest_results.db"
EVAL_SUMMARY_PATH = RESULTS_DIR / "eval_summary.json"

# Ensure directories exist
BACKTESTS_DIR.mkdir(parents=True, exist_ok=True)
DB_DIR.mkdir(parents=True, exist_ok=True)


# ---------------------------------------------------------------------------
# Data structures
# ---------------------------------------------------------------------------
@dataclass
class FactorWorkspace:
    """Represents a factor workspace with code and results."""
    workspace_hash: str
    factor_name: str
    workspace_path: Path
    result_path: Optional[Path] = None
    data_path: Optional[Path] = None


@dataclass
class EvalResult:
    """Evaluation result for a single factor."""
    factor_name: str
    workspace_hash: str
    status: str  # success, failed
    ic: Optional[float] = None
    rank_ic: Optional[float] = None
    sharpe: Optional[float] = None
    annualized_return: Optional[float] = None
    max_drawdown: Optional[float] = None
    win_rate: Optional[float] = None
    non_null_count: int = 0
    total_count: int = 0
    error_message: Optional[str] = None
    timestamp: str = field(default_factory=lambda: datetime.now().isoformat())

    def to_dict(self) -> dict:
        return {k: v for k, v in self.__dict__.items()}


# ---------------------------------------------------------------------------
# Workspace scanner
# ---------------------------------------------------------------------------
def scan_workspaces(workspace_dir: Path) -> List[FactorWorkspace]:
    """Scan workspace directories for factors with results."""
    workspaces = []
    for ws in workspace_dir.iterdir():
        if not ws.is_dir():
            continue
        result_file = ws / "result.h5"
        data_file = ws / "intraday_pv.h5"
        if not result_file.exists() or not data_file.exists():
            continue

        # Read factor name from result.h5
        try:
            result = pd.read_hdf(str(result_file), key="data")
            if result is not None and len(result.columns) > 0:
                factor_name = result.columns[0]
                workspaces.append(FactorWorkspace(
                    workspace_hash=ws.name,
                    factor_name=factor_name,
                    workspace_path=ws,
                    result_path=result_file,
                    data_path=data_file,
                ))
        except Exception:
            continue

    return workspaces


# ---------------------------------------------------------------------------
# Factor evaluator
# ---------------------------------------------------------------------------
def evaluate_factor(ws: FactorWorkspace, forward_return_bars: int = 96) -> EvalResult:
    """
    Evaluate a factor by computing IC and Sharpe from factor values.

    Parameters
    ----------
    ws : FactorWorkspace
        Workspace with result.h5 and intraday_pv.h5
    forward_return_bars : int
        Number of bars for forward return calculation (96 = 96 minutes for 1min data)

    Returns
    -------
    EvalResult
    """
    try:
        # Load data
        df = pd.read_hdf(str(ws.data_path), key="data")
        result = pd.read_hdf(str(ws.result_path), key="data")

        total_count = len(result)
        factor_val = result.iloc[:, 0]
        non_null_count = factor_val.notna().sum()

        # Skip if too few valid values
        if non_null_count < 100:
            return EvalResult(
                factor_name=ws.factor_name,
                workspace_hash=ws.workspace_hash,
                status="failed",
                non_null_count=non_null_count,
                total_count=total_count,
                error_message=f"Too few valid values: {non_null_count}",
            )

        # Compute forward returns
        # Handle column name escaping
        col_close = "$close"
        if col_close not in df.columns:
            # Try alternative column name
            col_close = next((c for c in df.columns if "close" in c.lower()), None)
            if col_close is None:
                return EvalResult(
                    factor_name=ws.factor_name,
                    workspace_hash=ws.workspace_hash,
                    status="failed",
                    non_null_count=non_null_count,
                    total_count=total_count,
                    error_message=f"No close column found. Columns: {list(df.columns)}",
                )

        close = df[col_close]
        forward_ret = close.groupby(level="instrument").shift(-forward_return_bars) / close - 1

        # Compute IC (Information Coefficient)
        valid_idx = factor_val.dropna().index.intersection(forward_ret.dropna().index)
        if len(valid_idx) < 100:
            return EvalResult(
                factor_name=ws.factor_name,
                workspace_hash=ws.workspace_hash,
                status="failed",
                non_null_count=non_null_count,
                total_count=total_count,
                error_message=f"Too little overlap: {len(valid_idx)}",
            )

        ic = factor_val.loc[valid_idx].corr(forward_ret.loc[valid_idx])
        rank_ic = factor_val.loc[valid_idx].corr(forward_ret.loc[valid_idx], method="spearman")

        # Compute factor-level Sharpe (mean/std of factor values)
        factor_mean = factor_val.loc[valid_idx].mean()
        factor_std = factor_val.loc[valid_idx].std()
        sharpe = factor_mean / factor_std if factor_std > 0 else 0

        # Annualized return (assuming 252 trading days, 1440 minutes per day)
        ann_factor = np.sqrt(252 * 1440 / forward_return_bars)
        annualized_return = float(factor_mean * ann_factor * 100)  # in percent

        # Max drawdown approximation (cumulative factor performance)
        cum_perf = factor_val.loc[valid_idx].cumsum()
        running_max = cum_perf.expanding().max()
        drawdown = (cum_perf - running_max) / running_max.replace(0, np.nan)
        max_drawdown = float(drawdown.min()) if len(drawdown) > 0 else 0

        # Win rate (percentage of positive factor values)
        win_rate = float((factor_val.loc[valid_idx] > 0).sum()) / len(valid_idx)

        return EvalResult(
            factor_name=ws.factor_name,
            workspace_hash=ws.workspace_hash,
            status="success",
            ic=float(ic) if ic is not None else None,
            rank_ic=float(rank_ic) if rank_ic is not None else None,
            sharpe=float(sharpe),
            annualized_return=annualized_return,
            max_drawdown=max_drawdown,
            win_rate=win_rate,
            non_null_count=non_null_count,
            total_count=total_count,
        )

    except Exception as e:
        return EvalResult(
            factor_name=ws.factor_name,
            workspace_hash=ws.workspace_hash,
            status="failed",
            error_message=str(e)[:500],
        )


# ---------------------------------------------------------------------------
# Parallel evaluation
# ---------------------------------------------------------------------------
def run_evaluation(
    workspaces: List[FactorWorkspace],
    n_workers: int = 4,
) -> List[EvalResult]:
    """Run factor evaluation in parallel."""
    results = []

    with Progress(
        SpinnerColumn(),
        TextColumn("[progress.description]{task.description}"),
        BarColumn(),
        TaskProgressColumn(),
        TimeElapsedColumn(),
        console=console,
    ) as progress:
        task = progress.add_task(f"Evaluating {len(workspaces)} factors...", total=len(workspaces))

        with ProcessPoolExecutor(max_workers=n_workers) as executor:
            futures = {executor.submit(evaluate_factor, ws): ws for ws in workspaces}

            for future in as_completed(futures):
                ws = futures[future]
                try:
                    result = future.result(timeout=300)
                    results.append(result)
                except Exception as e:
                    results.append(EvalResult(
                        factor_name=ws.factor_name,
                        workspace_hash=ws.workspace_hash,
                        status="failed",
                        error_message=f"Timeout/Exception: {str(e)[:300]}",
                    ))

                n_success = sum(1 for r in results if r.status == "success")
                n_fail = sum(1 for r in results if r.status == "failed")
                progress.update(
                    task,
                    advance=1,
                    description=f"Evaluating: {n_success}✅ {n_fail}❌ | {ws.factor_name[:40]}",
                )

    return results


# ---------------------------------------------------------------------------
# Results storage
# ---------------------------------------------------------------------------
def save_results(results: List[EvalResult]) -> None:
    """Save evaluation results to JSON and SQLite."""
    # Save as JSON
    successful = [r for r in results if r.status == "success"]
    failed = [r for r in results if r.status == "failed"]

    # Sort by IC
    successful.sort(key=lambda r: abs(r.ic) if r.ic is not None else 0, reverse=True)

    # Save individual results
    for r in successful[:50]:  # Top 50
        json_path = BACKTESTS_DIR / f"{r.factor_name}_{r.workspace_hash}.json"
        with open(json_path, "w", encoding="utf-8") as f:
            json.dump(r.to_dict(), f, indent=2, default=str)

    # Save summary
    valid_ic = [r.ic for r in results if r.ic is not None]
    valid_sharpe = [r.sharpe for r in results if r.sharpe is not None]

    summary = {
        "generated_at": datetime.now().isoformat(),
        "total_evaluated": len(results),
        "successful": len(successful),
        "failed": len(failed),
        "success_rate": len(successful) / len(results) if results else 0,
        "avg_ic": float(np.mean(valid_ic)) if valid_ic else 0,
        "best_ic": float(max(valid_ic, key=abs)) if valid_ic else 0,
        "avg_sharpe": float(np.mean(valid_sharpe)) if valid_sharpe else 0,
        "best_sharpe": float(max(valid_sharpe)) if valid_sharpe else 0,
        "top_20_by_ic": [r.to_dict() for r in successful[:20]],
        "all_results": [r.to_dict() for r in results],
    }

    with open(EVAL_SUMMARY_PATH, "w", encoding="utf-8") as f:
        json.dump(summary, f, indent=2, default=str)

    # Save to SQLite
    try:
        import sqlite3
        conn = sqlite3.connect(str(DB_PATH))
        c = conn.cursor()
        c.execute("""CREATE TABLE IF NOT EXISTS factor_evaluations (
            id INTEGER PRIMARY KEY,
            factor_name TEXT,
            workspace_hash TEXT,
            ic REAL,
            rank_ic REAL,
            sharpe REAL,
            annualized_return REAL,
            max_drawdown REAL,
            win_rate REAL,
            non_null_count INTEGER,
            total_count INTEGER,
            status TEXT,
            timestamp TEXT
        )""")

        for r in results:
            c.execute("""INSERT INTO factor_evaluations
                (factor_name, workspace_hash, ic, rank_ic, sharpe,
                 annualized_return, max_drawdown, win_rate,
                 non_null_count, total_count, status, timestamp)
                VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)""",
                (r.factor_name, r.workspace_hash, r.ic, r.rank_ic, r.sharpe,
                 r.annualized_return, r.max_drawdown, r.win_rate,
                 r.non_null_count, r.total_count, r.status, r.timestamp))

        conn.commit()
        conn.close()
    except Exception as e:
        console.print(f"[yellow]SQLite save warning: {e}[/yellow]")


# ---------------------------------------------------------------------------
# Display
# ---------------------------------------------------------------------------
def display_results(results: List[EvalResult]) -> None:
    """Display evaluation results as a table."""
    successful = [r for r in results if r.status == "success"]
    successful.sort(key=lambda r: abs(r.ic) if r.ic is not None else 0, reverse=True)

    table = Table(
        title="Factor Evaluation Results",
        show_header=True,
        header_style="bold cyan",
    )
    table.add_column("#", justify="center", width=4)
    table.add_column("Factor", width=40)
    table.add_column("IC", justify="right", width=10)
    table.add_column("Rank IC", justify="right", width=10)
    table.add_column("Sharpe", justify="right", width=10)
    table.add_column("Ann. Ret %", justify="right", width=10)
    table.add_column("Max DD", justify="right", width=10)
    table.add_column("Win Rate", justify="right", width=10)

    for i, r in enumerate(successful[:20], 1):
        table.add_row(
            str(i),
            r.factor_name[:38],
            f"{r.ic:.6f}" if r.ic is not None else "N/A",
            f"{r.rank_ic:.6f}" if r.rank_ic is not None else "N/A",
            f"{r.sharpe:.4f}" if r.sharpe is not None else "N/A",
            f"{r.annualized_return:.4f}" if r.annualized_return is not None else "N/A",
            f"{r.max_drawdown:.4f}" if r.max_drawdown is not None else "N/A",
            f"{r.win_rate:.2%}" if r.win_rate is not None else "N/A",
        )

    console.print()
    console.print(table)

    # Summary
    valid_ic = [r.ic for r in results if r.ic is not None]
    valid_sharpe = [r.sharpe for r in results if r.sharpe is not None]

    console.print(Panel(
        f"[bold]Evaluation Summary[/bold]\n"
        f"Total evaluated: {len(results)}\n"
        f"Successful: {len(successful)} ✅\n"
        f"Failed: {len(results) - len(successful)} ❌\n"
        f"Avg IC: {np.mean(valid_ic):.6f} (n={len(valid_ic)})\n"
        f"Best IC: {max(valid_ic, key=abs, default=0):.6f}\n"
        f"Avg Sharpe: {np.mean(valid_sharpe):.4f} (n={len(valid_sharpe)})\n"
        f"Best Sharpe: {max(valid_sharpe, default=0):.4f}\n"
        f"Saved to: {EVAL_SUMMARY_PATH}\n"
        f"Database: {DB_PATH}",
        border_style="green",
    ))


# ---------------------------------------------------------------------------
# Entry point
# ---------------------------------------------------------------------------
def main(
    top: int = 100,
    all_factors: bool = False,
    parallel: int = 4,
) -> None:
    """Main entry point."""
    console.print(Panel(
        "[bold cyan]Predix Simple Factor Evaluator[/bold cyan]\n"
        f"Scanning workspaces for generated factors...",
        border_style="cyan",
    ))

    # Scan workspaces
    workspaces = scan_workspaces(WORKSPACE_DIR)
    console.print(f"\n[bold]Total workspaces with results: {len(workspaces)}[/bold]")

    if not workspaces:
        console.print("[red]No factors found![/red]")
        return

    # Select factors to evaluate
    if all_factors:
        to_evaluate = workspaces
    else:
        # Deduplicate by factor name, keep first occurrence
        seen = set()
        unique = []
        for ws in workspaces:
            if ws.factor_name not in seen:
                seen.add(ws.factor_name)
                unique.append(ws)

        # Sort by non-null count (prefer factors with more valid values)
        to_evaluate = sorted(unique, key=lambda ws: 0, reverse=True)[:top]

    console.print(f"[bold green]Selected {len(to_evaluate)} factors for evaluation[/bold green]")
    console.print(f"  Using {parallel} parallel workers")

    # Run evaluation
    results = run_evaluation(to_evaluate, n_workers=parallel)

    # Save results
    console.print(f"\n[bold cyan]Saving results...[/bold cyan]")
    save_results(results)

    # Display
    display_results(results)


if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser(
        description="Predix Simple Factor Evaluator - Direct IC/Sharpe computation"
    )
    parser.add_argument(
        "--top", "-n",
        type=int,
        default=100,
        help="Number of factors to evaluate (default: 100)",
    )
    parser.add_argument(
        "--all", "-a",
        action="store_true",
        help="Evaluate all discovered factors",
    )
    parser.add_argument(
        "--parallel", "-p",
        type=int,
        default=4,
        help="Number of parallel workers (default: 4)",
    )

    args = parser.parse_args()

    main(
        top=args.top,
        all_factors=args.all,
        parallel=args.parallel,
    )
