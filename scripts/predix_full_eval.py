"""
Predix Full Data Factor Evaluator - Evaluate factors with FULL 1min data.

Evaluates factors using the complete intraday_pv.h5 dataset (2022-2026, ~2.26M rows)
instead of the debug dataset (2024 only, ~371K rows).

Usage:
    python predix_full_eval.py --top 100    # Evaluate top 100 factors with full data
    python predix_full_eval.py --all        # Evaluate all factors
    python predix_full_eval.py --parallel 4 # 4 parallel workers
"""

import json
import os
import sys
import warnings
import time
from concurrent.futures import ThreadPoolExecutor, as_completed
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

# FULL data file (2022-2026, ~72MB)
FULL_DATA_FILE = PROJECT_ROOT / "git_ignore_folder" / "factor_implementation_source_data" / "intraday_pv.h5"

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
class FactorInfo:
    """Factor information."""
    workspace_hash: str
    factor_name: str
    factor_code: str


@dataclass
class EvalResult:
    """Evaluation result for a single factor."""
    factor_name: str
    workspace_hash: str
    factor_code: str = ""
    factor_description: str = ""
    status: str = ""  # success, failed
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
# Factor description extractor
# ---------------------------------------------------------------------------
def _extract_factor_description(code: str) -> str:
    """Extract docstring or description from factor code."""
    import re
    # Try to extract docstring
    match = re.search(r'"""(.*?)"""', code, re.DOTALL)
    if match:
        return match.group(1).strip()[:500]
    # Try to extract from comments
    lines = code.split('\n')
    desc_lines = []
    for line in lines[:20]:
        if line.strip().startswith('#') and not line.strip().startswith('#!'):
            desc_lines.append(line.strip()[1:].strip())
    if desc_lines:
        return ' '.join(desc_lines)[:500]
    return "No description available"


# ---------------------------------------------------------------------------
# Factor scanner
# ---------------------------------------------------------------------------
def scan_factors(workspace_dir: Path, skip_evaluated: bool = True) -> List[FactorInfo]:
    """Scan workspace directories for unique factor codes.

    Parameters
    ----------
    workspace_dir : Path
        Path to workspace directory
    skip_evaluated : bool
        If True, skip factors that already have valid results in results/factors/

    Returns
    -------
    List[FactorInfo]
        List of factor information
    """
    factors = []
    seen_names = set()

    # Load already evaluated factors (if skip_evaluated is True)
    evaluated_factors = set()
    if skip_evaluated:
        project_root = Path(__file__).parent
        factors_dir = project_root / "results" / "factors"
        if factors_dir.exists():
            import json
            import glob
            for f in glob.glob(str(factors_dir / "*.json")):
                try:
                    with open(f) as fh:
                        data = json.load(fh)
                    if data.get("status") == "success" and data.get("ic") is not None:
                        evaluated_factors.add(data.get("factor_name"))
                except Exception:
                    logging.debug("Exception caught", exc_info=True)
            print(f"  Found {len(evaluated_factors)} already evaluated factors - skipping")

    for ws in workspace_dir.iterdir():
        if not ws.is_dir():
            continue
        factor_file = ws / "factor.py"
        result_file = ws / "result.h5"
        if not factor_file.exists():
            continue

        # Read factor name from result.h5
        factor_name = None
        if result_file.exists():
            try:
                result = pd.read_hdf(str(result_file), key="data")
                if result is not None and len(result.columns) > 0:
                    factor_name = result.columns[0]
            except Exception:
                logging.debug("Exception caught", exc_info=True)

        if factor_name is None:
            # Try to extract from code
            code = factor_file.read_text()
            import re
            match = re.search(r'def calculate_(\w+)', code)
            if match:
                factor_name = match.group(1)
            else:
                factor_name = f"factor_{ws.name}"

        # Skip duplicates
        if factor_name in seen_names:
            continue

        # Skip already evaluated factors
        if skip_evaluated and factor_name in evaluated_factors:
            continue

        seen_names.add(factor_name)

        factors.append(FactorInfo(
            workspace_hash=ws.name,
            factor_name=factor_name,
            factor_code=factor_file.read_text(),
        ))

    return factors


# ---------------------------------------------------------------------------
# Look-ahead bias detection for daily-constant factors
# ---------------------------------------------------------------------------
def _shift_daily_constant_factor_if_needed(factor_col: "pd.Series", factor_name: str) -> "pd.Series":
    """Detect daily-constant factors (look-ahead bias) and shift by 1 trading day."""
    sample_days = factor_col.index.get_level_values("datetime").normalize().unique()
    if len(sample_days) < 10:
        return factor_col
    rng = np.random.default_rng(42)
    days_to_check = rng.choice(sample_days, size=min(50, len(sample_days)), replace=False)
    constant_count = 0
    for day in days_to_check:
        day_mask = factor_col.index.get_level_values("datetime").normalize() == day
        day_vals = factor_col[day_mask].dropna()
        if len(day_vals) == 0:
            continue
        if day_vals.nunique() == 1:
            constant_count += 1
    fraction_constant = constant_count / len(days_to_check)
    if fraction_constant < 0.90:
        return factor_col
    # Shift by 1 trading day per instrument
    import logging
    logging.getLogger(__name__).info(
        "Factor '%s' is %.0f%% daily-constant — shifting 1 trading day to fix look-ahead bias",
        factor_name, fraction_constant * 100,
    )
    instruments = factor_col.index.get_level_values("instrument").unique() if "instrument" in factor_col.index.names else [None]
    shifted_parts = []
    for instr in instruments:
        if instr is not None:
            mask = factor_col.index.get_level_values("instrument") == instr
            col_instr = factor_col[mask]
        else:
            col_instr = factor_col
        dates = col_instr.index.get_level_values("datetime").normalize()
        trading_days = dates.unique().sort_values()
        day_first = col_instr.groupby(dates).first()
        day_first_shifted = day_first.shift(1)
        day_first_shifted.index = pd.to_datetime(day_first_shifted.index)
        day_map = day_first_shifted.reindex(pd.to_datetime(trading_days)).values
        new_vals = pd.Series(
            day_map[np.searchsorted(trading_days.values, dates.values)],
            index=col_instr.index,
        )
        shifted_parts.append(new_vals)
    return pd.concat(shifted_parts).sort_index()


# ---------------------------------------------------------------------------
# Factor evaluator
# ---------------------------------------------------------------------------
def evaluate_factor_full(factor: FactorInfo, full_data: pd.DataFrame,
                         forward_return_bars: int = 96) -> EvalResult:
    """
    Evaluate a factor using the FULL dataset.

    Parameters
    ----------
    factor : FactorInfo
        Factor information with code
    full_data : pd.DataFrame
        Full intraday_pv.h5 data
    forward_return_bars : int
        Number of bars for forward return calculation

    Returns
    -------
    EvalResult
    """
    import tempfile
    import subprocess  # nosec B404

    with tempfile.TemporaryDirectory(prefix="predix_full_") as tmp_dir:
        ws = Path(tmp_dir)

        try:
            # Copy full data to temp workspace
            import shutil
            shutil.copy(str(FULL_DATA_FILE), str(ws / "intraday_pv.h5"))

            # Write factor code
            (ws / "factor.py").write_text(factor.factor_code, encoding="utf-8")

            # Execute factor code
            proc = subprocess.run(
                [sys.executable, str(ws / "factor.py")],
                cwd=str(ws),
                capture_output=True,
                text=True,
                timeout=120,
            )

            if proc.returncode != 0:
                return EvalResult(
                    factor_name=factor.factor_name,
                    workspace_hash=factor.workspace_hash,
                    status="failed",
                    error_message=f"Execution failed: {proc.stderr[:300]}",
                )

            # Read result
            result_file = ws / "result.h5"
            if not result_file.exists():
                return EvalResult(
                    factor_name=factor.factor_name,
                    workspace_hash=factor.workspace_hash,
                    status="failed",
                    error_message="No result.h5 generated",
                )

            result = pd.read_hdf(str(result_file), key="data")
            total_count = len(result)
            factor_val = result.iloc[:, 0]
            factor_val = _shift_daily_constant_factor_if_needed(factor_val, factor.factor_name)
            non_null_count = factor_val.notna().sum()

            if non_null_count < 1000:
                return EvalResult(
                    factor_name=factor.factor_name,
                    workspace_hash=factor.workspace_hash,
                    status="failed",
                    non_null_count=non_null_count,
                    total_count=total_count,
                    error_message=f"Too few valid values: {non_null_count}",
                )

            # Compute forward returns
            col_close = "$close"
            if col_close not in full_data.columns:
                col_close = next((c for c in full_data.columns if "close" in c.lower()), None)
                if col_close is None:
                    return EvalResult(
                        factor_name=factor.factor_name,
                        workspace_hash=factor.workspace_hash,
                        status="failed",
                        error_message=f"No close column found",
                    )

            close = full_data[col_close]
            forward_ret = close.groupby(level="instrument").shift(-forward_return_bars) / close - 1

            # Compute IC
            valid_idx = factor_val.dropna().index.intersection(forward_ret.dropna().index)
            if len(valid_idx) < 1000:
                return EvalResult(
                    factor_name=factor.factor_name,
                    workspace_hash=factor.workspace_hash,
                    status="failed",
                    non_null_count=non_null_count,
                    total_count=total_count,
                    error_message=f"Too little overlap: {len(valid_idx)}",
                )

            with warnings.catch_warnings():
                warnings.simplefilter("ignore")
                ic = factor_val.loc[valid_idx].corr(forward_ret.loc[valid_idx])
                rank_ic = factor_val.loc[valid_idx].corr(forward_ret.loc[valid_idx], method="spearman")

            # Compute Sharpe
            factor_mean = factor_val.loc[valid_idx].mean()
            factor_std = factor_val.loc[valid_idx].std()
            sharpe = factor_mean / factor_std if factor_std > 0 else 0

            # Annualized return
            ann_factor = np.sqrt(252 * 1440 / forward_return_bars)
            annualized_return = float(factor_mean * ann_factor * 100)

            # Max drawdown
            cum_perf = factor_val.loc[valid_idx].cumsum()
            running_max = cum_perf.expanding().max()
            drawdown = (cum_perf - running_max) / running_max.replace(0, np.nan)
            max_drawdown = float(drawdown.min()) if len(drawdown) > 0 else 0

            # Win rate
            win_rate = float((factor_val.loc[valid_idx] > 0).sum()) / len(valid_idx)

            return EvalResult(
                factor_name=factor.factor_name,
                workspace_hash=factor.workspace_hash,
                factor_code=factor.factor_code,
                factor_description=_extract_factor_description(factor.factor_code),
                status="success",
                ic=float(ic) if ic is not None and not np.isnan(ic) else None,
                rank_ic=float(rank_ic) if rank_ic is not None and not np.isnan(rank_ic) else None,
                sharpe=float(sharpe) if sharpe is not None and not np.isnan(sharpe) else None,
                annualized_return=annualized_return,
                max_drawdown=max_drawdown,
                win_rate=win_rate,
                non_null_count=non_null_count,
                total_count=total_count,
            )

        except subprocess.TimeoutExpired:
            return EvalResult(
                factor_name=factor.factor_name,
                workspace_hash=factor.workspace_hash,
                status="failed",
                error_message="Execution timeout (120s)",
            )
        except Exception as e:
            return EvalResult(
                factor_name=factor.factor_name,
                workspace_hash=factor.workspace_hash,
                status="failed",
                error_message=str(e)[:500],
            )


# ---------------------------------------------------------------------------
# Parallel evaluation
# ---------------------------------------------------------------------------
def run_evaluation(
    factors: List[FactorInfo],
    full_data: pd.DataFrame,
    n_workers: int = 4,
) -> List[EvalResult]:
    """Run factor evaluation in parallel using threads."""
    results = []

    with Progress(
        SpinnerColumn(),
        TextColumn("[progress.description]{task.description}"),
        BarColumn(),
        TaskProgressColumn(),
        TimeElapsedColumn(),
        console=console,
    ) as progress:
        task = progress.add_task(f"Evaluating {len(factors)} factors with FULL data...", total=len(factors))

        with ThreadPoolExecutor(max_workers=n_workers) as executor:
            futures = {executor.submit(evaluate_factor_full, f, full_data): f for f in factors}

            for future in as_completed(futures):
                factor = futures[future]
                result = None
                try:
                    result = future.result(timeout=300)
                    results.append(result)
                except Exception as e:
                    # Handle timeout and other exceptions
                    fname = str(getattr(factor, 'factor_name', 'unknown'))[:40]
                    results.append(EvalResult(
                        factor_name=fname,
                        workspace_hash=getattr(factor, 'workspace_hash', 'unknown'),
                        status="failed",
                        error_message=f"Exception: {str(e)[:300]}",
                    ))
                    result = results[-1]

                n_success = sum(1 for r in results if r.status == "success")
                n_fail = sum(1 for r in results if r.status == "failed")

                # Save immediately after each factor
                if result is not None:
                    save_single_result(result)

                # Update progress with safe string handling
                fname = str(getattr(factor, 'factor_name', 'unknown'))[:40]
                progress.update(
                    task,
                    advance=1,
                    description=f"Evaluating: {n_success}✅ {n_fail}❌ | {fname}",
                )

    return results


# ---------------------------------------------------------------------------
# Results storage
# ---------------------------------------------------------------------------
FACTORS_DIR = RESULTS_DIR / "factors"
FACTORS_DIR.mkdir(parents=True, exist_ok=True)

def save_single_result(r: EvalResult) -> None:
    """Save a single factor result to results/factors/."""
    if r.status != "success":
        return
    safe_name = r.factor_name.replace("/", "_").replace("\\", "_").replace(" ", "_")[:100]
    json_path = FACTORS_DIR / f"{safe_name}.json"
    with open(json_path, "w", encoding="utf-8") as f:
        json.dump(r.to_dict(), f, indent=2, default=str)

def save_results(results: List[EvalResult]) -> None:
    """Save evaluation results to JSON and SQLite."""
    successful = [r for r in results if r.status == "success"]
    failed = [r for r in results if r.status == "failed"]

    # Sort by IC
    successful.sort(key=lambda r: abs(r.ic) if r.ic is not None else 0, reverse=True)

    # Save ALL successful results to results/factors/
    for r in successful:
        # Safe filename (remove special chars)
        safe_name = r.factor_name.replace("/", "_").replace("\\", "_").replace(" ", "_")[:100]
        json_path = FACTORS_DIR / f"{safe_name}.json"
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
        "best_ic": float(max(valid_ic, key=abs, default=0)),
        "avg_sharpe": float(np.mean(valid_sharpe)) if valid_sharpe else 0,
        "best_sharpe": float(max(valid_sharpe, default=0)),
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
        title="Factor Evaluation Results (FULL DATA)",
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
        f"[bold]Evaluation Summary (FULL DATA)[/bold]\n"
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
    force: bool = False,
) -> None:
    """Main entry point."""
    console.print(Panel(
        "[bold cyan]Predix Full Data Factor Evaluator[/bold cyan]\n"
        f"Using FULL 1min data: {FULL_DATA_FILE}",
        border_style="cyan",
    ))

    # Load full data
    if not FULL_DATA_FILE.exists():
        console.print(f"[red]Full data file not found: {FULL_DATA_FILE}[/red]")
        return

    console.print(f"\n[dim]Loading full data...[/dim]")
    full_data = pd.read_hdf(str(FULL_DATA_FILE), key="data")
    console.print(f"[bold green]✓ Loaded {len(full_data):,} rows ({full_data.index.get_level_values('datetime').min()} to {full_data.index.get_level_values('datetime').max()})[/bold green]")

    # Scan factors (skip already evaluated by default)
    console.print(f"\n[dim]Scanning workspaces...[/dim]")
    factors = scan_factors(WORKSPACE_DIR, skip_evaluated=not force)
    console.print(f"[bold]Total unique factors found: {len(factors)}[/bold]")
    if force:
        console.print("[yellow]⚠️  Force mode: Re-evaluating ALL factors[/yellow]")
    else:
        console.print("[dim]Skipping already evaluated factors[/dim]")

    if not factors:
        console.print("[red]No factors found![/red]")
        return

    # Select factors to evaluate
    if all_factors:
        to_evaluate = factors
    else:
        to_evaluate = factors[:top]

    console.print(f"\n[bold green]Selected {len(to_evaluate)} factors for evaluation[/bold green]")
    console.print(f"  Using {parallel} parallel workers")

    # Run evaluation
    results = run_evaluation(to_evaluate, full_data, n_workers=parallel)

    # Save results
    console.print(f"\n[bold cyan]Saving results...[/bold cyan]")
    save_results(results)

    # Display
    display_results(results)


if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser(
        description="Predix Full Data Factor Evaluator"
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
    parser.add_argument(
        "--force", "-f",
        action="store_true",
        help="Force re-evaluation of ALL factors (even already evaluated)",
    )

    args = parser.parse_args()

    main(
        top=args.top,
        all_factors=args.all,
        parallel=args.parallel,
        force=args.force,
    )
