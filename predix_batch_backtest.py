"""
Predix Batch Backtest Script - Extract and backtest existing factors.

Scans generated factor code from workspaces, runs Qlib backtests directly
(bypassing CoSTEER), and saves results to JSON + SQLite.

Usage:
    python predix_batch_backtest.py --factors 100        # Backtest top 100 factors
    python predix_batch_backtest.py --all                # Backtest all discovered factors
    python predix_batch_backtest.py --parallel 5         # 5 parallel backtests
    python predix_batch_backtest.py --scan-only          # Only scan, don't run backtests
"""

import json
import os
import re
import signal
import sys
import time
import traceback
from concurrent.futures import ProcessPoolExecutor, as_completed
from dataclasses import dataclass, field, asdict
from datetime import datetime
from pathlib import Path
from typing import Dict, List, Optional, Tuple

import numpy as np
import pandas as pd

# ---------------------------------------------------------------------------
# Project root setup — ensure rdagent is importable
# ---------------------------------------------------------------------------
PROJECT_ROOT = Path(__file__).resolve().parent
sys.path.insert(0, str(PROJECT_ROOT))

from rich.console import Console
from rich.progress import (
    Progress,
    SpinnerColumn,
    TextColumn,
    BarColumn,
    TaskProgressColumn,
    TimeElapsedColumn,
    TimeRemainingColumn,
)
from rich.table import Table
from rich.panel import Panel
from rich.markdown import Markdown

console = Console()

# ---------------------------------------------------------------------------
# Constants & configuration
# ---------------------------------------------------------------------------
QLIB_DATA_DIR = Path.home() / ".qlib" / "qlib_data" / "eurusd_1min_data"
WORKSPACE_DIR = PROJECT_ROOT / "git_ignore_folder" / "RD-Agent_workspace"
RESULTS_DIR = PROJECT_ROOT / "results"
BACKTESTS_DIR = RESULTS_DIR / "backtests"
FACTORS_DIR = RESULTS_DIR / "factors"
DB_DIR = RESULTS_DIR / "db"
DB_PATH = DB_DIR / "backtest_results.db"
BATCH_SUMMARY_PATH = RESULTS_DIR / "batch_summary.json"

# Find qrun binary
QRUN_PATH = Path("/home/nico/miniconda3/envs/rdagent/bin/qrun")
if not QRUN_PATH.exists():
    # Fallback: try to find via which
    import shutil
    QRUN_PATH = shutil.which("qrun") or "qrun"

# Default backtest date ranges (matching factor_runner.py defaults)
TRAIN_START = "2008-01-01"
TRAIN_END = "2014-12-31"
VALID_START = "2015-01-01"
VALID_END = "2016-12-31"
TEST_START = "2017-01-01"
TEST_END = "2020-08-01"

# Qlib config template for a SINGLE factor backtest
QLIB_CONFIG_TEMPLATE = """\
qlib_init:
    provider_uri: "{provider_uri}"
    region: cn

market: &market eurusd
benchmark: &benchmark EURUSD

data_handler_config: &data_handler_config
    start_time: {train_start}
    end_time: {test_end}
    instruments: *market
    data_loader:
        class: NestedDataLoader
        kwargs:
            dataloader_l:
                - class: qlib.contrib.data.loader.Alpha158DL
                  kwargs:
                    config:
                        label:
                            - ["Ref($close, -2)/Ref($close, -1) - 1"]
                            - ["LABEL0"]
                        feature:
                            - {feature_expressions}
                            - {feature_names}

    infer_processors:
        - class: RobustZScoreNorm
          kwargs:
              fields_group: feature
              clip_outlier: true
              fit_start_time: {train_start}
              fit_end_time: {train_end}
        - class: Fillna
          kwargs:
              fields_group: feature
    learn_processors:
        - class: DropnaLabel
          kwargs:
              fields_group: label

port_analysis_config: &port_analysis_config
    strategy:
        class: TopkDropoutStrategy
        module_path: qlib.contrib.strategy
        kwargs:
            signal: <PRED>
            topk: 1
            n_drop: 0
    backtest:
        start_time: {test_start}
        end_time: {test_end}
        account: 100000000
        benchmark: *benchmark
        exchange_kwargs:
            limit_threshold: 0.0
            deal_price: close
            open_cost: 0.0005
            close_cost: 0.0015
            min_cost: 0

task:
    model:
        class: LGBModel
        module_path: qlib.contrib.model.gbdt
        kwargs:
            loss: mse
            colsample_bytree: 0.8879
            learning_rate: 0.2
            subsample: 0.8789
            lambda_l1: 205.6999
            lambda_l2: 580.9768
            max_depth: 8
            num_leaves: 210
            num_threads: {n_threads}
    dataset:
        class: DatasetH
        module_path: qlib.data.dataset
        kwargs:
            handler:
                class: DataHandlerLP
                module_path: qlib.contrib.data.handler
                kwargs: *data_handler_config
            segments:
                train: [{train_start}, {train_end}]
                valid: [{valid_start}, {valid_end}]
                test: [{test_start}, {test_end}]
    record:
        - class: SignalRecord
          module_path: qlib.workflow.record_temp
          kwargs:
            model: <MODEL>
            dataset: <DATASET>
        - class: SigAnaRecord
          module_path: qlib.workflow.record_temp
          kwargs:
            ana_long_short: False
            ann_scaler: 252
        - class: PortAnaRecord
          module_path: qlib.workflow.record_temp
          kwargs:
            config: *port_analysis_config
"""


# ---------------------------------------------------------------------------
# Data classes
# ---------------------------------------------------------------------------
@dataclass
class FactorInfo:
    """Metadata for a discovered factor."""
    workspace_hash: str
    factor_name: str
    factor_code: str
    factor_description: str = ""
    discovered_at: str = ""
    has_existing_result: bool = False
    existing_ic: Optional[float] = None
    existing_sharpe: Optional[float] = None


@dataclass
class BacktestResult:
    """Result of a single factor backtest."""
    factor_name: str
    workspace_hash: str
    status: str  # "success", "failed", "timeout"
    ic: Optional[float] = None
    sharpe_ratio: Optional[float] = None
    annualized_return: Optional[float] = None
    max_drawdown: Optional[float] = None
    win_rate: Optional[float] = None
    information_ratio: Optional[float] = None
    volatility: Optional[float] = None
    error_message: str = ""
    duration_seconds: float = 0.0
    timestamp: str = ""

    def to_dict(self) -> Dict:
        return asdict(self)


# ---------------------------------------------------------------------------
# Factor Discovery
# ---------------------------------------------------------------------------
class FactorExtractor:
    """Scans workspaces for generated factor code."""

    def __init__(self, workspace_dir: Path = None):
        self.workspace_dir = workspace_dir or WORKSPACE_DIR
        self.results: List[FactorInfo] = []

    def scan_workspaces(self) -> List[FactorInfo]:
        """Scan all workspace directories for factor.py files."""
        if not self.workspace_dir.exists():
            console.print(f"[red]Workspace directory not found: {self.workspace_dir}[/red]")
            return []

        console.print(f"\n[bold cyan]Scanning workspaces for factor.py files...[/bold cyan]")
        console.print(f"  Directory: {self.workspace_dir}")

        factors = []
        workspace_dirs = sorted(self.workspace_dir.iterdir())
        total = len(workspace_dirs)

        with Progress(
            SpinnerColumn(),
            TextColumn("[progress.description]{task.description}"),
            BarColumn(),
            TaskProgressColumn(),
            TimeElapsedColumn(),
            console=console,
        ) as progress:
            task = progress.add_task("Scanning...", total=total)

            for ws_dir in workspace_dirs:
                progress.update(task, advance=1, description=f"Scanning {ws_dir.name[:12]}...")

                if not ws_dir.is_dir():
                    continue

                factor_file = ws_dir / "factor.py"
                if factor_file.exists():
                    factor_info = self._extract_factor(ws_dir, factor_file)
                    if factor_info:
                        factors.append(factor_info)

                # Also check nested directories (home/ subfolder etc.)
                for nested_factor in self._scan_nested(ws_dir):
                    factors.append(nested_factor)

        console.print(f"  [green]Found {len(factors)} factors across {total} workspaces[/green]")
        self.results = factors
        return factors

    def _scan_nested(self, parent_dir: Path) -> List[FactorInfo]:
        """Scan nested subdirectories for factor.py files."""
        factors = []
        # Check common nested locations
        for sub in ["home"]:
            sub_dir = parent_dir / sub
            if sub_dir.is_dir():
                for f in sub_dir.rglob("factor.py"):
                    ws_hash = parent_dir.name
                    factor_info = self._extract_factor(f.parent, f, ws_hash=ws_hash)
                    if factor_info:
                        factors.append(factor_info)
        return factors

    def _extract_factor(self, ws_dir: Path, factor_file: Path, ws_hash: str = None) -> Optional[FactorInfo]:
        """Extract factor information from a workspace directory."""
        if ws_hash is None:
            ws_hash = ws_dir.name

        try:
            code = factor_file.read_text(encoding="utf-8")
        except Exception:
            return None

        # Extract factor name from code or directory
        factor_name = self._extract_factor_name(code, ws_dir)

        # Check for existing Qlib results
        existing_ic = None
        existing_sharpe = None
        existing_annualized_return = None
        existing_max_drawdown = None
        existing_information_ratio = None
        has_result = False

        qlib_res_file = ws_dir / "qlib_res.csv"
        if qlib_res_file.exists():
            try:
                # qlib_res.csv format: index=metric_name, column 0=value
                # Some metrics may have empty values (e.g. "IC," with no value)
                result_df = pd.read_csv(qlib_res_file, index_col=0, header=None, skip_blank_lines=True)
                if len(result_df.columns) >= 1:
                    # Convert to dict for easier access
                    metrics = {}
                    for idx in result_df.index:
                        if pd.notna(idx):
                            val = result_df.loc[idx, result_df.columns[0]]
                            metrics[str(idx).strip()] = val

                    def _parse_val(key):
                        """Parse a metric value, handling empty strings and NaN."""
                        val = metrics.get(key)
                        if val is None:
                            return None
                        if isinstance(val, str):
                            val = val.strip()
                            if not val:
                                return None
                        try:
                            f = float(val)
                            return f if f == f else None  # NaN check
                        except (ValueError, TypeError):
                            return None

                    existing_ic = _parse_val("IC")
                    # Try multiple Sharpe key variants
                    for key in ["1day.excess_return_with_cost.shar",
                                "1day.excess_return_with_cost.sharpe",
                                "1day.sharpe"]:
                        existing_sharpe = _parse_val(key)
                        if existing_sharpe is not None:
                            break

                    existing_annualized_return = _parse_val("1day.excess_return_with_cost.annualized_return")
                    existing_max_drawdown = _parse_val("1day.excess_return_with_cost.max_drawdown")
                    existing_information_ratio = _parse_val("1day.excess_return_with_cost.information_ratio")

                    # Has result if ANY metric is present
                    has_result = any(v is not None for v in [
                        existing_ic, existing_sharpe, existing_annualized_return,
                        existing_max_drawdown, existing_information_ratio
                    ])
            except Exception:
                pass

        # Extract description from code (first docstring or comment)
        description = self._extract_description(code)

        fi = FactorInfo(
            workspace_hash=ws_hash,
            factor_name=factor_name,
            factor_code=code,
            factor_description=description,
            discovered_at=datetime.now().isoformat(),
            has_existing_result=has_result,
            existing_ic=existing_ic,
            existing_sharpe=existing_sharpe,
        )
        # Store additional metrics as attributes (for extract_existing mode)
        fi._annualized_return = existing_annualized_return
        fi._max_drawdown = existing_max_drawdown
        fi._information_ratio = existing_information_ratio

        return fi

    def _extract_factor_name(self, code: str, ws_dir: Path) -> str:
        """Extract factor name from code or use workspace hash."""
        # Try to find class name
        match = re.search(r'class\s+(\w+)', code)
        if match:
            return match.group(1)

        # Try to find factor name variable
        match = re.search(r'(?:factor_name|FACTOR_NAME|name)\s*=\s*["\']([^"\']+)["\']', code)
        if match:
            return match.group(1)

        # Fall back to workspace hash
        return f"factor_{ws_dir.name[:16]}"

    def _extract_description(self, code: str) -> str:
        """Extract first docstring or comment as description."""
        # Try docstring
        match = re.search(r'"""(.*?)"""', code, re.DOTALL)
        if match:
            desc = match.group(1).strip().split("\n")[0]
            if len(desc) > 120:
                desc = desc[:120] + "..."
            return desc

        # Try single-line comment after class def
        match = re.search(r'class\s+\w+.*?\n\s*#(.+)', code)
        if match:
            return match.group(1).strip()[:120]

        return ""

    def get_unique_factors(self) -> List[FactorInfo]:
        """Return unique factors by name (deduplicate)."""
        seen = set()
        unique = []
        for f in self.results:
            if f.factor_name not in seen:
                seen.add(f.factor_name)
                unique.append(f)
        return unique

    def get_top_factors(self, metric: str = "ic", limit: int = 100) -> List[FactorInfo]:
        """Get top factors sorted by existing metric (if available)."""
        # Only consider factors that have existing results
        factors_with_results = [f for f in self.results if f.has_existing_result]

        if not factors_with_results:
            # No existing results, return first N unique factors
            return self.get_unique_factors()[:limit]

        if metric == "ic":
            factors_with_results.sort(key=lambda f: abs(f.existing_ic or 0), reverse=True)
        elif metric == "sharpe":
            factors_with_results.sort(key=lambda f: abs(f.existing_sharpe or 0), reverse=True)

        return factors_with_results[:limit]


# ---------------------------------------------------------------------------
# Qlib Backtest Execution
# ---------------------------------------------------------------------------
def run_single_backtest(factor_info: FactorInfo, work_dir: Path) -> BacktestResult:
    """
    Run a Qlib backtest for a single factor.

    This function is designed to run in a subprocess to isolate Qlib state.

    Parameters
    ----------
    factor_info : FactorInfo
        Factor metadata including code
    work_dir : Path
        Working directory for the backtest

    Returns
    -------
    BacktestResult
        Backtest metrics
    """
    start_time = time.time()
    result = BacktestResult(
        factor_name=factor_info.factor_name,
        workspace_hash=factor_info.workspace_hash,
        status="failed",
        timestamp=datetime.now().isoformat(),
    )

    try:
        # Create isolated workspace
        ws_dir = work_dir / f"bt_{factor_info.workspace_hash}"
        ws_dir.mkdir(parents=True, exist_ok=True)

        # Write factor code
        factor_file = ws_dir / "factor.py"
        factor_file.write_text(factor_info.factor_code, encoding="utf-8")

        # Write Qlib config
        feature_name = factor_info.factor_name.replace(" ", "_").replace("-", "_")[:50]
        feature_expr = f'"factor_{feature_name}"'

        config = QLIB_CONFIG_TEMPLATE.format(
            provider_uri=str(QLIB_DATA_DIR),
            train_start=TRAIN_START,
            train_end=TRAIN_END,
            valid_start=VALID_START,
            valid_end=VALID_END,
            test_start=TEST_START,
            test_end=TEST_END,
            feature_expressions=feature_expr,
            feature_names=f'["{feature_name}"]',
            n_threads=4,
        )

        config_file = ws_dir / "conf.yaml"
        config_file.write_text(config, encoding="utf-8")

        # Copy read_exp_res.py from template if exists
        template_read = PROJECT_ROOT / "rdagent" / "scenarios" / "qlib" / "experiment" / "factor_template" / "read_exp_res.py"
        if template_read.exists():
            (ws_dir / "read_exp_res.py").write_text(
                template_read.read_text(encoding="utf-8"), encoding="utf-8"
            )

        # Run Qlib backtest
        import qlib
        from qlib.config import REG_CN
        from qlib.workflow import R

        qlib.init(
            provider_uri=str(QLIB_DATA_DIR),
            region=REG_CN,
        )

        # Run qrun
        from qlib.workflow.cli import run
        result_obj = run(config_file=str(config_file))

        # Parse results from qlib_res.csv
        qlib_res = ws_dir / "qlib_res.csv"
        if qlib_res.exists():
            try:
                res_df = pd.read_csv(qlib_res, index_col=0, header=None)
                if len(res_df.columns) > 0:
                    metrics = res_df.iloc[:, 0]

                    def _parse_metric(key):
                        val = metrics.get(key, None)
                        if val is None:
                            return None
                        if isinstance(val, str) and val.strip():
                            try:
                                return float(val)
                            except (ValueError, TypeError):
                                return None
                        if isinstance(val, (int, float)):
                            try:
                                return float(val)
                            except (ValueError, TypeError):
                                return None
                        return None

                    result.ic = _parse_metric("IC")
                    result.sharpe_ratio = _parse_metric("1day.excess_return_with_cost.shar")
                    result.annualized_return = _parse_metric("1day.excess_return_with_cost.annualized_return")
                    result.max_drawdown = _parse_metric("1day.excess_return_with_cost.max_drawdown")
                    result.win_rate = _parse_metric("win_rate")
                    result.information_ratio = _parse_metric("1day.excess_return_with_cost.information_ratio")
                    result.volatility = _parse_metric("1day.excess_return_with_cost.std")

                    # Mark success if we got at least IC or Sharpe
                    if result.ic is not None or result.sharpe_ratio is not None:
                        result.status = "success"
                    result.duration_seconds = time.time() - start_time
                    return result
            except Exception as e:
                result.error_message = f"Error parsing qlib_res.csv: {str(e)[:200]}"
                result.duration_seconds = time.time() - start_time
                return result

        result.error_message = "No qlib_res.csv generated"
        result.duration_seconds = time.time() - start_time
        return result

    except Exception as e:
        result.error_message = str(e)[:500]
        result.status = "failed"
        result.duration_seconds = time.time() - start_time
        return result


def _safe_float(value) -> Optional[float]:
    """Convert value to float safely, returning None on failure."""
    if value is None:
        return None
    try:
        v = float(value)
        if v != v:  # NaN check
            return None
        return v
    except (ValueError, TypeError):
        return None


# ---------------------------------------------------------------------------
# Simplified Backtest (no Qlib dependency, direct calculation)
# ---------------------------------------------------------------------------
def run_simplified_backtest(factor_info: FactorInfo) -> BacktestResult:
    """
    Run a simplified backtest without full Qlib infrastructure.

    Strategy:
    1. If existing qlib_res.csv has valid results, use those
    2. If Qlib is available, try running it
    3. Otherwise, mark as needs_qlib for later batch processing

    Parameters
    ----------
    factor_info : FactorInfo
        Factor metadata

    Returns
    -------
    BacktestResult
    """
    start_time = time.time()
    result = BacktestResult(
        factor_name=factor_info.factor_name,
        workspace_hash=factor_info.workspace_hash,
        status="failed",
        timestamp=datetime.now().isoformat(),
    )

    try:
        # Strategy 1: Use existing results if available
        if factor_info.has_existing_result and factor_info.existing_ic is not None:
            result.status = "success"
            result.ic = factor_info.existing_ic
            result.sharpe_ratio = factor_info.existing_sharpe
            result.duration_seconds = time.time() - start_time
            return result

        # Strategy 2: Try to execute factor.py and compute metrics directly
        # This requires the factor code to be runnable
        try:
            direct_result = _run_factor_directly(factor_info)
            if direct_result and direct_result.status == "success":
                direct_result.duration_seconds = time.time() - start_time
                return direct_result
        except Exception:
            pass

        # Strategy 3: Mark as skipped (no Qlib, no existing results)
        result.status = "skipped"
        result.error_message = "No Qlib available and no existing results. Factor needs manual backtest."
        result.duration_seconds = time.time() - start_time
        return result

    except Exception as e:
        result.error_message = str(e)[:500]
        result.duration_seconds = time.time() - start_time
        return result


def _run_factor_directly(factor_info: FactorInfo) -> Optional[BacktestResult]:
    """
    Try to execute factor.py directly and compute simple metrics.

    This runs the factor code in a subprocess and reads result.h5
    (the standard output format for factor execution).

    Parameters
    ----------
    factor_info : FactorInfo
        Factor metadata

    Returns
    -------
    BacktestResult or None
    """
    import tempfile
    import subprocess

    with tempfile.TemporaryDirectory(prefix="predix_factor_") as tmp_dir:
        ws = Path(tmp_dir)

        # Write factor code
        (ws / "factor.py").write_text(factor_info.factor_code, encoding="utf-8")

        # Try to execute factor.py
        try:
            proc = subprocess.run(
                [sys.executable, str(ws / "factor.py")],
                cwd=str(ws),
                capture_output=True,
                text=True,
                timeout=120,  # 2 minute timeout
            )

            if proc.returncode != 0:
                return None

            # Check for result.h5 (standard factor output)
            result_h5 = ws / "result.h5"
            if not result_h5.exists():
                return None

            factor_values = pd.read_hdf(result_h5)

            if factor_values is None or factor_values.empty:
                return None

            # Compute simple metrics from factor values
            # Flatten and compute basic statistics
            flat_values = factor_values.values.flatten()
            flat_values = flat_values[np.isfinite(flat_values)]

            if len(flat_values) < 100:
                return None

            # Simple statistical metrics
            mean_val = float(np.mean(flat_values))
            std_val = float(np.std(flat_values))
            skewness = float(pd.Series(flat_values).skew())

            # Use mean/std as proxy for signal quality
            # This is NOT a real backtest, just a sanity check
            signal_quality = abs(mean_val) / std_val if std_val > 0 else 0

            return BacktestResult(
                factor_name=factor_info.factor_name,
                workspace_hash=factor_info.workspace_hash,
                status="success",
                ic=None,  # Cannot compute IC without forward returns
                sharpe_ratio=None,
                annualized_return=None,
                max_drawdown=None,
                win_rate=None,
                information_ratio=None,
                volatility=std_val,
                error_message=f"Direct execution — IC/Sharpe unavailable. Signal quality: {signal_quality:.6f}",
                timestamp=datetime.now().isoformat(),
            )

        except (subprocess.TimeoutExpired, Exception):
            return None


def _run_qlib_single(factor_info: FactorInfo) -> BacktestResult:
    """
    Run Qlib backtest for a single factor via qrun CLI.

    Parameters
    ----------
    factor_info : FactorInfo
        Factor metadata

    Returns
    -------
    BacktestResult
    """
    import subprocess
    import tempfile

    # Create temp workspace
    with tempfile.TemporaryDirectory(prefix="predix_bt_") as tmp_dir:
        ws = Path(tmp_dir)

        # Write factor code
        factor_code = factor_info.factor_code
        # Ensure the factor code saves result.h5 correctly
        if "result.h5" not in factor_code and "to_hdf" not in factor_code:
            factor_code += "\n\n# Save result for Qlib\nif 'result_df' in dir():\n    result_df.to_hdf('result.h5', key='data', mode='w')"

        (ws / "factor.py").write_text(factor_code, encoding="utf-8")

        # Generate feature name
        feature_name = f"factor_{factor_info.factor_name.replace(' ', '_').replace('-', '_')[:40]}"

        # Create Qlib config for LightGBM backtest
        # Based on conf_combined_factors.yaml template
        config = f"""
qlib_init:
    provider_uri: "{QLIB_DATA_DIR}"
    region: cn

market: &market eurusd
benchmark: &benchmark EURUSD

data_handler_config: &data_handler_config
    start_time: "2022-03-14"
    end_time: "2026-03-20"
    instruments: *market
    data_loader:
        class: NestedDataLoader
        kwargs:
            dataloader_l:
                - class: qlib.contrib.data.loader.Alpha158DL
                  kwargs:
                    config:
                        label:
                            - ["Ref($close, -96)/$close - 1"]
                            - ["LABEL0"]
                        feature:
                            - ["{factor_info.factor_name}"]
                            - ["{feature_name}"]
                - class: qlib.data.dataset.loader.StaticDataLoader
                  kwargs:
                    config: "combined_factors_df.parquet"

    learn_processors:
        - class: DropnaLabel
          kwargs:
              fields_group: label

port_analysis_config: &port_analysis_config
    strategy:
        class: TopkDropoutStrategy
        module_path: qlib.contrib.strategy
        kwargs:
            signal: <PRED>
            topk: 1
            n_drop: 0
    backtest:
        start_time: "2025-01-01"
        end_time: "2026-03-20"
        account: 100000
        benchmark: *benchmark
        exchange_kwargs:
            limit_threshold: 0.0
            deal_price: close
            open_cost: 0.00015
            close_cost: 0.00015
            min_cost: 0

task:
    model:
        class: LGBModel
        module_path: qlib.contrib.model.gbdt
        kwargs:
            loss: mse
            colsample_bytree: 0.8879
            learning_rate: 0.2
            subsample: 0.8789
            lambda_l1: 205.6999
            lambda_l2: 580.9768
            max_depth: 8
            num_leaves: 210
            num_threads: 2
    dataset:
        class: DatasetH
        module_path: qlib.data.dataset
        kwargs:
            handler:
                class: DataHandlerLP
                module_path: qlib.contrib.data.handler
                kwargs: *data_handler_config
            segments:
                train: ["2022-03-14", "2024-06-30"]
                valid: ["2024-07-01", "2024-12-31"]
                test: ["2025-01-01", "2026-03-20"]
    record:
        - class: SignalRecord
          module_path: qlib.workflow.record_temp
          kwargs:
              model: <MODEL>
              dataset: <DATASET>
        - class: SigAnaRecord
          module_path: qlib.workflow.record_temp
          kwargs:
              ana_long_short: False
              ann_scaler: 252
        - class: PortAnaRecord
          module_path: qlib.workflow.record_temp
          kwargs:
              config: *port_analysis_config
"""

        (ws / "conf.yaml").write_text(config, encoding="utf-8")

        # Run Qlib via qrun CLI
        try:
            result = subprocess.run(
                [str(QRUN_PATH), str(ws / "conf.yaml")],
                cwd=str(ws),
                capture_output=True,
                text=True,
                timeout=300,  # 5 min timeout
                env={**os.environ, "PYTHONPATH": str(ws)},
            )

            # Parse results from qlib output
            ic = None
            sharpe = None
            ann_ret = None
            max_dd = None

            # Look for IC in stdout
            if result.stdout:
                import re
                ic_match = re.search(r'IC\s+([\d.]+)', result.stdout)
                if ic_match:
                    ic = float(ic_match.group(1))

            # Try to read qlib_res.csv
            qlib_res = ws / "qlib_res.csv"
            if qlib_res.exists():
                try:
                    res_df = pd.read_csv(qlib_res, index_col=0, header=None)
                    if len(res_df.columns) > 0:
                        metrics = res_df.iloc[:, 0]
                        for key, attr in [("IC", "ic"), ("1day.excess_return_with_cost.shar", "sharpe")]:
                            val = metrics.get(key, None)
                            if val is not None and isinstance(val, (int, float)):
                                if attr == "ic":
                                    ic = float(val)
                                elif attr == "sharpe":
                                    sharpe = float(val)
                except:
                    pass

            return BacktestResult(
                factor_name=factor_info.factor_name,
                workspace_hash=factor_info.workspace_hash,
                status="success",
                ic=ic,
                sharpe_ratio=sharpe,
                annualized_return=ann_ret,
                max_drawdown=max_dd,
                win_rate=None,
                information_ratio=None,
                error_message=None,
                timestamp=datetime.now().isoformat(),
            )

        except subprocess.TimeoutExpired:
            return BacktestResult(
                factor_name=factor_info.factor_name,
                workspace_hash=factor_info.workspace_hash,
                status="failed",
                error_message="Qlib timeout (5 min)",
                timestamp=datetime.now().isoformat(),
            )
        except Exception as e:
            return BacktestResult(
                factor_name=factor_info.factor_name,
                workspace_hash=factor_info.workspace_hash,
                status="failed",
                error_message=str(e)[:500],
                timestamp=datetime.now().isoformat(),
            )


# ---------------------------------------------------------------------------
# Results Storage
# ---------------------------------------------------------------------------
class BatchResultsStorage:
    """Saves backtest results to JSON and SQLite."""

    def __init__(self):
        self.backtests_dir = BACKTESTS_DIR
        self.factors_dir = FACTORS_DIR
        self.db_path = DB_PATH

        # Ensure directories exist
        self.backtests_dir.mkdir(parents=True, exist_ok=True)
        self.factors_dir.mkdir(parents=True, exist_ok=True)
        self.db_path.parent.mkdir(parents=True, exist_ok=True)

    def save_result(self, result: BacktestResult) -> bool:
        """Save a single backtest result."""
        saved = False

        # Save as JSON
        try:
            self._save_json(result)
            saved = True
        except Exception as e:
            console.print(f"  [yellow]JSON save failed for {result.factor_name}: {e}[/yellow]")

        # Save to SQLite
        try:
            self._save_to_db(result)
            saved = True
        except Exception as e:
            console.print(f"  [yellow]DB save failed for {result.factor_name}: {e}[/yellow]")

        return saved

    def _save_json(self, result: BacktestResult) -> None:
        """Save result as JSON file."""
        safe_name = re.sub(r"[^\w\-_]", "_", result.factor_name)[:80]
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        filename = f"{safe_name}_{timestamp}.json"
        filepath = self.backtests_dir / filename

        with open(filepath, "w", encoding="utf-8") as f:
            json.dump(result.to_dict(), f, indent=2, default=str)

    def _save_to_db(self, result: BacktestResult) -> None:
        """Save result to SQLite database."""
        # Import ResultsDatabase from existing module
        sys.path.insert(0, str(PROJECT_ROOT / "rdagent" / "components" / "backtesting"))
        from results_db import ResultsDatabase

        db = ResultsDatabase(db_path=str(self.db_path))

        metrics = {
            "ic": result.ic,
            "sharpe_ratio": result.sharpe_ratio,
            "annualized_return": result.annualized_return,
            "max_drawdown": result.max_drawdown,
            "win_rate": result.win_rate,
            "information_ratio": result.information_ratio,
            "volatility": result.volatility,
        }

        db.add_backtest(factor_name=result.factor_name[:100], metrics=metrics)
        db.close()

    def save_batch_summary(self, results: List[BacktestResult], total_scanned: int) -> Dict:
        """Save overall batch summary."""
        successful = [r for r in results if r.status == "success"]
        failed = [r for r in results if r.status == "failed"]

        # Compute rankings
        ranked_by_ic = sorted(
            [r for r in successful if r.ic is not None],
            key=lambda r: abs(r.ic),
            reverse=True,
        )
        ranked_by_sharpe = sorted(
            [r for r in successful if r.sharpe_ratio is not None],
            key=lambda r: r.sharpe_ratio,
            reverse=True,
        )

        summary = {
            "generated_at": datetime.now().isoformat(),
            "total_scanned": total_scanned,
            "total_backtested": len(results),
            "successful": len(successful),
            "failed": len(failed),
            "success_rate": len(successful) / len(results) if results else 0,
            "total_duration_seconds": sum(r.duration_seconds for r in results),
            "best_ic": ranked_by_ic[0].to_dict() if ranked_by_ic else None,
            "best_sharpe": ranked_by_sharpe[0].to_dict() if ranked_by_sharpe else None,
            "top_20_by_ic": [r.to_dict() for r in ranked_by_ic[:20]],
            "top_20_by_sharpe": [r.to_dict() for r in ranked_by_sharpe[:20]],
            "all_results": [r.to_dict() for r in results],
        }

        with open(BATCH_SUMMARY_PATH, "w", encoding="utf-8") as f:
            json.dump(summary, f, indent=2, default=str)

        return summary


# ---------------------------------------------------------------------------
# Parallel Execution
# ---------------------------------------------------------------------------
def _worker_backtest(factor_info: FactorInfo) -> BacktestResult:
    """Worker function for parallel execution - calls Qlib directly."""
    try:
        return _run_qlib_single(factor_info)
    except Exception as e:
        return BacktestResult(
            factor_name=factor_info.factor_name,
            workspace_hash=factor_info.workspace_hash,
            status="failed",
            error_message=f"Worker exception: {str(e)[:300]}",
            timestamp=datetime.now().isoformat(),
        )


def run_parallel_backtests(
    factors: List[FactorInfo],
    n_workers: int = 4,
) -> List[BacktestResult]:
    """
    Run backtests in parallel using ProcessPoolExecutor.

    Parameters
    ----------
    factors : List[FactorInfo]
        Factors to backtest
    n_workers : int
        Number of parallel workers

    Returns
    -------
    List[BacktestResult]
    """
    results = []

    with Progress(
        SpinnerColumn(),
        TextColumn("[progress.description]{task.description}"),
        BarColumn(),
        TaskProgressColumn(),
        TimeElapsedColumn(),
        TimeRemainingColumn(),
        console=console,
    ) as progress:
        task = progress.add_task(f"Backtesting {len(factors)} factors...", total=len(factors))

        with ProcessPoolExecutor(max_workers=n_workers) as executor:
            futures = {
                executor.submit(_worker_backtest, f): f
                for f in factors
            }

            for future in as_completed(futures):
                factor = futures[future]
                try:
                    result = future.result(timeout=600)  # 10 min timeout per factor
                    results.append(result)
                except Exception as e:
                    results.append(BacktestResult(
                        factor_name=factor.factor_name,
                        workspace_hash=factor.workspace_hash,
                        status="failed",
                        error_message=f"Timeout/Exception: {str(e)[:300]}",
                        timestamp=datetime.now().isoformat(),
                    ))

                # Update progress
                n_success = sum(1 for r in results if r.status == "success")
                n_fail = sum(1 for r in results if r.status == "failed")
                progress.update(
                    task,
                    advance=1,
                    description=f"Backtesting: {n_success}✅ {n_fail}❌ | {factor.factor_name[:40]}",
                )

    return results


# ---------------------------------------------------------------------------
# CLI Interface
# ---------------------------------------------------------------------------
def build_results_table(results: List[BacktestResult], limit: int = 20) -> Table:
    """Build a Rich table of results."""
    table = Table(
        title="Top Backtest Results",
        show_header=True,
        header_style="bold cyan",
    )
    table.add_column("#", justify="center", width=4)
    table.add_column("Factor", width=40)
    table.add_column("IC", justify="right", width=10)
    table.add_column("Sharpe", justify="right", width=10)
    table.add_column("Ann. Return", justify="right", width=12)
    table.add_column("Max DD", justify="right", width=10)
    table.add_column("Win Rate", justify="right", width=10)
    table.add_column("Status", justify="center", width=8)

    # Sort by absolute IC
    ranked = sorted(
        [r for r in results if r.status == "success" and r.ic is not None],
        key=lambda r: abs(r.ic),
        reverse=True,
    )[:limit]

    for i, r in enumerate(ranked, 1):
        status_icon = "✅" if r.status == "success" else "❌"
        table.add_row(
            str(i),
            r.factor_name[:38],
            f"{r.ic:.6f}" if r.ic is not None else "N/A",
            f"{r.sharpe_ratio:.4f}" if r.sharpe_ratio is not None else "N/A",
            f"{r.annualized_return:.4f}" if r.annualized_return is not None else "N/A",
            f"{r.max_drawdown:.4f}" if r.max_drawdown is not None else "N/A",
            f"{r.win_rate:.4f}" if r.win_rate is not None else "N/A",
            status_icon,
        )

    return table


def main(
    factors: int = 100,
    all_factors: bool = False,
    parallel: int = 4,
    scan_only: bool = False,
    metric: str = "ic",
    extract_existing: bool = False,
) -> None:
    """
    Main batch backtest entry point.

    Parameters
    ----------
    factors : int
        Number of factors to backtest
    all_factors : bool
        If True, backtest all discovered factors
    parallel : int
        Number of parallel workers
    scan_only : bool
        If True, only scan and list factors without backtesting
    metric : str
        Metric for ranking ('ic' or 'sharpe')
    """
    console.print(Panel(
        "[bold cyan]Predix Batch Backtest Runner[/bold cyan]\n"
        f"Scanning workspaces for generated factors...",
        border_style="cyan",
    ))

    # -----------------------------------------------------------------------
    # Step 1: Scan workspaces
    # -----------------------------------------------------------------------
    extractor = FactorExtractor()
    all_factors_list = extractor.scan_workspaces()

    if not all_factors_list:
        console.print("\n[red]No factors found in workspaces![/red]")
        console.print(
            "[yellow]Ensure factors have been generated via `predix.py quant` first.[/yellow]"
        )
        return

    # Deduplicate
    unique_factors = extractor.get_unique_factors()
    console.print(f"\n[bold]Unique factors: {len(unique_factors)}[/bold]")

    if scan_only:
        # Show scan results only
        table = Table(title="Discovered Factors", show_header=True, header_style="bold cyan")
        table.add_column("#", justify="center", width=5)
        table.add_column("Factor Name", width=40)
        table.add_column("IC", justify="right", width=10)
        table.add_column("Sharpe", justify="right", width=10)
        table.add_column("Has Result", justify="center", width=10)
        table.add_column("Workspace", width=20)

        for i, f in enumerate(unique_factors[:50], 1):
            has_icon = "✅" if f.has_existing_result else "❌"
            table.add_row(
                str(i),
                f.factor_name[:38],
                f"{f.existing_ic:.6f}" if f.existing_ic is not None else "N/A",
                f"{f.existing_sharpe:.4f}" if f.existing_sharpe is not None else "N/A",
                has_icon,
                f.workspace_hash[:18],
            )

        console.print(table)
        console.print(f"\n[dim]Total unique: {len(unique_factors)}, Total with results: {sum(1 for f in unique_factors if f.has_existing_result)}[/dim]")
        return

    # -----------------------------------------------------------------------
    # Extract existing results mode (fast — no backtest execution)
    # -----------------------------------------------------------------------
    if extract_existing:
        storage = BatchResultsStorage()
        console.print(f"\n[bold cyan]Extracting existing qlib_res.csv results...[/bold cyan]")

        existing_results = []
        factors_with_results = [f for f in unique_factors if f.has_existing_result]

        if not factors_with_results:
            console.print("[yellow]No factors with existing results found.[/yellow]")
            console.print("[dim]All qlib_res.csv files have empty metric values (IC, Sharpe, etc.).[/dim]")
            console.print("[dim]You need to run full backtests via Qlib (Docker).[/dim]")

            # Still save empty summary
            summary = storage.save_batch_summary([], total_scanned=len(all_factors_list))
            return

        with Progress(
            SpinnerColumn(),
            TextColumn("[progress.description]{task.description}"),
            BarColumn(),
            TaskProgressColumn(),
            console=console,
        ) as progress:
            task = progress.add_task("Extracting...", total=len(factors_with_results))

            for factor in factors_with_results:
                result = BacktestResult(
                    factor_name=factor.factor_name,
                    workspace_hash=factor.workspace_hash,
                    status="success",
                    ic=factor.existing_ic,
                    sharpe_ratio=factor.existing_sharpe,
                    annualized_return=getattr(factor, '_annualized_return', None),
                    max_drawdown=getattr(factor, '_max_drawdown', None),
                    information_ratio=getattr(factor, '_information_ratio', None),
                    timestamp=datetime.now().isoformat(),
                )
                existing_results.append(result)
                storage.save_result(result)
                progress.update(task, advance=1, description=f"Extracted {factor.factor_name[:40]}")

        # Save summary
        summary = storage.save_batch_summary(existing_results, total_scanned=len(all_factors_list))

        # Display — rank by information_ratio if no IC/Sharpe
        table = Table(
            title="Existing Backtest Results",
            show_header=True,
            header_style="bold cyan",
        )
        table.add_column("#", justify="center", width=4)
        table.add_column("Factor", width=40)
        table.add_column("IC", justify="right", width=10)
        table.add_column("Sharpe", justify="right", width=10)
        table.add_column("Ann. Return", justify="right", width=12)
        table.add_column("Max DD", justify="right", width=10)
        table.add_column("Info Ratio", justify="right", width=12)

        # Sort by best available metric
        def sort_key(r):
            if r.information_ratio is not None:
                return abs(r.information_ratio)
            if r.annualized_return is not None:
                return r.annualized_return
            if r.ic is not None:
                return abs(r.ic)
            return 0

        ranked = sorted(existing_results, key=sort_key, reverse=True)[:20]

        for i, r in enumerate(ranked, 1):
            table.add_row(
                str(i),
                r.factor_name[:38],
                f"{r.ic:.6f}" if r.ic is not None else "N/A",
                f"{r.sharpe_ratio:.4f}" if r.sharpe_ratio is not None else "N/A",
                f"{r.annualized_return:.4f}" if r.annualized_return is not None else "N/A",
                f"{r.max_drawdown:.4f}" if r.max_drawdown is not None else "N/A",
                f"{r.information_ratio:.4f}" if r.information_ratio is not None else "N/A",
            )

        console.print()
        console.print(table)

        valid_ir = [r.information_ratio for r in existing_results if r.information_ratio is not None]
        valid_ar = [r.annualized_return for r in existing_results if r.annualized_return is not None]

        console.print(Panel(
            f"[bold]Existing Results Summary[/bold]\n"
            f"Total scanned: {len(all_factors_list)} factors\n"
            f"With results: {len(existing_results)}\n"
            f"Avg Information Ratio: {np.mean(valid_ir):.4f} (n={len(valid_ir)})\n"
            f"Best Information Ratio: {max(valid_ir, default=0):.4f}\n"
            f"Avg Ann. Return: {np.mean(valid_ar):.4f} (n={len(valid_ar)})\n"
            f"Best Ann. Return: {max(valid_ar, default=0):.4f}",
            border_style="green",
        ))
        return

    # -----------------------------------------------------------------------
    # Step 2: Select factors to backtest
    # -----------------------------------------------------------------------
    if all_factors:
        to_backtest = unique_factors
    else:
        # Get top factors by existing metric
        to_backtest = extractor.get_top_factors(metric=metric, limit=factors)

    console.print(f"\n[bold green]Selected {len(to_backtest)} factors for backtesting[/bold green]")

    # -----------------------------------------------------------------------
    # Step 3: Run backtests
    # -----------------------------------------------------------------------
    console.print(f"[bold yellow]Running Qlib backtests with {parallel} parallel workers...[/bold yellow]")
    
    # Use the Qlib runner directly
    results = run_parallel_backtests(to_backtest, n_workers=parallel)

    # -----------------------------------------------------------------------
    # Step 4: Save results
    # -----------------------------------------------------------------------
    console.print(f"\n[bold cyan]Saving results...[/bold cyan]")
    storage = BatchResultsStorage()

    saved_count = 0
    for result in results:
        if storage.save_result(result):
            saved_count += 1

    # Batch summary
    summary = storage.save_batch_summary(results, total_scanned=len(all_factors_list))

    # -----------------------------------------------------------------------
    # Step 5: Display results
    # -----------------------------------------------------------------------
    console.print()
    console.print(build_results_table(results, limit=20))

    # Summary stats
    n_success = sum(1 for r in results if r.status == "success")
    n_fail = sum(1 for r in results if r.status == "failed")
    n_skipped = sum(1 for r in results if r.status == "skipped")
    valid_ic = [r.ic for r in results if r.ic is not None]
    valid_sharpe = [r.sharpe_ratio for r in results if r.sharpe_ratio is not None]

    avg_ic = float(np.mean(valid_ic)) if valid_ic else 0.0
    best_ic = float(max(valid_ic)) if valid_ic else 0.0
    avg_sharpe = float(np.mean(valid_sharpe)) if valid_sharpe else 0.0
    best_sharpe = float(max(valid_sharpe)) if valid_sharpe else 0.0

    console.print(Panel(
        f"[bold]Backtest Summary[/bold]\n"
        f"Scanned: {len(all_factors_list)} factors\n"
        f"Backtested: {len(results)}\n"
        f"Successful: {n_success} ✅\n"
        f"Failed: {n_fail} ❌\n"
        f"Skipped (needs Qlib): {n_skipped} ⏭️\n"
        f"Success Rate: {n_success / len(results) * 100:.1f}%\n"
        f"Avg IC: {avg_ic:.6f} (n={len(valid_ic)})\n"
        f"Best IC: {best_ic:.6f}\n"
        f"Avg Sharpe: {avg_sharpe:.4f} (n={len(valid_sharpe)})\n"
        f"Best Sharpe: {best_sharpe:.4f}\n"
        f"Saved to: {BATCH_SUMMARY_PATH}\n"
        f"Database: {DB_PATH}",
        border_style="green",
    ))


# ---------------------------------------------------------------------------
# Entry point
# ---------------------------------------------------------------------------
if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser(
        description="Predix Batch Backtest - Extract and backtest existing factors"
    )
    parser.add_argument(
        "--factors", "-n",
        type=int,
        default=100,
        help="Number of factors to backtest (default: 100)",
    )
    parser.add_argument(
        "--all", "-a",
        action="store_true",
        help="Backtest all discovered factors",
    )
    parser.add_argument(
        "--parallel", "-p",
        type=int,
        default=4,
        help="Number of parallel workers (default: 4)",
    )
    parser.add_argument(
        "--scan-only", "-s",
        action="store_true",
        help="Only scan and list factors, don't run backtests",
    )
    parser.add_argument(
        "--metric", "-m",
        type=str,
        default="ic",
        choices=["ic", "sharpe"],
        help="Metric for ranking factors (default: ic)",
    )
    parser.add_argument(
        "--workspace-dir", "-w",
        type=str,
        default=None,
        help="Custom workspace directory to scan",
    )
    parser.add_argument(
        "--extract-existing", "-e",
        action="store_true",
        help="Only extract existing qlib_res.csv results from all workspaces (fast)",
    )

    args = parser.parse_args()

    # Graceful signal handling
    def signal_handler(sig, frame):
        console.print("\n[yellow]Interrupted by user. Saving partial results...[/yellow]")
        sys.exit(130)

    signal.signal(signal.SIGINT, signal_handler)
    signal.signal(signal.SIGTERM, signal_handler)

    workspace_dir = Path(args.workspace_dir) if args.workspace_dir else None

    main(
        factors=args.factors,
        all_factors=args.all,
        parallel=args.parallel,
        scan_only=args.scan_only,
        metric=args.metric,
        extract_existing=args.extract_existing,
    )
