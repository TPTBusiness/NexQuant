#!/usr/bin/env python
"""
Extract Results from Qlib Workspace to ResultsDatabase

This script parses existing Qlib workspace results (from Docker backtests)
and imports them into the ResultsDatabase for querying and dashboard display.

It can:
1. Parse qlib_res.csv files from workspace directories
2. Parse ret.pkl files for portfolio analysis
3. Import results into the SQLite database
4. Handle both new and existing workspace structures

Usage:
    python scripts/extract_results.py [--workspace-dir PATH] [--dry-run] [--verbose]
"""

import argparse
import pickle
import sys
import traceback
from pathlib import Path
from datetime import datetime
from typing import Dict, Optional, List, Tuple

import pandas as pd

# Add project root to path
project_root = Path(__file__).parent.parent
sys.path.insert(0, str(project_root))


class WorkspaceResultExtractor:
    """
    Extract backtest results from Qlib workspace directories.

    Scans workspace directories for qlib_res.csv and ret.pkl files,
    parses metrics, and optionally imports them into ResultsDatabase.
    """

    def __init__(self, workspace_dir: Path, dry_run: bool = False, verbose: bool = False):
        """
        Initialize the extractor.

        Parameters
        ----------
        workspace_dir : Path
            Root directory of the Qlib workspace
        dry_run : bool
            If True, only scan and display results without importing to DB
        verbose : bool
            If True, print detailed extraction logs
        """
        self.workspace_dir = workspace_dir
        self.dry_run = dry_run
        self.verbose = verbose
        self.extracted_results: List[Dict] = []

    def scan_workspace(self) -> List[Path]:
        """
        Scan workspace for qlib_res.csv files.

        Returns
        -------
        List[Path]
            List of paths to qlib_res.csv files found
        """
        csv_files = list(self.workspace_dir.rglob("qlib_res.csv"))
        pkl_files = list(self.workspace_dir.rglob("ret.pkl"))

        if self.verbose:
            print(f"\nScanning workspace: {self.workspace_dir}")
            print(f"  Found {len(csv_files)} qlib_res.csv files")
            print(f"  Found {len(pkl_files)} ret.pkl files")

        return csv_files

    def extract_from_csv(self, csv_path: Path) -> Optional[Dict]:
        """
        Extract metrics from a qlib_res.csv file.

        Parameters
        ----------
        csv_path : Path
            Path to the qlib_res.csv file

        Returns
        -------
        Optional[Dict]
            Dictionary of metrics, or None if extraction failed
        """
        try:
            # Read CSV - format is: ,value (first column is metric name, second is value)
            # The CSV has no header, with index column 0 and value column 1
            df = pd.read_csv(csv_path, header=None)
            
            # Convert to dictionary {metric_name: value}
            # Column 0 is metric name, column 1 is value
            metrics = {}
            for _, row in df.iterrows():
                if len(row) >= 2:
                    metric_name = str(row[0]).strip()
                    metric_value = row[1]
                    # Only include non-empty metrics with valid names
                    if metric_name and metric_name.lower() != 'nan' and pd.notna(metric_value) and str(metric_value).strip() != '':
                        try:
                            metrics[metric_name] = float(metric_value)
                        except (ValueError, TypeError):
                            metrics[metric_name] = str(metric_value)

            # Skip if no meaningful metrics found
            if not metrics or len(metrics) < 2:
                if self.verbose:
                    print(f"\n  SKIPPING {csv_path}: No meaningful metrics found (empty or failed backtest)")
                return None

            # Parse important metrics - Qlib uses various naming conventions
            result = {
                'ic': self._safe_float(metrics.get('IC', None)),
                'sharpe_ratio': self._safe_float(
                    metrics.get('1day.excess_return_with_cost.shar', 
                    metrics.get('1day.excess_return_with_cost.sharpe', None))
                ),
                'annualized_return': self._safe_float(
                    metrics.get('1day.excess_return_with_cost.annualized_return', None)
                ),
                'max_drawdown': self._safe_float(
                    metrics.get('1day.excess_return_with_cost.max_drawdown', None)
                ),
                'win_rate': self._safe_float(metrics.get('win_rate', None)),
                'information_ratio': self._safe_float(
                    metrics.get('1day.excess_return_with_cost.information_ratio', None)
                ),
                'volatility': self._safe_float(
                    metrics.get('1day.excess_return_with_cost.std', 
                    metrics.get('1day.excess_return_with_cost.volatility', None))
                ),
                'raw_metrics': metrics,
                'source_file': str(csv_path),
            }

            if self.verbose:
                print(f"\n  Extracted from {csv_path.name}:")
                print(f"    IC: {result['ic']}")
                print(f"    Sharpe: {result['sharpe_ratio']}")
                print(f"    Annual Return: {result['annualized_return']}")
                print(f"    Max Drawdown: {result['max_drawdown']}")
                print(f"    Information Ratio: {result['information_ratio']}")
                print(f"    Volatility: {result['volatility']}")
                print(f"    Total metrics: {len(metrics)}")

            return result

        except Exception as e:
            if self.verbose:
                print(f"\n  ERROR extracting from {csv_path}: {e}")
                traceback.print_exc()
            return None

    def extract_from_pkl(self, pkl_path: Path) -> Optional[pd.DataFrame]:
        """
        Extract portfolio analysis from ret.pkl file.

        Parameters
        ----------
        pkl_path : Path
            Path to the ret.pkl file

        Returns
        -------
        Optional[pd.DataFrame]
            DataFrame with portfolio analysis, or None if failed
        """
        try:
            df = pd.read_pickle(pkl_path)
            if self.verbose:
                print(f"\n  Extracted ret.pkl from {pkl_path}:")
                print(f"    Shape: {df.shape}")
                print(f"    Columns: {list(df.columns)}")
            return df
        except Exception as e:
            if self.verbose:
                print(f"\n  ERROR extracting ret.pkl from {pkl_path}: {e}")
            return None

    def extract_factor_name_from_path(self, csv_path: Path) -> str:
        """
        Attempt to extract factor name from the file path.

        Parameters
        ----------
        csv_path : Path
            Path to qlib_res.csv

        Returns
        -------
        str
            Extracted factor name or 'unknown'
        """
        # Try to find factor name in directory structure
        # Common pattern: workspace/factor_name/qlib_res.csv
        parent_dir = csv_path.parent.name
        if parent_dir and parent_dir not in ['.', '..']:
            return parent_dir

        # Fallback to parent's parent
        grandparent = csv_path.parent.parent.name
        if grandparent:
            return grandparent

        return 'unknown'

    def extract_all(self) -> List[Dict]:
        """
        Extract all results from the workspace.

        Returns
        -------
        List[Dict]
            List of extracted result dictionaries
        """
        csv_files = self.scan_workspace()

        total = len(csv_files)
        print(f"\nProcessing {total} qlib_res.csv files...")

        for i, csv_path in enumerate(csv_files, 1):
            # Progress indicator for large workspaces
            if i % 100 == 0 or i == total:
                print(f"  Progress: {i}/{total} ({i*100//max(total,1)}%)")

            result = self.extract_from_csv(csv_path)
            if result is not None:
                result['factor_name'] = self.extract_factor_name_from_path(csv_path)
                result['extraction_time'] = datetime.now().isoformat()
                self.extracted_results.append(result)

        print(f"\nExtracted {len(self.extracted_results)} valid results from {total} files")
        print(f"  Skipped {total - len(self.extracted_results)} empty/failed backtests")
        return self.extracted_results

    def import_to_database(self) -> int:
        """
        Import extracted results to the ResultsDatabase.

        Returns
        -------
        int
            Number of results successfully imported
        """
        if self.dry_run:
            print("\n[DRY RUN] Skipping database import")
            return 0

        if not self.extracted_results:
            print("\nNo results to import")
            return 0

        try:
            from rdagent.components.backtesting import ResultsDatabase

            db = ResultsDatabase()
            imported = 0
            failed = 0
            total = len(self.extracted_results)

            print(f"\nImporting {total} results to database...")

            for i, result in enumerate(self.extracted_results, 1):
                # Progress indicator
                if i % 100 == 0 or i == total:
                    print(f"  DB Progress: {i}/{total} ({i*100//max(total,1)}%) - Imported: {imported}, Failed: {failed}")

                try:
                    factor_name = result.get('factor_name', 'unknown')[:100]
                    metrics = {
                        'ic': result.get('ic'),
                        'sharpe_ratio': result.get('sharpe_ratio'),
                        'annualized_return': result.get('annualized_return'),
                        'max_drawdown': result.get('max_drawdown'),
                        'win_rate': result.get('win_rate'),
                        'information_ratio': result.get('information_ratio'),
                        'volatility': result.get('volatility'),
                        'raw_metrics': result.get('raw_metrics'),
                    }

                    run_id = db.add_backtest(factor_name=factor_name, metrics=metrics)
                    if run_id > 0:
                        imported += 1
                        if self.verbose:
                            print(f"  Imported: {factor_name} (IC={metrics['ic']}, Sharpe={metrics['sharpe_ratio']})")
                    else:
                        failed += 1
                        if self.verbose:
                            print(f"  WARNING: Failed to import {factor_name}")

                except Exception as e:
                    failed += 1
                    if self.verbose:
                        print(f"  ERROR importing {result.get('factor_name', 'unknown')}: {e}")
                        traceback.print_exc()

            db.close()
            print(f"\n{'=' * 60}")
            print(f"IMPORT COMPLETE")
            print(f"{'=' * 60}")
            print(f"  Total processed: {total}")
            print(f"  Successfully imported: {imported}")
            print(f"  Failed: {failed}")
            print(f"  Database: {db.db_path}")
            print(f"{'=' * 60}")
            return imported

        except Exception as e:
            print(f"\nERROR: Failed to connect to database: {e}")
            traceback.print_exc()
            return 0

    def _safe_float(self, value) -> Optional[float]:
        """Safely convert value to float."""
        if value is None:
            return None
        try:
            f = float(value)
            if pd.isna(f) or f == float('inf') or f == float('-inf'):
                return None
            return f
        except (ValueError, TypeError):
            return None

    def display_summary(self):
        """Display a summary of extracted results."""
        if not self.extracted_results:
            print("\nNo results extracted")
            return

        print("\n" + "=" * 80)
        print("EXTRACTED RESULTS SUMMARY")
        print("=" * 80)

        df = pd.DataFrame([
            {
                'Factor': r.get('factor_name', 'unknown'),
                'IC': r.get('ic'),
                'Sharpe': r.get('sharpe_ratio'),
                'Ann. Return': r.get('annualized_return'),
                'Max DD': r.get('max_drawdown'),
                'Win Rate': r.get('win_rate'),
            }
            for r in self.extracted_results
        ])

        # Sort by Sharpe ratio
        if 'Sharpe' in df.columns:
            df = df.dropna(subset=['Sharpe']).sort_values('Sharpe', ascending=False)

        print(df.to_string(index=False))
        print(f"\nTotal results: {len(self.extracted_results)}")

        if len(df) > 0 and df['IC'].notna().any():
            print(f"Average IC: {df['IC'].mean():.4f}")
        if len(df) > 0 and df['Sharpe'].notna().any():
            print(f"Best Sharpe: {df['Sharpe'].max():.4f}")
        print("=" * 80)


def main():
    parser = argparse.ArgumentParser(
        description="Extract Qlib workspace results to ResultsDatabase"
    )
    parser.add_argument(
        '--workspace-dir',
        type=str,
        default='git_ignore_folder/RD-Agent_workspace',
        help='Path to workspace directory (default: git_ignore_folder/RD-Agent_workspace)'
    )
    parser.add_argument(
        '--dry-run',
        action='store_true',
        help='Scan and display results without importing to database'
    )
    parser.add_argument(
        '--verbose',
        action='store_true',
        help='Enable verbose output with detailed logs'
    )
    parser.add_argument(
        '--summary-only',
        action='store_true',
        help='Only show summary, skip extraction'
    )

    args = parser.parse_args()

    workspace_path = Path(args.workspace_dir).resolve()

    if not workspace_path.exists():
        print(f"ERROR: Workspace directory not found: {workspace_path}")
        print("Please ensure you have run at least one fin_quant loop")
        sys.exit(1)

    print(f"Workspace: {workspace_path}")
    print(f"Dry run: {args.dry_run}")
    print(f"Verbose: {args.verbose}")

    # Extract results
    extractor = WorkspaceResultExtractor(
        workspace_dir=workspace_path,
        dry_run=args.dry_run,
        verbose=args.verbose
    )

    if not args.summary_only:
        extractor.extract_all()
        extractor.import_to_database()

    extractor.display_summary()


if __name__ == "__main__":
    main()
