#!/usr/bin/env python
"""
Predix CLI - Wrapper for rdagent with LLM model selection.

Usage:
    predix quant                    # Local llama.cpp (default)
    predix quant --model local      # Explicit local
    predix quant --model openrouter # OpenRouter cloud model
    predix quant -d                 # With web dashboard
"""
import os
import sys
from pathlib import Path

from dotenv import load_dotenv
load_dotenv(Path(__file__).parent / ".env")

import typer
from rich.console import Console

app = typer.Typer(help="Predix - AI Quantitative Trading Agent")
console = Console()


@app.command()
def quant(
    model: str = typer.Option(
        "local",
        "--model", "-m",
        help="LLM backend: 'local' (llama.cpp) or 'openrouter' (cloud)",
    ),
    dashboard: bool = typer.Option(
        False,
        "--dashboard/-d",
        help="Start web dashboard",
    ),
    cli_dashboard: bool = typer.Option(
        False,
        "--cli-dashboard/-c",
        help="Start CLI dashboard",
    ),
    log_file: str = typer.Option(
        None,  # None means auto-detect based on run_id
        "--log-file",
        help="Log file path (default: auto-detected). Use 'none' to disable.",
    ),
    step_n: int = typer.Option(None, help="Number of steps to run"),
    loop_n: int = typer.Option(None, help="Number of loops to run"),
    run_id: int = typer.Option(
        0,
        "--run-id",
        help="Parallel run ID (for isolated results). 0 = single run mode.",
    ),
):
    """
    Start EURUSD quantitative trading loop.

    Examples:
        predix quant                          # Local llama.cpp
        predix quant -m openrouter            # OpenRouter cloud model
        predix quant -d                       # With web dashboard
        predix quant -m openrouter -d         # Both
        predix quant --run-id 1               # Parallel run #1 (isolated)
    """
    import subprocess
    import threading
    import time
    import sys

    # ---- Parallel Run Isolation ----
    # When run_id > 0, isolate all outputs (logs, results, workspace)
    if run_id > 0:
        os.environ["PARALLEL_RUN_ID"] = str(run_id)
        console.print(f"\n[bold yellow]🔀 Parallel Run Mode:[/bold yellow] [cyan]ID={run_id}[/cyan]")

        # Auto-detect log file for parallel run
        if log_file is None:
            log_file = f"fin_quant_run{run_id}.log"

        # Isolate results directories
        results_base = Path(__file__).parent / "results" / "runs" / f"run{run_id}"
        results_base.mkdir(parents=True, exist_ok=True)

        # Isolate workspace directory
        workspace_dir = Path(__file__).parent / f"RD-Agent_workspace_run{run_id}"
        os.environ["RD_AGENT_WORKSPACE"] = str(workspace_dir)

        console.print(f"   [dim]Log: {log_file}[/dim]")
        console.print(f"   [dim]Results: results/runs/run{run_id}/[/dim]")
        console.print(f"   [dim]Workspace: {workspace_dir.name}/[/dim]")
    else:
        # Single run mode: default log file
        if log_file is None:
            log_file = "fin_quant.log"

    # ---- Log File Setup ----
    if log_file.lower() != "none":
        log_path = Path(__file__).parent / log_file
        log_path.parent.mkdir(parents=True, exist_ok=True)

        # Open log file for appending
        log_f = open(log_path, "a", encoding="utf-8")

        # Redirect stdout and stderr to both console and log file
        class TeeWriter:
            def __init__(self, *streams):
                self._streams = streams

            def write(self, data):
                for s in self._streams:
                    try:
                        s.write(data)
                        s.flush()
                    except:
                        pass

            def flush(self):
                for s in self._streams:
                    try:
                        s.flush()
                    except:
                        pass

        sys.stdout = TeeWriter(sys.__stdout__, log_f)
        sys.stderr = TeeWriter(sys.__stderr__, log_f)

        console.print(f"\n[dim]📝 Logging to: {log_path}[/dim]")
    else:
        console.print("\n[dim]⚠️  Logging disabled (console only)[/dim]")

    # ---- LLM Model Selection ----
    if model == "openrouter":
        api_key = os.getenv("OPENROUTER_API_KEY", "")
        api_key_2 = os.getenv("OPENROUTER_API_KEY_2", "")
        if not api_key:
            console.print("\n[bold red]❌ OPENROUTER_API_KEY not set in .env[/bold red]")
            console.print("[yellow]Add your API key to .env:[/yellow]")
            console.print('  OPENROUTER_API_KEY=sk-or-your-key-here')
            raise typer.Exit(code=1)

        # Setup both API keys for load balancing
        os.environ["OPENAI_API_BASE"] = "https://openrouter.ai/api/v1"
        os.environ["CHAT_MODEL"] = os.getenv("OPENROUTER_MODEL", "openrouter/qwen/qwen3.6-plus:free")

        # If second key exists, configure LiteLLM for load balancing
        if api_key_2:
            os.environ["OPENAI_API_KEY"] = f"{api_key},{api_key_2}"
            os.environ["LITELLM_PARALLEL_CALLS"] = "2"
            console.print(f"\n[bold blue]🌐 Using OpenRouter (2 API Keys):[/bold blue] [cyan]{os.environ['CHAT_MODEL']}[/cyan]")
            console.print(f"   [dim]Keys: {api_key[:15]}*** + {api_key_2[:15]}***[/dim]")
            console.print(f"   [dim]Parallel: 2 concurrent requests[/dim]")
        else:
            os.environ["OPENAI_API_KEY"] = api_key
            console.print(f"\n[bold blue]🌐 Using OpenRouter:[/bold blue] [cyan]{os.environ['CHAT_MODEL']}[/cyan]")
            console.print(f"   [dim]Key: {api_key[:15]}***[/dim]")
    elif model == "local":
        os.environ["OPENAI_API_KEY"] = os.getenv("OPENAI_API_KEY", "local")
        os.environ["OPENAI_API_BASE"] = os.getenv("OPENAI_API_BASE", "http://localhost:8081/v1")
        os.environ["CHAT_MODEL"] = os.getenv("CHAT_MODEL", "openai/qwen3.5-35b")

        console.print(f"\n[bold green]🏠 Using local LLM:[/bold green] [cyan]{os.environ['CHAT_MODEL']}[/cyan]")
        console.print(f"   [dim]Base: {os.environ['OPENAI_API_BASE']}[/dim]")
    else:
        console.print(f"\n[yellow]⚠️  Unknown model: '{model}'. Using .env settings.[/yellow]")

    # ---- Dashboards ----
    if dashboard:
        def start_web_dashboard():
            console.print(f"\n[bold green]🚀 Web Dashboard: http://localhost:5000[/bold green]")
            subprocess.run(
                ["python", "web/dashboard_api.py"],
                cwd=str(Path(__file__).parent),
                env={**os.environ, "FLASK_ENV": "development"},
            )

        threading.Thread(target=start_web_dashboard, daemon=True).start()
        time.sleep(2)

    if cli_dashboard:
        def start_cli_dash():
            from rdagent.log.ui.predix_dashboard import run_dashboard
            run_dashboard(log_path="fin_quant.log", refresh_interval=3)

        threading.Thread(target=start_cli_dash, daemon=True).start()
        time.sleep(1)

    # ---- Start fin_quant ----
    from rdagent.app.qlib_rd_loop.quant import main as fin_quant

    console.print(f"\n[bold cyan]📊 Starting EURUSD Trading Loop...[/bold cyan]\n")

    fin_quant(
        step_n=step_n,
        loop_n=loop_n,
    )


@app.command()
def evaluate(
    top: int = typer.Option(
        100,
        "--top", "-n",
        help="Number of factors to evaluate (default: 100)",
    ),
    all_factors: bool = typer.Option(
        False,
        "--all", "-a",
        help="Evaluate all undiscovered factors",
    ),
    parallel: int = typer.Option(
        4,
        "--parallel", "-p",
        help="Number of parallel workers (default: 4)",
    ),
    force: bool = typer.Option(
        False,
        "--force", "-f",
        help="Force re-evaluation of ALL factors (even already evaluated)",
    ),
):
    """
    Evaluate existing factors with full 1min data (2020-2026).

    Computes IC, Sharpe, Max DD, Win Rate for each factor.
    Automatically skips already evaluated factors (use --force to re-evaluate).

    Examples:
        predix evaluate                   # Evaluate 100 NEW factors
        predix evaluate --top 500         # Evaluate 500 NEW factors
        predix evaluate --all             # Evaluate all NEW factors
        predix evaluate --force --top 50  # Re-evaluate 50 factors
        predix evaluate -p 8              # Use 8 parallel workers
    """
    from rich.panel import Panel

    console.print(Panel(
        "[bold cyan]📊 Predix Factor Evaluator[/bold cyan]\n"
        "Evaluating factors with FULL 1min data (2020-2026)\n"
        "Skips already evaluated factors automatically",
        border_style="cyan",
    ))

    # Import and run the evaluator
    from predix_full_eval import main as eval_main

    try:
        eval_main(
            top=top,
            all_factors=all_factors,
            parallel=parallel,
            force=force,
        )
    except KeyboardInterrupt:
        console.print("\n[yellow]Evaluation interrupted by user[/yellow]")
    except Exception as e:
        console.print(f"\n[bold red]Evaluation failed: {e}[/bold red]")
        import traceback
        console.print(traceback.format_exc())


@app.command()
def top(
    n: int = typer.Option(
        20,
        "--num", "-n",
        help="Number of top factors to show (default: 20)",
    ),
    metric: str = typer.Option(
        "ic",
        "--metric", "-m",
        help="Sort by metric: 'ic' or 'sharpe'",
    ),
):
    """
    Show top-performing factors by IC or Sharpe.

    Examples:
        predix top                      # Top 20 by IC
        predix top -n 50                # Top 50 by IC
        predix top -m sharpe            # Top 20 by Sharpe
    """
    import json
    import glob as glob_module
    import numpy as np
    from rich.table import Table
    from rich.panel import Panel

    factors_dir = Path(__file__).parent / "results" / "factors"
    if not factors_dir.exists():
        console.print("[red]No results found in results/factors/[/red]")
        return

    # Load all factor JSON files
    results = []
    for f in glob_module.glob(str(factors_dir / "*.json")):
        try:
            with open(f) as fh:
                data = json.load(fh)
            # Only include factors with valid IC
            if data.get("status") == "success" and data.get("ic") is not None:
                results.append(data)
        except Exception:
            continue

    if not results:
        console.print("[yellow]No evaluated factors found with valid IC[/yellow]")
        return

    # Sort by metric
    if metric == "sharpe":
        results.sort(key=lambda x: abs(x.get("sharpe", 0) or 0), reverse=True)
        sort_label = "Sharpe"
    else:
        results.sort(key=lambda x: abs(x.get("ic", 0) or 0), reverse=True)
        sort_label = "IC"

    # Display as table
    table = Table(
        title=f"Top {min(n, len(results))} Factors by {sort_label}",
        show_header=True,
        header_style="bold cyan",
    )
    table.add_column("#", justify="center", width=4)
    table.add_column("Factor", width=40)
    table.add_column("IC", justify="right", width=10)
    table.add_column("Sharpe", justify="right", width=10)
    table.add_column("Ann. Return %", justify="right", width=12)
    table.add_column("Max DD", justify="right", width=10)
    table.add_column("Win Rate", justify="right", width=10)

    for i, r in enumerate(results[:n], 1):
        ic = r.get("ic")
        sharpe = r.get("sharpe")
        ann_ret = r.get("annualized_return")
        max_dd = r.get("max_drawdown")
        win_rate = r.get("win_rate")

        table.add_row(
            str(i),
            r["factor_name"][:38],
            f"{ic:.6f}" if ic is not None else "N/A",
            f"{sharpe:.4f}" if sharpe is not None else "N/A",
            f"{ann_ret:.4f}" if ann_ret is not None else "N/A",
            f"{max_dd:.4f}" if max_dd is not None else "N/A",
            f"{win_rate:.2%}" if win_rate is not None else "N/A",
        )

    console.print(table)

    # Summary
    valid_ic = [r.get("ic") for r in results if r.get("ic") is not None]
    valid_sharpe = [r.get("sharpe") for r in results if r.get("sharpe") is not None]
    # Filter extreme outliers for average
    valid_sharpe_filtered = [s for s in valid_sharpe if abs(s or 0) < 1e6]

    console.print(Panel(
        f"[bold]Summary[/bold]\n"
        f"Total evaluated: {len(results)}\n"
        f"Avg IC: {np.mean(valid_ic):.6f} (n={len(valid_ic)})\n"
        f"Best IC: {max(valid_ic, key=abs, default=0):.6f}\n"
        f"Avg Sharpe: {np.mean(valid_sharpe_filtered):.4f} (n={len(valid_sharpe_filtered)})\n"
        f"Best Sharpe: {max(valid_sharpe, key=abs, default=0):.4f}",
        border_style="green",
    ))


@app.command()
def portfolio(
    top: int = typer.Option(
        50,
        "--top", "-n",
        help="Number of candidate factors to consider (default: 50)",
    ),
    target: int = typer.Option(
        10,
        "--target", "-t",
        help="Number of factors to select (default: 10)",
    ),
    max_corr: float = typer.Option(
        0.3,
        "--max-corr", "-c",
        help="Maximum allowed correlation between factors (default: 0.3)",
    ),
):
    """
    Select a diversified portfolio of uncorrelated factors.

    Analyzes the top factors by IC and selects a subset that are
    not highly correlated, reducing redundancy.

    Examples:
        predix portfolio                   # Select top 10 from top 50
        predix portfolio -n 100 -t 20      # Select top 20 from top 100
        predix portfolio -c 0.5            # Allow higher correlation
    """
    import json
    import glob as glob_module
    import subprocess
    import tempfile
    import shutil
    import numpy as np
    import pandas as pd
    from rich.table import Table
    from rich.panel import Panel
    from rich.progress import Progress, SpinnerColumn, TextColumn, BarColumn, TaskProgressColumn, TimeElapsedColumn

    factors_dir = Path(__file__).parent / "results" / "factors"
    if not factors_dir.exists():
        console.print("[red]No results found in results/factors/[/red]")
        return

    # 1. Load top factors by IC
    results = []
    for f in glob_module.glob(str(factors_dir / "*.json")):
        try:
            with open(f) as fh:
                data = json.load(fh)
            if data.get("status") == "success" and data.get("ic") is not None:
                results.append(data)
        except Exception:
            continue

    if not results:
        console.print("[red]No evaluated factors found with valid IC[/red]")
        return

    # Sort and select candidates
    results.sort(key=lambda x: abs(x.get("ic", 0) or 0), reverse=True)
    candidates = results[:top]

    console.print(f"Loaded {len(results)} factors. Selecting top {top} candidates...")

    # 2. Evaluate candidates to get time-series values for correlation
    # We need to run the factor code to get the series of values.
    # We do this sequentially to avoid OOM.
    
    # Locate data file
    data_file = Path(__file__).parent / "git_ignore_folder" / "factor_implementation_source_data" / "intraday_pv.h5"
    if not data_file.exists():
        data_file = Path(__file__).parent / "git_ignore_folder" / "factor_implementation_source_data_debug" / "intraday_pv.h5"
    
    if not data_file.exists():
        console.print("[red]Source data file (intraday_pv.h5) not found.[/red]")
        return

    factor_series = {} # name -> pd.Series
    errors = []

    with Progress(
        SpinnerColumn(),
        TextColumn("[progress.description]{task.description}"),
        BarColumn(),
        TaskProgressColumn(),
        TimeElapsedColumn(),
        console=console,
    ) as progress:
        task = progress.add_task(f"Computing values for {len(candidates)} factors...", total=len(candidates))
        
        for cand in candidates:
            fname = cand.get("factor_name", "unknown")
            fcode = cand.get("factor_code", "")
            
            if not fcode:
                errors.append((fname, "No code in JSON"))
                progress.advance(task)
                continue

            # Create temp workspace
            with tempfile.TemporaryDirectory() as tmpdir:
                tmp_path = Path(tmpdir)
                # Symlink data
                try:
                    os.symlink(str(data_file), str(tmp_path / "intraday_pv.h5"))
                except OSError:
                    # If symlink fails, copy the file
                    import shutil
                    shutil.copy(str(data_file), str(tmp_path / "intraday_pv.h5"))
                
                # Write code
                (tmp_path / "factor.py").write_text(fcode)
                
                try:
                    # Run factor
                    result = subprocess.run(
                        [sys.executable, "factor.py"],
                        cwd=tmp_path,
                        capture_output=True,
                        text=True,
                        timeout=120 # 2 min timeout per factor
                    )
                    
                    # Read result
                    res_file = tmp_path / "result.h5"
                    if res_file.exists():
                        df = pd.read_hdf(str(res_file), key="data")
                        # Get the series (first column)
                        series = df.iloc[:, 0]
                        
                        # Count non-NaN values
                        non_nan = series.count()
                        if non_nan < 1000:
                            errors.append((fname, f"Only {non_nan} valid values"))
                            progress.update(task, description=f"{fname}: {non_nan} values ⚠️")
                        else:
                            factor_series[fname] = series
                            progress.update(task, description=f"Computed {fname} ✅ ({non_nan} values)")
                    else:
                        # Check stderr for errors
                        stderr = result.stderr[:200] if result.stderr else "Unknown"
                        errors.append((fname, f"No result.h5. Error: {stderr}"))
                        progress.update(task, description=f"{fname} ❌ (No result)")
                except subprocess.TimeoutExpired:
                    errors.append((fname, "Timeout (2 min)"))
                    progress.update(task, description=f"{fname} ⏱️ (Timeout)")
                except Exception as e:
                    errors.append((fname, str(e)[:100]))
                    progress.update(task, description=f"{fname} ❌ (Error)")
            
            progress.advance(task)

    # Show summary of errors
    if errors:
        console.print(f"\n[yellow]Skipped {len(errors)} factors:[/yellow]")
        for fname, reason in errors[:5]:
            console.print(f"  • {fname}: {reason}")
        if len(errors) > 5:
            console.print(f"  ... and {len(errors)-5} more")

    if len(factor_series) < 3:
        console.print("[red]Not enough valid factor series to build portfolio (need at least 3).[/red]")
        console.print("[yellow]Tip: Factors might be producing mostly NaN values or failing execution.[/yellow]")
        
        # Fallback: Show top factors by IC without diversification
        console.print("\n[dim]Showing top factors by IC instead:[/dim]")
        table = Table(
            title=f"Top {min(20, len(candidates))} Factors by IC (No Diversification)",
            show_header=True,
            header_style="bold cyan",
        )
        table.add_column("#", justify="center", width=4)
        table.add_column("Factor", width=40)
        table.add_column("IC", justify="right", width=10)
        table.add_column("Sharpe", justify="right", width=10)

        for i, cand in enumerate(candidates[:20], 1):
            table.add_row(
                str(i),
                cand.get("factor_name", "unknown")[:38],
                f"{cand.get('ic', 0):.6f}",
                f"{cand.get('sharpe', 0):.4f}" if cand.get('sharpe') else "N/A",
            )
        
        console.print(table)
        return

    # 3. Build Correlation Matrix
    console.print(f"\n[dim]Building correlation matrix from {len(factor_series)} factors...[/dim]")
    
    # Align indices and drop NaN
    combined = pd.DataFrame(factor_series).dropna()
    
    if combined.empty or len(combined) < 100:
        console.print("[red]Not enough valid overlapping data to compute correlation.[/red]")
        console.print("[dim]This means the factors produce values at different times or have too many NaN values.[/dim]")
        return

    corr_matrix = combined.corr().fillna(0)
    ic_map = {cand['factor_name']: cand.get('ic', 0) for cand in candidates}

    # 4. Greedy Selection
    selected = []
    remaining = list(corr_matrix.columns)
    
    # Sort remaining by IC to prioritize high IC factors
    remaining.sort(key=lambda x: abs(ic_map.get(x, 0)), reverse=True)

    for factor in remaining:
        if len(selected) >= target:
            break
        
        # If it's the first one, just take it
        if not selected:
            selected.append(factor)
            continue
        
        # Check correlation with already selected
        # We want max(|corr|) < max_corr
        max_c = 0
        for sel in selected:
            c = abs(corr_matrix.loc[factor, sel])
            if c > max_c:
                max_c = c
        
        if max_c < max_corr:
            selected.append(factor)

    # 5. Display Results
    table = Table(
        title=f"Selected Diversified Portfolio (Top {len(selected)})",
        show_header=True,
        header_style="bold cyan",
    )
    table.add_column("#", justify="center", width=4)
    table.add_column("Factor", width=40)
    table.add_column("IC", justify="right", width=10)
    table.add_column("Sharpe", justify="right", width=10)
    table.add_column("Max Corr", justify="right", width=10)

    for i, fname in enumerate(selected, 1):
        # Find original data for display
        data = next((c for c in candidates if c['factor_name'] == fname), {})
        ic = data.get('ic')
        sharpe = data.get('sharpe')
        
        # Calculate max corr with other selected factors
        max_c_val = 0
        for s in selected:
            if s != fname:
                val = abs(corr_matrix.loc[fname, s])
                if val > max_c_val: max_c_val = val

        table.add_row(
            str(i),
            fname[:38],
            f"{ic:.6f}" if ic is not None else "N/A",
            f"{sharpe:.4f}" if sharpe is not None else "N/A",
            f"{max_c_val:.4f}" if max_c_val > 0 else "-"
        )

    console.print(table)

    # 6. Save Result
    portfolio_data = {
        "selected_factors": selected,
        "max_correlation": max_corr,
        "pool_size": top,
        "timestamp": pd.Timestamp.now().isoformat()
    }
    
    out_dir = Path(__file__).parent / "results" / "portfolio"
    out_dir.mkdir(parents=True, exist_ok=True)
    out_file = out_dir / "selected_factors.json"
    
    with open(out_file, "w") as f:
        json.dump(portfolio_data, f, indent=2)
    
    console.print(Panel(
        f"[bold]Portfolio saved to results/portfolio/selected_factors.json[/bold]\n"
        f"Selected {len(selected)} unique factors from {top} candidates.",
        border_style="green"
    ))


@app.command()
def health():
    """Check system health and configuration."""
    from rdagent.app.utils.health_check import health_check
    health_check()


@app.command()
def status():
    """Show current trading loop status."""
    import sqlite3

    # Process check
    result = subprocess.run(
        ["pgrep", "-f", "fin_quant"],
        capture_output=True, text=True
    )
    if result.returncode == 0:
        console.print("[bold green]✅ Trading Loop: RUNNING[/bold green]")
    else:
        console.print("[bold yellow]⏸️  Trading Loop: STOPPED[/bold yellow]")

    # DB stats
    db_path = Path(__file__).parent / "results" / "db" / "backtest_results.db"
    if db_path.exists():
        conn = sqlite3.connect(str(db_path))
        c = conn.cursor()
        c.execute("SELECT COUNT(*) FROM backtest_runs")
        runs = c.fetchone()[0]
        c.execute("SELECT COUNT(*) FROM factors")
        factors = c.fetchone()[0]
        conn.close()

        console.print(f"\n📊 Results:")
        console.print(f"  Backtest runs: {runs}")
        console.print(f"  Factors: {factors}")


if __name__ == "__main__":
    app()
