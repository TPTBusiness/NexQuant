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
    Start EUR/USD quantitative trading loop with LLM-powered factor generation.

    Executes the RD-Agent quantitative trading loop that uses large language models
    to generate, test, and iterate on alpha factors for EUR/USD trading. Supports
    both local llama.cpp inference and cloud-based OpenRouter models. Results are
    automatically logged and stored in the results directory.

    Args:
        model: LLM backend to use. 'local' for llama.cpp (requires local server
            running on OPENAI_API_BASE), 'openrouter' for cloud API. (default: "local")
        dashboard: If True, starts the Flask-based web dashboard on port 5000
            for real-time monitoring of the trading loop. (default: False)
        cli_dashboard: If True, starts the Rich-based CLI dashboard with a 3-second
            refresh interval for terminal-based monitoring. (default: False)
        log_file: Path for the log file. If None, auto-detects based on run_id
            (e.g., 'fin_quant.log' or 'fin_quant_run1.log'). Use 'none' to disable.
        step_n: Number of individual steps to execute within the loop. None means
            use the default from configuration.
        loop_n: Number of complete loops to run. Each loop generates and evaluates
            new alpha factors. None means use the default from configuration.
        run_id: Parallel run identifier for isolated execution. When > 0, creates
            separate log files, results directories, and workspace directories.
            0 = single run mode (default: 0)

    Examples:
        $ predix quant                          # Local llama.cpp, single run
        $ predix quant -m openrouter            # OpenRouter cloud model
        $ predix quant -d                       # With web dashboard on :5000
        $ predix quant -m openrouter -d         # Cloud model + web dashboard
        $ predix quant --run-id 1               # Parallel run #1 (isolated)
        $ predix quant --run-id 2 --loop-n 50   # Parallel run #2, 50 loops
        $ predix quant --log-file custom.log    # Custom log file path

    Expected Output:
        - Generated alpha factors saved to results/factors/ as JSON files
        - Backtest results stored in results/db/backtest_results.db
        - Log file created in project root (e.g., fin_quant.log)
        - Optional: Web dashboard at http://localhost:5000

    Estimated Time:
        ~5-15 minutes per loop depending on model and data size.
        Local models are faster but may have lower quality than cloud models.

    See Also:
        predix evaluate - Evaluate existing factors with full 1min data
        predix top - Show top-performing factors by IC or Sharpe
        predix health - Check system health and configuration
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

    # ---- Log File Setup (daily-rotated) ----
    from datetime import datetime as _dt
    _today = _dt.now().strftime("%Y-%m-%d")
    _daily_dir = Path(__file__).parent / "logs" / _today
    _daily_dir.mkdir(parents=True, exist_ok=True)

    if log_file.lower() != "none":
        log_path = _daily_dir / log_file
        # Open log file for appending (raw stdout/stderr capture)
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

        console.print(f"\n[dim]📝 Logging to: logs/{_today}/{log_file}[/dim]")
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
        os.environ["CHAT_MODEL"] = os.getenv("OPENROUTER_MODEL", "openrouter/google/gemma-4-26b-a4b-it:free")

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
    from rdagent.log.daily_log import session as _daily_session

    console.print(f"\n[bold cyan]📊 Starting EURUSD Trading Loop...[/bold cyan]\n")

    _ctx = {"model": model}
    if run_id:
        _ctx["run_id"] = run_id
    if loop_n:
        _ctx["loops"] = loop_n
    if step_n:
        _ctx["steps"] = step_n

    with _daily_session("fin_quant", **_ctx):
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
    Evaluate existing alpha factors with full 1-minute intraday data (2020-2026).

    Computes comprehensive performance metrics including Information Coefficient (IC),
    Sharpe Ratio, Maximum Drawdown, and Win Rate for each factor. Factors are loaded
    from JSON files in results/factors/ and executed against historical data to produce
    out-of-sample performance estimates. Already evaluated factors are automatically
    skipped unless --force is specified.

    Args:
        top: Number of unevaluated factors to process. Only applies when --all is
            not set. Higher values increase total runtime linearly. (default: 100)
        all_factors: If True, evaluates ALL unevaluated factors in the factors
            directory, ignoring the --top parameter. Use with caution as this
            may take hours for large factor sets. (default: False)
        parallel: Number of parallel worker processes for factor evaluation.
            Higher values speed up evaluation but increase memory usage.
            Recommended: 4-8 for most systems. (default: 4)
        force: If True, re-evaluates ALL factors including those that already
            have valid results. Useful when underlying data has changed or
            when recalculating with updated methodology. (default: False)

    Examples:
        $ predix evaluate                   # Evaluate 100 NEW factors
        $ predix evaluate --top 500         # Evaluate 500 NEW factors
        $ predix evaluate --all             # Evaluate all remaining factors
        $ predix evaluate --force --top 50  # Re-evaluate 50 factors
        $ predix evaluate -p 8              # Use 8 parallel workers

    Expected Output:
        - Updated JSON files in results/factors/ with IC, Sharpe, Max DD, Win Rate
        - Summary statistics printed to console
        - Factors with errors are logged and skipped gracefully

    Estimated Time:
        ~2-10 minutes per factor depending on complexity and data size.
        With --parallel 4, expect ~30-60 seconds per factor wall-clock time.

    See Also:
        predix top - Show top-performing factors by IC or Sharpe
        predix portfolio - Select a diversified portfolio of uncorrelated factors
        predix quant - Generate new factors via LLM trading loop
    """
    from rich.panel import Panel
    from rdagent.log.daily_log import session as _daily_session

    console.print(Panel(
        "[bold cyan]📊 Predix Factor Evaluator[/bold cyan]\n"
        "Evaluating factors with FULL 1min data (2020-2026)\n"
        "Skips already evaluated factors automatically",
        border_style="cyan",
    ))

    # Import and run the evaluator
    from predix_full_eval import main as eval_main

    _eval_ctx = {"top": "all" if all_factors else top, "workers": parallel}
    if force:
        _eval_ctx["force"] = True
    try:
        with _daily_session("evaluate", **_eval_ctx):
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
    Display top-performing alpha factors ranked by IC or Sharpe ratio.

    Loads all evaluated factor results from results/factors/ and presents them
    in a formatted table sorted by the chosen metric. Only factors with valid
    IC values (status='success') are included. This is useful for quickly
    identifying the most promising factors before building portfolios or strategies.

    Args:
        n: Number of top factors to display. Shows fewer if fewer exist in
            the results directory. (default: 20)
        metric: Sorting metric for ranking factors. 'ic' sorts by absolute
            Information Coefficient, 'sharpe' sorts by absolute Sharpe Ratio.
            IC measures predictive power, Sharpe measures risk-adjusted returns.
            (default: "ic")

    Examples:
        $ predix top                      # Top 20 factors by absolute IC
        $ predix top -n 50                # Top 50 factors by absolute IC
        $ predix top -m sharpe            # Top 20 factors by absolute Sharpe
        $ predix top -n 100 -m sharpe     # Top 100 factors by Sharpe

    Expected Output:
        - Formatted table showing Factor name, IC, Sharpe, Annualized Return,
          Max Drawdown, and Win Rate for each factor
        - Summary panel with average and best IC/Sharpe across all factors

    Estimated Time:
        Nearly instantaneous (< 1 second) for typical factor counts.
        May take a few seconds with thousands of factor files.

    See Also:
        predix evaluate - Evaluate factors to generate performance metrics
        predix portfolio - Select diversified portfolio from top factors
        predix build-strategies - Combine factors into trading strategies
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
    Select a diversified portfolio of uncorrelated alpha factors.

    Analyzes the top factors by IC and selects a subset that minimizes redundancy
    by calculating the correlation matrix of factor values. Uses a greedy selection
    algorithm that prioritizes high-IC factors while ensuring pairwise correlations
    stay below the specified threshold. This reduces overfitting risk and creates
    more robust composite signals.

    Args:
        top: Number of candidate factors to consider for portfolio construction.
            Factors are pre-selected by absolute IC before correlation analysis.
            Higher values provide more diversity but increase computation time.
            (default: 50)
        target: Number of factors to include in the final portfolio. The algorithm
            will attempt to select this many uncorrelated factors from the candidate
            pool. May return fewer if insufficient uncorrelated factors exist.
            (default: 10)
        max_corr: Maximum allowed absolute correlation between any two selected
            factors. Lower values produce more diverse portfolios but may exclude
            high-IC factors. Typical range: 0.2-0.5. (default: 0.3)

    Examples:
        $ predix portfolio                   # Select top 10 from top 50 candidates
        $ predix portfolio -n 100 -t 20      # Select top 20 from top 100
        $ predix portfolio -c 0.5            # Allow higher correlation (0.5)
        $ predix portfolio -n 200 -t 15 -c 0.2  # Strict diversification

    Expected Output:
        - Formatted table showing selected factors with IC, Sharpe, and max correlation
        - Portfolio saved to results/portfolio/selected_factors.json
        - Summary of skipped factors and errors (if any)

    Estimated Time:
        ~2-10 minutes depending on candidate count.
        Each factor must be re-evaluated to compute time-series values for correlation.

    See Also:
        predix portfolio-simple - Faster category-based diversification
        predix top - View top factors before portfolio selection
        predix build-strategies - Build strategies from selected factors
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
def portfolio_simple(
    top: int = typer.Option(
        100,
        "--top", "-n",
        help="Number of candidate factors to consider (default: 100)",
    ),
):
    """
    Select a diversified portfolio using keyword-based category grouping (fast method).

    Instead of computing expensive correlation matrices, this method groups factors
    by their names into categories (momentum, volatility, mean_reversion, session,
    volume, pattern) and selects the highest-IC factor from each category. This
    provides a quick approximation of diversification without re-evaluating factors.
    Falls back to 'other' category for factors that don't match any keywords.

    Args:
        top: Number of candidate factors to consider before categorization.
            Factors are pre-selected by absolute IC. Higher values increase
            the chance of finding factors in all categories. (default: 100)

    Examples:
        $ predix portfolio-simple              # Top factors from different categories
        $ predix portfolio-simple -n 200       # Consider top 200 factors
        $ predix portfolio-simple -n 50        # Quick selection from top 50

    Expected Output:
        - Formatted table showing selected factors with their category, IC, and Sharpe
        - Portfolio saved to results/portfolio/portfolio_simple.json
        - Categories include: Momentum, Volatility, Mean Reversion, Session,
          Volume, Pattern, and Other

    Estimated Time:
        Nearly instantaneous (< 1 second). No factor re-evaluation required.
        Only loads existing JSON results and performs keyword matching.

    See Also:
        predix portfolio - Correlation-based diversification (more accurate but slower)
        predix top - View top factors before portfolio selection
        predix build-strategies - Build strategies from selected factors
    """
    import json
    import glob as glob_module
    import re
    import numpy as np
    import pandas as pd
    from rich.table import Table
    from rich.panel import Panel

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

    # Sort by absolute IC
    results.sort(key=lambda x: abs(x.get("ic", 0) or 0), reverse=True)
    candidates = results[:top]

    # 2. Define categories based on keywords in factor names
    categories = {
        "momentum": ["mom", "return", "ret", "trend", "directional", "drift", "slope", "roc"],
        "volatility": ["vol", "std", "range", "dev", "risk", "variance"],
        "mean_reversion": ["ridge", "mean", "reversion", "revert", "resid", "resi", "norm"],
        "session": ["session", "london", "ny", "overlap", "asian", "intraday"],
        "volume": ["vol_", "volume", "flow", "pressure", "toxicity", "imbalance"],
        "pattern": ["pattern", "shape", "structure", "fractal"],
    }

    # 3. Assign each factor to a category
    categorized = {cat: [] for cat in categories}
    categorized["other"] = []

    for cand in candidates:
        fname = cand.get("factor_name", "").lower()
        assigned = False
        
        # Check each category's keywords
        for cat, keywords in categories.items():
            if any(kw in fname for kw in keywords):
                categorized[cat].append(cand)
                assigned = True
                break
        
        if not assigned:
            categorized["other"].append(cand)

    # 4. Select best factor from each category
    selected = []
    for cat in list(categories.keys()) + ["other"]:
        if categorized[cat]:
            best = categorized[cat][0]  # Already sorted by IC
            selected.append({
                "factor": best,
                "category": cat.capitalize() if cat != "other" else "Other"
            })

    # 5. Display Results
    table = Table(
        title=f"Simple Diversified Portfolio (Selected {len(selected)} factors)",
        show_header=True,
        header_style="bold cyan",
    )
    table.add_column("#", justify="center", width=4)
    table.add_column("Factor", width=40)
    table.add_column("Category", width=15)
    table.add_column("IC", justify="right", width=10)
    table.add_column("Sharpe", justify="right", width=10)

    for i, item in enumerate(selected, 1):
        cand = item["factor"]
        cat = item["category"]
        table.add_row(
            str(i),
            cand.get("factor_name", "unknown")[:38],
            cat,
            f"{cand.get('ic', 0):.6f}",
            f"{cand.get('sharpe', 0):.4f}" if cand.get('sharpe') else "N/A",
        )

    console.print(table)

    # 6. Save Result
    portfolio_data = {
        "selected_factors": [item["factor"]["factor_name"] for item in selected],
        "categories": {item["category"]: item["factor"]["factor_name"] for item in selected},
        "method": "simple_keyword_categorization",
        "timestamp": str(pd.Timestamp.now().isoformat())
    }

    out_dir = Path(__file__).parent / "results" / "portfolio"
    out_dir.mkdir(parents=True, exist_ok=True)
    out_file = out_dir / "portfolio_simple.json"

    with open(out_file, "w") as f:
        json.dump(portfolio_data, f, indent=2)

    console.print(Panel(
        f"[bold]Simple Portfolio saved to results/portfolio/portfolio_simple.json[/bold]\n"
        f"Selected {len(selected)} factors across {len([c for c in categorized if categorized[c]])} categories.",
        border_style="green"
    ))


@app.command()
def build_strategies(
    top: int = typer.Option(
        50,
        "--top", "-n",
        help="Number of top factors to consider (default: 50)",
    ),
    max_combo: int = typer.Option(
        2,
        "--max-combo", "-c",
        help="Maximum combination size: 2=pairs, 3=triplets (default: 2)",
    ),
    diversified: bool = typer.Option(
        False,
        "--diversified/-d",
        help="Only generate cross-category combinations",
    ),
):
    """
    Build trading strategies by systematically combining alpha factors.

    This command loads top evaluated factors, generates systematic combinations
    (pairs, triplets, etc.), and evaluates each combination using walk-forward
    validation. Results are ranked by Sharpe ratio and the best strategies are
    saved for later use. This is ideal for discovering synergies between factors
    that individually may have modest performance but work well together.

    Args:
        top: Number of top factors (by IC) to use as building blocks for
            strategy combinations. Higher values increase the number of
            combinations exponentially. (default: 50)
        max_combo: Maximum number of factors per combination. 2 creates only
            pairs, 3 creates pairs and triplets, etc. Higher values dramatically
            increase the combination count (n choose k). (default: 2)
        diversified: If True, only generates cross-category combinations,
            ensuring factors come from different groups (momentum, volatility,
            etc.). This reduces redundancy but may miss strong single-category
            strategies. (default: False)

    Examples:
        $ predix build-strategies                   # Build from top 50, pairs only
        $ predix build-strategies -n 100 -c 3       # Top 100, up to triplets
        $ predix build-strategies -d                # Diversified (cross-category) only
        $ predix build-strategies -n 30 -c 2 -d     # Top 30, diversified pairs

    Expected Output:
        - Formatted table of top strategies ranked by Sharpe ratio
        - Strategy files saved to results/strategies/
        - Summary with total combinations, success rate, avg/best Sharpe

    Estimated Time:
        ~1-5 minutes for pairs, ~10-30 minutes for triplets.
        Scales with O(n^k) where n=factors, k=max_combo_size.

    See Also:
        predix build-strategies-ai - AI-powered strategy generation via LLM
        predix portfolio - Select diversified factors before combining
        predix top - View top factors before building strategies
    """
    import pandas as pd
    import numpy as np
    from rich.table import Table
    from rich.panel import Panel

    from rdagent.scenarios.qlib.developer.strategy_builder import StrategyBuilder

    console.print(Panel(
        "[bold cyan]🏗️  Predix Strategy Builder[/bold cyan]\n"
        "Systematically combining factors into trading strategies",
        border_style="cyan",
    ))

    builder = StrategyBuilder()

    try:
        results = builder.build_strategies(
            top_n=top,
            max_combo_size=max_combo,
            diversified_only=diversified,
        )
    except Exception as e:
        console.print(f"[bold red]Strategy building failed: {e}[/bold red]")
        import traceback
        console.print(traceback.format_exc())
        return

    if not results:
        console.print("[yellow]No strategies built. Check if factor values exist.[/yellow]")
        return

    # Display top strategies
    successful = [r for r in results if r.get("status") == "success"]

    if successful:
        table = Table(
            title=f"Top {min(20, len(successful))} Strategies by Sharpe",
            show_header=True,
            header_style="bold cyan",
        )
        table.add_column("#", justify="center", width=4)
        table.add_column("Factors", width=50)
        table.add_column("Sharpe", justify="right", width=8)
        table.add_column("Ann. Ret %", justify="right", width=10)
        table.add_column("Max DD", justify="right", width=8)
        table.add_column("Win Rate", justify="right", width=8)

        for i, strat in enumerate(successful[:20], 1):
            factors_str = " + ".join(strat["factors"][:3])
            if len(strat["factors"]) > 3:
                factors_str += f" +{len(strat['factors'])-3}"

            table.add_row(
                str(i),
                factors_str,
                f"{strat.get('sharpe', 0):.4f}",
                f"{strat.get('annualized_return', 0):.4f}",
                f"{strat.get('max_drawdown', 0):.4f}",
                f"{strat.get('win_rate', 0):.2%}",
            )

        console.print(table)

        # Summary
        avg_sharpe = np.mean([s.get("sharpe", 0) for s in successful])
        best_sharpe = max(s.get("sharpe", 0) for s in successful)
        avg_dd = np.mean([s.get("max_drawdown", 0) for s in successful])

        console.print(Panel(
            f"[bold]Strategy Building Summary[/bold]\n"
            f"Total combinations: {len(results)}\n"
            f"Successful: {len(successful)}\n"
            f"Failed: {len(results) - len(successful)}\n"
            f"Avg Sharpe: {avg_sharpe:.4f}\n"
            f"Best Sharpe: {best_sharpe:.4f}\n"
            f"Avg Max DD: {avg_dd:.4f}\n"
            f"Saved to: results/strategies/",
            border_style="green",
        ))
    else:
        console.print("[yellow]No successful strategies. Check factor values exist.[/yellow]")


@app.command()
def build_strategies_ai(
    top: int = typer.Option(
        50,
        "--top", "-t",
        help="Number of top factors to use (default: 50)",
    ),
    max_loops: int = typer.Option(
        5,
        "--max-loops", "-l",
        help="Maximum improvement cycles (default: 5)",
    ),
    min_sharpe: float = typer.Option(
        1.5,
        "--min-sharpe",
        help="Minimum Sharpe ratio for acceptance (default: 1.5)",
    ),
    max_drawdown: float = typer.Option(
        -0.20,
        "--max-dd",
        help="Maximum acceptable drawdown (default: -0.20)",
    ),
    count: int = typer.Option(
        1,
        "--count", "-c",
        help="Number of strategies to generate (default: 1, use 0 for unlimited)",
    ),
):
    """
    Build trading strategies using AI-powered iterative improvement (StrategyCoSTEER).

    Uses a large language model to generate, test, and refine trading strategies
    from existing alpha factors. Follows the CoSTEER (Continuous Strategy
    Evolution via Evaluative Refinement) pattern: the LLM proposes strategy
    hypotheses and code, backtests are executed, results are fed back to the
    LLM for analysis and improvement, and the cycle repeats until acceptance
    criteria are met or max loops are reached. Requires OpenRouter API key.

    Args:
        top: Number of top factors (by IC) to provide as building blocks for
            the AI. The LLM will select from this pool to construct strategies.
            (default: 50)
        max_loops: Maximum number of improvement cycles per strategy. Each loop
            the LLM receives previous results and refines its approach. Higher
            values may find better strategies but cost more API calls. (default: 5)
        min_sharpe: Minimum Sharpe ratio threshold for strategy acceptance.
            Strategies below this threshold are rejected and the LLM attempts
            to improve them in subsequent loops. (default: 1.5)
        max_drawdown: Maximum acceptable drawdown threshold. Strategies exceeding
            this drawdown (more negative) are rejected. Expressed as a negative
            decimal (e.g., -0.20 = 20% max drawdown). (default: -0.20)
        count: Number of accepted strategies to generate. Set to 0 for unlimited
            mode (runs until max_batches or Ctrl+C). Each accepted strategy
            may require multiple improvement loops. (default: 1)

    Examples:
        $ predix build-strategies-ai                  # Generate 1 strategy, 5 loops max
        $ predix build-strategies-ai -t 100           # Use top 100 factors as pool
        $ predix build-strategies-ai -l 10            # Allow 10 improvement loops
        $ predix build-strategies-ai --min-sharpe 2.0 # Stricter Sharpe requirement
        $ predix build-strategies-ai --max-dd -0.15   # Tighter drawdown limit
        $ predix build-strategies-ai -c 5             # Generate 5 accepted strategies

    Expected Output:
        - Formatted table of accepted strategies with Sharpe, return, drawdown,
          win rate, and real IC from backtest
        - Strategy files saved to results/strategies/
        - Each strategy includes LLM-generated hypothesis and implementation code

    Estimated Time:
        ~5-20 minutes per accepted strategy depending on max_loops and backtest size.
        Each loop requires a full backtest execution plus LLM API calls.

    See Also:
        predix build-strategies - Systematic (non-AI) strategy combination
        predix quant - Generate new alpha factors via LLM trading loop
        predix evaluate - Evaluate factors before strategy building
    """
    from rich.panel import Panel
    from pathlib import Path

    console.print(Panel(
        "[bold cyan]🧠 StrategyCoSTEER - AI Strategy Builder[/bold cyan]\n"
        "Generating trading strategies from existing factors\n"
        "Uses LLM to combine factors, backtest, and improve",
        border_style="cyan",
    ))

    # Check if local module exists
    local_module = Path(__file__).parent / "rdagent" / "scenarios" / "qlib" / "local"
    if not local_module.exists():
        console.print("[bold red]❌ StrategyCoSTEER not available: local/ directory not found[/bold red]")
        console.print("[yellow]This is a closed-source feature. Contact development team.[/yellow]")
        return

    costeer_file = local_module / "strategy_coster.py"
    if not costeer_file.exists():
        console.print("[bold red]❌ strategy_coster.py not found[/bold red]")
        return

    # Load top factors
    factors_dir = Path(__file__).parent / "results" / "factors"

    # Setup LLM environment (same as quant command)
    api_key = os.getenv("OPENROUTER_API_KEY") or os.getenv("OPENAI_API_KEY", "")
    api_key_2 = os.getenv("OPENROUTER_API_KEY_2", "")
    
    if api_key and not api_key.startswith("sk-or-"):
        # OPENROUTER_API_KEY not set, try to use what we have
        api_key = os.getenv("OPENROUTER_API_KEY", api_key)

    if "openrouter" in os.getenv("CHAT_MODEL", "").lower() or "openrouter" in os.getenv("OPENAI_API_BASE", "").lower():
        # Already configured for OpenRouter
        console.print(f"\n[bold blue]🌐 Using OpenRouter: {os.getenv('CHAT_MODEL', 'unknown')}[/bold blue]")
    elif api_key:
        # Configure OpenRouter
        if api_key_2:
            os.environ["OPENAI_API_KEY"] = f"{api_key},{api_key_2}"
        else:
            os.environ["OPENAI_API_KEY"] = api_key
        os.environ["OPENAI_API_BASE"] = "https://openrouter.ai/api/v1"
        os.environ["CHAT_MODEL"] = os.getenv("OPENROUTER_MODEL", "openrouter/google/gemma-4-26b-a4b-it:free")
        console.print(f"\n[bold blue]🌐 Using OpenRouter: {os.environ['CHAT_MODEL']}[/bold blue]")
    else:
        console.print("[bold red]❌ No API key found. Set OPENROUTER_API_KEY in .env[/bold red]")
        return

    if not factors_dir.exists():
        console.print("[bold red]❌ No factors directory found at results/factors/[/bold red]")
        console.print("[yellow]Run 'predix quant' to generate factors first.[/yellow]")
        return

    # Load evaluated factors
    import json
    import glob as glob_module

    factors = []
    for f in glob_module.glob(str(factors_dir / "*.json")):
        try:
            with open(f) as fh:
                data = json.load(fh)
            if data.get("status") == "success" and data.get("ic") is not None:
                factors.append(data)
        except Exception:
            continue

    if len(factors) < 10:
        console.print(f"[bold red]❌ Only {len(factors)} evaluated factors found. Need at least 10.[/bold red]")
        console.print("[yellow]Run 'predix evaluate' or 'predix quant' to generate more factors.[/yellow]")
        return

    # Sort by IC and take top factors
    factors.sort(key=lambda x: abs(x.get("ic", 0) or 0), reverse=True)
    top_factors = factors[:top]

    console.print(f"\n[bold green]✓ Loaded {len(top_factors)} top factors[/bold green]")
    console.print(f"   Max loops: {max_loops}")
    console.print(f"   Target Sharpe: ≥ {min_sharpe}")
    console.print(f"   Max Drawdown: ≥ {max_drawdown:.2%}\n")

    # Run StrategyCoSTEER
    try:
        from rdagent.scenarios.qlib.local.strategy_coster import StrategyCoSTEER

        strategies_dir = Path(__file__).parent / "results" / "strategies"
        strategies_dir.mkdir(parents=True, exist_ok=True)

        costeer = StrategyCoSTEER(
            factors_dir=str(factors_dir),
            strategies_dir=str(strategies_dir),
            max_loops=max_loops,
            min_sharpe=min_sharpe,
            max_drawdown=max_drawdown,
        )

        # Generate strategies until we have enough
        all_results = []
        batch_idx = 0
        max_batches = count if count > 0 else 999  # Unlimited if count=0

        while len(all_results) < count or count == 0:
            if count == 0 and batch_idx >= max_batches:
                break  # Safety limit for unlimited mode
            if count > 0 and batch_idx >= count:
                break  # Already tried enough times

            batch_idx += 1
            console.print(f"\n[dim]━━━ Strategy Batch {batch_idx}/{count if count > 0 else '∞'} ━━━[/dim]")

            results = costeer.run(top_factors)
            all_results.extend(results)

            if count == 0:
                console.print(f"\n[dim]Generated {len(all_results)} strategies so far. Press Ctrl+C to stop.[/dim]")
            elif len(all_results) < count:
                console.print(f"\n[dim]Need {count - len(all_results)} more strategies...[/dim]")

        results = all_results[:count] if count > 0 else all_results  # Trim to exact count

        # Display results
        if results:
            console.print(f"\n[bold green]✓ Generated {len(results)} accepted strategies![/bold green]\n")

            from rich.table import Table
            table = Table(title="Accepted Strategies")
            table.add_column("#", style="dim")
            table.add_column("Strategy", style="cyan")
            table.add_column("Monthly %", justify="right", style="green")
            table.add_column("Trades", justify="right")
            table.add_column("Sharpe", justify="right")
            table.add_column("Max DD", justify="right", style="red")
            table.add_column("Win Rate", justify="right")
            table.add_column("Real IC", justify="right", style="magenta")
            table.add_column("Loop", justify="center")

            for i, r in enumerate(results, 1):
                # Monthly return: use real backtest if available, else estimate
                rb = r.get('real_backtest', {})
                if isinstance(rb, dict) and rb.get('status') == 'success':
                    monthly_pct = rb.get('monthly_return_pct', r.get('monthly_return_pct', 0))
                    n_trades = rb.get('n_trades', '-')
                    real_ic = rb.get('ic', 0)
                else:
                    monthly_pct = r.get('monthly_return_pct', r.get('real_monthly_return', 0))
                    n_trades = '-'
                    real_ic = rb.get('ic', 0) if isinstance(rb, dict) else 0

                table.add_row(
                    str(i),
                    r.get("strategy_name", "unknown")[:30],
                    f"{monthly_pct:.2f}%",
                    str(n_trades),
                    f"{r.get('sharpe', r.get('sharpe_ratio', 0)):.3f}",
                    f"{r.get('max_drawdown', r.get('est_max_drawdown', 0)):.2%}",
                    f"{r.get('win_rate', r.get('est_win_rate', 0)):.2%}",
                    f"{real_ic:.4f}" if real_ic else "-",
                    str(r.get("loop", "?")),
                )

            console.print(table)
            console.print(f"\n[dim]Strategies saved to: {strategies_dir}/[/dim]")
        else:
            console.print("[yellow]No strategies met acceptance criteria.[/yellow]")
            console.print("[dim]Check factor values in results/factors/values/[/dim]")

    except ImportError as e:
        console.print(f"[bold red]❌ Import failed: {e}[/bold red]")
    except Exception as e:
        console.print(f"[bold red]❌ Strategy building failed: {e}[/bold red]")
        import traceback
        console.print(traceback.format_exc())


@app.command()
def health():
    """Check system health and configuration status.

    Runs a comprehensive diagnostic check of the PREDIX trading system including
    Python version, installed dependencies, environment variables, database
    connectivity, data file availability, and LLM API configuration. This command
    helps identify setup issues before running computationally expensive operations.

    Examples:
        $ predix health                    # Run full system health check
        $ predix health --verbose          # Detailed output (if supported)

    Expected Output:
        - Python version and dependency status
        - Environment variable check (API keys, API base URLs)
        - Database connectivity test
        - Data file availability (OHLCV data)
        - LLM model connectivity test (if configured)
        - Overall health status: PASS or FAIL per check

    Estimated Time:
        ~5-15 seconds depending on network and database checks.

    See Also:
        predix status - Show current trading loop status and statistics
        predix quant - Main trading loop command
    """
    from rdagent.app.utils.health_check import health_check
    health_check()


@app.command()
def status():
    """Show current trading loop status and database statistics.

    Displays whether the quantitative trading loop (fin_quant) is currently
    running by checking active processes. Also connects to the SQLite results
    database and shows summary statistics including total backtest runs and
    number of evaluated factors. Useful for monitoring long-running sessions
    and verifying data persistence.

    Examples:
        $ predix status                    # Show current trading loop status
        $ predix status --json             # JSON output (if supported)

    Expected Output:
        - Trading loop process status: RUNNING or STOPPED
        - Number of backtest runs in database
        - Number of evaluated factors in database
        - Database file path

    Estimated Time:
        Nearly instantaneous (< 1 second).

    See Also:
        predix health - Check system health and configuration
        predix quant - Start the quantitative trading loop
        predix top - View top evaluated factors
    """
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
