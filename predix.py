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
    step_n: int = typer.Option(None, help="Number of steps to run"),
    loop_n: int = typer.Option(None, help="Number of loops to run"),
):
    """
    Start EURUSD quantitative trading loop.

    Examples:
        predix quant                          # Local llama.cpp
        predix quant -m openrouter            # OpenRouter cloud model
        predix quant -d                       # With web dashboard
        predix quant -m openrouter -d         # Both
    """
    import subprocess
    import threading
    import time

    # ---- LLM Model Selection ----
    if model == "openrouter":
        api_key = os.getenv("OPENROUTER_API_KEY", "")
        if not api_key:
            console.print("\n[bold red]❌ OPENROUTER_API_KEY not set in .env[/bold red]")
            console.print("[yellow]Add your API key to .env:[/yellow]")
            console.print('  OPENROUTER_API_KEY=sk-or-your-key-here')
            raise typer.Exit(code=1)

        os.environ["OPENAI_API_KEY"] = api_key
        os.environ["OPENAI_API_BASE"] = "https://openrouter.ai/api/v1"
        os.environ["CHAT_MODEL"] = os.getenv("OPENROUTER_MODEL", "openrouter/google/gemini-2.0-flash:free")

        console.print(f"\n[bold blue]🌐 Using OpenRouter:[/bold blue] [cyan]{os.environ['CHAT_MODEL']}[/cyan]")
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
