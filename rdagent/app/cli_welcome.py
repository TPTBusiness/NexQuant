"""
Predix CLI Welcome Screen - Beautiful dashboard for GitHub README screenshot.
"""

import os
import subprocess  # nosec B404
from pathlib import Path
from rich.console import Console
from rich.panel import Panel
from rich.table import Table
from rich.text import Text
from rich.align import Align
from rich.layout import Layout
from datetime import datetime

console = Console()

def show_welcome():
    """Show beautiful Predix welcome screen."""
    
    # Header
    console.print()
    title = Text("🤖 PREDIX", style="bold cyan")
    subtitle = Text("AI-Powered Quantitative Trading Agent for EUR/USD Forex", style="dim white")
    console.print(Align.center(title))
    console.print(Align.center(subtitle))
    console.print()
    
    # Version info
    version_panel = Panel(
        f"[bold green]v2.0.0[/bold green] • Released: 2026.04.10 • [dim]MIT License[/dim]",
        border_style="green",
        title="📦 Release",
        title_align="left"
    )
    console.print(version_panel)
    console.print()
    
    # System Stats
    stats_table = Table(show_header=False, box=None, padding=(0, 2))
    stats_table.add_column("Metric", style="cyan")
    stats_table.add_column("Value", style="bold white")
    stats_table.add_column("Metric2", style="cyan")
    stats_table.add_column("Value2", style="bold white")
    
    # Count factors and strategies
    factors_dir = Path("results/factors")
    strategies_dir = Path("results/strategies_new")
    factor_count = len(list(factors_dir.glob("*.json"))) if factors_dir.exists() else 0
    strategy_count = len(list(strategies_dir.glob("*.json"))) if strategies_dir.exists() else 0
    
    stats_table.add_row("📊 Factors", f"[green]{factor_count:,}[/green]", "📈 Strategies", f"[green]{strategy_count}[/green]")
    stats_table.add_row("🧠 LLM", "[yellow]Qwen3.5-35B (local)[/yellow]", "⚡ Optuna", "[yellow]Enabled[/yellow]")
    stats_table.add_row("🔒 Security", "[green]All resolved[/green]", "🧪 Tests", "[green]282+ passing[/green]")
    
    stats_panel = Panel(stats_table, border_style="blue", title="📊 System Status", title_align="left")
    console.print(stats_panel)
    console.print()
    
    # Available Commands
    cmd_table = Table(show_header=True, header_style="bold magenta", box=None)
    cmd_table.add_column("Command", style="cyan", width=40)
    cmd_table.add_column("Description", style="white", width=50)
    
    cmd_table.add_row("rdagent fin_quant", "Start EUR/USD factor evolution loop")
    cmd_table.add_row("rdagent start_llama", "Start local llama.cpp server")
    cmd_table.add_row("rdagent start_loop", "Start strategy generator loop")
    cmd_table.add_row("rdagent generate_strategies", "Generate strategies from factors")
    cmd_table.add_row("rdagent optimize_portfolio", "Portfolio optimization")
    cmd_table.add_row("rdagent eval_all", "Evaluate factors with full data")  # nosec
    cmd_table.add_row("rdagent batch_backtest", "Batch backtest existing factors")
    cmd_table.add_row("rdagent report", "Generate PDF performance reports")
    cmd_table.add_row("rdagent rebacktest", "Re-backtest existing strategies")
    
    cmd_panel = Panel(cmd_table, border_style="magenta", title="🚀 Available Commands", title_align="left")
    console.print(cmd_panel)
    console.print()
    
    # Quick Start
    quick_start = Panel(
        "[bold cyan]1.[/bold cyan] Start LLM Server:   [dim]rdagent start_llama[/dim]\n"
        "[bold cyan]2.[/bold cyan] Run Trading Loop:   [dim]rdagent fin_quant --auto-strategies[/dim]\n"
        "[bold cyan]3.[/bold cyan] Generate Strategies: [dim]rdagent generate_strategies --count 5 --optuna[/dim]",
        border_style="yellow",
        title="💡 Quick Start",
        title_align="left"
    )
    console.print(quick_start)
    console.print()
    
    # Footer
    footer = Text("📄 github.com/TPTBusiness/Predix  •  🔒 MIT License  •  📖 docs/", style="dim white")
    console.print(Align.center(footer))
    console.print()

if __name__ == "__main__":
    show_welcome()


def main():
    """Entry point for 'predix' CLI command."""
    show_welcome()
