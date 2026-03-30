#!/usr/bin/env python3
"""
Predix CLI Wrapper - Startet fin_quant mit Dashboard-Unterstützung

Verwendung:
  python predix.py fin_quant                  # Normal
  python predix.py fin_quant -d               # Web Dashboard
  python predix.py fin_quant -c               # CLI Dashboard
  python predix.py fin_quant -d -c            # Beide
  python predix.py fin_quant --help           # Hilfe
"""

import sys
import os
from pathlib import Path

# Parent-Directory zum Path hinzufügen
sys.path.insert(0, str(Path(__file__).parent))

# Environment laden
from dotenv import load_dotenv
load_dotenv(".env")

import subprocess
import threading
import time
from rich.console import Console

console = Console()

def start_web_dashboard(port=5000):
    """Starte Web Dashboard."""
    console.print(f"\n[bold green]🚀 Starting Web Dashboard on http://localhost:{port}...[/bold green]")
    console.print(f"   [cyan]Open: http://localhost:{port}/dashboard.html[/cyan]\n")
    subprocess.run(
        ["python", "web/dashboard_api.py"],
        cwd=str(Path(__file__).parent),
        env={**os.environ, "FLASK_ENV": "development"}
    )

def start_cli_dashboard():
    """Starte CLI Dashboard."""
    from rdagent.log.ui.predix_dashboard import run_dashboard
    run_dashboard(log_path="fin_quant.log", refresh_interval=3)

def fin_quant(path=None, step_n=None, loop_n=None, all_duration=None, checkout=True):
    """Starte fin_quant."""
    from rdagent.app.qlib_rd_loop.quant import main
    main(path=path, step_n=step_n, loop_n=loop_n, all_duration=all_duration, checkout=checkout)

def start_cli_dashboard_standalone():
    """
    Startet CLI Dashboard in einem SEPARATEN Terminal-Fenster.
    """
    import subprocess
    
    # Dashboard Script in neuem Terminal starten
    dashboard_script = Path(__file__).parent / "rdagent" / "log" / "ui" / "predix_dashboard.py"
    
    # Versuche verschiedene Terminal-Emulatoren
    terminal_commands = [
        ["gnome-terminal", "--", "python", str(dashboard_script)],
        ["konsole", "-e", "python", str(dashboard_script)],
        ["xterm", "-e", "python", str(dashboard_script)],
        ["tilix", "-e", "python", str(dashboard_script)],
    ]
    
    for cmd in terminal_commands:
        try:
            subprocess.Popen(cmd, start_new_session=True)
            console.print(f"[bold green]✓ Dashboard in neuem Terminal-Fenster gestartet[/bold green]")
            return True
        except FileNotFoundError:
            continue
    
    console.print("[yellow]⚠ Kein unterstütztes Terminal gefunden. Verwende Web Dashboard (-d) statt CLI.[/yellow]")
    return False


def main():
    import argparse
    
    parser = argparse.ArgumentParser(
        description="Predix EURUSD Trading",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  python predix.py fin_quant                      # Normal starten
  python predix.py fin_quant -d                   # Web Dashboard (empfohlen!)
  python predix.py fin_quant -c                   # CLI Dashboard (separates Terminal)
  python predix.py fin_quant -d -c                # Beide Dashboards
  python predix.py fin_quant --dashboard-port 5001  # Custom Port
        """
    )
    
    subparsers = parser.add_subparsers(dest='command', help='Commands')
    
    # fin_quant command
    fq_parser = subparsers.add_parser('fin_quant', help='Start EURUSD quantitative trading loop')
    fq_parser.add_argument('--path', type=str, default=None, help='Path')
    fq_parser.add_argument('--step-n', type=int, default=None, help='Number of steps')
    fq_parser.add_argument('--loop-n', type=int, default=None, help='Number of loops')
    fq_parser.add_argument('--all-duration', type=str, default=None, help='Duration')
    fq_parser.add_argument('--checkout', action='store_true', default=True, help='Checkout')
    fq_parser.add_argument('--no-checkout', action='store_false', dest='checkout', help='No checkout')
    fq_parser.add_argument('-d', '--with-dashboard', action='store_true', help='Start web dashboard')
    fq_parser.add_argument('-c', '--cli-dashboard', action='store_true', help='Start CLI dashboard in new terminal')
    fq_parser.add_argument('--dashboard-port', type=int, default=5000, help='Dashboard port')
    
    args = parser.parse_args()
    
    if args.command == 'fin_quant':
        # Start Web Dashboard wenn gewünscht
        if args.with_dashboard:
            dashboard_thread = threading.Thread(target=start_web_dashboard, args=(args.dashboard_port,), daemon=True)
            dashboard_thread.start()
            time.sleep(2)
            console.print(f"[bold green]✓ Web Dashboard gestartet: http://localhost:{args.dashboard_port}/dashboard.html[/bold green]")
        
        # Start CLI Dashboard in SEPARATEM Terminal wenn gewünscht
        if args.cli_dashboard:
            start_cli_dashboard_standalone()
            time.sleep(1)
        
        # Fin Quant starten
        console.print("\n[bold cyan]Starting fin_quant...[/bold cyan]\n")
        fin_quant(
            path=args.path,
            step_n=args.step_n,
            loop_n=args.loop_n,
            all_duration=args.all_duration,
            checkout=args.checkout
        )
    else:
        parser.print_help()

if __name__ == '__main__':
    main()
