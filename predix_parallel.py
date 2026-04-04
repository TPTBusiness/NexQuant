"""
Predix Parallel Runner - Run multiple factor experiments concurrently.

Spawns N subprocesses, each running `predix.py quant` with isolated config:
- Separate log files (fin_quant_run1.log, fin_quant_run2.log, etc.)
- Separate result directories (results/runs/run1/, results/runs/run2/, etc.)
- Separate workspace directories
- API key distribution across multiple keys (round-robin)

Usage:
    python predix_parallel.py --runs 5 --api-keys 2
    python predix_parallel.py --runs 3 --model openrouter
    python predix_parallel.py --runs 5 --model local --api-keys 1
"""
import os
import signal
import subprocess
import sys
import time
from datetime import datetime
from pathlib import Path
from typing import Dict, List, Optional

from dotenv import load_dotenv
from rich.console import Console
from rich.live import Live
from rich.panel import Panel
from rich.table import Table
from rich.markdown import Markdown
from rich.layout import Layout
from rich.progress import Progress, SpinnerColumn, TextColumn, BarColumn, TaskProgressColumn

# Load environment variables from .env file
load_dotenv(Path(__file__).parent / ".env")

console = Console()


class RunState:
    """Tracks the state of a single parallel run."""

    def __init__(self, run_id: int, api_key_idx: int, model: str):
        self.run_id = run_id
        self.api_key_idx = api_key_idx
        self.model = model
        self.process: Optional[subprocess.Popen] = None
        self.status: str = "pending"  # pending, running, success, failed, stopped
        self.start_time: Optional[datetime] = None
        self.end_time: Optional[datetime] = None
        self.exit_code: Optional[int] = None
        self.error_message: Optional[str] = None
        self.log_file: str = f"fin_quant_run{run_id}.log"

    @property
    def elapsed(self) -> str:
        """Get elapsed time as human-readable string."""
        if self.start_time is None:
            return "--:--:--"
        end = self.end_time or datetime.now()
        delta = end - self.start_time
        total_seconds = int(delta.total_seconds())
        hours, remainder = divmod(total_seconds, 3600)
        minutes, seconds = divmod(remainder, 60)
        return f"{hours:02d}:{minutes:02d}:{seconds:02d}"

    @property
    def status_icon(self) -> str:
        """Get icon for current status."""
        icons = {
            "pending": "⏳",
            "running": "🔄",
            "success": "✅",
            "failed": "❌",
            "stopped": "⏹️",
        }
        return icons.get(self.status, "❓")


class ParallelRunner:
    """
    Manages multiple concurrent factor experiment runs.

    Spawns subprocesses with isolated configurations, monitors progress,
    and handles graceful shutdown.
    """

    def __init__(
        self,
        num_runs: int = 5,
        num_api_keys: int = 2,
        model: str = "openrouter",
    ):
        """
        Initialize parallel runner.

        Parameters
        ----------
        num_runs : int
            Number of concurrent runs to spawn
        num_api_keys : int
            Number of API keys to distribute across (1 or 2)
        model : str
            LLM backend: 'local' (llama.cpp) or 'openrouter' (cloud)
        """
        self.num_runs = num_runs
        self.num_api_keys = num_api_keys
        self.model = model
        self.runs: List[RunState] = []
        self.project_root = Path(__file__).parent
        self._shutdown_requested = False

        # Read API keys from environment
        self.api_keys = self._load_api_keys()

        # Validate we have enough API keys
        if self.model == "openrouter" and len(self.api_keys) < num_api_keys:
            console.print(
                f"[yellow]⚠️  Requested {num_api_keys} API keys, but only {len(self.api_keys)} found in .env[/yellow]"
            )
            console.print(
                f"[dim]Distributing across {len(self.api_keys)} available key(s)[/dim]"
            )
            self.num_api_keys = len(self.api_keys)

        # Initialize run states
        for i in range(1, num_runs + 1):
            # Round-robin API key assignment
            api_key_idx = (i - 1) % max(len(self.api_keys), 1)
            run_state = RunState(run_id=i, api_key_idx=api_key_idx, model=model)
            self.runs.append(run_state)

    def _load_api_keys(self) -> List[str]:
        """Load API keys from environment variables."""
        keys = []

        if self.model == "openrouter":
            key1 = os.getenv("OPENROUTER_API_KEY", "")
            key2 = os.getenv("OPENROUTER_API_KEY_2", "")
            if key1:
                keys.append(key1)
            if key2:
                keys.append(key2)
        else:
            # For local mode, we just need the llama.cpp endpoint
            keys.append("local")

        if not keys or (len(keys) == 1 and keys[0] == "local"):
            keys = ["local"]

        return keys

    def _build_env(self, run_state: RunState) -> Dict[str, str]:
        """
        Build isolated environment for a subprocess.

        Parameters
        ----------
        run_state : RunState
            The run state object containing run configuration

        Returns
        -------
        dict
            Environment variables dict for subprocess
        """
        # Start with a copy of current environment
        env = os.environ.copy()

        # Set parallel run ID for isolation
        env["PARALLEL_RUN_ID"] = str(run_state.run_id)

        # Set workspace isolation
        workspace_dir = self.project_root / f"RD-Agent_workspace_run{run_state.run_id}"
        env["RD_AGENT_WORKSPACE"] = str(workspace_dir)

        # Configure API key for this run
        if self.model == "openrouter" and run_state.api_key_idx < len(self.api_keys):
            api_key = self.api_keys[run_state.api_key_idx]
            env["OPENAI_API_KEY"] = api_key
            env["OPENAI_API_BASE"] = "https://openrouter.ai/api/v1"
            env["CHAT_MODEL"] = os.getenv("OPENROUTER_MODEL", "openrouter/qwen/qwen3.6-plus:free")

            # If we have 2 API keys, configure LiteLLM for load balancing
            if len(self.api_keys) >= 2:
                env["OPENAI_API_KEY"] = f"{self.api_keys[0]},{self.api_keys[1]}"
                env["LITELLM_PARALLEL_CALLS"] = "2"
        elif self.model == "local":
            env["OPENAI_API_KEY"] = os.getenv("OPENAI_API_KEY", "local")
            env["OPENAI_API_BASE"] = os.getenv("OPENAI_API_BASE", "http://localhost:8081/v1")
            env["CHAT_MODEL"] = os.getenv("CHAT_MODEL", "openai/qwen3.5-35b")

        return env

    def _build_command(self, run_state: RunState) -> List[str]:
        """
        Build the subprocess command to run predix quant.

        Parameters
        ----------
        run_state : RunState
            The run state object containing run configuration

        Returns
        -------
        list
            Command list for subprocess.Popen
        """
        cmd = [
            sys.executable,  # Use same Python interpreter
            str(self.project_root / "predix.py"),
            "quant",
            "--model", run_state.model,
            "--run-id", str(run_state.run_id),
            "--log-file", run_state.log_file,
        ]
        return cmd

    def _start_run(self, run_state: RunState) -> None:
        """
        Start a single run as a subprocess.

        Parameters
        ----------
        run_state : RunState
            The run state to start
        """
        env = self._build_env(run_state)
        cmd = self._build_command(run_state)

        # Ensure results directory exists
        results_dir = self.project_root / "results" / "runs" / f"run{run_state.run_id}"
        results_dir.mkdir(parents=True, exist_ok=True)

        # Open log file for appending
        log_path = self.project_root / run_state.log_file
        log_f = open(log_path, "a", encoding="utf-8")

        # Start subprocess
        run_state.process = subprocess.Popen(
            cmd,
            env=env,
            cwd=str(self.project_root),
            stdout=log_f,
            stderr=subprocess.STDOUT,
        )
        run_state.status = "running"
        run_state.start_time = datetime.now()

        console.print(
            f"[dim]  ▶️  Run {run_state.run_id} started (PID: {run_state.process.pid}, "
            f"API Key: {run_state.api_key_idx + 1}, Model: {run_state.model})[/dim]"
        )

    def _check_run(self, run_state: RunState) -> None:
        """
        Check if a run is still running and update status.

        Parameters
        ----------
        run_state : RunState
            The run state to check
        """
        if run_state.status != "running" or run_state.process is None:
            return

        poll_result = run_state.process.poll()
        if poll_result is not None:
            # Process has finished
            run_state.exit_code = poll_result
            run_state.end_time = datetime.now()

            if poll_result == 0:
                run_state.status = "success"
                console.print(
                    f"[bold green]  ✅ Run {run_state.run_id} completed "
                    f"({run_state.elapsed})[/bold green]"
                )
            else:
                run_state.status = "failed"
                run_state.error_message = f"Exit code: {poll_result}"
                console.print(
                    f"[bold red]  ❌ Run {run_state.run_id} failed "
                    f"({run_state.elapsed}, exit code: {poll_result})[/bold red]"
                )

    def _stop_run(self, run_state: RunState) -> None:
        """
        Gracefully stop a running subprocess.

        Parameters
        ----------
        run_state : RunState
            The run state to stop
        """
        if run_state.process is None or run_state.status != "running":
            return

        try:
            # Try graceful termination first
            run_state.process.terminate()
            try:
                run_state.process.wait(timeout=10)
            except subprocess.TimeoutExpired:
                # Force kill if not responding
                run_state.process.kill()
                run_state.process.wait()
        except Exception as e:
            console.print(f"[yellow]  ⚠️  Error stopping run {run_state.run_id}: {e}[/yellow]")

        run_state.status = "stopped"
        run_state.end_time = datetime.now()

    def _render_dashboard(self) -> Panel:
        """
        Render the live dashboard panel showing all run states.

        Returns
        -------
        Panel
            Rich Panel object with dashboard content
        """
        # Summary stats
        pending = sum(1 for r in self.runs if r.status == "pending")
        running = sum(1 for r in self.runs if r.status == "running")
        success = sum(1 for r in self.runs if r.status == "success")
        failed = sum(1 for r in self.runs if r.status == "failed")
        stopped = sum(1 for r in self.runs if r.status == "stopped")

        # Build summary table
        table = Table(
            title="🔀 Predix Parallel Run Dashboard",
            show_header=True,
            header_style="bold cyan",
            expand=True,
        )
        table.add_column("Run", justify="center", width=6)
        table.add_column("Status", justify="center", width=10)
        table.add_column("Elapsed", justify="center", width=10)
        table.add_column("API Key", justify="center", width=8)
        table.add_column("Model", justify="center", width=12)
        table.add_column("Exit", justify="center", width=6)
        table.add_column("Log File", justify="left")

        for run in self.runs:
            table.add_row(
                f"#{run.run_id}",
                f"{run.status_icon} {run.status}",
                run.elapsed,
                str(run.api_key_idx + 1),
                run.model,
                str(run.exit_code) if run.exit_code is not None else "--",
                run.log_file,
            )

        # Summary panel
        total = len(self.runs)
        summary_text = (
            f"**Summary:** {total} total | "
            f"{success} done | "
            f"{running} running | "
            f"{pending} pending | "
            f"{failed} failed"
        )

        if self._shutdown_requested:
            summary_text += "\n⚠️  **Shutdown requested - stopping all runs...**"

        from rich.console import Group
        return Group(table, Panel(Markdown(summary_text), border_style="blue"))

    def _signal_handler(self, signum, frame) -> None:
        """Handle SIGINT/SIGTERM for graceful shutdown."""
        if self._shutdown_requested:
            # Second Ctrl+C - force kill everything
            console.print("\n[bold red]🛑 Force killing all runs![/bold red]")
            for run in self.runs:
                if run.process and run.status == "running":
                    run.process.kill()
            sys.exit(1)

        self._shutdown_requested = True
        console.print("\n[yellow]⏹️  Shutdown requested - gracefully stopping all runs...[/yellow]")
        console.print("[dim]Press Ctrl+C again to force kill[/dim]")

        for run in self.runs:
            if run.status == "running":
                self._stop_run(run)

    def run(self) -> Dict[str, int]:
        """
        Execute all parallel runs and show live dashboard.

        Returns
        -------
        dict
            Summary with keys: total, success, failed, stopped
        """
        # Register signal handlers
        signal.signal(signal.SIGINT, self._signal_handler)
        signal.signal(signal.SIGTERM, self._signal_handler)

        console.print(f"\n[bold cyan]{'=' * 60}[/bold cyan]")
        console.print(f"[bold cyan]🔀 Predix Parallel Runner[/bold cyan]")
        console.print(f"[bold cyan]{'=' * 60}[/bold cyan]")
        console.print(f"  Runs: {self.num_runs}")
        console.print(f"  API Keys: {self.num_api_keys} ({len(self.api_keys)} available)")
        console.print(f"  Model: {self.model}")
        console.print(f"  Log pattern: fin_quant_run{{1..{self.num_runs}}}.log")
        console.print(f"  Results: results/runs/run{{1..{self.num_runs}}}/")
        console.print()

        # Start all runs
        for run in self.runs:
            if self._shutdown_requested:
                break
            self._start_run(run)
            # Small delay to prevent overwhelming the system
            time.sleep(1)

        # Monitor loop with live dashboard
        with Live(refresh_per_second=2, screen=True) as live:
            live.update(self._render_dashboard())
            while True:
                if self._shutdown_requested:
                    # Check if all runs are stopped
                    all_stopped = all(
                        r.status in ("success", "failed", "stopped", "pending")
                        for r in self.runs
                    )
                    if all_stopped:
                        break

                # Update all run statuses
                for run in self.runs:
                    self._check_run(run)

                # Check if all runs are complete
                all_done = all(
                    r.status in ("success", "failed", "stopped")
                    for r in self.runs
                )
                if all_done:
                    break

                live.update(self._render_dashboard())
                time.sleep(0.5)

        # Final summary
        success_count = sum(1 for r in self.runs if r.status == "success")
        failed_count = sum(1 for r in self.runs if r.status == "failed")
        stopped_count = sum(1 for r in self.runs if r.status == "stopped")

        console.print(f"\n[bold cyan]{'=' * 60}[/bold cyan]")
        console.print(f"[bold cyan]📊 Parallel Run Summary[/bold cyan]")
        console.print(f"[bold cyan]{'=' * 60}[/bold cyan]")
        console.print(f"  ✅ Success: {success_count}/{self.num_runs}")
        console.print(f"  ❌ Failed: {failed_count}/{self.num_runs}")
        if stopped_count > 0:
            console.print(f"  ⏹️  Stopped: {stopped_count}/{self.num_runs}")

        total_time = None
        for run in self.runs:
            if run.start_time and run.end_time:
                delta = run.end_time - run.start_time
                console.print(
                    f"    Run #{run.run_id}: {run.status} ({delta.total_seconds():.0f}s)"
                )

        return {
            "total": self.num_runs,
            "success": success_count,
            "failed": failed_count,
            "stopped": stopped_count,
        }


def main(
    runs: int = 5,
    api_keys: int = 2,
    model: str = "openrouter",
) -> Dict[str, int]:
    """
    Run multiple factor experiments in parallel.

    Parameters
    ----------
    runs : int
        Number of concurrent runs to spawn
    api_keys : int
        Number of API keys to distribute across
    model : str
        LLM backend: 'local' (llama.cpp) or 'openrouter' (cloud)

    Returns
    -------
    dict
        Summary with keys: total, success, failed, stopped
    """
    runner = ParallelRunner(num_runs=runs, num_api_keys=api_keys, model=model)
    return runner.run()


if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser(
        description="Predix Parallel Runner - Run multiple factor experiments concurrently"
    )
    parser.add_argument(
        "--runs", "-n",
        type=int,
        default=5,
        help="Number of concurrent runs (default: 5)",
    )
    parser.add_argument(
        "--api-keys", "-k",
        type=int,
        default=2,
        help="Number of API keys to distribute across (default: 2)",
    )
    parser.add_argument(
        "--model", "-m",
        type=str,
        default="openrouter",
        choices=["local", "openrouter"],
        help="LLM backend: 'local' (llama.cpp) or 'openrouter' (cloud)",
    )

    args = parser.parse_args()
    result = main(runs=args.runs, api_keys=args.api_keys, model=args.model)

    # Exit with appropriate code
    if result["failed"] > 0:
        sys.exit(1)
    sys.exit(0)
