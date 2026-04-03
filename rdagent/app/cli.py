"""
CLI entrance for all rdagent application.

This will
- make rdagent a nice entry and
- autoamtically load dotenv
"""

import os
import sys
from pathlib import Path

from dotenv import load_dotenv

load_dotenv(".env")
# 1) Make sure it is at the beginning of the script so that it will load dotenv before initializing BaseSettings.
# 2) The ".env" argument is necessary to make sure it loads `.env` from the current directory.

import subprocess
from importlib.resources import path as rpath
from typing import Optional

import typer
from rich.console import Console
from typing_extensions import Annotated

from rdagent.app.data_science.loop import main as data_science
from rdagent.app.finetune.llm.loop import main as llm_finetune
from rdagent.app.general_model.general_model import (
    extract_models_and_implement as general_model,
)
from rdagent.app.qlib_rd_loop.factor import main as fin_factor
from rdagent.app.qlib_rd_loop.factor_from_report import main as fin_factor_report
from rdagent.app.qlib_rd_loop.model import main as fin_model
from rdagent.app.qlib_rd_loop.quant import main as fin_quant
from rdagent.app.utils.health_check import health_check
from rdagent.app.utils.info import collect_info
from rdagent.log.mle_summary import grade_summary as grade_summary

app = typer.Typer()

CheckoutOption = Annotated[bool, typer.Option("--checkout/--no-checkout", "-c/-C")]
CheckEnvOption = Annotated[bool, typer.Option("--check-env/--no-check-env", "-e/-E")]
CheckDockerOption = Annotated[bool, typer.Option("--check-docker/--no-check-docker", "-d/-D")]
CheckPortsOption = Annotated[bool, typer.Option("--check-ports/--no-check-ports", "-p/-P")]


def ui(port=19899, log_dir="", debug: bool = False, data_science: bool = False):
    """
    start web app to show the log traces.
    """
    if data_science:
        with rpath("rdagent.log.ui", "dsapp.py") as app_path:
            cmds = ["streamlit", "run", app_path, f"--server.port={port}"]
            subprocess.run(cmds)
        return
    with rpath("rdagent.log.ui", "app.py") as app_path:
        cmds = ["streamlit", "run", app_path, f"--server.port={port}"]
        if log_dir or debug:
            cmds.append("--")
        if log_dir:
            cmds.append(f"--log_dir={log_dir}")
        if debug:
            cmds.append("--debug")
        subprocess.run(cmds)


def server_ui(port=19899):
    """
    start the Flask log server in real time
    """
    from rdagent.log.server.app import main as log_server_main

    log_server_main(port=port)


def ds_user_interact(port=19900):
    """
    start web app to show the log traces in real time
    """
    commands = ["streamlit", "run", "rdagent/log/ui/ds_user_interact.py", f"--server.port={port}"]
    subprocess.run(commands)


@app.command(name="fin_factor")
def fin_factor_cli(
    path: Optional[str] = None,
    step_n: Optional[int] = None,
    loop_n: Optional[int] = None,
    all_duration: Optional[str] = None,
    checkout: CheckoutOption = True,
):
    fin_factor(path=path, step_n=step_n, loop_n=loop_n, all_duration=all_duration, checkout=checkout)


@app.command(name="fin_model")
def fin_model_cli(
    path: Optional[str] = None,
    step_n: Optional[int] = None,
    loop_n: Optional[int] = None,
    all_duration: Optional[str] = None,
    checkout: CheckoutOption = True,
):
    fin_model(path=path, step_n=step_n, loop_n=loop_n, all_duration=all_duration, checkout=checkout)


@app.command(name="fin_quant")
def fin_quant_cli(
    path: Optional[str] = None,
    step_n: Optional[int] = None,
    loop_n: Optional[int] = None,
    all_duration: Optional[str] = None,
    checkout: CheckoutOption = True,
    with_dashboard: bool = typer.Option(False, "--with-dashboard/-d", help="Start web dashboard automatically"),
    with_cli_dashboard: bool = typer.Option(False, "--cli-dashboard/-c", help="Show beautiful CLI dashboard"),
    dashboard_port: int = typer.Option(5000, "--dashboard-port", help="Dashboard port"),
):
    """
    Start EURUSD quantitative trading loop.
    
    Options:
      --with-dashboard/-d: Start web dashboard at http://localhost:5000
      --cli-dashboard/-c: Show beautiful terminal UI with live stats
    
    Examples:
      rdagent fin_quant
      rdagent fin_quant -d              # Web dashboard
      rdagent fin_quant -c              # CLI dashboard
      rdagent fin_quant -d -c           # Both dashboards
    """
    import subprocess
    import threading
    import time
    
    # Start Web Dashboard wenn gewünscht
    if with_dashboard:
        def start_web_dashboard():
            console = Console()
            console.print(f"\n[bold green]🚀 Starting Web Dashboard on http://localhost:{dashboard_port}...[/bold green]")
            console.print(f"   [cyan]Open: http://localhost:{dashboard_port}/dashboard.html[/cyan]\n")
            subprocess.run(
                ["python", "web/dashboard_api.py"],
                cwd=str(Path(__file__).parent.parent.parent),
                env={**os.environ, "FLASK_ENV": "development"}
            )
        
        dashboard_thread = threading.Thread(target=start_web_dashboard, daemon=True)
        dashboard_thread.start()
        time.sleep(2)
    
    # Start CLI Dashboard wenn gewünscht
    if with_cli_dashboard:
        def start_cli_dash():
            from rdagent.log.ui.predix_dashboard import run_dashboard
            run_dashboard(log_path="fin_quant.log", refresh_interval=3)
        
        cli_thread = threading.Thread(target=start_cli_dash, daemon=True)
        cli_thread.start()
        time.sleep(1)
    
    # Fin Quant starten
    fin_quant(path=path, step_n=step_n, loop_n=loop_n, all_duration=all_duration, checkout=checkout)


@app.command(name="fin_factor_report")
def fin_factor_report_cli(
    report_folder: Optional[str] = None,
    path: Optional[str] = None,
    all_duration: Optional[str] = None,
    checkout: CheckoutOption = True,
):
    fin_factor_report(report_folder=report_folder, path=path, all_duration=all_duration, checkout=checkout)


@app.command(name="general_model")
def general_model_cli(report_file_path: str):
    general_model(report_file_path)


@app.command(name="data_science")
def data_science_cli(
    path: Optional[str] = None,
    checkout: CheckoutOption = True,
    step_n: Optional[int] = None,
    loop_n: Optional[int] = None,
    timeout: Optional[str] = None,
    competition: Optional[str] = None,
):
    data_science(
        path=path,
        checkout=checkout,
        step_n=step_n,
        loop_n=loop_n,
        timeout=timeout,
        competition=competition,
    )


@app.command(name="llm_finetune")
def llm_finetune_cli(
    path: Optional[str] = None,
    checkout: CheckoutOption = True,
    benchmark: Optional[str] = None,
    benchmark_description: Optional[str] = None,
    dataset: Optional[str] = None,
    base_model: Optional[str] = None,
    upper_data_size_limit: Optional[int] = None,
    step_n: Optional[int] = None,
    loop_n: Optional[int] = None,
    timeout: Optional[str] = None,
):
    llm_finetune(
        path=path,
        checkout=checkout,
        benchmark=benchmark,
        benchmark_description=benchmark_description,
        dataset=dataset,
        base_model=base_model,
        upper_data_size_limit=upper_data_size_limit,
        step_n=step_n,
        loop_n=loop_n,
        timeout=timeout,
    )


@app.command(name="grade_summary")
def grade_summary_cli(log_folder: str):
    grade_summary(log_folder)


app.command(name="ui")(ui)
app.command(name="server_ui")(server_ui)


@app.command(name="health_check")
def health_check_cli(
    check_env: CheckEnvOption = True,
    check_docker: CheckDockerOption = True,
    check_ports: CheckPortsOption = True,
):
    health_check(check_env=check_env, check_docker=check_docker, check_ports=check_ports)


@app.command(name="collect_info")
def collect_info_cli():
    collect_info()


app.command(name="ds_user_interact")(ds_user_interact)


@app.command(name="rl_trading")
def rl_trading_cli(
    mode: str = typer.Option("train", help="Mode: train, backtest, live"),
    algorithm: str = typer.Option("PPO", help="RL algorithm: PPO, A2C, SAC"),
    model_path: str = typer.Option(None, help="Path to trained model"),
    total_timesteps: int = typer.Option(100000, help="Training timesteps"),
    data_config: str = typer.Option("data_config.yaml", help="Data config file"),
    with_protections: bool = typer.Option(True, help="Enable trading protections"),
    n_episodes: int = typer.Option(10, help="Number of evaluation episodes"),
):
    """
    RL Trading Agent - Train and run reinforcement learning trading agents.

    Examples:
        # Train new RL agent
        rdagent rl_trading --mode train --algorithm PPO --total-timesteps 100000

        # Run backtest with trained model
        rdagent rl_trading --mode backtest --model-path models/rl_trader.zip

        # Run with protections disabled
        rdagent rl_trading --mode backtest --no-with-protections
    """
    from pathlib import Path
    import yaml

    console = Console()

    # Load config
    config_path = Path(data_config)
    config = {}
    if config_path.exists():
        with open(config_path) as f:
            config = yaml.safe_load(f) or {}

    console.print(f"\n[bold blue]🤖 RL Trading Agent[/bold blue]")
    console.print(f"Mode: [cyan]{mode}[/cyan]")
    console.print(f"Algorithm: [cyan]{algorithm.upper()}[/cyan]")
    console.print(f"Protections: {'[green]Enabled[/green]' if with_protections else '[red]Disabled[/red]'}")

    try:
        from rdagent.components.coder.rl import RLTradingAgent, RLCosteer, TradingEnv
    except ImportError as e:
        console.print(f"[bold red]Error: RL components not available.[/bold red]")
        console.print(f"Details: {e}")
        console.print(f"\n[yellow]Install RL dependencies:[/yellow]")
        console.print(f"  pip install stable-baselines3 gymnasium")
        raise typer.Exit(code=1)

    if mode == "train":
        console.print("\n[yellow]📊 Training RL agent...[/yellow]")
        console.print(f"  Algorithm: {algorithm.upper()}")
        console.print(f"  Timesteps: {total_timesteps:,}")

        try:
            # Create RL agent
            agent = RLTradingAgent(algorithm=algorithm.upper())

            # Load data for environment
            console.print("[dim]Loading market data...[/dim]")
            # TODO: Load actual data from config
            # For now, create mock environment
            import numpy as np
            import gymnasium as gym

            # Create simple mock environment for demonstration
            class MockTradingEnv(gym.Env):
                """Mock environment for demonstration."""
                def __init__(self):
                    super().__init__()
                    self.action_space = gym.spaces.Box(low=-1.0, high=1.0, shape=(1,))
                    self.observation_space = gym.spaces.Box(low=-np.inf, high=np.inf, shape=(63,))
                    self.current_step = 0
                    self.max_steps = 1000

                def reset(self, seed=None):
                    super().reset(seed=seed)
                    self.current_step = 0
                    return np.zeros(63, dtype=np.float32), {}

                def step(self, action):
                    self.current_step += 1
                    reward = np.random.randn() * 0.01
                    done = self.current_step >= self.max_steps
                    obs = np.random.randn(63).astype(np.float32)
                    return obs, reward, done, False, {}

            env = MockTradingEnv()
            console.print("[dim]Environment created (mock for demonstration)[/dim]")

            # Train
            console.print("[yellow]Starting training...[/yellow]")
            result = agent.train(env, total_timesteps=total_timesteps)

            # Save model
            model_path_out = Path("models") / f"rl_{algorithm.lower()}_trained.zip"
            model_path_out.parent.mkdir(parents=True, exist_ok=True)
            agent.save(model_path_out)

            console.print(f"\n[bold green]✅ Training complete![/bold green]")
            console.print(f"Model saved to: [cyan]{model_path_out}[/cyan]")
            console.print(f"Algorithm: {result['algorithm']}")
            console.print(f"Timesteps: {result['total_timesteps']:,}")

        except Exception as e:
            console.print(f"\n[bold red]❌ Training failed: {e}[/bold red]")
            raise typer.Exit(code=1)

    elif mode == "backtest":
        console.print("\n[yellow]📈 Running RL backtest...[/yellow]")

        if model_path:
            console.print(f"  Model: [cyan]{model_path}[/cyan]")
        else:
            console.print("[yellow]No model specified, using untrained agent[/yellow]")

        try:
            # Load agent
            if model_path:
                agent = RLTradingAgent(algorithm=algorithm.upper())
                agent.load(Path(model_path))
            else:
                agent = RLTradingAgent(algorithm=algorithm.upper())

            # Run backtest
            from rdagent.components.backtesting import FactorBacktester
            import pandas as pd
            import numpy as np

            backtester = FactorBacktester()

            # Mock data for demonstration
            console.print("[dim]Loading market data...[/dim]")
            n_steps = 500
            mock_prices = pd.Series(100 + np.cumsum(np.random.randn(n_steps) * 0.5))
            mock_indicators = pd.DataFrame({
                'rsi': np.random.uniform(30, 70, n_steps),
                'macd': np.random.randn(n_steps) * 0.1,
            })

            console.print("[yellow]Running backtest...[/yellow]")
            metrics = backtester.run_rl_backtest(
                rl_agent=agent,
                prices=mock_prices,
                indicators=mock_indicators,
                enable_protections=with_protections,
            )

            console.print(f"\n[bold green]✅ Backtest complete![/bold green]")
            console.print(f"  Final Equity: [green]${metrics.get('final_equity', 0):,.2f}[/green]")
            console.print(f"  Sharpe Ratio: {metrics.get('sharpe_ratio', 0):.3f}")
            console.print(f"  Max Drawdown: {metrics.get('max_drawdown', 0):.2%}")
            console.print(f"  Win Rate: {metrics.get('win_rate', 0):.2%}")

        except Exception as e:
            console.print(f"\n[bold red]❌ Backtest failed: {e}[/bold red]")
            import traceback
            console.print(f"[dim]{traceback.format_exc()}[/dim]")
            raise typer.Exit(code=1)

    elif mode == "live":
        console.print("\n[yellow]🔴 Starting live RL trading...[/yellow]")
        console.print("[bold red]⚠️  WARNING: Live trading carries real financial risk![/bold red]")

        if not model_path:
            console.print("[bold red]Error: Live trading requires a trained model (--model-path)[/bold red]")
            raise typer.Exit(code=1)

        try:
            # Load costeer with protections
            costeer = RLCosteer(
                model_path=Path(model_path),
                algorithm=algorithm.upper(),
                enable_protections=with_protections,
            )

            console.print(f"  Model: [cyan]{model_path}[/cyan]")
            console.print(f"  Algorithm: [cyan]{algorithm.upper()}[/cyan]")
            console.print(f"  Protections: {'[green]Enabled[/green]' if with_protections else '[red]Disabled[/red]'}")

            # TODO: Implement live trading loop
            console.print("\n[yellow]Live trading mode initialized.[/yellow]")
            console.print("[dim]Connect to your broker API to execute trades.[/dim]")
            console.print("[dim]See documentation for broker integration guide.[/dim]")

        except Exception as e:
            console.print(f"\n[bold red]❌ Live trading setup failed: {e}[/bold red]")
            raise typer.Exit(code=1)

    else:
        console.print(f"[bold red]Error: Unknown mode '{mode}'[/bold red]")
        console.print("Valid modes: train, backtest, live")
        raise typer.Exit(code=1)


if __name__ == "__main__":
    app()
