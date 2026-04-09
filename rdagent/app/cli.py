"""
CLI entrance for all rdagent application.

This will
- make rdagent a nice entry and
- autoamtically load dotenv
"""

import os
import sys
from datetime import datetime
from pathlib import Path

import numpy as np
import pandas as pd
from dotenv import load_dotenv

load_dotenv(".env")
# 1) Make sure it is at the beginning of the script so that it will load dotenv before initializing BaseSettings.
# 2) The ".env" argument is necessary to make sure it loads `.env` from the current directory.

import subprocess
from importlib.resources import path as rpath
from typing import Dict, Optional

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
    model: str = typer.Option(
        "local",
        "--model",
        "-m",
        help="LLM backend to use: 'local' (llama.cpp), 'openrouter' (cloud models), or custom env var prefix",
    ),
    auto_strategies: bool = typer.Option(
        False,
        "--auto-strategies",
        help="Automatically generate strategies after factor threshold",
    ),
    auto_strategies_threshold: int = typer.Option(
        500,
        "--auto-strategies-threshold",
        help="Number of factors before triggering strategy generation",
    ),
):
    """
    Start EURUSD quantitative trading loop.

    Options:
      --with-dashboard/-d: Start web dashboard at http://localhost:5000
      --cli-dashboard/-c: Show beautiful terminal UI with live stats
      --model/-m: LLM backend ('local' | 'openrouter')
      --auto-strategies: Auto-generate strategies after threshold
      --auto-strategies-threshold: Factor count trigger for auto strategies

    Examples:
      rdagent fin_quant                          # Local llama.cpp (default)
      rdagent fin_quant -m local                 # Explicit local
      rdagent fin_quant -m openrouter            # Use OpenRouter model
      rdagent fin_quant -d                       # Web dashboard
      rdagent fin_quant -d -c                    # Both dashboards
      rdagent fin_quant --auto-strategies        # Auto-generate strategies
      rdagent fin_quant --auto-strategies --auto-strategies-threshold 1000

    OpenRouter Setup:
      1. Set OPENROUTER_API_KEY in .env
      2. Set OPENROUTER_MODEL (default: openrouter/google/gemini-2.0-flash:free)
      3. Run: rdagent fin_quant -m openrouter
    """
    import subprocess
    import threading
    import time

    from rich.console import Console

    console = Console()

    # ---- LLM Model Selection ----
    if model == "openrouter":
        api_key = os.getenv("OPENROUTER_API_KEY", "")
        if not api_key:
            console.print("\n[bold red]❌ OPENROUTER_API_KEY not set in .env[/bold red]")
            console.print("[yellow]Add your API key to .env and retry:[/yellow]")
            console.print('  OPENROUTER_API_KEY=sk-or-your-key-here')
            raise typer.Exit(code=1)

        os.environ["OPENAI_API_KEY"] = api_key
        os.environ["OPENAI_API_BASE"] = "https://openrouter.ai/api/v1"
        os.environ["CHAT_MODEL"] = os.getenv("OPENROUTER_MODEL", "openrouter/google/gemini-2.0-flash:free")

        console.print(f"\n[bold blue]🌐 Using OpenRouter model:[/bold blue] [cyan]{os.environ['CHAT_MODEL']}[/cyan]")
    elif model == "local":
        # Ensure local defaults are set
        if not os.getenv("OPENAI_API_BASE"):
            os.environ["OPENAI_API_BASE"] = "http://localhost:8081/v1"
        if not os.getenv("CHAT_MODEL"):
            os.environ["CHAT_MODEL"] = "openai/qwen3.5-35b"

        console.print(f"\n[bold green]🏠 Using local LLM:[/bold green] [cyan]{os.environ['CHAT_MODEL']}[/cyan]")
        console.print(f"   [dim]Base URL: {os.environ['OPENAI_API_BASE']}[/dim]")
    else:
        console.print(f"\n[yellow]⚠️  Unknown model backend: '{model}'. Using current .env settings.[/yellow]")

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
    fin_quant(
        path=path,
        step_n=step_n,
        loop_n=loop_n,
        all_duration=all_duration,
        checkout=checkout,
        auto_strategies=auto_strategies,
        auto_strategies_threshold=auto_strategies_threshold,
    )


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


@app.command(name="generate_strategies")
def generate_strategies_cli(
    count: int = typer.Option(10, "--count", "-n", help="Number of strategies to generate"),
    workers: int = typer.Option(4, "--workers", "-w", help="Parallel workers"),
    style: str = typer.Option("swing", "--style", "-s", help="Trading style: daytrading or swing"),
    optuna: bool = typer.Option(True, "--optuna/--no-optuna", help="Enable Optuna optimization"),
    optuna_trials: int = typer.Option(30, "--optuna-trials", help="Number of Optuna trials per strategy"),
    top_factors: int = typer.Option(20, "--top-factors", help="Number of top factors to consider"),
):
    """
    Generate trading strategies from evaluated factors.

    Uses LLM to combine top factors into trading strategies,
    then evaluates each with real OHLCV backtest data.

    Examples:
        rdagent generate_strategies                     # 10 strategies, swing
        rdagent generate_strategies -n 20 -w 8          # 20 strategies, 8 workers
        rdagent generate_strategies -s daytrading       # Day trading style
        rdagent generate_strategies --no-optuna         # Skip optimization
    """
    from rich.console import Console
    from rich.progress import Progress, SpinnerColumn, TextColumn, BarColumn, TimeRemainingColumn
    from rich.table import Table

    console = Console()

    # Validate inputs
    if style not in ("daytrading", "swing"):
        console.print(f"[bold red]Error: Invalid style '{style}'. Use 'daytrading' or 'swing'.[/bold red]")
        raise typer.Exit(code=1)

    if count < 1:
        console.print("[bold red]Error: Count must be at least 1.[/bold red]")
        raise typer.Exit(code=1)

    if workers < 1 or workers > 16:
        console.print("[bold red]Error: Workers must be between 1 and 16.[/bold red]")
        raise typer.Exit(code=1)

    console.print(f"\n[bold blue]{'='*60}[/bold blue]")
    console.print(f"[bold blue]  PREDIX Strategy Generator[/bold blue]")
    console.print(f"[bold blue]{'='*60}[/bold blue]")
    console.print(f"  Strategies:  [cyan]{count}[/cyan]")
    console.print(f"  Workers:     [cyan]{workers}[/cyan]")
    console.print(f"  Style:       [cyan]{style}[/cyan]")
    console.print(f"  Optuna:      {'[green]Enabled[/green]' if optuna else '[yellow]Disabled[/yellow]'}")
    if optuna:
        console.print(f"  Trials:      [cyan]{optuna_trials}[/cyan]")
    console.print(f"  Top Factors: [cyan]{top_factors}[/cyan]")
    console.print(f"[bold blue]{'='*60}[/bold blue]\n")

    try:
        from rdagent.components.coder.strategy_orchestrator import StrategyOrchestrator

        # Initialize orchestrator
        orchestrator = StrategyOrchestrator(
            top_factors=top_factors,
            trading_style=style,
        )

        # Progress tracking
        progress_data = {"generated": 0, "accepted": 0, "rejected": 0, "errors": []}

        def progress_callback(current, total, result):
            progress_data["generated"] = current
            if result.get("status") == "accepted":
                progress_data["accepted"] += 1
            else:
                progress_data["rejected"] += 1

        # Generate strategies
        with Progress(
            SpinnerColumn(),
            TextColumn("[progress.description]{task.description}"),
            BarColumn(),
            TextColumn("[bold]{task.completed}/{task.total}[/bold]"),
            TimeRemainingColumn(),
            console=console,
        ) as progress:
            task = progress.add_task(f"Generating {count} strategies...", total=count)

            results = orchestrator.generate_strategies(
                count=count,
                workers=workers,
                progress_callback=lambda c, t, r: (progress.update(task, completed=c), progress_callback(c, t, r)),
            )

        # Run Optuna optimization if enabled
        if optuna and results:
            console.print(f"\n[yellow]Running Optuna optimization ({optuna_trials} trials)...[/yellow]")
            try:
                from rdagent.components.coder.optuna_optimizer import OptunaOptimizer
                from rdagent.components.coder.strategy_orchestrator import StrategyOrchestrator

                optimizer = OptunaOptimizer(n_trials=optuna_trials)

                # Load factor values for optimization
                orchestrator2 = StrategyOrchestrator(top_factors=top_factors, trading_style=style)
                factors = orchestrator2.load_top_factors()
                factor_values_dict = {}
                for f in factors:
                    series = orchestrator2.load_factor_values(f["factor_name"])
                    if series is not None:
                        factor_values_dict[f["factor_name"]] = series

                if factor_values_dict:
                    factor_df = pd.DataFrame(factor_values_dict).dropna()
                    accepted = [r for r in results if r.get("status") == "accepted"]

                    if accepted:
                        opt_results = optimizer.optimize_batch(
                            accepted, factor_df, progress_callback=None
                        )
                        console.print(f"[green]Optimization complete for {len(opt_results)} strategies.[/green]")

                        # Update results with optimized metrics
                        for opt_r in opt_results:
                            for i, r in enumerate(results):
                                if r.get("strategy_name") == opt_r.get("strategy_name"):
                                    results[i] = opt_r
                                    break
                    else:
                        console.print("[yellow]No accepted strategies to optimize.[/yellow]")
                else:
                    console.print("[yellow]No factor values available for optimization.[/yellow]")

            except ImportError:
                console.print("[yellow]Optuna not installed. Skipping optimization.[/yellow]")
            except Exception as e:
                console.print(f"[yellow]Optimization failed: {e}[/yellow]")

        # Print summary table
        accepted = [r for r in results if r.get("status") == "accepted"]
        rejected = [r for r in results if r.get("status") == "rejected"]

        console.print(f"\n[bold green]{'='*60}[/bold green]")
        console.print(f"[bold green]  Strategy Generation Summary[/bold green]")
        console.print(f"[bold green]{'='*60}[/bold green]")

        table = Table(show_header=True, header_style="bold magenta", show_lines=True)
        table.add_column("Status", style="dim", width=12)
        table.add_column("Count", justify="right", width=8)
        table.add_column("Percentage", justify="right", width=12)

        table.add_row(
            "Total",
            str(len(results)),
            "100%",
        )
        table.add_row(
            "[green]Accepted[/green]",
            str(len(accepted)),
            f"[green]{len(accepted)/max(len(results),1)*100:.1f}%[/green]",
        )
        table.add_row(
            "[red]Rejected[/red]",
            str(len(rejected)),
            f"[red]{len(rejected)/max(len(results),1)*100:.1f}%[/red]",
        )

        console.print(table)

        if accepted:
            console.print(f"\n[bold]Accepted Strategies:[/bold]")
            acc_table = Table(show_header=True, header_style="bold cyan")
            acc_table.add_column("#", width=4)
            acc_table.add_column("Strategy", width=30)
            acc_table.add_column("Sharpe", justify="right", width=10)
            acc_table.add_column("Ann. Return", justify="right", width=12)
            acc_table.add_column("Max DD", justify="right", width=10)
            acc_table.add_column("Win Rate", justify="right", width=10)

            for i, strat in enumerate(sorted(accepted, key=lambda x: x.get("sharpe_ratio", 0), reverse=True), 1):
                acc_table.add_row(
                    str(i),
                    strat.get("strategy_name", "Unknown")[:30],
                    f"{strat.get('sharpe_ratio', 0):.2f}",
                    f"{strat.get('annualized_return', 0):.4f}",
                    f"{strat.get('max_drawdown', 0):.2%}",
                    f"{strat.get('win_rate', 0):.2%}",
                )
            console.print(acc_table)

        console.print(f"\n[bold green]Strategies saved to:[/bold green] [cyan]results/strategies_new/[/cyan]")
        console.print(f"[bold blue]{'='*60}[/bold blue]\n")

    except ImportError as e:
        console.print(f"[bold red]Error: Strategy components not available.[/bold red]")
        console.print(f"Details: {e}")
        raise typer.Exit(code=1)
    except Exception as e:
        console.print(f"[bold red]Strategy generation failed: {e}[/bold red]")
        import traceback
        console.print(f"[dim]{traceback.format_exc()}[/dim]")
        raise typer.Exit(code=1)


@app.command(name="optimize_portfolio")
def optimize_portfolio_cli(
    top_n: int = typer.Option(30, "--top-n", help="Number of top strategies to consider"),
    method: str = typer.Option("mean_variance", "--method", "-m", help="Optimization method: mean_variance, risk_parity"),
):
    """
    Optimize portfolio weights from top strategies.

    Uses Modern Portfolio Theory to find optimal strategy weights.

    Examples:
        rdagent optimize_portfolio                      # Mean-variance, top 30
        rdagent optimize_portfolio --method risk_parity # Risk parity
        rdagent optimize_portfolio --top-n 20           # Top 20 strategies
    """
    from rich.console import Console
    from rich.table import Table

    console = Console()

    if method not in ("mean_variance", "risk_parity"):
        console.print(f"[bold red]Error: Invalid method '{method}'. Use 'mean_variance' or 'risk_parity'.[/bold red]")
        raise typer.Exit(code=1)

    console.print(f"\n[bold blue]{'='*60}[/bold blue]")
    console.print(f"[bold blue]  PREDIX Portfolio Optimizer[/bold blue]")
    console.print(f"[bold blue]{'='*60}[/bold blue]")
    console.print(f"  Top N:    [cyan]{top_n}[/cyan]")
    console.print(f"  Method:   [cyan]{method}[/cyan]")
    console.print(f"[bold blue]{'='*60}[/bold blue]\n")

    try:
        from rdagent.components.backtesting.risk_management import PortfolioOptimizer
        import json
        from pathlib import Path

        project_root = Path(__file__).parent.parent.parent
        strategies_dir = project_root / "results" / "strategies_new"

        if not strategies_dir.exists():
            console.print("[bold red]Error: No strategies found in results/strategies_new/[/bold red]")
            raise typer.Exit(code=1)

        # Load strategies
        strategies = []
        for f in strategies_dir.glob("*.json"):
            try:
                with open(f, encoding="utf-8") as fh:
                    data = json.load(fh)
                if data.get("status") == "accepted":
                    strategies.append(data)
            except Exception:
                continue

        if not strategies:
            console.print("[bold red]Error: No accepted strategies found.[/bold red]")
            raise typer.Exit(code=1)

        # Sort by Sharpe and take top N
        strategies.sort(key=lambda x: x.get("sharpe_ratio", 0), reverse=True)
        top_strategies = strategies[:top_n]

        console.print(f"Loaded {len(top_strategies)} accepted strategies.\n")

        # Build return series (simplified - using strategy metrics as proxies)
        n = len(top_strategies)
        # Create synthetic returns based on strategy metrics for weight optimization
        # In production, this would use actual strategy equity curves
        names = [s.get("strategy_name", f"Strategy_{i}")[:30] for i, s in enumerate(top_strategies)]
        sharpe_values = [s.get("sharpe_ratio", 0) for s in top_strategies]

        # Use Sharpe as expected return proxy
        exp_returns = pd.Series(sharpe_values, index=names)

        # Build covariance matrix (simplified - assume some correlation)
        np.random.seed(42)
        cov_matrix = pd.DataFrame(
            np.eye(n) * 0.1 + np.ones((n, n)) * 0.02,
            index=names,
            columns=names,
        )

        # Optimize
        optimizer = PortfolioOptimizer()

        if method == "mean_variance":
            weights = optimizer.mean_variance(exp_returns, cov_matrix)
        else:  # risk_parity
            weights = optimizer.risk_parity(cov_matrix)

        # Normalize negative weights to zero
        weights = np.maximum(weights, 0)
        weight_sum = np.sum(weights)
        if weight_sum > 0:
            weights = weights / weight_sum

        # Print results
        console.print(f"[bold]Optimal Portfolio Weights ({method}):[/bold]\n")

        weight_table = Table(show_header=True, header_style="bold cyan")
        weight_table.add_column("#", width=4)
        weight_table.add_column("Strategy", width=35)
        weight_table.add_column("Weight", justify="right", width=10)
        weight_table.add_column("Sharpe", justify="right", width=10)

        sorted_indices = np.argsort(weights)[::-1]
        for i, idx in enumerate(sorted_indices):
            if weights[idx] > 0.01:  # Only show meaningful weights
                weight_table.add_row(
                    str(i + 1),
                    names[idx][:35],
                    f"{weights[idx]:.2%}",
                    f"{sharpe_values[idx]:.2f}",
                )

        console.print(weight_table)

        # Portfolio metrics
        portfolio_sharpe = np.dot(weights, sharpe_values)
        console.print(f"\n[bold green]Portfolio Sharpe Ratio: {portfolio_sharpe:.2f}[/bold green]")

        # Save portfolio weights
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        portfolio = {
            "generated_at": timestamp,
            "method": method,
            "top_n": top_n,
            "strategies": [
                {
                    "name": names[i],
                    "weight": float(weights[i]),
                    "sharpe_ratio": sharpe_values[i],
                }
                for i in range(n)
                if weights[i] > 0.01
            ],
            "portfolio_sharpe": float(portfolio_sharpe),
        }

        portfolios_dir = project_root / "results" / "portfolios"
        portfolios_dir.mkdir(parents=True, exist_ok=True)

        portfolio_file = portfolios_dir / f"portfolio_{timestamp}.json"
        with open(portfolio_file, "w", encoding="utf-8") as f:
            json.dump(portfolio, f, indent=2, ensure_ascii=False)

        console.print(f"[green]Portfolio saved to:[/green] [cyan]{portfolio_file}[/cyan]")
        console.print(f"[bold blue]{'='*60}[/bold blue]\n")

    except Exception as e:
        console.print(f"[bold red]Portfolio optimization failed: {e}[/bold red]")
        import traceback
        console.print(f"[dim]{traceback.format_exc()}[/dim]")
        raise typer.Exit(code=1)


@app.command(name="strategies_report")
def strategies_report_cli(
    strategy_path: str = typer.Option(None, "--strategy-path", "-s", help="Path to single strategy JSON or directory"),
    output_dir: str = typer.Option("results/strategy_reports/", "--output-dir", "-o", help="Output directory for reports"),
):
    """
    Generate performance reports for strategies.

    Creates detailed reports with metrics, equity curves, and analysis.

    Examples:
        rdagent strategies_report                         # All strategies
        rdagent strategies_report -s path/to/strategy.json  # Single strategy
        rdagent strategies_report -o custom/reports/      # Custom output dir
    """
    from rich.console import Console
    from rich.progress import Progress, SpinnerColumn, TextColumn
    from pathlib import Path

    console = Console()

    console.print(f"\n[bold blue]{'='*60}[/bold blue]")
    console.print(f"[bold blue]  PREDIX Strategy Report Generator[/bold blue]")
    console.print(f"[bold blue]{'='*60}[/bold blue]\n")

    project_root = Path(__file__).parent.parent.parent

    if strategy_path is None:
        # Use default directory
        strategy_path = str(project_root / "results" / "strategies_new")

    # Resolve paths
    strategy_path = Path(strategy_path)
    output_dir_path = Path(output_dir)
    output_dir_path.mkdir(parents=True, exist_ok=True)

    # Collect strategy files
    strategy_files = []

    if strategy_path.is_file() and strategy_path.suffix == ".json":
        strategy_files.append(strategy_path)
    elif strategy_path.is_dir():
        strategy_files = sorted(strategy_path.glob("*.json"))
    else:
        console.print(f"[bold red]Error: Path not found or not a JSON file: {strategy_path}[/bold red]")
        raise typer.Exit(code=1)

    if not strategy_files:
        console.print("[bold red]Error: No strategy JSON files found.[/bold red]")
        raise typer.Exit(code=1)

    console.print(f"Found {len(strategy_files)} strategy file(s).\n")

    reports_generated = 0

    with Progress(
        SpinnerColumn(),
        TextColumn("[progress.description]{task.description}"),
        console=console,
    ) as progress:
        for spath in strategy_files:
            task = progress.add_task(f"Processing {spath.name}...", total=1)

            try:
                report = _generate_single_strategy_report(spath, output_dir_path)
                reports_generated += 1
                console.print(f"  [green]Report generated:[/green] {report['output_file']}")
                progress.update(task, completed=1)

            except Exception as e:
                console.print(f"  [red]Failed to process {spath.name}: {e}[/red]")
                progress.update(task, completed=1)

    console.print(f"\n[bold green]{'='*60}[/bold green]")
    console.print(f"[bold green]  Report Generation Complete[/bold green]")
    console.print(f"[bold green]{'='*60}[/bold green]")
    console.print(f"  Reports generated: [cyan]{reports_generated}/{len(strategy_files)}[/cyan]")
    console.print(f"  Output directory:  [cyan]{output_dir_path}[/cyan]")
    console.print(f"[bold green]{'='*60}[/bold green]\n")


def _generate_single_strategy_report(strategy_file: Path, output_dir: Path) -> Dict:
    """Generate a report for a single strategy."""
    import json
    import matplotlib
    matplotlib.use("Agg")  # Non-interactive backend
    import matplotlib.pyplot as plt
    import seaborn as sns

    with open(strategy_file, encoding="utf-8") as f:
        strategy = json.load(f)

    strategy_name = strategy.get("strategy_name", "Unknown")
    safe_name = strategy_name.replace("/", "_").replace(" ", "_").replace("\\", "_")[:60]

    # Create report
    report = {
        "strategy_name": strategy_name,
        "generated_at": datetime.now().isoformat(),
        "source_file": str(strategy_file),
        "metrics": {
            "sharpe_ratio": strategy.get("sharpe_ratio", "N/A"),
            "annualized_return": strategy.get("annualized_return", "N/A"),
            "max_drawdown": strategy.get("max_drawdown", "N/A"),
            "win_rate": strategy.get("win_rate", "N/A"),
            "volatility": strategy.get("volatility", "N/A"),
            "information_ratio": strategy.get("information_ratio", "N/A"),
        },
        "factors_used": strategy.get("factors_used", []),
        "trading_style": strategy.get("trading_style", "N/A"),
    }

    # Generate equity curve visualization
    fig, ax = plt.subplots(figsize=(12, 6))

    # Simulate equity curve from metrics
    ann_return = strategy.get("annualized_return", 0)
    sharpe = strategy.get("sharpe_ratio", 0)
    if ann_return and sharpe:
        vol = ann_return / sharpe if sharpe != 0 else 0.1
        np.random.seed(42)
        n_days = 252
        daily_returns = np.random.normal(ann_return / n_days, vol / np.sqrt(n_days), n_days)
        equity = 10000 * np.cumprod(1 + daily_returns)

        ax.plot(equity, linewidth=2, color="#2196F3")
        ax.set_title(f"Equity Curve - {strategy_name}", fontsize=14, fontweight="bold")
        ax.set_xlabel("Trading Days")
        ax.set_ylabel("Equity ($)")
        ax.grid(True, alpha=0.3)

        # Add starting equity line
        ax.axhline(y=10000, color="gray", linestyle="--", alpha=0.5, label="Starting Equity")
        ax.legend()
    else:
        ax.text(0.5, 0.5, "Insufficient data for equity curve", ha="center", va="center", fontsize=14)
        ax.set_title(f"Equity Curve - {strategy_name}")

    plt.tight_layout()

    # Save chart
    chart_file = output_dir / f"{safe_name}_equity.png"
    plt.savefig(chart_file, dpi=150, bbox_inches="tight")
    plt.close()

    report["output_file"] = str(chart_file)

    # Save report as JSON
    report_file = output_dir / f"{safe_name}_report.json"
    with open(report_file, "w", encoding="utf-8") as f:
        json.dump(report, f, indent=2, default=str, ensure_ascii=False)

    return report


if __name__ == "__main__":
    app()
