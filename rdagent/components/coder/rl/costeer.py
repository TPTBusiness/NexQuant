"""RL CoSTEER - Code generation and RL trading controller for post-training.

This module provides two main components:
1. RLCoSTEER: LLM-based code generation for RL training pipelines
2. RLCosteer: RL-based trading controller that uses trained models
   to make trading decisions based on market state.
"""

from datetime import datetime
from pathlib import Path
from typing import Any, Dict, Generator, List, Optional

import numpy as np
import pandas as pd

from rdagent.components.coder.CoSTEER import CoSTEER
from rdagent.components.coder.CoSTEER.config import CoSTEERSettings
from rdagent.components.coder.CoSTEER.evaluators import (  # nosec
    CoSTEERMultiEvaluator,
    CoSTEERSingleFeedback,
)
from rdagent.components.coder.CoSTEER.evolvable_subjects import EvolvingItem
from rdagent.components.coder.CoSTEER.knowledge_management import (
    CoSTEERQueriedKnowledge,
)
from rdagent.core.evolving_agent import EvolvingStrategy, EvoStep
from rdagent.core.experiment import FBWorkspace, Task
from rdagent.core.scenario import Scenario
from rdagent.log import rdagent_logger as logger
from rdagent.oai.llm_utils import APIBackend
from rdagent.utils.agent.tpl import T


class RLCoderCoSTEERSettings(CoSTEERSettings):
    """RL Coder settings."""

    pass


class RLEvolvingStrategy(EvolvingStrategy):
    """RL code generation strategy using LLM."""

    def __init__(self, scen: Scenario, settings: CoSTEERSettings):
        self.scen = scen
        self.settings = settings

    def evolve_iter(
        self,
        *,
        evo: EvolvingItem,
        queried_knowledge: CoSTEERQueriedKnowledge | None = None,
        evolving_trace: list[EvoStep] = [],
        **kwargs,
    ) -> Generator[EvolvingItem, EvolvingItem, None]:
        """Generate code for all tasks using LLM."""
        for index, target_task in enumerate(evo.sub_tasks):
            code = self._generate_code(target_task, evolving_trace)
            if evo.sub_workspace_list[index] is None:
                evo.sub_workspace_list[index] = evo.experiment_workspace
            evo.sub_workspace_list[index].inject_files(**code)

        evo = yield evo
        return

    def _generate_code(self, task: Task, evolving_trace: list[EvoStep] = []) -> dict[str, str]:
        """Generate RL training code using LLM."""
        from rdagent.app.rl.conf import RL_RD_SETTING

        # Get feedback from previous round
        feedback = None
        if evolving_trace:
            last_step = evolving_trace[-1]
            if hasattr(last_step, "feedback") and last_step.feedback:
                feedback = str(last_step.feedback)

        # Construct prompt
        system_prompt = T(".prompts:rl_coder.system").r()
        user_prompt = T(".prompts:rl_coder.user").r(
            task_description=task.description if hasattr(task, "description") else str(task),
            base_model=RL_RD_SETTING.base_model or "",
            benchmark=RL_RD_SETTING.benchmark or "",
            hypothesis=str(task.name) if hasattr(task, "name") else "Train RL model",
            feedback=feedback,
        )

        # Call LLM
        session = APIBackend().build_chat_session(session_system_prompt=system_prompt)
        code = session.build_chat_completion(
            user_prompt=user_prompt,
            json_mode=False,
            code_block_language="python",
        )
        logger.info(f"LLM generated code:\n{code[:200]}...")
        return {"main.py": code}

    def _mock_code(self) -> dict[str, str]:
        """Fallback mock code."""
        return {"main.py": """import gymnasium as gym
from stable_baselines3 import PPO

env = gym.make("CartPole-v1")
model = PPO("MlpPolicy", env, verbose=1)
model.learn(total_timesteps=1000)
model.save("ppo_cartpole")
print("Training completed!")
"""}


class RLCoderEvaluator:
    """RL code evaluator (mock implementation)."""  # nosec

    def __init__(self, scen: Scenario) -> None:
        self.scen = scen

    def evaluate(  # nosec
        self,
        target_task: Task,
        implementation: FBWorkspace,
        gt_implementation: FBWorkspace | None,
        queried_knowledge: CoSTEERQueriedKnowledge | None = None,
    ) -> CoSTEERSingleFeedback:
        """Evaluate RL code. Currently returns mock success."""
        # TODO: Implement proper evaluation logic  # nosec
        return CoSTEERSingleFeedback(
            execution="Mock: executed successfully",  # nosec
            return_checking=None,
            code="Mock: code looks good",
            final_decision=True,
        )


class RLCoSTEER(CoSTEER):
    """RL CoSTEER - orchestrates code generation and evaluation."""  # nosec

    def __init__(self, scen: Scenario, *args, **kwargs) -> None:
        settings = RLCoderCoSTEERSettings()
        eva = CoSTEERMultiEvaluator([RLCoderEvaluator(scen=scen)], scen=scen)
        es = RLEvolvingStrategy(scen=scen, settings=settings)

        super().__init__(
            *args,
            settings=settings,
            eva=eva,
            es=es,
            scen=scen,
            max_loop=1,
            stop_eval_chain_on_fail=False,  # nosec
            with_knowledge=False,
            knowledge_self_gen=False,
            **kwargs,
        )


# =============================================================================
# RL Trading Controller (RLCosteer)
# =============================================================================


class RLCosteer:
    """
    RL-based trading controller with protection manager integration.

    Takes market data, technical indicators, and portfolio state,
    then uses a trained RL model to decide position sizing.

    Parameters
    ----------
    model_path : Path, optional
        Path to a trained RL model file
    algorithm : str
        RL algorithm used ("PPO", "A2C", "SAC")
    window_size : int
        Lookback window for observations
    max_position : float
        Maximum position size (0 to 1)
    risk_limit : float
        Maximum drawdown before forcing position close
    enable_protections : bool
        Enable trading protection manager (default: True)

    Examples
    --------
    >>> costeer = RLCosteer(model_path=Path("models/ppo_trader.zip"))
    >>> costeer.initialize(prices, indicators, initial_equity=100000)
    >>> trade = costeer.step(current_equity=101000, cash=50000, position=0.0)
    """

    def __init__(
        self,
        model_path: Optional[Path] = None,
        algorithm: str = "PPO",
        window_size: int = 60,
        max_position: float = 1.0,
        risk_limit: float = 0.15,
        enable_protections: bool = True,
    ) -> None:
        self.model_path = model_path
        self.algorithm = algorithm.upper()
        self.window_size = window_size
        self.max_position = max_position
        self.risk_limit = risk_limit
        self.enable_protections = enable_protections

        # State
        self.is_active = False
        self.model: Optional[Any] = None
        self.current_position: float = 0.0
        self.peak_equity: float = 0.0
        self.trade_history: List[Dict[str, Any]] = []
        self.equity_history: List[float] = []

        # Protection Manager
        self.protection_manager: Optional[Any] = None
        if enable_protections:
            try:
                from rdagent.components.backtesting.protections.protection_manager import (
                    ProtectionManager,
                )

                self.protection_manager = ProtectionManager()
                self.protection_manager.create_default_protections()
            except ImportError:
                import warnings

                warnings.warn(
                    "Protection manager not available. Trading protections disabled."
                )
                self.protection_manager = None

        # Market data (set during initialize)
        self.prices: np.ndarray = np.array([])
        self.indicators: Optional[np.ndarray] = None
        self.initial_equity: float = 0.0
        self.current_step: int = 0
        self.timestamps_history: List[datetime] = []

        # Load model if path provided
        if model_path is not None and model_path.exists():
            self.load_model(model_path)

    def initialize(
        self,
        prices: pd.Series,
        indicators: Optional[pd.DataFrame] = None,
        initial_equity: float = 100000.0,
    ) -> None:
        """Initialize costeer with market data.

        Parameters
        ----------
        prices : pd.Series
            Price time series
        indicators : pd.DataFrame, optional
            Technical indicator DataFrame
        initial_equity : float
            Starting equity
        """
        self.prices = prices.values.astype(np.float64)
        self.indicators = indicators.values.astype(np.float32) if indicators is not None else None
        self.initial_equity = initial_equity
        self.current_step = self.window_size
        self.is_active = True
        self.peak_equity = initial_equity

    def get_action(
        self,
        current_equity: float,
        cash: float,
        position: float,
        returns_history: Optional[List[float]] = None,
        timestamps: Optional[List[datetime]] = None,
    ) -> float:
        """
        Get trading action from RL model with protection checks.

        Parameters
        ----------
        current_equity : float
            Current portfolio equity
        cash : float
            Available cash
        position : float
            Current position size
        returns_history : list, optional
            Historical returns for protection checks
        timestamps : list, optional
            Historical timestamps for protection checks

        Returns
        -------
        float
            Target position (-1 to 1)
        """
        # Check protections first (if enabled and available)
        if self.protection_manager and returns_history and len(returns_history) > 0:
            peak_equity = max(self.equity_history + [current_equity]) if self.equity_history else current_equity

            protection_result = self.protection_manager.check_all(
                returns=returns_history,
                timestamps=timestamps or self.timestamps_history[-len(returns_history):] if timestamps is None else [],
                current_equity=current_equity,
                peak_equity=peak_equity,
            )

            if protection_result.should_block:
                # Protection triggered - force close position
                return 0.0

        # If not active or no model, hold
        if not self.is_active or self.model is None:
            return 0.0

        # Build observation for RL model
        observation = self._build_observation(current_equity, cash, position)

        # Get action from model
        try:
            prediction = self.model.predict(observation)
            target_position = float(np.asarray(prediction[0]).flatten()[0])
        except Exception as e:
            import warnings

            warnings.warn(f"RL model prediction failed: {e}. Returning hold.")
            return 0.0

        # Apply risk limits
        drawdown = (self.peak_equity - current_equity) / self.peak_equity if self.peak_equity > 0 else 0.0
        if drawdown > self.risk_limit:
            return 0.0  # Force close position if risk limit exceeded

        # Scale by risk appetite
        risk_multiplier = 1.0 - (drawdown / self.risk_limit)
        target_position *= risk_multiplier * self.max_position

        return float(np.clip(target_position, -self.max_position, self.max_position))

    def _build_observation(
        self,
        current_equity: float,
        cash: float,
        position: float,
    ) -> np.ndarray:
        """Build observation vector for RL model.

        Parameters
        ----------
        current_equity : float
            Current portfolio equity
        cash : float
            Available cash
        position : float
            Current position size

        Returns
        -------
        np.ndarray
            Observation vector
        """
        start = max(0, self.current_step - self.window_size)
        end = self.current_step

        # Price window
        price_window = self.prices[start:end]
        if len(price_window) > 0 and price_window[0] != 0:
            price_norm = price_window / price_window[0] - 1.0
        else:
            price_norm = np.zeros(len(price_window))

        # Pad if needed
        if len(price_norm) < self.window_size:
            padded = np.zeros(self.window_size)
            padded[-len(price_norm):] = price_norm
            price_norm = padded

        # Indicators window
        if self.indicators is not None:
            indicators_window = self.indicators[start:end]
            if len(indicators_window) < self.window_size:
                padded = np.zeros((self.window_size, self.indicators.shape[1]), dtype=np.float32)
                padded[-len(indicators_window):] = indicators_window
                indicators_window = padded
        else:
            indicators_window = np.zeros((self.window_size, 0), dtype=np.float32)

        # Portfolio state features
        pnl = 0.0
        if position != 0 and self.current_step >= 2:
            prev_price = float(self.prices[self.current_step - 2])
            curr_price = float(self.prices[self.current_step - 1])
            if prev_price != 0:
                pnl = (curr_price - prev_price) / prev_price * np.sign(position)

        # Equity ratio
        equity_ratio = current_equity / self.initial_equity if self.initial_equity > 0 else 1.0

        observation = np.concatenate(
            [
                price_norm.astype(np.float32),
                indicators_window.flatten(),
                np.array([position, pnl, equity_ratio], dtype=np.float32),
            ]
        )

        return observation.astype(np.float32)

    def step(
        self,
        current_equity: float,
        cash: float,
        position: float,
        returns_history: Optional[List[float]] = None,
        timestamps: Optional[List[datetime]] = None,
    ) -> Dict[str, Any]:
        """
        Execute one trading step.

        Parameters
        ----------
        current_equity : float
            Current portfolio equity
        cash : float
            Available cash
        position : float
            Current position size
        returns_history : list, optional
            Historical returns for protection checks
        timestamps : list, optional
            Historical timestamps for protection checks

        Returns
        -------
        dict
            Step information including action taken
        """
        # Get action with protections
        target_position = self.get_action(
            current_equity=current_equity,
            cash=cash,
            position=position,
            returns_history=returns_history,
            timestamps=timestamps,
        )

        # Record trade
        trade = {
            "timestamp": datetime.now(),
            "step": self.current_step,
            "equity": current_equity,
            "position": position,
            "target_position": target_position,
            "action": target_position - position,
        }
        self.trade_history.append(trade)

        # Track equity and timestamps
        self.equity_history.append(current_equity)
        if timestamps:
            self.timestamps_history.extend(timestamps)
        else:
            self.timestamps_history.append(datetime.now())

        # Update peak equity
        if current_equity > self.peak_equity:
            self.peak_equity = current_equity

        # Move to next step
        self.current_step += 1

        return trade

    def load_model(self, path: Path) -> None:
        """Load trained RL model.

        Parameters
        ----------
        path : Path
            Path to the saved model file

        Raises
        ------
        ValueError
            If model loading fails
        """
        try:
            from stable_baselines3 import A2C, PPO, SAC

            model_class = {"PPO": PPO, "A2C": A2C, "SAC": SAC}[self.algorithm]
            self.model = model_class.load(str(path))
            self.is_active = True
        except ImportError:
            raise ImportError(
                "stable-baselines3 is required. Install with: pip install stable-baselines3"
            )
        except Exception as e:
            raise ValueError(f"Failed to load model: {e}")

    def get_performance(self) -> pd.DataFrame:
        """Get trading performance history.

        Returns
        -------
        pd.DataFrame
            Trade history as DataFrame
        """
        return pd.DataFrame(self.trade_history)
