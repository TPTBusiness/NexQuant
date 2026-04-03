"""
Trading Environment for RL Agents.

Gym-compatible environment for training RL trading agents.
Supports single-asset (EUR/USD) trading with technical indicators
and portfolio state as observations.

Inspired by common RL trading environment patterns, implemented from scratch for Predix.
"""

import gymnasium as gym
import numpy as np
import pandas as pd
from dataclasses import dataclass, field
from typing import Dict, List, Optional, Tuple


@dataclass
class TradingState:
    """Current state of the trading environment."""

    position: float = 0.0  # Current position (-1 to 1 for short/long)
    cash: float = 100000.0  # Available cash
    equity: float = 100000.0  # Total equity (cash + position value)
    entry_price: float = 0.0  # Entry price of current position
    step: int = 0  # Current time step
    holdings_history: List[float] = field(default_factory=list)  # Historical holdings


class TradingEnv(gym.Env):
    """
    Trading environment for RL agents.

    State: price history, technical indicators, portfolio state
    Action: position size (-1 to 1, short to long)
    Reward: risk-adjusted return with transaction costs

    Parameters
    ----------
    prices : np.ndarray
        Array of asset prices (1D)
    indicators : np.ndarray, optional
        Array of technical indicators (n_steps x n_features)
    initial_balance : float
        Starting cash balance
    transaction_cost : float
        Cost per unit of position change (e.g. 0.0001 = 1 basis point)
    window_size : int
        Lookback window for observations
    max_steps : int
        Maximum steps per episode

    Examples
    --------
    >>> prices = np.random.randn(1000) + 100
    >>> env = TradingEnv(prices, window_size=30)
    >>> obs, info = env.reset()
    >>> action = np.array([0.5])
    >>> obs, reward, terminated, truncated, info = env.step(action)
    """

    metadata = {"render_modes": ["human"]}

    def __init__(
        self,
        prices: np.ndarray,
        indicators: Optional[np.ndarray] = None,
        initial_balance: float = 100000.0,
        transaction_cost: float = 0.0001,
        window_size: int = 60,
        max_steps: int = 10000,
    ) -> None:
        super().__init__()

        self.prices = np.asarray(prices, dtype=np.float64)
        self.indicators = np.asarray(indicators, dtype=np.float32) if indicators is not None else None
        self.initial_balance = initial_balance
        self.transaction_cost = transaction_cost
        self.window_size = window_size
        self.max_steps = max_steps

        # Environment state (reset each episode)
        self.current_step: int = 0
        self.balance: float = initial_balance
        self.position: float = 0.0
        self.entry_price: float = 0.0
        self.equity_history: List[float] = [initial_balance]
        self.trades: List[Dict] = []

        # Observation space: window_size x (1 + n_indicators + 3)
        n_indicators = self.indicators.shape[1] if self.indicators is not None else 0
        obs_dim = window_size * (1 + n_indicators + 3)  # +3 for position, pnl, step
        self.observation_space = gym.spaces.Box(
            low=-np.inf, high=np.inf, shape=(obs_dim,), dtype=np.float32
        )

        # Action space: continuous position from -1 (short) to 1 (long)
        self.action_space = gym.spaces.Box(low=-1.0, high=1.0, shape=(1,), dtype=np.float32)

    # ------------------------------------------------------------------ #
    #  Observation
    # ------------------------------------------------------------------ #

    def _get_observation(self) -> np.ndarray:
        """Build observation vector from current state.

        Returns
        -------
        np.ndarray
            Flattened observation vector of shape (window_size * (1 + n_ind + 3),)
        """
        start_idx = self.current_step
        end_idx = start_idx + self.window_size

        # Price window
        price_window = self.prices[start_idx:end_idx]

        # Normalize prices relative to first value
        if len(price_window) == self.window_size and price_window[0] != 0:
            price_norm = price_window / price_window[0] - 1.0
        else:
            price_norm = np.zeros(self.window_size)
            if len(price_window) > 0 and price_window[0] != 0:
                valid_len = len(price_window)
                price_norm[-valid_len:] = price_window / price_window[0] - 1.0

        # Indicators window
        if self.indicators is not None:
            indicators_window = self.indicators[start_idx:end_idx]
            if len(indicators_window) < self.window_size:
                padded = np.zeros((self.window_size, self.indicators.shape[1]), dtype=np.float32)
                padded[-len(indicators_window):] = indicators_window
                indicators_window = padded
        else:
            indicators_window = np.zeros((self.window_size, 0), dtype=np.float32)

        # Portfolio state features (repeated for each timestep in window)
        current_price = float(self.prices[min(self.current_step, len(self.prices) - 1)])
        position_feature = np.full(self.window_size, self.position, dtype=np.float32)

        if self.entry_price > 0 and self.position != 0:
            pnl = (current_price - self.entry_price) / self.entry_price * np.sign(self.position)
        else:
            pnl = 0.0
        pnl_feature = np.full(self.window_size, pnl, dtype=np.float32)

        step_feature = np.full(self.window_size, self.current_step / max(self.max_steps, 1), dtype=np.float32)

        # Combine all features
        observation = np.column_stack(
            [
                price_norm.astype(np.float32),
                indicators_window,
                position_feature.reshape(-1, 1),
                pnl_feature.reshape(-1, 1),
                step_feature.reshape(-1, 1),
            ]
        )

        return observation.flatten().astype(np.float32)

    # ------------------------------------------------------------------ #
    #  Reward
    # ------------------------------------------------------------------ #

    def _calculate_reward(self, new_equity: float, old_equity: float) -> float:
        """Calculate risk-adjusted reward with penalties.

        Parameters
        ----------
        new_equity : float
            Equity after the step
        old_equity : float
            Equity before the step

        Returns
        -------
        float
            Reward value
        """
        # Simple return
        if old_equity > 0:
            simple_return = (new_equity - old_equity) / old_equity
        else:
            simple_return = 0.0

        # Transaction cost penalty
        position_change = abs(self.position) if len(self.trades) > 0 else 0.0
        cost_penalty = -position_change * self.transaction_cost

        # Drawdown penalty
        if len(self.equity_history) > 1:
            max_equity = max(self.equity_history)
            if max_equity > 0:
                drawdown = (max_equity - new_equity) / max_equity
            else:
                drawdown = 0.0
            drawdown_penalty = -drawdown * 2.0  # Heavy penalty for drawdowns
        else:
            drawdown_penalty = 0.0

        return float(simple_return + cost_penalty + drawdown_penalty)

    # ------------------------------------------------------------------ #
    #  Gym API: reset / step
    # ------------------------------------------------------------------ #

    def reset(
        self, seed: Optional[int] = None, options: Optional[dict] = None
    ) -> Tuple[np.ndarray, dict]:
        """Reset environment to initial state.

        Parameters
        ----------
        seed : int, optional
            Random seed for reproducibility
        options : dict, optional
            Additional reset options (unused)

        Returns
        -------
        observation : np.ndarray
            Initial observation
        info : dict
            Additional info
        """
        super().reset(seed=seed)

        self.current_step = 0
        self.balance = self.initial_balance
        self.position = 0.0
        self.entry_price = 0.0
        self.equity_history = [self.initial_balance]
        self.trades = []

        return self._get_observation(), {}

    def step(
        self, action: np.ndarray
    ) -> Tuple[np.ndarray, float, bool, bool, dict]:
        """Execute one time step in the environment.

        Parameters
        ----------
        action : np.ndarray
            Target position size in [-1, 1]

        Returns
        -------
        observation : np.ndarray
            Next observation
        reward : float
            Step reward
        terminated : bool
            Whether episode ended due to terminal condition
        truncated : bool
            Whether episode ended due to time limit
        info : dict
            Additional info
        """
        current_price = float(self.prices[min(self.current_step, len(self.prices) - 1)])
        old_equity = self.balance + self.position * current_price

        # Execute action
        target_position = float(np.clip(action[0], -1.0, 1.0))

        # Calculate transaction costs when position changes significantly
        if abs(target_position - self.position) > 0.01:
            cost = (
                abs(target_position - self.position)
                * current_price
                * self.transaction_cost
            )
            self.balance -= cost
            self.trades.append(
                {"step": self.current_step, "action": target_position, "cost": cost}
            )

        # Update position
        self.position = target_position

        if abs(self.position) > 0.01:
            self.entry_price = current_price

        # Advance time
        self.current_step += 1

        # Calculate new equity
        if self.current_step < len(self.prices):
            next_price = float(self.prices[self.current_step])
            new_equity = self.balance + self.position * next_price
        else:
            new_equity = self.balance

        self.equity_history.append(new_equity)

        # Reward
        reward = self._calculate_reward(new_equity, old_equity)

        # Termination conditions
        terminated = new_equity < self.initial_balance * 0.5  # Liquidation
        truncated = self.current_step >= min(self.max_steps, len(self.prices) - 1)

        observation = self._get_observation()

        info = {
            "equity": new_equity,
            "balance": self.balance,
            "position": self.position,
            "trades_count": len(self.trades),
            "return": (new_equity - self.initial_balance) / self.initial_balance
            if self.initial_balance > 0
            else 0.0,
        }

        return observation, reward, terminated, truncated, info

    # ------------------------------------------------------------------ #
    #  Utility
    # ------------------------------------------------------------------ #

    def get_equity_curve(self) -> np.ndarray:
        """Return equity curve for the current episode.

        Returns
        -------
        np.ndarray
            Equity values over time
        """
        return np.array(self.equity_history, dtype=np.float64)

    def get_trade_log(self) -> List[Dict]:
        """Return trade log for the current episode.

        Returns
        -------
        list[dict]
            List of trade records
        """
        return list(self.trades)
