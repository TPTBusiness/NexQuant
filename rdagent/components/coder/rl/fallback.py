"""
Fallback RL implementation for users without stable-baselines3.

Provides simple rule-based trading when RL library is not available.
This ensures the NexQuant system works for all GitHub users, even
without the optional stable-baselines3 dependency.

The fallback implements a momentum-based strategy as a placeholder
for proper RL algorithms.
"""

import numpy as np
from typing import Optional, Any, Dict


class SimpleRLFallback:
    """
    Simple momentum-based trading as fallback when RL library unavailable.

    This is NOT a real RL algorithm. It provides basic functionality
    so the system doesn't break when stable-baselines3 is not installed.

    Strategy:
    - Positive momentum -> Long position
    - Negative momentum -> Short position
    - Zero momentum -> Hold

    Parameters
    ----------
    window_size : int
        Lookback window for momentum calculation
    momentum_threshold : float
        Threshold for entering positions (absolute value)
    max_position : float
        Maximum position size (-1 to 1)
    """

    def __init__(
        self,
        window_size: int = 20,
        momentum_threshold: float = 0.0,
        max_position: float = 1.0,
    ) -> None:
        self.window_size = window_size
        self.momentum_threshold = momentum_threshold
        self.max_position = max_position
        self.algorithm = "FALLBACK"  # Identify as fallback
        self.model = self  # Self-reference for compatibility
        self.is_trained = True  # Always "trained"

    def predict(
        self,
        observation: np.ndarray,
        deterministic: bool = True,
    ) -> np.ndarray:
        """
        Predict action from observation using momentum strategy.

        Parameters
        ----------
        observation : np.ndarray
            Observation vector (first window_size elements are prices)
        deterministic : bool
            Ignored (for API compatibility)

        Returns
        -------
        np.ndarray
            Action (-1 to 1)
        """
        # Extract prices from observation (first window_size elements)
        price_length = min(self.window_size, len(observation))
        prices = observation[:price_length]

        # Calculate momentum
        if len(prices) < 2 or prices[0] == 0:
            return np.array([0.0])

        momentum = (prices[-1] - prices[0]) / prices[0]

        # Apply threshold
        if abs(momentum) < self.momentum_threshold:
            return np.array([0.0])

        # Scale to position size
        position = np.clip(momentum, -self.max_position, self.max_position)

        return np.array([position])

    def learn(
        self,
        total_timesteps: int = 100000,
        *args: Any,
        **kwargs: Any,
    ) -> None:
        """
        No-op for compatibility with RLTradingAgent.train().

        The fallback doesn't actually train, it's a fixed strategy.
        """
        pass

    def save(self, path: str) -> None:
        """Save fallback config (no-op for compatibility)."""
        import json
        from pathlib import Path

        config = {
            "algorithm": "FALLBACK",
            "window_size": self.window_size,
            "momentum_threshold": self.momentum_threshold,
            "max_position": self.max_position,
        }

        path_obj = Path(path)
        path_obj.parent.mkdir(parents=True, exist_ok=True)

        # Save as JSON instead of model file
        with open(path_obj.with_suffix('.json'), 'w') as f:
            json.dump(config, f, indent=2)

    def load(self, path: str) -> None:
        """Load fallback config (no-op for compatibility)."""
        # Fallback doesn't need to load anything
        pass

    @staticmethod
    def is_available() -> bool:
        """
        Check if stable-baselines3 is available.

        Returns
        -------
        bool
            True if stable-baselines3 is installed
        """
        try:
            import stable_baselines3  # noqa: F401
            return True
        except ImportError:
            return False

    @staticmethod
    def get_recommendation() -> str:
        """
        Get recommendation for installing RL dependencies.

        Returns
        -------
        str
            Installation command
        """
        return "pip install stable-baselines3[extra] gymnasium"
