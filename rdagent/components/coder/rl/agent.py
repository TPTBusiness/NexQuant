"""
RL Trading Agent wrapper for Stable Baselines3.

Provides an easy-to-use interface for training, evaluating, and deploying  # nosec
RL trading agents within the Predix framework.

Supported algorithms:
- PPO: Proximal Policy Optimization (most stable, recommended for production)
- A2C: Advantage Actor-Critic (faster training)
- SAC: Soft Actor-Critic (best for continuous action spaces)
"""

from pathlib import Path
from typing import Any, Dict, Optional, Union

import numpy as np


class RLTradingAgent:
    """
    Wrapper for RL trading agents built on Stable Baselines3.

    Parameters
    ----------
    algorithm : str
        RL algorithm to use ("PPO", "A2C", or "SAC")
    policy : str
        Policy network type (default: "MlpPolicy")
    params : dict, optional
        Algorithm-specific hyperparameters (merged with defaults)
    verbose : int
        Verbosity level (0 = silent, 1 = info, 2 = debug)

    Examples
    --------
    >>> agent = RLTradingAgent("PPO")
    >>> agent.train(env, total_timesteps=50000)
    >>> action = agent.predict(observation)
    >>> agent.save("models/rl_trader.zip")
    """

    _DEFAULT_PARAMS: Dict[str, Dict[str, Any]] = {
        "PPO": {
            "learning_rate": 3e-4,
            "n_steps": 2048,
            "batch_size": 64,
            "n_epochs": 10,
            "gamma": 0.99,
            "clip_range": 0.2,
            "ent_coef": 0.0,
        },
        "A2C": {
            "learning_rate": 7e-4,
            "n_steps": 5,
            "gamma": 0.99,
            "ent_coef": 0.01,
        },
        "SAC": {
            "learning_rate": 3e-4,
            "buffer_size": 1_000_000,
            "batch_size": 256,
            "gamma": 0.99,
            "tau": 0.005,
            "train_freq": 1,
            "gradient_steps": 1,
        },
    }

    def __init__(
        self,
        algorithm: str = "PPO",
        policy: str = "MlpPolicy",
        params: Optional[Dict[str, Any]] = None,
        verbose: int = 0,
    ) -> None:
        self.algorithm = algorithm.upper()
        self.policy = policy
        self.verbose = verbose

        # Merge user params with defaults
        defaults = self._DEFAULT_PARAMS.get(self.algorithm, {})
        self.params = {**defaults, **(params or {})}

        self.model: Optional[Any] = None
        self.is_trained = False

    # ------------------------------------------------------------------ #
    #  Internal helpers
    # ------------------------------------------------------------------ #

    def _get_model_class(self) -> Any:
        """Return the SB3 model class for the selected algorithm."""
        try:
            from stable_baselines3 import A2C, PPO, SAC

            model_map = {"PPO": PPO, "A2C": A2C, "SAC": SAC}
            if self.algorithm not in model_map:
                raise ImportError(
                    f"Unknown algorithm '{self.algorithm}'. "
                    f"Supported: {', '.join(model_map.keys())}"
                )
            return model_map[self.algorithm]
        except ImportError:
            raise

    # ------------------------------------------------------------------ #
    #  Public API
    # ------------------------------------------------------------------ #

    def create_model(self, env: Any) -> None:
        """Create a new RL model instance.

        Parameters
        ----------
        env : gym.Env
            Trading environment compatible with Gymnasium API
        """
        model_class = self._get_model_class()
        self.model = model_class(
            self.policy,
            env,
            verbose=self.verbose,
            **self.params,
        )

    def train(
        self,
        env: Any,
        total_timesteps: int = 100_000,
        tb_log_name: Optional[str] = None,
        progress_bar: bool = False,
    ) -> Dict[str, Any]:
        """
        Train the RL agent.

        Parameters
        ----------
        env : gym.Env
            Trading environment
        total_timesteps : int
            Number of training timesteps
        tb_log_name : str, optional
            TensorBoard log name
        progress_bar : bool
            Show progress bar during training

        Returns
        -------
        dict
            Training metadata
        """
        if self.model is None:
            self.create_model(env)

        if self.model is None:
            raise RuntimeError("Model creation failed unexpectedly")

        self.model.learn(
            total_timesteps=total_timesteps,
            tb_log_name=tb_log_name,
            progress_bar=progress_bar,
        )

        self.is_trained = True
        return {
            "algorithm": self.algorithm,
            "policy": self.policy,
            "total_timesteps": total_timesteps,
            "is_trained": True,
        }

    def predict(
        self,
        observation: np.ndarray,
        deterministic: bool = True,
    ) -> np.ndarray:
        """
        Predict action from observation.

        Parameters
        ----------
        observation : np.ndarray
            Current state observation vector
        deterministic : bool
            Use deterministic action (recommended for inference)

        Returns
        -------
        np.ndarray
            Action to take
        """
        if self.model is None:
            raise ValueError("Model not trained or loaded. Call train() or load() first.")

        action, _ = self.model.predict(observation, deterministic=deterministic)
        return np.asarray(action)

    def save(self, path: Union[str, Path]) -> None:
        """Save trained model to disk.

        Parameters
        ----------
        path : str or Path
            Destination file path (e.g. "models/ppo_trader.zip")
        """
        if self.model is None:
            raise ValueError("No model to save. Train first.")

        path = Path(path)
        path.parent.mkdir(parents=True, exist_ok=True)
        self.model.save(str(path))

    def load(self, path: Union[str, Path]) -> None:
        """Load a trained model from disk.

        Parameters
        ----------
        path : str or Path
            Source file path (e.g. "models/ppo_trader.zip")
        """
        model_class = self._get_model_class()
        self.model = model_class.load(str(path))
        self.is_trained = True

    def evaluate(  # nosec
        self,
        env: Any,
        n_episodes: int = 10,
        deterministic: bool = True,
    ) -> Dict[str, float]:
        """
        Evaluate agent performance over multiple episodes.

        Parameters
        ----------
        env : gym.Env
            Trading environment for evaluation  # nosec
        n_episodes : int
            Number of evaluation episodes  # nosec
        deterministic : bool
            Use deterministic actions during evaluation  # nosec

        Returns
        -------
        dict
            Evaluation metrics (mean/std of rewards and returns)
        """
        if self.model is None:
            raise ValueError("Model not trained or loaded.")

        rewards: list[float] = []
        returns: list[float] = []

        for _ in range(n_episodes):
            obs, _ = env.reset()
            episode_reward = 0.0
            done = False
            info: Dict[str, Any] = {}

            while not done:
                action = self.predict(obs, deterministic=deterministic)
                obs, reward, terminated, truncated, info = env.step(action)
                episode_reward += float(reward)
                done = terminated or truncated

            rewards.append(episode_reward)
            returns.append(float(info.get("return", 0.0)))

        return {
            "mean_reward": float(np.mean(rewards)),
            "std_reward": float(np.std(rewards)),
            "mean_return": float(np.mean(returns)),
            "std_return": float(np.std(returns)),
            "n_episodes": n_episodes,
        }
