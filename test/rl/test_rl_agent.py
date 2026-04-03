"""
Tests for RL Trading Agent wrapper.

Covers:
- Agent creation with different algorithms
- Parameter merging with defaults
- Model creation (mocked, since SB3 may not be installed)
- Predict/Save/Load error handling
- Evaluation (mocked)
"""

from pathlib import Path
from unittest.mock import MagicMock, patch

import numpy as np
import pytest

from rdagent.components.coder.rl.agent import RLTradingAgent


# =============================================================================
# FIXTURES
# =============================================================================


@pytest.fixture
def mock_env() -> MagicMock:
    """Create a mock gym environment."""
    env = MagicMock()
    env.observation_space = MagicMock()
    env.action_space = MagicMock()
    return env


@pytest.fixture
def mock_model() -> MagicMock:
    """Create a mock SB3 model."""
    model = MagicMock()
    model.predict.return_value = (np.array([0.5]), None)
    return model


# =============================================================================
# AGENT CREATION
# =============================================================================


class TestAgentCreation:
    """Test agent initialization."""

    def test_default_creation(self) -> None:
        """Default agent should use PPO with standard params."""
        agent = RLTradingAgent()
        assert agent.algorithm == "PPO"
        assert agent.policy == "MlpPolicy"
        assert agent.verbose == 0
        assert agent.model is None
        assert agent.is_trained is False

    def test_algorithm_uppercase(self) -> None:
        """Algorithm name should be uppercased."""
        agent = RLTradingAgent(algorithm="ppo")
        assert agent.algorithm == "PPO"

    def test_custom_params_merge(self) -> None:
        """User params should merge with defaults."""
        agent = RLTradingAgent(
            algorithm="PPO",
            params={"learning_rate": 1e-3, "gamma": 0.95},
        )
        # User param should override
        assert agent.params["learning_rate"] == 1e-3
        assert agent.params["gamma"] == 0.95
        # Default should remain
        assert "n_steps" in agent.params
        assert agent.params["n_steps"] == 2048

    def test_a2c_default_params(self) -> None:
        """A2C should have its own default params."""
        agent = RLTradingAgent(algorithm="A2C")
        assert agent.params["n_steps"] == 5
        assert agent.params["learning_rate"] == 7e-4

    def test_sac_default_params(self) -> None:
        """SAC should have its own default params."""
        agent = RLTradingAgent(algorithm="SAC")
        assert agent.params["buffer_size"] == 1_000_000
        assert agent.params["batch_size"] == 256


# =============================================================================
# MODEL CREATION (MOCKED)
# =============================================================================


class TestModelCreation:
    """Test model creation with mocked SB3."""

    @patch("stable_baselines3.PPO")
    def test_create_model_ppo(self, mock_ppo: MagicMock, mock_env: MagicMock) -> None:
        """PPO model should be created with correct params."""
        agent = RLTradingAgent(algorithm="PPO", params={"n_steps": 100})
        agent.create_model(mock_env)
        mock_ppo.assert_called_once()
        call_kwargs = mock_ppo.call_args.kwargs
        assert call_kwargs["verbose"] == 0

    @patch("stable_baselines3.A2C")
    def test_create_model_a2c(self, mock_a2c: MagicMock, mock_env: MagicMock) -> None:
        """A2C model should be created with correct params."""
        agent = RLTradingAgent(algorithm="A2C")
        agent.create_model(mock_env)
        mock_a2c.assert_called_once()

    @patch("stable_baselines3.SAC")
    def test_create_model_sac(self, mock_sac: MagicMock, mock_env: MagicMock) -> None:
        """SAC model should be created with correct params."""
        agent = RLTradingAgent(algorithm="SAC")
        agent.create_model(mock_env)
        mock_sac.assert_called_once()

    def test_model_class_not_found(self) -> None:
        """Invalid algorithm should raise ImportError."""
        agent = RLTradingAgent(algorithm="INVALID")
        with pytest.raises(ImportError, match="Unknown algorithm"):
            agent._get_model_class()


# =============================================================================
# TRAINING (MOCKED)
# =============================================================================


class TestTraining:
    """Test training with mocked model."""

    @patch("stable_baselines3.PPO")
    def test_train_returns_metadata(
        self, mock_ppo: MagicMock, mock_env: MagicMock
    ) -> None:
        """Training should return metadata dict."""
        mock_model = MagicMock()
        mock_ppo.return_value = mock_model

        agent = RLTradingAgent(algorithm="PPO")
        result = agent.train(mock_env, total_timesteps=1000)

        assert result["algorithm"] == "PPO"
        assert result["total_timesteps"] == 1000
        assert result["is_trained"] is True
        assert agent.is_trained is True

    @patch("stable_baselines3.PPO")
    def test_train_calls_learn(
        self, mock_ppo: MagicMock, mock_env: MagicMock
    ) -> None:
        """Training should call model.learn."""
        mock_model = MagicMock()
        mock_ppo.return_value = mock_model

        agent = RLTradingAgent(algorithm="PPO")
        agent.train(mock_env, total_timesteps=5000)

        mock_model.learn.assert_called_once()
        call_kwargs = mock_model.learn.call_args.kwargs
        assert call_kwargs["total_timesteps"] == 5000


# =============================================================================
# PREDICTION
# =============================================================================


class TestPrediction:
    """Test prediction functionality."""

    def test_predict_without_model_raises(self) -> None:
        """Predict without model should raise ValueError."""
        agent = RLTradingAgent()
        obs = np.random.randn(120).astype(np.float32)
        with pytest.raises(ValueError, match="not trained or loaded"):
            agent.predict(obs)

    def test_predict_returns_action(self, mock_model: MagicMock) -> None:
        """Predict should return action array."""
        agent = RLTradingAgent()
        agent.model = mock_model

        obs = np.random.randn(120).astype(np.float32)
        action = agent.predict(obs)

        assert isinstance(action, np.ndarray)
        mock_model.predict.assert_called_once_with(obs, deterministic=True)

    def test_predict_non_deterministic(self, mock_model: MagicMock) -> None:
        """Predict with deterministic=False should pass flag."""
        agent = RLTradingAgent()
        agent.model = mock_model

        obs = np.random.randn(120).astype(np.float32)
        agent.predict(obs, deterministic=False)

        mock_model.predict.assert_called_once_with(obs, deterministic=False)


# =============================================================================
# SAVE / LOAD (MOCKED)
# =============================================================================


class TestSaveLoad:
    """Test model save and load."""

    def test_save_without_model_raises(self, tmp_path: Path) -> None:
        """Save without model should raise ValueError."""
        agent = RLTradingAgent()
        with pytest.raises(ValueError, match="No model to save"):
            agent.save(tmp_path / "model.zip")

    @patch("stable_baselines3.PPO")
    def test_save_creates_directory(self, mock_ppo: MagicMock, tmp_path: Path) -> None:
        """Save should create parent directories."""
        mock_model = MagicMock()
        mock_ppo.return_value = mock_model

        agent = RLTradingAgent(algorithm="PPO")
        agent.model = mock_model

        save_path = tmp_path / "subdir" / "model.zip"
        agent.save(save_path)

        mock_model.save.assert_called_once_with(str(save_path))

    @patch("stable_baselines3.PPO")
    def test_load_sets_trained_flag(self, mock_ppo: MagicMock, tmp_path: Path) -> None:
        """Load should set is_trained to True."""
        mock_model_class = MagicMock()
        mock_ppo.load.return_value = MagicMock()
        mock_ppo.return_value = mock_model_class

        agent = RLTradingAgent(algorithm="PPO")
        agent.load(tmp_path / "model.zip")

        assert agent.is_trained is True


# =============================================================================
# EVALUATION (MOCKED)
# =============================================================================


class TestEvaluation:
    """Test evaluation functionality."""

    def test_evaluate_without_model_raises(self, mock_env: MagicMock) -> None:
        """Evaluate without model should raise ValueError."""
        agent = RLTradingAgent()
        with pytest.raises(ValueError, match="not trained or loaded"):
            agent.evaluate(mock_env)

    @patch("stable_baselines3.PPO")
    def test_evaluate_returns_metrics(
        self, mock_ppo: MagicMock, mock_env: MagicMock
    ) -> None:
        """Evaluate should return metrics dict."""
        mock_model = MagicMock()
        mock_model.predict.return_value = (np.array([0.3]), None)
        mock_ppo.return_value = mock_model

        # Mock env.reset and env.step
        mock_env.reset.return_value = (np.random.randn(120).astype(np.float32), {})
        mock_env.step.return_value = (
            np.random.randn(120).astype(np.float32),
            0.1,  # reward
            False,  # terminated
            True,  # truncated (end after 1 step for simplicity)
            {"return": 0.05},
        )

        agent = RLTradingAgent(algorithm="PPO")
        agent.model = mock_model

        metrics = agent.evaluate(mock_env, n_episodes=3)

        assert "mean_reward" in metrics
        assert "std_reward" in metrics
        assert "mean_return" in metrics
        assert "std_return" in metrics
        assert metrics["n_episodes"] == 3
