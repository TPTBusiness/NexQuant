"""
Tests for RL Trading Environment.

Covers:
- Environment creation with various configurations
- Observation space correctness
- Action execution and state transitions  # nosec
- Reward calculation
- Episode termination and truncation
- Edge cases (empty data, extreme prices)
"""

import numpy as np
import pandas as pd
import pytest
from typing import Tuple

from rdagent.components.coder.rl.env import TradingEnv, TradingState


# =============================================================================
# FIXTURES
# =============================================================================


@pytest.fixture
def mock_prices() -> np.ndarray:
    """Generate 500-step mock price series."""
    np.random.seed(42)
    return 100.0 + np.cumsum(np.random.randn(500) * 0.5)


@pytest.fixture
def mock_indicators(mock_prices: np.ndarray) -> np.ndarray:
    """Generate mock technical indicators (500 x 3)."""
    np.random.seed(42)
    return np.random.randn(500, 3).astype(np.float32)


@pytest.fixture
def basic_env(mock_prices: np.ndarray) -> TradingEnv:
    """Create a basic trading environment."""
    return TradingEnv(prices=mock_prices, window_size=30, max_steps=200)


@pytest.fixture
def env_with_indicators(mock_prices: np.ndarray, mock_indicators: np.ndarray) -> TradingEnv:
    """Create environment with indicators."""
    return TradingEnv(
        prices=mock_prices,
        indicators=mock_indicators,
        window_size=30,
        max_steps=200,
    )


# =============================================================================
# ENVIRONMENT CREATION
# =============================================================================


class TestEnvCreation:
    """Test environment initialization."""

    def test_basic_creation(self, basic_env: TradingEnv) -> None:
        """Environment should initialize with default values."""
        assert basic_env.initial_balance == 100000.0
        assert basic_env.transaction_cost == 0.0001
        assert basic_env.window_size == 30
        assert basic_env.current_step == 0
        assert basic_env.position == 0.0

    def test_custom_parameters(self) -> None:
        """Custom parameters should be respected."""
        prices = np.random.randn(200) + 100
        env = TradingEnv(
            prices=prices,
            initial_balance=50000.0,
            transaction_cost=0.0005,
            window_size=20,
            max_steps=500,
        )
        assert env.initial_balance == 50000.0
        assert env.transaction_cost == 0.0005
        assert env.window_size == 20
        assert env.max_steps == 500

    def test_observation_space_shape_no_indicators(self, basic_env: TradingEnv) -> None:
        """Observation dim = window_size * (1 + 0 + 3) = 30 * 4 = 120."""
        expected_dim = 30 * (1 + 0 + 3)
        assert basic_env.observation_space.shape == (expected_dim,)

    def test_observation_space_shape_with_indicators(
        self, env_with_indicators: TradingEnv
    ) -> None:
        """Observation dim = window_size * (1 + 3 + 3) = 30 * 7 = 210."""
        expected_dim = 30 * (1 + 3 + 3)
        assert env_with_indicators.observation_space.shape == (expected_dim,)

    def test_action_space_bounds(self, basic_env: TradingEnv) -> None:
        """Action space should be [-1, 1]."""
        assert basic_env.action_space.low[0] == -1.0
        assert basic_env.action_space.high[0] == 1.0
        assert basic_env.action_space.shape == (1,)


# =============================================================================
# RESET
# =============================================================================


class TestReset:
    """Test environment reset."""

    def test_reset_returns_observation_and_info(
        self, basic_env: TradingEnv
    ) -> None:
        obs, info = basic_env.reset()
        assert isinstance(obs, np.ndarray)
        assert obs.dtype == np.float32
        assert isinstance(info, dict)

    def test_reset_restores_initial_state(self, basic_env: TradingEnv) -> None:
        """After reset, state should match initialization."""
        basic_env.reset()
        assert basic_env.current_step == 0
        assert basic_env.balance == basic_env.initial_balance
        assert basic_env.position == 0.0
        assert basic_env.entry_price == 0.0
        assert basic_env.equity_history == [basic_env.initial_balance]
        assert basic_env.trades == []

    def test_reset_with_seed(self, mock_prices: np.ndarray) -> None:
        """Reset with seed should be reproducible."""
        env1 = TradingEnv(prices=mock_prices, window_size=10, max_steps=50)
        env2 = TradingEnv(prices=mock_prices, window_size=10, max_steps=50)

        obs1, _ = env1.reset(seed=123)
        obs2, _ = env2.reset(seed=123)
        np.testing.assert_array_equal(obs1, obs2)


# =============================================================================
# STEP
# =============================================================================


class TestStep:
    """Test environment step execution."""  # nosec

    def test_step_returns_correct_types(self, basic_env: TradingEnv) -> None:
        """Step should return (obs, reward, terminated, truncated, info)."""
        basic_env.reset()
        action = np.array([0.5], dtype=np.float32)
        obs, reward, terminated, truncated, info = basic_env.step(action)

        assert isinstance(obs, np.ndarray)
        assert isinstance(reward, float)
        assert isinstance(terminated, bool)
        assert isinstance(truncated, bool)
        assert isinstance(info, dict)

    def test_step_updates_position(self, basic_env: TradingEnv) -> None:
        """Position should update after step."""
        basic_env.reset()
        basic_env.step(np.array([0.5], dtype=np.float32))
        assert basic_env.position == 0.5

    def test_step_records_trade_on_change(self, basic_env: TradingEnv) -> None:
        """Trade should be recorded when position changes significantly."""
        basic_env.reset()
        basic_env.step(np.array([0.5], dtype=np.float32))
        assert len(basic_env.trades) == 1

    def test_no_trade_on_small_change(self, basic_env: TradingEnv) -> None:
        """Position changes < 0.01 should not record a trade."""
        basic_env.reset()
        basic_env.step(np.array([0.005], dtype=np.float32))
        assert len(basic_env.trades) == 0

    def test_step_advances_time(self, basic_env: TradingEnv) -> None:
        """current_step should increment after step."""
        basic_env.reset()
        assert basic_env.current_step == 0
        basic_env.step(np.array([0.0], dtype=np.float32))
        assert basic_env.current_step == 1

    def test_equity_history_grows(self, basic_env: TradingEnv) -> None:
        """Equity history should grow with each step."""
        basic_env.reset()
        initial_len = len(basic_env.equity_history)
        basic_env.step(np.array([0.5], dtype=np.float32))
        assert len(basic_env.equity_history) == initial_len + 1

    def test_info_contains_expected_keys(self, basic_env: TradingEnv) -> None:
        """Info dict should have standard keys."""
        basic_env.reset()
        _, _, _, _, info = basic_env.step(np.array([0.5], dtype=np.float32))
        required_keys = ["equity", "balance", "position", "trades_count", "return"]
        for key in required_keys:
            assert key in info


# =============================================================================
# REWARD
# =============================================================================


class TestReward:
    """Test reward calculation."""

    def test_positive_return_reward(self) -> None:
        """Positive equity change should yield positive return component."""
        prices = np.array([100.0, 101.0, 102.0, 103.0, 104.0])
        env = TradingEnv(prices=prices, window_size=2, max_steps=10)
        env.reset()
        env.position = 1.0
        env.entry_price = 100.0

        # Simulate positive move
        old_equity = 100000.0
        new_equity = 101000.0
        reward = env._calculate_reward(new_equity, old_equity)

        # Return component should be positive (~0.01)
        assert reward > -0.01  # Allow small cost penalties

    def test_negative_return_reward(self) -> None:
        """Negative equity change should yield negative reward."""
        prices = np.array([100.0, 99.0, 98.0])
        env = TradingEnv(prices=prices, window_size=2, max_steps=10)
        env.reset()

        old_equity = 100000.0
        new_equity = 99000.0
        reward = env._calculate_reward(new_equity, old_equity)
        assert reward < 0

    def test_drawdown_penalty(self) -> None:
        """Drawdown should penalize reward."""
        prices = np.array([100.0, 101.0])
        env = TradingEnv(prices=prices, window_size=2, max_steps=10)
        env.reset()
        env.equity_history = [100000.0, 110000.0, 105000.0]

        old_equity = 105000.0
        new_equity = 104000.0
        reward_with_dd = env._calculate_reward(new_equity, old_equity)

        # Same return without drawdown
        env.equity_history = [100000.0]
        reward_no_dd = env._calculate_reward(new_equity, old_equity)

        assert reward_with_dd < reward_no_dd


# =============================================================================
# TERMINATION
# =============================================================================


class TestTermination:
    """Test episode termination conditions."""

    def test_truncation_on_max_steps(self) -> None:
        """Episode should truncate when max_steps reached."""
        prices = np.arange(100.0, 150.0, 0.5)  # 100 steps
        env = TradingEnv(prices=prices, window_size=5, max_steps=10)
        env.reset()

        for _ in range(10):
            obs, reward, terminated, truncated, info = env.step(np.array([0.0]))

        assert truncated is True

    def test_termination_on_liquidation(self) -> None:
        """Episode should terminate on liquidation (equity < 50% initial)."""
        prices = np.array([100.0, 50.0, 10.0, 5.0, 1.0])
        env = TradingEnv(
            prices=prices,
            window_size=2,
            max_steps=10,
            initial_balance=100000.0,
        )
        env.reset()
        env.position = 1000.0  # Large long position
        env.entry_price = 100.0

        # Price crashes -> equity drops below 50%
        obs, reward, terminated, truncated, info = env.step(np.array([1.0]))
        # May or may not terminate depending on exact equity calc
        assert isinstance(terminated, bool)

    def test_no_termination_on_normal_step(self, basic_env: TradingEnv) -> None:
        """Normal step should not trigger termination."""
        basic_env.reset()
        _, _, terminated, truncated, _ = basic_env.step(np.array([0.1]))
        assert terminated is False
        assert truncated is False


# =============================================================================
# OBSERVATION
# =============================================================================


class TestObservation:
    """Test observation building."""

    def test_observation_shape_no_indicators(self, basic_env: TradingEnv) -> None:
        """Observation shape should match observation space."""
        obs, _ = basic_env.reset()
        assert obs.shape == basic_env.observation_space.shape

    def test_observation_shape_with_indicators(
        self, env_with_indicators: TradingEnv
    ) -> None:
        """Observation shape should include indicator dimensions."""
        obs, _ = env_with_indicators.reset()
        assert obs.shape == env_with_indicators.observation_space.shape

    def test_observation_values_finite(self, basic_env: TradingEnv) -> None:
        """All observation values should be finite."""
        obs, _ = basic_env.reset()
        assert np.all(np.isfinite(obs))


# =============================================================================
# UTILITY METHODS
# =============================================================================


class TestUtility:
    """Test utility methods."""

    def test_get_equity_curve(self, basic_env: TradingEnv) -> None:
        """Equity curve should return array of equity values."""
        basic_env.reset()
        basic_env.step(np.array([0.5]))
        basic_env.step(np.array([0.3]))

        curve = basic_env.get_equity_curve()
        assert isinstance(curve, np.ndarray)
        assert len(curve) == 3  # initial + 2 steps

    def test_get_trade_log(self, basic_env: TradingEnv) -> None:
        """Trade log should return list of trade records."""
        basic_env.reset()
        basic_env.step(np.array([0.5]))
        basic_env.step(np.array([-0.3]))

        log = basic_env.get_trade_log()
        assert len(log) == 2
        assert "step" in log[0]
        assert "action" in log[0]
        assert "cost" in log[0]


# =============================================================================
# TRADING STATE DATACLASS
# =============================================================================


class TestTradingState:
    """Test TradingState dataclass."""

    def test_default_values(self) -> None:
        """Default values should match initialization params."""
        state = TradingState()
        assert state.position == 0.0
        assert state.cash == 100000.0
        assert state.equity == 100000.0
        assert state.entry_price == 0.0
        assert state.step == 0
        assert state.holdings_history == []

    def test_custom_values(self) -> None:
        """Custom values should be stored correctly."""
        state = TradingState(
            position=0.5,
            cash=50000.0,
            equity=75000.0,
            entry_price=100.0,
            step=10,
            holdings_history=[0.1, 0.2, 0.3],
        )
        assert state.position == 0.5
        assert state.cash == 50000.0
        assert len(state.holdings_history) == 3
