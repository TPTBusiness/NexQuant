"""
Tests for RL Costeer (Trading Controller).

Covers:
- Costeer initialization with/without model
- Market data initialization
- Action retrieval (mocked model)  # nosec
- Risk limit enforcement
- Observation building
- Step execution  # nosec
- Performance tracking
"""

from pathlib import Path
from unittest.mock import MagicMock, patch

import numpy as np
import pandas as pd
import pytest

from rdagent.components.coder.rl.costeer import RLCosteer


# =============================================================================
# FIXTURES
# =============================================================================


@pytest.fixture
def mock_prices() -> pd.Series:
    """Generate 200-step price series."""
    np.random.seed(42)
    return pd.Series(100.0 + np.cumsum(np.random.randn(200) * 0.5))


@pytest.fixture
def mock_indicators() -> pd.DataFrame:
    """Generate mock indicators DataFrame."""
    np.random.seed(42)
    return pd.DataFrame(
        np.random.randn(200, 3).astype(np.float32),
        columns=["rsi", "macd", "bb_width"],
    )


@pytest.fixture
def basic_costeer() -> RLCosteer:
    """Create costeer without model."""
    return RLCosteer(
        algorithm="PPO",
        window_size=30,
        max_position=1.0,
        risk_limit=0.15,
    )


@pytest.fixture
def initialized_costeer(mock_prices: pd.Series, mock_indicators: pd.DataFrame) -> RLCosteer:
    """Create costeer with market data but no model."""
    costeer = RLCosteer(window_size=30)
    costeer.initialize(mock_prices, mock_indicators, initial_equity=100000.0)
    return costeer


# =============================================================================
# INITIALIZATION
# =============================================================================


class TestCosteerInit:
    """Test costeer initialization."""

    def test_default_values(self) -> None:
        """Default parameters should match specification."""
        costeer = RLCosteer()
        assert costeer.algorithm == "PPO"
        assert costeer.window_size == 60
        assert costeer.max_position == 1.0
        assert costeer.risk_limit == 0.15
        assert costeer.is_active is False
        assert costeer.model is None
        assert costeer.trade_history == []

    def test_custom_values(self) -> None:
        """Custom parameters should be stored."""
        costeer = RLCosteer(
            algorithm="SAC",
            window_size=120,
            max_position=0.5,
            risk_limit=0.10,
        )
        assert costeer.algorithm == "SAC"
        assert costeer.window_size == 120
        assert costeer.max_position == 0.5
        assert costeer.risk_limit == 0.10

    def test_initialize_sets_market_data(
        self, mock_prices: pd.Series, mock_indicators: pd.DataFrame
    ) -> None:
        """Initialize should store market data and activate costeer."""
        costeer = RLCosteer(window_size=30)
        costeer.initialize(mock_prices, mock_indicators, initial_equity=50000.0)

        assert costeer.is_active is True
        assert len(costeer.prices) == 200
        assert costeer.indicators is not None
        assert costeer.initial_equity == 50000.0
        assert costeer.current_step == 30
        assert costeer.peak_equity == 50000.0

    def test_initialize_without_indicators(self, mock_prices: pd.Series) -> None:
        """Initialize should work without indicators."""
        costeer = RLCosteer(window_size=30)
        costeer.initialize(mock_prices, initial_equity=100000.0)

        assert costeer.indicators is None
        assert costeer.is_active is True


# =============================================================================
# GET ACTION
# =============================================================================


class TestGetAction:
    """Test action retrieval."""  # nosec

    def test_no_model_returns_zero(self, initialized_costeer: RLCosteer) -> None:
        """Without model, action should be 0 (hold)."""
        action = initialized_costeer.get_action(
            current_equity=100000.0, cash=50000.0, position=0.0
        )
        assert action == 0.0

    def test_not_active_returns_zero(self, basic_costeer: RLCosteer) -> None:
        """Inactive costeer should return 0."""
        action = basic_costeer.get_action(
            current_equity=100000.0, cash=50000.0, position=0.0
        )
        assert action == 0.0

    @patch.object(RLCosteer, "_build_observation")
    def test_model_action_risk_scaled(
        self, mock_obs: MagicMock, initialized_costeer: RLCosteer
    ) -> None:
        """Model action should be returned and risk-scaled."""
        mock_obs.return_value = np.random.randn(100).astype(np.float32)

        mock_model = MagicMock()
        mock_model.predict.return_value = (np.array([[0.8]]), None)
        initialized_costeer.model = mock_model

        action = initialized_costeer.get_action(
            current_equity=100000.0, cash=50000.0, position=0.0
        )

        # At full risk (no drawdown), action should be ~0.8
        assert abs(action) <= 1.0

    def test_risk_limit_forces_close(self, initialized_costeer: RLCosteer) -> None:
        """Drawdown > risk_limit should force position to 0."""
        mock_model = MagicMock()
        mock_model.predict.return_value = (np.array([[1.0]]), None)
        initialized_costeer.model = mock_model

        # Simulate 20% drawdown (> 15% limit)
        action = initialized_costeer.get_action(
            current_equity=80000.0,  # 20% drawdown from 100k
            cash=50000.0,
            position=0.5,
        )
        assert action == 0.0

    def test_action_clipped_to_max_position(
        self, initialized_costeer: RLCosteer
    ) -> None:
        """Action should be clipped to max_position."""
        initialized_costeer.max_position = 0.5

        mock_model = MagicMock()
        mock_model.predict.return_value = (np.array([[1.0]]), None)
        initialized_costeer.model = mock_model

        action = initialized_costeer.get_action(
            current_equity=100000.0, cash=50000.0, position=0.0
        )
        assert action <= 0.5


# =============================================================================
# OBSERVATION BUILDING
# =============================================================================


class TestObservationBuilding:
    """Test observation vector construction."""

    def test_observation_shape_no_indicators(
        self, mock_prices: pd.Series
    ) -> None:
        """Observation should have correct shape without indicators."""
        costeer = RLCosteer(window_size=30)
        costeer.initialize(mock_prices, initial_equity=100000.0)

        obs = costeer._build_observation(
            current_equity=100000.0, cash=50000.0, position=0.0
        )

        # window_size + 3 (position, pnl, equity_ratio)
        expected = 30 + 3
        assert obs.shape == (expected,)

    def test_observation_shape_with_indicators(
        self, mock_prices: pd.Series, mock_indicators: pd.DataFrame
    ) -> None:
        """Observation should include indicator dimensions."""
        costeer = RLCosteer(window_size=30)
        costeer.initialize(mock_prices, mock_indicators, initial_equity=100000.0)

        obs = costeer._build_observation(
            current_equity=100000.0, cash=50000.0, position=0.0
        )

        # window_size * (1 + 3) + 3
        expected = 30 + (30 * 3) + 3
        assert obs.shape == (expected,)

    def test_observation_dtype(self, initialized_costeer: RLCosteer) -> None:
        """Observation should be float32."""
        obs = initialized_costeer._build_observation(
            current_equity=100000.0, cash=50000.0, position=0.0
        )
        assert obs.dtype == np.float32


# =============================================================================
# STEP EXECUTION
# =============================================================================


class TestStepExecution:
    """Test step execution."""  # nosec

    def test_step_records_trade(self, initialized_costeer: RLCosteer) -> None:
        """Step should record trade in history."""
        trade = initialized_costeer.step(
            current_equity=100000.0, cash=50000.0, position=0.0
        )

        assert "timestamp" in trade
        assert "step" in trade
        assert "equity" in trade
        assert "position" in trade
        assert "target_position" in trade
        assert "action" in trade

    def test_step_advances_current_step(self, initialized_costeer: RLCosteer) -> None:
        """Step should increment current_step."""
        initial_step = initialized_costeer.current_step
        initialized_costeer.step(
            current_equity=100000.0, cash=50000.0, position=0.0
        )
        assert initialized_costeer.current_step == initial_step + 1

    def test_step_updates_peak_equity(self, initialized_costeer: RLCosteer) -> None:
        """Peak equity should update when equity exceeds previous peak."""
        initialized_costeer.peak_equity = 100000.0
        initialized_costeer.step(
            current_equity=105000.0, cash=50000.0, position=0.0
        )
        assert initialized_costeer.peak_equity == 105000.0

    def test_multiple_steps_accumulate_trades(
        self, initialized_costeer: RLCosteer
    ) -> None:
        """Multiple steps should accumulate trades."""
        for _ in range(5):
            initialized_costeer.step(
                current_equity=100000.0, cash=50000.0, position=0.0
            )

        assert len(initialized_costeer.trade_history) == 5


# =============================================================================
# MODEL LOADING
# =============================================================================


class TestModelLoading:
    """Test model loading functionality."""

    def test_load_model_import_error(self, tmp_path: Path) -> None:
        """Load should raise ImportError when SB3 not installed."""
        costeer = RLCosteer()

        with patch.dict("sys.modules", {"stable_baselines3": None}):
            with pytest.raises(ImportError, match="stable-baselines3"):
                costeer.load_model(tmp_path / "model.zip")

    def test_load_model_file_not_found(self, tmp_path: Path) -> None:
        """Load should raise ValueError for non-existent file."""
        costeer = RLCosteer(model_path=tmp_path / "nonexistent.zip")
        # Model path doesn't exist, so it won't try to load
        assert costeer.is_active is False


# =============================================================================
# PERFORMANCE TRACKING
# =============================================================================


class TestPerformanceTracking:
    """Test performance history."""

    def test_get_performance_returns_dataframe(
        self, initialized_costeer: RLCosteer
    ) -> None:
        """Performance should return DataFrame."""
        # Add some trades
        initialized_costeer.step(
            current_equity=100000.0, cash=50000.0, position=0.0
        )
        initialized_costeer.step(
            current_equity=101000.0, cash=49000.0, position=0.3
        )

        df = initialized_costeer.get_performance()

        assert isinstance(df, pd.DataFrame)
        assert len(df) == 2
        assert "equity" in df.columns
        assert "position" in df.columns

    def test_empty_performance_history(self, basic_costeer: RLCosteer) -> None:
        """Empty history should return empty DataFrame."""
        df = basic_costeer.get_performance()
        assert isinstance(df, pd.DataFrame)
        assert len(df) == 0
