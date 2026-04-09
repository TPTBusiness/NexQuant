"""
Tests for Optuna Parameter Optimizer.

Public test file - references closed-source module at rdagent/scenarios/qlib/local/optuna_optimizer.py

Tests cover:
- Parameter space definition and validation
- Parameter suggestion mechanisms
- Objective function calculation
- FTMO penalty logic
- Optuna study creation and configuration
- Parameter injection into strategy code
- Optimization run (mocked, small trial count)
- Result saving and loading
- Best parameter extraction
- Top trial retrieval
- Edge cases and error handling
- Strategy metadata updates
"""

import json
import os
import tempfile
from pathlib import Path
from unittest.mock import Mock, patch, MagicMock

import pytest
import numpy as np
import pandas as pd

try:
    import optuna
    OPTUNA_AVAILABLE = True
except ImportError:
    OPTUNA_AVAILABLE = False

from rdagent.scenarios.qlib.local.optuna_optimizer import (
    OptunaOptimizer,
    PARAMETER_SPACE,
    FTMO_MAX_STOP_LOSS,
    FTMO_MAX_DRAWDOWN,
    FTMO_MAX_DAILY_LOSS,
    PENALTY_MAX_DD,
    PENALTY_FTMO_VIOLATION,
    OPTUNA_AVAILABLE,
)


# =============================================================================
# Fixtures
# =============================================================================

@pytest.fixture
def sample_factors():
    """Sample factor values DataFrame."""
    dates = pd.date_range('2024-01-01', periods=1000, freq='1min')
    np.random.seed(42)
    return pd.DataFrame({
        'momentum_1d': np.random.randn(1000),
        'mean_reversion': np.random.randn(1000),
        'volatility': np.random.randn(1000),
    }, index=dates)


@pytest.fixture
def sample_close():
    """Sample OHLCV close price series."""
    dates = pd.date_range('2024-01-01', periods=1000, freq='1min')
    np.random.seed(42)
    prices = 1.0850 + np.cumsum(np.random.randn(1000) * 0.0001)
    return pd.Series(prices, index=dates, name='$close')


@pytest.fixture
def sample_strategy_code():
    """Sample strategy code for testing."""
    return '''
import pandas as pd
import numpy as np

# Strategy parameters
entry_threshold = 0.3
exit_threshold = 0.1
stop_loss = 0.02
take_profit = 0.04
trailing_stop = 0.015
short_window = 10

def generate_signal(factors, close):
    """Generate trading signals based on factors."""
    momentum = factors['momentum_1d']
    signal = pd.Series(0, index=close.index)
    signal[momentum > entry_threshold] = 1
    signal[momentum < -entry_threshold] = -1
    return signal

# Generate signal
signal = generate_signal(factors, close)
'''


@pytest.fixture
def sample_strategy_json(sample_strategy_code, tmp_path):
    """Sample strategy JSON file."""
    strategy = {
        'name': 'TestStrategy',
        'code': sample_strategy_code,
        'factor_names': ['momentum_1d', 'mean_reversion', 'volatility'],
        'parameters': {
            'entry_threshold': 0.3,
            'exit_threshold': 0.1,
        },
    }
    path = tmp_path / 'test_strategy.json'
    with open(path, 'w') as f:
        json.dump(strategy, f, indent=2)
    return str(path)


@pytest.fixture
def mock_backtest_engine():
    """Mock BacktestEngine from P1."""
    engine = Mock()

    def mock_run(**kwargs):
        # Simulate realistic backtest result
        code = kwargs.get('strategy_code', '')
        # Extract params from code for varied results
        np.random.seed(hash(code) % 2**32)
        sharpe = np.random.uniform(0.5, 2.5)
        ic = np.random.uniform(0.02, 0.15)
        n_trades = np.random.randint(10, 100)
        max_dd = np.random.uniform(-0.15, -0.03)

        return {
            'success': True,
            'sharpe_ratio': sharpe,
            'ic': ic,
            'max_drawdown': max_dd,
            'total_trades': n_trades,
            'wins': int(n_trades * 0.55),
            'losses': int(n_trades * 0.45),
            'win_rate': 0.55,
            'total_return': sharpe * 0.05,
            'final_equity': 1.0 + sharpe * 0.05,
        }

    engine.run_backtest.side_effect = mock_run
    return engine


@pytest.fixture
def optimizer():
    """Basic optimizer instance."""
    return OptunaOptimizer(seed=42)


@pytest.fixture
def optimizer_with_data(sample_strategy_code, sample_factors, sample_close):
    """Optimizer with strategy code and factor data."""
    opt = OptunaOptimizer(seed=42)
    opt.set_strategy_code(sample_strategy_code)
    opt.set_factor_data(sample_factors, sample_close)
    return opt


# =============================================================================
# Module Constants Tests
# =============================================================================

class TestParameterSpaceDefinition:
    """Test parameter space definition."""

    def test_parameter_space_is_dict(self):
        """Test that PARAMETER_SPACE is a dictionary."""
        assert isinstance(PARAMETER_SPACE, dict)

    def test_parameter_space_has_required_keys(self):
        """Test that all required parameters are defined."""
        required = [
            'entry_threshold', 'exit_threshold', 'short_window',
            'stop_loss', 'take_profit', 'trailing_stop',
        ]
        for key in required:
            assert key in PARAMETER_SPACE, f"Missing parameter: {key}"

    def test_parameter_space_entry_threshold_config(self):
        """Test entry_threshold parameter configuration."""
        config = PARAMETER_SPACE['entry_threshold']
        assert config['type'] == 'uniform'
        assert config['low'] == 0.1
        assert config['high'] == 0.5

    def test_parameter_space_exit_threshold_config(self):
        """Test exit_threshold parameter configuration."""
        config = PARAMETER_SPACE['exit_threshold']
        assert config['type'] == 'uniform'
        assert config['low'] == 0.0
        assert config['high'] == 0.3

    def test_parameter_space_short_window_config(self):
        """Test short_window parameter configuration."""
        config = PARAMETER_SPACE['short_window']
        assert config['type'] == 'categorical'
        assert config['choices'] == [5, 10, 15, 20]

    def test_parameter_space_stop_loss_config(self):
        """Test stop_loss parameter configuration (FTMO compliant)."""
        config = PARAMETER_SPACE['stop_loss']
        assert config['type'] == 'categorical'
        assert all(c <= FTMO_MAX_STOP_LOSS for c in config['choices'])

    def test_parameter_space_take_profit_config(self):
        """Test take_profit parameter configuration."""
        config = PARAMETER_SPACE['take_profit']
        assert config['type'] == 'categorical'
        assert config['choices'] == [0.02, 0.03, 0.04]

    def test_parameter_space_trailing_stop_config(self):
        """Test trailing_stop parameter configuration."""
        config = PARAMETER_SPACE['trailing_stop']
        assert config['type'] == 'categorical'
        assert config['choices'] == [0.01, 0.015]

    def test_ftmo_constants_correct(self):
        """Test FTMO compliance constants."""
        assert FTMO_MAX_STOP_LOSS == 0.02
        assert FTMO_MAX_DRAWDOWN == -0.10
        assert FTMO_MAX_DAILY_LOSS == 0.05

    def test_penalty_constants_correct(self):
        """Test penalty weight constants."""
        assert PENALTY_MAX_DD == -10.0
        assert PENALTY_FTMO_VIOLATION == -50.0


# =============================================================================
# Optimizer Initialization Tests
# =============================================================================

class TestOptimizerInitialization:
    """Test optimizer initialization."""

    @pytest.mark.skipif(not OPTUNA_AVAILABLE, reason="Optuna not installed")
    def test_init_defaults(self):
        """Test initialization with default parameters."""
        opt = OptunaOptimizer()
        assert opt.seed == 42
        assert opt.parameter_space == PARAMETER_SPACE
        assert opt.backtest_engine is None
        assert opt._study is None
        assert opt._best_params is None
        assert opt._optimization_history == []

    @pytest.mark.skipif(not OPTUNA_AVAILABLE, reason="Optuna not installed")
    def test_init_custom_parameters(self):
        """Test initialization with custom parameters."""
        custom_space = {'param1': {'type': 'uniform', 'low': 0, 'high': 1}}
        opt = OptunaOptimizer(
            parameter_space=custom_space,
            seed=123,
            study_name='custom_test',
        )
        assert opt.parameter_space == custom_space
        assert opt.seed == 123
        assert opt.study_name == 'custom_test'

    @pytest.mark.skipif(not OPTUNA_AVAILABLE, reason="Optuna not installed")
    def test_init_with_backtest_engine(self, mock_backtest_engine):
        """Test initialization with external backtest engine."""
        opt = OptunaOptimizer(backtest_engine=mock_backtest_engine)
        assert opt.backtest_engine is mock_backtest_engine

    @pytest.mark.skipif(not OPTUNA_AVAILABLE, reason="Optuna not installed")
    def test_parameter_names_property(self):
        """Test parameter_names property returns correct list."""
        opt = OptunaOptimizer()
        names = opt.parameter_names
        assert isinstance(names, list)
        assert 'entry_threshold' in names
        assert 'stop_loss' in names
        assert len(names) == len(PARAMETER_SPACE)


# =============================================================================
# Parameter Suggestion Tests
# =============================================================================

@pytest.mark.skipif(not OPTUNA_AVAILABLE, reason="Optuna not installed")
class TestParameterSuggestion:
    """Test parameter suggestion mechanism."""

    def test_suggest_params_returns_all_params(self, optimizer):
        """Test that suggest_params returns all defined parameters."""
        study = optuna.create_study(sampler=optuna.samplers.TPESampler(seed=42))

        def dummy_objective(trial):
            params = optimizer.suggest_params(trial)
            assert len(params) == len(optimizer.parameter_space)
            return 0.0

        study.optimize(dummy_objective, n_trials=1)

    def test_suggest_params_uniform_range(self, optimizer):
        """Test that uniform parameters are within defined range."""
        study = optuna.create_study(sampler=optuna.samplers.TPESampler(seed=42))
        captured = {}

        def dummy_objective(trial):
            params = optimizer.suggest_params(trial)
            captured.update(params)
            return 0.0

        study.optimize(dummy_objective, n_trials=5)

        # Check entry_threshold is within [0.1, 0.5]
        # Note: We check the trial params, not the captured ones directly
        for trial in study.trials:
            if 'entry_threshold' in trial.params:
                val = trial.params['entry_threshold']
                assert 0.1 <= val <= 0.5

    def test_suggest_params_categorical_choices(self, optimizer):
        """Test that categorical parameters use defined choices."""
        study = optuna.create_study(sampler=optuna.samplers.TPESampler(seed=42))

        def dummy_objective(trial):
            params = optimizer.suggest_params(trial)
            sl = params.get('stop_loss')
            if sl is not None:
                assert sl in [0.01, 0.015, 0.02]
            return 0.0

        study.optimize(dummy_objective, n_trials=5)


# =============================================================================
# Objective Function Tests
# =============================================================================

@pytest.mark.skipif(not OPTUNA_AVAILABLE, reason="Optuna not installed")
class TestObjectiveFunction:
    """Test objective function calculation."""

    def test_objective_with_mock_backtest(self, mock_backtest_engine):
        """Test objective function with mocked backtest engine."""
        opt = OptunaOptimizer(backtest_engine=mock_backtest_engine, seed=42)
        opt._strategy_code = "test_code"
        opt._factors = Mock()
        opt._close = Mock()

        study = optuna.create_study(sampler=optuna.samplers.TPESampler(seed=42))

        def test_objective(trial):
            return opt.objective(trial)

        study.optimize(test_objective, n_trials=1)

        # Check that trial was recorded in history
        assert len(opt._optimization_history) == 1
        trial_record = opt._optimization_history[0]
        assert 'objective' in trial_record
        assert 'sharpe_ratio' in trial_record
        assert 'ic' in trial_record

    def test_objective_formula(self, optimizer):
        """Test objective formula: sharpe * |IC| * sqrt(n_trades)."""
        # Create a trial with known params
        study = optuna.create_study(sampler=optuna.samplers.TPESampler(seed=42))

        # Mock the backtest result
        optimizer._strategy_code = "test"

        with patch.object(optimizer, '_run_backtest_with_params') as mock_bt:
            mock_bt.return_value = {
                'success': True,
                'sharpe_ratio': 1.5,
                'ic': 0.08,
                'total_trades': 25,
                'max_drawdown': -0.05,
                'total_return': 0.075,
                'win_rate': 0.56,
            }

            trial = study.ask()
            params = optimizer.suggest_params(trial)
            value = optimizer.objective(trial)

            # Expected: 1.5 * 0.08 * sqrt(25) = 1.5 * 0.08 * 5 = 0.6
            expected = 1.5 * 0.08 * np.sqrt(25)
            assert abs(value - expected) < 1e-6

    def test_objective_failed_backtest_returns_neg_inf(self, optimizer):
        """Test that failed backtests return -inf."""
        study = optuna.create_study(sampler=optuna.samplers.TPESampler(seed=42))

        with patch.object(optimizer, '_run_backtest_with_params') as mock_bt:
            mock_bt.return_value = {'success': False, 'error': 'Test failure'}

            trial = study.ask()
            value = optimizer.objective(trial)
            assert value == float('-inf')

    def test_objective_zero_trades_returns_neg_inf(self, optimizer):
        """Test that zero trades return -inf."""
        study = optuna.create_study(sampler=optuna.samplers.TPESampler(seed=42))

        with patch.object(optimizer, '_run_backtest_with_params') as mock_bt:
            mock_bt.return_value = {
                'success': True,
                'sharpe_ratio': 1.0,
                'ic': 0.05,
                'total_trades': 0,
                'max_drawdown': -0.03,
            }

            trial = study.ask()
            value = optimizer.objective(trial)
            assert value == float('-inf')


# =============================================================================
# FTMO Penalty Tests
# =============================================================================

@pytest.mark.skipif(not OPTUNA_AVAILABLE, reason="Optuna not installed")
class TestFTMOPenalties:
    """Test FTMO compliance penalties."""

    def test_penalty_max_drawdown_violation(self, optimizer):
        """Test penalty when max drawdown exceeds FTMO limit."""
        study = optuna.create_study(sampler=optuna.samplers.TPESampler(seed=42))

        with patch.object(optimizer, '_run_backtest_with_params') as mock_bt:
            mock_bt.return_value = {
                'success': True,
                'sharpe_ratio': 1.5,
                'ic': 0.08,
                'total_trades': 25,
                'max_drawdown': -0.12,  # Below FTMO_MAX_DRAWDOWN (-0.10)
            }

            trial = study.ask()
            optimizer.suggest_params(trial)
            value = optimizer.objective(trial)

            # Should have penalty applied
            history = optimizer._optimization_history[-1]
            assert history['penalty'] <= PENALTY_MAX_DD

    def test_penalty_stop_loss_violation(self, optimizer):
        """Test penalty when stop loss exceeds FTMO maximum."""
        study = optuna.create_study(sampler=optuna.samplers.TPESampler(seed=42))

        # Create a custom parameter space that allows FTMO-violating values
        violating_space = {
            **PARAMETER_SPACE,
            'stop_loss': {'type': 'categorical', 'choices': [0.01, 0.025, 0.03]},
        }
        optimizer.param_space_original = optimizer.parameter_space
        optimizer.parameter_space = violating_space

        with patch.object(optimizer, '_run_backtest_with_params') as mock_bt:
            mock_bt.return_value = {
                'success': True,
                'sharpe_ratio': 1.5,
                'ic': 0.08,
                'total_trades': 25,
                'max_drawdown': -0.05,
            }

            trial = study.ask()
            # Force stop_loss to violating value
            with patch.object(optimizer, 'suggest_params', return_value={'stop_loss': 0.025}):
                value = optimizer.objective(trial)

            history = optimizer._optimization_history[-1]
            assert history['penalty'] <= PENALTY_FTMO_VIOLATION

        # Restore original space
        optimizer.parameter_space = optimizer.param_space_original

    def test_no_penalty_compliant_strategy(self, optimizer):
        """Test no penalty for FTMO-compliant strategy."""
        study = optuna.create_study(sampler=optuna.samplers.TPESampler(seed=42))

        with patch.object(optimizer, '_run_backtest_with_params') as mock_bt:
            mock_bt.return_value = {
                'success': True,
                'sharpe_ratio': 1.5,
                'ic': 0.08,
                'total_trades': 25,
                'max_drawdown': -0.05,  # Within FTMO limit
            }

            trial = study.ask()
            with patch.object(optimizer, 'suggest_params', return_value={'stop_loss': 0.01}):
                value = optimizer.objective(trial)

            history = optimizer._optimization_history[-1]
            assert history['penalty'] == 0.0
            assert history['objective'] == history['base_objective']

    def test_combined_penalties(self, optimizer):
        """Test that both penalties can be applied simultaneously."""
        study = optuna.create_study(sampler=optuna.samplers.TPESampler(seed=42))

        violating_space = {
            **PARAMETER_SPACE,
            'stop_loss': {'type': 'categorical', 'choices': [0.01, 0.025, 0.03]},
        }
        optimizer.parameter_space = violating_space

        with patch.object(optimizer, '_run_backtest_with_params') as mock_bt:
            mock_bt.return_value = {
                'success': True,
                'sharpe_ratio': 1.5,
                'ic': 0.08,
                'total_trades': 25,
                'max_drawdown': -0.12,  # FTMO violation
            }

            trial = study.ask()
            with patch.object(optimizer, 'suggest_params', return_value={'stop_loss': 0.025}):
                value = optimizer.objective(trial)

            history = optimizer._optimization_history[-1]
            # Both penalties should apply
            expected_penalty = PENALTY_MAX_DD + PENALTY_FTMO_VIOLATION
            assert history['penalty'] == expected_penalty


# =============================================================================
# Parameter Injection Tests
# =============================================================================

class TestParameterInjection:
    """Test parameter injection into strategy code."""

    def test_inject_params_basic_replacement(self):
        """Test basic parameter replacement in code."""
        code = '''
entry_threshold = 0.3
exit_threshold = 0.1
'''
        params = {'entry_threshold': 0.4, 'exit_threshold': 0.15}
        result = OptunaOptimizer.inject_params(code, params)

        assert 'entry_threshold = 0.4' in result
        assert 'exit_threshold = 0.15' in result

    def test_inject_params_with_marker(self):
        """Test injection at PARAMS_INJECT marker."""
        code = '''
# PARAMS_INJECT

def generate_signal(factors, close):
    pass
'''
        params = {'entry_threshold': 0.35, 'stop_loss': 0.015}
        result = OptunaOptimizer.inject_params(code, params)

        assert '# PARAMS_INJECT' in result
        assert 'entry_threshold = 0.35' in result
        assert 'stop_loss = 0.015' in result

    def test_inject_params_preserves_indentation(self):
        """Test that indentation is preserved during injection."""
        code = '''
class Strategy:
    entry_threshold = 0.3
'''
        params = {'entry_threshold': 0.45}
        result = OptunaOptimizer.inject_params(code, params)

        assert '    entry_threshold = 0.45' in result

    def test_inject_params_no_match_returns_original(self):
        """Test that non-matching params return unchanged code."""
        code = 'my_param = 0.5\n'
        params = {'nonexistent_param': 0.1}
        result = OptunaOptimizer.inject_params(code, params)

        assert result == code

    def test_inject_params_float_values(self):
        """Test injection of float parameter values."""
        code = 'stop_loss = 0.02\n'
        params = {'stop_loss': 0.015}
        result = OptunaOptimizer.inject_params(code, params)

        assert 'stop_loss = 0.015' in result

    def test_inject_params_int_values(self):
        """Test injection of integer parameter values."""
        code = 'short_window = 10\n'
        params = {'short_window': 20}
        result = OptunaOptimizer.inject_params(code, params)

        assert 'short_window = 20' in result

    def test_inject_params_full_strategy(self, sample_strategy_code):
        """Test injection into a full strategy code."""
        params = {
            'entry_threshold': 0.4,
            'exit_threshold': 0.15,
            'stop_loss': 0.015,
            'take_profit': 0.03,
            'trailing_stop': 0.01,
            'short_window': 15,
        }
        result = OptunaOptimizer.inject_params(sample_strategy_code, params)

        assert 'entry_threshold = 0.4' in result
        assert 'exit_threshold = 0.15' in result
        assert 'stop_loss = 0.015' in result
        assert 'take_profit = 0.03' in result
        assert 'trailing_stop = 0.01' in result
        assert 'short_window = 15' in result


# =============================================================================
# Optuna Study Creation Tests
# =============================================================================

@pytest.mark.skipif(not OPTUNA_AVAILABLE, reason="Optuna not installed")
class TestStudyCreation:
    """Test Optuna study creation and configuration."""

    def test_study_uses_tpe_sampler(self, optimizer_with_data):
        """Test that study uses TPESampler."""
        # Run a tiny optimization to create the study
        with tempfile.TemporaryDirectory() as tmpdir:
            strategy_path = Path(tmpdir) / 'strategy.json'
            with open(strategy_path, 'w') as f:
                json.dump({'name': 'Test', 'code': optimizer_with_data._strategy_code}, f)

            study = optimizer_with_data.optimize(
                strategy_path=str(strategy_path),
                factors=optimizer_with_data._factors,
                close=optimizer_with_data._close,
                n_trials=2,
                show_progress=False,
            )

        assert isinstance(study.sampler, optuna.samplers.TPESampler)

    def test_study_uses_median_pruner(self, optimizer_with_data):
        """Test that study uses MedianPruner."""
        with tempfile.TemporaryDirectory() as tmpdir:
            strategy_path = Path(tmpdir) / 'strategy.json'
            with open(strategy_path, 'w') as f:
                json.dump({'name': 'Test', 'code': optimizer_with_data._strategy_code}, f)

            study = optimizer_with_data.optimize(
                strategy_path=str(strategy_path),
                factors=optimizer_with_data._factors,
                close=optimizer_with_data._close,
                n_trials=2,
                show_progress=False,
            )

        assert isinstance(study.pruner, optuna.pruners.MedianPruner)

    def test_study_direction_is_maximize(self, optimizer_with_data):
        """Test that study direction is maximize."""
        with tempfile.TemporaryDirectory() as tmpdir:
            strategy_path = Path(tmpdir) / 'strategy.json'
            with open(strategy_path, 'w') as f:
                json.dump({'name': 'Test', 'code': optimizer_with_data._strategy_code}, f)

            study = optimizer_with_data.optimize(
                strategy_path=str(strategy_path),
                factors=optimizer_with_data._factors,
                close=optimizer_with_data._close,
                n_trials=2,
                show_progress=False,
            )

        assert study.directions[0] == optuna.study.StudyDirection.MAXIMIZE


# =============================================================================
# Optimization Run Tests
# =============================================================================

@pytest.mark.skipif(not OPTUNA_AVAILABLE, reason="Optuna not installed")
class TestOptimizationRun:
    """Test optimization run with small trial count."""

    def test_optimize_runs_specified_trials(self, optimizer_with_data):
        """Test that optimization runs the specified number of trials."""
        with tempfile.TemporaryDirectory() as tmpdir:
            strategy_path = Path(tmpdir) / 'strategy.json'
            with open(strategy_path, 'w') as f:
                json.dump({'name': 'Test', 'code': optimizer_with_data._strategy_code}, f)

            study = optimizer_with_data.optimize(
                strategy_path=str(strategy_path),
                factors=optimizer_with_data._factors,
                close=optimizer_with_data._close,
                n_trials=10,
                show_progress=False,
            )

            assert len(study.trials) == 10

    def test_optimize_records_history(self, optimizer_with_data):
        """Test that optimization records trial history."""
        with tempfile.TemporaryDirectory() as tmpdir:
            strategy_path = Path(tmpdir) / 'strategy.json'
            with open(strategy_path, 'w') as f:
                json.dump({'name': 'Test', 'code': optimizer_with_data._strategy_code}, f)

            optimizer_with_data.optimize(
                strategy_path=str(strategy_path),
                factors=optimizer_with_data._factors,
                close=optimizer_with_data._close,
                n_trials=5,
                show_progress=False,
            )

            assert len(optimizer_with_data._optimization_history) == 5
            for trial in optimizer_with_data._optimization_history:
                assert 'trial_number' in trial
                assert 'params' in trial
                assert 'objective' in trial

    def test_optimize_sets_best_params(self, optimizer_with_data):
        """Test that best parameters are extracted after optimization."""
        with tempfile.TemporaryDirectory() as tmpdir:
            strategy_path = Path(tmpdir) / 'strategy.json'
            with open(strategy_path, 'w') as f:
                json.dump({'name': 'Test', 'code': optimizer_with_data._strategy_code}, f)

            optimizer_with_data.optimize(
                strategy_path=str(strategy_path),
                factors=optimizer_with_data._factors,
                close=optimizer_with_data._close,
                n_trials=5,
                show_progress=False,
            )

            best = optimizer_with_data.get_best_params()
            assert best is not None
            assert isinstance(best, dict)

    def test_optimize_with_missing_file_raises_error(self, optimizer):
        """Test that missing strategy file raises FileNotFoundError."""
        with pytest.raises(FileNotFoundError):
            optimizer.optimize(
                strategy_path='/nonexistent/strategy.json',
                n_trials=1,
            )

    def test_optimize_with_empty_code_raises_error(self, tmp_path):
        """Test that empty strategy code raises ValueError."""
        strategy_path = tmp_path / 'empty.json'
        with open(strategy_path, 'w') as f:
            json.dump({'name': 'Empty', 'code': ''}, f)

        optimizer = OptunaOptimizer(seed=42)
        with pytest.raises(ValueError, match="no code"):
            optimizer.optimize(
                strategy_path=str(strategy_path),
                n_trials=1,
            )


# =============================================================================
# Result Saving Tests
# =============================================================================

@pytest.mark.skipif(not OPTUNA_AVAILABLE, reason="Optuna not installed")
class TestResultSaving:
    """Test result persistence to JSON."""

    def test_save_results_creates_file(self, optimizer_with_data, tmp_path):
        """Test that save_results creates the output file."""
        # Run a tiny optimization first
        strategy_path = tmp_path / 'strategy.json'
        with open(strategy_path, 'w') as f:
            json.dump({'name': 'Test', 'code': optimizer_with_data._strategy_code}, f)

        optimizer_with_data.optimize(
            strategy_path=str(strategy_path),
            factors=optimizer_with_data._factors,
            close=optimizer_with_data._close,
            n_trials=3,
            show_progress=False,
        )

        output_path = tmp_path / 'results.json'
        result = optimizer_with_data.save_results(str(output_path))

        assert Path(result).exists()
        assert result == str(output_path)

    def test_save_results_valid_json(self, optimizer_with_data, tmp_path):
        """Test that saved results are valid JSON."""
        strategy_path = tmp_path / 'strategy.json'
        with open(strategy_path, 'w') as f:
            json.dump({'name': 'Test', 'code': optimizer_with_data._strategy_code}, f)

        optimizer_with_data.optimize(
            strategy_path=str(strategy_path),
            factors=optimizer_with_data._factors,
            close=optimizer_with_data._close,
            n_trials=3,
            show_progress=False,
        )

        output_path = tmp_path / 'results.json'
        optimizer_with_data.save_results(str(output_path))

        with open(output_path, 'r') as f:
            data = json.load(f)

        assert 'best_params' in data
        assert 'best_objective_value' in data
        assert 'optimization_history' in data
        assert 'top_trials' in data

    def test_save_results_contains_strategy_name(self, optimizer_with_data, tmp_path):
        """Test that saved results contain strategy name."""
        strategy_path = tmp_path / 'strategy.json'
        with open(strategy_path, 'w') as f:
            json.dump({'name': 'MyTestStrategy', 'code': optimizer_with_data._strategy_code}, f)

        optimizer_with_data.optimize(
            strategy_path=str(strategy_path),
            factors=optimizer_with_data._factors,
            close=optimizer_with_data._close,
            n_trials=2,
            show_progress=False,
        )

        output_path = tmp_path / 'results.json'
        optimizer_with_data.save_results(str(output_path))

        with open(output_path, 'r') as f:
            data = json.load(f)

        assert data['strategy_name'] == 'MyTestStrategy'

    def test_save_results_top_trials_count(self, optimizer_with_data, tmp_path):
        """Test that top_trials contains at most 5 entries."""
        strategy_path = tmp_path / 'strategy.json'
        with open(strategy_path, 'w') as f:
            json.dump({'name': 'Test', 'code': optimizer_with_data._strategy_code}, f)

        optimizer_with_data.optimize(
            strategy_path=str(strategy_path),
            factors=optimizer_with_data._factors,
            close=optimizer_with_data._close,
            n_trials=10,
            show_progress=False,
        )

        output_path = tmp_path / 'results.json'
        optimizer_with_data.save_results(str(output_path))

        with open(output_path, 'r') as f:
            data = json.load(f)

        assert len(data['top_trials']) <= 5

    def test_save_results_creates_parent_dirs(self, optimizer_with_data, tmp_path):
        """Test that save_results creates parent directories."""
        strategy_path = tmp_path / 'strategy.json'
        with open(strategy_path, 'w') as f:
            json.dump({'name': 'Test', 'code': optimizer_with_data._strategy_code}, f)

        optimizer_with_data.optimize(
            strategy_path=str(strategy_path),
            factors=optimizer_with_data._factors,
            close=optimizer_with_data._close,
            n_trials=2,
            show_progress=False,
        )

        output_path = tmp_path / 'nested' / 'dir' / 'results.json'
        result = optimizer_with_data.save_results(str(output_path))

        assert Path(result).exists()


# =============================================================================
# Strategy Metadata Update Tests
# =============================================================================

@pytest.mark.skipif(not OPTUNA_AVAILABLE, reason="Optuna not installed")
class TestStrategyMetadataUpdate:
    """Test strategy metadata updates after optimization."""

    def test_update_strategy_metadata_adds_optimization(self, optimizer_with_data, tmp_path):
        """Test that update adds optimization section to strategy."""
        strategy_path = tmp_path / 'strategy.json'
        with open(strategy_path, 'w') as f:
            json.dump({
                'name': 'TestStrategy',
                'code': optimizer_with_data._strategy_code,
            }, f)

        optimizer_with_data.optimize(
            strategy_path=str(strategy_path),
            factors=optimizer_with_data._factors,
            close=optimizer_with_data._close,
            n_trials=3,
            show_progress=False,
        )

        optimizer_with_data.update_strategy_metadata(str(strategy_path))

        with open(strategy_path, 'r') as f:
            strategy = json.load(f)

        assert 'optimization' in strategy
        assert 'best_params' in strategy['optimization']
        assert 'timestamp' in strategy['optimization']

    def test_update_strategy_metadata_adds_parameters(self, optimizer_with_data, tmp_path):
        """Test that update adds best params to strategy parameters."""
        strategy_path = tmp_path / 'strategy.json'
        with open(strategy_path, 'w') as f:
            json.dump({
                'name': 'TestStrategy',
                'code': optimizer_with_data._strategy_code,
            }, f)

        optimizer_with_data.optimize(
            strategy_path=str(strategy_path),
            factors=optimizer_with_data._factors,
            close=optimizer_with_data._close,
            n_trials=3,
            show_progress=False,
        )

        optimizer_with_data.update_strategy_metadata(str(strategy_path))

        with open(strategy_path, 'r') as f:
            strategy = json.load(f)

        assert 'parameters' in strategy
        if optimizer_with_data._best_params:
            for key in optimizer_with_data._best_params:
                assert key in strategy['parameters']

    def test_update_strategy_metadata_missing_file_raises_error(self, optimizer):
        """Test that missing strategy file raises FileNotFoundError."""
        with pytest.raises(FileNotFoundError):
            optimizer.update_strategy_metadata('/nonexistent/strategy.json')


# =============================================================================
# Accessor Method Tests
# =============================================================================

@pytest.mark.skipif(not OPTUNA_AVAILABLE, reason="Optuna not installed")
class TestAccessorMethods:
    """Test result accessor methods."""

    def test_get_best_params_before_optimization(self, optimizer):
        """Test get_best_params returns None before optimization."""
        assert optimizer.get_best_params() is None

    def test_get_best_value_before_optimization(self, optimizer):
        """Test get_best_value returns None before optimization."""
        assert optimizer.get_best_value() is None

    def test_get_study_before_optimization(self, optimizer):
        """Test get_study returns None before optimization."""
        assert optimizer.get_study() is None

    def test_get_optimization_history_before_optimization(self, optimizer):
        """Test get_optimization_history returns empty list before optimization."""
        assert optimizer.get_optimization_history() == []

    def test_get_top_trials_empty_before_optimization(self, optimizer):
        """Test get_top_trials returns empty list before optimization."""
        assert optimizer.get_top_trials() == []

    def test_get_top_trials_after_optimization(self, optimizer_with_data, tmp_path):
        """Test get_top_trials returns sorted results after optimization."""
        strategy_path = tmp_path / 'strategy.json'
        with open(strategy_path, 'w') as f:
            json.dump({'name': 'Test', 'code': optimizer_with_data._strategy_code}, f)

        optimizer_with_data.optimize(
            strategy_path=str(strategy_path),
            factors=optimizer_with_data._factors,
            close=optimizer_with_data._close,
            n_trials=5,
            show_progress=False,
        )

        top = optimizer_with_data.get_top_trials(3)
        assert len(top) <= 3

        # Verify sorting (descending by objective)
        for i in range(len(top) - 1):
            assert top[i]['objective'] >= top[i + 1]['objective']


# =============================================================================
# Edge Cases and Error Handling Tests
# =============================================================================

@pytest.mark.skipif(not OPTUNA_AVAILABLE, reason="Optuna not installed")
class TestEdgeCases:
    """Test edge cases and error handling."""

    def test_optimizer_with_custom_param_space(self):
        """Test optimizer with custom parameter space."""
        custom_space = {
            'my_param': {'type': 'uniform', 'low': 0, 'high': 1},
        }
        opt = OptunaOptimizer(parameter_space=custom_space, seed=42)

        assert opt.parameter_names == ['my_param']
        assert opt.parameter_space == custom_space

    def test_set_strategy_code(self, optimizer):
        """Test set_strategy_code method."""
        optimizer.set_strategy_code('print("hello")')
        assert optimizer._strategy_code == 'print("hello")'
        assert optimizer._strategy_name == 'manual_strategy'

    def test_set_factor_data(self, sample_factors, sample_close, optimizer):
        """Test set_factor_data method."""
        optimizer.set_factor_data(sample_factors, sample_close)
        assert optimizer._factors is sample_factors
        assert optimizer._close is sample_close

    def test_simple_backtest_without_data_raises_error(self, optimizer):
        """Test that simple backtest raises error without data."""
        optimizer.set_strategy_code("pass")
        with pytest.raises(RuntimeError, match="No factor data loaded"):
            optimizer._simple_backtest({'entry_threshold': 0.3})

    def test_backtest_engine_run_without_code_raises_error(self, mock_backtest_engine):
        """Test that backtest engine run raises error without strategy code."""
        opt = OptunaOptimizer(backtest_engine=mock_backtest_engine)
        opt._factors = Mock()
        opt._close = Mock()
        with pytest.raises(AttributeError):
            opt._backtest_engine_run({})

    def test_objective_exception_handling(self, optimizer):
        """Test that objective handles exceptions gracefully."""
        study = optuna.create_study(sampler=optuna.samplers.TPESampler(seed=42))

        with patch.object(optimizer, '_run_backtest_with_params') as mock_bt:
            mock_bt.side_effect = RuntimeError("Test exception")

            trial = study.ask()
            value = optimizer.objective(trial)

            # Should return -inf on exception
            assert value == float('-inf')
