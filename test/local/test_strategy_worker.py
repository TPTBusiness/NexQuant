"""
Tests for Strategy Worker (LLMStrategyGenerator, BacktestEngine, AcceptanceGate, StrategySaver).

Public test file - references closed-source module at rdagent/scenarios/qlib/local/strategy_worker.py
"""

import os
import json
import time
import tempfile
import subprocess
from pathlib import Path
from unittest.mock import Mock, patch, MagicMock

import pytest
import numpy as np
import pandas as pd

from rdagent.scenarios.qlib.local.strategy_worker import (
    LLMStrategyGenerator,
    BacktestEngine,
    AcceptanceGate,
    StrategySaver,
    StrategyWorker,
)


# =============================================================================
# Fixtures
# =============================================================================

@pytest.fixture
def sample_factors():
    """Sample factor metadata list."""
    return [
        {'name': 'momentum_1d', 'ic': 0.15, 'description': '1-day momentum'},
        {'name': 'mean_reversion', 'ic': -0.12, 'description': 'Mean reversion signal'},
        {'name': 'volatility', 'ic': 0.08, 'description': 'Volatility indicator'},
    ]


@pytest.fixture
def sample_close():
    """Sample OHLCV close price series."""
    dates = pd.date_range('2024-01-01', periods=1000, freq='1min')
    np.random.seed(42)
    prices = 1.0850 + np.cumsum(np.random.randn(1000) * 0.0001)
    return pd.Series(prices, index=dates, name='$close')


@pytest.fixture
def sample_factors_df(sample_close):
    """Sample factor values DataFrame."""
    np.random.seed(42)
    return pd.DataFrame({
        'momentum_1d': np.random.randn(len(sample_close)),
        'mean_reversion': np.random.randn(len(sample_close)),
        'volatility': np.random.randn(len(sample_close)),
    }, index=sample_close.index)


@pytest.fixture
def llm_generator():
    """LLM Strategy Generator instance."""
    return LLMStrategyGenerator()


@pytest.fixture
def backtest_engine():
    """Backtest Engine instance."""
    return BacktestEngine(timeout=60)


@pytest.fixture
def acceptance_gate():
    """Acceptance Gate instance."""
    return AcceptanceGate()


@pytest.fixture
def strategy_saver(tmp_path):
    """Strategy Saver instance with temp directory."""
    return StrategySaver(output_dir=str(tmp_path))


# =============================================================================
# LLMStrategyGenerator Tests
# =============================================================================

class TestLLMStrategyGenerator:
    """Test LLM strategy generation."""

    def test_init_defaults(self):
        """Test initialization with defaults."""
        gen = LLMStrategyGenerator()
        assert gen.llm_url == 'http://localhost:8081/v1/chat/completions'
        assert gen.model_name == 'qwen3.5-35b'
        assert gen.timeout == 120
        assert gen.max_tokens == 4096
        assert gen.temperature == 0.5

    def test_init_custom(self):
        """Test initialization with custom parameters."""
        gen = LLMStrategyGenerator(
            llm_url='http://custom:9999/v1',
            model_name='custom-model',
            timeout=60,
            max_tokens=2048,
            temperature=0.8,
        )
        assert gen.llm_url == 'http://custom:9999/v1'
        assert gen.model_name == 'custom-model'
        assert gen.timeout == 60
        assert gen.max_tokens == 2048
        assert gen.temperature == 0.8

    @patch('rdagent.scenarios.qlib.local.strategy_worker.requests.post')
    def test_generate_strategy_success(self, mock_post, llm_generator, sample_factors, sample_close):
        """Test successful strategy generation."""
        # Mock LLM response
        mock_response = Mock()
        mock_response.json.return_value = {
            'choices': [{
                'message': {
                    'content': '''Here is the strategy:

```python
def generate_signal(factors, close):
    import pandas as pd
    import numpy as np
    signal = pd.Series(0, index=close.index)
    signal[factors['momentum_1d'] > 0.5] = 1
    signal[factors['momentum_1d'] < -0.5] = -1
    return signal
```
'''
                }
            }]
        }
        mock_response.raise_for_status = Mock()
        mock_post.return_value = mock_response

        result = llm_generator.generate_strategy(
            factors=sample_factors,
            close=sample_close,
        )

        assert result['success'] is True
        assert 'def generate_signal' in result['code']
        assert result['error'] is None
        assert len(result['factor_names']) == 3
        assert result['attempt_time'] > 0

    @patch('rdagent.scenarios.qlib.local.strategy_worker.requests.post')
    def test_generate_strategy_no_code_block(self, mock_post, llm_generator, sample_factors, sample_close):
        """Test failure when no code block found."""
        mock_response = Mock()
        mock_response.json.return_value = {
            'choices': [{
                'message': {
                    'content': 'This is just text, no code.'
                }
            }]
        }
        mock_response.raise_for_status = Mock()
        mock_post.return_value = mock_response

        result = llm_generator.generate_strategy(
            factors=sample_factors,
            close=sample_close,
        )

        assert result['success'] is False
        assert 'No Python code block found' in result['error']

    @patch('rdagent.scenarios.qlib.local.strategy_worker.requests.post')
    def test_generate_strategy_syntax_error(self, mock_post, llm_generator, sample_factors, sample_close):
        """Test failure when generated code has syntax errors."""
        mock_response = Mock()
        mock_response.json.return_value = {
            'choices': [{
                'message': {
                    'content': '''```python
def generate_signal(factors, close):
    import pandas as pd
    if True
        print("missing colon")
```'''
                }
            }]
        }
        mock_response.raise_for_status = Mock()
        mock_post.return_value = mock_response

        result = llm_generator.generate_strategy(
            factors=sample_factors,
            close=sample_close,
        )

        assert result['success'] is False
        assert 'syntax errors' in result['error']

    @patch('rdagent.scenarios.qlib.local.strategy_worker.requests.post')
    def test_generate_strategy_timeout(self, mock_post, llm_generator, sample_factors, sample_close):
        """Test timeout handling."""
        import requests
        mock_post.side_effect = requests.Timeout('Request timed out')

        result = llm_generator.generate_strategy(
            factors=sample_factors,
            close=sample_close,
        )

        assert result['success'] is False
        assert 'Timeout' in result['error']

    @patch('rdagent.scenarios.qlib.local.strategy_worker.requests.post')
    def test_generate_strategy_connection_error(self, mock_post, llm_generator, sample_factors, sample_close):
        """Test connection error handling."""
        import requests
        mock_post.side_effect = requests.ConnectionError('Connection refused')

        result = llm_generator.generate_strategy(
            factors=sample_factors,
            close=sample_close,
        )

        assert result['success'] is False
        assert 'Connection failed' in result['error']

    def test_extract_code_python_block(self):
        """Test extracting code from ```python block."""
        gen = LLMStrategyGenerator()
        response = '''Some text

```python
def my_func():
    pass
```

More text'''

        code = gen._extract_code(response)
        assert code == 'def my_func():\n    pass'

    def test_extract_code_plain_block(self):
        """Test extracting code from ``` block."""
        gen = LLMStrategyGenerator()
        response = '''```
def my_func():
    pass
```'''

        code = gen._extract_code(response)
        assert code == 'def my_func():\n    pass'

    def test_extract_code_no_block(self):
        """Test when no code block present."""
        gen = LLMStrategyGenerator()
        response = 'Just plain text, no code.'

        code = gen._extract_code(response)
        assert code is None

    def test_validate_code_valid(self):
        """Test validation of valid code."""
        gen = LLMStrategyGenerator()
        assert gen._validate_code('def foo(): pass') is True

    def test_validate_code_invalid(self):
        """Test validation of invalid code."""
        gen = LLMStrategyGenerator()
        assert gen._validate_code('def foo(\n') is False

    @patch.object(LLMStrategyGenerator, 'generate_strategy')
    def test_generate_with_retry_success(self, mock_gen, llm_generator, sample_factors, sample_close):
        """Test successful generation with retry logic."""
        mock_gen.return_value = {
            'success': True,
            'code': 'def generate_signal(f, c): pass',
            'factor_names': ['f1'],
            'llm_response': 'response',
            'error': None,
            'attempt_time': 1.5,
        }

        result = llm_generator.generate_with_retry(
            factors=sample_factors,
            close=sample_close,
            max_retries=3,
        )

        assert result['success'] is True
        assert mock_gen.call_count == 1

    @patch.object(LLMStrategyGenerator, 'generate_strategy')
    def test_generate_with_retry_all_fail(self, mock_gen, llm_generator, sample_factors, sample_close):
        """Test all retries failing."""
        mock_gen.return_value = {
            'success': False,
            'code': '',
            'factor_names': ['f1'],
            'llm_response': '',
            'error': 'Test error',
            'attempt_time': 1.0,
        }

        result = llm_generator.generate_with_retry(
            factors=sample_factors,
            close=sample_close,
            max_retries=3,
        )

        assert result['success'] is False
        assert mock_gen.call_count == 3

    def test_format_factors(self, llm_generator, sample_factors):
        """Test factor formatting for LLM prompt."""
        text = llm_generator._format_factors(sample_factors)
        assert 'momentum_1d' in text
        assert 'IC=0.1500' in text
        assert 'Mean reversion signal' in text


# =============================================================================
# BacktestEngine Tests
# =============================================================================

class TestBacktestEngine:
    """Test backtest engine."""

    def test_init_defaults(self):
        """Test initialization with defaults."""
        engine = BacktestEngine()
        assert engine.stop_loss == 0.02
        assert engine.take_profit == 0.04
        assert engine.trail_activation == 0.015
        assert engine.trailing_stop == 0.015
        assert engine.transaction_cost == 0.00015
        assert engine.timeout == 300

    def test_init_custom(self):
        """Test initialization with custom parameters."""
        engine = BacktestEngine(
            stop_loss=0.01,
            take_profit=0.03,
            trail_activation=0.01,
            trailing_stop=0.01,
            transaction_cost=0.0001,
            timeout=120,
        )
        assert engine.stop_loss == 0.01
        assert engine.take_profit == 0.03
        assert engine.trail_activation == 0.01
        assert engine.trailing_stop == 0.01
        assert engine.transaction_cost == 0.0001
        assert engine.timeout == 120

    def test_run_backtest_valid_strategy(self, backtest_engine, sample_factors_df, sample_close):
        """Test backtest with valid strategy."""
        strategy_code = '''
def generate_signal(factors, close):
    import pandas as pd
    import numpy as np
    signal = pd.Series(0, index=close.index)
    signal[factors['momentum_1d'] > 0.5] = 1
    signal[factors['momentum_1d'] < -0.5] = -1
    return signal
'''

        result = backtest_engine.run_backtest(
            strategy_code=strategy_code,
            factors=sample_factors_df,
            close=sample_close,
        )

        assert result['success'] is True
        assert 'total_trades' in result
        assert 'sharpe_ratio' in result
        assert 'max_drawdown' in result
        assert isinstance(result['total_trades'], int)

    def test_run_backtest_invalid_strategy(self, backtest_engine, sample_factors_df, sample_close):
        """Test backtest with strategy that raises error."""
        strategy_code = '''
def generate_signal(factors, close):
    raise ValueError("Test error")
'''

        result = backtest_engine.run_backtest(
            strategy_code=strategy_code,
            factors=sample_factors_df,
            close=sample_close,
        )

        assert result['success'] is False
        assert 'error' in result

    @patch('rdagent.scenarios.qlib.local.strategy_worker.subprocess.run')
    def test_run_subprocess_timeout(self, mock_run, backtest_engine):
        """Test subprocess timeout handling."""
        import subprocess
        mock_run.side_effect = subprocess.TimeoutExpired(cmd='python', timeout=60)

        result = backtest_engine._run_subprocess(Path('/tmp/run.py'))

        assert result['success'] is False
        assert 'Timeout' in result['error']

    @patch('rdagent.scenarios.qlib.local.strategy_worker.subprocess.run')
    def test_run_subprocess_success(self, mock_run, backtest_engine):
        """Test successful subprocess execution."""
        mock_proc = Mock()
        mock_proc.returncode = 0
        mock_proc.stdout = json.dumps({
            'ic': 0.05,
            'sharpe_ratio': 1.2,
            'max_drawdown': -0.08,
            'win_rate': 0.55,
            'total_trades': 25,
            'wins': 14,
            'losses': 11,
            'total_return': 0.05,
            'final_equity': 1.05,
            'avg_trade_pnl': 0.002,
            'sl_pct': 0.02,
            'tp_pct': 0.04,
            'trail_activation': 0.015,
            'trail_stop': 0.015,
            'transaction_cost': 0.00015,
        })
        mock_run.return_value = mock_proc

        result = backtest_engine._run_subprocess(Path('/tmp/run.py'))

        assert result['success'] is True
        assert result['sharpe_ratio'] == 1.2
        assert result['total_trades'] == 25


# =============================================================================
# AcceptanceGate Tests
# =============================================================================

class TestAcceptanceGate:
    """Test acceptance gate."""

    def test_init_defaults(self):
        """Test initialization with defaults."""
        gate = AcceptanceGate()
        assert gate.min_ic == 0.02
        assert gate.min_sharpe == 0.5
        assert gate.min_trades == 10
        assert gate.max_drawdown == -0.15
        assert gate.ftmo_max_sl == 0.02
        assert gate.ftmo_max_daily_loss == 0.05
        assert gate.ftmo_max_dd == 0.10

    def test_evaluate_passing_strategy(self, acceptance_gate):
        """Test evaluation of passing strategy."""
        result = {
            'ic': 0.05,
            'sharpe_ratio': 1.2,
            'max_drawdown': -0.08,
            'total_trades': 25,
            'sl_pct': 0.02,
        }

        evaluation = acceptance_gate.evaluate(result)

        assert evaluation['passed'] is True
        assert len(evaluation['reasons']) == 0
        assert evaluation['checks']['ic']['passed'] is True
        assert evaluation['checks']['sharpe']['passed'] is True
        assert evaluation['checks']['trades']['passed'] is True
        assert evaluation['checks']['max_drawdown']['passed'] is True
        assert evaluation['checks']['ftmo_sl']['passed'] is True
        assert evaluation['checks']['ftmo_max_dd']['passed'] is True

    def test_evaluate_failing_ic(self, acceptance_gate):
        """Test failure due to low IC."""
        result = {
            'ic': 0.01,
            'sharpe_ratio': 1.2,
            'max_drawdown': -0.08,
            'total_trades': 25,
            'sl_pct': 0.02,
        }

        evaluation = acceptance_gate.evaluate(result)

        assert evaluation['passed'] is False
        assert any('IC' in r for r in evaluation['reasons'])
        assert evaluation['checks']['ic']['passed'] is False

    def test_evaluate_failing_sharpe(self, acceptance_gate):
        """Test failure due to low Sharpe."""
        result = {
            'ic': 0.05,
            'sharpe_ratio': 0.3,
            'max_drawdown': -0.08,
            'total_trades': 25,
            'sl_pct': 0.02,
        }

        evaluation = acceptance_gate.evaluate(result)

        assert evaluation['passed'] is False
        assert any('Sharpe' in r for r in evaluation['reasons'])
        assert evaluation['checks']['sharpe']['passed'] is False

    def test_evaluate_failing_trades(self, acceptance_gate):
        """Test failure due to insufficient trades."""
        result = {
            'ic': 0.05,
            'sharpe_ratio': 1.2,
            'max_drawdown': -0.08,
            'total_trades': 5,
            'sl_pct': 0.02,
        }

        evaluation = acceptance_gate.evaluate(result)

        assert evaluation['passed'] is False
        assert any('trades' in r.lower() for r in evaluation['reasons'])
        assert evaluation['checks']['trades']['passed'] is False

    def test_evaluate_failing_drawdown(self, acceptance_gate):
        """Test failure due to excessive drawdown."""
        result = {
            'ic': 0.05,
            'sharpe_ratio': 1.2,
            'max_drawdown': -0.20,
            'total_trades': 25,
            'sl_pct': 0.02,
        }

        evaluation = acceptance_gate.evaluate(result)

        assert evaluation['passed'] is False
        assert any('DD' in r or 'drawdown' in r.lower() for r in evaluation['reasons'])
        assert evaluation['checks']['max_drawdown']['passed'] is False
        assert evaluation['checks']['ftmo_max_dd']['passed'] is False

    def test_evaluate_failing_ftmo_sl(self, acceptance_gate):
        """Test FTMO stop loss violation."""
        result = {
            'ic': 0.05,
            'sharpe_ratio': 1.2,
            'max_drawdown': -0.08,
            'total_trades': 25,
            'sl_pct': 0.03,
        }

        evaluation = acceptance_gate.evaluate(result)

        assert evaluation['passed'] is False
        assert evaluation['checks']['ftmo_sl']['passed'] is False

    def test_evaluate_ic_none(self, acceptance_gate):
        """Test when IC is None."""
        result = {
            'ic': None,
            'sharpe_ratio': 1.2,
            'max_drawdown': -0.08,
            'total_trades': 25,
            'sl_pct': 0.02,
        }

        evaluation = acceptance_gate.evaluate(result)

        assert evaluation['passed'] is False
        assert any('IC is None' in r for r in evaluation['reasons'])


# =============================================================================
# StrategySaver Tests
# =============================================================================

class TestStrategySaver:
    """Test strategy saver."""

    def test_init_default(self):
        """Test initialization with defaults."""
        saver = StrategySaver()
        assert 'results/strategies_new' in str(saver.output_dir)

    def test_init_custom(self, tmp_path):
        """Test initialization with custom directory."""
        saver = StrategySaver(output_dir=str(tmp_path))
        assert saver.output_dir == tmp_path

    def test_save_strategy(self, strategy_saver):
        """Test saving accepted strategy."""
        filepath = strategy_saver.save_strategy(
            name='TestStrategy',
            code='def generate_signal(f, c): pass',
            factor_names=['f1', 'f2'],
            backtest_result={
                'ic': 0.05,
                'sharpe_ratio': 1.2,
                'max_drawdown': -0.08,
                'total_trades': 25,
                'sl_pct': 0.02,
                'tp_pct': 0.04,
            },
            acceptance_result={
                'passed': True,
                'checks': {},
            },
        )

        assert filepath.exists()
        assert filepath.suffix == '.json'

        # Verify content
        with open(filepath) as f:
            data = json.load(f)

        assert data['strategy_name'] == 'TestStrategy'
        assert 'def generate_signal' in data['code']
        assert data['factor_names'] == ['f1', 'f2']
        assert data['metrics']['ic'] == 0.05
        assert data['metrics']['sharpe_ratio'] == 1.2
        assert 'risk_config' in data
        assert 'acceptance_result' in data

    def test_save_strategy_with_metadata(self, strategy_saver):
        """Test saving strategy with additional metadata."""
        filepath = strategy_saver.save_strategy(
            name='TestStrategy',
            code='def generate_signal(f, c): pass',
            factor_names=['f1'],
            backtest_result={
                'ic': 0.05,
                'sharpe_ratio': 1.2,
                'max_drawdown': -0.08,
                'total_trades': 25,
                'sl_pct': 0.02,
                'tp_pct': 0.04,
            },
            acceptance_result={
                'passed': True,
                'checks': {},
            },
            metadata={'version': '1.0', 'author': 'NexQuant'},
        )

        with open(filepath) as f:
            data = json.load(f)

        assert data['metadata']['version'] == '1.0'
        assert data['metadata']['author'] == 'NexQuant'

    def test_save_strategy_with_llm_response(self, strategy_saver):
        """Test saving strategy with LLM response preview."""
        filepath = strategy_saver.save_strategy(
            name='TestStrategy',
            code='def generate_signal(f, c): pass',
            factor_names=['f1'],
            backtest_result={
                'ic': 0.05,
                'sharpe_ratio': 1.2,
                'max_drawdown': -0.08,
                'total_trades': 25,
                'sl_pct': 0.02,
                'tp_pct': 0.04,
            },
            acceptance_result={
                'passed': True,
                'checks': {},
            },
            llm_response='This is a very long LLM response...',
        )

        with open(filepath) as f:
            data = json.load(f)

        assert 'llm_response_preview' in data
        assert len(data['llm_response_preview']) <= 1000

    def test_filename_format(self, strategy_saver):
        """Test filename format: timestamp_name.json."""
        filepath = strategy_saver.save_strategy(
            name='My Strategy',
            code='pass',
            factor_names=['f1'],
            backtest_result={'ic': 0.05, 'sharpe_ratio': 1.2, 'max_drawdown': -0.08,
                            'total_trades': 25, 'sl_pct': 0.02, 'tp_pct': 0.04},
            acceptance_result={'passed': True, 'checks': {}},
        )

        # Should match: {timestamp}_{name}.json
        assert filepath.name.endswith('_My_Strategy.json')
        assert filepath.name.split('_')[0].isdigit()


# =============================================================================
# StrategyWorker Integration Tests
# =============================================================================

class TestStrategyWorker:
    """Test Strategy Worker integration."""

    def test_init_defaults(self):
        """Test initialization with defaults."""
        worker = StrategyWorker()
        assert isinstance(worker.llm_generator, LLMStrategyGenerator)
        assert isinstance(worker.backtest_engine, BacktestEngine)
        assert isinstance(worker.acceptance_gate, AcceptanceGate)
        assert isinstance(worker.strategy_saver, StrategySaver)

    def test_init_custom(self):
        """Test initialization with custom components."""
        custom_gen = Mock()
        custom_bt = Mock()
        custom_gate = Mock()
        custom_saver = Mock()

        worker = StrategyWorker(
            llm_generator=custom_gen,
            backtest_engine=custom_bt,
            acceptance_gate=custom_gate,
            strategy_saver=custom_saver,
        )

        assert worker.llm_generator == custom_gen
        assert worker.backtest_engine == custom_bt
        assert worker.acceptance_gate == custom_gate
        assert worker.strategy_saver == custom_saver

    @patch.object(StrategyWorker, '__init__', lambda self: None)
    def test_workflow_generation_failure(self, sample_factors, sample_factors_df, sample_close):
        """Test workflow when generation fails."""
        worker = StrategyWorker()
        worker.llm_generator = Mock()
        worker.llm_generator.generate_with_retry.return_value = {
            'success': False,
            'error': 'LLM error',
            'code': '',
        }

        result = worker.run_workflow(
            factors=sample_factors,
            factor_data=sample_factors_df,
            close=sample_close,
            strategy_name='TestStrategy',
        )

        assert result['success'] is False
        assert result['stage'] == 'generation'
        assert result['error'] == 'LLM error'

    @patch.object(StrategyWorker, '__init__', lambda self: None)
    def test_workflow_backtest_failure(self, sample_factors, sample_factors_df, sample_close):
        """Test workflow when backtest fails."""
        worker = StrategyWorker()
        worker.llm_generator = Mock()
        worker.llm_generator.generate_with_retry.return_value = {
            'success': True,
            'code': 'def generate_signal(f, c): pass',
            'factor_names': ['f1'],
            'llm_response': 'response',
            'error': None,
            'attempt_time': 1.0,
        }
        worker.backtest_engine = Mock()
        worker.backtest_engine.run_backtest.return_value = {
            'success': False,
            'error': 'Backtest error',
        }

        result = worker.run_workflow(
            factors=sample_factors,
            factor_data=sample_factors_df,
            close=sample_close,
            strategy_name='TestStrategy',
        )

        assert result['success'] is False
        assert result['stage'] == 'backtest'
        assert result['error'] == 'Backtest error'

    @patch.object(StrategyWorker, '__init__', lambda self: None)
    def test_workflow_acceptance_failure(self, sample_factors, sample_factors_df, sample_close):
        """Test workflow when acceptance gate rejects."""
        worker = StrategyWorker()
        worker.llm_generator = Mock()
        worker.llm_generator.generate_with_retry.return_value = {
            'success': True,
            'code': 'def generate_signal(f, c): pass',
            'factor_names': ['f1'],
            'llm_response': 'response',
            'error': None,
            'attempt_time': 1.0,
        }
        worker.backtest_engine = Mock()
        worker.backtest_engine.run_backtest.return_value = {
            'success': True,
            'ic': 0.01,
            'sharpe_ratio': 0.3,
            'max_drawdown': -0.08,
            'total_trades': 25,
            'sl_pct': 0.02,
        }
        worker.acceptance_gate = Mock()
        worker.acceptance_gate.evaluate.return_value = {
            'passed': False,
            'reasons': ['IC too low', 'Sharpe too low'],
            'checks': {},
        }

        result = worker.run_workflow(
            factors=sample_factors,
            factor_data=sample_factors_df,
            close=sample_close,
            strategy_name='TestStrategy',
        )

        assert result['success'] is False
        assert result['stage'] == 'acceptance'
        assert 'IC too low' in result['rejection_reasons']

    @patch.object(StrategyWorker, '__init__', lambda self: None)
    def test_workflow_success(self, sample_factors, sample_factors_df, sample_close, tmp_path):
        """Test successful workflow completion."""
        worker = StrategyWorker()
        worker.llm_generator = Mock()
        worker.llm_generator.generate_with_retry.return_value = {
            'success': True,
            'code': 'def generate_signal(f, c): pass',
            'factor_names': ['f1'],
            'llm_response': 'response',
            'error': None,
            'attempt_time': 1.0,
        }
        worker.backtest_engine = Mock()
        worker.backtest_engine.run_backtest.return_value = {
            'success': True,
            'ic': 0.05,
            'sharpe_ratio': 1.2,
            'max_drawdown': -0.08,
            'total_trades': 25,
            'sl_pct': 0.02,
            'tp_pct': 0.04,
            'trail_activation': 0.015,
            'trail_stop': 0.015,
            'transaction_cost': 0.00015,
        }
        worker.acceptance_gate = Mock()
        worker.acceptance_gate.evaluate.return_value = {
            'passed': True,
            'reasons': [],
            'checks': {},
        }
        worker.strategy_saver = StrategySaver(output_dir=str(tmp_path))
        worker.strategy_saver.save_strategy = Mock(return_value=tmp_path / 'test.json')

        result = worker.run_workflow(
            factors=sample_factors,
            factor_data=sample_factors_df,
            close=sample_close,
            strategy_name='TestStrategy',
        )

        assert result['success'] is True
        assert result['stage'] == 'complete'
        worker.strategy_saver.save_strategy.assert_called_once()
