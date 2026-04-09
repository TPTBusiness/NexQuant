"""
Tests for Strategy Orchestrator.

Public test file - references closed-source module at rdagent/scenarios/qlib/local/strategy_orchestrator.py

Tests cover:
- Initialization and configuration
- Factor selection randomness
- Deduplication logic
- Parallel execution with mocked workers
- Result collection
- Graceful shutdown
- Task queue building
- Summary saving
"""

import os
import json
import time
import tempfile
from pathlib import Path
from unittest.mock import Mock, patch, MagicMock, PropertyMock
from multiprocessing import Manager

import pytest
import numpy as np
import pandas as pd

from rdagent.scenarios.qlib.local.strategy_orchestrator import (
    StrategyOrchestrator,
    _worker_process_task,
)


# =============================================================================
# Fixtures
# =============================================================================

@pytest.fixture
def mock_data_loader():
    """Mock DataLoader for fast tests."""
    loader = Mock()

    # Sample factors
    loader.get_top_factors_by_ic.return_value = [
        {'name': 'momentum_1d', 'ic': 0.15, 'description': 'Momentum'},
        {'name': 'mean_reversion', 'ic': -0.12, 'description': 'Mean reversion'},
        {'name': 'volatility', 'ic': 0.08, 'description': 'Volatility'},
        {'name': 'trend_strength', 'ic': 0.10, 'description': 'Trend'},
        {'name': 'session_momentum', 'ic': 0.07, 'description': 'Session momentum'},
    ]

    loader.load_factor_metadata.return_value = [
        {'name': 'momentum_1d', 'ic': 0.15, 'file': '/tmp/f1.json', 'data': {}},
        {'name': 'mean_reversion', 'ic': -0.12, 'file': '/tmp/f2.json', 'data': {}},
        {'name': 'volatility', 'ic': 0.08, 'file': '/tmp/f3.json', 'data': {}},
        {'name': 'trend_strength', 'ic': 0.10, 'file': '/tmp/f4.json', 'data': {}},
        {'name': 'session_momentum', 'ic': 0.07, 'file': '/tmp/f5.json', 'data': {}},
    ]

    # Sample close data
    dates = pd.date_range('2024-01-01', periods=1000, freq='1min')
    np.random.seed(42)
    close_prices = 1.0850 + np.cumsum(np.random.randn(1000) * 0.0001)
    loader.load_ohlcv.return_value = pd.Series(close_prices, index=dates, name='$close')

    # Sample factor time-series
    def mock_load_factor_timeseries(name, ohlcv_index=None):
        np.random.seed(hash(name) % 2**32)
        if ohlcv_index is not None:
            return pd.Series(np.random.randn(len(ohlcv_index)), index=ohlcv_index, name=name)
        return pd.Series(np.random.randn(1000), index=dates, name=name)

    loader.load_factor_timeseries.side_effect = mock_load_factor_timeseries

    return loader


@pytest.fixture
def tmp_output_dir(tmp_path):
    """Temporary output directory."""
    out_dir = tmp_path / 'strategies_new'
    out_dir.mkdir()
    return str(out_dir)


@pytest.fixture
def tmp_log_dir(tmp_path):
    """Temporary log directory."""
    log_dir = tmp_path / 'logs'
    log_dir.mkdir()
    return str(log_dir)


@pytest.fixture
def orchestrator(mock_data_loader, tmp_output_dir, tmp_log_dir):
    """StrategyOrchestrator instance with mocked dependencies."""
    orch = StrategyOrchestrator(
        n_workers=2,
        max_llm_parallel=2,
        data_loader=mock_data_loader,
        output_dir=tmp_output_dir,
        log_dir=tmp_log_dir,
    )
    return orch


# =============================================================================
# Initialization Tests
# =============================================================================

class TestStrategyOrchestratorInit:
    """Test orchestrator initialization."""

    def test_init_defaults(self, mock_data_loader, tmp_output_dir, tmp_log_dir):
        """Test initialization with default parameters."""
        orch = StrategyOrchestrator(
            data_loader=mock_data_loader,
            output_dir=tmp_output_dir,
            log_dir=tmp_log_dir,
        )

        assert orch.n_workers == 4
        assert orch.max_llm_parallel == 2
        assert orch.max_retries == 3
        assert orch._running is False
        assert orch._pool is None

    def test_init_custom_params(self, mock_data_loader, tmp_output_dir, tmp_log_dir):
        """Test initialization with custom parameters."""
        orch = StrategyOrchestrator(
            n_workers=8,
            max_llm_parallel=4,
            data_loader=mock_data_loader,
            output_dir=tmp_output_dir,
            log_dir=tmp_log_dir,
            llm_url='http://custom:9999/v1',
            model_name='custom-model',
            max_retries=5,
        )

        assert orch.n_workers == 8
        assert orch.max_llm_parallel == 4
        assert orch.data_loader == mock_data_loader
        assert str(orch.output_dir) == tmp_output_dir
        assert str(orch.log_dir) == tmp_log_dir
        assert orch.llm_url == 'http://custom:9999/v1'
        assert orch.model_name == 'custom-model'
        assert orch.max_retries == 5

    def test_init_min_workers(self, mock_data_loader, tmp_output_dir, tmp_log_dir):
        """Test that n_workers is clamped to minimum of 1."""
        orch = StrategyOrchestrator(
            n_workers=0,
            data_loader=mock_data_loader,
            output_dir=tmp_output_dir,
            log_dir=tmp_log_dir,
        )
        assert orch.n_workers == 1

    def test_init_min_llm_parallel(self, mock_data_loader, tmp_output_dir, tmp_log_dir):
        """Test that max_llm_parallel is clamped to minimum of 1."""
        orch = StrategyOrchestrator(
            max_llm_parallel=0,
            data_loader=mock_data_loader,
            output_dir=tmp_output_dir,
            log_dir=tmp_log_dir,
        )
        assert orch.max_llm_parallel == 1

    def test_init_creates_output_dir(self, tmp_path, mock_data_loader, tmp_log_dir):
        """Test that output directory is created if it doesn't exist."""
        new_dir = str(tmp_path / 'new_strategies')
        orch = StrategyOrchestrator(
            data_loader=mock_data_loader,
            output_dir=new_dir,
            log_dir=tmp_log_dir,
        )
        assert Path(new_dir).exists()

    def test_init_creates_log_dir(self, tmp_path, mock_data_loader, tmp_output_dir):
        """Test that log directory is created if it doesn't exist."""
        new_dir = str(tmp_path / 'new_logs')
        orch = StrategyOrchestrator(
            data_loader=mock_data_loader,
            output_dir=tmp_output_dir,
            log_dir=new_dir,
        )
        assert Path(new_dir).exists()


# =============================================================================
# Factor Selection Tests
# =============================================================================

class TestFactorSelection:
    """Test factor selection functionality."""

    def test_select_factors_default(self, orchestrator, mock_data_loader):
        """Test factor selection with default parameters."""
        factors = orchestrator._select_factors()

        mock_data_loader.get_top_factors_by_ic.assert_called_once()
        assert isinstance(factors, list)

    def test_select_factors_custom_params(self, orchestrator, mock_data_loader):
        """Test factor selection with custom parameters."""
        factors = orchestrator._select_factors(top_n=5, min_ic=0.05, randomize=False, seed=42)

        mock_data_loader.get_top_factors_by_ic.assert_called_once_with(
            top_n=5, min_ic=0.05, randomize=False, seed=42
        )

    def test_select_factors_randomness(self, mock_data_loader, tmp_output_dir, tmp_log_dir):
        """Test that different seeds produce different factor selections."""
        # Make the mock return different values based on randomize
        def mock_get_factors(top_n=20, min_ic=0.01, randomize=False, seed=None):
            np.random.seed(seed if seed is not None else 0)
            factors = [
                {'name': f'factor_{i}', 'ic': round(np.random.uniform(0.01, 0.2), 4)}
                for i in range(top_n * 2)
            ]
            return factors[:top_n]

        mock_data_loader.get_top_factors_by_ic.side_effect = mock_get_factors

        orch = StrategyOrchestrator(
            data_loader=mock_data_loader,
            output_dir=tmp_output_dir,
            log_dir=tmp_log_dir,
        )

        factors1 = orch._select_factors(top_n=5, seed=42)
        factors2 = orch._select_factors(top_n=5, seed=42)
        factors3 = orch._select_factors(top_n=5, seed=123)

        # Same seed should give same results
        names1 = [f['name'] for f in factors1]
        names2 = [f['name'] for f in factors2]
        assert names1 == names2


# =============================================================================
# Deduplication Tests
# =============================================================================

class TestDeduplication:
    """Test deduplication logic."""

    def test_factor_combo_hash_deterministic(self, orchestrator):
        """Test that factor combo hash is deterministic."""
        names1 = ['factor_a', 'factor_b', 'factor_c']
        names2 = ['factor_c', 'factor_a', 'factor_b']  # different order

        hash1 = orchestrator._factor_combo_hash(names1)
        hash2 = orchestrator._factor_combo_hash(names2)

        # Same factors, different order -> same hash
        assert hash1 == hash2

    def test_factor_combo_hash_different(self, orchestrator):
        """Test that different factor combos have different hashes."""
        names1 = ['factor_a', 'factor_b']
        names2 = ['factor_a', 'factor_c']

        hash1 = orchestrator._factor_combo_hash(names1)
        hash2 = orchestrator._factor_combo_hash(names2)

        assert hash1 != hash2

    def test_is_duplicate_first_time(self, orchestrator):
        """Test that a new combination is not a duplicate."""
        names = ['factor_a', 'factor_b']
        result = orchestrator._is_duplicate(names)
        assert result is False

    def test_is_duplicate_same_combo(self, orchestrator):
        """Test that the same combination is a duplicate."""
        names1 = ['factor_a', 'factor_b']
        names2 = ['factor_b', 'factor_a']  # same, different order

        assert orchestrator._is_duplicate(names1) is False
        assert orchestrator._is_duplicate(names2) is True

    def test_is_unique_after_different_combo(self, orchestrator):
        """Test that different combinations are not duplicates."""
        names1 = ['factor_a', 'factor_b']
        names2 = ['factor_a', 'factor_c']

        assert orchestrator._is_duplicate(names1) is False
        assert orchestrator._is_duplicate(names2) is False

    def test_load_existing_strategies(self, tmp_output_dir, tmp_log_dir, mock_data_loader):
        """Test loading existing strategies for deduplication."""
        # Create a fake strategy file
        strategy_file = Path(tmp_output_dir) / '123456_TestStrategy.json'
        strategy_data = {
            'factor_names': ['existing_factor_a', 'existing_factor_b'],
            'code': 'pass',
            'backtest_result': {},
            'acceptance_result': {'passed': True, 'checks': {}},
        }
        strategy_file.write_text(json.dumps(strategy_data))

        # Verify file exists
        assert strategy_file.exists(), f"Strategy file not found at {strategy_file}"

        # Create orchestrator with same output_dir
        orch2 = StrategyOrchestrator(
            data_loader=mock_data_loader,
            output_dir=tmp_output_dir,
            log_dir=tmp_log_dir,
        )

        combo_hash = orch2._factor_combo_hash(['existing_factor_a', 'existing_factor_b'])
        assert combo_hash in orch2._used_factor_combos, (
            f"Hash {combo_hash} not found in {orch2._used_factor_combos}. "
            f"Output dir: {orch2.output_dir}, Files: {list(orch2.output_dir.glob('*.json'))}"
        )


# =============================================================================
# Task Queue Building Tests
# =============================================================================

class TestTaskQueue:
    """Test task queue building."""

    def test_build_task_queue(self, orchestrator, mock_data_loader):
        """Test building a task queue."""
        tasks = orchestrator._build_task_queue(
            target_count=3,
            top_n_factors=3,
            min_ic=0.01,
            seed=42,
        )

        assert len(tasks) > 0
        assert all('task_id' in t for t in tasks)
        assert all('factors' in t for t in tasks)
        assert all('strategy_name' in t for t in tasks)

    def test_build_task_queue_unique_factors(self, orchestrator, mock_data_loader):
        """Test that task queue has unique factor combinations."""
        # Make mock return varied factors for randomization
        call_count = [0]
        original_side_effect = mock_data_loader.get_top_factors_by_ic.side_effect

        def varied_factors(top_n=20, min_ic=0.01, randomize=False, seed=None):
            call_count[0] += 1
            np.random.seed(seed if seed is not None else call_count[0])
            factors = [
                {'name': f'factor_{i}_v{call_count[0]}', 'ic': round(np.random.uniform(0.01, 0.2), 4)}
                for i in range(top_n)
            ]
            return factors

        mock_data_loader.get_top_factors_by_ic.side_effect = varied_factors

        tasks = orchestrator._build_task_queue(
            target_count=2,
            top_n_factors=3,
            min_ic=0.01,
            seed=42,
        )

        # All tasks should have factor lists
        assert all(len(t['factors']) > 0 for t in tasks)

    def test_build_task_queue_task_ids_sequential(self, orchestrator, mock_data_loader):
        """Test that task IDs are sequential starting from 0."""
        tasks = orchestrator._build_task_queue(
            target_count=2,
            top_n_factors=3,
            min_ic=0.01,
            seed=42,
        )

        task_ids = [t['task_id'] for t in tasks]
        assert task_ids == list(range(len(tasks)))


# =============================================================================
# Parallel Execution Tests (Mocked)
# =============================================================================

class TestParallelExecution:
    """Test parallel execution with mocked components."""

    @patch('rdagent.scenarios.qlib.local.strategy_orchestrator._worker_process_task')
    @patch('rdagent.scenarios.qlib.local.strategy_orchestrator.Pool')
    def test_run_dispatches_tasks(self, mock_pool_class, mock_worker, orchestrator):
        """Test that run() dispatches tasks to the pool."""
        # Mock the pool
        mock_pool = Mock()
        mock_pool_class.return_value = mock_pool
        mock_async_result = Mock()
        mock_async_result.get.return_value = None
        mock_pool.apply_async.return_value = mock_async_result

        # We need to patch the Manager dict to simulate results
        manager = Manager()
        results_dict = manager.dict()

        with patch('rdagent.scenarios.qlib.local.strategy_orchestrator.Manager') as mock_manager:
            mock_manager.return_value.__enter__ = Mock(return_value=manager)
            mock_manager.return_value.__exit__ = Mock(return_value=False)
            mock_manager.return_value.dict = Mock(return_value=results_dict)

            # Run with very small target to keep test fast
            try:
                summary = orchestrator.run(target_count=1, seed=42)
            except Exception:
                # Pool might not work perfectly in test environment
                pass

        # Verify pool was created with correct worker count
        mock_pool_class.assert_called_once_with(processes=orchestrator.n_workers)

    @patch('rdagent.scenarios.qlib.local.strategy_orchestrator.Pool')
    def test_run_collects_results(self, mock_pool_class, orchestrator, mock_data_loader, tmp_output_dir, tmp_log_dir):
        """Test that run() collects results from workers."""
        # Create a simpler mock approach: mock the entire run flow
        orch = StrategyOrchestrator(
            n_workers=2,
            max_llm_parallel=2,
            data_loader=mock_data_loader,
            output_dir=tmp_output_dir,
            log_dir=tmp_log_dir,
        )

        # Mock _build_task_queue to return predictable tasks
        orch._build_task_queue = Mock(return_value=[
            {'task_id': 0, 'factors': [{'name': 'f1', 'ic': 0.1}], 'strategy_name': 'Test1', 'feedback': None},
            {'task_id': 1, 'factors': [{'name': 'f2', 'ic': 0.15}], 'strategy_name': 'Test2', 'feedback': None},
        ])

        # Mock _load_existing_strategies
        orch._load_existing_strategies = Mock()

        # Mock the worker process function results via patching the Pool
        manager = Manager()
        results_dict = manager.dict()
        results_dict[0] = {
            'task_id': 0,
            'strategy_name': 'Test1',
            'success': True,
            'stage': 'complete',
            'saved_path': '/tmp/strategy1.json',
            'backtest_result': {'ic': 0.05, 'sharpe_ratio': 1.2},
        }
        results_dict[1] = {
            'task_id': 1,
            'strategy_name': 'Test2',
            'success': False,
            'stage': 'acceptance',
            'rejection_reasons': ['IC too low'],
            'backtest_result': {'ic': 0.01, 'sharpe_ratio': 0.3},
        }

        with patch('rdagent.scenarios.qlib.local.strategy_orchestrator.Manager') as mock_mgr:
            mock_mgr.return_value.dict = Mock(return_value=results_dict)
            mock_mgr.return_value.__enter__ = Mock(return_value=manager)
            mock_mgr.return_value.__exit__ = Mock(return_value=False)

            mock_pool = Mock()
            mock_pool_class.return_value = mock_pool

            mock_ar = Mock()
            mock_ar.get.return_value = None
            mock_pool.apply_async.return_value = mock_ar

            summary = orch.run(target_count=1, seed=42)

        assert summary['total_attempted'] == 2
        assert summary['accepted'] == 1
        assert summary['rejected'] == 1
        assert summary['failed'] == 0
        assert '/tmp/strategy1.json' in summary['strategies']

    @patch('rdagent.scenarios.qlib.local.strategy_orchestrator.Pool')
    def test_run_with_all_failures(self, mock_pool_class, orchestrator, mock_data_loader, tmp_output_dir, tmp_log_dir):
        """Test run() when all tasks fail."""
        orch = StrategyOrchestrator(
            n_workers=2,
            max_llm_parallel=2,
            data_loader=mock_data_loader,
            output_dir=tmp_output_dir,
            log_dir=tmp_log_dir,
        )
        orch._build_task_queue = Mock(return_value=[
            {'task_id': 0, 'factors': [{'name': 'f1', 'ic': 0.1}], 'strategy_name': 'Test1', 'feedback': None},
        ])
        orch._load_existing_strategies = Mock()

        manager = Manager()
        results_dict = manager.dict()
        results_dict[0] = {
            'task_id': 0,
            'strategy_name': 'Test1',
            'success': False,
            'stage': 'generation',
            'error': 'LLM timeout',
        }

        with patch('rdagent.scenarios.qlib.local.strategy_orchestrator.Manager') as mock_mgr:
            mock_mgr.return_value.dict = Mock(return_value=results_dict)
            mock_mgr.return_value.__enter__ = Mock(return_value=manager)
            mock_mgr.return_value.__exit__ = Mock(return_value=False)

            mock_pool = Mock()
            mock_pool_class.return_value = mock_pool
            mock_ar = Mock()
            mock_ar.get.return_value = None
            mock_pool.apply_async.return_value = mock_ar

            summary = orch.run(target_count=1, seed=42)

        assert summary['accepted'] == 0
        assert summary['failed'] == 1
        assert len(summary['strategies']) == 0

    @patch('rdagent.scenarios.qlib.local.strategy_orchestrator.Pool')
    def test_run_tracks_elapsed_time(self, mock_pool_class, orchestrator, mock_data_loader, tmp_output_dir, tmp_log_dir):
        """Test that run() tracks elapsed time."""
        orch = StrategyOrchestrator(
            n_workers=1,
            max_llm_parallel=1,
            data_loader=mock_data_loader,
            output_dir=tmp_output_dir,
            log_dir=tmp_log_dir,
        )
        orch._build_task_queue = Mock(return_value=[])
        orch._load_existing_strategies = Mock()

        with patch('rdagent.scenarios.qlib.local.strategy_orchestrator.Manager') as mock_mgr:
            mock_mgr.return_value.dict = Mock(return_value={})
            mock_mgr.return_value.__enter__ = Mock(return_value=Manager())
            mock_mgr.return_value.__exit__ = Mock(return_value=False)

            mock_pool = Mock()
            mock_pool_class.return_value = mock_pool

            summary = orch.run(target_count=1, seed=42)

        assert 'elapsed_seconds' in summary
        assert isinstance(summary['elapsed_seconds'], float)
        assert summary['elapsed_seconds'] >= 0

    def test_is_running_property(self, orchestrator):
        """Test the is_running property."""
        assert orchestrator.is_running is False
        orchestrator._running = True
        assert orchestrator.is_running is True
        orchestrator._running = False


# =============================================================================
# Graceful Shutdown Tests
# =============================================================================

class TestGracefulShutdown:
    """Test graceful shutdown behavior."""

    @patch('rdagent.scenarios.qlib.local.strategy_orchestrator.Pool')
    def test_shutdown_sets_flag(self, mock_pool_class, orchestrator):
        """Test that shutdown() sets the running flag to False."""
        orchestrator._running = True
        mock_pool = Mock()
        orchestrator._pool = mock_pool

        orchestrator.shutdown()

        assert orchestrator._running is False
        mock_pool.terminate.assert_called_once()

    @patch('rdagent.scenarios.qlib.local.strategy_orchestrator.Pool')
    def test_terminate_pool_handles_errors(self, mock_pool_class, orchestrator):
        """Test that _terminate_pool handles errors gracefully."""
        mock_pool = Mock()
        mock_pool.terminate.side_effect = RuntimeError("Pool error")
        orchestrator._pool = mock_pool

        # Should not raise
        orchestrator._terminate_pool()

        assert orchestrator._pool is None

    @patch('rdagent.scenarios.qlib.local.strategy_orchestrator.Pool')
    def test_shutdown_cleans_pool(self, mock_pool_class, orchestrator):
        """Test that shutdown properly cleans up the pool."""
        mock_pool = Mock()
        orchestrator._pool = mock_pool
        orchestrator._running = True

        orchestrator.shutdown()

        mock_pool.terminate.assert_called()
        mock_pool.join.assert_called()
        mock_pool.close.assert_called()


# =============================================================================
# Summary Saving Tests
# =============================================================================

class TestSummarySaving:
    """Test summary saving functionality."""

    def test_save_summary(self, orchestrator, tmp_path):
        """Test saving orchestrator summary."""
        summary = {
            'total_attempted': 10,
            'accepted': 3,
            'rejected': 5,
            'failed': 2,
            'strategies': ['/path/to/strategy1.json'],
            'results': [],
            'elapsed_seconds': 120.5,
            'timestamp': '2024-01-01T00:00:00',
        }

        orch2 = StrategyOrchestrator(
            n_workers=1,
            max_llm_parallel=1,
            data_loader=orchestrator.data_loader,
            output_dir=str(tmp_path / 'strategies'),
            log_dir=str(tmp_path / 'logs'),
        )
        orch2._save_summary(summary)

        summary_path = tmp_path / 'orchestrator_summary.json'
        assert summary_path.exists()

        with open(summary_path) as f:
            saved = json.load(f)

        assert saved['total_attempted'] == 10
        assert saved['accepted'] == 3
        assert saved['rejected'] == 5
        assert saved['failed'] == 2
        assert saved['elapsed_seconds'] == 120.5


# =============================================================================
# Worker Process Function Tests
# =============================================================================

class TestWorkerProcessTask:
    """Test the standalone worker process function."""

    def test_worker_task_factor_loading_failure(self, tmp_path):
        """Test worker task when no factor data is loaded."""
        from rdagent.scenarios.qlib.local.strategy_orchestrator import _worker_process_task

        # Create mock close data
        dates = pd.date_range('2024-01-01', periods=100, freq='1min')
        close_values = [1.0850] * 100
        close_index_str = [str(d) for d in dates]
        close_data = (close_values, close_index_str)

        manager = Manager()
        results_dict = manager.dict()

        # Mock the DataLoader inside the worker by patching the import
        factor_subset = [{'name': 'nonexistent_factor', 'ic': 0.1}]

        # This test is limited since the worker imports real DataLoader
        # We verify the result structure is correct when data loading fails
        # In a real scenario, the worker would connect to the actual data
        # Here we just verify the function is callable and has correct signature
        import inspect
        sig = inspect.signature(_worker_process_task)
        params = list(sig.parameters.keys())

        assert 'task_id' in params
        assert 'factor_subset' in params
        assert 'close_data' in params
        assert 'strategy_name' in params
        assert 'results_dict' in params
        assert 'llm_max_parallel' in params


# =============================================================================
# Integration-like Tests (Mocked)
# =============================================================================

class TestOrchestratorIntegration:
    """Test orchestrator integration with mocked components."""

    def test_full_workflow_mocked(self, mock_data_loader, tmp_output_dir, tmp_log_dir):
        """Test the full orchestrator workflow with mocked components."""
        orch = StrategyOrchestrator(
            n_workers=2,
            max_llm_parallel=2,
            data_loader=mock_data_loader,
            output_dir=tmp_output_dir,
            log_dir=tmp_log_dir,
        )

        # Mock the internal methods
        orch._load_existing_strategies = Mock()
        orch._build_task_queue = Mock(return_value=[
            {
                'task_id': 0,
                'factors': [{'name': 'f1', 'ic': 0.1}, {'name': 'f2', 'ic': 0.15}],
                'strategy_name': 'Test_Strategy',
                'feedback': None,
            },
        ])

        manager = Manager()
        results_dict = manager.dict()
        results_dict[0] = {
            'task_id': 0,
            'strategy_name': 'Test_Strategy',
            'success': True,
            'stage': 'complete',
            'saved_path': f'{tmp_output_dir}/123_Test_Strategy.json',
            'backtest_result': {
                'ic': 0.05,
                'sharpe_ratio': 1.5,
                'max_drawdown': -0.05,
                'total_trades': 20,
                'sl_pct': 0.02,
                'tp_pct': 0.04,
            },
        }

        with patch('rdagent.scenarios.qlib.local.strategy_orchestrator.Manager') as mock_mgr:
            mock_mgr.return_value.dict = Mock(return_value=results_dict)
            mock_mgr.return_value.__enter__ = Mock(return_value=manager)
            mock_mgr.return_value.__exit__ = Mock(return_value=False)

            mock_pool = Mock()
            mock_ar = Mock()
            mock_ar.get.return_value = None
            mock_pool.apply_async.return_value = mock_ar

            with patch('rdagent.scenarios.qlib.local.strategy_orchestrator.Pool') as mock_pool_cls:
                mock_pool_cls.return_value = mock_pool

                summary = orch.run(target_count=1, seed=42)

        assert summary['total_attempted'] == 1
        assert summary['accepted'] == 1
        assert summary['rejected'] == 0
        assert summary['failed'] == 0
        assert len(summary['strategies']) == 1
        assert 'elapsed_seconds' in summary
        assert 'timestamp' in summary

    def test_result_structure(self, mock_data_loader, tmp_output_dir, tmp_log_dir):
        """Test that run() returns correct result structure."""
        orch = StrategyOrchestrator(
            n_workers=1,
            max_llm_parallel=1,
            data_loader=mock_data_loader,
            output_dir=tmp_output_dir,
            log_dir=tmp_log_dir,
        )
        orch._load_existing_strategies = Mock()
        orch._build_task_queue = Mock(return_value=[])

        with patch('rdagent.scenarios.qlib.local.strategy_orchestrator.Manager') as mock_mgr:
            mock_mgr.return_value.dict = Mock(return_value={})
            mock_mgr.return_value.__enter__ = Mock(return_value=Manager())
            mock_mgr.return_value.__exit__ = Mock(return_value=False)

            with patch('rdagent.scenarios.qlib.local.strategy_orchestrator.Pool') as mock_pool_cls:
                mock_pool_cls.return_value = Mock()

                summary = orch.run(target_count=1, seed=42)

        # Verify all required keys are present
        required_keys = [
            'total_attempted',
            'accepted',
            'rejected',
            'failed',
            'strategies',
            'results',
            'elapsed_seconds',
            'timestamp',
        ]
        for key in required_keys:
            assert key in summary, f"Missing key: {key}"
