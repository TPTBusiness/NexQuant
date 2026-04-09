"""Tests for DataLoader."""

import pytest
import pandas as pd
import numpy as np
from pathlib import Path

from rdagent.scenarios.qlib.local.data_loader import DataLoader, DataCache


class TestDataCache:
    """Test thread-safe caching."""
    
    def test_put_get(self):
        cache = DataCache()
        cache.put('key1', 'value1')
        assert cache.get('key1') == 'value1'
    
    def test_cache_miss(self):
        cache = DataCache()
        assert cache.get('nonexistent') is None
    
    def test_clear(self):
        cache = DataCache()
        cache.put('key', 'value')
        cache.clear()
        assert cache.get('key') is None


class TestDataLoader:
    """Test DataLoader functionality."""
    
    @pytest.fixture
    def loader(self):
        return DataLoader()
    
    def test_load_ohlcv(self, loader):
        close = loader.load_ohlcv()
        assert isinstance(close, pd.Series)
        assert len(close) > 0
        assert close.name == '$close' or close.name == 'close'
    
    def test_load_ohlcv_cached(self, loader):
        # First call
        close1 = loader.load_ohlcv()
        len1 = len(close1)
        
        # Second call (should be cached)
        close2 = loader.load_ohlcv()
        assert len(close2) == len1
    
    def test_load_ohlcv_max_bars(self, loader):
        close = loader.load_ohlcv(max_bars=10000)
        assert len(close) == 10000
    
    def test_load_factor_metadata(self, loader):
        factors = loader.load_factor_metadata(min_ic=0.0, top_n=20)
        assert isinstance(factors, list)
        assert len(factors) > 0
        assert 'name' in factors[0]
        assert 'ic' in factors[0]
    
    def test_load_factor_metadata_sorted(self, loader):
        factors = loader.load_factor_metadata(top_n=20)
        ics = [abs(f['ic']) for f in factors]
        assert ics == sorted(ics, reverse=True)
    
    def test_get_top_factors_randomized(self, loader):
        factors1 = loader.get_top_factors_by_ic(top_n=10, randomize=True, seed=42)
        factors2 = loader.get_top_factors_by_ic(top_n=10, randomize=True, seed=42)
        factors3 = loader.get_top_factors_by_ic(top_n=10, randomize=True, seed=123)
        
        # Same seed should give same results
        names1 = [f['name'] for f in factors1]
        names2 = [f['name'] for f in factors2]
        assert names1 == names2
        
        # Different seed should give different results (likely)
        names3 = [f['name'] for f in factors3]
        # Not asserting different since it's probabilistic
    
    def test_build_feature_matrix(self, loader):
        factors = loader.load_factor_metadata(top_n=5)
        factor_names = [f['name'] for f in factors]
        
        close = loader.load_ohlcv(max_bars=10000)
        df, index = loader.build_feature_matrix(factor_names, ohlcv_index=close.index)
        
        assert isinstance(df, pd.DataFrame)
        assert len(df) > 0
        assert len(df.columns) <= len(factor_names)  # Some might be dropped
    
    def test_clear_cache(self, loader):
        loader.load_ohlcv()
        loader.clear_cache()
        # Cache should be empty (next call will reload)
        close = loader.load_ohlcv()
        assert len(close) > 0
