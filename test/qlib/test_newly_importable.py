"""Tests for newly importable modules (yfinance, rank_bm25, gymnasium installed)."""

from __future__ import annotations

import sys
from pathlib import Path
from unittest.mock import MagicMock, patch

import numpy as np
import pandas as pd
import pytest

PROJECT_ROOT = Path(__file__).parent.parent.parent
sys.path.insert(0, str(PROJECT_ROOT))


# =============================================================================
# eurusd_macro (previously needed yfinance)
# =============================================================================


class TestEurusdMacro:
    def test_importable(self):
        from rdagent.components.coder.factor_coder import eurusd_macro
        assert eurusd_macro is not None

    def test_macro_agent_class_available(self):
        from rdagent.components.coder.factor_coder.eurusd_macro import EURUSDMacroAgent
        assert EURUSDMacroAgent is not None

    def test_macro_signal_class_available(self):
        from rdagent.components.coder.factor_coder.eurusd_macro import MacroSignal
        assert MacroSignal is not None


# =============================================================================
# eurusd_memory (previously needed rank_bm25)
# =============================================================================


class TestEurusdMemory:
    def test_importable(self):
        from rdagent.components.coder.factor_coder import eurusd_memory
        assert eurusd_memory is not None

    def test_memory_class_available(self):
        from rdagent.components.coder.factor_coder.eurusd_memory import EURUSDTradeMemory
        assert EURUSDTradeMemory is not None

    def test_add_and_get_similar(self):
        from rdagent.components.coder.factor_coder.eurusd_memory import EURUSDTradeMemory
        mem = EURUSDTradeMemory()
        mem.add_trade(
            situation="RSI at 30, strong momentum 0.05, low volatility",
            decision="long",
            outcome="win",
            reflection="good timing",
        )
        results = mem.get_similar_setups("RSI 32 momentum")
        assert isinstance(results, dict)
        assert "similar_setups" in results


# =============================================================================
# eurusd_reflection (depends on eurusd_memory)
# =============================================================================


class TestEurusdReflection:
    def test_importable(self):
        from rdagent.components.coder.factor_coder import eurusd_reflection
        assert eurusd_reflection is not None

    def test_reflection_class_available(self):
        from rdagent.components.coder.factor_coder.eurusd_reflection import TradeReflection
        assert TradeReflection is not None


# =============================================================================
# rl/indicators (already tested, now via normal import)
# =============================================================================


class TestRLIndicatorsDirect:
    def test_importable_normally(self):
        from rdagent.components.coder.rl.indicators import (
            calculate_rsi, calculate_macd, calculate_bollinger_bands,
            calculate_atr, calculate_cci, prepare_features,
        )
        assert calculate_rsi is not None
        assert calculate_macd is not None

    def test_rsi_integration(self):
        from rdagent.components.coder.rl.indicators import calculate_rsi
        prices = pd.Series(np.random.default_rng(42).normal(0, 1, 100).cumsum() + 100)
        rsi = calculate_rsi(prices, period=14)
        valid = rsi.dropna()
        assert (valid >= 0).all() and (valid <= 100).all()

    def test_prepare_features_integration(self):
        from rdagent.components.coder.rl.indicators import prepare_features
        df = pd.DataFrame({
            "close": np.random.default_rng(42).normal(0, 1, 200).cumsum() + 100,
            "high": np.random.default_rng(43).normal(0, 1, 200).cumsum() + 101,
            "low": np.random.default_rng(44).normal(0, 1, 200).cumsum() + 99,
        })
        features = prepare_features(df, ["rsi", "macd", "bollinger", "atr"])
        assert isinstance(features, pd.DataFrame)
        assert len(features.columns) > len(df.columns)  # more features added


# =============================================================================
# rl/env.py (now importable with gymnasium)
# =============================================================================


class TestTradingEnv:
    def test_importable(self):
        from rdagent.components.coder.rl.env import TradingEnv
        assert TradingEnv is not None

    def test_class_exists_with_correct_signature(self):
        from rdagent.components.coder.rl.env import TradingEnv
        import inspect
        params = inspect.signature(TradingEnv.__init__).parameters
        assert "prices" in params
        assert "indicators" in params
        assert "window_size" in params
        assert "initial_balance" in params

    def test_env_has_required_methods(self):
        from rdagent.components.coder.rl.env import TradingEnv
        for method in ["reset", "step", "close", "render"]:
            assert hasattr(TradingEnv, method), f"Missing method: {method}"


# =============================================================================
# Previously failing fin_quant integration tests
# =============================================================================


class TestPreviouslyFailingIntegrationTests:
    def test_indicators_module_importable(self):
        from rdagent.components.coder.rl.indicators import (
            calculate_rsi, calculate_macd, calculate_bollinger_bands,
            calculate_cci, calculate_atr, prepare_features,
        )
        assert calculate_rsi is not None

    def test_all_integration_modules_importable(self):
        from rdagent.components.backtesting.protections import ProtectionManager
        from rdagent.components.backtesting import ResultsDatabase
        from rdagent.components.model_loader import load_model, list_available_models
        from rdagent.components.coder.rl.indicators import calculate_rsi
        assert all([ProtectionManager, ResultsDatabase, load_model, list_available_models, calculate_rsi])
