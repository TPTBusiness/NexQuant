"""RL Trading Agent components for Predix.

This package provides:
- RLCoSTEER: LLM-based code generation for RL training pipelines
- RLCosteer: RL-based trading controller using trained models
- TradingEnv: Gym-compatible trading environment
- RLTradingAgent: Stable Baselines3 wrapper for PPO, A2C, SAC
- Technical indicators: RSI, MACD, Bollinger Bands, CCI, ATR
"""

from rdagent.components.coder.rl.agent import RLTradingAgent
from rdagent.components.coder.rl.costeer import RLCoSTEER, RLCosteer
from rdagent.components.coder.rl.env import TradingEnv
from rdagent.components.coder.rl.indicators import (
    calculate_atr,
    calculate_bollinger_bands,
    calculate_cci,
    calculate_macd,
    calculate_rsi,
    prepare_features,
)

__all__ = [
    "RLCoSTEER",
    "RLCosteer",
    "RLTradingAgent",
    "TradingEnv",
    "calculate_atr",
    "calculate_bollinger_bands",
    "calculate_cci",
    "calculate_macd",
    "calculate_rsi",
    "prepare_features",
]
