"""RL Trading Agent components for NexQuant.

This package provides reinforcement learning trading capabilities.
Works with or without stable-baselines3 (graceful fallback).

OPEN SOURCE: Full RL system works for all GitHub users.
- With stable-baselines3: Full PPO/A2C/SAC training
- Without stable-baselines3: Simple momentum fallback

CLOSED SOURCE: Your trained models in models/local/
"""

# Try to import stable-baselines3
try:
    import stable_baselines3  # noqa: F401

    HAS_STABLE_BASELINES3 = True
except ImportError:
    HAS_STABLE_BASELINES3 = False
    import warnings

    warnings.warn(
        "stable-baselines3 not installed. RL trading will use simple momentum fallback. "
        "Install with: pip install stable-baselines3[extra]",
        UserWarning,
    )

# Always import core components (work regardless of stable-baselines3)
from rdagent.components.coder.rl.env import TradingEnv
from rdagent.components.coder.rl.indicators import (
    calculate_atr,
    calculate_bollinger_bands,
    calculate_cci,
    calculate_macd,
    calculate_rsi,
    prepare_features,
)

# Import RL-specific components only if available
if HAS_STABLE_BASELINES3:
    from rdagent.components.coder.rl.agent import RLTradingAgent
    from rdagent.components.coder.rl.costeer import RLCoSTEER, RLCosteer
else:
    # Use fallback implementations
    from rdagent.components.coder.rl.fallback import SimpleRLFallback as RLTradingAgent
    from rdagent.components.coder.rl.costeer import RLCosteer

    # Create RLCoSTEER stub for when stable-baselines3 is not available
    class RLCoSTEER:  # type: ignore[no-redef]
        """
        Stub RLCoSTEER when stable-baselines3 is not available.

        This class exists only for import compatibility.
        Use RLCosteer with SimpleRLFallback instead.
        """

        def __init__(self, *args, **kwargs):
            raise NotImplementedError(
                "RLCoSTEER requires stable-baselines3. "
                "Install with: pip install stable-baselines3[extra]"
            )


__all__ = [
    "HAS_STABLE_BASELINES3",
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

__version__ = "1.0.0"
