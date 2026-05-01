"""
Predix Quant Loop Factory - Selects appropriate workflow based on available components.

This module is the entry point for the quantitative trading loop.
It automatically selects between:
1. Standard Loop (Open Source) - Factor generation + backtesting
2. Advanced Loop (Local/Closed Source) - Full ML pipeline with portfolio & strategy

The selection is based on:
- Availability of local components (ml_trainer, portfolio_optimizer)
- Number of valid factors (threshold: 5000 for advanced loop)

Usage:
    from rdagent.scenarios.qlib.quant_loop_factory import create_quant_loop

    loop = create_quant_loop(scenario)
    loop.run()
"""

from pathlib import Path
from typing import Optional

from rdagent.log import rdagent_logger as logger


# Threshold for advanced loop activation
ADVANCED_LOOP_FACTOR_THRESHOLD = 5000


def has_local_components() -> bool:
    """
    Check if local (closed source) components are available.

    Returns True if:
    - rdagent/scenarios/qlib/local/ml_trainer.py exists
    - rdagent/scenarios/qlib/local/portfolio_optimizer.py exists
    """
    local_dir = Path(__file__).parent / "local"
    if not local_dir.exists():
        return False

    required_files = [
        "ml_trainer.py",
        "portfolio_optimizer.py",
    ]

    for fname in required_files:
        if not (local_dir / fname).exists():
            return False

    return True


def count_valid_factors() -> int:
    """
    Count the number of valid (successful) factors in results/factors/.

    Returns
    -------
    int
        Number of valid factors
    """
    import json
    from glob import glob

    project_root = Path(__file__).parent.parent.parent.parent
    factors_dir = project_root / "results" / "factors"

    if not factors_dir.exists():
        return 0

    count = 0
    for json_file in glob(str(factors_dir / "*.json")):
        try:
            with open(json_file) as f:
                data = json.load(f)
            if data.get("status") == "success" and data.get("ic") is not None:
                count += 1
        except Exception:
            logger.warning("Failed to load factor file %s", json_file, exc_info=True)
            continue

    return count


def create_quant_loop(scenario) -> "BaseQuantLoop":
    """
    Create the appropriate QuantLoop based on available components.

    Priority:
    1. Advanced Loop (if local components exist AND 5000+ factors)
    2. Standard Loop (always available)

    Parameters
    ----------
    scenario : Scenario
        The trading scenario

    Returns
    -------
    BaseQuantLoop
        The appropriate quant loop instance
    """
    local_available = has_local_components()
    factor_count = count_valid_factors()

    logger.info(
        f"Quant Loop Factory: local_components={local_available}, "
        f"valid_factors={factor_count}, threshold={ADVANCED_LOOP_FACTOR_THRESHOLD}"
    )

    if local_available and factor_count >= ADVANCED_LOOP_FACTOR_THRESHOLD:
        logger.info("Creating AdvancedQuantLoop (ML + Portfolio + Strategy)")
        from rdagent.scenarios.qlib.local.quant_loop_advanced import AdvancedQuantLoop
        return AdvancedQuantLoop(scenario)
    else:
        if not local_available:
            logger.info("Local components not found — using StandardQuantLoop")
        else:
            logger.info(
                f"Only {factor_count}/{ADVANCED_LOOP_FACTOR_THRESHOLD} factors — "
                f"using StandardQuantLoop (need {ADVANCED_LOOP_FACTOR_THRESHOLD - factor_count} more)"
            )

        from rdagent.app.qlib_rd_loop.quant import QuantRDLoop
        return QuantRDLoop


# Base class for type hints
class BaseQuantLoop:
    """Base class for quant loops."""
    pass
