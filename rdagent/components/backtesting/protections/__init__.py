"""
Trading Protection System for Predix.

Prevents excessive losses by automatically pausing trading
when risk thresholds are exceeded.

Usage:
    from rdagent.components.backtesting.protections import (
        ProtectionManager,
        MaxDrawdownProtection,
        CooldownProtection,
        StoplossGuardProtection,
        LowPerformanceProtection,
    )

    manager = ProtectionManager()
    manager.create_default_protections()

    result = manager.check_all(
        returns=[0.01, -0.02, 0.015],
        timestamps=[...],
        current_equity=98000,
        peak_equity=100000
    )

    if result.should_block:
        print(f"Trading blocked: {result.reason}")
"""

from .base import (
    BaseProtection,
    ProtectionConfig,
    ProtectionResult,
    ProtectionType,
    ProtectionScope,
)
from .max_drawdown import MaxDrawdownProtection, MaxDrawdownConfig
from .cooldown import CooldownProtection, CooldownConfig
from .stoploss_guard import StoplossGuardProtection, StoplossGuardConfig
from .low_performance import LowPerformanceProtection, LowPerformanceConfig
from .protection_manager import ProtectionManager

__all__ = [
    "BaseProtection",
    "ProtectionConfig",
    "ProtectionResult",
    "ProtectionType",
    "ProtectionScope",
    "MaxDrawdownProtection",
    "MaxDrawdownConfig",
    "CooldownProtection",
    "CooldownConfig",
    "StoplossGuardProtection",
    "StoplossGuardConfig",
    "LowPerformanceProtection",
    "LowPerformanceConfig",
    "ProtectionManager",
]
