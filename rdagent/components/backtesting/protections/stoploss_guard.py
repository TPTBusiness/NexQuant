"""
Stoploss Guard Protection

Detects clusters of stoplosses and blocks trading.
"""

from dataclasses import dataclass
from datetime import datetime
from .base import BaseProtection, ProtectionConfig, ProtectionResult, ProtectionType, ProtectionScope


@dataclass
class StoplossGuardConfig(ProtectionConfig):
    """Configuration for StoplossGuard protection."""
    max_stoplosses_in_period: int = 5  # Max stoplosses allowed
    stoploss_threshold_pct: float = 0.02  # What counts as stoploss (2%)


class StoplossGuardProtection(BaseProtection):
    """
    Detects stoploss clusters and blocks trading.

    Multiple stoplosses in short time indicates bad market conditions
    or strategy no longer working.
    """

    def __init__(self, config: StoplossGuardConfig):
        super().__init__(config)
        self.config: StoplossGuardConfig = config

    @property
    def scope(self) -> ProtectionScope:
        return ProtectionScope.GLOBAL

    def check(
        self,
        returns: list[float],
        timestamps: list[datetime],
        current_equity: float,
        peak_equity: float,
        **kwargs
    ) -> ProtectionResult:
        """Check for stoploss clusters."""
        self.record_check()

        if not self.config.enabled:
            return ProtectionResult(
                should_block=False,
                reason="Protection disabled",
                protection_type=ProtectionType.STOPLOSS_GUARD
            )

        # Count stoplosses (large losses)
        stoplosses = [
            r for r in returns
            if r < -self.config.stoploss_threshold_pct
        ]

        if len(stoplosses) > self.config.max_stoplosses_in_period:
            severity = len(stoplosses) / self.config.max_stoplosses_in_period

            result = ProtectionResult(
                should_block=True,
                reason=f"{len(stoplosses)} stoplosses detected (max {self.config.max_stoplosses_in_period})",
                protection_type=ProtectionType.STOPLOSS_GUARD,
                severity=severity
            )
            self.record_check(blocked=True)
            return result

        return ProtectionResult(
            should_block=False,
            reason=f"{len(stoplosses)} stoplosses (within limit)",
            protection_type=ProtectionType.STOPLOSS_GUARD,
            severity=len(stoplosses) / max(1, self.config.max_stoplosses_in_period)
        )
