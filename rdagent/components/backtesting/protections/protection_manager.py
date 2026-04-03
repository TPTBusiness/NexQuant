"""
Protection Manager

Orchestrates multiple trading protections.
"""

from datetime import datetime
from typing import Optional, List
from .base import (
    BaseProtection,
    ProtectionConfig,
    ProtectionResult,
    ProtectionType,
    ProtectionScope
)
from .max_drawdown import MaxDrawdownProtection, MaxDrawdownConfig
from .cooldown import CooldownProtection, CooldownConfig
from .stoploss_guard import StoplossGuardProtection, StoplossGuardConfig
from .low_performance import LowPerformanceProtection, LowPerformanceConfig


class ProtectionManager:
    """
    Manages multiple trading protections.

    Run all active protections and aggregate their results.
    If ANY protection returns should_block=True, trading is blocked.
    """

    def __init__(self):
        self.protections: List[BaseProtection] = []
        self.active_blocks: List[ProtectionResult] = []

    def add_protection(self, protection: BaseProtection):
        """Add a protection to the manager."""
        self.protections.append(protection)

    def remove_protection(self, protection_type: ProtectionType):
        """Remove a protection by type."""
        type_to_name = {
            ProtectionType.MAX_DRAWDOWN: "MaxDrawdownProtection",
            ProtectionType.COOLDOWN: "CooldownProtection",
            ProtectionType.STOPLOSS_GUARD: "StoplossGuardProtection",
            ProtectionType.LOW_PERFORMANCE: "LowPerformanceProtection",
        }
        class_name = type_to_name.get(protection_type, protection_type.value)
        self.protections = [
            p for p in self.protections
            if p.__class__.__name__ != class_name
        ]

    def check_all(
        self,
        returns: list[float],
        timestamps: list[datetime],
        current_equity: float,
        peak_equity: float,
        **kwargs
    ) -> ProtectionResult:
        """
        Run all protections and aggregate results.

        Returns
        -------
        ProtectionResult
            Combined result from all protections
        """
        all_results = []

        for protection in self.protections:
            result = protection.check(
                returns=returns,
                timestamps=timestamps,
                current_equity=current_equity,
                peak_equity=peak_equity,
                **kwargs
            )
            all_results.append(result)

        # Check if any protection is blocking
        blocking = [r for r in all_results if r.should_block]

        if blocking:
            # Find most severe block
            most_severe = max(blocking, key=lambda r: r.severity)

            # Clean up expired blocks
            self.active_blocks = [
                b for b in self.active_blocks
                if b.until is None or datetime.now() < b.until
            ]

            # Add new block
            self.active_blocks.append(most_severe)

            # Combine reasons
            reasons = [r.reason for r in blocking]

            return ProtectionResult(
                should_block=True,
                reason=f"Trading blocked: {'; '.join(reasons)}",
                until=most_severe.until,
                severity=most_severe.severity
            )

        return ProtectionResult(
            should_block=False,
            reason="All protections passed",
            severity=0.0
        )

    def get_active_blocks(self) -> List[ProtectionResult]:
        """Get currently active protection blocks."""
        # Clean up expired
        self.active_blocks = [
            b for b in self.active_blocks
            if b.until is None or datetime.now() < b.until
        ]
        return self.active_blocks

    def get_stats(self) -> dict:
        """Get statistics for all protections."""
        return {
            "total_protections": len(self.protections),
            "active_blocks": len(self.get_active_blocks()),
            "protections": [p.get_stats() for p in self.protections]
        }

    def create_default_protections(self):
        """Create standard protection setup."""
        # Max Drawdown: 15% threshold
        self.add_protection(
            MaxDrawdownProtection(
                MaxDrawdownConfig(
                    enabled=True,
                    max_drawdown_pct=0.15,
                    lookback_period_hours=168  # 1 week
                )
            )
        )

        # Cooldown: 4 hours after 5% loss
        self.add_protection(
            CooldownProtection(
                CooldownConfig(
                    enabled=True,
                    cooldown_after_loss_pct=0.05,
                    cooldown_duration_hours=4
                )
            )
        )

        # Stoploss Guard: Max 5 stoplosses per day
        self.add_protection(
            StoplossGuardProtection(
                StoplossGuardConfig(
                    enabled=True,
                    max_stoplosses_in_period=5,
                    stoploss_threshold_pct=0.02,
                    lookback_period_hours=24
                )
            )
        )

        # Low Performance: Filter bad factors
        self.add_protection(
            LowPerformanceProtection(
                LowPerformanceConfig(
                    enabled=True,
                    min_sharpe_ratio=0.5,
                    min_win_rate=0.40,
                    min_trades=20,
                    lookback_period_hours=720  # 30 days
                )
            )
        )
