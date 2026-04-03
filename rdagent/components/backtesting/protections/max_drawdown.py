"""
Maximum Drawdown Protection

Blocks trading when portfolio drawdown exceeds threshold.
"""

from dataclasses import dataclass
from datetime import datetime, timedelta
from .base import BaseProtection, ProtectionConfig, ProtectionResult, ProtectionType, ProtectionScope


@dataclass
class MaxDrawdownConfig(ProtectionConfig):
    """Configuration for MaxDrawdown protection."""
    max_drawdown_pct: float = 0.15  # Block if drawdown > 15%


class MaxDrawdownProtection(BaseProtection):
    """
    Blocks trading when drawdown exceeds safe threshold.

    This prevents the system from continuing to trade during a losing streak,
    giving the market time to stabilize.
    """

    def __init__(self, config: MaxDrawdownConfig):
        super().__init__(config)
        self.config: MaxDrawdownConfig = config

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
        """Check if drawdown exceeds threshold."""
        self.record_check()

        if not self.config.enabled:
            return ProtectionResult(
                should_block=False,
                reason="Protection disabled",
                protection_type=ProtectionType.MAX_DRAWDOWN
            )

        drawdown = self.calculate_drawdown(current_equity, peak_equity)
        severity = abs(drawdown) / self.config.max_drawdown_pct if self.config.max_drawdown_pct > 0 else 0

        if abs(drawdown) > self.config.max_drawdown_pct:
            # Calculate block duration based on severity
            block_hours = int(self.config.lookback_period_hours * severity)
            block_hours = min(block_hours, 168)  # Max 1 week

            result = ProtectionResult(
                should_block=True,
                reason=f"Drawdown {abs(drawdown)*100:.1f}% exceeds max {self.config.max_drawdown_pct*100:.1f}%",
                until=datetime.now() + timedelta(hours=block_hours),
                protection_type=ProtectionType.MAX_DRAWDOWN,
                severity=severity
            )
            self.record_check(blocked=True)
            return result

        return ProtectionResult(
            should_block=False,
            reason=f"Drawdown {abs(drawdown)*100:.1f}% within safe range",
            protection_type=ProtectionType.MAX_DRAWDOWN,
            severity=severity
        )
