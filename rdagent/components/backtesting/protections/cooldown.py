"""
Cooldown Period Protection

Enforces mandatory rest periods after losses.
"""

from dataclasses import dataclass
from datetime import datetime, timedelta
from typing import Optional
from .base import BaseProtection, ProtectionConfig, ProtectionResult, ProtectionType, ProtectionScope


@dataclass
class CooldownConfig(ProtectionConfig):
    """Configuration for Cooldown protection."""
    cooldown_after_loss_pct: float = 0.05  # Cooldown after 5% loss
    cooldown_duration_hours: int = 4  # How long to wait


class CooldownProtection(BaseProtection):
    """
    Enforces cooling-off period after significant losses.

    Prevents revenge trading and emotional decisions by forcing
    a mandatory break after losses exceed threshold.
    """

    def __init__(self, config: CooldownConfig):
        super().__init__(config)
        self.config: CooldownConfig = config
        self.last_loss_time: Optional[datetime] = None
        self.last_loss_pct: float = 0.0

    @property
    def scope(self) -> ProtectionScope:
        return ProtectionScope.FACTOR

    def check(
        self,
        returns: list[float],
        timestamps: list[datetime],
        current_equity: float,
        peak_equity: float,
        **kwargs
    ) -> ProtectionResult:
        """Check if cooldown should be triggered."""
        self.record_check()

        if not self.config.enabled:
            return ProtectionResult(
                should_block=False,
                reason="Protection disabled",
                protection_type=ProtectionType.COOLDOWN
            )

        # Check most recent return
        if returns:
            latest_return = returns[-1]
            latest_time = timestamps[-1] if timestamps else datetime.now()

            if latest_return < -self.config.cooldown_after_loss_pct:
                self.last_loss_time = latest_time
                self.last_loss_pct = latest_return

        # If recently had big loss, enforce cooldown
        if self.last_loss_time:
            time_since_loss = datetime.now() - self.last_loss_time
            if time_since_loss < timedelta(hours=self.config.cooldown_duration_hours):
                remaining = timedelta(hours=self.config.cooldown_duration_hours) - time_since_loss

                result = ProtectionResult(
                    should_block=True,
                    reason=f"Loss of {abs(self.last_loss_pct)*100:.1f}% - cooling down for {remaining.seconds // 3600}h",
                    until=self.last_loss_time + timedelta(hours=self.config.cooldown_duration_hours),
                    protection_type=ProtectionType.COOLDOWN,
                    severity=abs(self.last_loss_pct) / self.config.cooldown_after_loss_pct
                )
                self.record_check(blocked=True)
                return result

        return ProtectionResult(
            should_block=False,
            reason="No recent significant losses",
            protection_type=ProtectionType.COOLDOWN,
            severity=0.0
        )
