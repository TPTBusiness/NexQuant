"""
Trading Protection System

Prevents excessive losses by automatically pausing trading when risk thresholds are exceeded.

Inspired by common trading protection patterns, implemented from scratch for Predix.
"""

from abc import ABC, abstractmethod
from dataclasses import dataclass, field
from datetime import datetime, timedelta
from typing import Optional
from enum import Enum


class ProtectionType(Enum):
    """Type of protection mechanism."""
    MAX_DRAWDOWN = "max_drawdown"
    COOLDOWN = "cooldown"
    STOPLOSS_GUARD = "stoploss_guard"
    LOW_PERFORMANCE = "low_performance"


class ProtectionScope(Enum):
    """What scope does this protection apply to?"""
    GLOBAL = "global"  # All trading
    FACTOR = "factor"  # Per-factor
    PORTFOLIO = "portfolio"  # Per portfolio


@dataclass
class ProtectionResult:
    """Result of a protection check."""
    should_block: bool  # True if trading should be blocked
    reason: str  # Why was it blocked?
    until: Optional[datetime] = None  # If time-based, when does it expire?
    protection_type: Optional[ProtectionType] = None
    severity: float = 0.0  # How severe is the issue (0-1)

    @property
    def is_active(self) -> bool:
        """Check if protection is currently active."""
        if not self.until:
            return self.should_block
        return datetime.now() < self.until and self.should_block


@dataclass
class ProtectionConfig:
    """Base configuration for a protection."""
    enabled: bool = True
    lookback_period_hours: int = 24  # How far back to look
    severity_threshold: float = 0.8  # At what severity to block


class BaseProtection(ABC):
    """
    Base class for all trading protections.

    Each protection checks specific conditions and returns ProtectionResult.
    Multiple protections can be combined in ProtectionManager.
    """

    def __init__(self, config: ProtectionConfig):
        self.config = config
        self.last_check: Optional[datetime] = None
        self.total_checks: int = 0
        self.total_blocks: int = 0

    @abstractmethod
    def check(
        self,
        returns: list[float],
        timestamps: list[datetime],
        current_equity: float,
        peak_equity: float,
        **kwargs
    ) -> ProtectionResult:
        """
        Check if protection should be triggered.

        Parameters
        ----------
        returns : list[float]
            Historical returns in lookback period
        timestamps : list[datetime]
            Timestamps of returns
        current_equity : float
            Current equity value
        peak_equity : float
            Peak equity value (highest ever)
        **kwargs
            Additional context (factor name, portfolio ID, etc.)

        Returns
        -------
        ProtectionResult
            Decision on whether to block trading
        """
        pass

    def calculate_drawdown(self, current: float, peak: float) -> float:
        """Calculate drawdown percentage (negative value)."""
        if peak == 0:
            return 0.0
        return (current - peak) / peak

    def calculate_max_consecutive_losses(self, returns: list[float]) -> int:
        """Find maximum consecutive losing trades."""
        max_losses = 0
        current_losses = 0

        for ret in returns:
            if ret < 0:
                current_losses += 1
                max_losses = max(max_losses, current_losses)
            else:
                current_losses = 0

        return max_losses

    def calculate_recent_loss_rate(self, returns: list[float]) -> float:
        """Calculate percentage of losing trades."""
        if not returns:
            return 0.0
        losses = sum(1 for r in returns if r < 0)
        return losses / len(returns)

    def record_check(self, blocked: bool = False):
        """Record that a check was performed."""
        self.total_checks += 1
        self.last_check = datetime.now()
        if blocked:
            self.total_blocks += 1

    def get_stats(self) -> dict:
        """Get protection statistics."""
        return {
            "type": self.__class__.__name__,
            "enabled": self.config.enabled,
            "total_checks": self.total_checks,
            "total_blocks": self.total_blocks,
            "block_rate": self.total_blocks / max(1, self.total_checks),
            "last_check": self.last_check.isoformat() if self.last_check else None
        }
