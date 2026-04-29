"""
Low Performance Filter

Blocks trading for factors/portfolios with consistently poor performance.
"""

from dataclasses import dataclass
from datetime import datetime
from .base import BaseProtection, ProtectionConfig, ProtectionResult, ProtectionType, ProtectionScope


@dataclass
class LowPerformanceConfig(ProtectionConfig):
    """Configuration for LowPerformance protection."""
    min_sharpe_ratio: float = 0.5  # Minimum acceptable Sharpe
    min_win_rate: float = 0.40  # Minimum 40% win rate
    min_trades: int = 20  # Need at least this many trades to evaluate  # nosec


class LowPerformanceProtection(BaseProtection):
    """
    Filters out consistently underperforming factors.

    Prevents wasting resources on factors that statistical analysis
    shows are unlikely to become profitable.
    """

    def __init__(self, config: LowPerformanceConfig):
        super().__init__(config)
        self.config: LowPerformanceConfig = config

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
        """Check if performance is below minimum standards."""
        self.record_check()

        if not self.config.enabled:
            return ProtectionResult(
                should_block=False,
                reason="Protection disabled",
                protection_type=ProtectionType.LOW_PERFORMANCE
            )

        # Need minimum number of trades
        if len(returns) < self.config.min_trades:
            return ProtectionResult(
                should_block=False,
                reason=f"Insufficient data ({len(returns)} < {self.config.min_trades} trades)",
                protection_type=ProtectionType.LOW_PERFORMANCE,
                severity=0.0
            )

        # Calculate metrics
        import numpy as np
        returns_array = np.array(returns)

        # Win rate
        wins = int(np.sum(returns_array > 0))
        win_rate = wins / len(returns)

        # Sharpe ratio (annualized, assuming daily returns)
        mean_return = float(np.mean(returns_array))
        std_return = float(np.std(returns_array))
        sharpe = (mean_return / std_return * np.sqrt(252)) if std_return > 0 else 0

        # Check thresholds
        reasons = []
        severity = 0.0

        if sharpe < self.config.min_sharpe_ratio:
            reasons.append(f"Sharpe {sharpe:.2f} < {self.config.min_sharpe_ratio}")
            severity = max(severity, (self.config.min_sharpe_ratio - sharpe) / self.config.min_sharpe_ratio)

        if win_rate < self.config.min_win_rate:
            reasons.append(f"Win rate {win_rate*100:.1f}% < {self.config.min_win_rate*100:.1f}%")
            severity = max(severity, (self.config.min_win_rate - win_rate) / self.config.min_win_rate)

        if reasons:
            result = ProtectionResult(
                should_block=True,
                reason=" | ".join(reasons),
                protection_type=ProtectionType.LOW_PERFORMANCE,
                severity=severity
            )
            self.record_check(blocked=True)
            return result

        return ProtectionResult(
            should_block=False,
            reason=f"Performance acceptable (Sharpe: {sharpe:.2f}, Win rate: {win_rate*100:.1f}%)",
            protection_type=ProtectionType.LOW_PERFORMANCE,
            severity=severity
        )
