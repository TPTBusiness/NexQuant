"""
Tests for Trading Protection System

Covers all protection types and ProtectionManager.
"""

import pytest
import numpy as np
from datetime import datetime, timedelta
from rdagent.components.backtesting.protections.base import (
    ProtectionResult, ProtectionType, ProtectionScope, ProtectionConfig
)
from rdagent.components.backtesting.protections.max_drawdown import (
    MaxDrawdownProtection, MaxDrawdownConfig
)
from rdagent.components.backtesting.protections.cooldown import (
    CooldownProtection, CooldownConfig
)
from rdagent.components.backtesting.protections.stoploss_guard import (
    StoplossGuardProtection, StoplossGuardConfig
)
from rdagent.components.backtesting.protections.low_performance import (
    LowPerformanceProtection, LowPerformanceConfig
)
from rdagent.components.backtesting.protections.protection_manager import (
    ProtectionManager
)


class TestMaxDrawdownProtection:
    """Test MaxDrawdown protection."""

    def test_within_threshold(self):
        """Test that trading continues when drawdown is acceptable."""
        config = MaxDrawdownConfig(max_drawdown_pct=0.15)
        protection = MaxDrawdownProtection(config)

        # 10% drawdown (below 15% threshold)
        result = protection.check(
            returns=[-0.05, 0.03, -0.02],
            timestamps=[datetime.now()] * 3,
            current_equity=90000,
            peak_equity=100000
        )

        assert not result.should_block
        assert result.protection_type == ProtectionType.MAX_DRAWDOWN

    def test_exceeds_threshold(self):
        """Test that trading blocks when drawdown too high."""
        config = MaxDrawdownConfig(max_drawdown_pct=0.15)
        protection = MaxDrawdownProtection(config)

        # 20% drawdown (above 15% threshold)
        result = protection.check(
            returns=[-0.10, -0.05, -0.05],
            timestamps=[datetime.now()] * 3,
            current_equity=80000,
            peak_equity=100000
        )

        assert result.should_block
        assert result.until is not None
        assert result.severity > 1.0

    def test_disabled_protection(self):
        """Test that disabled protection never blocks."""
        config = MaxDrawdownConfig(max_drawdown_pct=0.15, enabled=False)
        protection = MaxDrawdownProtection(config)

        result = protection.check(
            returns=[],
            timestamps=[],
            current_equity=50000,  # 50% drawdown!
            peak_equity=100000
        )

        assert not result.should_block

    def test_calculate_drawdown(self):
        """Test drawdown calculation."""
        config = MaxDrawdownConfig()
        protection = MaxDrawdownProtection(config)

        assert protection.calculate_drawdown(90000, 100000) == -0.10
        assert protection.calculate_drawdown(100000, 100000) == 0.0
        assert protection.calculate_drawdown(0, 100000) == -1.0

    def test_block_duration_scales_with_severity(self):
        """More severe drawdowns get longer blocks."""
        config = MaxDrawdownConfig(max_drawdown_pct=0.15)
        protection = MaxDrawdownProtection(config)

        # Mild breach (16%)
        result_mild = protection.check(
            returns=[],
            timestamps=[],
            current_equity=84000,
            peak_equity=100000
        )

        # Severe breach (30%)
        result_severe = protection.check(
            returns=[],
            timestamps=[],
            current_equity=70000,
            peak_equity=100000
        )

        assert result_severe.until > result_mild.until


class TestCooldownProtection:
    """Test Cooldown protection."""

    def test_no_recent_loss(self):
        """Test that no loss means no cooldown."""
        config = CooldownConfig(cooldown_after_loss_pct=0.05)
        protection = CooldownProtection(config)

        result = protection.check(
            returns=[0.01, 0.02, 0.01],
            timestamps=[datetime.now()] * 3,
            current_equity=103000,
            peak_equity=103000
        )

        assert not result.should_block

    def test_triggers_after_loss(self):
        """Test that cooldown activates after loss."""
        config = CooldownConfig(
            cooldown_after_loss_pct=0.05,
            cooldown_duration_hours=4
        )
        protection = CooldownProtection(config)

        result = protection.check(
            returns=[-0.06],
            timestamps=[datetime.now()],
            current_equity=94000,
            peak_equity=100000
        )

        assert result.should_block
        assert "cooling down" in result.reason.lower()

    def test_expires_after_duration(self):
        """Test that cooldown expires after time."""
        config = CooldownConfig(
            cooldown_after_loss_pct=0.05,
            cooldown_duration_hours=1
        )
        protection = CooldownProtection(config)

        # Trigger loss in the past (2 hours ago, cooldown is 1 hour)
        past_time = datetime.now() - timedelta(hours=2)
        protection.last_loss_time = past_time
        protection.last_loss_pct = -0.06

        # Should have expired
        result = protection.check(
            returns=[0.01],
            timestamps=[datetime.now()],
            current_equity=94500,
            peak_equity=100000
        )

        assert not result.should_block


class TestStoplossGuardProtection:
    """Test StoplossGuard protection."""

    def test_within_limit(self):
        """Test that few stoplosses don't block."""
        config = StoplossGuardConfig(max_stoplosses_in_period=5)
        protection = StoplossGuardProtection(config)

        result = protection.check(
            returns=[-0.01, -0.015, -0.02],  # 3 stoplosses
            timestamps=[],
            current_equity=95000,
            peak_equity=100000
        )

        assert not result.should_block

    def test_exceeds_limit(self):
        """Test that too many stoplosses block."""
        config = StoplossGuardConfig(
            max_stoplosses_in_period=3,
            stoploss_threshold_pct=0.02
        )
        protection = StoplossGuardProtection(config)

        result = protection.check(
            returns=[-0.03, -0.025, -0.04, -0.021],  # 4 stoplosses (all < -2%)
            timestamps=[],
            current_equity=90000,
            peak_equity=100000
        )

        assert result.should_block
        assert "4 stoplosses" in result.reason

    def test_disabled_protection(self):
        """Test that disabled protection never blocks."""
        config = StoplossGuardConfig(max_stoplosses_in_period=1, enabled=False)
        protection = StoplossGuardProtection(config)

        result = protection.check(
            returns=[-0.03, -0.04, -0.05],  # Many stoplosses
            timestamps=[],
            current_equity=80000,
            peak_equity=100000
        )

        assert not result.should_block


class TestLowPerformanceProtection:
    """Test LowPerformance protection."""

    def test_insufficient_data(self):
        """Test that few trades don't trigger."""
        config = LowPerformanceConfig(min_trades=20)
        protection = LowPerformanceProtection(config)

        result = protection.check(
            returns=[-0.1, -0.1, -0.1],  # Bad but few
            timestamps=[],
            current_equity=70000,
            peak_equity=100000
        )

        assert not result.should_block
        assert "insufficient" in result.reason.lower()

    def test_blocks_poor_sharpe(self):
        """Test that low Sharpe blocks."""
        config = LowPerformanceConfig(
            min_sharpe_ratio=0.5,
            min_trades=20
        )
        protection = LowPerformanceProtection(config)

        # Generate 30 losing trades
        returns = [-0.01] * 30

        result = protection.check(
            returns=returns,
            timestamps=[],
            current_equity=70000,
            peak_equity=100000
        )

        assert result.should_block
        assert "sharpe" in result.reason.lower()

    def test_blocks_low_winrate(self):
        """Test that low win rate blocks."""
        config = LowPerformanceConfig(
            min_win_rate=0.40,
            min_trades=20
        )
        protection = LowPerformanceProtection(config)

        # 10% win rate (90% losses)
        returns = [-0.01] * 27 + [0.01] * 3

        result = protection.check(
            returns=returns,
            timestamps=[],
            current_equity=97000,
            peak_equity=100000
        )

        assert result.should_block
        assert "win rate" in result.reason.lower()

    def test_acceptable_performance(self):
        """Test that good performance passes."""
        config = LowPerformanceConfig(
            min_sharpe_ratio=0.5,
            min_win_rate=0.40,
            min_trades=20
        )
        protection = LowPerformanceProtection(config)

        # 60% win rate
        returns = [0.02] * 30 + [-0.01] * 20

        result = protection.check(
            returns=returns,
            timestamps=[],
            current_equity=110000,
            peak_equity=110000
        )

        assert not result.should_block

    def test_disabled_protection(self):
        """Test that disabled protection never blocks."""
        config = LowPerformanceConfig(
            min_sharpe_ratio=0.5,
            min_win_rate=0.40,
            min_trades=20,
            enabled=False
        )
        protection = LowPerformanceProtection(config)

        # Generate 30 losing trades
        returns = [-0.01] * 30

        result = protection.check(
            returns=returns,
            timestamps=[],
            current_equity=70000,
            peak_equity=100000
        )

        assert not result.should_block


class TestProtectionResult:
    """Test ProtectionResult dataclass."""

    def test_is_active_with_no_until(self):
        """Test is_active when no time-based block."""
        result = ProtectionResult(should_block=True, reason="test")
        assert result.is_active

        result = ProtectionResult(should_block=False, reason="test")
        assert not result.is_active

    def test_is_active_with_future_until(self):
        """Test is_active with future expiration."""
        result = ProtectionResult(
            should_block=True,
            reason="test",
            until=datetime.now() + timedelta(hours=1)
        )
        assert result.is_active

    def test_is_active_with_past_until(self):
        """Test is_active with past expiration."""
        result = ProtectionResult(
            should_block=True,
            reason="test",
            until=datetime.now() - timedelta(hours=1)
        )
        assert not result.is_active


class TestProtectionManager:
    """Test ProtectionManager integration."""

    def test_all_protections_pass(self):
        """Test that good conditions pass all protections."""
        manager = ProtectionManager()
        manager.create_default_protections()

        result = manager.check_all(
            returns=[0.01, 0.02, 0.015],
            timestamps=[datetime.now()] * 3,
            current_equity=105000,
            peak_equity=105000
        )

        assert not result.should_block

    def test_one_protection_blocks(self):
        """Test that single protection blocks all trading."""
        manager = ProtectionManager()
        manager.create_default_protections()

        # Trigger max drawdown
        result = manager.check_all(
            returns=[-0.10, -0.05, -0.05],
            timestamps=[datetime.now()] * 3,
            current_equity=80000,
            peak_equity=100000
        )

        assert result.should_block
        assert "blocked" in result.reason.lower()

    def test_get_stats(self):
        """Test statistics collection."""
        manager = ProtectionManager()
        manager.create_default_protections()

        # Run some checks
        for _ in range(5):
            manager.check_all(
                returns=[0.01],
                timestamps=[datetime.now()],
                current_equity=101000,
                peak_equity=101000
            )

        stats = manager.get_stats()

        assert stats["total_protections"] == 4
        assert "protections" in stats
        assert len(stats["protections"]) == 4

    def test_add_remove_protection(self):
        """Test adding and removing protections."""
        manager = ProtectionManager()

        # Add custom protection
        config = MaxDrawdownConfig(max_drawdown_pct=0.10)
        protection = MaxDrawdownProtection(config)
        manager.add_protection(protection)

        assert len(manager.protections) == 1

        # Remove
        manager.remove_protection(ProtectionType.MAX_DRAWDOWN)
        assert len(manager.protections) == 0

    def test_get_active_blocks(self):
        """Test active blocks retrieval."""
        manager = ProtectionManager()
        manager.create_default_protections()

        # Initially no blocks
        assert len(manager.get_active_blocks()) == 0

        # Trigger a block
        manager.check_all(
            returns=[-0.10, -0.05, -0.05],
            timestamps=[datetime.now()] * 3,
            current_equity=80000,
            peak_equity=100000
        )

        # Should have active block
        blocks = manager.get_active_blocks()
        assert len(blocks) >= 1

    def test_empty_manager_passes(self):
        """Test that empty manager always passes."""
        manager = ProtectionManager()

        result = manager.check_all(
            returns=[],
            timestamps=[],
            current_equity=100000,
            peak_equity=100000
        )

        assert not result.should_block
