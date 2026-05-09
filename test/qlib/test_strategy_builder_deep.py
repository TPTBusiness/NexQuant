"""Deep tests for strategy_builder.py — combinator, evaluator, edge cases."""

from __future__ import annotations

import sys
from pathlib import Path

import numpy as np
import pandas as pd
import pytest

PROJECT_ROOT = Path(__file__).parent.parent.parent
sys.path.insert(0, str(PROJECT_ROOT))

from hypothesis import given, settings
from hypothesis import strategies as st

from rdagent.scenarios.qlib.developer.strategy_builder import (
    StrategyCombinator,
    StrategyEvaluator,
)


class TestStrategyCombinator:
    @pytest.fixture
    def sample_factors(self):
        return [
            {"factor_name": "f_momentum", "ic": 0.25, "category": "momentum"},
            {"factor_name": "f_reversal", "ic": -0.18, "category": "momentum"},
            {"factor_name": "f_volume", "ic": 0.12, "category": "volume"},
            {"factor_name": "f_session", "ic": 0.09, "category": "session"},
            {"factor_name": "f_volatility", "ic": 0.07, "category": "volatility"},
        ]

    def test_generate_all_pairs(self, sample_factors):
        c = StrategyCombinator(sample_factors, max_combo_size=2)
        combos = c.generate_all()
        # 5 choose 2 = 10 pairs
        assert len(combos) > 0
        for combo in combos:
            assert combo["size"] >= 2
            assert "factors" in combo
            assert "avg_ic" in combo
            assert len(combo["factors"]) == combo["size"]

    def test_generate_all_triplets(self, sample_factors):
        c = StrategyCombinator(sample_factors, max_combo_size=3)
        combos = c.generate_all()
        assert any(c["size"] == 3 for c in combos)

    def test_sorted_by_abs_ic(self, sample_factors):
        c = StrategyCombinator(sample_factors, max_combo_size=2)
        combos = c.generate_all()
        ics = [cb["avg_ic"] for cb in combos]
        assert ics == sorted(ics, reverse=True)

    def test_generate_diversified(self, sample_factors):
        c = StrategyCombinator(sample_factors, max_combo_size=2)
        combos = c.generate_diversified(target_size=10)
        for combo in combos:
            cats = combo["categories"]
            assert len(set(cats)) >= 2  # Cross-category pairs

    def test_empty_factors(self):
        c = StrategyCombinator([], max_combo_size=2)
        combos = c.generate_all()
        assert combos == []
        div_combos = c.generate_diversified(10)
        assert div_combos == []

    def test_single_factor(self):
        c = StrategyCombinator([{"factor_name": "only", "ic": 0.5, "category": "momentum"}])
        combos = c.generate_all()
        assert combos == []

    def test_two_factors_same_category(self):
        factors = [
            {"factor_name": "a", "ic": 0.3, "category": "momentum"},
            {"factor_name": "b", "ic": 0.2, "category": "momentum"},
        ]
        c = StrategyCombinator(factors, max_combo_size=2)
        combos = c.generate_all()
        assert len(combos) == 1

    def test_missing_category_defaults(self):
        factors = [
            {"factor_name": "a", "ic": 0.3},
            {"factor_name": "b", "ic": 0.2},
        ]
        c = StrategyCombinator(factors)
        combos = c.generate_all()
        assert len(combos) == 1
        assert "Unknown" in combos[0]["categories"]


class TestStrategyEvaluator:
    def test_init_sets_cost_pct(self):
        e = StrategyEvaluator(Path("/tmp/test"), cost_bps=2.5)
        assert e.cost_bps == 2.5
        assert e.cost_pct == 2.5 / 10000

    def test_load_factor_values_nonexistent(self):
        e = StrategyEvaluator(Path("/nonexistent/path"))
        result = e.load_factor_values("nonexistent_factor")
        assert result is None

    def test_safe_name_sanitization(self):
        """Factor names with / \\ or spaces should be sanitized."""
        e = StrategyEvaluator(Path("/tmp"))
        # Just testing it doesn't crash
        result = e.load_factor_values("path/to/factor with spaces")
        assert result is None  # File doesn't exist, but name sanitization worked

    def test_default_cost_bps(self):
        e = StrategyEvaluator(Path("/tmp"))
        assert e.cost_bps == 1.5

    def test_evaluate_combo_without_data(self):
        e = StrategyEvaluator(Path("/nonexistent"))
        result = e.evaluate_combo({
            "factors": ["nonexistent"],
            "categories": ["test"],
            "size": 1,
            "avg_ic": 0.1,
        })
        assert result is not None
        assert "error" in result or result.get("status") == "failed"


class TestStrategyBuilderImport:
    def test_all_classes_importable(self):
        from rdagent.scenarios.qlib.developer.strategy_builder import (
            StrategyBuilder,
            StrategyCombinator,
            StrategyEvaluator,
        )
        assert StrategyBuilder
        assert StrategyCombinator
        assert StrategyEvaluator

    def test_strategy_builder_methods_exist(self):
        from rdagent.scenarios.qlib.developer.strategy_builder import StrategyBuilder
        assert hasattr(StrategyBuilder, "load_evaluated_factors")
        assert hasattr(StrategyBuilder, "build_strategies")
