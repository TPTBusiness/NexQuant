"""Tests for fx_validator config (no langchain dependency)."""

from __future__ import annotations

import sys
from pathlib import Path

import pytest

PROJECT_ROOT = Path(__file__).parent.parent.parent
sys.path.insert(0, str(PROJECT_ROOT))


class TestFXConfig:
    """Test config.py directly (bypasses __init__ which imports langchain)."""

    def test_config_is_dict(self):
        # Import config module directly, not via package __init__
        import importlib.util
        spec = importlib.util.spec_from_file_location(
            "fx_validator_config",
            PROJECT_ROOT / "rdagent/scenarios/qlib/fx_validator/config.py",
        )
        mod = importlib.util.module_from_spec(spec)
        spec.loader.exec_module(mod)
        assert isinstance(mod.FX_CONFIG, dict)

    def test_required_keys(self):
        import importlib.util
        spec = importlib.util.spec_from_file_location(
            "fx_validator_config",
            PROJECT_ROOT / "rdagent/scenarios/qlib/fx_validator/config.py",
        )
        mod = importlib.util.module_from_spec(spec)
        spec.loader.exec_module(mod)
        cfg = mod.FX_CONFIG
        assert "instrument" in cfg
        assert "max_debate_rounds" in cfg
        assert "sessions" in cfg
        assert "spread_bps" in cfg

    def test_sessions_have_four_zones(self):
        import importlib.util
        spec = importlib.util.spec_from_file_location(
            "fx_validator_config",
            PROJECT_ROOT / "rdagent/scenarios/qlib/fx_validator/config.py",
        )
        mod = importlib.util.module_from_spec(spec)
        spec.loader.exec_module(mod)
        sessions = mod.FX_CONFIG["sessions"]
        for zone in ("asian", "london", "ny", "overlap"):
            start, end = sessions[zone]
            assert isinstance(start, str)
            assert isinstance(end, str)

    def test_max_debate_rounds_positive(self):
        import importlib.util
        spec = importlib.util.spec_from_file_location(
            "fx_validator_config",
            PROJECT_ROOT / "rdagent/scenarios/qlib/fx_validator/config.py",
        )
        mod = importlib.util.module_from_spec(spec)
        spec.loader.exec_module(mod)
        assert mod.FX_CONFIG["max_debate_rounds"] > 0
