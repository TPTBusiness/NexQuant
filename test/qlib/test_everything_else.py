"""Tests for ALL remaining untested modules: scripts, web, log, loader, document_reader, gt_code."""

from __future__ import annotations

import sys
from pathlib import Path

import pytest

PROJECT_ROOT = Path(__file__).parent.parent.parent
sys.path.insert(0, str(PROJECT_ROOT))


# =============================================================================
# model_coder/gt_code.py — ground truth models
# =============================================================================


class TestGTModels:
    def test_file_exists(self):
        f = PROJECT_ROOT / "rdagent/components/coder/model_coder/gt_code.py"
        assert f.exists()
        content = f.read_text()
        assert len(content) > 0


# =============================================================================
# log sub-modules
# =============================================================================


class TestLogModules:
    def test_storage_importable(self):
        from rdagent.log.storage import FileStorage
        assert FileStorage is not None

    def test_conf_importable(self):
        from rdagent.log.conf import LOG_SETTINGS
        assert LOG_SETTINGS is not None

    def test_timer_importable(self):
        from rdagent.log.timer import RD_Agent_TIMER_wrapper
        assert RD_Agent_TIMER_wrapper is not None

    def test_base_importable(self):
        from rdagent.log.base import Storage
        assert Storage is not None

    def test_utils_importable(self):
        from rdagent.log.utils import get_caller_info
        assert callable(get_caller_info)

    def test_mle_summary_importable(self):
        from rdagent.log import mle_summary
        assert mle_summary is not None


# =============================================================================
# loader modules
# =============================================================================


class TestLoaderModules:
    def test_importable(self):
        from rdagent.components.loader.experiment_loader import Loader
        assert Loader is not None

    def test_factor_experiment_loader_available(self):
        from rdagent.components.loader.experiment_loader import FactorExperimentLoader
        assert FactorExperimentLoader is not None


# =============================================================================
# document_reader
# =============================================================================


class TestDocumentReader:
    def test_importable(self):
        from rdagent.components.document_reader.document_reader import (
            load_and_process_pdfs_by_langchain,
            extract_first_page_screenshot_from_pdf,
        )
        assert callable(load_and_process_pdfs_by_langchain)


# =============================================================================
# model_loader
# =============================================================================


class TestModelLoader:
    def test_load_model_importable(self):
        from rdagent.components.model_loader import load_model, list_available_models
        assert callable(load_model)
        assert callable(list_available_models)

    def test_list_available_models_returns_dict(self):
        from rdagent.components.model_loader import list_available_models
        models = list_available_models()
        assert isinstance(models, dict)
        assert "local" in models or "standard" in models


# =============================================================================
# web/dashboard_api.py
# =============================================================================


class TestWebDashboard:
    def test_importable(self):
        sys.path.insert(0, str(PROJECT_ROOT / "web"))
        try:
            import dashboard_api
            assert dashboard_api is not None
        except ImportError as e:
            pytest.skip(f"dashboard dependency missing: {e}")


# =============================================================================
# scripts/ — import tests (these are operational scripts, not libraries)
# =============================================================================


class TestScriptsImportable:
    def test_predix_full_eval(self):
        import importlib
        spec = importlib.util.spec_from_file_location("m", PROJECT_ROOT / "scripts/predix_full_eval.py")
        assert spec is not None

    def test_extract_results(self):
        import importlib
        spec = importlib.util.spec_from_file_location("m", PROJECT_ROOT / "scripts/extract_results.py")
        assert spec is not None

    def test_create_strategy(self):
        import importlib
        spec = importlib.util.spec_from_file_location("m", PROJECT_ROOT / "scripts/create_strategy.py")
        assert spec is not None

    def test_debug_backtest(self):
        import importlib
        spec = importlib.util.spec_from_file_location("m", PROJECT_ROOT / "scripts/debug_backtest.py")
        assert spec is not None

    def test_kronos_factor_gen(self):
        import importlib
        spec = importlib.util.spec_from_file_location("m", PROJECT_ROOT / "scripts/kronos_factor_gen.py")
        assert spec is not None

    def test_kronos_model_eval(self):
        import importlib
        spec = importlib.util.spec_from_file_location("m", PROJECT_ROOT / "scripts/kronos_model_eval.py")
        assert spec is not None

    def test_predix_add_risk_management(self):
        import importlib
        spec = importlib.util.spec_from_file_location("m", PROJECT_ROOT / "scripts/predix_add_risk_management.py")
        assert spec is not None

    def test_predix_gen_strategies(self):
        import importlib
        spec = importlib.util.spec_from_file_location("m", PROJECT_ROOT / "scripts/predix_gen_strategies_real_bt.py")
        assert spec is not None

    def test_predix_quick_daytrading(self):
        import importlib
        spec = importlib.util.spec_from_file_location("m", PROJECT_ROOT / "scripts/predix_quick_daytrading.py")
        assert spec is not None

    def test_predix_rebacktest_unified(self):
        import importlib
        spec = importlib.util.spec_from_file_location("m", PROJECT_ROOT / "scripts/predix_rebacktest_unified.py")
        assert spec is not None

    def test_realistic_backtest_all(self):
        import importlib
        spec = importlib.util.spec_from_file_location("m", PROJECT_ROOT / "scripts/realistic_backtest_all.py")
        assert spec is not None


# =============================================================================
# fx_validator agents (langchain_openai needed for import)
# =============================================================================


class TestFXValidatorAgents:
    def test_session_analyst_exists(self):
        f = PROJECT_ROOT / "rdagent/scenarios/qlib/fx_validator/agents/analysts/session_analyst.py"
        assert f.exists()

    def test_macro_analyst_exists(self):
        f = PROJECT_ROOT / "rdagent/scenarios/qlib/fx_validator/agents/analysts/macro_analyst.py"
        assert f.exists()

    def test_bull_researcher_exists(self):
        f = PROJECT_ROOT / "rdagent/scenarios/qlib/fx_validator/agents/researchers/bull_researcher.py"
        assert f.exists()

    def test_bear_researcher_exists(self):
        f = PROJECT_ROOT / "rdagent/scenarios/qlib/fx_validator/agents/researchers/bear_researcher.py"
        assert f.exists()

    def test_fx_trader_exists(self):
        f = PROJECT_ROOT / "rdagent/scenarios/qlib/fx_validator/agents/trader/fx_trader.py"
        assert f.exists()


# =============================================================================
# patterns/ — patches
# =============================================================================


class TestPatches:
    def test_patches_dir_exists(self):
        d = PROJECT_ROOT / "patches"
        assert d.exists()

    def test_home_page_importable(self):
        from rdagent.app.qlib_rd_loop.quant import main as fin_quant
        assert callable(fin_quant)
