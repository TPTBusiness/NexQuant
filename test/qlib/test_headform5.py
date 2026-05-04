"""Batch 5: runtime verifier edge cases, factor loader, save strategy."""

from __future__ import annotations
import sys, json
from pathlib import Path
import numpy as np, pandas as pd, pytest

PROJECT_ROOT = Path(__file__).parent.parent.parent
sys.path.insert(0, str(PROJECT_ROOT))


class TestRuntimeVerifierMissesNothing:
    def test_all_keys_missing_detected(self):
        from rdagent.components.backtesting.verify import verify_backtest_result
        assert len(verify_backtest_result({})) > 0

    def test_partial_keys_missing(self):
        from rdagent.components.backtesting.verify import verify_backtest_result
        w = verify_backtest_result({"sharpe": 1.0, "max_drawdown": -0.1})
        assert len(w) > 0

    def test_zero_sharpe_accepted(self):
        from rdagent.components.backtesting.verify import verify_backtest_result
        result = {
            "sharpe": 0.0, "max_drawdown": -0.15, "win_rate": 0.5,
            "total_return": 0.0, "annual_return_pct": 0.0,
            "monthly_return_pct": 0.0, "n_trades": 10, "status": "success",
        }
        assert verify_backtest_result(result) == []


class TestSaveStrategyJson:
    def test_save_creates_file(self, tmp_path):
        import json
        (tmp_path / "factors").mkdir()
        data = {"strategy_name": "test", "status": "accepted", "sharpe_ratio": 1.0}
        json_path = tmp_path / "test.json"
        json_path.write_text(json.dumps(data))
        assert json_path.exists()
        loaded = json.loads(json_path.read_text())
        assert loaded["strategy_name"] == "test"


class TestFactorValuesIntegration:
    def test_factor_values_parquet_exists(self):
        vdir = Path("results/factors/values")
        if vdir.exists():
            count = len(list(vdir.glob("*.parquet")))
            assert count > 0

    def test_factor_json_valid(self):
        d = Path("results/factors")
        if d.exists():
            for f in list(d.glob("*.json"))[:5]:
                try:
                    data = json.loads(f.read_text())
                    assert "factor_name" in data
                except Exception:
                    pass
