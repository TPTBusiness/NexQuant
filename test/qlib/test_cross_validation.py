"""Cross-validation tests: verify metrics are computed correctly."""

from __future__ import annotations

import sys
from pathlib import Path

import numpy as np
import pandas as pd
import pytest

PROJECT_ROOT = Path(__file__).parent.parent.parent
sys.path.insert(0, str(PROJECT_ROOT))


@pytest.fixture
def synthetic_data():
    """Create synthetic multi-index data with known predictive signal."""
    rng = np.random.default_rng(42)
    n_bars = 2000
    dates = pd.date_range("2024-01-01", periods=n_bars, freq="1min")
    idx = pd.MultiIndex.from_arrays([dates, ["EURUSD"] * n_bars], names=["datetime", "instrument"])
    close = 1.10 + rng.normal(0, 0.001, n_bars).cumsum()
    df = pd.DataFrame({"$close": close}, index=idx)
    return df


class TestDirectEvalMetricsCorrectness:
    def test_perfect_predictor_gives_high_ic(self, synthetic_data):
        """Factor predicting sign of next return should have high |IC|."""
        df = synthetic_data
        close = df["$close"]
        fwd = close.groupby(level="instrument").shift(-96) / close - 1
        signal = pd.Series(np.sign(fwd.values), index=df.index)
        signal[pd.isna(signal)] = 0
        
        valid = signal.dropna().index.intersection(fwd.dropna().index)
        if len(valid) < 100:
            pytest.skip("Not enough data")
        ic = signal.loc[valid].corr(fwd.loc[valid])
        assert abs(ic) > 0.3, f"|IC| should be > 0.3, got {ic:.4f}"

    def test_noisy_factor_lower_sharpe(self, synthetic_data):
        """Noisy version should have lower Sharpe than perfect predictor."""
        df = synthetic_data
        close = df["$close"]
        fwd = close.groupby(level="instrument").shift(-96) / close - 1
        signal = pd.Series(np.sign(fwd.values), index=df.index).fillna(0)
        rng = np.random.default_rng(99)
        noisy = signal + rng.normal(0, 0.5, len(signal))

        valid = signal.dropna().index.intersection(fwd.dropna().index)
        if len(valid) < 100:
            pytest.skip("Not enough data")

        ann = np.sqrt(252 * 1440 / 96)
        ret_perfect = np.where(signal.loc[valid] > 0, 1.0, -1.0) * fwd.loc[valid]
        ret_noisy = np.where(noisy.loc[valid] > 0, 1.0, -1.0) * fwd.loc[valid]

        sp = ret_perfect.mean() / ret_perfect.std() * ann if ret_perfect.std() > 0 else 0
        sn = ret_noisy.mean() / ret_noisy.std() * ann if ret_noisy.std() > 0 else 0
        assert sp > sn, f"Perfect Sharpe ({sp:.4f}) > Noisy ({sn:.4f})"

    def test_constant_factor_nan_ic(self):
        """Constant factor should produce NaN IC (zero variance)."""
        dates = pd.date_range("2024-01-01", periods=200, freq="1min")
        idx = pd.MultiIndex.from_arrays([dates, ["EURUSD"] * 200], names=["datetime", "instrument"])
        close = pd.Series(1.10 + np.arange(200) * 0.0001, index=idx)
        fwd = close.groupby(level="instrument").shift(-96) / close - 1
        factor = pd.Series(np.ones(200), index=idx, name="const")
        valid = factor.dropna().index.intersection(fwd.dropna().index)
        if len(valid) < 10:
            pytest.skip("Not enough data")
        ic = factor.loc[valid].corr(fwd.loc[valid])
        assert np.isnan(ic), f"Constant factor should have NaN IC, got {ic}"

    def test_drawdown_bounded(self, synthetic_data):
        """MaxDD on equity must be in [-1, 0]."""
        df = synthetic_data
        close = df["$close"]
        fwd = close.groupby(level="instrument").shift(-96) / close - 1
        factor = pd.Series(np.random.default_rng(99).normal(0, 1, len(df)), index=df.index)

        valid = factor.dropna().index.intersection(fwd.dropna().index)
        if len(valid) < 100:
            pytest.skip("Not enough data")
        signal = np.where(factor.loc[valid] > 0, 1.0, -1.0)
        strategy_ret = signal * fwd.loc[valid]
        equity = (1.0 + strategy_ret).cumprod()
        running_max = equity.expanding().max()
        dd = (equity - running_max) / running_max.replace(0, np.nan)
        assert dd.min() >= -1.0, f"MaxDD {dd.min():.4f} must be >= -1"

    def test_win_rate_not_same_as_factor_sign(self, synthetic_data):
        """Win rate counts profitable strategy periods, not positive factor values."""
        df = synthetic_data
        close = df["$close"]
        fwd = close.groupby(level="instrument").shift(-96) / close - 1
        factor = pd.Series(np.random.default_rng(88).normal(0, 1, len(df)), index=df.index)

        valid = factor.dropna().index.intersection(fwd.dropna().index)
        if len(valid) < 100:
            pytest.skip("Not enough data")
        signal = np.where(factor.loc[valid] > 0, 1.0, -1.0)
        strategy_ret = signal * fwd.loc[valid]
        wr_strategy = (strategy_ret > 0).sum() / len(strategy_ret)
        wr_factor_sign = (factor.loc[valid] > 0).sum() / len(valid)
        # These should differ because factor sign != trade P&L
        assert abs(wr_strategy - wr_factor_sign) > 0.001


class TestCrossValidation:
    def test_ic_and_sharpe_calculable(self, synthetic_data):
        """Verify IC and Sharpe can be computed without errors."""
        df = synthetic_data
        close = df["$close"]
        fwd = close.groupby(level="instrument").shift(-96) / close - 1
        factor = pd.Series(np.random.default_rng(77).normal(0, 1, len(df)), index=df.index)

        valid = factor.dropna().index.intersection(fwd.dropna().index)
        if len(valid) < 100:
            pytest.skip("Not enough data")
        ic = factor.loc[valid].corr(fwd.loc[valid])
        signal = np.where(factor.loc[valid] > 0, 1.0, -1.0)
        strategy_ret = signal * fwd.loc[valid]
        ann = np.sqrt(252 * 1440 / 96)
        sharpe = strategy_ret.mean() / strategy_ret.std() * ann if strategy_ret.std() > 0 else 0
        assert np.isfinite(ic), f"IC should be finite, got {ic}"
        assert np.isfinite(sharpe), f"Sharpe should be finite, got {sharpe}"

    def test_all_metrics_finite(self, synthetic_data):
        """No metric should be inf or NaN for normal data."""
        df = synthetic_data
        close = df["$close"]
        fwd = close.groupby(level="instrument").shift(-96) / close - 1
        factor = pd.Series(np.random.default_rng(66).normal(0, 1, len(df)), index=df.index)

        valid = factor.dropna().index.intersection(fwd.dropna().index)
        if len(valid) < 100:
            pytest.skip("Not enough data")
        signal = np.where(factor.loc[valid] > 0, 1.0, -1.0)
        ret = signal * fwd.loc[valid]
        ann = np.sqrt(252 * 1440 / 96)
        sharpe = ret.mean() / ret.std() * ann if ret.std() > 0 else 0
        equity = (1.0 + ret).cumprod()
        dd = (equity - equity.expanding().max()) / equity.expanding().max().replace(0, np.nan)
        wr = (ret > 0).sum() / len(ret)

        for name, val in [("sharpe", sharpe), ("max_dd", dd.min()), ("win_rate", wr)]:
            assert np.isfinite(val), f"{name} should be finite, got {val}"

    def test_max_dd_bounded(self, synthetic_data):
        """MaxDD on equity between -1.0 and 0.0."""
        df = synthetic_data
        close = df["$close"]
        fwd = close.groupby(level="instrument").shift(-96) / close - 1
        factor = pd.Series(np.random.default_rng(55).normal(0, 1, len(df)), index=df.index)

        valid = factor.dropna().index.intersection(fwd.dropna().index)
        if len(valid) < 100:
            pytest.skip("Not enough data")
        signal = np.where(factor.loc[valid] > 0, 1.0, -1.0)
        ret = signal * fwd.loc[valid]
        equity = (1.0 + ret).cumprod()
        dd = (equity - equity.expanding().max()) / equity.expanding().max().replace(0, np.nan)
        assert -1.0 <= dd.min() <= 0.0, f"MaxDD {dd.min():.4f} not in [-1, 0]"
