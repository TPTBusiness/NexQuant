"""
Tests for MLTrainer - ML Training Pipeline for Predix quant trading system.

Tests cover:
- Feature matrix building
- Train/validation split (time-series)
- LightGBM training (mocked)
- Feature importance extraction
- Model saving/loading
- Feedback generation
- Edge cases and error handling

Run: pytest test/local/test_ml_trainer.py -v
"""

import json
import tempfile
from pathlib import Path
from unittest.mock import MagicMock, patch

import numpy as np
import pandas as pd
import pytest

from rdagent.scenarios.qlib.local.ml_trainer import MLTrainer, get_ml_trainer


# ---------------------------------------------------------------------------
# Fixtures
# ---------------------------------------------------------------------------

@pytest.fixture
def tmp_models_dir(tmp_path):
    """Temporary directory for model output."""
    return str(tmp_path / "models")


@pytest.fixture
def trainer(tmp_models_dir):
    """MLTrainer instance with temp directory."""
    return MLTrainer(models_dir=tmp_models_dir, random_state=42)


@pytest.fixture
def sample_factors():
    """List of sample factor info dicts."""
    return [
        {
            "factor_name": "momentum_5min",
            "ic": 0.12,
            "sharpe_ratio": 1.5,
            "status": "success",
            "workspace_hash": "abc123",
        },
        {
            "factor_name": "mean_reversion_zscore",
            "ic": 0.08,
            "sharpe_ratio": 1.2,
            "status": "success",
            "workspace_hash": "def456",
        },
        {
            "factor_name": "volatility_atr",
            "ic": -0.05,
            "sharpe_ratio": 0.8,
            "status": "success",
            "workspace_hash": "ghi789",
        },
        {
            "factor_name": "low_quality_factor",
            "ic": 0.005,
            "status": "success",
            "workspace_hash": "jkl012",
        },
        {
            "factor_name": "failed_factor",
            "status": "failed",
        },
    ]


@pytest.fixture
def sample_feature_matrix():
    """Sample X, y data for training tests."""
    np.random.seed(42)
    n_samples = 1000
    n_features = 5

    X = pd.DataFrame(
        np.random.randn(n_samples, n_features),
        columns=[
            "momentum_5min",
            "momentum_15min",
            "zscore_20",
            "atr_ratio",
            "volume_spike",
        ],
    )
    # Create y with some signal from X
    y = pd.Series(
        0.5 * X["momentum_5min"]
        + 0.3 * X["zscore_20"]
        + 0.2 * X["atr_ratio"]
        + np.random.randn(n_samples) * 0.1
    )
    return X, y


@pytest.fixture
def mock_model_info(sample_feature_matrix):
    """Mock model info for testing without actual training."""
    X, y = sample_feature_matrix
    return {
        "model": MagicMock(),
        "feature_names": list(X.columns),
        "feature_importance": {
            "momentum_5min": {"gain": 500.0, "split": 30, "gain_normalized": 0.50},
            "momentum_15min": {"gain": 200.0, "split": 15, "gain_normalized": 0.20},
            "zscore_20": {"gain": 150.0, "split": 12, "gain_normalized": 0.15},
            "atr_ratio": {"gain": 100.0, "split": 8, "gain_normalized": 0.10},
            "volume_spike": {"gain": 50.0, "split": 5, "gain_normalized": 0.05},
        },
        "feature_ranking": [
            "momentum_5min",
            "momentum_15min",
            "zscore_20",
            "atr_ratio",
            "volume_spike",
        ],
        "ic_train": 0.15,
        "ic_valid": 0.10,
        "rank_ic_train": 0.12,
        "rank_ic_valid": 0.08,
        "sharpe_valid": 0.05,
        "mse_train": 0.01,
        "mse_valid": 0.015,
        "n_features": 5,
        "n_samples_train": 800,
        "n_samples_valid": 200,
        "trained_at": "2026-04-09T12:00:00",
        "params": {
            "n_estimators": 500,
            "max_depth": 8,
            "learning_rate": 0.03,
            "subsample": 0.8,
            "colsample_bytree": 0.8,
            "random_state": 42,
        },
    }


# ---------------------------------------------------------------------------
# Test: MLTrainer Initialization
# ---------------------------------------------------------------------------

class TestMLTrainerInit:
    """Test MLTrainer initialization."""

    def test_default_init(self):
        """Test default initialization."""
        trainer = MLTrainer()
        assert trainer.random_state == 42
        assert "results" in str(trainer.models_dir)
        assert "models" in str(trainer.models_dir)

    def test_custom_models_dir(self, tmp_path):
        """Test custom models directory."""
        custom_dir = str(tmp_path / "custom_models")
        trainer = MLTrainer(models_dir=custom_dir, random_state=123)
        assert str(trainer.models_dir) == custom_dir
        assert trainer.random_state == 123

    def test_models_dir_created(self, tmp_path):
        """Test models directory is created if missing."""
        new_dir = str(tmp_path / "nested" / "models")
        trainer = MLTrainer(models_dir=new_dir)
        assert Path(new_dir).exists()


# ---------------------------------------------------------------------------
# Test: load_top_factors
# ---------------------------------------------------------------------------

class TestLoadTopFactors:
    """Test loading top factors from JSON files."""

    def test_load_factors_from_dir(self, trainer, tmp_path, sample_factors):
        """Test loading factors from directory."""
        # Create factor JSON files
        factors_dir = tmp_path / "factors"
        factors_dir.mkdir()
        for i, factor in enumerate(sample_factors[:3]):
            (factors_dir / f"factor_{i}.json").write_text(
                json.dumps(factor), encoding="utf-8"
            )

        result = trainer.load_top_factors(
            top_n=2, min_ic=0.01, factors_dir=str(factors_dir)
        )

        assert len(result) == 2
        # Sorted by |IC| descending
        assert abs(result[0]["ic"]) >= abs(result[1]["ic"])
        assert result[0]["factor_name"] == "momentum_5min"

    def test_filters_by_status(self, trainer, tmp_path, sample_factors):
        """Test filtering out failed factors."""
        factors_dir = tmp_path / "factors"
        factors_dir.mkdir()
        # Write one success, one failed
        (factors_dir / "success.json").write_text(
            json.dumps(sample_factors[0]), encoding="utf-8"
        )
        (factors_dir / "failed.json").write_text(
            json.dumps(sample_factors[4]), encoding="utf-8"
        )

        result = trainer.load_top_factors(factors_dir=str(factors_dir))
        assert len(result) == 1
        assert result[0]["status"] == "success"

    def test_filters_by_min_ic(self, trainer, tmp_path, sample_factors):
        """Test filtering by minimum IC threshold."""
        factors_dir = tmp_path / "factors"
        factors_dir.mkdir()
        # Write factors with different IC
        (factors_dir / "high_ic.json").write_text(
            json.dumps(sample_factors[0]), encoding="utf-8"
        )
        (factors_dir / "low_ic.json").write_text(
            json.dumps(sample_factors[3]), encoding="utf-8"
        )

        result = trainer.load_top_factors(min_ic=0.05, factors_dir=str(factors_dir))
        assert len(result) == 1
        assert result[0]["factor_name"] == "momentum_5min"

    def test_empty_directory(self, trainer, tmp_path):
        """Test loading from empty directory."""
        factors_dir = tmp_path / "empty"
        factors_dir.mkdir()

        result = trainer.load_top_factors(factors_dir=str(factors_dir))
        assert result == []

    def test_nonexistent_directory(self, trainer):
        """Test loading from non-existent directory."""
        result = trainer.load_top_factors(factors_dir="/nonexistent/path")
        assert result == []

    def test_top_n_limit(self, trainer, tmp_path, sample_factors):
        """Test top_n limit is respected."""
        factors_dir = tmp_path / "factors"
        factors_dir.mkdir()
        for i in range(10):
            factor = {
                "factor_name": f"factor_{i}",
                "ic": 0.1 - i * 0.005,
                "status": "success",
                "workspace_hash": f"hash_{i}",
            }
            (factors_dir / f"factor_{i}.json").write_text(
                json.dumps(factor), encoding="utf-8"
            )

        result = trainer.load_top_factors(top_n=3, min_ic=0.01, factors_dir=str(factors_dir))
        assert len(result) == 3


# ---------------------------------------------------------------------------
# Test: build_feature_matrix
# ---------------------------------------------------------------------------

class TestBuildFeatureMatrix:
    """Test feature matrix building."""

    def test_data_file_not_found(self, trainer, sample_factors):
        """Test when data file does not exist."""
        X, y = trainer.build_feature_matrix(
            sample_factors, data_file="/nonexistent/data.h5"
        )
        assert X is None
        assert y is None

    def test_no_workspace_hash(self, trainer, tmp_path, sample_factors):
        """Test handling of factors without workspace_hash."""
        factors_no_hash = [{"factor_name": "test", "ic": 0.1, "status": "success"}]
        X, y = trainer.build_feature_matrix(
            factors_no_hash, data_file=str(tmp_path / "data.h5")
        )
        assert X is None
        assert y is None


# ---------------------------------------------------------------------------
# Test: train_lightgbm
# ---------------------------------------------------------------------------

class TestTrainLightGBM:
    """Test LightGBM training."""

    def _mock_lgbm_regressor(self):
        """Helper to create a properly configured mock LGBMRegressor."""
        mock_model = MagicMock()
        mock_model.predict.side_effect = lambda x: np.random.randn(len(x)) * 0.1
        mock_booster = MagicMock()
        mock_booster.feature_importance.return_value = np.array([100, 80, 60, 40, 20])
        mock_model.booster_ = mock_booster
        return mock_model

    def test_train_with_mock(self, trainer, sample_feature_matrix):
        """Test training with mocked LightGBM."""
        X, y = sample_feature_matrix
        mock_model = self._mock_lgbm_regressor()

        with patch("lightgbm.LGBMRegressor", return_value=mock_model):
            result = trainer.train_lightgbm(X, y)

        assert result is not None
        assert "model" in result
        assert "feature_importance" in result
        assert "ic_train" in result
        assert "ic_valid" in result
        assert result["n_features"] == 5

    def test_time_series_split(self, trainer, sample_feature_matrix):
        """Test chronological train/val split."""
        X, y = sample_feature_matrix
        mock_model = self._mock_lgbm_regressor()

        with patch("lightgbm.LGBMRegressor", return_value=mock_model):
            result = trainer.train_lightgbm(X, y, time_series_split=True)

        assert result is not None
        # 80/20 split
        assert result["n_samples_train"] == 800
        assert result["n_samples_valid"] == 200

    def test_custom_params(self, trainer, sample_feature_matrix):
        """Test training with custom hyperparameters."""
        X, y = sample_feature_matrix
        mock_model = self._mock_lgbm_regressor()

        with patch("lightgbm.LGBMRegressor", return_value=mock_model):
            custom_params = {
                "n_estimators": 100,
                "max_depth": 4,
                "learning_rate": 0.1,
            }
            result = trainer.train_lightgbm(X, y, params=custom_params)

        assert result is not None
        assert result["params"]["n_estimators"] == 100
        assert result["params"]["max_depth"] == 4

    def test_lightgbm_not_installed(self, trainer, sample_feature_matrix):
        """Test graceful degradation when LightGBM is missing."""
        X, y = sample_feature_matrix

        import sys
        orig = sys.modules.get("lightgbm")
        sys.modules["lightgbm"] = None  # type: ignore[assignment]
        try:
            # Force fresh import check in train_lightgbm
            import importlib
            import rdagent.scenarios.qlib.local.ml_trainer as mt
            importlib.reload(mt)
            trainer2 = mt.MLTrainer(models_dir=trainer.models_dir)
            result = trainer2.train_lightgbm(X, y)
            assert result is None
        finally:
            if orig is not None:
                sys.modules["lightgbm"] = orig
            elif "lightgbm" in sys.modules:
                del sys.modules["lightgbm"]

    def test_feature_importance_extraction(self, trainer, sample_feature_matrix):
        """Test feature importance is correctly extracted."""
        X, y = sample_feature_matrix
        mock_model = self._mock_lgbm_regressor()

        with patch("lightgbm.LGBMRegressor", return_value=mock_model):
            result = trainer.train_lightgbm(X, y)

        assert "momentum_5min" in result["feature_importance"]
        assert result["feature_importance"]["momentum_5min"]["gain"] == 100.0
        assert "gain_normalized" in result["feature_importance"]["momentum_5min"]
        assert len(result["feature_ranking"]) == 5
        assert result["feature_ranking"][0] == "momentum_5min"

    def test_metrics_calculated(self, trainer, sample_feature_matrix):
        """Test that all metrics are calculated."""
        X, y = sample_feature_matrix
        mock_model = MagicMock()
        # Predictions correlated with y for valid metrics
        mock_model.predict.side_effect = lambda x: x.iloc[:, 0].values * 0.5
        mock_booster = MagicMock()
        mock_booster.feature_importance.return_value = np.array([100, 100, 100, 100, 100])
        mock_model.booster_ = mock_booster

        with patch("lightgbm.LGBMRegressor", return_value=mock_model):
            result = trainer.train_lightgbm(X, y)

        assert "ic_train" in result
        assert "ic_valid" in result
        assert "rank_ic_train" in result
        assert "rank_ic_valid" in result
        assert "sharpe_valid" in result
        assert "mse_train" in result
        assert "mse_valid" in result


# ---------------------------------------------------------------------------
# Test: extract_feature_importance
# ---------------------------------------------------------------------------

class TestExtractFeatureImportance:
    """Test feature importance extraction."""

    def test_extract_all(self, trainer, mock_model_info):
        """Test extracting all feature importances."""
        df = trainer.extract_feature_importance(mock_model_info)
        assert len(df) == 5
        assert df.iloc[0]["feature"] == "momentum_5min"
        assert "gain" in df.columns
        assert "split" in df.columns

    def test_extract_top_n(self, trainer, mock_model_info):
        """Test extracting only top N features."""
        df = trainer.extract_feature_importance(mock_model_info, top_n=3)
        assert len(df) == 3
        assert df.iloc[0]["feature"] == "momentum_5min"

    def test_empty_importance(self, trainer):
        """Test handling empty importance dict."""
        df = trainer.extract_feature_importance({})
        assert df.empty

    def test_sorted_by_gain(self, trainer, mock_model_info):
        """Test that result is sorted by gain descending."""
        df = trainer.extract_feature_importance(mock_model_info)
        gains = df["gain"].values
        assert all(gains[i] >= gains[i + 1] for i in range(len(gains) - 1))


# ---------------------------------------------------------------------------
# Test: save_model
# ---------------------------------------------------------------------------

class TestSaveModel:
    """Test model persistence."""

    def test_save_model(self, trainer, mock_model_info):
        """Test saving model to directory."""
        mock_model_info["model"].save_model = MagicMock()

        path = trainer.save_model(mock_model_info, model_name="test_model")

        assert path is not None
        assert path.exists()
        assert (path / "model.txt").exists() or mock_model_info["model"].save_model.called
        assert (path / "metadata.json").exists()
        assert (path / "feature_importance.json").exists()

    def test_save_model_metadata(self, trainer, mock_model_info, tmp_path):
        """Test metadata is correctly saved."""
        mock_model_info["model"].save_model = MagicMock()

        path = trainer.save_model(mock_model_info, model_name="test_meta")

        metadata_file = path / "metadata.json"
        with open(metadata_file, encoding="utf-8") as fh:
            metadata = json.load(fh)

        assert metadata["model_type"] == "LightGBM"
        assert metadata["ic_valid"] == 0.10
        assert metadata["n_features"] == 5

    def test_save_model_feature_importance(self, trainer, mock_model_info):
        """Test feature importance JSON is saved."""
        mock_model_info["model"].save_model = MagicMock()

        path = trainer.save_model(mock_model_info, model_name="test_importance")

        imp_file = path / "feature_importance.json"
        with open(imp_file, encoding="utf-8") as fh:
            importance = json.load(fh)

        assert "momentum_5min" in importance
        assert importance["momentum_5min"]["gain"] == 500.0

    def test_save_model_csv(self, trainer, mock_model_info):
        """Test feature importance CSV is saved."""
        mock_model_info["model"].save_model = MagicMock()

        path = trainer.save_model(mock_model_info, model_name="test_csv")

        csv_file = path / "feature_importance.csv"
        assert csv_file.exists()
        df = pd.read_csv(csv_file)
        assert "feature" in df.columns
        assert "gain" in df.columns

    def test_save_model_none_info(self, trainer):
        """Test saving with None model_info."""
        path = trainer.save_model(None)
        assert path is None

    def test_save_model_missing_model(self, trainer):
        """Test saving without model object."""
        path = trainer.save_model({"feature_names": []})
        assert path is None

    def test_save_model_default_name(self, trainer, mock_model_info):
        """Test auto-generated model name."""
        mock_model_info["model"].save_model = MagicMock()

        path = trainer.save_model(mock_model_info)

        assert path is not None
        assert "lgbm_" in path.name


# ---------------------------------------------------------------------------
# Test: load_model
# ---------------------------------------------------------------------------

class TestLoadModel:
    """Test model loading."""

    def test_load_model_by_name(self, trainer, mock_model_info, tmp_path):
        """Test loading model by name."""
        # Train a real model to have a valid model.txt file
        import lightgbm as lgb

        X = pd.DataFrame(np.random.randn(200, 3), columns=["f1", "f2", "f3"])
        y = pd.Series(np.random.randn(200))
        real_model = lgb.LGBMRegressor(n_estimators=10, max_depth=3, verbose=-1)
        real_model.fit(X, y)

        model_dir = trainer.models_dir / "load_test"
        model_dir.mkdir(parents=True, exist_ok=True)
        # Save via booster
        real_model.booster_.save_model(str(model_dir / "model.txt"))

        metadata = {
            "model_type": "LightGBM",
            "ic_valid": 0.1,
            "n_features": 3,
            "feature_names": ["f1", "f2", "f3"],
            "feature_importance": {"f1": {"gain": 100, "split": 5}},
        }
        with open(model_dir / "metadata.json", "w", encoding="utf-8") as fh:
            json.dump(metadata, fh)

        loaded = trainer.load_model(model_name="load_test")

        assert loaded is not None
        assert "model" in loaded
        assert loaded["model_type"] == "LightGBM"

    def test_load_model_nonexistent(self, trainer):
        """Test loading non-existent model."""
        loaded = trainer.load_model(model_name="nonexistent")
        assert loaded is None

    def test_load_model_no_models_dir(self, trainer, tmp_path):
        """Test loading when no models exist."""
        empty_dir = tmp_path / "empty"
        empty_dir.mkdir()
        trainer.models_dir = empty_dir

        loaded = trainer.load_model()
        assert loaded is None


# ---------------------------------------------------------------------------
# Test: generate_feedback
# ---------------------------------------------------------------------------

class TestGenerateFeedback:
    """Test feedback generation for factor generation loop."""

    def test_generate_feedback_success(self, trainer, mock_model_info):
        """Test feedback generation with valid model info."""
        feedback = trainer.generate_feedback(mock_model_info)

        assert feedback["status"] == "success"
        assert "suggestions" in feedback
        assert len(feedback["suggestions"]) > 0
        assert "factor_type_analysis" in feedback
        assert feedback["n_high_importance"] > 0

    def test_feedback_contains_suggestions(self, trainer, mock_model_info):
        """Test that feedback contains actionable suggestions."""
        feedback = trainer.generate_feedback(mock_model_info)

        assert len(feedback["suggestions"]) >= 1
        assert all(isinstance(s, str) for s in feedback["suggestions"])

    def test_feedback_top_5_features(self, trainer, mock_model_info):
        """Test top 5 features are included."""
        feedback = trainer.generate_feedback(mock_model_info)

        assert "top_5_features" in feedback
        assert len(feedback["top_5_features"]) <= 5

    def test_feedback_factor_type_analysis(self, trainer, mock_model_info):
        """Test factor type analysis is included."""
        feedback = trainer.generate_feedback(mock_model_info)

        analysis = feedback["factor_type_analysis"]
        assert "momentum" in analysis
        assert "mean_reversion" in analysis
        assert "volatility" in analysis
        assert "volume" in analysis

    def test_feedback_low_ic_warning(self, trainer, mock_model_info):
        """Test feedback warns about low IC."""
        mock_model_info["ic_valid"] = 0.02  # Below threshold
        feedback = trainer.generate_feedback(mock_model_info)

        assert any("IC is low" in s for s in feedback["suggestions"])

    def test_feedback_empty_importance(self, trainer):
        """Test feedback with no importance data."""
        feedback = trainer.generate_feedback({"feature_importance": {}})

        assert feedback["status"] == "no_importance_data"
        assert feedback["suggestions"] == []

    def test_feedback_min_importance_threshold(self, trainer, mock_model_info):
        """Test min_importance_threshold affects classification."""
        feedback_low = trainer.generate_feedback(
            mock_model_info, min_importance_threshold=0.01
        )
        feedback_high = trainer.generate_feedback(
            mock_model_info, min_importance_threshold=0.50
        )

        assert feedback_low["n_high_importance"] >= feedback_high["n_high_importance"]


# ---------------------------------------------------------------------------
# Test: save_feedback
# ---------------------------------------------------------------------------

class TestSaveFeedback:
    """Test feedback persistence."""

    def test_save_feedback(self, trainer, mock_model_info):
        """Test saving feedback to JSON."""
        feedback = trainer.generate_feedback(mock_model_info)
        path = trainer.save_feedback(feedback)

        assert path.exists()
        assert path.suffix == ".json"

        with open(path, encoding="utf-8") as fh:
            loaded = json.load(fh)

        assert loaded["status"] == "success"

    def test_save_feedback_custom_path(self, trainer, mock_model_info, tmp_path):
        """Test saving feedback to custom path."""
        feedback = trainer.generate_feedback(mock_model_info)
        custom_path = str(tmp_path / "custom_feedback.json")
        path = trainer.save_feedback(feedback, output_path=custom_path)

        assert str(path) == custom_path
        assert path.exists()


# ---------------------------------------------------------------------------
# Test: train_top_factors (full pipeline)
# ---------------------------------------------------------------------------

class TestTrainTopFactors:
    """Test complete training pipeline."""

    @patch.object(MLTrainer, "load_top_factors")
    @patch.object(MLTrainer, "build_feature_matrix")
    @patch.object(MLTrainer, "train_lightgbm")
    @patch.object(MLTrainer, "save_model")
    @patch.object(MLTrainer, "generate_feedback")
    @patch.object(MLTrainer, "save_feedback")
    def test_full_pipeline(
        self,
        mock_save_feedback,
        mock_gen_feedback,
        mock_save_model,
        mock_train,
        mock_build,
        mock_load,
        trainer,
        sample_feature_matrix,
        mock_model_info,
    ):
        """Test complete pipeline with all steps mocked."""
        X, y = sample_feature_matrix
        mock_load.return_value = [{"factor_name": "f1", "ic": 0.1, "status": "success"}]
        mock_build.return_value = (X, y)
        mock_train.return_value = mock_model_info
        mock_save_model.return_value = Path("/fake/path")
        mock_gen_feedback.return_value = {
            "status": "success",
            "suggestions": ["test"],
        }
        mock_save_feedback.return_value = Path("/fake/feedback.json")

        result = trainer.train_top_factors(top_n=10, min_ic=0.05)

        assert result is not None
        assert "feedback" in result
        assert "model_path" in result
        assert "feedback_path" in result

    def test_pipeline_no_factors(self, trainer):
        """Test pipeline when no factors found."""
        with patch.object(MLTrainer, "load_top_factors", return_value=[]):
            result = trainer.train_top_factors()
            assert result is None

    def test_pipeline_no_feature_matrix(self, trainer):
        """Test pipeline when feature matrix build fails."""
        with patch.object(MLTrainer, "load_top_factors", return_value=[{"ic": 0.1}]):
            with patch.object(MLTrainer, "build_feature_matrix", return_value=(None, None)):
                result = trainer.train_top_factors()
                assert result is None

    def test_pipeline_training_fails(self, trainer):
        """Test pipeline when training fails."""
        with patch.object(MLTrainer, "load_top_factors", return_value=[{"ic": 0.1}]):
            with patch.object(
                MLTrainer, "build_feature_matrix", return_value=(pd.DataFrame(), pd.Series())
            ):
                with patch.object(MLTrainer, "train_lightgbm", return_value=None):
                    result = trainer.train_top_factors()
                    assert result is None


# ---------------------------------------------------------------------------
# Test: get_ml_trainer factory
# ---------------------------------------------------------------------------

class TestGetMLTrainer:
    """Test factory function."""

    def test_factory_returns_trainer_when_lgb_available(self):
        """Test factory returns MLTrainer when LightGBM available."""
        try:
            import lightgbm  # noqa: F401
            has_lgb = True
        except ImportError:
            has_lgb = False

        if has_lgb:
            trainer = get_ml_trainer()
            assert trainer is not None
            # Check it has the expected methods
            assert hasattr(trainer, "train_lightgbm")
            assert hasattr(trainer, "load_top_factors")
            assert hasattr(trainer, "generate_feedback")
        else:
            # Skip test if LightGBM not installed
            pytest.skip("LightGBM not installed")

    def test_factory_returns_none_without_lgb(self):
        """Test factory returns None when LightGBM missing."""
        import sys

        orig = sys.modules.get("lightgbm")
        # Temporarily remove lightgbm from modules
        if "lightgbm" in sys.modules:
            del sys.modules["lightgbm"]
        sys.modules["lightgbm"] = None  # type: ignore[assignment]
        try:
            # Force reimport in get_ml_trainer
            result = get_ml_trainer()
            assert result is None
        finally:
            if orig is not None:
                sys.modules["lightgbm"] = orig
            elif "lightgbm" in sys.modules:
                del sys.modules["lightgbm"]
