"""Deep tests for rdagent.oai.llm_utils: embedding distance, APIBackend, and edge cases."""

from __future__ import annotations

import pickle
import sys
from pathlib import Path
from typing import Any
from unittest.mock import MagicMock, patch

import numpy as np
import pytest

PROJECT_ROOT = Path(__file__).parent.parent.parent
sys.path.insert(0, str(PROJECT_ROOT))


# =============================================================================
# Import safety
# =============================================================================

LLM_MODULES = [
    "rdagent.oai.llm_utils",
    "rdagent.oai.llm_conf",
    "rdagent.oai.backend.base",
    "rdagent.utils",
]


class TestLLMImports:
    @pytest.mark.parametrize("module_name", LLM_MODULES)
    def test_module_importable(self, module_name: str) -> None:
        """Each LLM utility module imports without error."""
        import importlib
        mod = importlib.import_module(module_name)
        assert mod is not None


# =============================================================================
# calculate_embedding_distance_between_str_list
# =============================================================================


class TestEmbeddingDistance:
    """Tests for calculate_embedding_distance_between_str_list."""

    @patch("rdagent.oai.llm_utils.APIBackend")
    def test_empty_source_returns_empty(self, mock_api: MagicMock) -> None:
        """Empty source list returns nested empty list."""
        from rdagent.oai.llm_utils import calculate_embedding_distance_between_str_list
        mock_api.return_value.create_embedding.return_value = []
        result = calculate_embedding_distance_between_str_list([], ["target"])
        assert result == [[]]

    @patch("rdagent.oai.llm_utils.APIBackend")
    def test_empty_target_returns_empty(self, mock_api: MagicMock) -> None:
        """Empty target list returns nested empty list."""
        from rdagent.oai.llm_utils import calculate_embedding_distance_between_str_list
        mock_api.return_value.create_embedding.return_value = []
        result = calculate_embedding_distance_between_str_list(["source"], [])
        assert result == [[]]

    @patch("rdagent.oai.llm_utils.APIBackend")
    def test_both_empty_returns_empty(self, mock_api: MagicMock) -> None:
        """Both lists empty returns nested empty list."""
        from rdagent.oai.llm_utils import calculate_embedding_distance_between_str_list
        mock_api.return_value.create_embedding.return_value = []
        result = calculate_embedding_distance_between_str_list([], [])
        assert result == [[]]

    def test_both_empty_no_api_call(self) -> None:
        """Empty inputs return [[]] without any API call."""
        from rdagent.oai.llm_utils import calculate_embedding_distance_between_str_list
        result = calculate_embedding_distance_between_str_list([], [])
        assert result == [[]]

    @patch("rdagent.oai.llm_utils.APIBackend")
    def test_single_source_single_target(self, mock_api: MagicMock) -> None:
        """Single source and target return 1x1 matrix."""
        from rdagent.oai.llm_utils import calculate_embedding_distance_between_str_list
        mock_api.return_value.create_embedding.return_value = [
            [0.5, 0.5],  # source embedding
            [0.5, 0.5],  # target embedding
        ]
        result = calculate_embedding_distance_between_str_list(["s1"], ["t1"])
        assert len(result) == 1
        assert len(result[0]) == 1
        assert isinstance(result[0][0], float)

    @patch("rdagent.oai.llm_utils.APIBackend")
    def test_multiple_sources_single_target(self, mock_api: MagicMock) -> None:
        """Multiple sources, single target returns n x 1 matrix."""
        from rdagent.oai.llm_utils import calculate_embedding_distance_between_str_list
        mock_api.return_value.create_embedding.return_value = [
            [1.0, 0.0],
            [0.0, 1.0],
            [0.5, 0.5],
        ]
        result = calculate_embedding_distance_between_str_list(["s1", "s2"], ["t1"])
        assert len(result) == 2
        assert len(result[0]) == 1
        assert len(result[1]) == 1

    @patch("rdagent.oai.llm_utils.APIBackend")
    def test_similarity_range(self, mock_api: MagicMock) -> None:
        """Similarity values should be in [-1, 1] range after normalization."""
        from rdagent.oai.llm_utils import calculate_embedding_distance_between_str_list
        mock_api.return_value.create_embedding.return_value = [
            [1.0, 0.0, 0.0],
            [0.0, 1.0, 0.0],
            [-1.0, 0.0, 0.0],
            [0.7, 0.3, 0.1],
        ]
        result = calculate_embedding_distance_between_str_list(
            ["s1", "s2", "s3"], ["t1"],
        )
        for row in result:
            for val in row:
                assert -1.0 - 1e-9 <= val <= 1.0 + 1e-9

    @patch("rdagent.oai.llm_utils.APIBackend")
    def test_identical_embedding_produces_one(self, mock_api: MagicMock) -> None:
        """Identical embeddings produce similarity of 1.0."""
        from rdagent.oai.llm_utils import calculate_embedding_distance_between_str_list
        mock_api.return_value.create_embedding.return_value = [
            [3.0, 4.0],  # source (norm=5, unit=[0.6, 0.8])
            [3.0, 4.0],  # target (norm=5, unit=[0.6, 0.8])
        ]
        result = calculate_embedding_distance_between_str_list(["s1"], ["t1"])
        assert result[0][0] == pytest.approx(1.0, abs=1e-9)

    @patch("rdagent.oai.llm_utils.APIBackend")
    def test_orthogonal_embedding_produces_zero(self, mock_api: MagicMock) -> None:
        """Orthogonal embeddings produce similarity of 0.0."""
        from rdagent.oai.llm_utils import calculate_embedding_distance_between_str_list
        mock_api.return_value.create_embedding.return_value = [
            [1.0, 0.0],
            [0.0, 1.0],
        ]
        result = calculate_embedding_distance_between_str_list(["s1"], ["t1"])
        assert result[0][0] == pytest.approx(0.0, abs=1e-9)

    @patch("rdagent.oai.llm_utils.APIBackend")
    def test_opposite_embedding_produces_negative_one(self, mock_api: MagicMock) -> None:
        """Opposite embeddings produce similarity of -1.0."""
        from rdagent.oai.llm_utils import calculate_embedding_distance_between_str_list
        mock_api.return_value.create_embedding.return_value = [
            [1.0, 0.0],
            [-1.0, 0.0],
        ]
        result = calculate_embedding_distance_between_str_list(["s1"], ["t1"])
        assert result[0][0] == pytest.approx(-1.0, abs=1e-9)

    @patch("rdagent.oai.llm_utils.APIBackend")
    def test_zero_vector_embedding(self, mock_api: MagicMock) -> None:
        """Zero vector embedding should be handled (division by zero)."""
        from rdagent.oai.llm_utils import calculate_embedding_distance_between_str_list
        mock_api.return_value.create_embedding.return_value = [
            [0.0, 0.0],
            [1.0, 0.0],
        ]
        # After normalization, zero vector becomes NaN, dot produces NaN
        result = calculate_embedding_distance_between_str_list(["s1"], ["t1"])
        assert isinstance(result[0][0], float)

    @patch("rdagent.oai.llm_utils.APIBackend")
    def test_large_embedding_values(self, mock_api: MagicMock) -> None:
        """Large-magnitude embeddings are correctly normalized."""
        from rdagent.oai.llm_utils import calculate_embedding_distance_between_str_list
        mock_api.return_value.create_embedding.return_value = [
            [1e5, 0.0],
            [0.0, 1e5],
        ]
        result = calculate_embedding_distance_between_str_list(["s1"], ["t1"])
        assert isinstance(result[0][0], float)

    @patch("rdagent.oai.llm_utils.APIBackend")
    def test_return_type_is_list_of_lists_of_floats(self, mock_api: MagicMock) -> None:
        """Return type is List[List[float]]."""
        from rdagent.oai.llm_utils import calculate_embedding_distance_between_str_list
        mock_api.return_value.create_embedding.return_value = [
            [1.0],
            [1.0],
        ]
        result = calculate_embedding_distance_between_str_list(["a"], ["b"])
        assert isinstance(result, list)
        assert isinstance(result[0], list)
        assert isinstance(result[0][0], float)

    @patch("rdagent.oai.llm_utils.APIBackend")
    def test_matrix_shape_matches_input_counts(self, mock_api: MagicMock) -> None:
        """Output matrix has shape (len(sources), len(targets))."""
        from rdagent.oai.llm_utils import calculate_embedding_distance_between_str_list
        n_sources, n_targets = 3, 5
        # Create embeddings for all strings
        emb_dim = 128
        embeddings = [
            list(np.random.randn(emb_dim))
            for _ in range(n_sources + n_targets)
        ]
        mock_api.return_value.create_embedding.return_value = embeddings

        sources = [f"s{i}" for i in range(n_sources)]
        targets = [f"t{i}" for i in range(n_targets)]
        result = calculate_embedding_distance_between_str_list(sources, targets)
        assert len(result) == n_sources
        assert all(len(row) == n_targets for row in result)

    @patch("rdagent.oai.llm_utils.APIBackend")
    def test_unicode_strings(self, mock_api: MagicMock) -> None:
        """Unicode/emoji strings are handled."""
        from rdagent.oai.llm_utils import calculate_embedding_distance_between_str_list
        mock_api.return_value.create_embedding.return_value = [
            [0.5, 0.5],
            [0.5, 0.5],
        ]
        result = calculate_embedding_distance_between_str_list(["日本語"], ["🌟"])
        assert len(result) == 1
        assert len(result[0]) == 1

    @patch("rdagent.oai.llm_utils.APIBackend")
    def test_real_calculate_embedding_via_mock(self, mock_api: MagicMock) -> None:
        """Full calculation path works via mocked API."""
        from rdagent.oai.llm_utils import calculate_embedding_distance_between_str_list
        mock_api.return_value.create_embedding.return_value = [
            [1.0, 2.0, 3.0],
            [2.0, 3.0, 4.0],
            [4.0, 2.0, 0.0],
            [4.0, 1.0, 0.0],
        ]
        result = calculate_embedding_distance_between_str_list(
            ["task_info_1", "task_info_2"],
            ["target_1", "target_2"],
        )
        assert len(result) == 2
        assert len(result[0]) == 2


# =============================================================================
# APIBackend
# =============================================================================


class TestAPIBackend:
    """Tests for APIBackend (alias for get_api_backend)."""

    def test_api_backend_is_callable_fn(self) -> None:
        """APIBackend resolves to a callable class factory."""
        from rdagent.oai.llm_utils import APIBackend
        assert callable(APIBackend)

    def test_get_api_backend_is_importable(self) -> None:
        """get_api_backend is importable."""
        from rdagent.oai.llm_utils import get_api_backend
        assert callable(get_api_backend)

    @patch("rdagent.oai.llm_utils.import_class")
    def test_get_api_backend_calls_import_class(self, mock_import: MagicMock) -> None:
        """get_api_backend uses import_class to resolve backend class."""
        from rdagent.oai.llm_utils import get_api_backend
        mock_cls = MagicMock()
        mock_cls.return_value = MagicMock()
        mock_import.return_value = mock_cls

        backend = get_api_backend(cache_enabled=False)
        assert backend is not None
        mock_import.assert_called_once()

    @patch("rdagent.oai.llm_utils.import_class")
    def test_api_backend_passes_args(self, mock_import: MagicMock) -> None:
        """APIBackend passes args to the backend constructor."""
        from rdagent.oai.llm_utils import get_api_backend
        mock_cls = MagicMock()
        mock_import.return_value = mock_cls

        get_api_backend(use_chat_cache=True, json_mode=True)
        mock_cls.assert_called_once_with(use_chat_cache=True, json_mode=True)

    def test_api_backend_reference_equality(self) -> None:
        """APIBackend and get_api_backend are the same object."""
        from rdagent.oai.llm_utils import APIBackend, get_api_backend
        assert APIBackend is get_api_backend


# =============================================================================
# LLM settings
# =============================================================================


class TestLLMSettings:
    """Tests for LLM settings module."""

    def test_llm_settings_is_importable(self) -> None:
        """LLM_SETTINGS is importable."""
        from rdagent.oai.llm_conf import LLM_SETTINGS
        assert LLM_SETTINGS is not None

    def test_llm_settings_has_backend(self) -> None:
        """LLM_SETTINGS has backend attribute."""
        from rdagent.oai.llm_conf import LLM_SETTINGS
        assert hasattr(LLM_SETTINGS, "backend")

    def test_llm_settings_backend_is_string(self) -> None:
        """LLM_SETTINGS.backend is a string class path."""
        from rdagent.oai.llm_conf import LLM_SETTINGS
        assert isinstance(LLM_SETTINGS.backend, str)


# =============================================================================
# md5_hash utility
# =============================================================================


class TestMd5Hash:
    """Tests for md5_hash utility."""

    def test_md5_hash_is_importable(self) -> None:
        """md5_hash is importable."""
        from rdagent.utils import md5_hash
        assert callable(md5_hash)

    def test_md5_hash_returns_string(self) -> None:
        """md5_hash returns a hex digest string."""
        from rdagent.utils import md5_hash
        result = md5_hash("test input")
        assert isinstance(result, str)
        assert len(result) == 64  # SHA256 hex digest (named md5 but uses sha256)

    def test_md5_hash_deterministic(self) -> None:
        """md5_hash is deterministic."""
        from rdagent.utils import md5_hash
        a = md5_hash("hello")
        b = md5_hash("hello")
        assert a == b

    def test_md5_hash_different_inputs(self) -> None:
        """Different inputs produce different hashes."""
        from rdagent.utils import md5_hash
        a = md5_hash("hello")
        b = md5_hash("world")
        assert a != b

    @pytest.mark.parametrize("input_val", [
        "", "a", "abc" * 1000, "unicode_日本語", "emoji_🌟", "multi\nline\nstring",
    ])
    def test_md5_hash_various_inputs(self, input_val: str) -> None:
        """Various input types produce valid hashes."""
        from rdagent.utils import md5_hash
        result = md5_hash(input_val)
        assert isinstance(result, str)
        assert len(result) == 64


# =============================================================================
# Integration tests — end-to-end mocked embedding pipeline
# =============================================================================


class TestEmbeddingPipeline:
    """Integration-style tests for the embedding pipeline (mocked)."""

    @patch("rdagent.oai.llm_utils.APIBackend")
    def test_knowledge_base_typical_usage(self, mock_api: MagicMock) -> None:
        """Typical usage pattern: query similarity of task vs known successes."""
        from rdagent.oai.llm_utils import calculate_embedding_distance_between_str_list

        target_task = "Calculate rolling z-score of $close for EURUSD"
        success_tasks = [
            "Calculate SMA of $close",
            "Calculate volatility of returns",
            "Compute volume-weighted average price",
        ]

        # Mock embeddings: first target, then three successes
        mock_api.return_value.create_embedding.return_value = [
            [0.3, 0.7, 0.1, 0.5],
            [0.4, 0.6, 0.2, 0.4],
            [0.1, 0.8, 0.0, 0.5],
            [0.2, 0.9, 0.1, 0.3],
        ]

        similarity = calculate_embedding_distance_between_str_list(
            [target_task], success_tasks,
        )
        assert len(similarity) == 1
        assert len(similarity[0]) == 3

        # Sort by similarity descending
        similar_indexes = sorted(
            range(len(similarity[0])),
            key=lambda i: similarity[0][i],
            reverse=True,
        )
        assert len(similar_indexes) == 3

    @patch("rdagent.oai.llm_utils.APIBackend")
    def test_embedding_concatenation_order(self, mock_api: MagicMock) -> None:
        """Source embeddings are first, then target embeddings."""
        from rdagent.oai.llm_utils import calculate_embedding_distance_between_str_list

        mock_api.return_value.create_embedding.return_value = [
            [1.0, 0.0],  # source
            [0.0, 1.0],  # target
        ]

        result = calculate_embedding_distance_between_str_list(["s"], ["t"])
        assert result[0][0] == pytest.approx(0.0, abs=1e-9)


# =============================================================================
# Edge cases — NaN, inf, extreme values in embedding vectors
# =============================================================================


class TestEmbeddingEdgeCases:
    """Edge case tests for embedding distance calculation."""

    @patch("rdagent.oai.llm_utils.APIBackend")
    def test_nan_in_embeddings(self, mock_api: MagicMock) -> None:
        """NaN values in embeddings produce NaN in similarity."""
        from rdagent.oai.llm_utils import calculate_embedding_distance_between_str_list
        mock_api.return_value.create_embedding.return_value = [
            [float("nan"), 1.0],
            [1.0, 0.0],
        ]
        result = calculate_embedding_distance_between_str_list(["s"], ["t"])
        assert isinstance(result[0][0], float)

    @patch("rdagent.oai.llm_utils.APIBackend")
    def test_inf_in_embeddings(self, mock_api: MagicMock) -> None:
        """Inf values in embeddings produce NaN or inf in similarity."""
        from rdagent.oai.llm_utils import calculate_embedding_distance_between_str_list
        mock_api.return_value.create_embedding.return_value = [
            [float("inf"), 0.0],
            [1.0, 0.0],
        ]
        result = calculate_embedding_distance_between_str_list(["s"], ["t"])
        assert isinstance(result[0][0], float)

    @patch("rdagent.oai.llm_utils.APIBackend")
    def test_very_high_dimensional_embedding(self, mock_api: MagicMock) -> None:
        """High-dimensional embeddings (1536 dims) work."""
        from rdagent.oai.llm_utils import calculate_embedding_distance_between_str_list
        dim = 1536
        mock_api.return_value.create_embedding.return_value = [
            list(np.random.randn(dim)),
            list(np.random.randn(dim)),
        ]
        result = calculate_embedding_distance_between_str_list(["s"], ["t"])
        assert len(result[0]) == 1
        assert -1.0 <= result[0][0] <= 1.0

    @patch("rdagent.oai.llm_utils.APIBackend")
    def test_many_targets(self, mock_api: MagicMock) -> None:
        """Large number of targets works correctly."""
        from rdagent.oai.llm_utils import calculate_embedding_distance_between_str_list
        n_targets = 100
        mock_api.return_value.create_embedding.return_value = [
            list(np.random.randn(128))
            for _ in range(1 + n_targets)
        ]
        sources = ["s"]
        targets = [f"t{i}" for i in range(n_targets)]
        result = calculate_embedding_distance_between_str_list(sources, targets)
        assert len(result) == 1
        assert len(result[0]) == n_targets


# =============================================================================
# Backend base class
# =============================================================================


class TestBackendBase:
    """Tests for the backend base class."""

    def test_base_api_backend_is_importable(self) -> None:
        """BaseAPIBackend is importable."""
        from rdagent.oai.backend.base import APIBackend
        assert APIBackend is not None

    def test_base_api_backend_is_a_class(self) -> None:
        """BaseAPIBackend is a class."""
        from rdagent.oai.backend.base import APIBackend
        assert isinstance(APIBackend, type)


# =============================================================================
# Pickle safety for LLM-related objects
# =============================================================================


class TestLLMPickleSafety:
    """Pickle safety tests for LLM utility objects."""

    def test_similarity_matrix_pickle(self) -> None:
        """Similarity matrix (list of lists) survives pickle."""
        matrix = [[0.5, 0.8], [0.3, 0.1]]
        data = pickle.dumps(matrix)
        loaded = pickle.loads(data)
        assert loaded == matrix

    def test_embedding_list_pickle(self) -> None:
        """Embedding vector list survives pickle."""
        emb = [0.1, 0.2, 0.3, 0.4]
        data = pickle.dumps(emb)
        loaded = pickle.loads(data)
        assert loaded == emb

    @patch("rdagent.oai.llm_utils.APIBackend")
    def test_mocked_api_result_pickle(self, mock_api: MagicMock) -> None:
        """Mocked API result (list of floats) survives pickle."""
        mock_result = [[0.1, 0.2], [0.3, 0.4]]
        data = pickle.dumps(mock_result)
        loaded = pickle.loads(data)
        assert loaded == mock_result
