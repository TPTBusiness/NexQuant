"""Tests for remaining untested modules: conf, knowledge_base, interactor, utils/fmt, llm_utils, experiment utils."""

from __future__ import annotations

import sys
from pathlib import Path
from unittest.mock import MagicMock, patch

import pytest

PROJECT_ROOT = Path(__file__).parent.parent.parent
sys.path.insert(0, str(PROJECT_ROOT))


# =============================================================================
# RDAgentSettings (core/conf.py)
# =============================================================================


class TestRDAgentSettings:
    def test_defaults(self):
        from rdagent.core.conf import RDAgentSettings
        s = RDAgentSettings()
        assert s.multi_proc_n == 1
        assert s.cache_with_pickle is True
        assert s.use_file_lock is True
        assert s.enable_mlflow is False
        assert s.step_semaphore == 1
        assert s.subproc_step is False

    def test_get_max_parallel_int(self):
        from rdagent.core.conf import RDAgentSettings
        s = RDAgentSettings()
        s.step_semaphore = 5
        assert s.get_max_parallel() == 5

    def test_get_max_parallel_dict(self):
        from rdagent.core.conf import RDAgentSettings
        s = RDAgentSettings()
        s.step_semaphore = {"coding": 3, "running": 2}
        assert s.get_max_parallel() == 3

    def test_is_force_subproc_subproc_step(self):
        from rdagent.core.conf import RDAgentSettings
        s = RDAgentSettings()
        s.subproc_step = True
        assert s.is_force_subproc() is True

    def test_is_force_subproc_parallel(self):
        from rdagent.core.conf import RDAgentSettings
        s = RDAgentSettings()
        s.subproc_step = False
        s.step_semaphore = 4
        assert s.is_force_subproc() is True

    def test_singleton(self):
        from rdagent.core.conf import RD_AGENT_SETTINGS, RDAgentSettings
        assert isinstance(RD_AGENT_SETTINGS, RDAgentSettings)

    def test_workspace_path_is_path(self):
        from rdagent.core.conf import RDAgentSettings
        s = RDAgentSettings()
        assert isinstance(s.workspace_path, Path)

    def test_env_prefix(self, monkeypatch):
        from rdagent.core.conf import RDAgentSettings
        monkeypatch.setenv("multi_proc_n", "4")
        s = RDAgentSettings()
        assert s.multi_proc_n == 4


# =============================================================================
# KnowledgeBase (core/knowledge_base.py)
# =============================================================================


class TestKnowledgeBase:
    def test_init_without_path(self):
        from rdagent.core.knowledge_base import KnowledgeBase
        kb = KnowledgeBase()
        assert kb.path is None

    def test_init_with_path(self, tmp_path):
        from rdagent.core.knowledge_base import KnowledgeBase
        p = tmp_path / "kb.pkl"
        kb = KnowledgeBase(path=p)
        assert kb.path == p

    def test_dump_creates_file(self, tmp_path):
        from rdagent.core.knowledge_base import KnowledgeBase
        p = tmp_path / "kb.pkl"
        kb = KnowledgeBase(path=p)
        kb.foo = "bar"
        kb.dump()
        assert p.exists()

    def test_load_restores_state(self, tmp_path):
        from rdagent.core.knowledge_base import KnowledgeBase
        p = tmp_path / "kb.pkl"
        kb1 = KnowledgeBase(path=p)
        kb1.foo = "hello"
        kb1.dump()
        kb2 = KnowledgeBase(path=p)
        assert kb2.foo == "hello"


# =============================================================================
# Interactor (core/interactor.py)
# =============================================================================


class TestInteractor:
    def test_abstract_class(self):
        from rdagent.core.interactor import Interactor

        class MyInteractor(Interactor):
            def interact(self, exp, trace=None):
                return exp

        i = MyInteractor(scen=MagicMock())
        assert i.scen is not None


# =============================================================================
# shrink_text (utils/fmt.py)
# =============================================================================


class TestShrinkText:
    def test_short_text_unchanged(self):
        from rdagent.utils.fmt import shrink_text
        result = shrink_text("hello", context_lines=10, line_len=100)
        assert result == "hello"

    def test_long_lines_shrunk(self):
        from rdagent.utils.fmt import shrink_text
        result = shrink_text("a" * 100, context_lines=10, line_len=20)
        assert "chars are hidden" in result

    def test_many_lines_shrunk(self):
        from rdagent.utils.fmt import shrink_text
        text = "\n".join([f"line{i}" for i in range(100)])
        result = shrink_text(text, context_lines=10, line_len=1000)
        assert "lines are hidden" in result

    def test_row_shrink_disabled(self):
        from rdagent.utils.fmt import shrink_text
        text = "\n".join([f"line{i}" for i in range(100)])
        result = shrink_text(text, context_lines=10, line_len=1000, row_shrink=False)
        assert "lines are hidden" not in result


# =============================================================================
# md5_hash (utils/__init__.py)
# =============================================================================


class TestMD5Hash:
    def test_returns_string(self):
        from rdagent.utils import md5_hash
        h = md5_hash("hello")
        assert isinstance(h, str)
        assert len(h) == 64  # actually SHA-256, name is historical

    def test_deterministic(self):
        from rdagent.utils import md5_hash
        assert md5_hash("hello") == md5_hash("hello")

    def test_different_inputs(self):
        from rdagent.utils import md5_hash
        assert md5_hash("a") != md5_hash("b")


# =============================================================================
# convert2bool (utils/__init__.py)
# =============================================================================


class TestConvert2Bool:
    def test_true_strings(self):
        from rdagent.utils import convert2bool
        for v in ("yes", "true", "True", "YES", "ok"):
            assert convert2bool(v) is True

    def test_false_strings(self):
        from rdagent.utils import convert2bool
        for v in ("no", "false", "False", "NO"):
            assert convert2bool(v) is False

    def test_boolean_passthrough(self):
        from rdagent.utils import convert2bool
        assert convert2bool(True) is True
        assert convert2bool(False) is False

    def test_invalid_raises(self):
        from rdagent.utils import convert2bool
        with pytest.raises(ValueError):
            convert2bool("maybe")


# =============================================================================
# calculate_embedding_distance (llm_utils.py)
# =============================================================================


class TestEmbeddingDistance:
    def test_empty_lists_returns_empty(self):
        from rdagent.oai.llm_utils import calculate_embedding_distance_between_str_list
        result = calculate_embedding_distance_between_str_list([], [])
        assert result == [[]]

    def test_one_empty_list(self):
        from rdagent.oai.llm_utils import calculate_embedding_distance_between_str_list
        result = calculate_embedding_distance_between_str_list(["a"], [])
        assert result == [[]]


# =============================================================================
# get_file_desc (experiment/utils.py) — pure logic
# =============================================================================


class TestGetFileDesc:
    def test_md_file(self, tmp_path):
        from rdagent.scenarios.qlib.experiment.utils import get_file_desc
        md_file = tmp_path / "test.md"
        md_file.write_text("# Hello\ncontent")
        desc = get_file_desc(md_file)
        assert "Markdown Documentation" in desc
        assert "Hello" in desc

    def test_unsupported_extension_raises(self, tmp_path):
        from rdagent.scenarios.qlib.experiment.utils import get_file_desc
        f = tmp_path / "test.txt"
        f.write_text("hello")
        with pytest.raises(NotImplementedError):
            get_file_desc(f)


# =============================================================================
# workflow/conf.py
# =============================================================================


class TestBasePropSetting:
    def test_importable(self):
        from rdagent.components.workflow.conf import BasePropSetting
        assert BasePropSetting is not None

    def test_has_evolving_n(self):
        from rdagent.components.workflow.conf import BasePropSetting
        s = BasePropSetting()
        assert hasattr(s, "evolving_n")
