"""Tests for final untested modules: runtime_info, repo_utils, json_loader, cli_welcome."""

from __future__ import annotations

import json
import sys
import tempfile
from pathlib import Path
from unittest.mock import patch, MagicMock

import pytest

PROJECT_ROOT = Path(__file__).parent.parent.parent
sys.path.insert(0, str(PROJECT_ROOT))


class TestRuntimeInfo:
    def test_get_runtime_info_has_keys(self):
        from rdagent.scenarios.shared.runtime_info import get_runtime_info
        info = get_runtime_info()
        assert "python_version" in info
        assert "os" in info
        assert isinstance(info["python_version"], str)

    def test_get_gpu_info_returns_dict(self):
        from rdagent.scenarios.shared.runtime_info import get_gpu_info
        info = get_gpu_info()
        assert isinstance(info, dict)
        assert "source" in info

    def test_get_gpu_info_no_pytorch_fallback(self):
        with patch("rdagent.scenarios.shared.runtime_info.torch", None, create=True):
            with patch("subprocess.run", side_effect=FileNotFoundError):
                from importlib import reload
                import rdagent.scenarios.shared.runtime_info as ri
                reload(ri)
                info = ri.get_gpu_info()
                assert info["source"] in ("nvidia-smi", "pytorch")


class TestRepoAnalyzer:
    def test_repo_analyzer_init(self, tmp_path):
        from rdagent.utils.repo.repo_utils import RepoAnalyzer
        (tmp_path / "test.py").write_text("def foo(): pass\n")
        ra = RepoAnalyzer(str(tmp_path))
        assert ra.repo_path == tmp_path

    def test_summarize_repo(self, tmp_path):
        from rdagent.utils.repo.repo_utils import RepoAnalyzer
        (tmp_path / "test.py").write_text("def foo(x: int) -> int:\n    '''Return x.'''\n    return x\n")
        ra = RepoAnalyzer(str(tmp_path))
        summary = ra.summarize_repo(verbose_level=1, doc_str_level=1, sign_level=1)
        assert "Workspace Summary" in summary
        assert "test.py" in summary
        assert "foo" in summary

    def test_summarize_with_class(self, tmp_path):
        from rdagent.utils.repo.repo_utils import RepoAnalyzer
        (tmp_path / "test.py").write_text("class A:\n    '''Class doc.'''\n    def m(self): pass\n")
        ra = RepoAnalyzer(str(tmp_path))
        summary = ra.summarize_repo(verbose_level=2, doc_str_level=1, sign_level=1)
        assert "Class: A" in summary

    def test_highlight(self, tmp_path):
        from rdagent.utils.repo.repo_utils import RepoAnalyzer
        (tmp_path / "test.py").write_text("x = 1\n")
        ra = RepoAnalyzer(str(tmp_path))
        result = ra.highlight("test.py")
        assert "x = 1" in result["test.py"]

    def test_tree_structure(self, tmp_path):
        from rdagent.utils.repo.repo_utils import RepoAnalyzer
        sub = tmp_path / "sub"
        sub.mkdir()
        (sub / "mod.py").write_text("pass\n")
        ra = RepoAnalyzer(str(tmp_path))
        tree = ra._generate_tree_structure()
        assert "sub/" in tree
        assert "mod.py" in tree


class TestJsonLoader:
    def test_load_from_dict(self):
        from rdagent.scenarios.qlib.factor_experiment_loader.json_loader import FactorExperimentLoaderFromDict
        loader = FactorExperimentLoaderFromDict()
        factor_dict = {
            "f1": {"description": "desc1", "formulation": "form1", "variables": {}},
            "f2": {"description": "desc2", "formulation": "form2", "variables": {"x": 1}},
        }
        exp = loader.load(factor_dict)
        assert len(exp.sub_tasks) == 2
        assert exp.sub_tasks[0].factor_name == "f1"

    def test_load_from_json_string(self):
        from rdagent.scenarios.qlib.factor_experiment_loader.json_loader import FactorExperimentLoaderFromJsonString
        loader = FactorExperimentLoaderFromJsonString()
        json_str = json.dumps({
            "f1": {"description": "d", "formulation": "f", "variables": {}},
        })
        exp = loader.load(json_str)
        assert len(exp.sub_tasks) == 1

    def test_load_from_json_file(self, tmp_path):
        from rdagent.scenarios.qlib.factor_experiment_loader.json_loader import FactorExperimentLoaderFromJsonFile
        json_file = tmp_path / "factors.json"
        json_file.write_text(json.dumps({
            "f1": {"description": "d", "formulation": "f", "variables": {}},
        }))
        loader = FactorExperimentLoaderFromJsonFile()
        exp = loader.load(json_file)
        assert len(exp.sub_tasks) == 1


class TestCLIWelcome:
    def test_cli_welcome_importable(self):
        from rdagent.app import cli_welcome
        assert cli_welcome is not None
    def test_cli_welcome_importable(self):
        from rdagent.app import cli_welcome
        assert cli_welcome is not None

class TestGetRuntimeInfoShared:
    def test_module_importable(self):
        from rdagent.scenarios.shared import get_runtime_info
        assert get_runtime_info is not None


class TestGeneralModel:
    def test_module_importable(self):
        from rdagent.app.general_model import general_model
        assert general_model is not None
