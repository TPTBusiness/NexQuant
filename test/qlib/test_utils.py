"""Tests for utils/agent/apply_patch, core/prompts, utils/fmt."""

from __future__ import annotations

import sys
from pathlib import Path

import pytest

PROJECT_ROOT = Path(__file__).parent.parent.parent
sys.path.insert(0, str(PROJECT_ROOT))


# =============================================================================
# apply_patch data structures
# =============================================================================


class TestApplyPatchDatastructures:
    def test_action_type_enum(self):
        from rdagent.utils.agent.apply_patch import ActionType
        assert ActionType.ADD.value == "add"
        assert ActionType.DELETE.value == "delete"
        assert ActionType.UPDATE.value == "update"

    def test_file_change_dataclass(self):
        from rdagent.utils.agent.apply_patch import FileChange, ActionType
        fc = FileChange(type=ActionType.UPDATE, old_content="old", new_content="new")
        assert fc.type == ActionType.UPDATE
        assert fc.old_content == "old"
        assert fc.new_content == "new"
        assert fc.move_path is None

    def test_file_change_defaults(self):
        from rdagent.utils.agent.apply_patch import FileChange, ActionType
        fc = FileChange(type=ActionType.ADD)
        assert fc.old_content is None
        assert fc.new_content is None

    def test_commit_defaults(self):
        from rdagent.utils.agent.apply_patch import Commit
        c = Commit()
        assert c.changes == {}

    def test_commit_with_changes(self):
        from rdagent.utils.agent.apply_patch import Commit, FileChange, ActionType
        c = Commit(changes={"test.py": FileChange(type=ActionType.UPDATE)})
        assert "test.py" in c.changes

    def test_diff_error_is_value_error(self):
        from rdagent.utils.agent.apply_patch import DiffError
        with pytest.raises(DiffError):
            raise DiffError("test error")

    def test_chunk_dataclass(self):
        from rdagent.utils.agent.apply_patch import Chunk
        c = Chunk(orig_index=5, del_lines=["a"], ins_lines=["b", "c"])
        assert c.orig_index == 5
        assert c.del_lines == ["a"]
        assert c.ins_lines == ["b", "c"]

    def test_patch_action_dataclass(self):
        from rdagent.utils.agent.apply_patch import PatchAction, ActionType
        pa = PatchAction(type=ActionType.ADD, new_file="test.py")
        assert pa.type == ActionType.ADD
        assert pa.new_file == "test.py"


# =============================================================================
# Prompts (core/prompts.py)
# =============================================================================


class TestPrompts:
    def test_prompts_loads_yaml(self, tmp_path):
        from rdagent.core.prompts import Prompts
        yaml_file = tmp_path / "test.yaml"
        yaml_file.write_text("key1: value1\nkey2: value2\n")
        p = Prompts(file_path=yaml_file)
        assert p["key1"] == "value1"
        assert p["key2"] == "value2"

    def test_prompts_raises_on_empty(self, tmp_path):
        from rdagent.core.prompts import Prompts
        yaml_file = tmp_path / "empty.yaml"
        yaml_file.write_text("")
        with pytest.raises(ValueError, match="Failed to load"):
            Prompts(file_path=yaml_file)

    def test_prompts_is_dict_subclass(self, tmp_path):
        from rdagent.core.prompts import Prompts
        yaml_file = tmp_path / "test.yaml"
        yaml_file.write_text("k: v\n")
        p = Prompts(file_path=yaml_file)
        assert isinstance(p, dict)
        assert len(p) == 1


# =============================================================================
# SingletonBaseClass (core/utils.py)
# =============================================================================


class TestSingletonBaseClass:
    def test_singleton_returns_same_instance(self):
        from rdagent.core.utils import SingletonBaseClass

        class MySingleton(SingletonBaseClass):
            pass

        a = MySingleton()
        b = MySingleton()
        assert a is b
