"""Tests for knowledge graph and task_loader."""

from __future__ import annotations

import sys
from pathlib import Path
from unittest.mock import MagicMock, patch

import pytest

PROJECT_ROOT = Path(__file__).parent.parent.parent
sys.path.insert(0, str(PROJECT_ROOT))


class TestUndirectedNode:
    def test_init(self):
        from rdagent.components.knowledge_management.graph import UndirectedNode
        n = UndirectedNode(content="test", label="component")
        assert n.content == "test"
        assert n.label == "component"
        assert n.id is not None
        assert isinstance(n.neighbors, set)
        assert len(n.neighbors) == 0

    def test_add_neighbor_bidirectional(self):
        from rdagent.components.knowledge_management.graph import UndirectedNode
        a = UndirectedNode(content="a")
        b = UndirectedNode(content="b")
        a.add_neighbor(b)
        assert b in a.neighbors
        assert a in b.neighbors

    def test_remove_neighbor(self):
        from rdagent.components.knowledge_management.graph import UndirectedNode
        a = UndirectedNode(content="a")
        b = UndirectedNode(content="b")
        a.add_neighbor(b)
        a.remove_neighbor(b)
        assert b not in a.neighbors
        assert a not in b.neighbors

    def test_remove_nonexistent_noop(self):
        from rdagent.components.knowledge_management.graph import UndirectedNode
        a = UndirectedNode(content="a")
        b = UndirectedNode(content="b")
        a.remove_neighbor(b)  # should not raise

    def test_get_neighbors(self):
        from rdagent.components.knowledge_management.graph import UndirectedNode
        a = UndirectedNode(content="a")
        b = UndirectedNode(content="b")
        c = UndirectedNode(content="c")
        a.add_neighbor(b)
        a.add_neighbor(c)
        assert a.get_neighbors() == {b, c}

    def test_rejects_non_string_content(self):
        from rdagent.components.knowledge_management.graph import UndirectedNode
        with pytest.raises(TypeError, match="string"):
            UndirectedNode(content=123)

    def test_string_representation(self):
        from rdagent.components.knowledge_management.graph import UndirectedNode
        n = UndirectedNode(content="hello", label="test")
        s = str(n)
        assert "UndirectedNode" in s
        assert "hello" in s
        assert "test" in s


class TestGraphBase:
    def test_init_empty(self):
        from rdagent.components.knowledge_management.graph import Graph
        g = Graph()
        assert g.size() == 0
        assert g.get_all_nodes() == []

    def test_get_node_nonexistent(self):
        from rdagent.components.knowledge_management.graph import Graph
        g = Graph()
        assert g.get_node("nonexistent") is None

    def test_get_all_nodes_by_label_list(self, tmp_path):
        from rdagent.components.knowledge_management.graph import Graph, UndirectedNode
        g = Graph()

        # Add nodes directly to internal dict
        n1 = UndirectedNode(content="a", label="component")
        n2 = UndirectedNode(content="b", label="error")
        n3 = UndirectedNode(content="c", label="component")
        g.nodes[n1.id] = n1
        g.nodes[n2.id] = n2
        g.nodes[n3.id] = n3

        components = g.get_all_nodes_by_label_list(["component"])
        assert len(components) == 2
        labels = [n.label for n in components]
        assert all(l == "component" for l in labels)


class TestVectorBase:
    def test_cosine_distance_identical_is_zero(self):
        from rdagent.components.knowledge_management.vector_base import cosine
        import numpy as np
        dist = cosine(np.array([1.0, 0.0]), np.array([1.0, 0.0]))
        assert dist == pytest.approx(0.0)  # cosine distance, not similarity

    def test_cosine_distance_orthogonal(self):
        from rdagent.components.knowledge_management.vector_base import cosine
        import numpy as np
        dist = cosine(np.array([1.0, 0.0]), np.array([0.0, 1.0]))
        assert dist == pytest.approx(1.0)
