import tempfile
import unittest
from pathlib import Path

import pytest

from rdagent.core.utils import SingletonBaseClass, import_class, safe_resolve_path


class A(SingletonBaseClass):
    def __init__(self, **kwargs):
        print(self, "__init__", kwargs)  # make sure the __init__ is called only once.
        self.kwargs = kwargs

    def __str__(self) -> str:
        return f"{self.__class__.__name__}.{getattr(self, 'kwargs', None)}"

    def __repr__(self) -> str:
        return self.__str__()


@pytest.mark.offline
class MiscTest(unittest.TestCase):
    def test_singleton(self):
        print("a1=================")
        a1 = A()
        print("a2=================")
        a2 = A()
        print("a3=================")
        a3 = A(x=3)
        print("a4=================")
        a4 = A(x=2)
        print("a5=================")
        a5 = A(b=3)
        print("a6=================")
        a6 = A(x=3)

        # Check that a1 and a2 are the same instance
        self.assertIs(a1, a2)

        # Check that a3 and a6 are the same instance
        self.assertIs(a3, a6)

        # Check that a1 and a3 are different instances
        self.assertIsNot(a1, a3)

        # Check that a3 and a4 are different instances
        self.assertIsNot(a3, a4)

        # Check that a4 and a5 are different instances
        self.assertIsNot(a4, a5)

        # Check that a5 and a6 are different instances
        self.assertIsNot(a5, a6)

        print(id(a1), id(a2), id(a3), id(a4), id(a5), id(a6))

        print("...................... Start testing pickle ......................")

        # Test pickle
        import pickle

        with self.assertRaises(pickle.PicklingError):
            with open("a3.pkl", "wb") as f:
                pickle.dump(a3, f)
        # NOTE: If the pickle feature is not disabled,
        # loading a3.pkl will return a1, and a1 will be updated with a3's attributes.
        # print(a1.kwargs)
        # with open("a3.pkl", "rb") as f:
        #     a3_pkl = pickle.load(f)
        # print(id(a3), id(a3_pkl))  # not the same object
        # print(a1.kwargs)  # a1 will be changed.


class TestSafeResolvePath:
    """Tests for safe_resolve_path — path traversal prevention."""

    def test_inside_root_returns_absolute(self):
        with tempfile.TemporaryDirectory() as tmpdir:
            root = Path(tmpdir)
            result = safe_resolve_path(root / "subdir" / "file.txt", safe_root=root)
            assert result.is_absolute()
            assert str(result).startswith(str(root.resolve()))

    def test_no_safe_root_just_resolves(self):
        result = safe_resolve_path(Path("/tmp/nonexistent_test"), safe_root=None)
        assert result.is_absolute()

    def test_path_traversal_raises(self):
        with tempfile.TemporaryDirectory() as tmpdir:
            root = Path(tmpdir)
            with pytest.raises(ValueError, match="outside allowed root"):
                safe_resolve_path(root / ".." / "etc" / "passwd", safe_root=root)

    def test_symlink_outside_root_raises(self):
        with tempfile.TemporaryDirectory() as tmpdir:
            root = Path(tmpdir)
            inside = root / "inside"
            inside.mkdir()
            link = inside / "escape"
            link.symlink_to("/etc/passwd")
            with pytest.raises(ValueError, match="outside allowed root"):
                safe_resolve_path(link, safe_root=root)

    def test_root_itself_is_valid(self):
        with tempfile.TemporaryDirectory() as tmpdir:
            root = Path(tmpdir)
            result = safe_resolve_path(root, safe_root=root)
            assert result == root.resolve()

    def test_expanduser_resolves_home(self):
        result = safe_resolve_path(Path("~/nonexistent_test"), safe_root=None)
        assert str(result).startswith(str(Path.home()))


class TestImportClass:
    """Tests for import_class — dynamic class loading."""

    def test_valid_class_import(self):
        cls = import_class("pathlib.Path")
        assert cls is Path

    def test_builtin_class_import(self):
        cls = import_class("collections.OrderedDict")
        from collections import OrderedDict
        assert cls is OrderedDict

    def test_invalid_module_raises_import_error(self):
        with pytest.raises(ImportError, match="Module not found"):
            import_class("nonexistent.module.ClassName")

    def test_missing_class_raises_import_error(self):
        with pytest.raises(ImportError, match="Class not found"):
            import_class("pathlib.NonExistentClass")

    def test_invalid_format_raises_import_error(self):
        with pytest.raises(ImportError, match="Invalid class path"):
            import_class("no_dots_at_all")

    def test_pandas_class_import(self):
        cls = import_class("pandas.DataFrame")
        import pandas as pd
        assert cls is pd.DataFrame


if __name__ == "__main__":
    unittest.main()
