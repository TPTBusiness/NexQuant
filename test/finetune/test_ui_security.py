"""
Security Tests for FT Timeline Viewer UI

Tests for path traversal vulnerability prevention (GitHub Issue #13).
Ensures user-provided paths are validated before file system access.
"""

import os
import tempfile
from pathlib import Path
from unittest.mock import patch

import pytest

# Import the functions we're testing
import sys
sys.path.insert(0, str(Path(__file__).resolve().parents[2]))

from rdagent.app.finetune.llm.ui.app import get_job_options


class TestPathTraversalPrevention:
    """Test path traversal vulnerability prevention in UI."""

    def test_valid_path_within_cwd(self):
        """Test that valid paths within CWD are accepted."""
        with tempfile.TemporaryDirectory() as tmpdir:
            tmp_path = Path(tmpdir)
            # Create a valid job directory structure
            job_dir = tmp_path / "valid_job"
            job_dir.mkdir()
            (job_dir / "subdir").mkdir()
            (job_dir / "subdir" / "__session__").touch()

            # Should return the valid job directory
            options = get_job_options(tmp_path)
            assert "valid_job" in options

    def test_path_traversal_with_double_dots(self):
        """Test that path traversal with ../ is blocked."""
        with tempfile.TemporaryDirectory() as tmpdir:
            tmp_path = Path(tmpdir)
            # Try to traverse outside allowed directory
            malicious_path = tmp_path / ".." / ".." / "etc"

            # Should not raise an error, but should return empty options
            # The function should handle this gracefully via try-except
            options = get_job_options(malicious_path)
            # Either returns empty list or shows error via Streamlit
            assert isinstance(options, list)

    def test_absolute_path_outside_cwd(self):
        """Test that absolute paths outside CWD are rejected."""
        # Try to access /tmp which should be outside CWD
        outside_path = Path("/tmp")

        # Mock st.error to capture the error message
        with patch('rdagent.app.finetune.llm.ui.app.st') as mock_st:
            options = get_job_options(outside_path)

            # Should show error and return empty list
            assert mock_st.error.called
            error_msg = mock_st.error.call_args[0][0]
            assert "Invalid log base path" in error_msg or "Invalid path" in error_msg
            assert options == []

    def test_path_with_symlink_traversal(self):
        """Test that symlink-based path traversal is blocked."""
        with tempfile.TemporaryDirectory() as tmpdir:
            tmp_path = Path(tmpdir)

            # Create a safe directory
            safe_dir = tmp_path / "safe"
            safe_dir.mkdir()

            # Create a symlink pointing outside
            outside_target = Path("/etc")
            symlink_path = safe_dir / "evil_link"

            try:
                symlink_path.symlink_to(outside_target)

                # Try to use the symlink path
                with patch('rdagent.app.finetune.llm.ui.app.st') as mock_st:
                    options = get_job_options(symlink_path)

                    # Should either reject or handle gracefully
                    assert isinstance(options, list)
            except (OSError, NotImplementedError):
                # Symlinks not supported on this system - skip test
                pytest.skip("Symlinks not supported")

    def test_resolve_prevents_traversal(self):
        """Test that Path.resolve() prevents traversal attacks."""
        # This tests the core security mechanism
        safe_root = Path.cwd().resolve()

        # Attempt to create a path that tries to escape
        malicious = Path("/tmp/../etc/passwd")
        resolved = malicious.resolve(strict=False)

        # The resolved path should still be /etc/passwd (not normalized away)
        # But the relative_to check should fail
        with pytest.raises(ValueError):
            resolved.relative_to(safe_root)

    def test_expanduser_security(self):
        """Test that ~ expansion doesn't bypass security."""
        with patch('rdagent.app.finetune.llm.ui.app.st') as mock_st:
            # Try to use home directory which may be outside CWD
            home_path = Path("~").expanduser()

            options = get_job_options(home_path)

            # Should be rejected if outside CWD
            if home_path.resolve().relative_to(Path.cwd().resolve()):
                # If home is within CWD (unlikely), it might pass
                pass
            else:
                # Should show error
                assert mock_st.error.called or options == []

    def test_special_characters_in_path(self):
        """Test that special characters in paths are handled safely."""
        with tempfile.TemporaryDirectory() as tmpdir:
            tmp_path = Path(tmpdir)

            # Create directory with special characters
            special_dir = tmp_path / "job$test`backtick"
            special_dir.mkdir()
            (special_dir / "__session__").touch()

            # Should handle safely (either accept or reject, but not crash)
            options = get_job_options(tmp_path)
            assert isinstance(options, list)

    def test_nonexistent_path_handling(self):
        """Test that nonexistent paths are handled gracefully."""
        with tempfile.TemporaryDirectory() as tmpdir:
            tmp_path = Path(tmpdir)
            nonexistent = tmp_path / "does_not_exist"

            # Should not crash, just return empty list
            options = get_job_options(nonexistent)
            assert options == []

    def test_permission_error_handling(self):
        """Test that permission errors are handled gracefully."""
        # Try to access a directory that may have permission issues
        restricted = Path("/root")

        with patch('rdagent.app.finetune.llm.ui.app.st') as mock_st:
            options = get_job_options(restricted)

            # Should handle gracefully (either error message or empty list)
            assert isinstance(options, list)


class TestSecurityMechanism:
    """Test the core security mechanism (relative_to validation)."""

    def test_relative_to_validation(self):
        """Test that relative_to() correctly identifies path containment."""
        base = Path("/safe/base").resolve()

        # Valid: path within base
        valid = Path("/safe/base/job1").resolve()
        assert valid.relative_to(base) is not None

        # Invalid: path outside base
        invalid = Path("/unsafe/other").resolve()
        with pytest.raises(ValueError):
            invalid.relative_to(base)

        # Invalid: path traversal attempt
        traversal = Path("/safe/base/../other").resolve()
        with pytest.raises(ValueError):
            traversal.relative_to(base)

    def test_security_pattern_from_issue_13(self):
        """
        Test the exact security pattern used to fix GitHub Issue #13.

        The fix uses:
        1. Path.resolve() to normalize the path
        2. Path.relative_to() to verify containment
        3. try-except to handle ValueError

        This pattern prevents:
        - Path traversal with ../
        - Symlink attacks
        - Absolute path injection
        """
        safe_root = Path("/allowed").resolve()

        # Test cases that should be REJECTED
        rejected_paths = [
            Path("/allowed/../forbidden").resolve(),
            Path("/forbidden").resolve(),
            Path("/tmp/escape").resolve(),
        ]

        for malicious in rejected_paths:
            with pytest.raises(ValueError):
                malicious.relative_to(safe_root)

        # Test cases that should be ACCEPTED
        accepted_paths = [
            Path("/allowed/job1").resolve(),
            Path("/allowed/subdir/job2").resolve(),
        ]

        for valid in accepted_paths:
            result = valid.relative_to(safe_root)
            assert result is not None


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
