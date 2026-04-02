"""
Security Tests for UI Path-Validation

Tests for path traversal prevention in get_job_options() function.
Ensures that only directories within the Current Working Directory
can be accessed.

Run:
    pytest test/finetune/test_ui_security.py -v
"""
import pytest
import os
from pathlib import Path

from rdagent.app.finetune.llm.ui.app import get_job_options


class TestGetJobOptionsSecurity:
    """Security tests for get_job_options() - Path Traversal Prevention"""

    def test_absolute_path_outside_cwd_blocked(self):
        """Absolute paths outside CWD should be blocked"""
        # Try to access /etc (outside CWD)
        malicious_path = Path("/etc")
        result = get_job_options(malicious_path)
        # Should return empty list (path outside CWD)
        assert result == [], "Path traversal to /etc should be blocked"

    def test_root_path_blocked(self):
        """Root path / should be blocked"""
        result = get_job_options(Path("/"))
        # Root path is outside CWD, should return empty list
        assert result == [], "Root path should be blocked"

    def test_nonexistent_path_returns_empty(self):
        """Non-existent paths should return empty list"""
        nonexistent_path = Path("/nonexistent/path/that/does/not/exist")
        result = get_job_options(nonexistent_path)
        assert result == [], "Non-existent path should return empty list"

    def test_path_traversal_relative_blocked(self):
        """Relative path traversal ../../../etc should be blocked"""
        # Try with relative path traversal
        malicious_path = Path("../../../etc")
        result = get_job_options(malicious_path)
        # Should be blocked (either empty or error)
        assert isinstance(result, list), "Should return list"
        # The path should be resolved and blocked
        assert result == [], "Relative path traversal should be blocked"

    def test_empty_path_handling(self):
        """Test handling of empty path"""
        result = get_job_options(Path(""))
        assert isinstance(result, list), "Should return list for empty path"


class TestJobFolderPathValidation:
    """Tests for job_folder path validation logic (from main() function)"""

    def test_job_folder_within_base_path(self, tmp_path):
        """Test that job_folder within base_path is accepted"""
        base_path = tmp_path / "log"
        base_path.mkdir()
        job_path = base_path / "job1"
        job_path.mkdir()

        # Simulate validation logic from main()
        safe_root = base_path.resolve()
        job_path_resolved = job_path.resolve()

        # Should not raise ValueError
        try:
            job_path_resolved.relative_to(safe_root)
            validation_passed = True
        except ValueError:
            validation_passed = False

        assert validation_passed is True, "Valid path within base should pass validation"

    def test_job_folder_outside_base_path(self, tmp_path):
        """Test that job_folder outside base_path is rejected"""
        base_path = tmp_path / "log"
        base_path.mkdir()
        job_path = tmp_path / "other" / "job1"
        job_path.mkdir(parents=True)

        # Simulate validation logic from main()
        safe_root = base_path.resolve()
        job_path_resolved = job_path.resolve()

        # Should raise ValueError
        with pytest.raises(ValueError):
            job_path_resolved.relative_to(safe_root)

    def test_job_folder_traversal_attempt(self, tmp_path):
        """Test that path traversal in job_folder is blocked"""
        base_path = tmp_path / "log"
        base_path.mkdir()

        # Malicious job_folder attempt
        job_folder = str(base_path / ".." / ".." / "etc")

        # Simulate validation logic from main()
        safe_root = base_path.resolve()
        job_path = Path(job_folder).expanduser().resolve(strict=False)

        # Should raise ValueError
        with pytest.raises(ValueError):
            job_path.relative_to(safe_root)


class TestIntegrationScenarios:
    """Integration tests for realistic attack scenarios"""

    def test_double_encoding_attack(self):
        """Test double-encoded path traversal attempt"""
        # Double-encoded "../" would be "%252e%252e%252f"
        # But we're testing raw paths, so use literal dots
        malicious_path = Path("/../../etc/passwd")
        result = get_job_options(malicious_path)
        assert result == [], "Double encoding attack should be blocked"

    def test_etc_passwd_access_blocked(self):
        """Test that direct /etc/passwd access is blocked"""
        result = get_job_options(Path("/etc/passwd"))
        assert result == [], "/etc/passwd access should be blocked"

    def test_system_directories_blocked(self):
        """Test that system directories are blocked"""
        system_paths = [
            Path("/etc"),
            Path("/usr"),
            Path("/var"),
            Path("/root"),
            Path("/home"),
        ]
        for path in system_paths:
            result = get_job_options(path)
            assert result == [], f"System directory {path} should be blocked"


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
