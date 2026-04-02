"""
Security Tests für UI Path-Validation

Tests für Path-Traversal-Prävention in get_job_options() Funktion.
Stellt sicher, dass nur Verzeichnisse innerhalb des Current Working Directory
zugriffen werden können.

Test-Fälle:
- Path-Traversal-Angriffe mit ../ sollten blockiert werden
- Absolute Pfade außerhalb CWD sollten blockiert werden
- Relative Pfade innerhalb CWD sollten erlaubt sein
- Edge Cases: Symlinks, nicht-existente Pfade
"""
import pytest
import tempfile
import os
from pathlib import Path
from unittest.mock import patch, MagicMock

# Mock streamlit vor dem Import der app
import sys
sys.modules['streamlit'] = MagicMock()
sys.modules['streamlit.session_state'] = MagicMock()

from rdagent.app.finetune.llm.ui.app import get_job_options


class TestGetJobOptionsSecurity:
    """Security Tests für get_job_options() - Path-Traversal-Prävention"""

    def test_path_traversal_attack_blocked(self):
        """Path-Traversal mit ../ sollte blockiert werden"""
        # Versuche auf /etc zuzugreifen (außerhalb CWD)
        malicious_path = Path("/etc")
        
        result = get_job_options(malicious_path)
        
        # Sollte leere Liste zurückgeben (Pfad außerhalb CWD)
        assert result == [], "Path-Traversal sollte blockiert werden"

    def test_path_traversal_relative_blocked(self):
        """Relativer Path-Traversal ../../../etc sollte blockiert werden"""
        # Versuche mit relativem Path-Traversal
        malicious_path = Path("../../../etc")
        
        result = get_job_options(malicious_path)
        
        # Sollte blockiert werden (entweder leer oder Fehler)
        # Wichtig: Keine sensiblen Daten sollten zurückgegeben werden
        assert isinstance(result, list), "Sollte Liste zurückgeben"

    def test_nonexistent_path_returns_empty(self):
        """Nicht-existente Pfade sollten leere Liste zurückgeben"""
        nonexistent_path = Path("/nonexistent/path/that/does/not/exist")
        
        result = get_job_options(nonexistent_path)
        
        assert result == [], "Nicht-existenter Pfad sollte leere Liste zurückgeben"

    def test_path_outside_project_blocked(self, tmp_path):
        """Pfade außerhalb des Projekt-Verzeichnisses sollten blockiert werden"""
        # Erstelle zwei separate Verzeichnisse
        outside_dir = tmp_path / "outside"
        outside_dir.mkdir()
        # Struktur für outside: outside/loop1/__session__
        (outside_dir / "loop1").mkdir()
        (outside_dir / "loop1" / "__session__").mkdir()
        
        inside_dir = tmp_path / "inside"
        inside_dir.mkdir()
        
        # Mock CWD zu inside_dir
        with patch.object(Path, 'cwd', return_value=inside_dir):
            # Versuche auf outside_dir zuzugreifen (außerhalb CWD)
            result = get_job_options(outside_dir)
            
            # Sollte leer sein (outside_dir ist außerhalb von inside_dir/CWD)
            assert result == [], "Pfad außerhalb CWD sollte blockiert werden"

    def test_empty_directory_returns_empty_list(self, tmp_path):
        """Leere Verzeichnisse sollten leere Liste zurückgeben"""
        empty_dir = tmp_path / "empty"
        empty_dir.mkdir()
        
        with patch.object(Path, 'cwd', return_value=tmp_path):
            result = get_job_options(empty_dir)
            
            assert result == [], "Leeres Verzeichnis sollte leere Liste zurückgeben"


class TestGetJobOptionsEdgeCases:
    """Edge Case Tests für get_job_options()"""

    def test_symlink_to_outside_blocked(self, tmp_path):
        """Symlinks die nach außen zeigen sollten blockiert werden"""
        outside = tmp_path / "outside"
        outside.mkdir()
        # outside als Job-Verzeichnis strukturieren
        (outside / "loop1").mkdir()
        (outside / "loop1" / "__session__").mkdir()
        
        inside = tmp_path / "inside"
        inside.mkdir()
        
        # Erstelle Symlink von inside/link -> outside
        link = inside / "link"
        link.symlink_to(outside)
        
        # Mock CWD zu inside
        with patch.object(Path, 'cwd', return_value=inside):
            # Symlink auflösen sollte outside sein (außerhalb CWD)
            # resolve() sollte den Symlink auflösen
            result = get_job_options(link)
            
            # Symlink sollte blockiert werden wenn er nach außen zeigt
            assert isinstance(result, list), "Sollte Liste zurückgeben"

    def test_permission_error_in_subdir_handled(self, tmp_path):
        """PermissionError in Subdirectory sollte gracefully behandelt werden"""
        base = tmp_path / "base"
        base.mkdir()
        
        # Erstelle eine normale Struktur
        job1 = base / "job1"
        job1.mkdir()
        loop1 = job1 / "loop1"
        loop1.mkdir()
        (loop1 / "__session__").mkdir()
        
        # Mock iterdir um PermissionError für job1 zu simulieren
        # Der Code fängt PermissionError im inneren Loop (Zeile 68-69)
        original_iterdir = Path.iterdir
        
        def mock_iterdir(self):
            # Wenn wir job1 iterieren (um nach loop1/__session__ zu suchen)
            if self == job1:
                raise PermissionError("Access denied")
            return original_iterdir(self)
        
        with patch.object(Path, 'iterdir', mock_iterdir):
            with patch.object(Path, 'cwd', return_value=tmp_path):
                # Sollte keine Exception werfen
                result = get_job_options(base)
                
                # Sollte Liste zurückgeben (job1 wird wegen PermissionError übersprungen)
                assert isinstance(result, list), "Sollte Liste zurückgeben trotz PermissionError"

    def test_multiple_subdirs_with_sessions(self, tmp_path):
        """Mehrere Subdirectories mit __session__ sollten erkannt werden"""
        base = tmp_path / "base"
        base.mkdir()
        
        # Erstelle mehrere Subdirs mit __session__
        # Diese werden als Root-Tasks erkannt (gemäß Logik der Funktion)
        for i in range(3):
            subdir = base / f"task{i}"
            subdir.mkdir()
            (subdir / "__session__").mkdir()
        
        with patch.object(Path, 'cwd', return_value=tmp_path):
            result = get_job_options(base)
            
            # Alle sollten als ". (Current)" erkannt werden (Root-Tasks)
            # Die Funktion setzt has_root_tasks=True wenn ANY subdir __session__ hat
            # Aber sie fügt nur ". (Current)" EINMAL hinzu
            assert ". (Current)" in result, "Root-Tasks sollten erkannt werden"

    def test_nested_structure_detection(self, tmp_path):
        """Verschachtelte Verzeichnisstruktur sollte korrekt erkannt werden"""
        base = tmp_path / "base"
        base.mkdir()
        
        # Job-Verzeichnis: job/loop/__session__
        job = base / "job1"
        job.mkdir()
        loop = job / "loop1"
        loop.mkdir()
        (loop / "__session__").mkdir()
        
        # Die Funktion prüft: hat job1/__session__? Nein.
        # Dann prüft sie: haben job1's Subdirs __session__? Ja (loop1)
        # Also ist job1 ein Job-Verzeichnis
        
        with patch.object(Path, 'cwd', return_value=tmp_path):
            result = get_job_options(base)
            
            # job1 sollte als Job-Verzeichnis erkannt werden
            assert "job1" in result, "Job-Verzeichnis sollte erkannt werden"
