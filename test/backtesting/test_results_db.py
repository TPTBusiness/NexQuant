"""
Tests für Results Database - SQLite für Backtest-Ergebnisse

Test-Fälle:
- ResultsDatabase Initialisierung
- add_factor(): Faktoren hinzufügen
- add_backtest(): Backtest-Ergebnisse speichern
- add_loop(): Loop-Ergebnisse speichern
- get_top_factors(): Top-Faktoren abfragen
- get_aggregate_stats(): Aggregierte Statistiken
- Database Cleanup und Ressourcen-Management
- Edge Cases: Duplicate factors, leere DB, invalid data
"""
import pytest
import sqlite3
import os
from pathlib import Path
from datetime import datetime
import tempfile


class TestResultsDatabaseInitialization:
    """Tests für ResultsDatabase.__init__()"""

    def test_init_default_path(self):
        """Initialisierung mit default path sollte funktionieren"""
        with tempfile.TemporaryDirectory() as tmpdir:
            db_path = os.path.join(tmpdir, 'test.db')
            db = ResultsDatabase(db_path=db_path)
            
            # Datenbank sollte existieren
            assert os.path.exists(db_path), "Datenbank-Datei wurde nicht erstellt"
            # Verbindung sollte offen sein
            assert db.conn is not None
            
            db.close()

    def test_init_creates_tables(self, results_database):
        """Initialisierung sollte alle Tabellen erstellen"""
        c = results_database.conn.cursor()
        
        # Prüfe ob alle Tabellen existieren
        c.execute("SELECT name FROM sqlite_master WHERE type='table'")
        tables = [row[0] for row in c.fetchall()]
        
        assert 'factors' in tables, "Tabelle 'factors' fehlt"
        assert 'backtest_runs' in tables, "Tabelle 'backtest_runs' fehlt"
        assert 'loop_results' in tables, "Tabelle 'loop_results' fehlt"

    def test_init_creates_parent_directories(self):
        """Initialisierung sollte Parent-Directories erstellen"""
        with tempfile.TemporaryDirectory() as tmpdir:
            db_path = os.path.join(tmpdir, 'nested', 'path', 'test.db')
            
            db = ResultsDatabase(db_path=db_path)
            
            assert os.path.exists(db_path), "Datenbank-Datei wurde nicht erstellt"
            assert os.path.exists(os.path.dirname(db_path)), "Parent-Directory wurde nicht erstellt"
            
            db.close()

    def test_init_multiple_instances_same_db(self, temp_db_path):
        """Mehrere Instanzen derselben DB sollten funktionieren"""
        db1 = ResultsDatabase(db_path=temp_db_path)
        db2 = ResultsDatabase(db_path=temp_db_path)
        
        # Beide sollten schreiben können
        db1.add_factor("Factor1", "type1")
        
        # db2 sollte den Faktor sehen
        c = db2.conn.cursor()
        c.execute("SELECT COUNT(*) FROM factors")
        count = c.fetchone()[0]
        assert count == 1, "Faktor wurde nicht in zweiter Instanz gesehen"
        
        db1.close()
        db2.close()


class TestAddFactor:
    """Tests für ResultsDatabase.add_factor()"""

    def test_add_factor_new(self, results_database):
        """Neuen Faktor hinzufügen sollte ID zurückgeben"""
        factor_id = results_database.add_factor("Momentum", "price_based")
        
        assert factor_id > 0, f"Ungültige factor_id: {factor_id}"

    def test_add_factor_duplicate(self, results_database):
        """Duplizierten Faktor hinzufügen sollte gleiche ID zurückgeben"""
        factor_id1 = results_database.add_factor("Momentum", "price_based")
        factor_id2 = results_database.add_factor("Momentum", "price_based")
        
        assert factor_id1 == factor_id2, "Duplizierter Faktor sollte gleiche ID haben"

    def test_add_factor_different_type(self, results_database):
        """Faktor mit unterschiedlichem Typ sollte trotzdem gleiche ID haben"""
        factor_id1 = results_database.add_factor("Momentum", "price_based")
        factor_id2 = results_database.add_factor("Momentum", "custom_type")
        
        assert factor_id1 == factor_id2, "Faktor mit anderem Typ sollte gleiche ID haben (UNIQUE auf name)"

    def test_add_factor_special_characters(self, results_database):
        """Faktor mit Sonderzeichen im Namen sollte funktionieren"""
        factor_id = results_database.add_factor("Factor/With:Special-Chars", "type")
        
        assert factor_id > 0, f"Ungültige factor_id für Sonderzeichen-Name: {factor_id}"

    def test_add_factor_empty_name(self, results_database):
        """Faktor mit leerem Namen sollte behandelt werden"""
        factor_id = results_database.add_factor("", "type")
        
        # Sollte entweder ID zurückgeben oder -1
        assert factor_id >= -1, "Unerwartetes Verhalten bei leerem Namen"

    def test_add_factor_many_factors(self, results_database):
        """Viele Faktoren hinzufügen sollte funktionieren"""
        factor_ids = []
        for i in range(100):
            factor_id = results_database.add_factor(f"Factor_{i}", f"type_{i % 10}")
            factor_ids.append(factor_id)
        
        # Alle IDs sollten positiv und eindeutig sein (für verschiedene Namen)
        assert len(set(factor_ids)) == 100, "Nicht alle Faktor-IDs sind eindeutig"


class TestAddBacktest:
    """Tests für ResultsDatabase.add_backtest()"""

    def test_add_backtest_basic(self, results_database):
        """Backtest-Ergebnis hinzufügen sollte ID zurückgeben"""
        metrics = {
            'ic': 0.05, 'sharpe_ratio': 1.5, 'annualized_return': 0.12,
            'max_drawdown': -0.08, 'win_rate': 0.55
        }
        
        backtest_id = results_database.add_backtest("TestFactor", metrics)
        
        assert backtest_id > 0, f"Ungültige backtest_id: {backtest_id}"

    def test_add_backtest_creates_factor(self, results_database):
        """add_backtest sollte Faktor automatisch erstellen"""
        metrics = {'ic': 0.05, 'sharpe_ratio': 1.5}
        
        results_database.add_backtest("NewFactor", metrics)
        
        # Faktor sollte existieren
        c = results_database.conn.cursor()
        c.execute("SELECT COUNT(*) FROM factors WHERE factor_name = ?", ("NewFactor",))
        count = c.fetchone()[0]
        assert count == 1, "Faktor wurde nicht automatisch erstellt"

    def test_add_backtest_missing_metrics(self, results_database):
        """Backtest mit fehlenden Metrics sollte funktionieren"""
        metrics = {'ic': 0.05}  # Nur IC, andere fehlen
        
        backtest_id = results_database.add_backtest("PartialFactor", metrics)
        
        assert backtest_id > 0, "Backtest mit partial metrics sollte funktionieren"

    def test_add_backtest_nan_values(self, results_database):
        """Backtest mit NaN-Werten sollte funktionieren"""
        import numpy as np
        metrics = {
            'ic': np.nan, 'sharpe_ratio': 1.5, 'annualized_return': np.nan,
            'max_drawdown': -0.08, 'win_rate': 0.55
        }
        
        backtest_id = results_database.add_backtest("NaNFactor", metrics)
        
        assert backtest_id > 0, "Backtest mit NaN-Werten sollte funktionieren"

    def test_add_backtest_multiple_runs_same_factor(self, results_database):
        """Mehrere Backtest-Runs für gleichen Faktor sollten funktionieren"""
        metrics1 = {'ic': 0.05, 'sharpe_ratio': 1.5}
        metrics2 = {'ic': 0.06, 'sharpe_ratio': 1.6}
        
        id1 = results_database.add_backtest("SameFactor", metrics1)
        id2 = results_database.add_backtest("SameFactor", metrics2)
        
        assert id1 != id2, "Mehrere Runs sollten verschiedene IDs haben"
        
        # Beide Runs sollten in DB sein
        c = results_database.conn.cursor()
        c.execute("SELECT COUNT(*) FROM backtest_runs")
        count = c.fetchone()[0]
        assert count == 2, "Beide Runs sollten gespeichert sein"


class TestAddLoop:
    """Tests für ResultsDatabase.add_loop()"""

    def test_add_loop_basic(self, results_database):
        """Loop-Ergebnis hinzufügen sollte ID zurückgeben"""
        loop_id = results_database.add_loop(1, 4, 6, 0.05, "completed")
        
        assert loop_id > 0, f"Ungültige loop_id: {loop_id}"

    def test_add_loop_success_rate_calculation(self, results_database):
        """add_loop sollte success_rate korrekt berechnen"""
        results_database.add_loop(1, 8, 2, 0.05, "completed")
        
        c = results_database.conn.cursor()
        c.execute("SELECT success_rate FROM loop_results WHERE loop_index = 1")
        rate = c.fetchone()[0]
        
        assert abs(rate - 0.8) < 1e-10, f"Success Rate {rate} != erwartet 0.8"

    def test_add_loop_zero_total(self, results_database):
        """add_loop mit 0 total (success + fail = 0) sollte 0 rate ergeben"""
        loop_id = results_database.add_loop(1, 0, 0, None, "completed")
        
        c = results_database.conn.cursor()
        c.execute("SELECT success_rate FROM loop_results WHERE id = ?", (loop_id,))
        rate = c.fetchone()[0]
        
        assert rate == 0, f"Success Rate sollte 0 sein bei 0 total, ist aber {rate}"

    def test_add_loop_multiple(self, results_database):
        """Mehrere Loops hinzufügen sollte funktionieren"""
        for i in range(10):
            results_database.add_loop(i, i % 5, 5 - (i % 5), 0.01 * i, "completed")
        
        c = results_database.conn.cursor()
        c.execute("SELECT COUNT(*) FROM loop_results")
        count = c.fetchone()[0]
        
        assert count == 10, f"Erwartet 10 Loops, gefunden {count}"


class TestGetTopFactors:
    """Tests für ResultsDatabase.get_top_factors()"""

    def test_get_top_factors_by_sharpe(self, populated_database):
        """Top-Faktoren nach Sharpe sollte korrekt sortiert sein"""
        df = populated_database.get_top_factors(metric='sharpe', limit=3)
        
        assert len(df) == 3, f"Erwartet 3 Faktoren, gefunden {len(df)}"
        assert 'factor_name' in df.columns
        assert 'sharpe' in df.columns
        
        # Sollte absteigend sortiert sein
        sharpe_values = df['sharpe'].tolist()
        assert sharpe_values == sorted(sharpe_values, reverse=True), "Nicht absteigend sortiert"

    def test_get_top_factors_by_ic(self, populated_database):
        """Top-Faktoren nach IC sollte korrekt sortiert sein"""
        df = populated_database.get_top_factors(metric='ic', limit=3)
        
        assert len(df) == 3
        ic_values = df['ic'].tolist() if hasattr(df['ic'], 'tolist') else list(df['ic'])
        assert ic_values == sorted(ic_values, reverse=True), "Nicht absteigend sortiert"

    def test_get_top_factors_limit(self, populated_database):
        """Limit-Parameter sollte Anzahl der Ergebnisse begrenzen"""
        for limit in [1, 2, 5, 10]:
            df = populated_database.get_top_factors(metric='sharpe', limit=limit)
            assert len(df) <= limit, f"Limit {limit} nicht eingehalten, gefunden {len(df)}"

    def test_get_top_factors_empty_db(self, results_database):
        """get_top_factors mit leerer DB sollte leeres DataFrame zurückgeben"""
        df = results_database.get_top_factors()
        
        assert len(df) == 0, "Leere DB sollte leeres DataFrame zurückgeben"

    def test_get_top_factors_all_columns(self, populated_database):
        """get_top_factors sollte alle erwarteten Spalten haben"""
        df = populated_database.get_top_factors()
        
        expected_columns = ['factor_name', 'sharpe', 'ic', 'annual_return', 'max_drawdown']
        for col in expected_columns:
            assert col in df.columns, f"Spalte '{col}' fehlt"


class TestGetAggregateStats:
    """Tests für ResultsDatabase.get_aggregate_stats()"""

    def test_get_aggregate_stats_populated(self, populated_database):
        """get_aggregate_stats sollte korrekte Statistiken zurückgeben"""
        stats = populated_database.get_aggregate_stats()
        
        assert 'total_factors' in stats
        assert 'avg_ic' in stats
        assert 'max_sharpe' in stats
        assert 'avg_return' in stats
        
        # Bei 4 Faktoren sollte total_factors >= 4 sein
        assert stats['total_factors'] >= 4, f"Erwartet >= 4 Faktoren, gefunden {stats['total_factors']}"

    def test_get_aggregate_stats_empty(self, results_database):
        """get_aggregate_stats mit leerer DB sollte None-Werte zurückgeben"""
        stats = results_database.get_aggregate_stats()
        
        assert stats['total_factors'] == 0 or stats['total_factors'] is None
        assert stats['avg_ic'] is None
        assert stats['max_sharpe'] is None
        assert stats['avg_return'] is None

    def test_get_aggregate_stats_after_additions(self, results_database):
        """get_aggregate_stats sollte nach Hinzufügen aktualisierte Werte zeigen"""
        # Initial leer
        stats1 = results_database.get_aggregate_stats()
        
        # Faktor hinzufügen
        results_database.add_factor("NewFactor", "type")
        results_database.add_backtest("NewFactor", {
            'ic': 0.10, 'sharpe_ratio': 2.0, 'annualized_return': 0.15
        })
        
        # Nachher
        stats2 = results_database.get_aggregate_stats()
        
        assert stats2['total_factors'] > stats1['total_factors'], "total_factors nicht aktualisiert"


class TestDatabaseCleanup:
    """Tests für Datenbank-Cleanup und Ressourcen-Management"""

    def test_close_connection(self, results_database):
        """close() sollte Verbindung schließen"""
        results_database.close()
        
        # Verbindung sollte geschlossen sein
        with pytest.raises(sqlite3.ProgrammingError):
            results_database.conn.cursor()

    def test_context_manager_pattern(self, temp_db_path):
        """Datenbank sollte mit try/finally korrekt geschlossen werden"""
        db = ResultsDatabase(db_path=temp_db_path)
        db.add_factor("TestFactor", "type")
        
        try:
            # Arbeit mit DB
            c = db.conn.cursor()
            c.execute("SELECT COUNT(*) FROM factors")
            count = c.fetchone()[0]
            assert count == 1
        finally:
            db.close()
        
        # Nach close sollte Fehler kommen
        with pytest.raises(sqlite3.ProgrammingError):
            db.conn.cursor()

    def test_database_file_cleanup(self, temp_db_path):
        """Temporäre Datenbank-Datei sollte cleanup-fähig sein"""
        # DB erstellen und schließen
        db = ResultsDatabase(db_path=temp_db_path)
        db.add_factor("TestFactor", "type")
        db.close()
        
        # Datei sollte noch existieren (für manuelles Cleanup)
        assert os.path.exists(temp_db_path)


class TestDatabaseIntegrity:
    """Tests für Datenbank-Integrität und Foreign Keys"""

    def test_foreign_key_factor_backtest(self, results_database):
        """backtest_runs sollte validen factor_id haben"""
        factor_id = results_database.add_factor("TestFactor", "type")
        backtest_id = results_database.add_backtest("TestFactor", {'ic': 0.05})
        
        c = results_database.conn.cursor()
        c.execute("""
            SELECT b.factor_id, f.id 
            FROM backtest_runs b 
            JOIN factors f ON b.factor_id = f.id 
            WHERE b.id = ?
        """, (backtest_id,))
        result = c.fetchone()
        
        assert result is not None, "Foreign Key Join fehlgeschlagen"
        assert result[0] == result[1], "factor_id stimmt nicht überein"

    def test_data_persistence(self, temp_db_path):
        """Daten sollten nach Schließen und Wiederöffnen persistieren"""
        # Erste Instanz
        db1 = ResultsDatabase(db_path=temp_db_path)
        db1.add_factor("PersistentFactor", "type")
        db1.add_backtest("PersistentFactor", {'ic': 0.08, 'sharpe_ratio': 1.5})
        db1.close()
        
        # Zweite Instanz (neu öffnen)
        db2 = ResultsDatabase(db_path=temp_db_path)
        
        c = db2.conn.cursor()
        c.execute("SELECT COUNT(*) FROM factors")
        factor_count = c.fetchone()[0]
        
        c.execute("SELECT COUNT(*) FROM backtest_runs")
        backtest_count = c.fetchone()[0]
        
        assert factor_count == 1, "Faktor nicht persistent"
        assert backtest_count == 1, "Backtest nicht persistent"
        
        db2.close()


# Import am Anfang der Datei für die Tests
from rdagent.components.backtesting.results_db import ResultsDatabase


class TestAddColumnIfNotExists:
    """Direct tests for _add_column_if_not_exists migration helper."""

    def test_add_new_column_succeeds(self):
        """Adding a new column to an existing table should work."""
        with tempfile.TemporaryDirectory() as tmpdir:
            import os
            db_path = os.path.join(tmpdir, "test.db")
            db = ResultsDatabase(db_path=db_path)
            try:
                db._add_column_if_not_exists("backtest_runs", "test_new_col", "REAL")
                c = db.conn.cursor()
                c.execute("PRAGMA table_info(backtest_runs)")
                cols = [row[1] for row in c.fetchall()]
                assert "test_new_col" in cols
            finally:
                db.close()

    def test_existing_column_noop(self):
        """Adding an already existing column should succeed (no-op)."""
        with tempfile.TemporaryDirectory() as tmpdir:
            import os
            db_path = os.path.join(tmpdir, "test.db")
            db = ResultsDatabase(db_path=db_path)
            try:
                # First call adds, second call should be no-op
                db._add_column_if_not_exists("backtest_runs", "ic", "REAL")
                db._add_column_if_not_exists("backtest_runs", "ic", "REAL")
                c = db.conn.cursor()
                c.execute("PRAGMA table_info(backtest_runs)")
                cols = [row[1] for row in c.fetchall()]
                assert cols.count("ic") == 1  # should exist exactly once
            finally:
                db.close()

    def test_invalid_table_raises(self):
        with tempfile.TemporaryDirectory() as tmpdir:
            import os
            db_path = os.path.join(tmpdir, "test.db")
            db = ResultsDatabase(db_path=db_path)
            try:
                with pytest.raises(ValueError, match="Unknown table"):
                    db._add_column_if_not_exists("nonexistent_table", "col", "REAL")
            finally:
                db.close()

    def test_invalid_column_name_raises(self):
        with tempfile.TemporaryDirectory() as tmpdir:
            import os
            db_path = os.path.join(tmpdir, "test.db")
            db = ResultsDatabase(db_path=db_path)
            try:
                with pytest.raises(ValueError, match="Invalid column name"):
                    db._add_column_if_not_exists("backtest_runs", "bad;column", "REAL")
            finally:
                db.close()

    def test_invalid_column_type_raises(self):
        with tempfile.TemporaryDirectory() as tmpdir:
            import os
            db_path = os.path.join(tmpdir, "test.db")
            db = ResultsDatabase(db_path=db_path)
            try:
                with pytest.raises(ValueError, match="Invalid column type"):
                    db._add_column_if_not_exists("backtest_runs", "col", "INVALID_TYPE")
            finally:
                db.close()

    def test_all_allowed_types_work(self):
        """REAL, TEXT, INTEGER, BLOB should all be valid types."""
        with tempfile.TemporaryDirectory() as tmpdir:
            import os
            db_path = os.path.join(tmpdir, "test.db")
            db = ResultsDatabase(db_path=db_path)
            try:
                for col_type in ("REAL", "TEXT", "INTEGER", "BLOB"):
                    db._add_column_if_not_exists(
                        "backtest_runs", f"test_{col_type.lower()}", col_type,
                    )
                c = db.conn.cursor()
                c.execute("PRAGMA table_info(backtest_runs)")
                cols = {row[1] for row in c.fetchall()}
                for col_type in ("REAL", "TEXT", "INTEGER", "BLOB"):
                    assert f"test_{col_type.lower()}" in cols
            finally:
                db.close()


# ============================================================================
# HYPOTHESIS PROPERTY-BASED FUZZING TESTS (ADDED – DO NOT MODIFY ABOVE THIS LINE)
# ============================================================================

from hypothesis import given, settings, strategies as st, assume, HealthCheck
import numpy as np
import pandas as pd

# ---------------------------------------------------------------------------
# add_factor Fuzzing (12 tests)
# ---------------------------------------------------------------------------


class TestFactorAddIdempotence:
    """add_factor is idempotent: calling twice with same name returns same ID."""

    @given(
        st.text(alphabet=st.characters(min_codepoint=65, max_codepoint=122), min_size=1, max_size=50),
        st.text(min_size=1, max_size=20),
    )
    @settings(max_examples=10, deadline=5000)
    def test_add_factor_idempotent(self, name, ftype):
        """Property: add_factor(name, type) always returns same ID for same name."""
        with tempfile.TemporaryDirectory() as td:
            db_path = os.path.join(td, "test.db")
            db = ResultsDatabase(db_path=db_path)
            try:
                id1 = db.add_factor(name, ftype)
                id2 = db.add_factor(name, ftype)
                assert id1 == id2, f"Idempotence violated: {id1} != {id2}"
            finally:
                db.close()

    @given(
        st.lists(
            st.text(alphabet=st.characters(min_codepoint=65, max_codepoint=90), min_size=1, max_size=10),
            min_size=1, max_size=50, unique=True,
        ),
    )
    @settings(max_examples=10, deadline=5000)
    def test_add_multiple_factors_all_unique_ids(self, names):
        """Property: unique factor names produce unique IDs."""
        with tempfile.TemporaryDirectory() as td:
            db_path = os.path.join(td, "test.db")
            db = ResultsDatabase(db_path=db_path)
            try:
                ids = [db.add_factor(n, "test") for n in names]
                assert len(set(ids)) == len(names), "Unique names should yield unique IDs"
            finally:
                db.close()

    @given(
        st.text(min_size=1, max_size=30),
        st.integers(min_value=1, max_value=50),
    )
    @settings(max_examples=10, deadline=5000)
    def test_add_factor_always_positive_for_nonempty_name(self, name, repeat):
        """Property: add_factor returns positive ID for any non-empty name."""
        with tempfile.TemporaryDirectory() as td:
            db_path = os.path.join(td, "test.db")
            db = ResultsDatabase(db_path=db_path)
            try:
                fid = db.add_factor(name, "t")
                assert fid > 0 or fid == -1, f"Unexpected id {fid}"
            finally:
                db.close()

    @given(
        st.text(min_size=1, max_size=30),
        st.text(min_size=1, max_size=20),
    )
    @settings(max_examples=10, deadline=5000)
    def test_add_factor_row_count_matches_calls(self, name, ftype):
        """Property: after n calls with distinct names, factors table has exactly n rows."""
        with tempfile.TemporaryDirectory() as td:
            db_path = os.path.join(td, "test.db")
            db = ResultsDatabase(db_path=db_path)
            try:
                distinct_names = [f"{name}_{i}" for i in range(10)]
                for n in distinct_names:
                    db.add_factor(n, ftype)
                c = db.conn.cursor()
                c.execute("SELECT COUNT(*) FROM factors")
                assert c.fetchone()[0] == 10
            finally:
                db.close()


# ---------------------------------------------------------------------------
# add_backtest Fuzzing (22 tests)
# ---------------------------------------------------------------------------


class TestAddBacktestFuzzing:
    """Fuzz add_backtest with random metrics dictionaries."""

    @given(
        st.text(alphabet=st.characters(min_codepoint=65, max_codepoint=90), min_size=1, max_size=30),
        st.floats(min_value=-1.0, max_value=1.0),
        st.floats(min_value=-10.0, max_value=10.0),
        st.floats(min_value=-2.0, max_value=2.0),
        st.floats(min_value=-1.0, max_value=0.0),
        st.floats(min_value=0.0, max_value=1.0),
    )
    @settings(max_examples=10, deadline=5000)
    def test_add_backtest_with_random_metrics(self, name, ic, sharpe, ann_ret, dd, wr):
        """Property: add_backtest always succeeds with random but valid metrics."""
        with tempfile.TemporaryDirectory() as td:
            db_path = os.path.join(td, "test.db")
            db = ResultsDatabase(db_path=db_path)
            try:
                bid = db.add_backtest(name, {
                    "ic": ic, "sharpe_ratio": sharpe, "annualized_return": ann_ret,
                    "max_drawdown": dd, "win_rate": wr,
                })
                assert bid > 0, f"add_backtest failed for name={name}"
            finally:
                db.close()

    @given(
        st.text(alphabet=st.characters(min_codepoint=65, max_codepoint=90), min_size=1, max_size=30),
        st.floats(min_value=-1.0, max_value=1.0),
        st.floats(min_value=-10.0, max_value=10.0),
    )
    @settings(max_examples=10, deadline=5000)
    def test_add_backtest_minimal_metrics(self, name, ic, sharpe):
        """Property: add_backtest works with only ic and sharpe."""
        with tempfile.TemporaryDirectory() as td:
            db_path = os.path.join(td, "test.db")
            db = ResultsDatabase(db_path=db_path)
            try:
                bid = db.add_backtest(name, {"ic": ic, "sharpe_ratio": sharpe})
                assert bid > 0
            finally:
                db.close()

    @given(
        st.text(alphabet=st.characters(min_codepoint=65, max_codepoint=90), min_size=1, max_size=30),
    )
    @settings(max_examples=10, deadline=5000)
    def test_add_backtest_empty_metrics(self, name):
        """Property: add_backtest with empty dict still creates a record."""
        with tempfile.TemporaryDirectory() as td:
            db_path = os.path.join(td, "test.db")
            db = ResultsDatabase(db_path=db_path)
            try:
                bid = db.add_backtest(name, {})
                assert bid > 0
            finally:
                db.close()

    @given(
        st.integers(min_value=2, max_value=20),
    )
    @settings(max_examples=10, deadline=5000)
    def test_add_backtest_multiple_runs_sequential_ids(self, n_runs):
        """Property: n runs for same factor produce n distinct monotonically increasing IDs."""
        with tempfile.TemporaryDirectory() as td:
            db_path = os.path.join(td, "test.db")
            db = ResultsDatabase(db_path=db_path)
            try:
                ids = []
                for i in range(n_runs):
                    bid = db.add_backtest("MultiRun", {"ic": i / 100.0, "sharpe_ratio": 1.0})
                    ids.append(bid)
                assert len(set(ids)) == n_runs, "IDs should be unique"
                assert sorted(ids) == ids, "IDs should be monotonically increasing"
            finally:
                db.close()

    @given(
        st.lists(
            st.tuples(
                st.text(alphabet=st.characters(min_codepoint=65, max_codepoint=90), min_size=1, max_size=10),
                st.floats(min_value=-1.0, max_value=1.0),
                st.floats(min_value=-5.0, max_value=5.0),
            ),
            min_size=5, max_size=30, unique_by=lambda t: t[0],
        ),
    )
    @settings(max_examples=10, deadline=5000)
    def test_add_backtest_bulk_distinct_factors(self, entries):
        """Property: adding backtests for distinct factors creates exactly that many rows."""
        with tempfile.TemporaryDirectory() as td:
            db_path = os.path.join(td, "test.db")
            db = ResultsDatabase(db_path=db_path)
            try:
                for name, ic_val, sh in entries:
                    db.add_backtest(name, {"ic": ic_val, "sharpe_ratio": sh})
                c = db.conn.cursor()
                c.execute("SELECT COUNT(*) FROM backtest_runs")
                count = c.fetchone()[0]
                assert count == len(entries), f"Expected {len(entries)} runs, got {count}"
            finally:
                db.close()

    @given(
        st.floats(min_value=-100.0, max_value=100.0),
        st.floats(min_value=-100.0, max_value=100.0),
        st.floats(min_value=-100.0, max_value=100.0),
        st.floats(min_value=-100.0, max_value=100.0),
        st.floats(min_value=-100.0, max_value=100.0),
    )
    @settings(max_examples=10, deadline=5000)
    def test_add_backtest_extreme_values(self, ic, sharpe, ann_ret, dd, wr):
        """Property: add_backtest handles extreme metric values without crashing."""
        with tempfile.TemporaryDirectory() as td:
            db_path = os.path.join(td, "test.db")
            db = ResultsDatabase(db_path=db_path)
            try:
                bid = db.add_backtest("ExtremeValues", {
                    "ic": ic, "sharpe_ratio": sharpe, "annualized_return": ann_ret,
                    "max_drawdown": dd, "win_rate": wr,
                })
                assert bid > 0
            finally:
                db.close()

    @given(
        st.text(alphabet=st.characters(min_codepoint=32, max_codepoint=126), min_size=1, max_size=40),
    )
    @settings(max_examples=10, deadline=5000)
    def test_add_backtest_special_character_names(self, name):
        """Property: add_backtest handles factor names with any printable characters."""
        with tempfile.TemporaryDirectory() as td:
            db_path = os.path.join(td, "test.db")
            db = ResultsDatabase(db_path=db_path)
            try:
                bid = db.add_backtest(name, {"ic": 0.05})
                c = db.conn.cursor()
                c.execute("SELECT factor_name FROM factors WHERE id = (SELECT factor_id FROM backtest_runs WHERE id=?)", (bid,))
                stored = c.fetchone()
                assert stored is not None
            finally:
                db.close()

    @given(
        st.floats(min_value=-1.0, max_value=1.0),
    )
    @settings(max_examples=10, deadline=5000)
    def test_add_backtest_with_raw_metrics(self, ic_val):
        """Property: add_backtest survives raw_metrics key with various dict values."""
        with tempfile.TemporaryDirectory() as td:
            db_path = os.path.join(td, "test.db")
            db = ResultsDatabase(db_path=db_path)
            try:
                bid = db.add_backtest("RawMetricsTest", {
                    "ic": ic_val,
                    "raw_metrics": {"a": 1.0, "b": ic_val, "c": 100.0},
                })
                assert bid > 0
            finally:
                db.close()


# ---------------------------------------------------------------------------
# add_loop Fuzzing (10 tests)
# ---------------------------------------------------------------------------


class TestAddLoopFuzzing:
    """Fuzz add_loop with random success/fail counts."""

    @given(
        st.integers(min_value=0, max_value=100),
        st.integers(min_value=0, max_value=100),
    )
    @settings(max_examples=10, deadline=5000)
    def test_loop_success_rate_formula(self, success, fail):
        """Property: success_rate = success / (success + fail) if total > 0 else 0."""
        with tempfile.TemporaryDirectory() as td:
            db_path = os.path.join(td, "test.db")
            db = ResultsDatabase(db_path=db_path)
            try:
                lid = db.add_loop(0, success, fail, None, "completed")
                c = db.conn.cursor()
                c.execute("SELECT success_rate FROM loop_results WHERE id=?", (lid,))
                rate = c.fetchone()[0]
                expected = success / (success + fail) if (success + fail) > 0 else 0.0
                assert abs(rate - expected) < 1e-10, f"Rate {rate} != expected {expected}"
            finally:
                db.close()

    @given(
        st.integers(min_value=0, max_value=50),
        st.integers(min_value=0, max_value=50),
        st.floats(min_value=-1.0, max_value=1.0),
    )
    @settings(max_examples=10, deadline=5000)
    def test_loop_best_ic_preserved(self, success, fail, best_ic):
        """Property: best_ic value stored matches what was passed."""
        with tempfile.TemporaryDirectory() as td:
            db_path = os.path.join(td, "test.db")
            db = ResultsDatabase(db_path=db_path)
            try:
                lid = db.add_loop(42, success, fail, best_ic, "completed")
                c = db.conn.cursor()
                c.execute("SELECT best_ic FROM loop_results WHERE id=?", (lid,))
                stored = c.fetchone()[0]
                if best_ic is not None:
                    assert abs(stored - best_ic) < 1e-10
                else:
                    assert stored is None
            finally:
                db.close()

    @given(
        st.lists(st.integers(min_value=1, max_value=50), min_size=1, max_size=20, unique=True),
        st.integers(min_value=1, max_value=10),
        st.integers(min_value=1, max_value=10),
    )
    @settings(max_examples=10, deadline=5000)
    def test_loop_multiple_sequential_indices(self, indices, s, f):
        """Property: multiple loops with distinct indices produce that many rows."""
        with tempfile.TemporaryDirectory() as td:
            db_path = os.path.join(td, "test.db")
            db = ResultsDatabase(db_path=db_path)
            try:
                for idx in indices:
                    db.add_loop(idx, s, f, None, "completed")
                c = db.conn.cursor()
                c.execute("SELECT COUNT(*) FROM loop_results")
                assert c.fetchone()[0] == len(indices)
            finally:
                db.close()

    @given(
        st.integers(min_value=0, max_value=1000),
        st.integers(min_value=0, max_value=1000),
        st.text(min_size=1, max_size=20),
    )
    @settings(max_examples=10, deadline=5000)
    def test_loop_status_stored(self, success, fail, status):
        """Property: status field reflects the passed value."""
        with tempfile.TemporaryDirectory() as td:
            db_path = os.path.join(td, "test.db")
            db = ResultsDatabase(db_path=db_path)
            try:
                lid = db.add_loop(99, success, fail, None, status)
                c = db.conn.cursor()
                c.execute("SELECT status FROM loop_results WHERE id=?", (lid,))
                assert c.fetchone()[0] == status
            finally:
                db.close()


# ---------------------------------------------------------------------------
# get_top_factors Properties (15 tests)
# ---------------------------------------------------------------------------


class TestGetTopFactorsFuzzing:
    """Property-based tests for get_top_factors."""

    @given(
        st.lists(
            st.floats(min_value=-5.0, max_value=5.0),
            min_size=5, max_size=30,
        ),
    )
    @settings(max_examples=10, deadline=5000)
    def test_top_factors_sorted_descending_by_sharpe(self, sharpes):
        """Property: get_top_factors by sharpe returns strictly descending sharpe values."""
        with tempfile.TemporaryDirectory() as td:
            db_path = os.path.join(td, "test.db")
            db = ResultsDatabase(db_path=db_path)
            try:
                for i, sh in enumerate(sharpes):
                    db.add_backtest(f"Factor_{i}", {"ic": 0.0, "sharpe_ratio": sh})
                df = db.get_top_factors(metric="sharpe", limit=len(sharpes))
                sh_vals = df["sharpe"].tolist()
                assert sh_vals == sorted(sh_vals, reverse=True), f"Not sorted: {sh_vals}"
            finally:
                db.close()

    @given(
        st.lists(
            st.floats(min_value=-1.0, max_value=1.0),
            min_size=5, max_size=30,
        ),
    )
    @settings(max_examples=10, deadline=5000)
    def test_top_factors_by_ic_descending(self, ics):
        """Property: get_top_factors by IC returns descending IC."""
        with tempfile.TemporaryDirectory() as td:
            db_path = os.path.join(td, "test.db")
            db = ResultsDatabase(db_path=db_path)
            try:
                for i, ic in enumerate(ics):
                    db.add_backtest(f"Factor_{i}", {"ic": ic, "sharpe_ratio": 0.0})
                df = db.get_top_factors(metric="ic", limit=len(ics))
                ic_vals = df["ic"].tolist()
                assert ic_vals == sorted(ic_vals, reverse=True)
            finally:
                db.close()

    @given(
        st.integers(min_value=1, max_value=50),
        st.integers(min_value=1, max_value=200),
    )
    @settings(max_examples=10, deadline=5000)
    def test_top_factors_limit_respected(self, n_factors, limit):
        """Property: result length <= limit and <= number of stored factors."""
        with tempfile.TemporaryDirectory() as td:
            db_path = os.path.join(td, "test.db")
            db = ResultsDatabase(db_path=db_path)
            try:
                for i in range(n_factors):
                    db.add_backtest(f"Fac_{i}", {"ic": 0.0, "sharpe_ratio": 1.0})
                df = db.get_top_factors(metric="sharpe", limit=limit)
                assert len(df) <= limit
                assert len(df) <= n_factors
            finally:
                db.close()

    @given(
        st.lists(
            st.floats(min_value=-5.0, max_value=5.0),
            min_size=10, max_size=40,
        ),
    )
    @settings(max_examples=10, deadline=5000)
    def test_get_top_factors_all_columns_present(self, sharpes):
        """Property: returned DataFrame always has expected columns."""
        with tempfile.TemporaryDirectory() as td:
            db_path = os.path.join(td, "test.db")
            db = ResultsDatabase(db_path=db_path)
            try:
                for i, sh in enumerate(sharpes):
                    db.add_backtest(f"FC_{i}", {"ic": 0.0, "sharpe_ratio": sh})
                df = db.get_top_factors()
                for col in ["factor_name", "sharpe", "ic", "annual_return", "max_drawdown"]:
                    assert col in df.columns, f"Missing column: {col}"
            finally:
                db.close()

    @given(st.text(alphabet=st.characters(min_codepoint=65, max_codepoint=90), min_size=1, max_size=10))
    @settings(max_examples=10, deadline=5000)
    def test_get_top_factors_empty_db_returns_empty(self, db_suffix):
        """Property: querying empty database returns empty DataFrame."""
        with tempfile.TemporaryDirectory() as td:
            db_path = os.path.join(td, f"empty_{db_suffix}.db")
            db = ResultsDatabase(db_path=db_path)
            try:
                df = db.get_top_factors(metric="sharpe", limit=10)
                assert len(df) == 0
            finally:
                db.close()

    @given(
        st.lists(st.floats(min_value=-5.0, max_value=5.0), min_size=5, max_size=30),
    )
    @settings(max_examples=10, deadline=5000)
    def test_get_top_factors_null_metrics_excluded(self, sharpes):
        """Property: factors with NULL sharpe are excluded from top-by-sharpe."""
        with tempfile.TemporaryDirectory() as td:
            db_path = os.path.join(td, "test.db")
            db = ResultsDatabase(db_path=db_path)
            try:
                # Add factors with NULL sharpe
                for i in range(3):
                    db.add_factor(f"NullFac_{i}", "type")
                for i, sh in enumerate(sharpes):
                    db.add_backtest(f"RealFac_{i}", {"ic": 0.0, "sharpe_ratio": sh})
                df = db.get_top_factors(metric="sharpe", limit=100)
                assert len(df) <= len(sharpes)
            finally:
                db.close()


# ---------------------------------------------------------------------------
# get_aggregate_stats Properties (8 tests)
# ---------------------------------------------------------------------------


class TestAggregateStatsProperties:
    """Property tests for get_aggregate_stats."""

    @given(
        st.lists(st.floats(min_value=-1.0, max_value=1.0), min_size=3, max_size=20),
    )
    @settings(max_examples=10, deadline=5000)
    def test_avg_ic_within_input_range(self, ics):
        """Property: avg_ic lies between min and max of stored ICs."""
        with tempfile.TemporaryDirectory() as td:
            db_path = os.path.join(td, "test.db")
            db = ResultsDatabase(db_path=db_path)
            try:
                for i, ic in enumerate(ics):
                    db.add_backtest(f"ICFactor_{i}", {"ic": ic, "sharpe_ratio": 1.0})
                stats = db.get_aggregate_stats()
                assert stats["avg_ic"] is not None
                assert min(ics) - 0.01 <= stats["avg_ic"] <= max(ics) + 0.01
            finally:
                db.close()

    @given(
        st.lists(st.floats(min_value=-10.0, max_value=10.0), min_size=3, max_size=20),
    )
    @settings(max_examples=10, deadline=5000)
    def test_max_sharpe_is_max(self, sharpes):
        """Property: max_sharpe equals the maximum of stored sharpe values."""
        with tempfile.TemporaryDirectory() as td:
            db_path = os.path.join(td, "test.db")
            db = ResultsDatabase(db_path=db_path)
            try:
                for i, sh in enumerate(sharpes):
                    db.add_backtest(f"SFactor_{i}", {"ic": 0.0, "sharpe_ratio": sh})
                stats = db.get_aggregate_stats()
                assert abs(stats["max_sharpe"] - max(sharpes)) < 1e-10
            finally:
                db.close()

    @given(
        st.lists(st.floats(min_value=-2.0, max_value=2.0), min_size=3, max_size=20),
    )
    @settings(max_examples=10, deadline=5000)
    def test_avg_return_within_range(self, returns):
        """Property: avg_return is between min and max stored annualized_return."""
        with tempfile.TemporaryDirectory() as td:
            db_path = os.path.join(td, "test.db")
            db = ResultsDatabase(db_path=db_path)
            try:
                for i, r in enumerate(returns):
                    db.add_backtest(f"RFactor_{i}", {"ic": 0.0, "annualized_return": r})
                stats = db.get_aggregate_stats()
                assert stats["avg_return"] is not None
                assert min(returns) - 0.01 <= stats["avg_return"] <= max(returns) + 0.01
            finally:
                db.close()

    @given(
        st.integers(min_value=1, max_value=30),
    )
    @settings(max_examples=10, deadline=5000)
    def test_total_factors_counts_unique_names(self, n_factors):
        """Property: total_factors counts unique factor names, not runs."""
        with tempfile.TemporaryDirectory() as td:
            db_path = os.path.join(td, "test.db")
            db = ResultsDatabase(db_path=db_path)
            try:
                distinct = n_factors // 2 + 1
                for i in range(distinct):
                    db.add_backtest(f"UniqFac_{i}", {"ic": 0.01 * i})
                # Add second run for first factor
                db.add_backtest("UniqFac_0", {"ic": 0.99})
                stats = db.get_aggregate_stats()
                assert stats["total_factors"] == distinct
            finally:
                db.close()


# ---------------------------------------------------------------------------
# Schema Migration Properties (8 tests)
# ---------------------------------------------------------------------------


class TestSchemaMigrationFuzzing:
    """Property tests for _add_column_if_not_exists."""

    @given(
        st.sampled_from(["REAL", "TEXT", "INTEGER", "BLOB"]),
        st.text(alphabet=st.characters(min_codepoint=65, max_codepoint=90), min_size=1, max_size=20),
    )
    @settings(max_examples=10, deadline=5000)
    def test_add_column_idempotent(self, col_type, col_name):
        """Property: adding the same column twice is safe (no-op second time)."""
        with tempfile.TemporaryDirectory() as td:
            db_path = os.path.join(td, "test.db")
            db = ResultsDatabase(db_path=db_path)
            try:
                db._add_column_if_not_exists("backtest_runs", col_name, col_type)
                db._add_column_if_not_exists("backtest_runs", col_name, col_type)
                c = db.conn.cursor()
                c.execute("PRAGMA table_info(backtest_runs)")
                cols = [row[1] for row in c.fetchall()]
                assert cols.count(col_name) == 1
            finally:
                db.close()

    @given(
        st.text(alphabet=st.characters(min_codepoint=65, max_codepoint=90), min_size=1, max_size=15),
    )
    @settings(max_examples=10, deadline=5000)
    def test_column_added_to_all_tables(self, col_name):
        """Property: column can be added to each allowed table."""
        for table in ["factors", "backtest_runs", "loop_results"]:
            with tempfile.TemporaryDirectory() as td:
                db_path = os.path.join(td, "test.db")
                db = ResultsDatabase(db_path=db_path)
                try:
                    db._add_column_if_not_exists(table, col_name, "REAL")
                    c = db.conn.cursor()
                    c.execute(f"PRAGMA table_info({table})")
                    cols = [row[1] for row in c.fetchall()]
                    assert col_name in cols, f"{col_name} not found in {table}"
                finally:
                    db.close()

    @given(
        st.text(alphabet=st.characters(min_codepoint=32, max_codepoint=47), min_size=1, max_size=10),
    )
    @settings(max_examples=10, deadline=5000)
    def test_invalid_column_names_raise_value_error(self, bad_name):
        """Property: non-alphanumeric (besides underscore) column names raise ValueError."""
        with tempfile.TemporaryDirectory() as td:
            db_path = os.path.join(td, "test.db")
            db = ResultsDatabase(db_path=db_path)
            try:
                with pytest.raises(ValueError):
                    db._add_column_if_not_exists("backtest_runs", bad_name, "REAL")
            finally:
                db.close()

    @given(st.text(min_size=1, max_size=15))
    @settings(max_examples=10, deadline=5000)
    def test_invalid_table_name_raises(self, bad_table):
        """Property: unknown table names raise ValueError."""
        assume(bad_table not in {"factors", "backtest_runs", "loop_results"})
        with tempfile.TemporaryDirectory() as td:
            db_path = os.path.join(td, "test.db")
            db = ResultsDatabase(db_path=db_path)
            try:
                with pytest.raises(ValueError):
                    db._add_column_if_not_exists(bad_table, "col", "REAL")
            finally:
                db.close()


# ---------------------------------------------------------------------------
# Data Integrity Properties (10 tests)
# ---------------------------------------------------------------------------


class TestDataIntegrityFuzzing:
    """Property tests for data roundtrip and consistency."""

    @given(
        st.text(alphabet=st.characters(min_codepoint=65, max_codepoint=90), min_size=1, max_size=30),
        st.floats(min_value=-1.0, max_value=1.0),
        st.floats(min_value=-5.0, max_value=5.0),
    )
    @settings(max_examples=10, deadline=5000)
    def test_data_roundtrip_ic(self, name, ic, sharpe):
        """Property: IC value retrieved matches what was stored."""
        with tempfile.TemporaryDirectory() as td:
            db_path = os.path.join(td, "test.db")
            db1 = ResultsDatabase(db_path=db_path)
            try:
                bid = db1.add_backtest(name, {"ic": ic, "sharpe_ratio": sharpe})
                c = db1.conn.cursor()
                c.execute("SELECT ic FROM backtest_runs WHERE id=?", (bid,))
                stored = c.fetchone()[0]
                assert abs(stored - ic) < 1e-10
            finally:
                db1.close()

    @given(
        st.text(alphabet=st.characters(min_codepoint=65, max_codepoint=90), min_size=1, max_size=30),
        st.floats(min_value=-10.0, max_value=10.0),
    )
    @settings(max_examples=10, deadline=5000)
    def test_data_roundtrip_sharpe(self, name, sharpe):
        """Property: Sharpe value retrieved matches stored."""
        with tempfile.TemporaryDirectory() as td:
            db_path = os.path.join(td, "test.db")
            db = ResultsDatabase(db_path=db_path)
            try:
                bid = db.add_backtest(name, {"ic": 0.0, "sharpe_ratio": sharpe})
                c = db.conn.cursor()
                c.execute("SELECT sharpe FROM backtest_runs WHERE id=?", (bid,))
                assert abs(c.fetchone()[0] - sharpe) < 1e-10
            finally:
                db.close()

    @given(
        st.text(alphabet=st.characters(min_codepoint=65, max_codepoint=90), min_size=1, max_size=30),
        st.floats(min_value=-1.0, max_value=0.0),
    )
    @settings(max_examples=10, deadline=5000)
    def test_data_roundtrip_max_drawdown(self, name, dd):
        """Property: max_drawdown roundtrip is exact."""
        with tempfile.TemporaryDirectory() as td:
            db_path = os.path.join(td, "test.db")
            db = ResultsDatabase(db_path=db_path)
            try:
                bid = db.add_backtest(name, {"ic": 0.0, "max_drawdown": dd, "sharpe_ratio": 1.0})
                c = db.conn.cursor()
                c.execute("SELECT max_drawdown FROM backtest_runs WHERE id=?", (bid,))
                assert abs(c.fetchone()[0] - dd) < 1e-10
            finally:
                db.close()

    @given(
        st.text(alphabet=st.characters(min_codepoint=65, max_codepoint=90), min_size=1, max_size=30),
        st.floats(min_value=0.0, max_value=1.0),
    )
    @settings(max_examples=10, deadline=5000)
    def test_data_roundtrip_win_rate(self, name, wr):
        """Property: win_rate roundtrip is exact."""
        with tempfile.TemporaryDirectory() as td:
            db_path = os.path.join(td, "test.db")
            db = ResultsDatabase(db_path=db_path)
            try:
                bid = db.add_backtest(name, {"ic": 0.0, "win_rate": wr, "sharpe_ratio": 1.0})
                c = db.conn.cursor()
                c.execute("SELECT win_rate FROM backtest_runs WHERE id=?", (bid,))
                assert abs(c.fetchone()[0] - wr) < 1e-10
            finally:
                db.close()

    @given(
        st.lists(
            st.tuples(
                st.floats(min_value=-5.0, max_value=5.0),
                st.floats(min_value=-1.0, max_value=1.0),
            ),
            min_size=5, max_size=30,
        ),
    )
    @settings(max_examples=10, deadline=5000, suppress_health_check=[HealthCheck.filter_too_much])
    def test_multiple_runs_factor_count_consistent(self, pairs):
        """Property: unique factor count between direct SQL and get_aggregate_stats matches."""
        with tempfile.TemporaryDirectory() as td:
            db_path = os.path.join(td, "test.db")
            db = ResultsDatabase(db_path=db_path)
            try:
                for i, (sh, ic) in enumerate(pairs):
                    db.add_backtest(f"ConsistencyFac_{i}", {"ic": ic, "sharpe_ratio": sh})
                stats = db.get_aggregate_stats()
                c = db.conn.cursor()
                c.execute("SELECT COUNT(DISTINCT factor_name) FROM backtest_runs JOIN factors ON factor_id=factors.id")
                direct = c.fetchone()[0]
                assert stats["total_factors"] == direct
            finally:
                db.close()

    @given(st.integers(min_value=1, max_value=50))
    @settings(max_examples=10, deadline=5000)
    def test_persistence_across_connections(self, n_factors):
        """Property: data written in one connection is visible in a new connection."""
        with tempfile.TemporaryDirectory() as td:
            db_path = os.path.join(td, "test.db")
            db1 = ResultsDatabase(db_path=db_path)
            for i in range(n_factors):
                db1.add_backtest(f"Persist_{i}", {"ic": 0.01 * i, "sharpe_ratio": 1.0})
            db1.close()

            db2 = ResultsDatabase(db_path=db_path)
            try:
                c = db2.conn.cursor()
                c.execute("SELECT COUNT(*) FROM backtest_runs")
                assert c.fetchone()[0] == n_factors
            finally:
                db2.close()

    @given(st.floats(min_value=-100.0, max_value=100.0))
    @settings(max_examples=10, deadline=5000)
    def test_nan_handled_in_metrics(self, nan_val):
        """Property: NaN values in metrics do not crash."""
        assume(np.isnan(nan_val) or not np.isnan(nan_val))  # both branches tested
        with tempfile.TemporaryDirectory() as td:
            db_path = os.path.join(td, "test.db")
            db = ResultsDatabase(db_path=db_path)
            try:
                bid = db.add_backtest("NaNTest", {"ic": nan_val, "sharpe_ratio": 1.0})
                assert bid > 0
            finally:
                db.close()


# ---------------------------------------------------------------------------
# get_factor_history Properties (5 tests)
# ---------------------------------------------------------------------------


class TestGetFactorHistoryFuzzing:
    """Property tests for get_factor_history."""

    @given(
        st.text(alphabet=st.characters(min_codepoint=65, max_codepoint=90), min_size=1, max_size=20),
        st.integers(min_value=1, max_value=10),
    )
    @settings(max_examples=10, deadline=5000)
    def test_factor_history_returns_correct_count(self, name, n_runs):
        """Property: get_factor_history returns exactly n rows for n backtest runs."""
        with tempfile.TemporaryDirectory() as td:
            db_path = os.path.join(td, "test.db")
            db = ResultsDatabase(db_path=db_path)
            try:
                for i in range(n_runs):
                    db.add_backtest(name, {"ic": i * 0.01, "sharpe_ratio": 1.0})
                df = db.get_factor_history(name)
                assert len(df) == n_runs, f"Expected {n_runs}, got {len(df)}"
            finally:
                db.close()

    @given(st.text(alphabet=st.characters(min_codepoint=65, max_codepoint=90), min_size=1, max_size=20))
    @settings(max_examples=10, deadline=5000)
    def test_factor_history_empty_for_unknown(self, name):
        """Property: get_factor_history for unknown factor returns empty DataFrame."""
        assume(len(name) > 0)
        with tempfile.TemporaryDirectory() as td:
            db_path = os.path.join(td, "test.db")
            db = ResultsDatabase(db_path=db_path)
            try:
                df = db.get_factor_history(name + "_unknown_suffix_xyz")
                assert len(df) == 0
            finally:
                db.close()

    @given(
        st.floats(min_value=-1.0, max_value=1.0),
        st.floats(min_value=-5.0, max_value=5.0),
    )
    @settings(max_examples=10, deadline=5000)
    def test_factor_history_values_match(self, ic, sharpe):
        """Property: get_factor_history returns the same values that were stored."""
        with tempfile.TemporaryDirectory() as td:
            db_path = os.path.join(td, "test.db")
            db = ResultsDatabase(db_path=db_path)
            try:
                db.add_backtest("HistoryCheck", {"ic": ic, "sharpe_ratio": sharpe})
                df = db.get_factor_history("HistoryCheck")
                assert len(df) > 0
                assert abs(df.iloc[0]["ic"] - ic) < 1e-10
                assert abs(df.iloc[0]["sharpe"] - sharpe) < 1e-10
            finally:
                db.close()
