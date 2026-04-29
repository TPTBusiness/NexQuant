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
        c.execute("SELECT name FROM sqlite_master WHERE type='table'")  # nosec
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
        c.execute("SELECT COUNT(*) FROM factors")  # nosec
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
        c.execute("SELECT COUNT(*) FROM factors WHERE factor_name = ?", ("NewFactor",))  # nosec
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
        c.execute("SELECT COUNT(*) FROM backtest_runs")  # nosec
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
        c.execute("SELECT success_rate FROM loop_results WHERE loop_index = 1")  # nosec
        rate = c.fetchone()[0]
        
        assert abs(rate - 0.8) < 1e-10, f"Success Rate {rate} != erwartet 0.8"

    def test_add_loop_zero_total(self, results_database):
        """add_loop mit 0 total (success + fail = 0) sollte 0 rate ergeben"""
        loop_id = results_database.add_loop(1, 0, 0, None, "completed")
        
        c = results_database.conn.cursor()
        c.execute("SELECT success_rate FROM loop_results WHERE id = ?", (loop_id,))  # nosec
        rate = c.fetchone()[0]
        
        assert rate == 0, f"Success Rate sollte 0 sein bei 0 total, ist aber {rate}"

    def test_add_loop_multiple(self, results_database):
        """Mehrere Loops hinzufügen sollte funktionieren"""
        for i in range(10):
            results_database.add_loop(i, i % 5, 5 - (i % 5), 0.01 * i, "completed")
        
        c = results_database.conn.cursor()
        c.execute("SELECT COUNT(*) FROM loop_results")  # nosec
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
            c.execute("SELECT COUNT(*) FROM factors")  # nosec
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
        c.execute("""  # nosec
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
        c.execute("SELECT COUNT(*) FROM factors")  # nosec
        factor_count = c.fetchone()[0]
        
        c.execute("SELECT COUNT(*) FROM backtest_runs")  # nosec
        backtest_count = c.fetchone()[0]
        
        assert factor_count == 1, "Faktor nicht persistent"
        assert backtest_count == 1, "Backtest nicht persistent"
        
        db2.close()


# Import am Anfang der Datei für die Tests
from rdagent.components.backtesting.results_db import ResultsDatabase
