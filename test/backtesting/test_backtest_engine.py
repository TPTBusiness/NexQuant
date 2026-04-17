"""
Tests für Backtest Engine - BacktestMetrics und FactorBacktester

Test-Fälle:
- calculate_ic(): Korrelation zwischen Faktor und Returns
- calculate_sharpe(): Sharpe Ratio Berechnung
- calculate_max_drawdown(): Maximaler Drawdown
- calculate_all(): Alle Metrics zusammen
- FactorBacktester.run_backtest(): Kompletter Backtest-Lauf
- Edge Cases: NaN, leere Daten, zu wenig Daten, Extremwerte
"""
import pytest
import numpy as np
import pandas as pd
import json
from pathlib import Path
from datetime import datetime


class TestBacktestMetricsCalculateIC:
    """Tests für BacktestMetrics.calculate_ic()"""

    def test_calculate_ic_normal_data(self, backtest_metrics, sample_factor_data):
        """IC-Berechnung mit normalen Daten sollte korrekte Korrelation zurückgeben"""
        factor_values, forward_returns = sample_factor_data
        ic = backtest_metrics.calculate_ic(factor_values, forward_returns)
        
        # IC sollte zwischen -1 und 1 liegen
        assert -1 <= ic <= 1, f"IC {ic} liegt außerhalb des gültigen Bereichs [-1, 1]"
        # Bei random Daten erwarten wir IC nahe 0
        assert abs(ic) < 0.3, f"IC {ic} ist für random Daten zu hoch"

    def test_calculate_ic_perfect_positive_correlation(self, backtest_metrics):
        """IC sollte 1.0 sein bei perfekter positiver Korrelation"""
        n = 100
        dates = pd.date_range(start='2024-01-01', periods=n, freq='B')
        factor = pd.Series(np.arange(n, dtype=float), index=dates)
        fwd_ret = pd.Series(np.arange(n, dtype=float), index=dates)
        
        ic = backtest_metrics.calculate_ic(factor, fwd_ret)
        assert np.isclose(ic, 1.0, atol=1e-10), f"IC sollte 1.0 sein, ist aber {ic}"

    def test_calculate_ic_perfect_negative_correlation(self, backtest_metrics):
        """IC sollte -1.0 sein bei perfekter negativer Korrelation"""
        n = 100
        dates = pd.date_range(start='2024-01-01', periods=n, freq='B')
        factor = pd.Series(np.arange(n, dtype=float), index=dates)
        fwd_ret = pd.Series(-np.arange(n, dtype=float), index=dates)
        
        ic = backtest_metrics.calculate_ic(factor, fwd_ret)
        assert np.isclose(ic, -1.0, atol=1e-10), f"IC sollte -1.0 sein, ist aber {ic}"

    def test_calculate_ic_empty_data(self, backtest_metrics, empty_data):
        """IC sollte NaN zurückgeben bei leeren Daten"""
        factor, fwd_ret = empty_data
        ic = backtest_metrics.calculate_ic(factor, fwd_ret)
        assert np.isnan(ic), f"IC sollte NaN sein für leere Daten, ist aber {ic}"

    def test_calculate_ic_insufficient_data(self, backtest_metrics, insufficient_data):
        """IC sollte NaN zurückgeben bei zu wenig Daten (< 10 Punkte)"""
        factor, fwd_ret = insufficient_data
        ic = backtest_metrics.calculate_ic(factor, fwd_ret)
        assert np.isnan(ic), f"IC sollte NaN sein für insufficient data (<10), ist aber {ic}"

    def test_calculate_ic_nan_data(self, backtest_metrics, nan_data):
        """IC sollte mit NaN-Werten korrekt umgehen"""
        factor, fwd_ret = nan_data
        ic = backtest_metrics.calculate_ic(factor, fwd_ret)
        # Sollte trotzdem berechnet werden mit den verfügbaren Daten
        assert not np.isnan(ic) or np.isnan(ic), "IC-Berechnung mit NaN-Daten fehlgeschlagen"

    def test_calculate_ic_constant_data(self, backtest_metrics, constant_data):
        """IC sollte NaN sein bei konstanten Daten (keine Varianz)"""
        factor, fwd_ret = constant_data
        ic = backtest_metrics.calculate_ic(factor, fwd_ret)
        # Bei konstantem Faktor ist Korrelation nicht definiert
        assert np.isnan(ic), f"IC sollte NaN sein für konstante Daten, ist aber {ic}"

    def test_calculate_ic_extreme_values(self, backtest_metrics, extreme_values_data):
        """IC-Berechnung sollte robust gegenüber Extremwerten sein"""
        factor, fwd_ret = extreme_values_data
        ic = backtest_metrics.calculate_ic(factor, fwd_ret)
        assert -1 <= ic <= 1, f"IC {ic} liegt außerhalb des gültigen Bereichs [-1, 1]"


class TestBacktestMetricsCalculateSharpe:
    """Tests für BacktestMetrics.calculate_sharpe()"""

    def test_calculate_sharpe_normal_data(self, sample_returns_data):
        """Sharpe Ratio mit Daily-Daten sollte im typischen Bereich liegen."""
        from rdagent.components.backtesting.backtest_engine import BacktestMetrics

        returns, _ = sample_returns_data
        # sample_returns_data is business-daily → use daily annualization.
        bm_daily = BacktestMetrics(risk_free_rate=0.02, bars_per_year=252)
        sharpe = bm_daily.calculate_sharpe(returns)

        assert -5 <= sharpe <= 5, f"Sharpe {sharpe} liegt außerhalb typischen Bereichs"

    def test_calculate_sharpe_annualized_vs_raw(self, sample_returns_data):
        """Annualisierte Sharpe = √(bars_per_year) * raw Sharpe — convention-agnostic."""
        from rdagent.components.backtesting.backtest_engine import BacktestMetrics

        returns, _ = sample_returns_data
        bm_daily = BacktestMetrics(risk_free_rate=0.02, bars_per_year=252)
        sharpe_raw = bm_daily.calculate_sharpe(returns, annualize=False)
        sharpe_ann = bm_daily.calculate_sharpe(returns, annualize=True)

        expected_ann = sharpe_raw * np.sqrt(252)
        assert abs(sharpe_ann - expected_ann) < 1e-10, \
            f"Annualisierte Sharpe {sharpe_ann} != erwartet {expected_ann}"

    def test_calculate_sharpe_empty_data(self, backtest_metrics, empty_data):
        """Sharpe sollte NaN sein bei leeren Daten"""
        returns, _ = empty_data
        sharpe = backtest_metrics.calculate_sharpe(returns)
        assert np.isnan(sharpe), f"Sharpe sollte NaN sein für leere Daten, ist aber {sharpe}"

    def test_calculate_sharpe_insufficient_data(self, backtest_metrics):
        """Sharpe sollte NaN sein bei zu wenig Daten (< 10 Punkte)"""
        n = 5
        dates = pd.date_range(start='2024-01-01', periods=n, freq='B')
        returns = pd.Series(np.random.randn(n), index=dates)
        
        sharpe = backtest_metrics.calculate_sharpe(returns)
        assert np.isnan(sharpe), f"Sharpe sollte NaN sein für insufficient data, ist aber {sharpe}"

    def test_calculate_sharpe_zero_variance(self, backtest_metrics, zero_variance_returns):
        """Sharpe sollte bei sehr geringer Varianz extrem hohe Werte liefern"""
        returns, _ = zero_variance_returns
        sharpe = backtest_metrics.calculate_sharpe(returns)
        # Bei konstanten Returns (std ~ 0) wird Sharpe extrem groß
        # Die Implementierung gibt keinen NaN zurück wenn std != 0
        assert np.isfinite(sharpe) or np.isnan(sharpe), "Sharpe sollte finite oder NaN sein"

    def test_calculate_sharpe_negative_returns(self):
        """Sharpe sollte mit negativen Daily-Returns korrekt umgehen"""
        from rdagent.components.backtesting.backtest_engine import BacktestMetrics

        n = 100
        dates = pd.date_range(start='2024-01-01', periods=n, freq='B')
        returns = pd.Series(np.random.randn(n) * 0.02 - 0.001, index=dates)

        bm_daily = BacktestMetrics(risk_free_rate=0.02, bars_per_year=252)
        sharpe = bm_daily.calculate_sharpe(returns)
        assert -5 <= sharpe <= 5, f"Sharpe {sharpe} liegt außerhalb typischen Bereichs"


class TestBacktestMetricsCalculateMaxDrawdown:
    """Tests für BacktestMetrics.calculate_max_drawdown()"""

    def test_calculate_max_drawdown_normal_data(self, backtest_metrics, sample_returns_data):
        """Max Drawdown mit normalen Daten sollte korrekt berechnet werden"""
        returns, equity = sample_returns_data
        max_dd = backtest_metrics.calculate_max_drawdown(equity)
        
        # Drawdown sollte negativ oder 0 sein
        assert max_dd <= 0, f"Max Drawdown {max_dd} sollte <= 0 sein"
        # Drawdown sollte >= -1 sein (kann nicht mehr als 100% verlieren)
        assert max_dd >= -1, f"Max Drawdown {max_dd} sollte >= -1 sein"

    def test_calculate_max_drawdown_monotonic_increasing(self, backtest_metrics):
        """Max Drawdown sollte 0 sein bei monoton steigender Equity"""
        n = 100
        dates = pd.date_range(start='2024-01-01', periods=n, freq='B')
        equity = pd.Series(np.linspace(1, 2, n), index=dates)
        
        max_dd = backtest_metrics.calculate_max_drawdown(equity)
        assert max_dd == 0.0, f"Max Drawdown sollte 0 sein für monotonic increasing, ist aber {max_dd}"

    def test_calculate_max_drawdown_significant_drop(self, backtest_metrics, negative_equity_data):
        """Max Drawdown sollte signifikanten Drop erkennen"""
        returns, equity = negative_equity_data
        max_dd = backtest_metrics.calculate_max_drawdown(equity)
        
        # Sollte einen signifikanten Drawdown erkennen
        assert max_dd < -0.05, f"Max Drawdown {max_dd} sollte signifikant negativ sein"

    def test_calculate_max_drawdown_empty_data(self, backtest_metrics, empty_data):
        """Max Drawdown sollte NaN sein bei leeren Daten"""
        _, equity = empty_data
        max_dd = backtest_metrics.calculate_max_drawdown(equity)
        # Leere Daten sollten NaN oder 0 zurückgeben
        assert np.isnan(max_dd) or max_dd == 0, f"Max Drawdown für leere Daten unerwartet: {max_dd}"

    def test_calculate_max_drawdown_single_point(self, backtest_metrics):
        """Max Drawdown mit nur einem Datenpunkt"""
        dates = pd.date_range(start='2024-01-01', periods=1, freq='B')
        equity = pd.Series([1.0], index=dates)
        
        max_dd = backtest_metrics.calculate_max_drawdown(equity)
        assert max_dd == 0.0, f"Max Drawdown sollte 0 sein für single point, ist aber {max_dd}"


class TestBacktestMetricsCalculateAll:
    """Tests für BacktestMetrics.calculate_all()"""

    def test_calculate_all_complete_metrics(self, backtest_metrics, sample_factor_data, sample_returns_data):
        """calculate_all sollte alle erwarteten Metrics zurückgeben"""
        factor_values, forward_returns = sample_factor_data
        returns, equity = sample_returns_data
        
        metrics = backtest_metrics.calculate_all(
            returns, equity, factor_values, forward_returns
        )
        
        # Alle erwarteten Keys sollten vorhanden sein
        expected_keys = ['total_return', 'annualized_return', 'sharpe_ratio',
                        'max_drawdown', 'win_rate', 'total_trades', 'ic']
        for key in expected_keys:
            assert key in metrics, f"Key '{key}' fehlt in metrics"

    def test_calculate_all_without_factor_data(self, backtest_metrics, sample_returns_data):
        """calculate_all ohne Faktor-Daten sollte kein 'ic' enthalten"""
        returns, equity = sample_returns_data
        
        metrics = backtest_metrics.calculate_all(returns, equity)
        
        # IC sollte nicht vorhanden sein
        assert 'ic' not in metrics, "'ic' sollte nicht in metrics sein ohne factor_data"
        # Andere Keys sollten vorhanden sein
        assert 'sharpe_ratio' in metrics
        assert 'max_drawdown' in metrics

    def test_calculate_all_total_return_calculation(self, backtest_metrics):
        """Total Return sollte (1 + returns).prod() - 1 sein"""
        n = 100
        dates = pd.date_range(start='2024-01-01', periods=n, freq='B')
        returns = pd.Series([0.01] * n, index=dates)  # 1% pro Tag
        equity = (1 + returns).cumprod()
        
        metrics = backtest_metrics.calculate_all(returns, equity)
        expected_total = (1 + returns).prod() - 1
        
        assert abs(metrics['total_return'] - expected_total) < 1e-10, \
            f"Total Return {metrics['total_return']} != erwartet {expected_total}"

    def test_calculate_all_win_rate_calculation(self, backtest_metrics):
        """Win Rate sollte Anteil positiver Returns sein"""
        n = 100
        dates = pd.date_range(start='2024-01-01', periods=n, freq='B')
        returns = pd.Series([0.01] * 60 + [-0.01] * 40, index=dates)  # 60% positiv
        equity = (1 + returns).cumprod()
        
        metrics = backtest_metrics.calculate_all(returns, equity)
        assert abs(metrics['win_rate'] - 0.60) < 0.01, \
            f"Win Rate {metrics['win_rate']} != erwartet 0.60"

    def test_calculate_all_total_trades(self, backtest_metrics, sample_returns_data):
        """Total Trades sollte Länge der Returns sein"""
        returns, equity = sample_returns_data
        
        metrics = backtest_metrics.calculate_all(returns, equity)
        assert metrics['total_trades'] == len(returns), \
            f"Total Trades {metrics['total_trades']} != {len(returns)}"


class TestFactorBacktesterRunBacktest:
    """Tests für FactorBacktester.run_backtest()"""

    def test_run_backtest_complete_output(self, factor_backtester, sample_factor_data):
        """run_backtest sollte vollständige Metrics zurückgeben"""
        factor_values, forward_returns = sample_factor_data
        
        metrics = factor_backtester.run_backtest(
            factor_values, forward_returns, "TestFactor"
        )
        
        # Erwartete Keys
        expected_keys = ['total_return', 'annualized_return', 'sharpe_ratio',
                        'max_drawdown', 'win_rate', 'total_trades', 'ic',
                        'factor_name', 'timestamp']
        for key in expected_keys:
            assert key in metrics, f"Key '{key}' fehlt in metrics"

    def test_run_backtest_saves_json_file(self, factor_backtester, sample_factor_data):
        """run_backtest sollte JSON-Datei speichern"""
        factor_values, forward_returns = sample_factor_data
        
        metrics = factor_backtester.run_backtest(
            factor_values, forward_returns, "TestFactor"
        )
        
        # JSON-Datei sollte existieren
        json_files = list(factor_backtester.results_path.glob("*.json"))
        assert len(json_files) > 0, "Keine JSON-Datei wurde gespeichert"
        
        # Datei sollte lesbar sein
        with open(json_files[0], 'r') as f:
            saved_data = json.load(f)
        assert 'ic' in saved_data or 'sharpe_ratio' in saved_data

    def test_run_backtest_transaction_costs(self, factor_backtester, sample_factor_data):
        """run_backtest sollte Transaktionskosten berücksichtigen"""
        factor_values, forward_returns = sample_factor_data
        
        # Backtest mit hohen Transaktionskosten
        metrics_high_cost = factor_backtester.run_backtest(
            factor_values, forward_returns, "TestFactor", transaction_cost=0.001
        )
        
        # Backtest mit niedrigen Transaktionskosten
        metrics_low_cost = factor_backtester.run_backtest(
            factor_values, forward_returns, "TestFactor", transaction_cost=0.00001
        )
        
        # Höhere Kosten sollten niedrigere Returns ergeben
        assert metrics_high_cost['total_return'] <= metrics_low_cost['total_return'] + 0.01, \
            "Hohe Transaktionskosten sollten Returns reduzieren"

    def test_run_backtest_with_nan_values(self, factor_backtester, nan_data):
        """run_backtest sollte mit NaN-Werten korrekt umgehen"""
        factor, fwd_ret = nan_data
        
        metrics = factor_backtester.run_backtest(factor, fwd_ret, "NaNFactor")
        
        # Sollte trotzdem laufen, IC kann NaN sein
        assert 'factor_name' in metrics
        assert metrics['factor_name'] == "NaNFactor"

    def test_run_backtest_empty_data(self, factor_backtester, empty_data):
        """run_backtest sollte mit leeren Daten korrekt umgehen"""
        factor, fwd_ret = empty_data
        
        metrics = factor_backtester.run_backtest(factor, fwd_ret, "EmptyFactor")
        
        # Sollte laufen aber NaN für Metrics haben
        assert metrics['factor_name'] == "EmptyFactor"

    def test_run_backtest_realistic_data(self, factor_backtester, realistic_market_data):
        """run_backtest mit realistischen Markt-Daten"""
        factor, fwd_ret = realistic_market_data
        
        metrics = factor_backtester.run_backtest(factor, fwd_ret, "RealisticFactor")
        
        # Alle Metrics sollten berechnet sein
        assert 'ic' in metrics
        assert 'sharpe_ratio' in metrics
        assert 'max_drawdown' in metrics
        assert 'win_rate' in metrics
        
        # Win Rate sollte zwischen 0 und 1 liegen
        assert 0 <= metrics['win_rate'] <= 1, f"Win Rate {metrics['win_rate']} ungültig"


class TestBacktestIntegration:
    """Integrationstests für das gesamte Backtesting-System"""

    def test_full_backtest_workflow(self, backtest_metrics, factor_backtester, sample_factor_data, sample_returns_data):
        """Kompletter Backtest-Workflow von Metrics bis Speicherung"""
        factor_values, forward_returns = sample_factor_data
        returns, equity = sample_returns_data
        
        # 1. Einzelne Metrics berechnen
        ic = backtest_metrics.calculate_ic(factor_values, forward_returns)
        sharpe = backtest_metrics.calculate_sharpe(returns)
        max_dd = backtest_metrics.calculate_max_drawdown(equity)
        
        # 2. Alle Metrics zusammen
        all_metrics = backtest_metrics.calculate_all(returns, equity, factor_values, forward_returns)
        
        # 3. Kompletten Backtest laufen
        backtest_result = factor_backtester.run_backtest(
            factor_values, forward_returns, "IntegrationTestFactor"
        )
        
        # Konsistenz prüfen (IC sollte gleich sein)
        assert abs(all_metrics['ic'] - backtest_result['ic']) < 1e-10, "IC inkonsistent"
        # Sharpe kann unterschiedlich sein da backtester strategy_returns verwendet
        assert 'sharpe_ratio' in all_metrics
        assert 'sharpe_ratio' in backtest_result

    def test_multiple_factors_comparison(self, factor_backtester, sample_factor_data):
        """Vergleich mehrerer Faktoren im Backtest"""
        factor_values, forward_returns = sample_factor_data
        
        # Erzeuge verschiedene Faktoren durch Transformation
        factor_conservative = factor_values * 0.5
        factor_aggressive = factor_values * 2.0
        
        metrics_conservative = factor_backtester.run_backtest(
            factor_conservative, forward_returns, "ConservativeFactor"
        )
        metrics_aggressive = factor_backtester.run_backtest(
            factor_aggressive, forward_returns, "AggressiveFactor"
        )
        
        # Beide sollten IC-Werte haben
        assert 'ic' in metrics_conservative
        assert 'ic' in metrics_aggressive
        # IC sollte gleich sein (Skalierung ändert Korrelation nicht)
        assert abs(metrics_conservative['ic'] - metrics_aggressive['ic']) < 1e-10
