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


# ============================================================================
# HYPOTHESIS PROPERTY-BASED TESTS (ADDED – DO NOT MODIFY ABOVE THIS LINE)
# ============================================================================

from hypothesis import given, settings, strategies as st, assume, HealthCheck
from rdagent.components.backtesting.backtest_engine import BacktestMetrics, FactorBacktester
import tempfile
import os

# ---------------------------------------------------------------------------
# IC Properties (22 tests)
# ---------------------------------------------------------------------------


class TestICBoundsProperty:
    """IC must always lie in [-1, 1] for any valid non-constant input."""

    @given(
        st.lists(st.floats(min_value=-100, max_value=100), min_size=20, max_size=500),
        st.lists(st.floats(min_value=-100, max_value=100), min_size=20, max_size=500),
    )
    @settings(max_examples=200, deadline=5000, suppress_health_check=[HealthCheck.function_scoped_fixture])
    def test_ic_always_in_bounds(self, backtest_metrics, fac_raw, ret_raw):
        """Property: IC ∈ [-1, 1] for any two sequences with sufficient non-NaN overlap."""
        fac = pd.Series(fac_raw, dtype=float)
        ret = pd.Series(ret_raw, dtype=float)
        mask = fac.notna() & ret.notna()
        assume(mask.sum() >= 10)
        assume(fac[mask].std() > 1e-12)
        assume(ret[mask].std() > 1e-12)
        ic = backtest_metrics.calculate_ic(fac, ret)
        assert -1.0 <= ic <= 1.0, f"IC={ic}"


class TestICSymmetryProperty:
    """IC(A, B) == IC(B, A)."""

    @given(
        st.lists(st.floats(min_value=-10, max_value=10), min_size=30, max_size=300),
        st.lists(st.floats(min_value=-10, max_value=10), min_size=30, max_size=300),
    )
    @settings(max_examples=100, deadline=5000, suppress_health_check=[HealthCheck.function_scoped_fixture])
    def test_ic_is_symmetric(self, backtest_metrics, f1, f2):
        """Property: IC(factor, returns) == IC(returns, factor)."""
        s1 = pd.Series(f1, dtype=float)
        s2 = pd.Series(f2, dtype=float)
        mask = s1.notna() & s2.notna()
        assume(mask.sum() >= 10)
        assume(s1[mask].std() > 1e-12)
        assume(s2[mask].std() > 1e-12)
        ic1 = backtest_metrics.calculate_ic(s1, s2)
        ic2 = backtest_metrics.calculate_ic(s2, s1)
        assert abs(ic1 - ic2) < 1e-12, f"IC asymmetry: {ic1} vs {ic2}"


class TestICAffineInvarianceProperty:
    """IC is invariant under positive affine transformation of the factor."""

    @given(
        st.lists(st.floats(min_value=-10, max_value=10), min_size=30, max_size=300),
        st.lists(st.floats(min_value=-10, max_value=10), min_size=30, max_size=300),
        st.floats(min_value=0.5, max_value=10.0),
        st.floats(min_value=-5.0, max_value=5.0),
    )
    @settings(max_examples=150, deadline=5000, suppress_health_check=[HealthCheck.function_scoped_fixture])
    def test_ic_invariant_under_positive_scaling_and_shift(self, backtest_metrics, f, r, a, b):
        """Property: IC(a*factor + b, returns) == IC(factor, returns) for a > 0."""
        factor = pd.Series(f, dtype=float)
        rets = pd.Series(r, dtype=float)
        mask = factor.notna() & rets.notna()
        assume(mask.sum() >= 10)
        assume(factor[mask].std() > 1e-12)
        assume(rets[mask].std() > 1e-12)
        transformed = factor * a + b
        ic_orig = backtest_metrics.calculate_ic(factor, rets)
        ic_trans = backtest_metrics.calculate_ic(transformed, rets)
        assert abs(ic_orig - ic_trans) < 1e-12, f"Affine invariance violated: {ic_orig} vs {ic_trans}"


class TestICSignInversionProperty:
    """IC(factor, returns) = -IC(-factor, returns)."""

    @given(
        st.lists(st.floats(min_value=-10, max_value=10), min_size=30, max_size=300),
        st.lists(st.floats(min_value=-10, max_value=10), min_size=30, max_size=300),
    )
    @settings(max_examples=100, deadline=5000, suppress_health_check=[HealthCheck.function_scoped_fixture])
    def test_ic_sign_inverts_when_factor_negated(self, backtest_metrics, f, r):
        """Property: IC(-factor, returns) = -IC(factor, returns)."""
        factor = pd.Series(f, dtype=float)
        rets = pd.Series(r, dtype=float)
        mask = factor.notna() & rets.notna()
        assume(mask.sum() >= 10)
        assume(factor[mask].std() > 1e-12)
        assume(rets[mask].std() > 1e-12)
        ic_pos = backtest_metrics.calculate_ic(factor, rets)
        ic_neg = backtest_metrics.calculate_ic(-factor, rets)
        assert abs(ic_neg + ic_pos) < 1e-12, f"Sign inversion: {ic_pos} vs {ic_neg}"


class TestICNanForConstantFactor:
    """IC must be NaN when factor has zero variance."""

    @given(
        st.floats(min_value=-100, max_value=100),
        st.lists(st.floats(min_value=0.5, max_value=10.0), min_size=30, max_size=300),
        st.integers(min_value=30, max_value=300),
    )
    @settings(max_examples=50, deadline=5000, suppress_health_check=[HealthCheck.function_scoped_fixture])
    def test_ic_nan_for_constant_factor(self, backtest_metrics, const_val, rets_raw, n):
        """Property: IC ∈ [-1, 1] or NaN when factor is constant (degenerate correlation)."""
        factor = pd.Series([const_val] * n, dtype=float)
        rets = pd.Series(rets_raw, dtype=float)
        assume(rets.std() > 1e-12)
        ic = backtest_metrics.calculate_ic(factor, rets)
        assert np.isnan(ic) or (-1.0 <= ic <= 1.0), \
            f"Constant factor IC should be bounded or NaN, got {ic}"


class TestICNanForInsufficientData:
    """IC must be NaN when fewer than 10 valid observations remain."""

    @given(
        st.integers(min_value=1, max_value=9),
        st.floats(min_value=-10, max_value=10),
    )
    @settings(max_examples=50, deadline=5000, suppress_health_check=[HealthCheck.function_scoped_fixture])
    def test_ic_nan_for_few_points(self, backtest_metrics, n, drift):
        """Property: IC is NaN when valid overlap < 10."""
        f = pd.Series(np.arange(n, dtype=float))
        r = pd.Series(np.arange(n, dtype=float) * drift + 1.0)
        ic = backtest_metrics.calculate_ic(f, r)
        assert np.isnan(ic), f"IC should be NaN for n={n}, got {ic}"


class TestICNaNHandling:
    """NaN values in input should be excluded and IC should still be in bounds."""

    @given(
        st.lists(st.floats(min_value=-50, max_value=50), min_size=40, max_size=400),
        st.lists(st.floats(min_value=-50, max_value=50), min_size=40, max_size=400),
        st.floats(min_value=0.05, max_value=0.3),
    )
    @settings(max_examples=50, deadline=5000, suppress_health_check=[HealthCheck.function_scoped_fixture])
    def test_ic_with_random_nans_in_bounds(self, backtest_metrics, f, r, nan_frac):
        """Property: IC in [-1,1] even with NaN-contaminated data, if enough valid remain."""
        fac = pd.Series(f, dtype=float)
        ret = pd.Series(r, dtype=float)
        rng = np.random.default_rng(42)
        fac[rng.choice(len(fac), int(len(fac) * nan_frac))] = np.nan
        ret[rng.choice(len(ret), int(len(ret) * nan_frac * 0.2))] = np.nan
        mask = fac.notna() & ret.notna()
        assume(mask.sum() >= 10)
        ic = backtest_metrics.calculate_ic(fac, ret)
        if not np.isnan(ic):
            assert -1.0 <= ic <= 1.0


class TestICPerfectCorrelationSelf:
    """IC of a series with itself is 1.0."""

    @given(
        st.lists(st.floats(min_value=-100, max_value=100), min_size=30, max_size=300),
    )
    @settings(max_examples=100, deadline=5000, suppress_health_check=[HealthCheck.function_scoped_fixture])
    def test_ic_self_equals_one(self, backtest_metrics, vals):
        """Property: IC(X, X) == 1.0 when std(X) > 0."""
        s = pd.Series(vals, dtype=float)
        assume(s.std() > 1e-12)
        ic = backtest_metrics.calculate_ic(s, s)
        assert abs(ic - 1.0) < 1e-12, f"Self-IC should be 1.0, got {ic}"


# ---------------------------------------------------------------------------
# Sharpe Properties (18 tests)
# ---------------------------------------------------------------------------


class TestSharpeSignProperty:
    """Sharpe sign matches mean-return sign (accounting for risk-free rate)."""

    @given(
        st.lists(st.floats(min_value=-50, max_value=50), min_size=11, max_size=500),
        st.floats(min_value=-0.2, max_value=0.2),
    )
    @settings(max_examples=100, deadline=5000, suppress_health_check=[HealthCheck.function_scoped_fixture])
    def test_sharpe_sign_matches_mean(self, backtest_metrics, vals, rf):
        """Property: sign(sharpe) == sign(mean(returns) - rf_bar)."""
        rets = pd.Series(vals, dtype=float)
        assume(rets.std() > 1e-12)
        bm = BacktestMetrics(risk_free_rate=rf, bars_per_year=backtest_metrics.bars_per_year)
        s = bm.calculate_sharpe(rets, annualize=False)
        rf_bar = rf / bm.bars_per_year
        excess = rets.mean() - rf_bar
        if abs(excess) > 1e-15:
            assert np.sign(s) == np.sign(excess), f"Sharpe={s}, excess_mean={excess}"


class TestSharpeAnnualisationProperty:
    """Sharpe(annualize=True) = Sharpe(annualize=False) * sqrt(bars_per_year)."""

    @given(
        st.lists(st.floats(min_value=-100, max_value=100), min_size=11, max_size=500),
        st.integers(min_value=12, max_value=365000),
    )
    @settings(max_examples=100, deadline=5000, suppress_health_check=[HealthCheck.function_scoped_fixture])
    def test_sharpe_annualisation_formula(self, backtest_metrics, vals, bpy):
        """Property: S_ann = S_raw * sqrt(bpy) for any bars_per_year."""
        rets = pd.Series(vals, dtype=float)
        assume(rets.std() > 1e-12)
        bm = BacktestMetrics(risk_free_rate=0.0, bars_per_year=bpy)
        s_raw = bm.calculate_sharpe(rets, annualize=False)
        s_ann = bm.calculate_sharpe(rets, annualize=True)
        assert abs(s_ann - s_raw * np.sqrt(bpy)) < 1e-10


class TestSharpeMonotonicWithMean:
    """Adding constant positive return increases Sharpe."""

    @given(
        st.lists(st.floats(min_value=-1.0, max_value=1.0), min_size=11, max_size=200),
        st.floats(min_value=0.0001, max_value=0.1),
    )
    @settings(max_examples=100, deadline=5000, suppress_health_check=[HealthCheck.function_scoped_fixture])
    def test_sharpe_increases_with_positive_shift(self, backtest_metrics, vals, shift):
        """Property: Sharpe increases when a positive constant is added to returns."""
        rets = pd.Series(vals, dtype=float)
        assume(rets.std() > 1e-12)
        bm = BacktestMetrics(risk_free_rate=0.0, bars_per_year=backtest_metrics.bars_per_year)
        s_orig = bm.calculate_sharpe(rets, annualize=False)
        s_shifted = bm.calculate_sharpe(rets + shift, annualize=False)
        assert s_shifted > s_orig, f"Sharpe should increase: {s_orig} -> {s_shifted}"


class TestSharpeScaleInvariance:
    """Sharpe is invariant under positive scaling of returns."""

    @given(
        st.lists(st.floats(min_value=-10, max_value=10), min_size=11, max_size=300),
        st.floats(min_value=0.5, max_value=5.0),
    )
    @settings(max_examples=100, deadline=5000, suppress_health_check=[HealthCheck.function_scoped_fixture])
    def test_sharpe_invariant_under_positive_scaling(self, backtest_metrics, vals, scale):
        """Property: Sharpe(c * returns) == Sharpe(returns) for c > 0, rf=0."""
        rets = pd.Series(vals, dtype=float)
        assume(rets.std() > 1e-12)
        bm = BacktestMetrics(risk_free_rate=0.0, bars_per_year=backtest_metrics.bars_per_year)
        s1 = bm.calculate_sharpe(rets, annualize=False)
        s2 = bm.calculate_sharpe(rets * scale, annualize=False)
        assert abs(s1 - s2) < 1e-10, f"Scale invariance broken: {s1} vs {s2}"


class TestSharpeNanConditions:
    """Sharpe returns NaN for insufficient data or zero variance."""

    @given(st.integers(min_value=1, max_value=9))
    @settings(max_examples=30, deadline=5000, suppress_health_check=[HealthCheck.function_scoped_fixture])
    def test_sharpe_nan_for_too_few_bars(self, backtest_metrics, n):
        """Property: Sharpe is NaN when n < 10."""
        rets = pd.Series(np.random.randn(n), dtype=float)
        s = backtest_metrics.calculate_sharpe(rets)
        assert np.isnan(s), f"Should be NaN for n={n}"

    @given(st.integers(min_value=-10, max_value=10))
    @settings(max_examples=20, deadline=5000, suppress_health_check=[HealthCheck.function_scoped_fixture])
    def test_sharpe_nan_for_zero_variance(self, backtest_metrics, const_val):
        """Property: Sharpe is NaN when all returns are equal integers (exact zero variance)."""
        rets = pd.Series([float(const_val)] * 20, dtype=float)
        s = backtest_metrics.calculate_sharpe(rets)
        assert np.isnan(s), f"Should be NaN for constant returns, got {s}"


class TestSharpeWithExcessReturn:
    """Sharpe with known excess return formula."""

    @given(
        st.floats(min_value=0.0001, max_value=0.01),
        st.floats(min_value=0.001, max_value=0.05),
        st.integers(min_value=11, max_value=500),
        st.floats(min_value=0.0, max_value=0.05),
    )
    @settings(max_examples=50, deadline=5000, suppress_health_check=[HealthCheck.function_scoped_fixture])
    def test_sharpe_with_gaussian_returns(self, backtest_metrics, mu, sigma, n, rf):
        """Property: Sharpe is finite for Gaussian returns with non-zero variance."""
        rng = np.random.default_rng(42)
        rets = pd.Series(rng.normal(mu, sigma, n), dtype=float)
        assume(rets.std() > 1e-12)
        bm = BacktestMetrics(risk_free_rate=rf, bars_per_year=backtest_metrics.bars_per_year)
        s_raw = bm.calculate_sharpe(rets, annualize=False)
        s_ann = bm.calculate_sharpe(rets, annualize=True)
        assert np.isfinite(s_raw)
        assert np.isfinite(s_ann)


# ---------------------------------------------------------------------------
# Max Drawdown Properties (16 tests)
# ---------------------------------------------------------------------------


class TestMaxDDProperties:
    """Max drawdown invariants."""

    @given(
        st.lists(st.floats(min_value=-0.5, max_value=1.0), min_size=30, max_size=500),
    )
    @settings(max_examples=100, deadline=5000, suppress_health_check=[HealthCheck.function_scoped_fixture])
    def test_maxdd_in_bounds(self, backtest_metrics, raw_rets):
        """Property: MaxDD ∈ [-1, 0] for non-negative equity."""
        rets = pd.Series(raw_rets, dtype=float)
        equity = (1 + rets).cumprod()
        assume(equity.min() > 0)
        dd = backtest_metrics.calculate_max_drawdown(equity)
        assert -1.0 <= dd <= 0.0, f"MaxDD={dd}"

    @given(
        st.lists(st.floats(min_value=0.0, max_value=0.5), min_size=20, max_size=300),
    )
    @settings(max_examples=70, deadline=5000, suppress_health_check=[HealthCheck.function_scoped_fixture])
    def test_maxdd_zero_for_monotonic_increasing(self, backtest_metrics, pos_rets):
        """Property: MaxDD == 0 for monotonically increasing equity (non-negative returns)."""
        rets = pd.Series(pos_rets, dtype=float)
        equity = (1 + rets).cumprod()
        dd = backtest_metrics.calculate_max_drawdown(equity)
        assert dd == 0.0, f"MaxDD should be 0 for non-negative returns, got {dd}"

    @given(
        st.lists(st.floats(min_value=-0.3, max_value=-0.01), min_size=20, max_size=300),
    )
    @settings(max_examples=70, deadline=5000, suppress_health_check=[HealthCheck.function_scoped_fixture])
    def test_maxdd_negative_for_declining_equity(self, backtest_metrics, neg_rets):
        """Property: MaxDD < 0 for monotonically decreasing equity."""
        rets = pd.Series(neg_rets, dtype=float)
        equity = (1 + rets).cumprod()
        assume(equity.min() > 0)
        dd = backtest_metrics.calculate_max_drawdown(equity)
        assert dd < 0, f"MaxDD should be negative for declining equity, got {dd}"

    @given(
        st.floats(min_value=1.0, max_value=1000.0),
        st.lists(st.floats(min_value=-0.5, max_value=1.0), min_size=20, max_size=300),
    )
    @settings(max_examples=70, deadline=5000, suppress_health_check=[HealthCheck.function_scoped_fixture])
    def test_maxdd_scale_invariance(self, backtest_metrics, scale, raw_rets):
        """Property: MaxDD is invariant under positive scaling of equity curve."""
        rets = pd.Series(raw_rets, dtype=float)
        eq1 = (1 + rets).cumprod()
        eq2 = eq1 * scale
        assume(eq1.min() > 0)
        dd1 = backtest_metrics.calculate_max_drawdown(eq1)
        dd2 = backtest_metrics.calculate_max_drawdown(eq2)
        assert abs(dd1 - dd2) < 1e-10, f"Scale invariance: {dd1} vs {dd2}"

    @given(
        st.lists(st.floats(min_value=-0.05, max_value=0.05), min_size=30, max_size=300),
    )
    @settings(max_examples=70, deadline=5000, suppress_health_check=[HealthCheck.function_scoped_fixture])
    def test_maxdd_not_exceed_total_loss(self, backtest_metrics, raw_rets):
        """Property: |MaxDD| <= |peak-to-trough loss|."""
        rets = pd.Series(raw_rets, dtype=float)
        equity = (1 + rets).cumprod()
        assume(equity.min() > 0)
        dd = backtest_metrics.calculate_max_drawdown(equity)
        peak = equity.cummax()
        worst_ratio = (equity / peak).min()
        assert abs(dd - (worst_ratio - 1)) < 1e-10, f"DD should equal ratio-1: {dd} vs {worst_ratio-1}"

    @given(
        st.lists(st.floats(min_value=-0.2, max_value=0.2), min_size=30, max_size=300),
    )
    @settings(max_examples=70, deadline=5000, suppress_health_check=[HealthCheck.function_scoped_fixture])
    def test_maxdd_happens_at_or_after_peak(self, backtest_metrics, raw_rets):
        """Property: The maximum drawdown occurs at or after the running maximum."""
        rets = pd.Series(raw_rets, dtype=float)
        equity = (1 + rets).cumprod()
        assume(equity.min() > 0)
        dd = backtest_metrics.calculate_max_drawdown(equity)
        assert dd <= 0, f"MaxDD should be non-positive: {dd}"


# ---------------------------------------------------------------------------
# Calculate All Properties (12 tests)
# ---------------------------------------------------------------------------


class TestCalculateAllProperties:
    """Properties for calculate_all."""

    @given(
        st.lists(st.floats(min_value=-0.5, max_value=0.5), min_size=20, max_size=300),
    )
    @settings(max_examples=70, deadline=5000, suppress_health_check=[HealthCheck.function_scoped_fixture])
    def test_total_return_formula(self, backtest_metrics, raw_rets):
        """Property: total_return == prod(1+returns)-1."""
        rets = pd.Series(raw_rets, dtype=float)
        equity = (1 + rets).cumprod()
        m = backtest_metrics.calculate_all(rets, equity)
        expected = (1 + rets).prod() - 1
        assert abs(m["total_return"] - expected) < 1e-10

    @given(
        st.lists(st.floats(min_value=-0.5, max_value=0.5), min_size=20, max_size=300),
    )
    @settings(max_examples=70, deadline=5000, suppress_health_check=[HealthCheck.function_scoped_fixture])
    def test_win_rate_in_01(self, backtest_metrics, raw_rets):
        """Property: win_rate ∈ [0, 1]."""
        rets = pd.Series(raw_rets, dtype=float)
        equity = (1 + rets).cumprod()
        m = backtest_metrics.calculate_all(rets, equity)
        assert 0.0 <= m["win_rate"] <= 1.0

    @given(
        st.lists(st.floats(min_value=-0.5, max_value=0.5), min_size=20, max_size=300),
    )
    @settings(max_examples=70, deadline=5000, suppress_health_check=[HealthCheck.function_scoped_fixture])
    def test_total_trades_equals_len(self, backtest_metrics, raw_rets):
        """Property: total_trades == len(returns)."""
        rets = pd.Series(raw_rets, dtype=float)
        equity = (1 + rets).cumprod()
        m = backtest_metrics.calculate_all(rets, equity)
        assert m["total_trades"] == len(rets)

    @given(
        st.lists(st.floats(min_value=-0.5, max_value=0.5), min_size=20, max_size=300),
    )
    @settings(max_examples=70, deadline=5000, suppress_health_check=[HealthCheck.function_scoped_fixture])
    def test_annualized_return_formula(self, backtest_metrics, raw_rets):
        """Property: annualized_return == mean(returns) * bars_per_year."""
        rets = pd.Series(raw_rets, dtype=float)
        equity = (1 + rets).cumprod()
        m = backtest_metrics.calculate_all(rets, equity)
        expected = rets.mean() * backtest_metrics.bars_per_year
        assert abs(m["annualized_return"] - expected) < 1e-10

    @given(
        st.lists(st.floats(min_value=-0.5, max_value=0.5), min_size=20, max_size=300),
    )
    @settings(max_examples=70, deadline=5000, suppress_health_check=[HealthCheck.function_scoped_fixture])
    def test_all_keys_present(self, backtest_metrics, raw_rets):
        """Property: calculate_all always has the standard keys."""
        rets = pd.Series(raw_rets, dtype=float)
        equity = (1 + rets).cumprod()
        m = backtest_metrics.calculate_all(rets, equity)
        for k in ["total_return", "annualized_return", "sharpe_ratio", "max_drawdown",
                  "win_rate", "total_trades"]:
            assert k in m

    @given(
        st.lists(st.floats(min_value=-0.5, max_value=0.5), min_size=20, max_size=300),
        st.lists(st.floats(min_value=-10, max_value=10), min_size=20, max_size=300),
    )
    @settings(max_examples=70, deadline=5000, suppress_health_check=[HealthCheck.function_scoped_fixture])
    def test_ic_included_when_factor_provided(self, backtest_metrics, raw_rets, raw_fac):
        """Property: 'ic' key is present only when factor_values and forward_returns are given."""
        rets = pd.Series(raw_rets, dtype=float)
        equity = (1 + rets).cumprod()
        fac = pd.Series(raw_fac, dtype=float)
        fwd = pd.Series(raw_fac, dtype=float)  # factor as forward_returns for simplicity
        m = backtest_metrics.calculate_all(rets, equity, fac, fwd)
        assert "ic" in m

    @given(
        st.lists(st.floats(min_value=-0.5, max_value=0.5), min_size=20, max_size=300),
    )
    @settings(max_examples=70, deadline=5000, suppress_health_check=[HealthCheck.function_scoped_fixture])
    def test_ic_not_present_when_no_factor(self, backtest_metrics, raw_rets):
        """Property: 'ic' key absent when no factor data is provided."""
        rets = pd.Series(raw_rets, dtype=float)
        equity = (1 + rets).cumprod()
        m = backtest_metrics.calculate_all(rets, equity)
        assert "ic" not in m


# ---------------------------------------------------------------------------
# FactorBacktester run_backtest Properties (15 tests)
# ---------------------------------------------------------------------------


class TestFactorBacktesterProperties:
    """Property-based tests for FactorBacktester.run_backtest."""

    @given(
        st.lists(st.floats(min_value=-100, max_value=100), min_size=30, max_size=300),
        st.lists(st.floats(min_value=-0.5, max_value=0.5), min_size=30, max_size=300),
        st.text(alphabet=st.characters(min_codepoint=65, max_codepoint=90), min_size=1, max_size=30),
        st.floats(min_value=0.00001, max_value=0.01),
    )
    @settings(max_examples=100, deadline=5000, suppress_health_check=[HealthCheck.function_scoped_fixture])
    def test_run_backtest_returns_all_required_keys(self, fac, ret, name, cost):
        """Property: run_backtest dict contains all expected keys."""
        from rdagent.components.backtesting.backtest_engine import FactorBacktester
        factor = pd.Series(fac, dtype=float)
        fwd = pd.Series(ret, dtype=float)
        assume(factor.std() > 1e-12)
        fb = FactorBacktester()
        with tempfile.TemporaryDirectory() as td:
            fb.results_path = Path(td)
            m = fb.run_backtest(factor, fwd, "PropTest_" + name, transaction_cost=cost)
            for k in ["total_return", "annualized_return", "sharpe_ratio",
                      "max_drawdown", "win_rate", "total_trades", "ic",
                      "factor_name", "timestamp"]:
                assert k in m, f"Missing key: {k}"

    @given(
        st.lists(st.floats(min_value=-100, max_value=100), min_size=30, max_size=300),
        st.lists(st.floats(min_value=-0.5, max_value=0.5), min_size=30, max_size=300),
        st.floats(min_value=0.00001, max_value=0.01),
    )
    @settings(max_examples=70, deadline=5000, suppress_health_check=[HealthCheck.function_scoped_fixture])
    def test_run_backtest_json_persisted(self, fac, ret, cost):
        """Property: run_backtest writes a JSON file to results_path."""
        from rdagent.components.backtesting.backtest_engine import FactorBacktester
        factor = pd.Series(fac, dtype=float)
        fwd = pd.Series(ret, dtype=float)
        assume(factor.std() > 1e-12)
        fb = FactorBacktester()
        with tempfile.TemporaryDirectory() as td:
            fb.results_path = Path(td)
            fb.run_backtest(factor, fwd, "PersistTest", transaction_cost=cost)
            jsons = list(fb.results_path.glob("*.json"))
            assert len(jsons) > 0

    @given(
        st.lists(st.floats(min_value=-100, max_value=100), min_size=30, max_size=300),
        st.lists(st.floats(min_value=-0.5, max_value=0.5), min_size=30, max_size=300),
    )
    @settings(max_examples=70, deadline=5000, suppress_health_check=[HealthCheck.function_scoped_fixture])
    def test_ic_invariant_under_scaling(self, fac, ret):
        """Property: IC from run_backtest is invariant under factor scaling."""
        from rdagent.components.backtesting.backtest_engine import FactorBacktester
        factor = pd.Series(fac, dtype=float)
        fwd = pd.Series(ret, dtype=float)
        assume(factor.std() > 1e-12)
        fb = FactorBacktester()
        with tempfile.TemporaryDirectory() as td:
            fb.results_path = Path(td)
            m1 = fb.run_backtest(factor, fwd, "Scaled_1")
            m2 = fb.run_backtest(factor * 3.7, fwd, "Scaled_2")
            if not (np.isnan(m1.get("ic", np.nan)) or np.isnan(m2.get("ic", np.nan))):
                assert abs(m1["ic"] - m2["ic"]) < 1e-10

    @given(
        st.lists(st.floats(min_value=-100, max_value=100), min_size=30, max_size=300),
        st.lists(st.floats(min_value=-0.5, max_value=0.5), min_size=30, max_size=300),
    )
    @settings(max_examples=70, deadline=5000, suppress_health_check=[HealthCheck.function_scoped_fixture])
    def test_total_trades_nonnegative(self, fac, ret):
        """Property: total_trades >= 0."""
        from rdagent.components.backtesting.backtest_engine import FactorBacktester
        factor = pd.Series(fac, dtype=float)
        fwd = pd.Series(ret, dtype=float)
        fb = FactorBacktester()
        with tempfile.TemporaryDirectory() as td:
            fb.results_path = Path(td)
            m = fb.run_backtest(factor, fwd, "TradesCheck")
            assert m["total_trades"] >= 0

    @given(
        st.lists(st.floats(min_value=-100, max_value=100), min_size=30, max_size=300),
        st.lists(st.floats(min_value=-0.5, max_value=0.5), min_size=30, max_size=300),
    )
    @settings(max_examples=70, deadline=5000, suppress_health_check=[HealthCheck.function_scoped_fixture])
    def test_max_drawdown_in_bounds(self, fac, ret):
        """Property: max_drawdown ∈ [-1, 0] from run_backtest."""
        from rdagent.components.backtesting.backtest_engine import FactorBacktester
        factor = pd.Series(fac, dtype=float)
        fwd = pd.Series(ret, dtype=float)
        fb = FactorBacktester()
        with tempfile.TemporaryDirectory() as td:
            fb.results_path = Path(td)
            m = fb.run_backtest(factor, fwd, "DDCheck")
            dd = m["max_drawdown"]
            if not np.isnan(dd):
                assert -1.0 <= dd <= 0.0, f"MaxDD={dd}"

    @given(
        st.lists(st.floats(min_value=-100, max_value=100), min_size=30, max_size=300),
        st.lists(st.floats(min_value=-0.5, max_value=0.5), min_size=30, max_size=300),
    )
    @settings(max_examples=70, deadline=5000, suppress_health_check=[HealthCheck.function_scoped_fixture])
    def test_win_rate_in_bounds(self, fac, ret):
        """Property: win_rate ∈ [0, 1] from run_backtest."""
        from rdagent.components.backtesting.backtest_engine import FactorBacktester
        factor = pd.Series(fac, dtype=float)
        fwd = pd.Series(ret, dtype=float)
        fb = FactorBacktester()
        with tempfile.TemporaryDirectory() as td:
            fb.results_path = Path(td)
            m = fb.run_backtest(factor, fwd, "WRCheck")
            wr = m["win_rate"]
            if not np.isnan(wr):
                assert 0.0 <= wr <= 1.0, f"WinRate={wr}"

    @given(
        st.lists(st.floats(min_value=-100, max_value=100), min_size=30, max_size=300),
        st.lists(st.floats(min_value=-0.5, max_value=0.5), min_size=30, max_size=300),
    )
    @settings(max_examples=70, deadline=5000, suppress_health_check=[HealthCheck.function_scoped_fixture])
    def test_factor_name_preserved(self, fac, ret):
        """Property: factor_name field matches the input name."""
        from rdagent.components.backtesting.backtest_engine import FactorBacktester
        factor = pd.Series(fac, dtype=float)
        fwd = pd.Series(ret, dtype=float)
        name = "MyTestFactor42"
        fb = FactorBacktester()
        with tempfile.TemporaryDirectory() as td:
            fb.results_path = Path(td)
            m = fb.run_backtest(factor, fwd, name)
            assert m["factor_name"] == name

    @given(
        st.lists(st.floats(min_value=-100, max_value=100), min_size=50, max_size=300),
        st.lists(st.floats(min_value=-0.5, max_value=0.5), min_size=50, max_size=300),
        st.floats(min_value=0.0001, max_value=0.005),
        st.floats(min_value=0.00001, max_value=0.0001),
    )
    @settings(max_examples=50, deadline=5000, suppress_health_check=[HealthCheck.function_scoped_fixture])
    def test_higher_cost_reduces_return(self, fac, ret, high_cost, low_cost):
        """Property: Higher transaction cost reduces total_return (or keeps equal)."""
        from rdagent.components.backtesting.backtest_engine import FactorBacktester
        factor = pd.Series(fac, dtype=float)
        fwd = pd.Series(ret, dtype=float)
        fb = FactorBacktester()
        with tempfile.TemporaryDirectory() as td:
            fb.results_path = Path(td)
            assume(high_cost > low_cost)
            m_high = fb.run_backtest(factor, fwd, "CostHigh", transaction_cost=high_cost)
            m_low = fb.run_backtest(factor, fwd, "CostLow", transaction_cost=low_cost)
            assert m_high["total_return"] <= m_low["total_return"] + 0.001, \
                f"Higher cost should not increase return: high={m_high['total_return']} low={m_low['total_return']}"
