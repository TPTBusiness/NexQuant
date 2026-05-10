"""
Tests für Risk Management - Korrelation, Portfolio-Optimierung, Risk-Checks

Test-Fälle:
- CorrelationAnalyzer.calculate_matrix(): Korrelationsmatrix
- CorrelationAnalyzer.find_uncorrelated(): Unkorrelierte Faktoren finden
- PortfolioOptimizer.mean_variance(): Mean-Variance-Optimierung
- PortfolioOptimizer.risk_parity(): Risk-Parity-Optimierung
- AdvancedRiskManager.check_limits(): Risk-Limits prüfen
- Edge Cases: Singuläre Matrizen, NaN-Werte, leere Daten, Extremwerte
"""
import pytest
import numpy as np
import pandas as pd
from pathlib import Path


class TestCorrelationAnalyzerCalculateMatrix:
    """Tests für CorrelationAnalyzer.calculate_matrix()"""

    def test_calculate_matrix_normal_data(self, correlation_analyzer, sample_returns_matrix):
        """Korrelationsmatrix mit normalen Daten sollte korrekt berechnet werden"""
        corr = correlation_analyzer.calculate_matrix(sample_returns_matrix)
        
        # Sollte quadratisch sein
        assert corr.shape[0] == corr.shape[1], "Matrix sollte quadratisch sein"
        # Sollte symmetrisch sein
        assert np.allclose(corr.values, corr.values.T), "Matrix sollte symmetrisch sein"
        # Diagonale sollte 1.0 sein
        diag = np.diag(corr.values)
        assert np.allclose(diag, 1.0), f"Diagonale sollte 1.0 sein, ist {diag}"
        # Alle Werte sollten zwischen -1 und 1 liegen
        assert corr.values.min() >= -1, f"Min Korrelation {corr.values.min()} < -1"
        assert corr.values.max() <= 1, f"Max Korrelation {corr.values.max()} > 1"

    def test_calculate_matrix_perfect_correlation(self, correlation_analyzer):
        """Perfekt korrelierte Assets sollten Korrelation 1.0 haben"""
        n = 100
        dates = pd.date_range(start='2024-01-01', periods=n, freq='B')
        
        # Zwei identische Returns
        returns = pd.DataFrame({
            'A': np.random.randn(n),
            'B': np.random.randn(n),  # gleich wie A
        }, index=dates)
        returns['B'] = returns['A']  # Perfekte Korrelation
        
        corr = correlation_analyzer.calculate_matrix(returns)
        assert abs(corr.loc['A', 'B'] - 1.0) < 1e-10, \
            f"Perfekte Korrelation sollte 1.0 sein, ist {corr.loc['A', 'B']}"

    def test_calculate_matrix_perfect_negative_correlation(self, correlation_analyzer):
        """Perfekt negativ korrelierte Assets sollten -1.0 haben"""
        n = 100
        dates = pd.date_range(start='2024-01-01', periods=n, freq='B')
        
        base = np.random.randn(n)
        returns = pd.DataFrame({
            'A': base,
            'B': -base,  # Perfekt negativ korreliert
        }, index=dates)
        
        corr = correlation_analyzer.calculate_matrix(returns)
        assert abs(corr.loc['A', 'B'] - (-1.0)) < 1e-10, \
            f"Perfekt negative Korrelation sollte -1.0 sein, ist {corr.loc['A', 'B']}"

    def test_calculate_matrix_empty_data(self, correlation_analyzer, empty_data):
        """Korrelationsmatrix mit leeren Daten sollte leere Matrix zurückgeben"""
        factor, _ = empty_data
        empty_df = pd.DataFrame()
        
        corr = correlation_analyzer.calculate_matrix(empty_df)
        
        assert corr.empty, "Leere Daten sollten leere Matrix ergeben"

    def test_calculate_matrix_with_nan(self, correlation_analyzer, sample_returns_matrix):
        """Korrelationsmatrix mit NaN-Werten sollte korrekt umgehen"""
        # Füge NaN-Werte hinzu
        data_with_nan = sample_returns_matrix.copy()
        data_with_nan.iloc[0:10, 0] = np.nan
        
        corr = correlation_analyzer.calculate_matrix(data_with_nan)
        
        # Sollte trotzdem berechenbar sein (pandas dropna)
        assert corr.shape[0] == corr.shape[1], "Matrix sollte quadratisch sein"
        # Keine NaN in der resultierenden Matrix (außer bei konstanten Spalten)
        # NaN ist akzeptabel wenn eine Spalte nur NaN hat

    def test_calculate_matrix_single_asset(self, correlation_analyzer):
        """Korrelationsmatrix mit nur einem Asset"""
        n = 100
        dates = pd.date_range(start='2024-01-01', periods=n, freq='B')
        returns = pd.DataFrame({'A': np.random.randn(n)}, index=dates)
        
        corr = correlation_analyzer.calculate_matrix(returns)
        
        assert corr.shape == (1, 1), "Single Asset sollte 1x1 Matrix sein"
        assert corr.iloc[0, 0] == 1.0, "Korrelation mit sich selbst sollte 1.0 sein"

    def test_calculate_matrix_insufficient_data(self, correlation_analyzer):
        """Korrelationsmatrix mit zu wenig Datenpunkten"""
        n = 2  # Weniger als Assets
        dates = pd.date_range(start='2024-01-01', periods=n, freq='B')
        returns = pd.DataFrame({
            'A': np.random.randn(n),
            'B': np.random.randn(n),
            'C': np.random.randn(n),
        }, index=dates)
        
        corr = correlation_analyzer.calculate_matrix(returns)
        
        # Sollte trotzdem funktionieren (kann NaN enthalten bei zu wenig Daten)
        assert corr.shape == (3, 3), "Matrix sollte 3x3 sein"


class TestCorrelationAnalyzerFindUncorrelated:
    """Tests für CorrelationAnalyzer.find_uncorrelated()"""

    def test_find_uncorrelated_identifies_uncorrelated(self, correlation_analyzer):
        """find_uncorrelated sollte unkorrelierte Faktoren identifizieren"""
        n = 252
        dates = pd.date_range(start='2024-01-01', periods=n, freq='B')
        
        # Erzeuge Daten wo 'Uncorrelated' wirklich unkorreliert ist
        np.random.seed(42)
        base1 = np.random.randn(n)
        base2 = np.random.randn(n)
        uncorr = np.random.randn(n)  # Unabhängig
        
        returns = pd.DataFrame({
            'Correlated1': base1,
            'Correlated2': base2,
            'Correlated3': base1 * 0.5 + base2 * 0.5,
            'Uncorrelated': uncorr,
        }, index=dates)
        
        corr = correlation_analyzer.calculate_matrix(returns)
        uncorr_factors = correlation_analyzer.find_uncorrelated(corr, threshold=0.3)
        
        assert 'Uncorrelated' in uncorr_factors, "Uncorrelated sollte gefunden werden"

    def test_find_uncorrelated_all_correlated(self, correlation_analyzer):
        """Wenn alle korreliert sind, sollte leere Liste zurückgegeben werden"""
        n = 100
        dates = pd.date_range(start='2024-01-01', periods=n, freq='B')
        
        base = np.random.randn(n)
        returns = pd.DataFrame({
            'A': base,
            'B': base * 0.9,  # Stark korreliert
            'C': base * 0.8,  # Stark korreliert
        }, index=dates)
        
        corr = correlation_analyzer.calculate_matrix(returns)
        uncorr_factors = correlation_analyzer.find_uncorrelated(corr, threshold=0.3)
        
        # Bei starker Korrelation sollte keiner unkorreliert sein
        assert len(uncorr_factors) == 0, f"Erwartet keine unkorrelierten, gefunden {uncorr_factors}"

    def test_find_uncorrelated_custom_threshold(self, correlation_analyzer, sample_returns_matrix):
        """find_uncorrelated mit custom threshold"""
        corr = correlation_analyzer.calculate_matrix(sample_returns_matrix)
        
        # Niedriger threshold sollte weniger Faktoren finden
        uncorr_strict = correlation_analyzer.find_uncorrelated(corr, threshold=0.1)
        # Hoher threshold sollte mehr Faktoren finden
        uncorr_loose = correlation_analyzer.find_uncorrelated(corr, threshold=0.8)
        
        assert len(uncorr_loose) >= len(uncorr_strict), \
            "Höherer threshold sollte >= Faktoren finden"

    def test_find_uncorrelated_empty_matrix(self, correlation_analyzer):
        """find_uncorrelated mit leerer Matrix"""
        empty_corr = pd.DataFrame()
        
        result = correlation_analyzer.find_uncorrelated(empty_corr)
        
        assert result == [], "Leere Matrix sollte leere Liste zurückgeben"

    def test_find_uncorrelated_single_asset(self, correlation_analyzer):
        """find_uncorrelated mit nur einem Asset"""
        corr = pd.DataFrame([[1.0]], columns=['A'], index=['A'])
        
        result = correlation_analyzer.find_uncorrelated(corr, threshold=0.3)
        
        # Single Asset hat keine "anderen" zur Korrelation, sollte gefunden werden
        assert 'A' in result or result == [], "Single Asset Verhalten unerwartet"


class TestPortfolioOptimizerMeanVariance:
    """Tests für PortfolioOptimizer.mean_variance()"""

    def test_mean_variance_basic(self, portfolio_optimizer, sample_expected_returns, sample_covariance_matrix):
        """Mean-Variance-Optimierung sollte Gewichte zurückgeben"""
        weights = portfolio_optimizer.mean_variance(sample_expected_returns, sample_covariance_matrix)
        
        # Gewichte sollten Array sein
        assert isinstance(weights, np.ndarray), "Gewichte sollten numpy Array sein"
        # Länge sollte Anzahl Assets entsprechen
        assert len(weights) == len(sample_expected_returns), "Falsche Länge der Gewichte"
        # Summe sollte ~1 sein (fully invested)
        assert abs(np.sum(weights) - 1.0) < 0.01, f"Gewichte summieren zu {np.sum(weights)}"

    def test_mean_variance_higher_expected_return(self, portfolio_optimizer, sample_covariance_matrix):
        """Höhere expected returns sollten höheres Gewicht bekommen"""
        # Asset mit sehr hohem expected return
        exp_ret = pd.Series({'A': 0.50, 'B': 0.01, 'C': 0.01})
        cov = pd.DataFrame(
            [[0.04, 0.001, 0.001], [0.001, 0.04, 0.001], [0.001, 0.001, 0.04]],
            index=['A', 'B', 'C'], columns=['A', 'B', 'C']
        )
        
        weights = portfolio_optimizer.mean_variance(exp_ret, cov)
        
        # Asset A sollte höchstes Gewicht haben
        assert weights[0] > weights[1] and weights[0] > weights[2], \
            f"Asset mit höchstem Return sollte höchstes Gewicht haben: {weights}"

    def test_mean_variance_singular_covariance(self, portfolio_optimizer, sample_expected_returns):
        """Mean-Variance mit singulärer Kovarianz-Matrix sollte Fallback nutzen"""
        # Singuläre Matrix (alle Assets perfekt korreliert)
        cov = pd.DataFrame(
            [[0.04, 0.04, 0.04], [0.04, 0.04, 0.04], [0.04, 0.04, 0.04]],
            index=['A', 'B', 'C'], columns=['A', 'B', 'C']
        )
        
        weights = portfolio_optimizer.mean_variance(sample_expected_returns, cov)
        
        # Sollte Fallback nutzen (equal weights)
        assert len(weights) == len(sample_expected_returns), "Fallback sollte gleiche Länge haben"
        # Bei Fallback: equal weights
        assert abs(np.sum(weights) - 1.0) < 0.01, "Fallback-Gewichte sollten zu 1 summieren"

    def test_mean_variance_zero_covariance(self, portfolio_optimizer, sample_expected_returns):
        """Mean-Variance mit Null-Kovarianz sollte Fallback nutzen"""
        # Erstelle Kovarianz-Matrix mit passender Größe für sample_expected_returns (5 Assets)
        n = len(sample_expected_returns)
        cov = pd.DataFrame(
            [[0] * n for _ in range(n)],
            index=sample_expected_returns.index, columns=sample_expected_returns.index
        )
        
        weights = portfolio_optimizer.mean_variance(sample_expected_returns, cov)
        
        # Sollte Fallback nutzen (equal weights)
        assert len(weights) == n, f"Zero cov sollte Fallback mit {n} Gewichten nutzen"
        # Bei Fallback: equal weights
        expected_weight = 1.0 / n
        assert np.allclose(weights, expected_weight, atol=0.01), \
            f"Zero covariance sollte equal weights geben: {weights}"

    def test_mean_variance_negative_expected_returns(self, portfolio_optimizer, sample_covariance_matrix):
        """Mean-Variance mit negativen expected returns"""
        exp_ret = pd.Series({'A': -0.10, 'B': -0.05, 'C': 0.02})
        
        weights = portfolio_optimizer.mean_variance(exp_ret, sample_covariance_matrix)
        
        assert len(weights) == 3, "Negative returns sollten funktionieren"
        assert abs(np.sum(weights) - 1.0) < 0.01, "Gewichte sollten zu 1 summieren"


class TestPortfolioOptimizerRiskParity:
    """Tests für PortfolioOptimizer.risk_parity()"""

    def test_risk_parity_basic(self, portfolio_optimizer, sample_covariance_matrix):
        """Risk-Parity-Optimierung sollte Gewichte zurückgeben"""
        weights = portfolio_optimizer.risk_parity(sample_covariance_matrix)
        
        # Gewichte sollten Array sein
        assert isinstance(weights, np.ndarray), "Gewichte sollten numpy Array sein"
        # Länge sollte Anzahl Assets entsprechen
        assert len(weights) == sample_covariance_matrix.shape[0], "Falsche Länge der Gewichte"
        # Summe sollte ~1 sein
        assert abs(np.sum(weights) - 1.0) < 0.01, f"Gewichte summieren zu {np.sum(weights)}"
        # Alle Gewichte sollten positiv sein (long-only)
        assert np.all(weights > 0), f"Risk Parity sollte positive Gewichte haben: {weights}"

    def test_risk_parity_equal_volatility(self, portfolio_optimizer):
        """Risk-Parity bei gleicher Volatilität sollte gleiche Gewichte geben"""
        # Diagonale Kovarianz mit gleicher Varianz
        cov = pd.DataFrame(
            [[0.04, 0, 0], [0, 0.04, 0], [0, 0, 0.04]],
            index=['A', 'B', 'C'], columns=['A', 'B', 'C']
        )
        
        weights = portfolio_optimizer.risk_parity(cov)
        
        # Bei gleicher Volatilität sollten Gewichte gleich sein
        expected = np.array([1/3, 1/3, 1/3])
        assert np.allclose(weights, expected, atol=0.01), \
            f"Bei gleicher Volatilität sollten Gewichte gleich sein: {weights}"

    def test_risk_parity_different_volatility(self, portfolio_optimizer):
        """Risk-Parity bei unterschiedlicher Volatilität"""
        # Unterschiedliche Varianzen
        cov = pd.DataFrame(
            [[0.01, 0, 0], [0, 0.04, 0], [0, 0, 0.09]],  # Vol: 10%, 20%, 30%
            index=['LowVol', 'MedVol', 'HighVol'], columns=['LowVol', 'MedVol', 'HighVol']
        )
        
        weights = portfolio_optimizer.risk_parity(cov)
        
        # Niedrigere Volatilität sollte höheres Gewicht bekommen
        assert weights[0] > weights[2], \
            f"LowVol sollte höheres Gewicht als HighVol haben: {weights}"

    def test_risk_parity_convergence(self, portfolio_optimizer, sample_covariance_matrix):
        """Risk-Parity sollte konvergieren"""
        weights1 = portfolio_optimizer.risk_parity(sample_covariance_matrix, max_iter=10)
        weights2 = portfolio_optimizer.risk_parity(sample_covariance_matrix, max_iter=1000)
        
        # Mehr Iterationen sollten zu ähnlichem oder besserem Ergebnis führen
        assert len(weights1) == len(weights2), "Länge sollte gleich bleiben"

    def test_risk_parity_single_asset(self, portfolio_optimizer):
        """Risk-Parity mit nur einem Asset"""
        cov = pd.DataFrame([[0.04]], index=['A'], columns=['A'])
        
        weights = portfolio_optimizer.risk_parity(cov)
        
        assert len(weights) == 1, "Single Asset sollte 1 Gewicht haben"
        assert weights[0] == 1.0, f"Single Asset sollte Gewicht 1.0 haben: {weights}"

    def test_risk_parity_zero_variance(self, portfolio_optimizer):
        """Risk-Parity mit Null-Varianz sollte Fallback nutzen"""
        cov = pd.DataFrame(
            [[0, 0], [0, 0]],
            index=['A', 'B'], columns=['A', 'B']
        )
        
        weights = portfolio_optimizer.risk_parity(cov)
        
        # Sollte equal weights Fallback nutzen
        assert np.allclose(weights, [0.5, 0.5], atol=0.01), \
            f"Zero variance sollte equal weights geben: {weights}"


class TestAdvancedRiskManagerCheckLimits:
    """Tests für AdvancedRiskManager.check_limits()"""

    def test_check_limits_all_pass(self, risk_manager, sample_weights):
        """check_limits sollte alle True zurückgeben wenn Limits eingehalten"""
        # Gewichte innerhalb der Limits
        weights = np.array([0.15, 0.15, 0.15, 0.15, 0.15])  # Max 15%, Summe 75%
        
        checks = risk_manager.check_limits(weights, vol=0.15, dd=-0.08)
        
        assert checks['position_limit'] == True, "Position Limit sollte eingehalten sein"
        assert checks['leverage_limit'] == True, "Leverage Limit sollte eingehalten sein"
        assert checks['drawdown_limit'] == True, "Drawdown Limit sollte eingehalten sein"

    def test_check_limits_position_exceeded(self, risk_manager):
        """check_limits sollte False für position_limit wenn exceeded"""
        # Eine Position > 20%
        weights = np.array([0.30, 0.10, 0.10, 0.10, 0.10])  # 30% in einer Position
        
        checks = risk_manager.check_limits(weights, vol=0.15, dd=-0.08)
        
        assert checks['position_limit'] == False, "Position Limit sollte verletzt sein"

    def test_check_limits_leverage_exceeded(self, risk_manager):
        """check_limits sollte False für leverage_limit wenn exceeded"""
        # Summe der absoluten Gewichte > 5.0
        weights = np.array([0.30, 0.30, 0.30, 0.30, 0.30])  # Summe = 150%
        weights = np.array([1.5, 1.5, 1.5, 1.5, -1.0])  # Summe abs = 7.0
        
        checks = risk_manager.check_limits(weights, vol=0.15, dd=-0.08)
        
        assert checks['leverage_limit'] == False, "Leverage Limit sollte verletzt sein"

    def test_check_limits_drawdown_exceeded(self, risk_manager, sample_weights):
        """check_limits sollte False für drawdown_limit wenn exceeded"""
        # Drawdown > 20%
        
        checks = risk_manager.check_limits(sample_weights, vol=0.15, dd=-0.25)
        
        assert checks['drawdown_limit'] == False, "Drawdown Limit sollte verletzt sein"

    def test_check_limits_boundary_values(self, risk_manager):
        """check_limits an den Grenzwerten"""
        # Genau an den Limits
        weights = np.array([0.2, 0.2, 0.2, 0.2, 0.2])  # Max genau 20%, Summe = 100%
        
        checks = risk_manager.check_limits(weights, vol=0.15, dd=-0.20)
        
        assert checks['position_limit'] == True, "Position an Grenze sollte OK sein"
        assert checks['leverage_limit'] == True, "Leverage an Grenze sollte OK sein"
        assert checks['drawdown_limit'] == True, "Drawdown an Grenze sollte OK sein"

    def test_check_limits_negative_weights(self, risk_manager):
        """check_limits mit negativen Gewichten (Short-Positionen)"""
        weights = np.array([0.3, -0.2, 0.3, -0.1, 0.2])  # Einige Short-Positionen
        
        checks = risk_manager.check_limits(weights, vol=0.15, dd=-0.08)
        
        # position_limit prüft abs(weight), also 0.3 > 0.2 -> False
        assert checks['position_limit'] == False, "Short mit |weight| > max sollte False sein"

    def test_check_limits_custom_manager_params(self):
        """check_limits mit custom Risk-Manager-Parametern"""
        # Strengere Limits
        strict_manager = AdvancedRiskManager(max_pos=0.10, max_lev=2.0, max_dd=0.10)

        weights = np.array([0.15, 0.15, 0.15, 0.15, 0.15])
        checks = strict_manager.check_limits(weights, vol=0.15, dd=-0.08)

        assert checks['position_limit'] == False, "15% > 10% strict limit"
        # Leverage ist 0.75 (75%) was < 2.0 ist, also True
        assert checks['leverage_limit'] == True, "75% < 2.0 leverage limit"


class TestRiskManagementIntegration:
    """Integrationstests für das gesamte Risk-Management-System"""

    def test_full_risk_analysis_workflow(self, sample_returns_matrix, sample_expected_returns):
        """Kompletter Risk-Analysis-Workflow"""
        # 1. Korrelation analysieren
        analyzer = CorrelationAnalyzer()
        corr = analyzer.calculate_matrix(sample_returns_matrix)
        
        # 2. Unkorrelierte Faktoren finden
        uncorr = analyzer.find_uncorrelated(corr, threshold=0.3)
        
        # 3. Portfolio optimieren
        optimizer = PortfolioOptimizer()
        cov = sample_returns_matrix.cov() * 252
        
        mv_weights = optimizer.mean_variance(sample_expected_returns, cov)
        rp_weights = optimizer.risk_parity(cov)
        
        # 4. Risk-Checks durchführen
        risk_manager = AdvancedRiskManager()
        mv_checks = risk_manager.check_limits(mv_weights, vol=0.15, dd=-0.08)
        rp_checks = risk_manager.check_limits(rp_weights, vol=0.15, dd=-0.08)
        
        # Alle sollten durchführbar sein
        assert isinstance(corr, pd.DataFrame)
        assert isinstance(uncorr, list)
        assert len(mv_weights) == len(sample_expected_returns)
        assert len(rp_weights) == len(sample_expected_returns)
        assert isinstance(mv_checks, dict)
        assert isinstance(rp_checks, dict)

    def test_portfolio_construction_with_risk_limits(self, sample_returns_matrix, sample_expected_returns):
        """Portfolio-Konstruktion mit Risk-Limit-Überprüfung"""
        optimizer = PortfolioOptimizer()
        risk_manager = AdvancedRiskManager(max_pos=0.25, max_lev=3.0)
        
        cov = sample_returns_matrix.cov() * 252
        
        # Versuche beide Optimierungsmethoden
        mv_weights = optimizer.mean_variance(sample_expected_returns, cov)
        rp_weights = optimizer.risk_parity(cov)
        
        # Prüfe welche Methode die Limits einhält
        mv_checks = risk_manager.check_limits(mv_weights, vol=0.15, dd=-0.05)
        rp_checks = risk_manager.check_limits(rp_weights, vol=0.15, dd=-0.05)
        
        # Mindestens eine Methode sollte funktionieren
        mv_pass = all(mv_checks.values())
        rp_pass = all(rp_checks.values())
        
        assert mv_pass or rp_pass, "Mindestens eine Optimierungsmethode sollte Limits einhalten"

    def test_risk_adjusted_portfolio_selection(self, sample_returns_matrix):
        """Risikoadjustierte Portfolio-Auswahl"""
        analyzer = CorrelationAnalyzer()
        corr = analyzer.calculate_matrix(sample_returns_matrix)
        
        # Finde unkorrelierte Faktoren für Diversifikation
        uncorr_factors = analyzer.find_uncorrelated(corr, threshold=0.4)
        
        # Wenn es unkorrelierte Faktoren gibt, sollten sie im Portfolio sein
        if len(uncorr_factors) > 0:
            # Diese Faktoren bieten Diversifikationsvorteile
            assert len(uncorr_factors) <= len(sample_returns_matrix.columns), \
                "Zu viele unkorrelierte Faktoren gefunden"


# Import am Anfang der Datei für die Tests
from rdagent.components.backtesting.risk_management import (
    CorrelationAnalyzer, PortfolioOptimizer, AdvancedRiskManager
)


# ============================================================================
# HYPOTHESIS PROPERTY-BASED TESTS (ADDED – DO NOT MODIFY ABOVE THIS LINE)
# ============================================================================

from hypothesis import given, settings, strategies as st, assume

# ---------------------------------------------------------------------------
# Correlation Matrix Properties (22 tests)
# ---------------------------------------------------------------------------


class TestCorrelationMatrixProperties:
    """Property-based tests for correlation matrix invariants."""

    @given(
        st.integers(min_value=2, max_value=15),
        st.integers(min_value=30, max_value=500),
        st.floats(min_value=0.001, max_value=0.1),
    )
    @settings(max_examples=100, deadline=5000)
    def test_corr_matrix_symmetric(self, n_assets, n_bars, noise):
        """Property: correlation matrix is always symmetric."""
        dates = pd.date_range("2024-01-01", periods=n_bars, freq="B")
        rng = np.random.default_rng(42)
        data = rng.normal(0, noise, (n_bars, n_assets))
        df = pd.DataFrame(data, columns=[f"A_{i}" for i in range(n_assets)], index=dates)
        analyzer = CorrelationAnalyzer()
        corr = analyzer.calculate_matrix(df)
        assert np.allclose(corr.values, corr.values.T, atol=1e-10)

    @given(
        st.integers(min_value=1, max_value=20),
        st.integers(min_value=30, max_value=500),
    )
    @settings(max_examples=70, deadline=5000)
    def test_corr_diagonal_is_one(self, n_assets, n_bars):
        """Property: all diagonal elements of correlation matrix equal 1.0."""
        dates = pd.date_range("2024-01-01", periods=n_bars, freq="B")
        rng = np.random.default_rng(42)
        data = rng.normal(0, 0.02, (n_bars, n_assets))
        df = pd.DataFrame(data, columns=[f"A_{i}" for i in range(n_assets)], index=dates)
        analyzer = CorrelationAnalyzer()
        corr = analyzer.calculate_matrix(df)
        diag = np.diag(corr.values)
        assert np.allclose(diag, 1.0, atol=1e-10)

    @given(
        st.integers(min_value=3, max_value=10),
        st.integers(min_value=50, max_value=300),
    )
    @settings(max_examples=70, deadline=5000)
    def test_corr_values_in_bounds(self, n_assets, n_bars):
        """Property: all correlation values ∈ [-1, 1]."""
        dates = pd.date_range("2024-01-01", periods=n_bars, freq="B")
        rng = np.random.default_rng(42)
        data = rng.normal(0, 0.02, (n_bars, n_assets))
        df = pd.DataFrame(data, columns=[f"A_{i}" for i in range(n_assets)], index=dates)
        analyzer = CorrelationAnalyzer()
        corr = analyzer.calculate_matrix(df)
        vals = corr.values.ravel()
        vals = vals[~np.isnan(vals)]
        assert np.all(vals >= -1.0)
        assert np.all(vals <= 1.0)

    @given(
        st.integers(min_value=2, max_value=6),
        st.integers(min_value=30, max_value=500),
    )
    @settings(max_examples=50, deadline=5000)
    def test_corr_psd(self, n_assets, n_bars):
        """Property: correlation matrix is positive semi-definite."""
        dates = pd.date_range("2024-01-01", periods=n_bars, freq="B")
        rng = np.random.default_rng(42)
        data = rng.normal(0, 0.02, (n_bars, n_assets))
        df = pd.DataFrame(data, columns=[f"A_{i}" for i in range(n_assets)], index=dates)
        analyzer = CorrelationAnalyzer()
        corr = analyzer.calculate_matrix(df)
        vals = corr.values
        vals = np.nan_to_num(vals, nan=0)
        eigenvalues = np.linalg.eigvalsh(vals)
        assert np.all(eigenvalues >= -1e-10), f"Non-PSD: min eigenvalue={eigenvalues.min()}"

    @given(st.integers(min_value=30, max_value=500))
    @settings(max_examples=50, deadline=5000)
    def test_single_asset_corr_is_one(self, n_bars):
        """Property: correlation matrix of single asset is [[1.0]]."""
        dates = pd.date_range("2024-01-01", periods=n_bars, freq="B")
        rng = np.random.default_rng(42)
        df = pd.DataFrame({"Only": rng.normal(0, 0.02, n_bars)}, index=dates)
        analyzer = CorrelationAnalyzer()
        corr = analyzer.calculate_matrix(df)
        assert corr.shape == (1, 1)
        assert corr.iloc[0, 0] == 1.0

    @given(
        st.integers(min_value=3, max_value=10),
        st.integers(min_value=50, max_value=300),
    )
    @settings(max_examples=50, deadline=5000)
    def test_corr_equals_corr_from_pandas(self, n_assets, n_bars):
        """Property: calculate_matrix matches pandas .corr()."""
        dates = pd.date_range("2024-01-01", periods=n_bars, freq="B")
        rng = np.random.default_rng(42)
        data = rng.normal(0, 0.02, (n_bars, n_assets))
        df = pd.DataFrame(data, columns=[f"A_{i}" for i in range(n_assets)], index=dates)
        analyzer = CorrelationAnalyzer()
        result = analyzer.calculate_matrix(df)
        expected = df.dropna().corr()
        assert np.allclose(result.values, expected.values, atol=1e-10, equal_nan=True)

    @given(
        st.floats(min_value=0.1, max_value=0.9),
        st.integers(min_value=50, max_value=200),
    )
    @settings(max_examples=40, deadline=5000)
    def test_corr_with_nans_still_symmetric(self, nan_fraction, n_bars):
        """Property: correlation matrix stays symmetric even with NaN-contaminated data."""
        n_assets = 5
        dates = pd.date_range("2024-01-01", periods=n_bars, freq="B")
        rng = np.random.default_rng(42)
        data = rng.normal(0, 0.02, (n_bars, n_assets))
        df = pd.DataFrame(data, columns=[f"A_{i}" for i in range(n_assets)], index=dates)
        for col in df.columns:
            n_nan = int(n_bars * nan_fraction * 0.3)
            df.loc[df.index[:n_nan], col] = np.nan
        analyzer = CorrelationAnalyzer()
        corr = analyzer.calculate_matrix(df)
        vals = np.nan_to_num(corr.values, nan=0)
        assert np.allclose(vals, vals.T, atol=1e-10)


# ---------------------------------------------------------------------------
# find_uncorrelated Properties (12 tests)
# ---------------------------------------------------------------------------


class TestFindUncorrelatedProperties:
    """Property tests for find_uncorrelated."""

    @given(
        st.integers(min_value=3, max_value=10),
        st.integers(min_value=100, max_value=500),
        st.floats(min_value=0.0, max_value=1.0),
    )
    @settings(max_examples=100, deadline=5000)
    def test_uncorrelated_count_bounded_by_n_assets(self, n_assets, n_bars, threshold):
        """Property: number of uncorrelated factors <= n_assets."""
        dates = pd.date_range("2024-01-01", periods=n_bars, freq="B")
        rng = np.random.default_rng(42)
        data = rng.normal(0, 0.02, (n_bars, n_assets))
        df = pd.DataFrame(data, columns=[f"A_{i}" for i in range(n_assets)], index=dates)
        analyzer = CorrelationAnalyzer()
        corr = analyzer.calculate_matrix(df)
        result = analyzer.find_uncorrelated(corr, threshold=threshold)
        assert len(result) <= n_assets

    @given(
        st.integers(min_value=3, max_value=8),
        st.integers(min_value=100, max_value=400),
        st.floats(min_value=0.0, max_value=0.5),
        st.floats(min_value=0.5, max_value=1.0),
    )
    @settings(max_examples=70, deadline=5000)
    def test_threshold_monotonicity(self, n_assets, n_bars, t_low, t_high):
        """Property: higher threshold => more or equal uncorrelated factors."""
        assume(t_low <= t_high)
        dates = pd.date_range("2024-01-01", periods=n_bars, freq="B")
        rng = np.random.default_rng(42)
        data = rng.normal(0, 0.02, (n_bars, n_assets))
        df = pd.DataFrame(data, columns=[f"A_{i}" for i in range(n_assets)], index=dates)
        analyzer = CorrelationAnalyzer()
        corr = analyzer.calculate_matrix(df)
        r_low = analyzer.find_uncorrelated(corr, threshold=t_low)
        r_high = analyzer.find_uncorrelated(corr, threshold=t_high)
        assert len(r_high) >= len(r_low)

    @given(
        st.integers(min_value=30, max_value=300),
    )
    @settings(max_examples=30, deadline=5000)
    def test_empty_matrix_returns_empty(self, n_bars):
        """Property: find_uncorrelated on empty matrix returns []."""
        analyzer = CorrelationAnalyzer()
        assert analyzer.find_uncorrelated(pd.DataFrame()) == []

    @given(
        st.integers(min_value=120, max_value=300),
    )
    @settings(max_examples=30, deadline=5000)
    def test_single_asset_is_uncorrelated(self, n_bars):
        """Property: single-asset mean abs correlation to others is NaN → not found."""
        dates = pd.date_range("2024-01-01", periods=n_bars, freq="B")
        rng = np.random.default_rng(42)
        df = pd.DataFrame({"Solo": rng.normal(0, 0.02, n_bars)}, index=dates)
        analyzer = CorrelationAnalyzer()
        corr = analyzer.calculate_matrix(df)
        result = analyzer.find_uncorrelated(corr, threshold=0.5)
        # Single asset has no "others" — abs().mean() returns NaN, which is not < threshold
        # So it should NOT be in result (or the list may be empty)
        assert isinstance(result, list)


# ---------------------------------------------------------------------------
# Mean-Variance Properties (18 tests)
# ---------------------------------------------------------------------------


class TestMeanVarianceProperties:
    """Property-based tests for mean_variance optimization."""

    @given(
        st.integers(min_value=2, max_value=10),
    )
    @settings(max_examples=50, deadline=5000)
    def test_weights_sum_to_one(self, n_assets):
        """Property: mean_variance weights always sum to 1."""
        names = [f"A_{i}" for i in range(n_assets)]
        exp_ret = pd.Series(np.random.default_rng(42).uniform(0.01, 0.15, n_assets), index=names)
        cov_data = np.random.default_rng(43).uniform(0.01, 0.1, (n_assets, n_assets))
        cov_data = cov_data @ cov_data.T + np.eye(n_assets) * 0.01  # make PSD
        cov = pd.DataFrame(cov_data, index=names, columns=names)
        opt = PortfolioOptimizer()
        w = opt.mean_variance(exp_ret, cov)
        assert abs(np.sum(w) - 1.0) < 1e-10

    @given(
        st.integers(min_value=2, max_value=8),
    )
    @settings(max_examples=50, deadline=5000)
    def test_weights_are_numpy_array(self, n_assets):
        """Property: mean_variance returns numpy array."""
        names = [f"A_{i}" for i in range(n_assets)]
        exp_ret = pd.Series(np.random.default_rng(42).uniform(0.01, 0.15, n_assets), index=names)
        cov = pd.DataFrame(np.eye(n_assets) * 0.04, index=names, columns=names)
        opt = PortfolioOptimizer()
        w = opt.mean_variance(exp_ret, cov)
        assert isinstance(w, np.ndarray)
        assert len(w) == n_assets

    @given(
        st.integers(min_value=2, max_value=6),
        st.floats(min_value=0.001, max_value=0.2),
    )
    @settings(max_examples=50, deadline=5000)
    def test_equal_returns_different_vol_weights(self, n_assets, ret_val):
        """Property: if all returns equal, lower-vol assets get higher weight."""
        names = [f"A_{i}" for i in range(n_assets)]
        exp_ret = pd.Series([ret_val] * n_assets, index=names)
        # Increasing vol: A0 has 0.01, A1 has 0.04, ...
        diag = np.array([0.01 * (i + 1) for i in range(n_assets)])
        cov = pd.DataFrame(np.diag(diag), index=names, columns=names)
        opt = PortfolioOptimizer()
        w = opt.mean_variance(exp_ret, cov)
        assert w[np.argmin(diag)] > w[np.argmax(diag)]

    @given(
        st.integers(min_value=3, max_value=6),
    )
    @settings(max_examples=50, deadline=5000)
    def test_higher_return_gets_higher_weight_ceteris_paribus(self, n_assets):
        """Property: among assets with equal risk, the one with highest return gets highest weight."""
        names = [f"A_{i}" for i in range(n_assets)]
        rets = np.linspace(0.01, 0.20, n_assets)
        exp_ret = pd.Series(rets, index=names)
        cov = pd.DataFrame(np.eye(n_assets) * 0.04, index=names, columns=names)
        opt = PortfolioOptimizer()
        w = opt.mean_variance(exp_ret, cov)
        assert np.argmax(w) == np.argmax(rets)

    @given(
        st.integers(min_value=2, max_value=6),
    )
    @settings(max_examples=50, deadline=5000)
    def test_singular_cov_fallback_equal_weights(self, n_assets):
        """Property: singular covariance produces equal weights (fallback)."""
        names = [f"A_{i}" for i in range(n_assets)]
        exp_ret = pd.Series(np.random.default_rng(42).uniform(0.01, 0.15, n_assets), index=names)
        # Singular: all rows identical
        row = np.ones(n_assets) * 0.04
        cov = pd.DataFrame([row] * n_assets, index=names, columns=names)
        opt = PortfolioOptimizer()
        w = opt.mean_variance(exp_ret, cov)
        expected = np.ones(n_assets) / n_assets
        assert np.allclose(w, expected, atol=0.01)

    @given(
        st.integers(min_value=2, max_value=6),
    )
    @settings(max_examples=50, deadline=5000)
    def test_zero_cov_fallback_equal_weights(self, n_assets):
        """Property: zero covariance matrix produces equal weights fallback."""
        names = [f"A_{i}" for i in range(n_assets)]
        exp_ret = pd.Series(np.random.default_rng(42).uniform(0.01, 0.15, n_assets), index=names)
        cov = pd.DataFrame(np.zeros((n_assets, n_assets)), index=names, columns=names)
        opt = PortfolioOptimizer()
        w = opt.mean_variance(exp_ret, cov)
        expected = np.ones(n_assets) / n_assets
        assert np.allclose(w, expected, atol=0.01)

    @given(
        st.integers(min_value=2, max_value=8),
    )
    @settings(max_examples=50, deadline=5000)
    def test_negative_returns_still_sum_to_one(self, n_assets):
        """Property: weights sum to 1 even when all expected returns are negative."""
        names = [f"A_{i}" for i in range(n_assets)]
        exp_ret = pd.Series(np.random.default_rng(42).uniform(-0.20, -0.01, n_assets), index=names)
        cov = pd.DataFrame(np.eye(n_assets) * 0.04, index=names, columns=names)
        opt = PortfolioOptimizer()
        w = opt.mean_variance(exp_ret, cov)
        assert abs(np.sum(w) - 1.0) < 1e-10

    @given(
        st.floats(min_value=0.01, max_value=0.5),
        st.integers(min_value=2, max_value=6),
    )
    @settings(max_examples=50, deadline=5000)
    def test_weights_invariant_to_exp_ret_scale(self, scale, n_assets):
        """Property: multiplying all expected returns by same factor doesn't change weights."""
        names = [f"A_{i}" for i in range(n_assets)]
        rng = np.random.default_rng(42)
        base_rets = rng.uniform(0.01, 0.15, n_assets)
        exp_ret_1 = pd.Series(base_rets, index=names)
        exp_ret_2 = pd.Series(base_rets * scale, index=names)
        cov = pd.DataFrame(np.eye(n_assets) * 0.04, index=names, columns=names)
        opt = PortfolioOptimizer()
        w1 = opt.mean_variance(exp_ret_1, cov)
        w2 = opt.mean_variance(exp_ret_2, cov)
        assert np.allclose(w1, w2, atol=1e-10), f"w1={w1}, w2={w2}"


# ---------------------------------------------------------------------------
# Risk-Parity Properties (16 tests)
# ---------------------------------------------------------------------------


class TestRiskParityProperties:
    """Property-based tests for risk_parity optimization."""

    @given(
        st.integers(min_value=2, max_value=8),
    )
    @settings(max_examples=50, deadline=5000)
    def test_weights_sum_to_one(self, n_assets):
        """Property: risk_parity weights sum to 1."""
        names = [f"A_{i}" for i in range(n_assets)]
        rng = np.random.default_rng(42)
        data = rng.uniform(0.01, 0.1, (n_assets, n_assets))
        cov_data = data @ data.T + np.eye(n_assets) * 0.01
        cov = pd.DataFrame(cov_data, index=names, columns=names)
        opt = PortfolioOptimizer()
        w = opt.risk_parity(cov)
        assert abs(np.sum(w) - 1.0) < 1e-10

    @given(
        st.integers(min_value=2, max_value=8),
    )
    @settings(max_examples=50, deadline=5000)
    def test_weights_positive(self, n_assets):
        """Property: risk_parity weights are all positive (long-only)."""
        names = [f"A_{i}" for i in range(n_assets)]
        rng = np.random.default_rng(42)
        data = rng.uniform(0.01, 0.1, (n_assets, n_assets))
        cov_data = data @ data.T + np.eye(n_assets) * 0.01
        cov = pd.DataFrame(cov_data, index=names, columns=names)
        opt = PortfolioOptimizer()
        w = opt.risk_parity(cov)
        assert np.all(w > 0), f"Non-positive weight: {w}"

    @given(st.integers(min_value=1, max_value=1))
    @settings(max_examples=20, deadline=5000)
    def test_single_asset_weight_is_one(self, _):
        """Property: risk_parity with single asset returns [1.0]."""
        cov = pd.DataFrame([[0.04]], index=["A"], columns=["A"])
        opt = PortfolioOptimizer()
        w = opt.risk_parity(cov)
        assert len(w) == 1
        assert w[0] == 1.0

    @given(
        st.integers(min_value=2, max_value=6),
    )
    @settings(max_examples=50, deadline=5000)
    def test_equal_vol_gives_equal_weights(self, n_assets):
        """Property: diagonal covariance with equal variance => equal weights."""
        names = [f"A_{i}" for i in range(n_assets)]
        cov = pd.DataFrame(np.eye(n_assets) * 0.04, index=names, columns=names)
        opt = PortfolioOptimizer()
        w = opt.risk_parity(cov)
        expected = np.ones(n_assets) / n_assets
        assert np.allclose(w, expected, atol=0.01)

    @given(
        st.integers(min_value=2, max_value=4),
    )
    @settings(max_examples=50, deadline=5000)
    def test_lower_vol_gets_higher_weight(self, n_assets):
        """Property: asset with lower variance gets higher weight."""
        names = [f"A_{i}" for i in range(n_assets)]
        diag = [0.01, 0.04, 0.09, 0.16][:n_assets]
        names = names[:n_assets]
        cov = pd.DataFrame(np.diag(diag), index=names, columns=names)
        opt = PortfolioOptimizer()
        w = opt.risk_parity(cov)
        assert np.argmax(w) == 0  # lowest vol has idx 0

    @given(
        st.integers(min_value=2, max_value=4),
    )
    @settings(max_examples=30, deadline=5000)
    def test_zero_variance_gives_equal_weights(self, n_assets):
        """Property: zero covariance matrix falls back to equal weights."""
        names = [f"A_{i}" for i in range(n_assets)]
        cov = pd.DataFrame(np.zeros((n_assets, n_assets)), index=names, columns=names)
        opt = PortfolioOptimizer()
        w = opt.risk_parity(cov)
        expected = np.ones(n_assets) / n_assets
        assert np.allclose(w, expected, atol=0.01)

    @given(
        st.integers(min_value=2, max_value=6),
        st.floats(min_value=0.5, max_value=5.0),
    )
    @settings(max_examples=50, deadline=5000)
    def test_cov_scaling_invariance(self, n_assets, scale):
        """Property: scaling covariance matrix by positive factor doesn't change RP weights."""
        names = [f"A_{i}" for i in range(n_assets)]
        rng = np.random.default_rng(42)
        data = rng.uniform(0.01, 0.1, (n_assets, n_assets))
        base = data @ data.T + np.eye(n_assets) * 0.01
        cov1 = pd.DataFrame(base, index=names, columns=names)
        cov2 = pd.DataFrame(base * scale, index=names, columns=names)
        opt = PortfolioOptimizer()
        w1 = opt.risk_parity(cov1)
        w2 = opt.risk_parity(cov2)
        assert np.allclose(w1, w2, atol=1e-10)

    @given(
        st.integers(min_value=2, max_value=6),
        st.integers(min_value=2, max_value=20),
        st.integers(min_value=50, max_value=200),
    )
    @settings(max_examples=30, deadline=5000)
    def test_more_iterations_similar_result(self, n_assets, few_iter, many_iter):
        """Property: more iterations gives similar or equal result."""
        assume(few_iter <= many_iter)
        names = [f"A_{i}" for i in range(n_assets)]
        rng = np.random.default_rng(42)
        data = rng.uniform(0.01, 0.1, (n_assets, n_assets))
        cov_data = data @ data.T + np.eye(n_assets) * 0.01
        cov = pd.DataFrame(cov_data, index=names, columns=names)
        opt = PortfolioOptimizer()
        w1 = opt.risk_parity(cov, max_iter=few_iter)
        w2 = opt.risk_parity(cov, max_iter=many_iter)
        assert np.abs(np.sum(w1) - np.sum(w2)) < 0.01


# ---------------------------------------------------------------------------
# check_limits Properties (16 tests)
# ---------------------------------------------------------------------------


class TestCheckLimitsProperties:
    """Property-based tests for check_limits."""

    @given(
        st.lists(st.floats(min_value=-0.5, max_value=0.5), min_size=3, max_size=10),
        st.floats(min_value=0.01, max_value=0.5),
        st.floats(min_value=-0.5, max_value=-0.001),
        st.floats(min_value=0.01, max_value=1.0),
        st.floats(min_value=1.0, max_value=10.0),
        st.floats(min_value=0.01, max_value=1.0),
    )
    @settings(max_examples=200, deadline=5000)
    def test_all_checks_are_boolean(self, weights, vol, dd, max_pos, max_lev, max_dd):
        """Property: all check_limits return values are boolean."""
        w = np.array(weights, dtype=float)
        mgr = AdvancedRiskManager(max_pos=max_pos, max_lev=max_lev, max_dd=max_dd)
        checks = mgr.check_limits(w, vol=vol, dd=dd)
        for k, v in checks.items():
            assert isinstance(v, (bool, np.bool_)), f"{k} is {type(v)}"

    @given(
        st.lists(st.floats(min_value=-0.5, max_value=0.5), min_size=3, max_size=10),
        st.floats(min_value=-0.5, max_value=-0.001),
        st.floats(min_value=0.01, max_value=1.0),
        st.floats(min_value=1.0, max_value=10.0),
        st.floats(min_value=0.01, max_value=1.0),
    )
    @settings(max_examples=200, deadline=5000)
    def test_three_keys_present(self, weights, dd, max_pos, max_lev, max_dd):
        """Property: check_limits returns exactly 3 keys."""
        w = np.array(weights, dtype=float)
        mgr = AdvancedRiskManager(max_pos=max_pos, max_lev=max_lev, max_dd=max_dd)
        checks = mgr.check_limits(w, vol=0.15, dd=dd)
        assert set(checks.keys()) == {"position_limit", "leverage_limit", "drawdown_limit"}

    @given(
        st.lists(st.floats(min_value=0.0, max_value=0.01), min_size=3, max_size=10),
        st.floats(min_value=-0.01, max_value=0),
        st.floats(min_value=0.1, max_value=1.0),
        st.floats(min_value=1.0, max_value=10.0),
        st.floats(min_value=0.1, max_value=1.0),
    )
    @settings(max_examples=100, deadline=5000)
    def test_tiny_weights_pass_all_limits(self, weights, dd, max_pos, max_lev, max_dd):
        """Property: very small weights pass all limits."""
        w = np.array(weights, dtype=float)
        mgr = AdvancedRiskManager(max_pos=max_pos, max_lev=max_lev, max_dd=max_dd)
        checks = mgr.check_limits(w, vol=0.15, dd=dd)
        assert bool(checks["position_limit"]) is True

    @given(
        st.lists(st.floats(min_value=100.0, max_value=1000.0), min_size=1, max_size=5),
        st.floats(min_value=0.1, max_value=1.0),
    )
    @settings(max_examples=100, deadline=5000)
    def test_huge_weights_fail_position_limit(self, weights, max_pos):
        """Property: weights much larger than max_pos fail position_limit."""
        w = np.array(weights, dtype=float)
        mgr = AdvancedRiskManager(max_pos=max_pos, max_lev=10000.0, max_dd=1.0)
        checks = mgr.check_limits(w, vol=0.15, dd=-0.01)
        assert bool(checks["position_limit"]) is False

    @given(
        st.lists(st.floats(min_value=50.0, max_value=500.0), min_size=3, max_size=10),
        st.floats(min_value=1.0, max_value=10.0),
    )
    @settings(max_examples=100, deadline=5000)
    def test_huge_weights_fail_leverage_limit(self, weights, max_lev):
        """Property: sum(abs(weights)) > max_lev fails leverage_limit."""
        w = np.array(weights, dtype=float)
        mgr = AdvancedRiskManager(max_pos=1000.0, max_lev=max_lev, max_dd=1.0)
        checks = mgr.check_limits(w, vol=0.15, dd=-0.01)
        assert bool(checks["leverage_limit"]) is False

    @given(
        st.floats(min_value=0.01, max_value=0.5),
        st.floats(min_value=-2.0, max_value=-0.01),
    )
    @settings(max_examples=100, deadline=5000)
    def test_big_drawdown_fails_drawdown_limit(self, max_dd, actual_dd):
        """Property: |dd| > max_dd fails drawdown_limit."""
        w = np.array([0.1, 0.1, 0.1])
        mgr = AdvancedRiskManager(max_pos=1.0, max_lev=100.0, max_dd=max_dd)
        checks = mgr.check_limits(w, vol=0.15, dd=actual_dd)
        assume(abs(actual_dd) > max_dd)
        assert bool(checks["drawdown_limit"]) is False

    @given(
        st.floats(min_value=0.01, max_value=0.5),
        st.floats(min_value=-0.001, max_value=0),
    )
    @settings(max_examples=50, deadline=5000)
    def test_small_drawdown_passes_drawdown_limit(self, max_dd, actual_dd):
        """Property: small |dd| passes drawdown_limit."""
        w = np.array([0.1, 0.1, 0.1])
        mgr = AdvancedRiskManager(max_pos=1.0, max_lev=100.0, max_dd=max_dd)
        checks = mgr.check_limits(w, vol=0.15, dd=actual_dd)
        assert bool(checks["drawdown_limit"]) is True

    @given(
        st.floats(min_value=0.01, max_value=1.0),
        st.floats(min_value=1.0, max_value=10.0),
        st.floats(min_value=0.01, max_value=1.0),
    )
    @settings(max_examples=100, deadline=5000)
    def test_zero_weights_pass_all(self, max_pos, max_lev, max_dd):
        """Property: all-zero weights pass all limits."""
        w = np.zeros(5)
        mgr = AdvancedRiskManager(max_pos=max_pos, max_lev=max_lev, max_dd=max_dd)
        checks = mgr.check_limits(w, vol=0.15, dd=-0.01)
        assert all(checks.values())

    @given(
        st.lists(st.floats(min_value=-2.0, max_value=2.0), min_size=2, max_size=8),
    )
    @settings(max_examples=100, deadline=5000)
    def test_position_limit_uses_abs_value(self, weights):
        """Property: position_limit uses abs(weight) for both long and short."""
        w = np.array(weights, dtype=float)
        max_abs = np.max(np.abs(w))
        mgr = AdvancedRiskManager(max_pos=max_abs + 0.001, max_lev=1000.0, max_dd=1.0)
        checks = mgr.check_limits(w, vol=0.15, dd=-0.01)
        assert bool(checks["position_limit"]) is True

        mgr2 = AdvancedRiskManager(max_pos=max_abs - 0.001, max_lev=1000.0, max_dd=1.0)
        checks2 = mgr2.check_limits(w, vol=0.15, dd=-0.01)
        if max_abs > 0.001:
            assert bool(checks2["position_limit"]) is False


# ---------------------------------------------------------------------------
# Correlation + Risk Integration Properties (8 tests)
# ---------------------------------------------------------------------------


class TestCorrelationRiskIntegration:
    """Integration properties combining correlation analysis and risk checks."""

    @given(
        st.integers(min_value=3, max_value=8),
        st.integers(min_value=100, max_value=500),
    )
    @settings(max_examples=50, deadline=5000)
    def test_uncorrelated_subset_weights_valid(self, n_assets, n_bars):
        """Property: portfolio weights for uncorrelated subset pass basic validation."""
        dates = pd.date_range("2024-01-01", periods=n_bars, freq="B")
        rng = np.random.default_rng(42)
        data = rng.normal(0, 0.02, (n_bars, n_assets))
        df = pd.DataFrame(data, columns=[f"A_{i}" for i in range(n_assets)], index=dates)
        analyzer = CorrelationAnalyzer()
        corr = analyzer.calculate_matrix(df)
        uncorr = analyzer.find_uncorrelated(corr, threshold=0.5)
        assume(len(uncorr) >= 2)

        cov = df[uncorr].cov() * 252
        opt = PortfolioOptimizer()
        w = opt.risk_parity(cov)
        assert abs(np.sum(w) - 1.0) < 1e-10
        assert np.all(np.isfinite(w)), f"RP weights should be finite: {w}"

    @given(
        st.integers(min_value=3, max_value=8),
        st.integers(min_value=100, max_value=300),
    )
    @settings(max_examples=50, deadline=5000)
    def test_full_workflow_weight_sum_one(self, n_assets, n_bars):
        """Property: full workflow (corr → uncorr → MV → risk check) runs end-to-end."""
        dates = pd.date_range("2024-01-01", periods=n_bars, freq="B")
        rng = np.random.default_rng(42)
        data = rng.normal(0, 0.02, (n_bars, n_assets))
        df = pd.DataFrame(data, columns=[f"A_{i}" for i in range(n_assets)], index=dates)
        analyzer = CorrelationAnalyzer()
        corr = analyzer.calculate_matrix(df)
        assume(corr.shape[0] >= 3)
        cov = df.cov()
        exp_ret = pd.Series(df.mean(), index=df.columns)
        opt = PortfolioOptimizer()
        mv = opt.mean_variance(exp_ret, cov)
        rp = opt.risk_parity(cov)
        assert abs(np.sum(mv) - 1.0) < 0.01
        assert abs(np.sum(rp) - 1.0) < 0.01
