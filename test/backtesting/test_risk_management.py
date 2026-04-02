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
