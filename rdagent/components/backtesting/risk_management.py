"""
Predix Risk Management - Korrelation, Portfolio-Optimierung
"""

import numpy as np
import pandas as pd


class CorrelationAnalyzer:
    def __init__(self, lookback: int = 60):
        self.lookback = lookback

    def calculate_matrix(self, returns: pd.DataFrame) -> pd.DataFrame:
        return returns.dropna().corr()

    def find_uncorrelated(self, corr: pd.DataFrame, threshold: float = 0.3) -> list[str]:
        result = []
        for f in corr.columns:
            others = [x for x in corr.columns if x != f]
            if corr.loc[f, others].abs().mean() < threshold:
                result.append(f)
        return result

class PortfolioOptimizer:
    def mean_variance(self, exp_ret: pd.Series, cov: pd.DataFrame) -> np.ndarray:
        try:
            w = np.linalg.inv(cov.values) @ exp_ret.values
            return w / np.sum(w)
        except (np.linalg.LinAlgError, ValueError):
            return np.ones(len(exp_ret)) / len(exp_ret)

    def risk_parity(self, cov: pd.DataFrame, max_iter: int = 100) -> np.ndarray:
        n = cov.shape[0]
        w = np.ones(n) / n
        for _ in range(max_iter):
            marginal = cov.values @ w
            vol = np.sqrt(w @ cov.values @ w)
            if vol == 0: break
            risk_contrib = w * marginal / vol
            scale = np.sum(risk_contrib) / (n * risk_contrib + 1e-10)
            new_w = w * scale
            new_w = new_w / np.sum(new_w)
            if np.max(np.abs(new_w - w)) < 1e-6: break
            w = new_w
        return w

class AdvancedRiskManager:
    def __init__(self, max_pos: float = 0.2, max_lev: float = 5.0, max_dd: float = 0.20):
        self.max_pos = max_pos
        self.max_lev = max_lev
        self.max_dd = max_dd
        self.corr_analyzer = CorrelationAnalyzer()
        self.optimizer = PortfolioOptimizer()

    def check_limits(self, weights: np.ndarray, vol: float, dd: float) -> dict[str, bool]:
        return {
            "position_limit": np.max(np.abs(weights)) <= self.max_pos,
            "leverage_limit": np.sum(np.abs(weights)) <= self.max_lev,
            "drawdown_limit": abs(dd) <= self.max_dd,
        }

if __name__ == "__main__":
    print("=== Risk Test ===")
    np.random.seed(42)
    n, names = 252, ["Mom", "MeanRev", "Vol", "Volu", "ML"]
    ret = pd.DataFrame(np.random.randn(n, 5), columns=names)

    corr = CorrelationAnalyzer().calculate_matrix(ret)
    print("Korrelationsmatrix:")
    print(corr.round(2))

    opt = PortfolioOptimizer()
    exp_ret = pd.Series([0.1, 0.08, 0.06, 0.07, 0.12], index=names)
    cov = ret.cov() * 252

    mv = opt.mean_variance(exp_ret, cov)
    print("\nMean-Variance:")
    for n, w in zip(names, mv): print(f"  {n}: {w:.2%}")

    rp = opt.risk_parity(cov)
    print("\nRisk Parity:")
    for n, w in zip(names, rp): print(f"  {n}: {w:.2%}")

    rm = AdvancedRiskManager()
    checks = rm.check_limits(mv, 0.15, -0.08)
    print(f"\nLimits OK: {all(checks.values())}")
    print("✅ Test bestanden!")
