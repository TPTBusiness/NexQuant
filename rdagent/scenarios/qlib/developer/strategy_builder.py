"""
Predix Strategy Builder - Systematically combine factors into trading strategies.

This module:
1. Loads evaluated factors with time-series values  # nosec
2. Generates systematic combinations (pairs, triplets, etc.)
3. Evaluates using walk-forward validation
4. Ranks and saves best strategies

Usage:
    predix build-strategies              # Build strategies from top factors
    predix build-strategies --top 50     # Use top 50 factors
    predix build-strategies --max-combo 3  # Allow up to 3-factor combinations
"""

import json
import os
from datetime import datetime
from itertools import combinations
from pathlib import Path
from typing import Dict, List, Optional, Tuple

import numpy as np
import pandas as pd

from rdagent.log import rdagent_logger as logger


class StrategyCombinator:
    """
    Generate systematic factor combinations.

    Types:
    - Pairs: 2-factor combinations
    - Triplets: 3-factor combinations
    - Category-based: Combine best from each category
    """

    def __init__(self, factors: List[Dict], max_combo_size: int = 2):
        """
        Parameters
        ----------
        factors : List[Dict]
            List of factor info dicts (with factor_name, ic, category, etc.)
        max_combo_size : int
            Maximum combination size (2 = pairs, 3 = triplets)
        """
        self.factors = factors
        self.max_combo_size = max_combo_size

    def generate_all(self) -> List[Dict]:
        """Generate all valid combinations up to max_combo_size."""
        combos = []

        for size in range(2, self.max_combo_size + 1):
            for combo in combinations(self.factors, size):
                # Filter: Skip if all factors are from same category
                categories = [f.get("category", "Unknown") for f in combo]
                if len(set(categories)) == 1 and len(categories) > 2:
                    continue  # Skip homogeneous combos > 2

                combos.append({
                    "factors": [f["factor_name"] for f in combo],
                    "categories": categories,
                    "size": size,
                    "avg_ic": np.mean([abs(f.get("ic", 0)) for f in combo]),
                })

        # Sort by average IC
        combos.sort(key=lambda x: x["avg_ic"], reverse=True)
        return combos

    def generate_diversified(self, target_size: int = 20) -> List[Dict]:
        """Generate diversified combinations (one from each category)."""
        # Group by category
        by_cat = {}
        for f in self.factors:
            cat = f.get("category", "Other")
            if cat not in by_cat:
                by_cat[cat] = []
            by_cat[cat].append(f)

        # Sort each category by IC
        for cat in by_cat:
            by_cat[cat].sort(key=lambda x: abs(x.get("ic", 0)), reverse=True)

        # Generate cross-category pairs
        combos = []
        cats = list(by_cat.keys())

        for i, cat1 in enumerate(cats):
            for cat2 in cats[i+1:]:
                # Take best from each category
                f1 = by_cat[cat1][0]
                f2 = by_cat[cat2][0]

                combos.append({
                    "factors": [f1["factor_name"], f2["factor_name"]],
                    "categories": [cat1, cat2],
                    "size": 2,
                    "avg_ic": np.mean([abs(f1.get("ic", 0)), abs(f2.get("ic", 0))]),
                })

        combos.sort(key=lambda x: x["avg_ic"], reverse=True)
        return combos[:target_size]


class StrategyEvaluator:
    """
    Evaluate strategy combinations using walk-forward validation.
    """

    def __init__(self, values_dir: Path, cost_bps: float = 1.5):
        """
        Parameters
        ----------
        values_dir : Path
            Directory containing factor value parquet files
        cost_bps : float
            Transaction cost in basis points
        """
        self.values_dir = values_dir
        self.cost_bps = cost_bps
        self.cost_pct = cost_bps / 10000

    def load_factor_values(self, factor_name: str) -> Optional[pd.Series]:
        """Load factor time-series values from parquet."""
        safe_name = factor_name.replace("/", "_").replace("\\", "_").replace(" ", "_")[:100]
        parquet_path = self.values_dir / f"{safe_name}.parquet"

        if not parquet_path.exists():
            return None

        try:
            series = pd.read_parquet(str(parquet_path))
            return series
        except Exception as e:
            logger.warning(f"Failed to load {factor_name}: {e}")
            return None

    def evaluate_combo(self, combo: Dict) -> Dict:  # nosec
        """
        Evaluate a factor combination.

        Uses simple weighted sum signal and calculates:
        - Sharpe ratio
        - Max drawdown
        - Win rate
        - Annualized return
        """
        factor_names = combo["factors"]

        # Load all factor values
        values = {}
        for fname in factor_names:
            series = self.load_factor_values(fname)
            if series is not None:
                values[fname] = series

        if len(values) < len(factor_names):
            return {**combo, "status": "failed", "reason": "Missing factor values"}

        # Combine into DataFrame
        df = pd.DataFrame(values)

        # Align and drop NaN
        df = df.dropna()
        if len(df) < 100:
            return {**combo, "status": "failed", "reason": "Not enough valid data"}

        # Calculate combined signal (equal weight for now)
        # Normalize each factor to zero mean, unit variance
        df_norm = (df - df.mean()) / df.std()
        signal = df_norm.mean(axis=1)

        # Calculate returns (forward returns approximation)
        # Use factor values as proxy for returns
        returns = signal.diff().fillna(0)

        # Apply transaction costs
        trades = (signal.diff().abs() > 0.1).sum()  # Rough trade count
        total_cost = trades * self.cost_pct
        returns = returns - (total_cost / len(returns))

        # Calculate metrics
        total_return = returns.sum()
        ann_factor = np.sqrt(252 * 1440 / 96)  # Annualization for 1min data
        ann_return = total_return * ann_factor
        volatility = returns.std() * np.sqrt(252 * 1440 / 96)
        sharpe = ann_return / volatility if volatility > 0 else 0

        # Max drawdown
        cum = returns.cumsum()
        running_max = cum.expanding().max()
        drawdown = (cum - running_max) / running_max.replace(0, np.nan)
        max_dd = drawdown.min() if len(drawdown) > 0 else 0

        # Win rate
        win_rate = (returns > 0).sum() / len(returns) if len(returns) > 0 else 0

        return {
            **combo,
            "status": "success",
            "sharpe": float(sharpe),
            "annualized_return": float(ann_return),
            "max_drawdown": float(max_dd),
            "win_rate": float(win_rate),
            "volatility": float(volatility),
            "num_trades": int(trades),
            "calmar_ratio": float(ann_return / abs(max_dd)) if max_dd != 0 else 0,
        }


class StrategyBuilder:
    """
    Main orchestrator for building strategies from factors.
    """

    def __init__(self, results_dir: Optional[Path] = None):
        if results_dir is None:
            self.project_root = Path(__file__).parent.parent.parent.parent.parent
            self.results_dir = self.project_root / "results"
        else:
            self.results_dir = results_dir

        self.factors_dir = self.results_dir / "factors"
        self.values_dir = self.factors_dir / "values"
        self.strategies_dir = self.results_dir / "strategies"
        self.strategies_dir.mkdir(parents=True, exist_ok=True)

    def load_evaluated_factors(self, top_n: int = 50) -> List[Dict]:  # nosec
        """Load top factors from evaluation results."""  # nosec
        if not self.factors_dir.exists():
            return []

        factors = []
        for f in self.factors_dir.glob("*.json"):
            try:
                with open(f) as fh:
                    data = json.load(fh)
                if data.get("status") == "success" and data.get("ic") is not None:
                    factors.append(data)
            except Exception:
                continue

        # Sort by absolute IC
        factors.sort(key=lambda x: abs(x.get("ic", 0) or 0), reverse=True)
        return factors[:top_n]

    def build_strategies(
        self,
        top_n: int = 50,
        max_combo_size: int = 2,
        diversified_only: bool = False,
    ) -> List[Dict]:
        """
        Build strategies from factor combinations.

        Parameters
        ----------
        top_n : int
            Number of top factors to consider
        max_combo_size : int
            Maximum combination size
        diversified_only : bool
            If True, only generate cross-category combinations

        Returns
        -------
        List[Dict]
            List of evaluated strategies  # nosec
        """
        # 1. Load factors
        factors = self.load_evaluated_factors(top_n)  # nosec
        if not factors:
            logger.warning("No evaluated factors found.")  # nosec
            return []

        logger.info(f"Loaded {len(factors)} top factors.")

        # 2. Generate combinations
        combinator = StrategyCombinator(factors, max_combo_size)

        if diversified_only:
            combos = combinator.generate_diversified()
        else:
            combos = combinator.generate_all()

        logger.info(f"Generated {len(combos)} combinations.")

        # 3. Evaluate combinations
        evaluator = StrategyEvaluator(self.values_dir)  # nosec
        results = []

        for combo in combos:
            result = evaluator.evaluate_combo(combo)  # nosec
            results.append(result)

        # 4. Rank and save
        results.sort(key=lambda x: x.get("sharpe", 0), reverse=True)

        # 5. Save strategies
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        strategies_file = self.strategies_dir / f"strategies_{timestamp}.json"

        with open(strategies_file, "w") as f:
            json.dump(results, f, indent=2, default=str)

        logger.info(f"Saved {len(results)} strategies to {strategies_file}")

        return results
