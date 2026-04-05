# StrategyBuilder — Architektur-Design

## Überblick

Der **StrategyBuilder** kombiniert existierende Faktoren systematisch zu handelbaren Strategien. 
Im Gegensatz zum ML-Trainer (der ein einzelnes Modell auf Top-Faktoren trainiert) testet der 
StrategyBuilder **explizite Kombinationsregeln** mit Walk-Forward-Validierung.

---

## 1. Klassen-Design

### 1.1 StrategyCombinator

**Zweck:** Generiert systematische Faktorkombinationen nach verschiedenen Strategien.

```python
# rdagent/scenarios/qlib/developer/strategy_builder.py

class CombinationStrategy(Enum):
    """Supported combination methods."""
    PAIR = "pair"                    # Top-N pairs by IC product
    TRIPLET = "triplet"              # Top triplets
    CATEGORY = "category"            # All factors of same type
    TEMPORAL = "temporal"            # Session/time-specific combos
    CUSTOM = "custom"                # User-defined combinations


@dataclass
class StrategySpec:
    """Defines a single strategy configuration."""
    name: str
    factors: List[str]               # Factor names to combine
    combination_type: str            # "weighted_sum", "regime_switch", etc.
    weighting: str                   # "equal", "ic_weighted", "risk_parity"
    metadata: Dict[str, Any]         # Additional context (category, session, etc.)


class StrategyCombinator:
    """Generate factor combinations systematically."""

    def __init__(
        self,
        factors_db: ResultsDatabase,
        min_ic: float = 0.02,
        max_factors_per_strategy: int = 5,
    ) -> None: ...

    def load_valid_factors(self, min_ic: float = 0.02) -> pd.DataFrame:
        """Load all factors with IC >= threshold from DB."""
        ...

    def generate_pairs(
        self,
        top_n: int = 50,
        max_correlation: float = 0.7,
    ) -> List[StrategySpec]:
        """
        Generate pairwise combinations.
        
        Rules:
        - Take top_n factors by |IC|
        - Filter pairs with correlation < max_correlation
        - Score by |IC1 * IC2| (both must have predictive power)
        - Prefer complementary pairs (one positive IC, one negative)
        """
        ...

    def generate_triplets(
        self,
        top_n: int = 30,
        max_pairwise_corr: float = 0.5,
    ) -> List[StrategySpec]:
        """
        Generate triplet combinations.
        
        Rules:
        - Top 30 factors by |IC|
        - All pairwise correlations < max_pairwise_corr
        - Score by geometric mean of |IC|
        """
        ...

    def generate_category_combos(
        self,
        category: str,
        min_factors: int = 2,
        max_factors: int = 5,
    ) -> List[StrategySpec]:
        """
        Combine all factors within a category.
        
        Categories (inferred from factor names):
        - "Momentum": mom_*, trend_*
        - "Mean Reversion": mean_rev_*, reversal_*
        - "Volatility": vol_*, std_*
        - "Session": session_*, intraday_*
        - "Volume": volume_*, turnover_*
        """
        ...

    def generate_temporal_combos(
        self,
        session_filters: Dict[str, Callable],
    ) -> List[StrategySpec]:
        """
        Generate session-specific combinations.
        
        Example strategies:
        - "London Open": Use momentum factors 07:00-09:00 UTC
        - "NY Close": Use mean reversion 14:00-16:00 UTC
        - "Asian Session": Use volatility factors 00:00-06:00 UTC
        """
        ...

    def generate_custom_combo(
        self,
        factor_names: List[str],
        weighting: str = "equal",
    ) -> StrategySpec:
        """User-defined combination for testing specific hypotheses."""
        ...

    def generate_all(
        self,
        strategies: List[CombinationStrategy] = None,
    ) -> List[StrategySpec]:
        """
        Run all enabled combination strategies.
        
        Default: PAIR + TRIPLET + CATEGORY
        Returns list of all StrategySpec objects.
        """
        ...
```

---

### 1.2 StrategyEvaluator

**Zweck:** Walk-Forward-Backtesting für Strategien mit Transaktionskosten.

```python
@dataclass
class WalkForwardConfig:
    """Walk-forward validation configuration."""
    train_window: int = 30           # Days for training
    test_window: int = 5             # Days for out-of-sample testing
    step_size: int = 5               # Days to slide forward
    min_train_periods: int = 3       # Minimum windows before first test


@dataclass
class TransactionCostModel:
    """Realistic transaction cost modeling."""
    cost_per_trade_bps: float = 1.5  # 1.5 bps per trade
    slippage_bps: float = 0.5        # Additional slippage
    min_trade_size: float = 0.01     # Minimum position size


class StrategyMetrics:
    """Complete metrics for a validated strategy."""
    
    def __init__(self, strategy_name: str) -> None: ...
    
    def update(
        self,
        window_idx: int,
        in_sample_ic: float,
        out_of_sample_ic: float,
        oos_sharpe: float,
        oos_return: float,
        oos_drawdown: float,
        n_trades: int,
        transaction_costs: float,
    ) -> None: ...
    
    def finalize(self) -> Dict[str, Any]:
        """
        Calculate aggregate metrics:
        
        - Mean OOS IC
        - IC decay (IS IC vs OOS IC)
        - Mean OOS Sharpe
        - Worst OOS Drawdown
        - Calmar Ratio (Ann Return / Max DD)
        - Total transaction costs
        - Win rate across windows
        - Consistency score (% windows with positive IC)
        """
        ...


class StrategyEvaluator:
    """Walk-forward backtesting for strategy combinations."""

    def __init__(
        self,
        data_source: str,              # Path to intraday_pv.h5
        wf_config: WalkForwardConfig = None,
        cost_model: TransactionCostModel = None,
    ) -> None: ...

    def load_factor_values(
        self,
        factor_names: List[str],
    ) -> Dict[str, pd.Series]:
        """Load time series values for each factor."""
        ...

    def compute_combined_signal(
        self,
        factor_values: Dict[str, pd.Series],
        weights: Dict[str, float],
        combination_type: str = "weighted_sum",
    ) -> pd.Series:
        """
        Combine factors into single signal.
        
        Types:
        - "weighted_sum": sum(w_i * factor_i)
        - "regime_switch": use different factors per regime
        - "timing": use volatility to scale momentum
        """
        ...

    def walk_forward_backtest(
        self,
        strategy_spec: StrategySpec,
    ) -> StrategyMetrics:
        """
        Run walk-forward validation for a single strategy.
        
        Process:
        1. Split time series into rolling windows
        2. For each window:
           a. Optimize weights on train period
           b. Test on out-of-sample period
           c. Apply transaction costs
           d. Record metrics
        3. Aggregate across all windows
        
        Returns StrategyMetrics with full validation results.
        """
        ...

    def backtest_single_window(
        self,
        train_data: pd.DataFrame,
        test_data: pd.DataFrame,
        strategy_spec: StrategySpec,
    ) -> Dict[str, float]:
        """
        Backtest strategy on single train/test split.
        
        Steps:
        1. Compute factor values on train period
        2. Optimize weights (IC-weighted or risk parity)
        3. Apply to test period
        4. Calculate returns with transaction costs
        5. Return metrics
        """
        ...

    def apply_transaction_costs(
        self,
        raw_returns: pd.Series,
        signals: pd.Series,
        cost_model: TransactionCostModel,
    ) -> pd.Series:
        """
        Deduct transaction costs from returns.
        
        Cost = (signal changes) * (cost_per_trade + slippage)
        Only charged when position actually changes.
        """
        ...
```

---

### 1.3 StrategySelector

**Zweck:** Selektiere beste Strategien nach Out-of-Sample-Performance.

```python
@dataclass
class StrategyRanking:
    """Ranking criteria for strategies."""
    primary_metric: str = "oos_sharpe"     # oos_sharpe, calmar, oos_ic
    min_oos_ic: float = 0.02               # Minimum OOS IC
    max_drawdown: float = -0.15            # Maximum allowed drawdown
    min_consistency: float = 0.6           # % of windows with positive IC
    min_windows: int = 3                   # Minimum validation windows


class StrategySelector:
    """Select and rank best strategies based on walk-forward results."""

    def __init__(
        self,
        ranking: StrategyRanking = None,
    ) -> None: ...

    def rank_strategies(
        self,
        strategy_results: List[Dict[str, Any]],
    ) -> pd.DataFrame:
        """
        Rank strategies by primary metric.
        
        Filters:
        - OOS IC >= min_oos_ic
        - Max DD <= max_drawdown threshold
        - Consistency >= min_consistency
        - At least min_windows validated
        
        Returns sorted DataFrame with:
        - strategy_name
        - oos_sharpe (primary)
        - oos_ic_mean
        - ic_decay (IS vs OOS gap)
        - calmar_ratio
        - max_drawdown
        - consistency_score
        - n_windows
        - total_transaction_costs
        """
        ...

    def select_top_k(
        self,
        ranked: pd.DataFrame,
        k: int = 10,
    ) -> List[Dict[str, Any]]:
        """Return top K strategies passing all filters."""
        ...

    def identify_overfitting(
        self,
        strategy_results: List[Dict[str, Any]],
        ic_decay_threshold: float = 0.5,
    ) -> List[str]:
        """
        Flag strategies where OOS IC < 50% of IS IC.
        Indicates overfitting to training period.
        """
        ...

    def recommend_ensemble(
        self,
        ranked: pd.DataFrame,
        max_correlation: float = 0.3,
        max_strategies: int = 3,
    ) -> List[str]:
        """
        Recommend ensemble of uncorrelated strategies.
        
        Select up to max_strategies with:
        - Highest combined Sharpe
        - Pairwise correlation < max_correlation
        """
        ...
```

---

### 1.4 StrategySaver

**Zweck:** Persistiert Strategien in `results/strategies/`.

```python
class StrategySaver:
    """Save validated strategies to results/strategies/."""

    def __init__(
        self,
        strategies_dir: Optional[str] = None,
    ) -> None:
        project_root = Path(__file__).parent.parent.parent.parent
        self.strategies_dir = Path(strategies_dir) if strategies_dir \
            else project_root / "results" / "strategies"
        self.strategies_dir.mkdir(parents=True, exist_ok=True)

    def save_strategy(
        self,
        strategy_spec: StrategySpec,
        metrics: Dict[str, Any],
        ranking: Dict[str, Any] = None,
    ) -> Path:
        """
        Save complete strategy to JSON.
        
        JSON structure:
        {
            "name": "momentum_mean_rev_pair",
            "created_at": "2026-04-05T12:00:00",
            "combination_type": "pair",
            "factors": ["Momentum_v3", "MeanReversion_v2"],
            "weights": {"Momentum_v3": 0.63, "MeanReversion_v2": 0.37},
            "weighting_method": "ic_weighted",
            
            "walk_forward": {
                "train_window_days": 30,
                "test_window_days": 5,
                "n_windows": 8,
                "total_test_days": 40
            },
            
            "metrics": {
                "oos_ic_mean": 0.045,
                "oos_ic_std": 0.012,
                "is_ic_mean": 0.062,
                "ic_decay": 0.27,
                "oos_sharpe": 2.15,
                "oos_annualized_return": 0.128,
                "oos_max_drawdown": -0.089,
                "calmar_ratio": 1.44,
                "consistency_score": 0.875,
                "win_rate": 0.58,
                "total_transaction_costs_bps": 12.4,
                "net_sharpe": 1.98
            },
            
            "per_window_metrics": [
                {"window": 0, "oos_ic": 0.051, "oos_sharpe": 2.3, ...},
                {"window": 1, "oos_ic": 0.038, "oos_sharpe": 1.9, ...},
                ...
            ],
            
            "ranking": {
                "rank_by_sharpe": 3,
                "rank_by_ic": 5,
                "rank_by_calmar": 2,
                "passes_filters": true
            }
        }
        """
        ...

    def load_all_strategies(
        self,
        min_oos_sharpe: float = None,
    ) -> List[Dict[str, Any]]:
        """Load all saved strategies, optionally filtered."""
        ...

    def load_best_strategy(self) -> Optional[Dict[str, Any]]:
        """Load the single best strategy by OOS Sharpe."""
        ...
```

---

## 2. Kombinations-Logik

### 2.1 Faktor-Auswahl für Kombinationen

```python
def select_factors_for_combination(
    factors_df: pd.DataFrame,
    min_ic: float = 0.02,
    max_correlation: float = 0.7,
) -> Tuple[List[str], pd.DataFrame]:
    """
    Select factors suitable for combination.
    
    Algorithm:
    1. Filter: |IC| >= min_ic
    2. Compute correlation matrix
    3. Cluster factors by correlation (hierarchical clustering)
    4. From each cluster, pick factor with highest |IC|
    5. Return selected factors + correlation matrix
    
    Rationale:
    - Avoid combining highly correlated factors (redundant)
    - Ensure each selected factor has standalone predictive power
    - Maximize diversity in combinations
    """
    ...
```

### 2.2 Pair-Strategie

```
Regel: Kombiniere Faktor A + B wenn:
  1. |IC_A| >= 0.02 UND |IC_B| >= 0.02
  2. Korrelation(A, B) < 0.7
  3. Score = |IC_A * IC_B| * (1 - corr(A, B))
  
Priorisiere:
  - Momentum + Mean Reversion (komplementär)
  - Volatility + Momentum (Timing)
  - Session + Hauptfaktor (Filter)
```

### 2.3 Triplet-Strategie

```
Regel: Kombiniere Faktor A + B + C wenn:
  1. Alle |IC| >= 0.02
  2. Alle pairwise Korrelationen < 0.5
  3. Score = (|IC_A| * |IC_B| * |IC_C|)^(1/3) * diversity_factor
  
Priorisiere:
  - Momentum + Mean Reversion + Volatility
  - Hauptfaktor + Session + Volatility
  - Drei unkorrelierte Alpha-Faktoren
```

### 2.4 Gewichtungsmethoden

```python
def compute_weights(
    factor_ics: Dict[str, float],
    factor_correlations: pd.DataFrame,
    method: str = "ic_weighted",
) -> Dict[str, float]:
    """
    Compute factor weights.
    
    Methods:
    
    1. "equal": w_i = 1/N
    
    2. "ic_weighted": w_i = |IC_i| / sum(|IC|)
       - Simple, effective when ICs are reliable
       
    3. "risk_parity": 
       - w_i proportional to 1/vol_i
       - Equalize risk contribution from each factor
       - Requires factor return covariance matrix
       
    4. "sharpe_weighted": w_i = Sharpe_i / sum(Sharpe)
       - Weight by risk-adjusted performance
       
    Returns normalized weights summing to 1.0
    """
    ...
```

---

## 3. Walk-Forward-Validierung

### 3.1 Schema

```
Zeitachse (Beispiel: 90 Tage Daten):

[---- Train 30d ----][Test 5d][---- Train 30d ----][Test 5d]...
      Window 0                   Window 1

Gesamt: ~8 Walks bei 90 Tagen
```

### 3.2 Ablauf pro Window

```python
for window_idx in range(n_windows):
    # 1. Define train/test periods
    train_start = window_idx * step_size
    train_end = train_start + train_window
    test_start = train_end
    test_end = test_start + test_window
    
    # 2. Optimize weights on train period
    weights = optimize_weights(
        factor_values[train_start:train_end],
        forward_returns[train_start:train_end],
        method=strategy_spec.weighting,
    )
    
    # 3. Generate signal on test period
    signal = compute_combined_signal(
        factor_values[test_start:test_end],
        weights,
    )
    
    # 4. Calculate returns with costs
    raw_returns = signal.shift(1) * forward_returns[test_start:test_end]
    net_returns = apply_transaction_costs(raw_returns, signal, cost_model)
    
    # 5. Record metrics
    metrics.update(
        window_idx=window_idx,
        in_sample_ic=compute_ic(train_period),
        out_of_sample_ic=compute_ic(test_period),
        oos_sharpe=calculate_sharpe(net_returns),
        oos_drawdown=calculate_max_drawdown(net_returns),
        n_trades=count_signal_changes(signal),
        transaction_costs=raw_returns.sum() - net_returns.sum(),
    )
```

### 3.3 Aggregierte Metriken

```python
final_metrics = {
    # Primary
    "oos_ic_mean": mean(window_oos_ics),
    "oos_ic_std": std(window_oos_ics),
    "oos_sharpe": mean(window_sharpes),
    
    # Overfitting detection
    "is_ic_mean": mean(window_is_ics),
    "ic_decay": 1 - (oos_ic_mean / is_ic_mean),  # < 0.5 good
    
    # Risk
    "oos_max_drawdown": min(window_drawdowns),
    "calmar_ratio": annualized_return / abs(max_drawdown),
    
    # Consistency
    "consistency_score": sum(ic > 0 for ic in window_oos_ics) / n_windows,
    
    # Costs
    "total_transaction_costs_bps": sum(window_costs),
    "net_sharpe": sharpe_after_costs,
}
```

---

## 4. Integrationspunkte mit factor_runner.py

### 4.1 Wo passt der StrategyBuilder hin?

```
Bestehender Flow (factor_runner.py):
┌─────────────────────────────────────────┐
│ 1. Hypothesis Gen → Factor Hypothesis   │
│ 2. Factor Coder → Generate factor code  │
│ 3. Factor Runner → Docker backtest      │
│ 4. Protection Check → Risk validation   │
│ 5. Save to DB → ResultsDatabase         │
│ 6. Feedback → Guide next hypothesis     │
└─────────────────────────────────────────┘

NEUER Flow (StrategyBuilder):
┌─────────────────────────────────────────┐
│ 7. StrategyCombinator → Combos          │  ← AFTER factor generation
│ 8. StrategyEvaluator → Walk-forward     │  ← SEPARATE phase
│ 9. StrategySelector → Rank strategies   │
│ 10. StrategySaver → results/strategies/ │
└─────────────────────────────────────────┘
```

### 4.2 Konkrete Integration

```python
# Option A: Eigenständiger CLI-Befehl (empfohlen)
# rdagent/build_strategies --top-n 100 --walk-forward

# Option B: Integration in QuantRDLoop
class QuantRDLoop:
    def running(self, prev_out):
        # ... existing factor runner code ...
        exp = self.factor_runner.develop(prev_out["coding"])
        
        # NEW: Periodically run strategy builder
        if self.should_build_strategies():
            self._run_strategy_builder()
        
        return exp
    
    def should_build_strategies(self) -> bool:
        """Check if enough factors exist to build strategies."""
        n_factors = self.trace.get_valid_factor_count()
        return n_factors >= 100 and self.loop_idx % 50 == 0
    
    def _run_strategy_builder(self) -> None:
        """Trigger strategy building process."""
        from rdagent.scenarios.qlib.developer.strategy_builder import (
            StrategyBuilder,
        )
        
        builder = StrategyBuilder(
            db=self.results_db,
            data_source=self.data_path,
        )
        builder.run(top_n=100)
```

### 4.3 Datenabhängigkeiten

```python
# Benötigt von factor_runner.py:
# ✅ ResultsDatabase → already exists, factor_runner schreibt dort
# ✅ Factor JSON files → already in results/factors/
# ✅ Factor values → Müssen aus workspace/result.h5 geladen werden

# Neue Abhängigkeit:
# ⚠️ Factor time series values → Müssen für Walk-Forward verfügbar sein
#   Lösung: Factor values beim Speichern in DB auch als Parquet schreiben
```

---

## 5. Integration in QuantRDLoop Workflow

### 5.1 Erweiterte Loop-Phasen

```
Phase 1: Factor Generation (EXISTIEREND)
    └─ Generate → Code → Backtest → Save to DB
    └─ Continue until N factors reached (z.B. 500)

Phase 2: Strategy Building (NEU)
    └─ Load top factors from DB
    └─ Generate combinations (pairs, triplets, categories)
    └─ Walk-forward validation
    └─ Save strategies to results/strategies/

Phase 3: Strategy Selection (NEU)
    └─ Rank by OOS Sharpe
    └─ Filter by max drawdown, consistency
    └─ Select top 3 strategies for live trading

Phase 4: ML Training (EXISTIEREND, optional)
    └─ Train ML model on top strategies' factors
    
Phase 5: Live Trading (ZUKUNFT)
    └─ Paper trade selected strategies
    └─ Monitor and adapt
```

### 5.2 Haupt-CLI-Befehl

```python
# rdagent/scenarios/qlib/developer/strategy_builder.py

class StrategyBuilder:
    """Main orchestrator for strategy building process."""
    
    def __init__(
        self,
        db: ResultsDatabase,
        data_source: str,
        output_dir: Optional[str] = None,
    ) -> None:
        self.db = db
        self.data_source = data_source
        self.combinator = StrategyCombinator(db)
        self.evaluator = StrategyEvaluator(data_source)
        self.selector = StrategySelector()
        self.saver = StrategySaver(output_dir)
    
    def run(
        self,
        top_n: int = 100,
        min_ic: float = 0.02,
        strategies: List[CombinationStrategy] = None,
        save: bool = True,
    ) -> pd.DataFrame:
        """
        Complete strategy building pipeline.
        
        Steps:
        1. Load top N factors from DB
        2. Generate combinations
        3. Walk-forward validate each
        4. Rank and filter
        5. Save top strategies
        6. Return ranked results
        """
        logger.info(f"=== Strategy Builder: Top {top_n} factors ===")
        
        # Step 1: Load factors
        factors = self.combinator.load_valid_factors(min_ic=min_ic)
        logger.info(f"Loaded {len(factors)} valid factors")
        
        # Step 2: Generate combinations
        combos = self.combinator.generate_all(strategies)
        logger.info(f"Generated {len(combos)} strategy combinations")
        
        # Step 3: Walk-forward validate
        results = []
        for spec in combos:
            logger.info(f"Evaluating: {spec.name}")
            metrics = self.evaluator.walk_forward_backtest(spec)
            results.append(metrics.finalize())
        
        # Step 4: Rank
        ranked = self.selector.rank_strategies(results)
        
        # Step 5: Save
        if save:
            for _, row in ranked.iterrows():
                spec = next(s for s in combos if s.name == row["strategy_name"])
                self.saver.save_strategy(spec, row)
        
        logger.info(f"=== Top 5 Strategies ===")
        logger.info(ranked.head(5).to_string())
        
        return ranked


def build_strategies(
    top_n: int = 100,
    min_ic: float = 0.02,
    data_source: str = None,
) -> None:
    """CLI entry point: rdagent build_strategies"""
    from rdagent.components.backtesting.results_db import ResultsDatabase
    
    db = ResultsDatabase()
    
    if data_source is None:
        data_source = str(Path(__file__).parent.parent.parent.parent.parent 
                          / "git_ignore_folder" 
                          / "factor_implementation_source_data" 
                          / "intraday_pv.h5")
    
    builder = StrategyBuilder(db=db, data_source=data_source)
    ranked = builder.run(top_n=top_n, min_ic=min_ic)
    
    logger.info(f"\nStrategy building complete. Results in results/strategies/")
```

### 5.3 Config-Erweiterung

```python
# rdagent/app/qlib_rd_loop/conf.py

@dataclass
class StrategyBuilderSetting:
    """Configuration for strategy building."""
    top_n_factors: int = 100
    min_ic_threshold: float = 0.02
    max_correlation: float = 0.7
    train_window_days: int = 30
    test_window_days: int = 5
    step_size_days: int = 5
    transaction_cost_bps: float = 1.5
    min_oos_sharpe: float = 1.0
    max_drawdown_threshold: float = -0.15
    combination_strategies: List[str] = None  # ["pair", "triplet", "category"]
```

---

## 6. Datei-Struktur

```
rdagent/scenarios/qlib/developer/
└── strategy_builder.py              # Hauptmodul (alle Klassen)

# ODER aufgeteilt:
rdagent/scenarios/qlib/developer/
└── strategy_builder/
    ├── __init__.py
    ├── combinator.py                # StrategyCombinator
    ├── evaluator.py                 # StrategyEvaluator
    ├── selector.py                  # StrategySelector
    ├── saver.py                     # StrategySaver
    └── builder.py                   # StrategyBuilder (Orchestrator)

results/
└── strategies/
    ├── momentum_mean_rev_pair.json
    ├── momentum_vol_timing.json
    ├── session_alpha_combo.json
    └── strategy_ranking.json        # Summary aller Strategien
```

---

## 7. Nächste Schritte

1. **Implementierung Phase 1:** StrategyCombinator + einfache Pair-Tests
2. **Implementierung Phase 2:** StrategyEvaluator mit Walk-Forward
3. **Implementierung Phase 3:** StrategySelector + Saver
4. **Integration:** CLI-Befehl `rdagent build_strategies`
5. **Validierung:** Top-Strategien gegen Hold-out Periode testen
6. **Dashboard:** Web-UI zur Strategie-Anzeige (erweitert)

---

## 8. Offene Fragen

- **Factor Values:** Woher kommen die Zeitreihen-Werte für jeden Faktor?
  - Aktuell: Nur in workspace/result.h5 gespeichert (nicht persistent)
  - Lösung: Beim Speichern in DB auch als Parquet in results/factors/values/ ablegen

- **Performance:** 100 Faktoren → ~5000 Pairs → 8 Walks each = 40.000 Backtests
  - Lösung: Parallelisierung (multiprocessing), Top-1000 Paare vorher filtern

- **Regime Detection:** Wie erkennen wir Markt-Regimes?
  - Vorschlag: Volatility-based (high/low vol), Trend-based (uptrend/downtrend)
  - Später: ML-basiert (HMM, Clustering)
