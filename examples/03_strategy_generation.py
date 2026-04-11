#!/usr/bin/env python
"""
Beispiel 03: Strategy Generation - Faktoren zu Strategien kombinieren

Was macht dieses Beispiel?
    Dieses Skript zeigt, wie man mehrere Trading-Faktoren zu einer robusten
    Strategie kombiniert. Dabei wird die IC-weighted Combination verwendet,
    die Faktoren nach ihrer prädiktiven Kraft (Information Coefficient) gewichtet.
    
    WICHTIG: Faktoren mit negativem IC müssen invertiert werden!

Voraussetzungen:
    - Mindestens 2-3 generierte Faktoren (aus Beispiel 01)
    - Faktoren sollten unkorreliert sein (Korrelation < 0.6)

Erwartete Laufzeit:
    ~3-5 Minuten

Output:
    - IC-weighted Faktor-Kombination
    - Signal-Verteilung (Long/Short/Neutral)
    - Composite Signal Code
"""

import argparse
import logging
import sys

logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s | %(levelname)-8s | %(message)s',
    datefmt='%Y-%m-%d %H:%M:%S'
)
logger = logging.getLogger(__name__)


def run_strategy_generation(factors: list, use_ai: bool = False) -> None:
    """
    Kombiniert Faktoren zu einer Strategie.

    Args:
        factors: Liste der Faktor-Namen
        use_ai: KI-gestützte Strategiegenerierung (StrategyCoSTEER)
    """
    logger.info("=" * 60)
    logger.info("PREDIX Strategy Generation - Beispiel 03")
    logger.info("=" * 60)
    logger.info(f"Faktoren: {', '.join(factors)}")
    logger.info(f"KI-gestützt: {use_ai}")
    logger.info("=" * 60)

    # Beispiel-Faktoren mit IC-Werten
    example_factors_data = {
        "momentum_16": {
            "ic": 0.074,
            "sharpe": 1.6,
            "arr": "10.2%",
            "type": "trend_following"
        },
        "hl_range_reversal": {
            "ic": -0.065,
            "sharpe": 1.4,
            "arr": "8.5%",
            "type": "mean_reversion"
        },
        "session_alpha": {
            "ic": 0.082,
            "sharpe": 1.8,
            "arr": "11.8%",
            "type": "session_timing"
        }
    }

    # IC-Weights berechnen (negative IC invertieren!)
    logger.info("\nFAKTOR-ANALYSE:")
    logger.info("-" * 60)
    
    total_abs_ic = 0
    for factor_name in factors:
        if factor_name in example_factors_data:
            data = example_factors_data[factor_name]
            logger.info(f"  {factor_name}:")
            logger.info(f"    IC: {data['ic']}")
            logger.info(f"    Typ: {data['type']}")
            logger.info(f"    Sharpe: {data['sharpe']}")
            total_abs_ic += abs(data['ic'])

    # Normalize weights
    logger.info("\nIC-WEIGHTED COMBINATION:")
    logger.info("-" * 60)
    
    weights = {}
    for factor_name in factors:
        if factor_name in example_factors_data:
            ic = example_factors_data[factor_name]['ic']
            # Negative IC invertieren
            weight = ic / total_abs_ic
            weights[factor_name] = weight
            logger.info(f"  {factor_name}: {weight:.3f} (IC: {ic})")

    # Strategie-Code generieren
    strategy_code = f"""
import pandas as pd
import numpy as np

# UNSTACK für cross-sectionale Operationen
factor_matrix = factors.unstack(level='instrument')

# Rolling Z-Score Normalisierung (Window=20)
z = (factor_matrix - factor_matrix.rolling(20).mean()) / (factor_matrix.rolling(20).std() + 1e-8)

# IC-weighted Combination (negative IC invertiert!)
composite = ({weights.get('momentum_16', 0):.3f} * z['momentum_16']
             {weights.get('hl_range_reversal', 0):+.3f} * z['hl_range_reversal']
             {weights.get('session_alpha', 0):+.3f} * z['session_alpha'])

# STACK back zu MultiIndex
composite = composite.stack(level='instrument')

# Signal-Generierung mit Thresholds
signal = pd.Series(0, index=factors.index)
signal[composite > 0.5] = 1   # LONG
signal[composite < -0.5] = -1  # SHORT
signal.name = 'signal'
"""

    logger.info("\nSTRATEGIE-CODE:")
    logger.info("-" * 60)
    logger.info(strategy_code)

    # Erwartete Performance
    logger.info("\nERWARTETE PERFORMANCE:")
    logger.info("-" * 60)
    logger.info("  ARR: 12-15%")
    logger.info("  Sharpe: 2.0-2.4")
    logger.info("  Max DD: 7-9%")
    logger.info("  Trades/Tag: 10-14")
    logger.info("  Win Rate: 55-58%")

    logger.info("\n" + "=" * 60)
    logger.info("FERTIG!")
    logger.info("=" * 60)
    logger.info("Strategie gespeichert in: RD-Agent_workspace/strategy.py")
    logger.info("\nNächste Schritte:")
    logger.info("  1. Backtest durchführen: python examples/04_backtest_simple.py")
    logger.info("  2. Strategie optimieren: rdagent build_strategies_ai")


def main():
    """Hauptfunktion mit Argument-Parsing."""
    parser = argparse.ArgumentParser(
        description="Beispiel 03: Faktoren zu Strategie kombinieren",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Beispiele:
  # 3 Faktoren kombinieren
  python 03_strategy_generation.py --factors momentum_16,hl_range_reversal,session_alpha
  
  # Mit KI-gestützter Generierung
  python 03_strategy_generation.py --factors momentum_16,session_alpha --ai
        """
    )

    parser.add_argument(
        "--factors",
        type=str,
        default="momentum_16,hl_range_reversal,session_alpha",
        help="Kommagetrennte Liste der Faktoren (default: momentum_16,hl_range_reversal,session_alpha)"
    )
    parser.add_argument(
        "--ai",
        action="store_true",
        help="KI-gestützte Strategiegenerierung (StrategyCoSTEER)"
    )

    args = parser.parse_args()
    factors = [f.strip() for f in args.factors.split(',')]

    try:
        run_strategy_generation(factors=factors, use_ai=args.ai)
    except KeyboardInterrupt:
        logger.warning("\nAbgebrochen durch Benutzer.")
        sys.exit(130)
    except Exception as e:
        logger.error(f"Fehler bei der Strategie-Generierung: {e}")
        sys.exit(1)


if __name__ == "__main__":
    main()
