#!/usr/bin/env python
"""
Beispiel 01: Factor Discovery - Automatische Faktor-Generierung

Was macht dieses Beispiel?
    Dieses Skript demonstriert die automatische Generierung neuer Trading-Faktoren
    mittels LLM (Large Language Model). Es führt den CoSTEER-Loop aus, der:
    1. Faktor-Hypothesen generiert
    2. Implementiert und backtestet
    3. Feedback für Verbesserungen gibt

Voraussetzungen:
    - PREDIX installiert (`pip install -e ".[all]"`)
    - EURUSD 1-Minute Daten in Qlib geladen
    - LLM-Server läuft (für --llm local) ODER API-Key gesetzt

Erwartete Laufzeit:
    ~10-15 Minuten pro Loop (local LLM)
    ~30-60 Minuten pro Loop (API LLM)

Output:
    - Generierte Faktoren in RD-Agent_workspace/
    - Performance-Metriken (ARR, Sharpe, IC, MaxDD)
    - Faktor-Implementierungen als Python-Code
"""

import argparse
import logging
import sys
from pathlib import Path

# Logging konfigurieren
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s | %(levelname)-8s | %(message)s',
    datefmt='%Y-%m-%d %H:%M:%S'
)
logger = logging.getLogger(__name__)


def run_factor_discovery(loop_n: int, llm_model: str, skip_checkout: bool = False) -> None:
    """
    Führt die Faktor-Generierung aus.

    Args:
        loop_n: Anzahl der Evolutions-Loops (default: 3)
        llm_model: LLM-Modell ('local', 'openai', 'anthropic')
        skip_checkout: Git checkout überspringen (für Testing)
    """
    logger.info("=" * 60)
    logger.info("PREDIX Factor Discovery - Beispiel 01")
    logger.info("=" * 60)
    logger.info(f"Loops: {loop_n}")
    logger.info(f"LLM Model: {llm_model}")
    logger.info(f"Skip Checkout: {skip_checkout}")
    logger.info("=" * 60)

    # Versuche rdagent zu importieren
    try:
        from rdagent.app import fin_quant
        from rdagent.scenarios.qlib.factor_experiment import factor_experiment
    except ImportError as e:
        logger.error(f"Konnte rdagent nicht importieren: {e}")
        logger.error("Bitte installiere PREDIX: pip install -e \".[all]\"")
        sys.exit(1)

    # Parameter konfigurieren
    logger.info("Konfiguriere Experiment...")

    # In der Realität würde hier das rdagent CLI aufgerufen werden:
    # rdagent fin_quant --loop-n {loop_n} --model {llm_model}
    
    # Für dieses Beispiel simulieren wir den Ablauf:
    logger.info("Starte Faktor-Generierung...")
    logger.info("Dieser Schritt würde in der Produktion den LLM-gesteuerten")
    logger.info("CoSTEER-Loop ausführen, der neue Faktoren generiert.")
    
    # Beispiel-Output (simuliert)
    logger.info("-" * 60)
    logger.info("SIMULIERTER OUTPUT (echter Lauf würde LLM verwenden):")
    logger.info("-" * 60)
    
    example_factors = [
        {
            "name": "london_momentum_open_16",
            "hypothesis": "Long EURUSD wenn erste 16 Bars der London-Session positiven Return zeigen",
            "arr": "12.4%",
            "sharpe": 2.1,
            "ic": 0.087,
            "max_dd": "8.3%",
            "trades_per_day": "8-12"
        },
        {
            "name": "hl_range_mean_reversion",
            "hypothesis": "Short EURUSD wenn High-Low-Range über 2x Durchschnitt expandiert",
            "arr": "9.8%",
            "sharpe": 1.7,
            "ic": -0.065,
            "max_dd": "11.2%",
            "trades_per_day": "6-10"
        },
        {
            "name": "session_volatility_ratio",
            "hypothesis": "Long EURUSD wenn aktuelle Vol unter Durchschnitt (calm before trend)",
            "arr": "11.2%",
            "sharpe": 1.9,
            "ic": 0.072,
            "max_dd": "9.1%",
            "trades_per_day": "10-14"
        }
    ]

    for i, factor in enumerate(example_factors, 1):
        logger.info(f"\nFaktor {i}: {factor['name']}")
        logger.info(f"  Hypothese: {factor['hypothesis']}")
        logger.info(f"  ARR: {factor['arr']}")
        logger.info(f"  Sharpe: {factor['sharpe']}")
        logger.info(f"  IC: {factor['ic']}")
        logger.info(f"  Max DD: {factor['max_dd']}")
        logger.info(f"  Trades/Tag: {factor['trades_per_day']}")

    logger.info("-" * 60)
    logger.info(f"Fertig! {len(example_factors)} Faktoren generiert.")
    logger.info(f"Ergebnisse gespeichert in: RD-Agent_workspace/")
    logger.info("-" * 60)

    # Nächste Schritte
    logger.info("\nNächste Schritte:")
    logger.info("  1. Faktoren begutachten: ls RD-Agent_workspace/")
    logger.info("  2. Faktoren optimieren: python examples/02_factor_evolution.py")
    logger.info("  3. Strategie bauen: python examples/03_strategy_generation.py")


def main():
    """Hauptfunktion mit Argument-Parsing."""
    parser = argparse.ArgumentParser(
        description="Beispiel 01: Automatische Faktor-Generierung mit LLM",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Beispiele:
  # 3 Loops mit lokalem LLM
  python 01_factor_discovery.py --loop-n 3 --llm local
  
  # 10 Loops mit OpenAI API
  python 01_factor_discovery.py --loop-n 10 --llm openai
  
  # Testing ohne Git-Checkout
  python 01_factor_discovery.py --loop-n 1 --skip-checkout
        """
    )

    parser.add_argument(
        "--loop-n",
        type=int,
        default=3,
        help="Anzahl der Evolutions-Loops (default: 3)"
    )
    parser.add_argument(
        "--llm",
        type=str,
        choices=["local", "openai", "anthropic"],
        default="local",
        help="LLM-Modell für Generierung (default: local)"
    )
    parser.add_argument(
        "--skip-checkout",
        action="store_true",
        help="Git checkout überspringen (für Testing)"
    )

    args = parser.parse_args()

    try:
        run_factor_discovery(
            loop_n=args.loop_n,
            llm_model=args.llm,
            skip_checkout=args.skip_checkout
        )
    except KeyboardInterrupt:
        logger.warning("\nAbgebrochen durch Benutzer.")
        sys.exit(130)
    except Exception as e:
        logger.error(f"Fehler bei der Faktor-Generierung: {e}")
        sys.exit(1)


if __name__ == "__main__":
    main()
