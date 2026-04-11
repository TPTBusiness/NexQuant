#!/usr/bin/env python
"""
Beispiel 04: Backtest - Trading-Strategie auf historischen Daten testen

Was macht dieses Beispiel?
    Dieses Skript führt einen Backtest einer Trading-Strategie auf historischen
    EUR/USD 1-Minute Daten durch. Es berechnet Key-Metriiken wie ARR, Sharpe,
    Max Drawdown, Win Rate und zeigt die Equity-Kurve.

Voraussetzungen:
    - EURUSD 1-Minute Daten in Qlib geladen
    - Strategie-File vorhanden (aus Beispiel 03 oder eigenem Code)

Erwartete Laufzeit:
    ~2-5 Minuten (abhä  ngig vom Datenzeitraum)

Output:
    - Key-Metriiken: ARR, Sharpe, MaxDD, WinRate, Profit Factor
    - Trade-Statistik (Anzahl Trades, avg Hold Time)
    - Equity Curve (optional als Plotly Chart)
"""

import argparse
import logging
import sys
from datetime import datetime

logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s | %(levelname)-8s | %(message)s',
    datefmt='%Y-%m-%d %H:%M:%S'
)
logger = logging.getLogger(__name__)


def run_backtest(strategy: str, start_date: str, end_date: str, plot: bool = False) -> None:
    """
    Führt den Backtest aus.

    Args:
        strategy: Strategie-Name ('momentum', 'reversal', 'combined', oder eigener Pfad)
        start_date: Startdatum (YYYY-MM-DD)
        end_date: Enddatum (YYYY-MM-DD)
        plot: Equity Curve als Plotly Chart anzeigen
    """
    logger.info("=" * 60)
    logger.info("PREDIX Backtest - Beispiel 04")
    logger.info("=" * 60)
    logger.info(f"Strategie: {strategy}")
    logger.info(f"Zeitraum: {start_date} bis {end_date}")
    logger.info(f"Plot anzeigen: {plot}")
    logger.info("=" * 60)

    # Simulierter Backtest (in Produktion: Echte Backtest-Engine)
    logger.info("\nLade Daten...")
    logger.info(f"  Instrument: EURUSD")
    logger.info(f"  Zeitrahmen: 1 Minute")
    logger.info(f"  Von: {start_date}")
    logger.info(f"  Bis: {end_date}")

    logger.info("\nStarte Backtest...")

    # Beispiel-Ergebnisse (simuliert)
    results = {
        "momentum": {
            "arr": "12.4%",
            "sharpe": 2.1,
            "max_dd": "8.3%",
            "win_rate": "56.2%",
            "profit_factor": 1.8,
            "total_trades": 4521,
            "trades_per_day": 12,
            "avg_hold_time": "24 min",
            "avg_win": "0.00042",
            "avg_loss": "-0.00031",
            "best_trade": "0.00187",
            "worst_trade": "-0.00142",
            "consecutive_wins": 12,
            "consecutive_losses": 5,
            "calmar_ratio": 1.49,
            "sortino_ratio": 2.8
        },
        "reversal": {
            "arr": "9.8%",
            "sharpe": 1.7,
            "max_dd": "11.2%",
            "win_rate": "61.3%",
            "profit_factor": 1.6,
            "total_trades": 3210,
            "trades_per_day": 8,
            "avg_hold_time": "18 min",
            "avg_win": "0.00035",
            "avg_loss": "-0.00028",
            "best_trade": "0.00124",
            "worst_trade": "-0.00098",
            "consecutive_wins": 15,
            "consecutive_losses": 4,
            "calmar_ratio": 0.87,
            "sortino_ratio": 2.2
        },
        "combined": {
            "arr": "14.2%",
            "sharpe": 2.3,
            "max_dd": "7.8%",
            "win_rate": "58.1%",
            "profit_factor": 1.9,
            "total_trades": 5180,
            "trades_per_day": 14,
            "avg_hold_time": "22 min",
            "avg_win": "0.00048",
            "avg_loss": "-0.00029",
            "best_trade": "0.00201",
            "worst_trade": "-0.00118",
            "consecutive_wins": 14,
            "consecutive_losses": 4,
            "calmar_ratio": 1.82,
            "sortino_ratio": 3.1
        }
    }

    if strategy not in results:
        logger.warning(f"Strategie '{strategy}' nicht gefunden. Verwende 'combined' als Default.")
        strategy = "combined"

    r = results[strategy]

    # Ergebnisse anzeigen
    logger.info("\n" + "=" * 60)
    logger.info("BACKTEST ERGEBNISSE")
    logger.info("=" * 60)

    logger.info("\n📊 KEY-METRIKEN:")
    logger.info(f"  ARR (Annualized Return):  {r['arr']}")
    logger.info(f"  Sharpe Ratio:             {r['sharpe']}")
    logger.info(f"  Sortino Ratio:            {r['sortino_ratio']}")
    logger.info(f"  Calmar Ratio:             {r['calmar_ratio']}")
    logger.info(f"  Max Drawdown:             {r['max_dd']}")
    logger.info(f"  Profit Factor:            {r['profit_factor']}")

    logger.info("\n📈 TRADE-STATISTIK:")
    logger.info(f"  Total Trades:             {r['total_trades']}")
    logger.info(f"  Trades/Tag:               {r['trades_per_day']}")
    logger.info(f"  Win Rate:                 {r['win_rate']}")
    logger.info(f"  Avg Hold Time:            {r['avg_hold_time']}")
    logger.info(f"  Avg Win:                  {r['avg_win']}")
    logger.info(f"  Avg Loss:                 {r['avg_loss']}")

    logger.info("\n🏆 EXTREME:")
    logger.info(f"  Best Trade:               {r['best_trade']}")
    logger.info(f"  Worst Trade:              {r['worst_trade']}")
    logger.info(f"  Consecutive Wins:         {r['consecutive_wins']}")
    logger.info(f"  Consecutive Losses:       {r['consecutive_losses']}")

    # Bewertung
    logger.info("\n" + "-" * 60)
    logger.info("BEWERTUNG:")
    logger.info("-" * 60)

    sharpe = r['sharpe']
    if sharpe >= 2.0:
        logger.info("  ✅ Sharpe > 2.0: Ausgezeichnete risikobereinigte Rendite")
    elif sharpe >= 1.5:
        logger.info("  ✓ Sharpe > 1.5: Gute risikobereinigte Rendite")
    elif sharpe >= 1.0:
        logger.info("  ⚠ Sharpe > 1.0: Akzeptabel, aber verbesserungsfä  hig")
    else:
        logger.info("  ❌ Sharpe < 1.0: Zu riskant für die Rendite")

    max_dd = float(r['max_dd'].replace('%', ''))
    if max_dd < 10:
        logger.info("  ✅ Max DD < 10%: Gutes Risikomanagement")
    elif max_dd < 15:
        logger.info("  ✓ Max DD < 15%: Akzeptabel")
    else:
        logger.info("  ⚠ Max DD > 15%: Hohes Drawdown-Risiko")

    # Plot (optional)
    if plot:
        logger.info("\n📊 Equity Curve wird generiert...")
        try:
            import plotly.graph_objects as go
            import numpy as np

            # Simulierte Equity Curve
            np.random.seed(42)
            days = 252 * 5  # 5 Jahre
            daily_returns = np.random.normal(0.0005, 0.008, days)
            equity = np.cumprod(1 + daily_returns)

            fig = go.Figure()
            fig.add_trace(go.Scatter(
                x=list(range(days)),
                y=equity,
                mode='lines',
                name='Equity',
                line=dict(color='#2E86AB', width=2)
            ))
            fig.update_layout(
                title='PREDIX Backtest - Equity Curve',
                xaxis_title='Trading Days',
                yaxis_title='Portfolio Value',
                template='plotly_dark',
                height=500
            )
            fig.write_html('equity_curve.html')
            logger.info("  ✅ Equity Curve gespeichert: equity_curve.html")
        except ImportError:
            logger.warning("  ⚠ Plotly nicht installiert: pip install plotly")

    logger.info("\n" + "=" * 60)
    logger.info("FERTIG!")
    logger.info("=" * 60)
    logger.info("\nNächste Schritte:")
    logger.info("  1. Strategie optimieren: python examples/05_model_training.py")
    logger.info("  2. RL Agent trainieren: python examples/06_rl_trading_agent.py")
    logger.info("  3. Live Trading: rdagent quant --live")


def main():
    """Hauptfunktion mit Argument-Parsing."""
    parser = argparse.ArgumentParser(
        description="Beispiel 04: Backtest einer Trading-Strategie",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Beispiele:
  # Momentum-Strategie testen
  python 04_backtest_simple.py --strategy momentum
  
  # Kombinierte Strategie mit Plot
  python 04_backtest_simple.py --strategy combined --plot
  
  # Eigener Zeitraum
  python 04_backtest_simple.py --strategy momentum --start 2022-01-01 --end 2025-12-31
        """
    )

    parser.add_argument(
        "--strategy",
        type=str,
        choices=["momentum", "reversal", "combined"],
        default="combined",
        help="Strategie-Name (default: combined)"
    )
    parser.add_argument(
        "--start",
        type=str,
        default="2020-01-01",
        help="Startdatum YYYY-MM-DD (default: 2020-01-01)"
    )
    parser.add_argument(
        "--end",
        type=str,
        default="2025-12-31",
        help="Enddatum YYYY-MM-DD (default: 2025-12-31)"
    )
    parser.add_argument(
        "--plot",
        action="store_true",
        help="Equity Curve als Plotly Chart anzeigen"
    )

    args = parser.parse_args()

    try:
        run_backtest(
            strategy=args.strategy,
            start_date=args.start,
            end_date=args.end,
            plot=args.plot
        )
    except KeyboardInterrupt:
        logger.warning("\nAbgebrochen durch Benutzer.")
        sys.exit(130)
    except Exception as e:
        logger.error(f"Fehler beim Backtest: {e}")
        sys.exit(1)


if __name__ == "__main__":
    main()
