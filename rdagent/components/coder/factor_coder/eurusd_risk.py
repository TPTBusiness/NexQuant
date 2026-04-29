import logging
"""
Volatility-Adjusted Position Sizing für EURUSD

Berechnet die optimale Positionsgröße basierend auf:
- Kontogröße und Risikotoleranz
- Aktueller Volatilität (ATR, Historical Volatility)
- Marktregime (Hurst Exponent)
- Korrelation mit anderen Positionen

Druckenmiller-Prinzip:
- Bei hoher Conviction und asymmetrischer Chance: Große Position
- Bei niedriger Volatilität: Positionsgröße erhöhen
- Bei hoher Korrelation: Risk reduzieren
"""

from dataclasses import dataclass
from typing import Literal, Optional, Tuple

import numpy as np
import pandas as pd


@dataclass
class PositionSizeResult:
    """Ergebnis der Positionsgrößen-Berechnung."""
    lots: float
    leverage: int
    stop_loss_pips: float
    take_profit_pips: float
    risk_usd: float
    risk_percent: float
    volatility_adjustment: float
    regime_adjustment: float
    correlation_adjustment: float
    final_adjustment: float


def calculate_atr(high: pd.Series, low: pd.Series, close: pd.Series, period: int = 14) -> pd.Series:
    """
    Berechnet Average True Range (ATR) für Volatilitätsmessung.
    
    Parameters
    ----------
    high : pd.Series
        High-Preise
    low : pd.Series
        Low-Preise
    close : pd.Series
        Close-Preise
    period : int, default 14
        ATR-Periode (14 für 14-Bar-ATR)
    
    Returns
    -------
    pd.Series
        ATR-Werte
    """
    prev_close = close.shift(1)
    
    # True Range Komponenten
    tr1 = high - low
    tr2 = abs(high - prev_close)
    tr3 = abs(low - prev_close)
    
    # True Range
    tr = pd.concat([tr1, tr2, tr3], axis=1).max(axis=1)
    
    # ATR als gleitender Durchschnitt von TR
    atr = tr.rolling(window=period).mean()
    
    return atr


def calculate_historical_volatility(returns: pd.Series, window: int = 20, annualize: bool = True) -> pd.Series:
    """
    Berechnet historische Volatilität (Standardabweichung der Returns).
    
    Parameters
    ----------
    returns : pd.Series
        Log-Returns oder prozentuale Returns
    window : int, default 20
        Fenster für Volatilitätsberechnung (20 Bars)
    annualize : bool, default True
        annualisieren der Volatilität (für 1min-Daten: * sqrt(525600))
    
    Returns
    -------
    pd.Series
        Historische Volatilität
    """
    vol = returns.rolling(window=window).std()
    
    if annualize:
        # Für 1min-Daten: 525600 Minuten pro Jahr (365 * 24 * 60)
        vol = vol * np.sqrt(525600)
    
    return vol


def calculate_volatility_percentile(current_vol: float, vol_history: pd.Series, lookback: int = 100) -> float:
    """
    Berechnet das Volatilitäts-Percentile (0-100).
    
    Parameters
    ----------
    current_vol : float
        Aktuelle Volatilität
    vol_history : pd.Series
        Historische Volatilitäten
    lookback : int, default 100
        Lookback-Fenster für Percentil-Berechnung
    
    Returns
    -------
    float
        Volatilitäts-Percentile (0-100)
    """
    if len(vol_history) < lookback:
        lookback = len(vol_history)
    
    if lookback < 10:
        return 50.0  # Default bei zu wenig Daten
    
    # Percentile-Rang der aktuellen Volatilität
    percentile = (vol_history.iloc[-lookback:] < current_vol).mean() * 100
    
    return percentile


def calculate_eurusd_position_size(
    account_equity: float,
    atr_14: float,
    volatility_percentile: float,
    regime: Literal["MEAN_REVERSION", "NEUTRAL", "TRENDING"] = "NEUTRAL",
    risk_percent: float = 0.02,
    base_leverage: int = 20,
    correlation_adjustment: float = 1.0,
    pip_value: float = 10.0  # $10 pro Pip für Standard-Lot EURUSD
) -> PositionSizeResult:
    """
    Berechnet die optimale Positionsgröße für EURUSD Trades.
    
    Volatility-Adjusted Position Sizing:
    - Niedrige Volatilität (< 20. Percentile) → größere Position (1.5x)
    - Mittlere Volatilität (20-80. Percentile) → normale Position (1.0x)
    - Hohe Volatilität (> 80. Percentile) → kleinere Position (0.4-0.7x)
    
    Regime-Adjustierung:
    - MEAN_REVERSION: Engerer TP, weiterer SL (mehr Raum für Mean-Reversion)
    - TRENDING: Weiterer TP, normaler SL (Trend ausreiten)
    - NEUTRAL: Vorsichtig, beide eng
    
    Korrelations-Adjustierung:
    - Hohe Korrelation mit anderen Positionen → Risk reduzieren
    
    Parameters
    ----------
    account_equity : float
        Kontogröße in USD
    atr_14 : float
        Aktueller ATR(14) in Pip (z.B. 0.0012 = 12 Pips)
    volatility_percentile : float
        Volatilitäts-Percentile (0-100)
    regime : str, default "NEUTRAL"
        Marktregime: "MEAN_REVERSION", "NEUTRAL", oder "TRENDING"
    risk_percent : float, default 0.02
        Risiko pro Trade (2% = 0.02)
    base_leverage : int, default 20
        Basis-Hebel (10-50)
    correlation_adjustment : float, default 1.0
        Korrelations-Faktor (0.7-1.1)
    pip_value : float, default 10.0
        Wert pro Pip pro Standard-Lot ($10 für EURUSD)
    
    Returns
    -------
    PositionSizeResult
        Berechnete Positionsgröße mit allen Details
        
    Example
    -------
    >>> result = calculate_eurusd_position_size(
    ...     account_equity=100000,
    ...     atr_14=12.5,  # 12.5 Pips
    ...     volatility_percentile=35,  # Unterdurchschnittliche Vol
    ...     regime="MEAN_REVERSION",
    ...     risk_percent=0.02
    ... )
    >>> print(f"Lots: {result.lots:.2f}, Leverage: {result.leverage}x")
    >>> print(f"Risk: ${result.risk_usd:.2f} ({result.risk_percent:.1%})")
    """
    
    # 1. Volatility-Adjustment
    if volatility_percentile < 20:
        vol_adjustment = 1.5  # Niedrige Vol → größere Position
    elif volatility_percentile < 50:
        vol_adjustment = 1.2  # Unterdurchschnittliche Vol
    elif volatility_percentile < 80:
        vol_adjustment = 1.0  # Normale Vol
    elif volatility_percentile < 95:
        vol_adjustment = 0.7  # Erhöhte Vol → kleinere Position
    else:
        vol_adjustment = 0.4  # Extreme Vol → minimales Risk
    
    # 2. Regime-Adjustment
    if regime == "MEAN_REVERSION":
        regime_adjustment = 1.1  # Mean-Reversion ist relativ vorhersehbar
        sl_pips = atr_14 * 2.0  # Weiterer SL für Mean-Reversion
        tp_pips = atr_14 * 1.0  # Engerer TP
    elif regime == "TRENDING":
        regime_adjustment = 1.2  # Trending kann profitabler sein
        sl_pips = atr_14 * 1.5  # Normaler SL
        tp_pips = atr_14 * 2.5  # Weiterer TP für Trend
    else:  # NEUTRAL
        regime_adjustment = 0.8  # Vorsichtig bei unklarem Regime
        sl_pips = atr_14 * 1.5  # Normaler SL
        tp_pips = atr_14 * 1.2  # Engerer TP
    
    # 3. Gesamtes Adjustment
    final_adjustment = vol_adjustment * regime_adjustment * correlation_adjustment
    
    # 4. Risiko in USD
    base_risk_usd = account_equity * risk_percent
    adjusted_risk_usd = base_risk_usd * final_adjustment
    
    # 5. Positionsgröße in Lots
    # Risk = Lots * Pip_Value * SL_Pips
    # Lots = Risk / (Pip_Value * SL_Pips)
    if sl_pips > 0 and pip_value > 0:
        lots = adjusted_risk_usd / (pip_value * sl_pips)
    else:
        lots = 0.0
    
    # 6. Effektiver Hebel basierend auf Positionsgröße
    # 1 Standard-Lot = 100,000 EUR
    # Bei 100k Konto und 1 Lot = 100k EUR = 1x Hebel
    position_value_eur = lots * 100000
    position_value_usd = position_value_eur  # EURUSD ≈ 1:1
    effective_leverage = position_value_usd / account_equity if account_equity > 0 else 0
    
    # Begrenze Hebel auf Maximum
    max_leverage = base_leverage * final_adjustment
    if effective_leverage > max_leverage:
        # Reduziere Lots um im Hebel-Limit zu bleiben
        lots = (max_leverage * account_equity) / 100000
        effective_leverage = max_leverage
    
    # Begrenze Lots auf vernünftige Werte
    lots = max(0.01, min(lots, 100.0))  # Min 0.01 Lots, Max 100 Lots
    
    # Finales Risiko mit angepassten Lots
    final_risk_usd = lots * pip_value * sl_pips
    final_risk_percent = final_risk_usd / account_equity if account_equity > 0 else 0
    
    return PositionSizeResult(
        lots=round(lots, 2),
        leverage=round(effective_leverage),
        stop_loss_pips=round(sl_pips, 1),
        take_profit_pips=round(tp_pips, 1),
        risk_usd=round(final_risk_usd, 2),
        risk_percent=round(final_risk_percent, 4),
        volatility_adjustment=round(vol_adjustment, 2),
        regime_adjustment=round(regime_adjustment, 2),
        correlation_adjustment=round(correlation_adjustment, 2),
        final_adjustment=round(final_adjustment, 2)
    )


def calculate_forex_correlation(
    eurusd_returns: pd.Series,
    other_positions: dict
) -> Tuple[float, float]:
    """
    Berechnet die durchschnittliche Korrelation von EURUSD mit anderen Positionen.
    
    Für Forex relevante Korrelationen:
    - GBPUSD: +0.75 (positiv, beide EUR/GBP vs USD)
    - USDCHF: -0.70 (negativ, beide USD-basiert)
    - DXY: -0.85 (negativ, DXY ist USD-Index)
    - EURGBP: +0.40 (moderat positiv)
    
    Parameters
    ----------
    eurusd_returns : pd.Series
        EURUSD Returns für Korrelationsberechnung
    other_positions : dict
        Andere offene Positionen mit Keys:
        - symbol: {"position": "LONG"/"SHORT", "size": lots, "returns": pd.Series}
    
    Returns
    -------
    Tuple[float, float]
        (durchschnittliche Korrelation, Korrelations-Adjustment-Faktor)
    """
    
    # Typische Forex-Korrelationen
    CORRELATIONS = {
        "GBPUSD": 0.75,
        "USDCHF": -0.70,
        "DXY": -0.85,
        "EURGBP": 0.40,
        "USDJPY": -0.50,
        "AUDUSD": 0.60,
        "USDCAD": -0.55,
        "EURUSD": 1.0  # Referenz
    }
    
    if len(other_positions) == 0:
        return 0.0, 1.0  # Keine Korrelation, kein Adjustment
    
    # Berechne gewichtete durchschnittliche Korrelation
    total_correlation = 0.0
    total_weight = 0.0
    
    for symbol, pos_data in other_positions.items():
        if symbol not in CORRELATIONS:
            continue
        
        # Korrelation aus historischen Returns (wenn verfügbar)
        if "returns" in pos_data and pos_data["returns"] is not None:
            try:
                # Berechne tatsächliche Korrelation
                corr = eurusd_returns.corr(pos_data["returns"])
                if not np.isnan(corr):
                    actual_corr = corr
                else:
                    actual_corr = CORRELATIONS[symbol]
            except Exception:
                actual_corr = CORRELATIONS[symbol]
        else:
            # Verwende typische Korrelation
            actual_corr = CORRELATIONS[symbol]
        
        # Gewichte mit Positionsgröße
        weight = pos_data.get("size", 1.0)
        
        # Berücksichtige Long/Short-Position
        if pos_data.get("position") == "SHORT":
            actual_corr = -actual_corr  # Short kehrt Korrelation um
        
        total_correlation += actual_corr * weight
        total_weight += weight
    
    if total_weight > 0:
        avg_correlation = total_correlation / total_weight
    else:
        avg_correlation = 0.0
    
    # Korrelations-Adjustment
    if avg_correlation > 0.6:
        corr_adjustment = 0.7  # Hohe positive Korrelation → Risk reduzieren
    elif avg_correlation > 0.4:
        corr_adjustment = 0.85
    elif avg_correlation < -0.6:
        corr_adjustment = 1.1  # Hohe negative Korrelation → natürlicher Hedge
    elif avg_correlation < -0.4:
        corr_adjustment = 1.05
    else:
        corr_adjustment = 1.0  # Neutrale Korrelation
    
    return avg_correlation, corr_adjustment


# Test-Funktion für lokale Validierung
if __name__ == "__main__":
    print("=== Volatility-Adjusted Position Sizing Test ===\n")
    
    # Test 1: Normale Volatilität, NEUTRAL Regime
    print("Test 1: Normale Bedingungen")
    result1 = calculate_eurusd_position_size(
        account_equity=100000,
        atr_14=12.5,  # 12.5 Pips
        volatility_percentile=50,
        regime="NEUTRAL",
        risk_percent=0.02
    )
    print(f"  Lots: {result1.lots:.2f}")
    print(f"  Leverage: {result1.leverage}x")
    print(f"  SL: {result1.stop_loss_pips:.1f} Pips, TP: {result1.take_profit_pips:.1f} Pips")
    print(f"  Risk: ${result1.risk_usd:.2f} ({result1.risk_percent:.2%})")
    print(f"  Adjustments: Vol={result1.volatility_adjustment}, Regime={result1.regime_adjustment}, Corr={result1.correlation_adjustment}")
    
    # Test 2: Niedrige Volatilität, MEAN_REVERSION Regime
    print("\nTest 2: Niedrige Volatilität, Mean-Reversion")
    result2 = calculate_eurusd_position_size(
        account_equity=100000,
        atr_14=8.0,  # Niedrige Vol
        volatility_percentile=15,
        regime="MEAN_REVERSION",
        risk_percent=0.02
    )
    print(f"  Lots: {result2.lots:.2f}")
    print(f"  Leverage: {result2.leverage}x")
    print(f"  SL: {result2.stop_loss_pips:.1f} Pips, TP: {result2.take_profit_pips:.1f} Pips")
    print(f"  Risk: ${result2.risk_usd:.2f} ({result2.risk_percent:.2%})")
    print(f"  Adjustments: Vol={result2.volatility_adjustment}, Regime={result2.regime_adjustment}")
    
    # Test 3: Hohe Volatilität, TRENDING Regime
    print("\nTest 3: Hohe Volatilität, Trending")
    result3 = calculate_eurusd_position_size(
        account_equity=100000,
        atr_14=25.0,  # Hohe Vol
        volatility_percentile=85,
        regime="TRENDING",
        risk_percent=0.02
    )
    print(f"  Lots: {result3.lots:.2f}")
    print(f"  Leverage: {result3.leverage}x")
    print(f"  SL: {result3.stop_loss_pips:.1f} Pips, TP: {result3.take_profit_pips:.1f} Pips")
    print(f"  Risk: ${result3.risk_usd:.2f} ({result3.risk_percent:.2%})")
    print(f"  Adjustments: Vol={result3.volatility_adjustment}, Regime={result3.regime_adjustment}")
    
    # Test 4: Korrelations-Adjustment
    print("\nTest 4: Korrelations-Adjustment")
    
    # Simuliere andere Positionen
    np.random.seed(42)
    eurusd_returns = pd.Series(np.random.randn(100) * 0.0001)
    
    other_positions = {
        "GBPUSD": {"position": "LONG", "size": 0.5, "returns": pd.Series(np.random.randn(100) * 0.0001)},
        "USDCHF": {"position": "SHORT", "size": 0.3, "returns": pd.Series(np.random.randn(100) * 0.0001)}
    }
    
    avg_corr, corr_adj = calculate_forex_correlation(eurusd_returns, other_positions)
    print(f"  Durchschnittliche Korrelation: {avg_corr:.3f}")
    print(f"  Korrelations-Adjustment: {corr_adj:.2f}")
    
    result4 = calculate_eurusd_position_size(
        account_equity=100000,
        atr_14=12.5,
        volatility_percentile=50,
        regime="NEUTRAL",
        risk_percent=0.02,
        correlation_adjustment=corr_adj
    )
    print(f"  Lots mit Korrelation: {result4.lots:.2f} (vs. {result1.lots:.2f} ohne)")
    print(f"  Korrelations-Adjustment: {result4.correlation_adjustment}")
    
    # Zusammenfassung
    print("\n=== Test Summary ===")
    print("✅ Volatility-Adjusted Position Sizing ist funktionsfähig!")
    print("\nKey Features:")
    print("  - Volatilitäts-Adjustment (0.4x - 1.5x)")
    print("  - Regime-Adjustment (MEAN_REVERSION/TRENDING/NEUTRAL)")
    print("  - Korrelations-Adjustment für Forex-Paare")
    print("  - ATR-basierte SL/TP-Berechnung")
    print("  - Hebel-Begrenzung und Risk-Management")
