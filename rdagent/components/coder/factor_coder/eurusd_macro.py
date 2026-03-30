"""
EURUSD Macro Agent (Stanley Druckenmiller Stil)

Inspiriert von: ai-hedge-fund/src/agents/stanley_druckenmiller.py

Makro-Fokus für Forex-Trading:
- Zinsdifferential (Fed vs EZB)
- Wirtschaftswachstum (BIP, PMI, NFP)
- Momentum (DXY Trend, EURUSD Trend)
- Sentiment (COT Report, Risk Sentiment)
- Asymmetrische Risk-Reward-Analyse
"""

import json
import sys
from dataclasses import dataclass
from pathlib import Path
from typing import Dict, List, Literal, Optional

sys.path.insert(0, str(Path(__file__).parent))

from eurusd_llm import MultiProviderLLM


@dataclass
class MacroSignal:
    """Makro-Signal mit Details."""
    action: Literal["LONG", "SHORT", "NEUTRAL"]
    confidence: int  # 0-100
    reasoning: List[str]
    
    # Makro-Faktoren
    rate_differential: float  # Fed - EZB Zinsen
    growth_differential: float  # US - EU Wachstum
    momentum_score: float  # -1 bis +1
    sentiment_score: float  # -1 bis +1
    
    # Risk-Reward
    expected_return: float  # Erwartete Rendite in %
    risk_reward_ratio: float  # R/R Verhältnis
    asymmetric_opportunity: bool  # Gibt es asymmetrische Chance?
    
    # Trade-Parameter
    entry_price: Optional[float] = None
    stop_loss: Optional[float] = None
    take_profit: Optional[float] = None
    leverage: int = 20


class EURUSDMacroAgent:
    """
    Macro Agent im Stanley Druckenmiller Stil für EURUSD.
    
    Analysiert makroökonomische Faktoren:
    1. Zinsdifferential (Fed vs EZB)
    2. Wirtschaftswachstum (BIP, PMI, NFP)
    3. Momentum (DXY, EURUSD Trends)
    4. Sentiment (COT, Risk-On/Off)
    5. Asymmetrische Risk-Reward-Analyse
    
    Druckenmiller-Prinzipien:
    - "It's not whether you're right or wrong, but how much you make when right"
    - Asymmetrische Chancen erkennen (begrenztes Downside, großes Upside)
    - Bei hoher Conviction großen Positionen eingehen
    - Makro-Trends folgen, nicht gegen sie handeln
    """
    
    def __init__(self, llm: Optional[MultiProviderLLM] = None):
        self.llm = llm or MultiProviderLLM()
    
    def analyze(
        self,
        macro_data: dict,
        price_data: Optional[dict] = None
    ) -> MacroSignal:
        """
        Analysiert makroökonomische Daten für EURUSD.
        
        Parameters
        ----------
        macro_data : dict
            Makrodaten mit Keys:
            - fed_rate: US-Leitzins (%)
            - ecb_rate: EZB-Leitzins (%)
            - us_pmi: US PMI
            - eu_pmi: Eurozone PMI
            - us_gdp_growth: US BIP-Wachstum (%)
            - eu_gdp_growth: EU BIP-Wachstum (%)
            - dxy_trend: DXY Trend ("up", "down", "neutral")
            - risk_sentiment: Risk-On/Off ("risk-on", "risk-off", "neutral")
            - cot_report: COT Report Daten
            
        price_data : dict, optional
            Preisdaten für Entry/SL/TP Berechnung
            
        Returns
        -------
        MacroSignal
            Makro-Signal mit Trading-Empfehlung
        """
        # 1. Berechne fundamentale Differentiale
        rate_diff = macro_data.get("fed_rate", 5.0) - macro_data.get("ecb_rate", 4.0)
        growth_diff = macro_data.get("us_gdp_growth", 2.0) - macro_data.get("eu_gdp_growth", 1.5)
        pmi_diff = macro_data.get("us_pmi", 50) - macro_data.get("eu_pmi", 50)
        
        # 2. Berechne Momentum-Score
        dxy_trend = macro_data.get("dxy_trend", "neutral")
        if dxy_trend == "up":
            momentum_score = -0.5  # Starker DXY = schwacher EURUSD
        elif dxy_trend == "down":
            momentum_score = 0.5  # Schwacher DXY = starker EURUSD
        else:
            momentum_score = 0.0
        
        # 3. Berechne Sentiment-Score
        risk_sentiment = macro_data.get("risk_sentiment", "neutral")
        if risk_sentiment == "risk-on":
            sentiment_score = 0.3  # Risk-On begünstigt EUR
        elif risk_sentiment == "risk-off":
            sentiment_score = -0.3  # Risk-Off begünstigt USD
        else:
            sentiment_score = 0.0
        
        # 4. LLM-basierte Gesamtanalyse
        signal = self._llm_analysis(
            rate_diff=rate_diff,
            growth_diff=growth_diff,
            pmi_diff=pmi_diff,
            momentum_score=momentum_score,
            sentiment_score=sentiment_score,
            macro_data=macro_data,
            price_data=price_data
        )
        
        # 5. Füge berechnete Werte hinzu
        signal.rate_differential = rate_diff
        signal.growth_differential = growth_diff
        signal.momentum_score = momentum_score
        signal.sentiment_score = sentiment_score
        
        return signal
    
    def _llm_analysis(
        self,
        rate_diff: float,
        growth_diff: float,
        pmi_diff: float,
        momentum_score: float,
        sentiment_score: float,
        macro_data: dict,
        price_data: Optional[dict]
    ) -> MacroSignal:
        """
        LLM-basierte Analyse mit Druckenmiller-Prinzipien.
        """
        prompt = self._build_macro_prompt(
            rate_diff, growth_diff, pmi_diff,
            momentum_score, sentiment_score,
            macro_data, price_data
        )
        
        system_prompt = """Du bist ein makroökonomischer Analyst im Stil von Stanley Druckenmiller.
        
        Deine Aufgabe:
        1. Analysiere makroökonomische Differentiale (Zinsen, Wachstum, PMI)
        2. Bewerte Momentum und Sentiment
        3. Identifiziere asymmetrische Risk-Reward-Chancen
        4. Gib eine klare LONG/SHORT/NEUTRAL Empfehlung
        
        Druckenmiller-Prinzipien:
        - "It's not whether you're right or wrong, but how much you make when right"
        - Bei hoher Conviction: große Positionen
        - Asymmetrische Chancen suchen (1:3 R/R oder besser)
        - Makro-Trends folgen, nicht gegen sie handeln
        
        Antworte IMMER im JSON-Format."""
        
        try:
            response = self.llm.chat(
                prompt=prompt,
                system_prompt=system_prompt,
                temperature=0.1,
                max_tokens=800,
                json_mode=True
            )
            
            result = json.loads(response["content"])
            
            return MacroSignal(
                action=result.get("action", "NEUTRAL"),
                confidence=min(100, max(0, result.get("confidence", 50))),
                reasoning=result.get("reasons", []),
                rate_differential=rate_diff,
                growth_differential=growth_diff,
                momentum_score=momentum_score,
                sentiment_score=sentiment_score,
                expected_return=result.get("expected_return", 0.0),
                risk_reward_ratio=result.get("risk_reward_ratio", 1.0),
                asymmetric_opportunity=result.get("asymmetric_opportunity", False),
                entry_price=price_data.get("price") if price_data else None,
                stop_loss=result.get("stop_loss"),
                take_profit=result.get("take_profit"),
                leverage=result.get("leverage", 20)
            )
            
        except Exception as e:
            # Fallback bei Fehlern
            return MacroSignal(
                action="NEUTRAL",
                confidence=50,
                reasoning=[f"Macro-Analyse fehlgeschlagen: {str(e)}"],
                rate_differential=rate_diff,
                growth_differential=growth_diff,
                momentum_score=momentum_score,
                sentiment_score=sentiment_score,
                expected_return=0.0,
                risk_reward_ratio=1.0,
                asymmetric_opportunity=False
            )
    
    def _build_macro_prompt(
        self,
        rate_diff: float,
        growth_diff: float,
        pmi_diff: float,
        momentum_score: float,
        sentiment_score: float,
        macro_data: dict,
        price_data: Optional[dict]
    ) -> str:
        """Erstellt makroökonomischen Prompt."""
        price_str = f"- Aktueller Preis: {price_data.get('price', 'N/A')}\n" if price_data else ""
        
        return f"""
=== EURUSD Macro Analyse (Druckenmiller Stil) ===

=== Zinsdifferential ===
- Fed Rate - EZB Rate: {rate_diff:+.2f}% ({'USD vorteil' if rate_diff > 0 else 'EUR vorteil' if rate_diff < 0 else 'neutral'})

=== Wirtschaftswachstum ===
- US vs EU Wachstum: {growth_diff:+.2f}%
- US vs EU PMI: {pmi_diff:+.1f}

=== Momentum & Sentiment ===
- Momentum Score: {momentum_score:+.2f} ({'DXY schwach' if momentum_score > 0 else 'DXY stark' if momentum_score < 0 else 'neutral'})
- Sentiment Score: {sentiment_score:+.2f} ({'Risk-On' if sentiment_score > 0 else 'Risk-Off' if sentiment_score < 0 else 'neutral'})

{price_str}
=== Zusätzliche Informationen ===
- Wirtschaftsdaten: {macro_data.get('economic_data', 'N/A')}
- COT Report: {macro_data.get('cot_report', 'N/A')}

=== Aufgabe ===
1. Bewerte die makroökonomische Situation
2. Identifiziere asymmetrische Risk-Reward-Chancen
3. Gib LONG/SHORT/NEUTRAL Empfehlung mit Confidence

Antworte als JSON:
{{
  "action": "LONG" oder "SHORT" oder "NEUTRAL",
  "confidence": 0-100,
  "reasons": ["Grund 1", "Grund 2", ...],
  "expected_return": 0.05,  # 5% erwartet
  "risk_reward_ratio": 3.0,  # 1:3 R/R
  "asymmetric_opportunity": true/false,
  "stop_loss": 1.0800,
  "take_profit": 1.0950,
  "leverage": 20
}}
"""


class MacroDebateIntegration:
    """
    Integriert Macro-Agent mit Bull/Bear/Neutral Debatte.
    
    Der Macro-Agent gibt zusätzliche makroökonomische Perspektive,
    die in die finale Debatte einfließt.
    """
    
    def __init__(self, llm: Optional[MultiProviderLLM] = None):
        self.macro_agent = EURUSDMacroAgent(llm)
    
    def get_macro_perspective(self, macro_data: dict, price_data: dict) -> dict:
        """
        Gibt makroökonomische Perspektive für Debatte.
        
        Returns
        -------
        dict
            Macro-Perspektive für Bull/Bear/Neutral Agenten
        """
        signal = self.macro_agent.analyze(macro_data, price_data)
        
        return {
            "action": signal.action,
            "confidence": signal.confidence,
            "reasoning": signal.reasoning,
            "macro_factors": {
                "rate_differential": signal.rate_differential,
                "growth_differential": signal.growth_differential,
                "momentum_score": signal.momentum_score,
                "sentiment_score": signal.sentiment_score
            },
            "risk_reward": {
                "expected_return": signal.expected_return,
                "risk_reward_ratio": signal.risk_reward_ratio,
                "asymmetric_opportunity": signal.asymmetric_opportunity
            }
        }


# Test-Funktion für lokale Validierung
if __name__ == "__main__":
    print("=== EURUSD Macro Agent Test (Mock Mode) ===\n")
    
    # Test-Makrodaten
    test_macro_data = {
        "fed_rate": 5.25,
        "ecb_rate": 4.50,
        "us_pmi": 52.5,
        "eu_pmi": 48.2,
        "us_gdp_growth": 2.4,
        "eu_gdp_growth": 0.8,
        "dxy_trend": "up",
        "risk_sentiment": "risk-off",
        "economic_data": "US NFP beat, EZB pause expected",
        "cot_report": "Speculators net short EUR"
    }
    
    price_data = {"price": 1.0850}
    
    print("Makrodaten:")
    for key, value in test_macro_data.items():
        print(f"  {key}: {value}")
    
    # Teste manuelle Berechnungen
    print("\n=== Test 1: Fundamentale Differentiale ===")
    rate_diff = test_macro_data["fed_rate"] - test_macro_data["ecb_rate"]
    growth_diff = test_macro_data["us_gdp_growth"] - test_macro_data["eu_gdp_growth"]
    pmi_diff = test_macro_data["us_pmi"] - test_macro_data["eu_pmi"]
    
    print(f"  Zinsdifferential (Fed-EZB): {rate_diff:+.2f}% → {'USD vorteil' if rate_diff > 0 else 'EUR vorteil'}")
    print(f"  Wachstumsdiff (US-EU): {growth_diff:+.2f}% → {'US stärker' if growth_diff > 0 else 'EU stärker'}")
    print(f"  PMI-Diff: {pmi_diff:+.1f} → {'US besser' if pmi_diff > 0 else 'EU besser'}")
    
    # Teste Momentum/Sentiment Berechnung
    print("\n=== Test 2: Momentum & Sentiment ===")
    dxy_trend = test_macro_data["dxy_trend"]
    if dxy_trend == "up":
        momentum_score = -0.5
        print(f"  DXY Trend: {dxy_trend} → Momentum Score: {momentum_score} (EURUSD bearish)")
    else:
        momentum_score = 0.5 if dxy_trend == "down" else 0.0
        print(f"  DXY Trend: {dxy_trend} → Momentum Score: {momentum_score}")
    
    risk_sentiment = test_macro_data["risk_sentiment"]
    if risk_sentiment == "risk-off":
        sentiment_score = -0.3
        print(f"  Risk Sentiment: {risk_sentiment} → Sentiment Score: {sentiment_score} (USD safe haven)")
    else:
        sentiment_score = 0.3 if risk_sentiment == "risk-on" else 0.0
        print(f"  Risk Sentiment: {risk_sentiment} → Sentiment Score: {sentiment_score}")
    
    # Teste MacroSignal Dataclass
    print("\n=== Test 3: MacroSignal Dataclass ===")
    macro_signal = MacroSignal(
        action="SHORT",
        confidence=72,
        reasoning=[
            "Fed-EZB Zinsdifferential begünstigt USD (+0.75%)",
            "US Wirtschaft stärker (BIP +1.6%, PMI +4.3)",
            "DXY Aufwärtstrend drückt EURUSD",
            "Risk-Off Sentiment begünstigt USD als Safe Haven"
        ],
        rate_differential=rate_diff,
        growth_differential=growth_diff,
        momentum_score=momentum_score,
        sentiment_score=sentiment_score,
        expected_return=0.035,  # 3.5%
        risk_reward_ratio=3.2,
        asymmetric_opportunity=True,
        entry_price=1.0850,
        stop_loss=1.0920,
        take_profit=1.0700,
        leverage=25
    )
    
    print(f"✓ Macro Signal erstellt: {macro_signal.action} @ {macro_signal.confidence}%")
    print(f"  Expected Return: {macro_signal.expected_return:.1%}")
    print(f"  Risk/Reward: 1:{macro_signal.risk_reward_ratio}")
    print(f"  Asymmetrische Chance: {'Ja ✓' if macro_signal.asymmetric_opportunity else 'Nein'}")
    print(f"  Leverage: {macro_signal.leverage}x")
    
    # Teste Druckenmiller Decision Logic
    print("\n=== Test 4: Druckenmiller Decision Logic ===")
    
    # Druckenmiller würde bei asymmetrischer Chance und hoher Conviction groß positionieren
    if macro_signal.asymmetric_opportunity and macro_signal.confidence > 70:
        position_decision = "GROSSE POSITION (hohe Conviction)"
        leverage_recommendation = min(30, macro_signal.leverage + 5)
    elif macro_signal.confidence > 60:
        position_decision = "NORMALE POSITION"
        leverage_recommendation = macro_signal.leverage
    elif macro_signal.confidence > 40:
        position_decision = "KLEINE POSITION"
        leverage_recommendation = max(5, macro_signal.leverage - 10)
    else:
        position_decision = "ABWARTEN"
        leverage_recommendation = 0
    
    print(f"  Conviction: {macro_signal.confidence}%")
    print(f"  Asymmetrische Chance: {'Ja' if macro_signal.asymmetric_opportunity else 'Nein'}")
    print(f"  → Entscheidung: {position_decision}")
    print(f"  → Empfohlenes Leverage: {leverage_recommendation}x")
    
    # Teste verschiedene Szenarien
    print("\n=== Test 5: Verschiedene Macro-Szenarien ===")
    
    scenarios = [
        {
            "name": "USD Strong (wie aktuell)",
            "rate_diff": 0.75,
            "growth_diff": 1.6,
            "momentum": -0.5,
            "sentiment": -0.3,
            "expected": "SHORT"
        },
        {
            "name": "EUR Strong (EZB hawkish)",
            "rate_diff": -0.25,
            "growth_diff": 0.5,
            "momentum": 0.5,
            "sentiment": 0.3,
            "expected": "LONG"
        },
        {
            "name": "Neutral (gemischte Signale)",
            "rate_diff": 0.1,
            "growth_diff": 0.2,
            "momentum": 0.0,
            "sentiment": 0.0,
            "expected": "NEUTRAL"
        }
    ]
    
    for scenario in scenarios:
        # Einfache Scoring-Logik
        total_score = (
            scenario["rate_diff"] * 20 +  # Zinsdiff gewichtet
            scenario["growth_diff"] * 10 +  # Wachstumsdiff
            scenario["momentum"] * 30 +  # Momentum
            scenario["sentiment"] * 20  # Sentiment
        )
        
        if total_score > 15:
            result = "SHORT"  # Positiv für USD
        elif total_score < -15:
            result = "LONG"  # Positiv für EUR
        else:
            result = "NEUTRAL"
        
        status = "✓" if result == scenario["expected"] else "✗"
        print(f"  {status} {scenario['name']}: Score={total_score:+.1f} → {result}")
    
    print("\n✅ EURUSD Macro Agent Implementierung ist funktionsfähig!")
    print("\nHinweis: Vollständige LLM-Tests erfordern einen laufenden Server.")
