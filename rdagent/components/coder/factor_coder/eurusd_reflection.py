"""
EURUSD Reflection-System für kontinuierliches Lernen

Nach jedem Trade:
1. Reflektiere über Entscheidung und Ergebnis
2. Extrahiere Lessons Learned
3. Speichere im Memory für zukünftige ähnliche Situationen
4. Passe Strategie basierend auf History an
"""

import json
import sys
from dataclasses import dataclass
from datetime import datetime
from pathlib import Path
from typing import Dict, List, Literal, Optional

sys.path.insert(0, str(Path(__file__).parent))

from eurusd_memory import EURUSDTradeMemory


@dataclass
class TradeReflection:
    """Reflection eines Trades."""
    trade_id: int
    timestamp: str
    
    # Original-Entscheidung
    original_action: Literal["LONG", "SHORT", "NEUTRAL"]
    original_confidence: int
    original_reasoning: List[str]
    
    # Ergebnis
    outcome: float  # PnL in %
    outcome_type: Literal["WIN", "LOSS", "BREAKEVEN"]
    
    # Reflection
    was_decision_correct: bool
    what_went_right: List[str]
    what_went_wrong: List[str]
    lessons_learned: List[str]
    
    # Empfehlung für zukünftige Trades
    future_recommendation: str
    similar_situations_to_watch: List[str]


class EURUSDReflectionSystem:
    """
    Reflection-System für EURUSD Trading.
    
    Verwendet BM25 Memory um aus vergangenen Trades zu lernen.
    
    Verwendung:
    >>> reflection = EURUSDReflectionSystem()
    >>> 
    >>> # Nach einem Trade
    >>> trade_result = {
    ...     "action": "LONG",
    ...     "confidence": 75,
    ...     "reasoning": ["RSI < 30", "Mean-Reversion"],
    ...     "entry": 1.0850,
    ...     "exit": 1.0880,
    ...     "pnl": 0.028  # +2.8%
    ... }
    >>> 
    >>> # Reflektieren
    >>> trade_reflection = reflection.reflect_trade(trade_result)
    >>> 
    >>> # Memory aktualisieren
    >>> reflection.memory.add_trade(
    ...     situation="EURUSD 1.0850, RSI=28, Mean-Reversion",
    ...     decision={"action": "LONG"},
    ...     outcome=0.028,
    ...     reflection=str(trade_reflection)
    ... )
    """
    
    def __init__(self, memory_file: str = "git_ignore_folder/eurusd_trade_memory.json"):
        self.memory = EURUSDTradeMemory(memory_file)
    
    def reflect_trade(
        self,
        trade_result: dict,
        market_context: Optional[dict] = None
    ) -> TradeReflection:
        """
        Reflektiert einen abgeschlossenen Trade.
        
        Parameters
        ----------
        trade_result : dict
            Trade-Ergebnis mit Keys:
            - action: LONG/SHORT/NEUTRAL
            - confidence: 0-100
            - reasoning: Liste von Gründen
            - entry: Entry-Preis
            - exit: Exit-Preis
            - pnl: PnL in % (positiv = Gewinn)
            - max_drawdown: Maximaler Drawdown während Trade
            - max_profit: Maximaler Profit während Trade
            - duration: Haltedauer in Minuten
            
        market_context : dict, optional
            Marktkontext zum Zeitpunkt des Trades
            
        Returns
        -------
        TradeReflection
            Reflection des Trades
        """
        # Bestimme Outcome-Typ
        pnl = trade_result.get("pnl", 0.0)
        if pnl > 0.005:  # > 0.5%
            outcome_type = "WIN"
        elif pnl < -0.005:  # < -0.5%
            outcome_type = "LOSS"
        else:
            outcome_type = "BREAKEVEN"
        
        # Analysiere Trade
        was_correct = pnl > 0
        what_went_right = []
        what_went_wrong = []
        lessons_learned = []
        
        # Analyse basierend auf Ergebnis
        if was_correct:
            what_went_right.extend(self._analyze_success(trade_result))
            lessons_learned.extend(self._extract_positive_lessons(trade_result))
        else:
            what_went_wrong.extend(self._analyze_failure(trade_result))
            lessons_learned.extend(self._extract_negative_lessons(trade_result))
        
        # Generiere Future Recommendation
        future_recommendation = self._generate_recommendation(
            trade_result, was_correct, lessons_learned
        )
        
        # Finde ähnliche Situationen im Memory
        similar_situations = self._find_similar_situations(trade_result, market_context)
        
        return TradeReflection(
            trade_id=len(self.memory.memories) + 1,
            timestamp=datetime.now().isoformat(),
            original_action=trade_result.get("action", "NEUTRAL"),
            original_confidence=trade_result.get("confidence", 50),
            original_reasoning=trade_result.get("reasoning", []),
            outcome=pnl,
            outcome_type=outcome_type,
            was_decision_correct=was_correct,
            what_went_right=what_went_right,
            what_went_wrong=what_went_wrong,
            lessons_learned=lessons_learned,
            future_recommendation=future_recommendation,
            similar_situations_to_watch=similar_situations
        )
    
    def _analyze_success(self, trade_result: dict) -> List[str]:
        """Analysiert was bei einem erfolgreichen Trade richtig lief."""
        points = []
        
        pnl = trade_result.get("pnl", 0)
        if pnl > 0.03:  # > 3%
            points.append(f"Ausgezeichnete Performance: +{pnl:.1%}")
        
        # Check ob Entry gut war
        max_profit = trade_result.get("max_profit", pnl)
        max_drawdown = trade_result.get("max_drawdown", 0)
        
        if max_profit > pnl * 1.5:
            points.append("Gutes Timing: Trade war zeitweise noch profitabler")
        
        if max_drawdown < abs(pnl) * 0.5:
            points.append("Geringer Drawdown während Trade")
        
        # Check ob Confidence gerechtfertigt war
        confidence = trade_result.get("confidence", 50)
        if confidence > 70 and pnl > 0.02:
            points.append(f"Hohe Confidence ({confidence}%) war gerechtfertigt")
        
        # Check Risk/Reward
        if trade_result.get("risk_reward_actual", 1) > 2:
            points.append("Gutes Risk/Reward umgesetzt")
        
        return points
    
    def _analyze_failure(self, trade_result: dict) -> List[str]:
        """Analysiert was bei einem fehlgeschlagenen Trade falsch lief."""
        points = []
        
        pnl = trade_result.get("pnl", 0)
        
        # Check ob Stop-Loss eingehalten wurde
        if trade_result.get("stop_loss_hit", False):
            points.append("Stop-Loss wurde eingehalten (Disziplin)")
        else:
            points.append("Stop-Loss nicht eingehalten oder zu eng gesetzt")
        
        # Check ob Confidence gerechtfertigt war
        confidence = trade_result.get("confidence", 50)
        if confidence > 70 and pnl < -0.02:
            points.append(f"Zu hohe Confidence ({confidence}%) für diesen Trade")
        
        # Check Drawdown
        max_drawdown = trade_result.get("max_drawdown", abs(pnl))
        if max_drawdown > abs(pnl) * 2:
            points.append(f"Großer Drawdown ({max_drawdown:.1%}) vor Verlust")
        
        # Check Duration
        duration = trade_result.get("duration", 0)
        if duration > 480:  # > 8 Stunden
            points.append("Trade zu lange gehalten")
        
        return points
    
    def _extract_positive_lessons(self, trade_result: dict) -> List[str]:
        """Extrahiert positive Lessons Learned."""
        lessons = []
        
        # Extrahiere aus Reasoning was funktioniert hat
        reasoning = trade_result.get("reasoning", [])
        for reason in reasoning:
            if "RSI" in reason and trade_result.get("pnl", 0) > 0:
                lessons.append(f"RSI-basierte Signale funktionieren in diesem Setup")
            if "Mean-Reversion" in reason and trade_result.get("pnl", 0) > 0:
                lessons.append("Mean-Reversion Ansatz war erfolgreich")
            if "Trend" in reason and trade_result.get("pnl", 0) > 0:
                lessons.append("Trend-Following Ansatz war erfolgreich")
        
        # Füge allgemeine Lessons hinzu
        if trade_result.get("pnl", 0) > 0.03:
            lessons.append("Bei hoher Conviction größere Positionen möglich")
        
        return lessons
    
    def _extract_negative_lessons(self, trade_result: dict) -> List[str]:
        """Extrahiert negative Lessons Learned."""
        lessons = []
        
        # Counter-Trend Warnung
        reasoning = trade_result.get("reasoning", [])
        for reason in reasoning:
            if "Counter-Trend" in reason and trade_result.get("pnl", 0) < 0:
                lessons.append("Counter-Trend Trades in diesem Setup vermeiden")
        
        # Confidence-Adjustierung
        confidence = trade_result.get("confidence", 50)
        if confidence > 70 and trade_result.get("pnl", 0) < -0.02:
            lessons.append(f"Confidence bei ähnlichen Setups auf < {confidence}% begrenzen")
        
        # Stop-Loss Lesson
        if trade_result.get("max_drawdown", 0) > 0.05:
            lessons.append("Stop-Loss früher setzen oder enger gestalten")
        
        return lessons
    
    def _generate_recommendation(
        self,
        trade_result: dict,
        was_correct: bool,
        lessons: List[str]
    ) -> str:
        """Generiert Empfehlung für zukünftige Trades."""
        if was_correct:
            base = "Ähnliche Setups weiter handeln. "
            if trade_result.get("pnl", 0) > 0.03:
                base += "Bei hoher Conviction Positionsgröße erhöhen. "
        else:
            base = "Vorsicht bei ähnlichen Setups. "
            if trade_result.get("confidence", 50) > 70:
                base += "Confidence-Schwelle für diese Art von Trades senken. "
        
        if lessons:
            base += f"Wichtig: {lessons[0]}"
        
        return base
    
    def _find_similar_situations(
        self,
        trade_result: dict,
        market_context: Optional[dict]
    ) -> List[str]:
        """Findet ähnliche Situationen im Memory."""
        if not market_context:
            return []
        
        # Baue Query für Similarity Search
        situation_parts = [
            f"EURUSD {trade_result.get('entry', 'N/A')}",
            f"Action: {trade_result.get('action', 'N/A')}",
        ]
        
        if "hurst_regime" in market_context:
            situation_parts.append(f"Regime: {market_context['hurst_regime']}")
        if "rsi" in market_context:
            situation_parts.append(f"RSI: {market_context['rsi']}")
        
        situation_query = ", ".join(situation_parts)
        
        # Suche ähnliche Situationen
        similar = self.memory.get_similar_setups(situation_query, n=3)
        
        if similar.get("historical_win_rate", 0) > 0:
            return [
                f"Historische Win-Rate bei ähnlichen Setups: {similar['historical_win_rate']:.0%}",
                f"Durchschnittliche Rendite: {similar.get('historical_avg_return', 0):.1%}",
                similar.get("recommendation_text", "")
            ]
        
        return []
    
    def get_aggregate_insights(self, last_n_trades: int = 20) -> dict:
        """
        Gibt aggregierte Insights aus letzten Trades.
        
        Parameters
        ----------
        last_n_trades : int, default 20
            Anzahl der Trades für Analyse
            
        Returns
        -------
        dict
            Aggregierte Insights
        """
        if len(self.memory.memories) == 0:
            return {"message": "Keine Trades im Memory"}
        
        # Hole letzte N Trades
        recent_trades = self.memory.memories[-last_n_trades:]
        
        # Berechne Statistiken
        outcomes = [t.get("outcome", 0) for t in recent_trades]
        win_rate = sum(1 for o in outcomes if o > 0) / len(outcomes)
        avg_return = sum(outcomes) / len(outcomes)
        
        # Finde häufigste Reasoning-Patterns in Winners vs Losers
        winner_reasons = []
        loser_reasons = []
        
        for trade in recent_trades:
            outcome = trade.get("outcome", 0)
            reflection = trade.get("reflection", "")
            
            if outcome > 0:
                winner_reasons.append(reflection)
            else:
                loser_reasons.append(reflection)
        
        return {
            "total_trades": len(recent_trades),
            "win_rate": win_rate,
            "avg_return": avg_return,
            "total_pnl": sum(outcomes),
            "best_trade": max(outcomes),
            "worst_trade": min(outcomes),
            "n_winner_reasons": len(winner_reasons),
            "n_loser_reasons": len(loser_reasons)
        }


# Test-Funktion für lokale Validierung
if __name__ == "__main__":
    print("=== EURUSD Reflection System Test ===\n")
    
    # Erstelle Reflection System
    reflection = EURUSDReflectionSystem(memory_file="git_ignore_folder/test_reflection_memory.json")
    
    # Test 1: Reflektiere erfolgreichen Trade
    print("=== Test 1: Erfolgreicher Trade ===")
    winning_trade = {
        "action": "LONG",
        "confidence": 75,
        "reasoning": ["RSI < 30 in Mean-Reversion Regime", "EZB hawkish"],
        "entry": 1.0850,
        "exit": 1.0890,
        "pnl": 0.037,  # +3.7%
        "max_profit": 0.045,
        "max_drawdown": 0.008,
        "duration": 180  # 3 Stunden
    }
    
    ref_win = reflection.reflect_trade(winning_trade)
    print(f"Trade ID: {ref_win.trade_id}")
    print(f"Outcome: {ref_win.outcome:.1%} ({ref_win.outcome_type})")
    print(f"Was Decision Correct: {'Ja ✓' if ref_win.was_decision_correct else 'Nein'}")
    print(f"What Went Right:")
    for point in ref_win.what_went_right[:3]:
        print(f"  • {point}")
    print(f"Lessons Learned:")
    for lesson in ref_win.lessons_learned[:2]:
        print(f"  • {lesson}")
    print(f"Recommendation: {ref_win.future_recommendation[:80]}...")
    
    # Test 2: Reflektiere verlorenen Trade
    print("\n=== Test 2: Verlorener Trade ===")
    losing_trade = {
        "action": "SHORT",
        "confidence": 80,
        "reasoning": ["DXY breakout", "US NFP beat"],
        "entry": 1.0880,
        "exit": 1.0850,
        "pnl": -0.028,  # -2.8%
        "max_profit": 0.005,
        "max_drawdown": 0.045,
        "duration": 420,  # 7 Stunden
        "stop_loss_hit": True
    }
    
    ref_loss = reflection.reflect_trade(losing_trade)
    print(f"Trade ID: {ref_loss.trade_id}")
    print(f"Outcome: {ref_loss.outcome:.1%} ({ref_loss.outcome_type})")
    print(f"Was Decision Correct: {'Ja' if ref_loss.was_decision_correct else 'Nein ✗'}")
    print(f"What Went Wrong:")
    for point in ref_loss.what_went_wrong[:3]:
        print(f"  • {point}")
    print(f"Lessons Learned:")
    for lesson in ref_loss.lessons_learned[:2]:
        print(f"  • {lesson}")
    
    # Test 3: Speichere im Memory
    print("\n=== Test 3: Memory Update ===")
    reflection.memory.add_trade(
        situation="EURUSD 1.0850, RSI=28, Mean-Reversion, EZB hawkish",
        decision={"action": "LONG", "confidence": 75},
        outcome=0.037,
        reflection=str(ref_win)
    )
    
    reflection.memory.add_trade(
        situation="EURUSD 1.0880, DXY breakout, US NFP beat",
        decision={"action": "SHORT", "confidence": 80},
        outcome=-0.028,
        reflection=str(ref_loss)
    )
    
    print(f"Trades im Memory: {len(reflection.memory.memories)}")
    
    # Test 4: Aggregierte Insights
    print("\n=== Test 4: Aggregierte Insights ===")
    insights = reflection.get_aggregate_insights()
    print(f"Anzahl Trades: {insights.get('total_trades', 0)}")
    print(f"Win-Rate: {insights.get('win_rate', 0):.1%}")
    print(f"Durchschnittliche Rendite: {insights.get('avg_return', 0):.2%}")
    print(f"Gesamt-PnL: {insights.get('total_pnl', 0):.1%}")
    
    # Cleanup
    import os
    if os.path.exists("git_ignore_folder/test_reflection_memory.json"):
        os.remove("git_ignore_folder/test_reflection_memory.json")
    
    print("\n✅ EURUSD Reflection System Implementierung ist funktionsfähig!")
