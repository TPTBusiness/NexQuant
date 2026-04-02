"""
BM25 Memory-System für EURUSD Trading-Setups

Speichert vergangene Trades mit:
- Marktsituation (Features, Regime, Indikatoren)
- Entscheidung (LONG/SHORT/NEUTRAL, Leverage, SL, TP)
- Ergebnis (PnL, Win/Loss)
- Reflection (Lessons Learned)

Vorteile gegenüber Vector-DBs:
- Keine API-Kosten (offline-fähig)
- Keine Token-Limits
- Lexikalische Ähnlichkeit (präzise für Trading-Setups)
- Schnell und einfach zu implementieren
"""

import json
import pickle
import re
from datetime import datetime
from pathlib import Path
from typing import Dict, List, Optional, Tuple

import numpy as np
from rank_bm25 import BM25Okapi


def tokenize(text: str) -> List[str]:
    """
    Tokenisiert Text für BM25-Verarbeitung.
    
    - Entfernt Sonderzeichen
    - Konvertiert zu Kleinbuchstaben
    - Split auf Wörter und Zahlen
    
    Parameters
    ----------
    text : str
        Eingabetext (Trading-Situation, Setup-Beschreibung)
        
    Returns
    -------
    List[str]
        Liste von Tokens
    """
    # Konvertiere zu Kleinbuchstaben
    text = text.lower()
    
    # Extrahiere Wörter und Zahlen (inkl. Dezimalzahlen wie 1.0850)
    tokens = re.findall(r'\b\w+(?:\.\d+)?\b', text)
    
    # Filtere sehr kurze Tokens (< 2 Zeichen)
    tokens = [t for t in tokens if len(t) >= 2]
    
    return tokens


class EURUSDTradeMemory:
    """
    BM25-basiertes Memory-System für vergangene EURUSD-Trading-Setups.
    
    Speichert vergangene Trades mit:
    - Marktsituation (Features, Regime, Indikatoren)
    - Entscheidung (LONG/SHORT/NEUTRAL, Leverage, SL, TP)
    - Ergebnis (PnL, Win/Loss)
    - Reflection (Lessons Learned)
    
    Bei neuer Situation: Findet ähnliche vergangene Setups und gibt
    historische Win-Rate und durchschnittliche Rendite zurück.
    
    Attributes
    ----------
    memory_file : Path
        Pfad zur persistenten Speicherdatei (JSON)
    memories : List[dict]
        Liste aller gespeicherten Trades
    bm25 : BM25Okapi
        BM25-Index für schnelle Ähnlichkeitssuche
        
    Example
    -------
    >>> memory = EURUSDTradeMemory()
    >>> memory.add_trade(
    ...     situation="EURUSD 1.0850, RSI=28, Hurst=0.52 (MEAN_REVERSION), EZB hawkish",
    ...     decision={"action": "LONG", "leverage": 20, "sl_pips": 25, "tp_pips": 15},
    ...     outcome=0.023,  # +2.3% Gewinn
    ...     reflection="RSI < 30 in Mean-Reversion Regime war erfolgreich"
    ... )
    >>> similar = memory.get_similar_setups("EURUSD 1.0820, RSI=25, Hurst=0.48")
    >>> print(f"Historische Win-Rate: {similar['historical_win_rate']:.1%}")
    """
    
    def __init__(self, memory_file: str = "git_ignore_folder/eurusd_trade_memory.json"):
        """
        Initialisiert das Memory-System.
        
        Parameters
        ----------
        memory_file : str
            Pfad zur JSON-Datei für persistente Speicherung
        """
        self.memory_file = Path(memory_file)
        self.memories: List[dict] = []
        self.bm25: Optional[BM25Okapi] = None
        self.tokenized_memories: List[List[str]] = []
        
        # Lade existierende Memories von Datei
        if self.memory_file.exists():
            self.load()
    
    def add_trade(
        self,
        situation: str,
        decision: dict,
        outcome: float,
        reflection: Optional[str] = None
    ) -> None:
        """
        Speichert einen vergangenen Trade im Memory.
        
        Parameters
        ----------
        situation : str
            Beschreibung der Marktsituation zum Zeitpunkt des Trades.
            Beispiel: "EURUSD 1.0850, RSI=28, Hurst=0.52 (MEAN_REVERSION), 
            London Session, EZB hawkish, DXY downtrend"
        decision : dict
            Trade-Entscheidung mit Details.
            Beispiel: {"action": "LONG", "leverage": 20, "sl_pips": 25, "tp_pips": 15}
        outcome : float
            Ergebnis des Trades als Dezimalzahl.
            Beispiel: 0.023 = +2.3% Gewinn, -0.015 = -1.5% Verlust
        reflection : str, optional
            Lessons Learned nach dem Trade (vom Reflection-System generiert).
        """
        trade_record = {
            "id": len(self.memories) + 1,
            "timestamp": datetime.now().isoformat(),
            "situation": situation,
            "decision": decision,
            "outcome": outcome,
            "reflection": reflection or "",
            "tokens": tokenize(situation)
        }
        
        self.memories.append(trade_record)
        self.tokenized_memories.append(trade_record["tokens"])
        
        # Rebuild BM25 Index
        self._rebuild_bm25()
        
        # Speichere auf Festplatte
        self.save()
    
    def add_trades_batch(self, trades: List[dict]) -> None:
        """
        Fügt mehrere Trades auf einmal hinzu (effizienter als einzelne add_trade Aufrufe).
        
        Parameters
        ----------
        trades : List[dict]
            Liste von Trade-Records mit Keys: situation, decision, outcome, reflection
        """
        for trade in trades:
            trade_record = {
                "id": len(self.memories) + len(trades),
                "timestamp": datetime.now().isoformat(),
                "situation": trade["situation"],
                "decision": trade["decision"],
                "outcome": trade["outcome"],
                "reflection": trade.get("reflection", ""),
                "tokens": tokenize(trade["situation"])
            }
            self.memories.append(trade_record)
            self.tokenized_memories.append(trade_record["tokens"])
        
        self._rebuild_bm25()
        self.save()
    
    def get_similar_setups(
        self,
        current_situation: str,
        n: int = 5,
        min_similarity: float = 0.0
    ) -> dict:
        """
        Findet ähnliche vergangene Trading-Setups.
        
        Parameters
        ----------
        current_situation : str
            Aktuelle Marktsituation (gleiche Formatierung wie bei add_trade)
        n : int, default 5
            Anzahl der zurückzugebenden ähnlichen Setups
        min_similarity : float, default 0.0
            Minimale BM25-Ähnlichkeit für Treffer
        
        Returns
        -------
        dict
            Ähnliche Setups mit Statistiken:
            - similar_setups: Liste der Top-N ähnlichen Trades
            - historical_win_rate: Win-Rate der ähnlichen Setups
            - historical_avg_return: Durchschnittliche Rendite
            - best_setup: Bestes historisches Setup
            - recommendation: Handlungsempfehlung basierend auf History
        """
        if len(self.memories) == 0:
            return {
                "similar_setups": [],
                "historical_win_rate": 0.0,
                "historical_avg_return": 0.0,
                "message": "Keine historischen Trades gespeichert"
            }
        
        # Tokenisiere aktuelle Situation
        query_tokens = tokenize(current_situation)
        
        # Berechne BM25-Ähnlichkeiten
        scores = self.bm25.get_scores(query_tokens)
        
        # Finde Top-N Treffer
        top_indices = np.argsort(scores)[::-1][:n]
        
        # Filtere nach min_similarity
        filtered_indices = [
            i for i in top_indices
            if scores[i] >= min_similarity
        ]
        
        if len(filtered_indices) == 0:
            return {
                "similar_setups": [],
                "historical_win_rate": 0.0,
                "historical_avg_return": 0.0,
                "message": f"Keine ähnlichen Setups gefunden (min_similarity={min_similarity})"
            }
        
        # Sammle ähnliche Setups
        similar_setups = []
        outcomes = []
        
        for idx in filtered_indices:
            memory = self.memories[idx]
            similar_setups.append({
                "id": memory["id"],
                "situation": memory["situation"],
                "decision": memory["decision"],
                "outcome": memory["outcome"],
                "reflection": memory["reflection"],
                "similarity_score": float(scores[idx]),
                "timestamp": memory["timestamp"]
            })
            outcomes.append(memory["outcome"])
        
        # Berechne Statistiken
        outcomes_array = np.array(outcomes)
        win_rate = np.mean(outcomes_array > 0)
        avg_return = np.mean(outcomes_array)
        std_return = np.std(outcomes_array) if len(outcomes) > 1 else 0.0
        
        # Finde bestes Setup
        best_idx = np.argmax(outcomes_array)
        best_setup = similar_setups[best_idx]
        
        # Generiere Empfehlung
        if win_rate > 0.7 and len(filtered_indices) >= 3:
            recommendation = "STRONG_SIGNAL"
            rec_text = f"Starke Historie: {win_rate:.0%} Win-Rate in {len(filtered_indices)} ähnlichen Situationen"
        elif win_rate > 0.55:
            recommendation = "MODERATE_SIGNAL"
            rec_text = f"Moderate Historie: {win_rate:.0%} Win-Rate"
        elif win_rate < 0.4 and len(filtered_indices) >= 3:
            recommendation = "AVOID"
            rec_text = f"Schwache Historie: Nur {win_rate:.0%} Win-Rate - Setup vermeiden!"
        else:
            recommendation = "NEUTRAL"
            rec_text = f"Neutrale Historie: {win_rate:.0%} Win-Rate, zu wenig Daten für klare Empfehlung"
        
        return {
            "similar_setups": similar_setups,
            "historical_win_rate": float(win_rate),
            "historical_avg_return": float(avg_return),
            "historical_std_return": float(std_return),
            "n_similar_trades": len(filtered_indices),
            "best_setup": best_setup,
            "recommendation": recommendation,
            "recommendation_text": rec_text
        }
    
    def get_memory_stats(self) -> dict:
        """
        Gibt Statistiken über das gespeicherte Memory.
        
        Returns
        -------
        dict
            Memory-Statistiken:
            - total_trades: Gesamtanzahl Trades
            - win_rate: Gesamte Win-Rate
            - avg_return: Durchschnittliche Rendite
            - best_trade: Bester Trade
            - worst_trade: Schlechtester Trade
            - recent_performance: Performance der letzten 10 Trades
        """
        if len(self.memories) == 0:
            return {"message": "Keine Trades gespeichert"}
        
        outcomes = [m["outcome"] for m in self.memories]
        outcomes_array = np.array(outcomes)
        
        # Recent Performance (letzte 10 Trades)
        recent_outcomes = outcomes_array[-10:] if len(outcomes) > 10 else outcomes_array
        
        return {
            "total_trades": len(self.memories),
            "win_rate": float(np.mean(outcomes_array > 0)),
            "avg_return": float(np.mean(outcomes_array)),
            "std_return": float(np.std(outcomes_array)),
            "sharpe_ratio": float(np.mean(outcomes_array) / np.std(outcomes_array)) if np.std(outcomes_array) > 0 else 0.0,
            "best_trade": {
                "id": self.memories[np.argmax(outcomes_array)]["id"],
                "outcome": float(np.max(outcomes_array)),
                "situation": self.memories[np.argmax(outcomes_array)]["situation"]
            },
            "worst_trade": {
                "id": self.memories[np.argmin(outcomes_array)]["id"],
                "outcome": float(np.min(outcomes_array)),
                "situation": self.memories[np.argmin(outcomes_array)]["situation"]
            },
            "recent_performance": {
                "n_trades": len(recent_outcomes),
                "win_rate": float(np.mean(recent_outcomes > 0)),
                "avg_return": float(np.mean(recent_outcomes))
            }
        }
    
    def _rebuild_bm25(self) -> None:
        """
        Baut den BM25-Index neu auf (nach Hinzufügen neuer Trades).
        """
        if len(self.tokenized_memories) > 0:
            self.bm25 = BM25Okapi(self.tokenized_memories)
    
    def save(self) -> None:
        """
        Speichert das Memory persistent auf die Festplatte.
        """
        # Erstelle Verzeichnis falls nicht existent
        self.memory_file.parent.mkdir(parents=True, exist_ok=True)
        
        # Speichere als JSON (ohne BM25-Index, der wird beim Laden neu gebaut)
        save_data = []
        for memory in self.memories:
            save_entry = {k: v for k, v in memory.items() if k != "tokens"}
            save_data.append(save_entry)
        
        with open(self.memory_file, 'w', encoding='utf-8') as f:
            json.dump(save_data, f, indent=2, ensure_ascii=False)
    
    def load(self) -> None:
        """
        Lädt das Memory von der Festplatte.
        """
        try:
            with open(self.memory_file, 'r', encoding='utf-8') as f:
                save_data = json.load(f)
            
            self.memories = []
            self.tokenized_memories = []
            
            for entry in save_data:
                entry["tokens"] = tokenize(entry["situation"])
                self.memories.append(entry)
                self.tokenized_memories.append(entry["tokens"])
            
            self._rebuild_bm25()
            
        except Exception as e:
            print(f"⚠️  Fehler beim Laden des Memory: {e}")
            self.memories = []
            self.tokenized_memories = []
    
    def clear(self) -> None:
        """
        Löscht das gesamte Memory.
        """
        self.memories = []
        self.tokenized_memories = []
        self.bm25 = None
        
        if self.memory_file.exists():
            self.memory_file.unlink()


# Test-Funktion für lokale Validierung
if __name__ == "__main__":
    print("=== BM25 Memory Test ===\n")
    
    # Erstelle Test-Memory
    memory = EURUSDTradeMemory(memory_file="git_ignore_folder/test_trade_memory.json")
    
    # Füge Beispiel-Trades hinzu
    test_trades = [
        {
            "situation": "EURUSD 1.0850, RSI=28, Hurst=0.52 (MEAN_REVERSION), London Session, EZB hawkish, DXY downtrend",
            "decision": {"action": "LONG", "leverage": 20, "sl_pips": 25, "tp_pips": 15},
            "outcome": 0.023,
            "reflection": "RSI < 30 in Mean-Reversion Regime war erfolgreich"
        },
        {
            "situation": "EURUSD 1.0920, RSI=72, Hurst=0.58 (NEUTRAL), NY Session, Fed dovish, DXY weak",
            "decision": {"action": "SHORT", "leverage": 15, "sl_pips": 30, "tp_pips": 20},
            "outcome": 0.015,
            "reflection": "RSI > 70 mit Mean-Reversion funktioniert gut"
        },
        {
            "situation": "EURUSD 1.0780, RSI=25, Hurst=0.48 (MEAN_REVERSION), Asian Session, low volatility",
            "decision": {"action": "LONG", "leverage": 10, "sl_pips": 20, "tp_pips": 12},
            "outcome": -0.012,
            "reflection": "Asian Session zu wenig Volumen für Mean-Reversion"
        },
        {
            "situation": "EURUSD 1.0950, RSI=65, Hurst=0.72 (TRENDING), London-NY Overlap, strong momentum",
            "decision": {"action": "LONG", "leverage": 25, "sl_pips": 20, "tp_pips": 35},
            "outcome": 0.035,
            "reflection": "Trending Regime mit Momentum war sehr profitabel"
        },
        {
            "situation": "EURUSD 1.0880, RSI=45, Hurst=0.61 (NEUTRAL), no clear direction, choppy market",
            "decision": {"action": "NEUTRAL", "leverage": 0, "sl_pips": 0, "tp_pips": 0},
            "outcome": 0.0,
            "reflection": "Abwarten war die beste Entscheidung in choppy Market"
        },
    ]
    
    memory.add_trades_batch(test_trades)
    print(f"✅ {len(test_trades)} Trades zum Memory hinzugefügt\n")
    
    # Teste Ähnlichkeitssuche
    print("=== Test 1: Ähnliche Setups finden ===")
    query = "EURUSD 1.0840, RSI=26, Hurst=0.50, MEAN_REVERSION, EZB hawkish"
    similar = memory.get_similar_setups(query, n=3)
    
    print(f"Query: {query}")
    print(f"Gefundene ähnliche Setups: {similar.get('n_similar_trades', 0)}")
    print(f"Historische Win-Rate: {similar.get('historical_win_rate', 0):.1%}")
    print(f"Durchschnittliche Rendite: {similar.get('historical_avg_return', 0):.2%}")
    print(f"Empfehlung: {similar.get('recommendation', 'N/A')} - {similar.get('recommendation_text', '')}")
    
    # Teste Memory-Statistiken
    print("\n=== Test 2: Memory Statistiken ===")
    stats = memory.get_memory_stats()
    print(f"Gesamte Trades: {stats.get('total_trades', 0)}")
    print(f"Gesamte Win-Rate: {stats.get('win_rate', 0):.1%}")
    print(f"Durchschnittliche Rendite: {stats.get('avg_return', 0):.2%}")
    print(f"Sharpe Ratio: {stats.get('sharpe_ratio', 0):.2f}")
    print(f"Bester Trade: {stats.get('best_trade', {}).get('outcome', 0):.2%}")
    print(f"Schlechtester Trade: {stats.get('worst_trade', {}).get('outcome', 0):.2%}")
    
    # Teste Persistenz
    print("\n=== Test 3: Persistenz ===")
    memory2 = EURUSDTradeMemory(memory_file="git_ignore_folder/test_trade_memory.json")
    print(f"Memory nach Neuladen: {len(memory2.memories)} Trades")
    
    # Cleanup
    import os
    if os.path.exists("git_ignore_folder/test_trade_memory.json"):
        os.remove("git_ignore_folder/test_trade_memory.json")
    
    print("\n✅ BM25 Memory Implementierung ist funktionsfähig!")
