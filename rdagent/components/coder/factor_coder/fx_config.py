"""
FX Config - Zentrale Konfiguration für EURUSD Trading

Wird verwendet von:
- Macro Agent (Live-Daten)
- Debate Team (Session-Analyse)
- Position Sizing (Spread, Costs)
- Web Dashboard (Zielwerte)
"""

import os
from dataclasses import dataclass
from typing import Dict, Tuple


@dataclass
class FXConfig:
    """Zentrale FX-Konfiguration."""
    
    # Instrument & Daten
    instrument: str = "EURUSD=X"
    frequency: str = "1min"
    data_path: str = os.path.expanduser("~/.qlib/qlib_data/eurusd_1min_data")
    
    # LLM Provider
    llm_provider: str = "openai"
    backend_url: str = os.getenv("OPENAI_API_BASE", "http://localhost:8081/v1")
    api_key: str = os.getenv("OPENAI_API_KEY", "local")
    chat_model: str = os.getenv("CHAT_MODEL", "qwen3.5-35b")
    embedding_model: str = os.getenv("EMBEDDING_MODEL", "nomic-embed-text")
    
    # Trading-Parameter
    spread_bps: float = 1.5  # 1.5 bps Spread
    target_arr: float = 9.62  # Ziel: 9.62% annualisierte Rendite
    max_drawdown: float = 20.0  # Max 20% Drawdown
    cost_rate: float = 0.00015  # 0.015% pro Trade
    
    # Sessions (UTC)
    sessions: Dict[str, Tuple[str, str]] = None
    
    # Debate & Risk
    max_debate_rounds: int = 2
    max_risk_discuss_rounds: int = 1
    
    # Memory & Reflection
    memory_file: str = "git_ignore_folder/eurusd_trade_memory.json"
    reflection_enabled: bool = True
    
    def __post_init__(self):
        if self.sessions is None:
            self.sessions = {
                "asian": ("00:00", "08:00"),
                "london": ("08:00", "16:00"),
                "ny": ("13:00", "21:00"),
                "overlap": ("13:00", "16:00"),
            }
    
    def get_current_session(self) -> str:
        """Bestimmt aktuelle FX-Session basierend auf UTC-Zeit."""
        from datetime import datetime, timezone
        
        hour_utc = datetime.now(timezone.utc).hour
        
        if 0 <= hour_utc < 8:
            return "asian"
        elif 8 <= hour_utc < 13:
            return "london"
        elif 13 <= hour_utc < 16:
            return "overlap"
        elif 16 <= hour_utc < 21:
            return "ny"
        else:
            return "after_hours"
    
    def get_session_description(self, session: str = None) -> dict:
        """Gibt Beschreibung der Session."""
        if session is None:
            session = self.get_current_session()
        
        descriptions = {
            "asian": {
                "name": "Asian Session",
                "hours": "00:00-08:00 UTC",
                "characteristics": "Low volume, ranging market",
                "recommended_strategy": "Mean Reversion",
                "avoid": "Momentum strategies"
            },
            "london": {
                "name": "London Session",
                "hours": "08:00-16:00 UTC",
                "characteristics": "High volume, trending market",
                "recommended_strategy": "Momentum/Trend-Following",
                "avoid": "Counter-trend trades"
            },
            "overlap": {
                "name": "London-NY Overlap",
                "hours": "13:00-16:00 UTC",
                "characteristics": "Highest volume, strong directional moves",
                "recommended_strategy": "Strong Momentum",
                "avoid": "Range trading"
            },
            "ny": {
                "name": "NY Session",
                "hours": "13:00-21:00 UTC",
                "characteristics": "Moderate volume, reversals after London close",
                "recommended_strategy": "Momentum/Reversal",
                "avoid": "Late entries after 20:00"
            },
            "after_hours": {
                "name": "After Hours",
                "hours": "21:00-00:00 UTC",
                "characteristics": "Very low volume, wide spreads",
                "recommended_strategy": "Avoid trading",
                "avoid": "All strategies"
            }
        }
        
        return descriptions.get(session, descriptions["after_hours"])


# Globale Instanz
fx_config = FXConfig()


def get_fx_config() -> FXConfig:
    """Gibt globale FX-Config zurück."""
    return fx_config


# Test
if __name__ == "__main__":
    config = get_fx_config()
    
    print("=== FX Config Test ===\n")
    print(f"Instrument: {config.instrument}")
    print(f"Frequency: {config.frequency}")
    print(f"Target ARR: {config.target_arr}%")
    print(f"Max Drawdown: {config.max_drawdown}%")
    print(f"Spread: {config.spread_bps} bps")
    
    print(f"\nAktuelle Session: {config.get_current_session()}")
    session_desc = config.get_session_description()
    print(f"  Name: {session_desc['name']}")
    print(f"  Hours: {session_desc['hours']}")
    print(f"  Characteristics: {session_desc['characteristics']}")
    print(f"  Recommended: {session_desc['recommended_strategy']}")
    
    print("\n✅ FX Config funktioniert!")
