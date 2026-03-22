"""
FX Validator Configuration
Angepasst für EURUSD 15min intraday trading
"""
import os

FX_CONFIG = {
    "instrument": "EURUSD=X",
    "frequency": "15min",
    "llm_provider": "openai",
    "backend_url": os.getenv("OPENAI_API_BASE", "http://localhost:8081/v1"),
    "api_key": os.getenv("OPENAI_API_KEY", "local"),
    "deep_think_llm": os.getenv("CHAT_MODEL", "openai/qwen3.5-35b"),
    "quick_think_llm": os.getenv("CHAT_MODEL", "openai/qwen3.5-35b"),
    "max_debate_rounds": 2,
    "max_risk_discuss_rounds": 1,
    # FX-spezifisch
    "spread_bps": 1.5,
    "target_arr": 9.62,
    "max_drawdown": 20.0,
    "sessions": {
        "asian":   ("00:00", "08:00"),
        "london":  ("08:00", "16:00"),
        "ny":      ("13:00", "21:00"),
        "overlap": ("13:00", "16:00"),
    },
}
