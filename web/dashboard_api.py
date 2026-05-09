"""
NexQuant Dashboard API

Flask-Backend für das Web-Dashboard.
Zeigt COMPLETE Progress von EURUSD Trading-Agent.

Features:
- Live Trading Progress (Loop, Step, Faktor)
- Performance Metrics (Win-Rate, PnL, Sharpe)
- Live Macro Daten (EURUSD, DXY, Volatility)
- Session Info
- Memory Statistics
- Debate Status
"""

import json
import os
import sys
from datetime import datetime, timezone
from pathlib import Path

from flask import Flask, jsonify
from flask_cors import CORS

# Parent-Directory zum Path hinzufügen
sys.path.insert(0, str(Path(__file__).parent.parent))

app = Flask(__name__)
CORS(app)

# Importiere unsere Module
try:
    from rdagent.components.coder.factor_coder.fx_config import get_fx_config
    from rdagent.components.coder.factor_coder.eurusd_macro import get_live_fx_data
    from rdagent.components.coder.factor_coder.eurusd_debate import get_current_session_info
    from rdagent.components.coder.factor_coder.eurusd_memory import EURUSDTradeMemory
    
    MODULES_AVAILABLE = True
except ImportError as e:
    print(f"⚠️  Module nicht verfügbar: {e}")
    MODULES_AVAILABLE = False


def parse_fin_quant_log(log_path: str, lines: int = 100) -> dict:
    """
    Parst die letzten N Zeilen der fin_quant.log.
    
    Extrahiert:
    - Aktueller Loop/Step
    - Letzter Faktor
    - Status (SUCCESS/FAILED/PENDING)
    """
    result = {
        "current_loop": "N/A",
        "current_step": "N/A",
        "progress_percent": 0,
        "last_factor": "N/A",
        "last_status": "N/A",
        "recent_factors": []
    }
    
    try:
        if not os.path.exists(log_path):
            return result
        
        with open(log_path, 'r', encoding='utf-8', errors='ignore') as f:
            # Hole letzte N Zeilen
            all_lines = f.readlines()
            recent_lines = all_lines[-lines:] if len(all_lines) > lines else all_lines
            log_content = ''.join(recent_lines)
        
        # Extrahiere Loop/Step
        import re
        
        # Workflow Progress
        progress_match = re.search(r'Workflow Progress:\s+(\d+)%.*loop_index=(\d+).*step_index=(\d+).*step_name=(\w+)', log_content)
        if progress_match:
            result["progress_percent"] = int(progress_match.group(1))
            result["current_loop"] = int(progress_match.group(2))
            result["current_step"] = f"{int(progress_match.group(3)) + 1}/4 ({progress_match.group(4)})"
        
        # Extrahiere Faktor-Namen
        factor_matches = re.findall(r'factor_name:\s*(\w+)', log_content)
        if factor_matches:
            result["last_factor"] = factor_matches[-1]
            result["recent_factors"] = list(reversed(factor_matches[-5:]))
        
        # Extrahiere Status
        if "This implementation is SUCCESS" in log_content:
            result["last_status"] = "SUCCESS"
        elif "This implementation is FAIL" in log_content:
            result["last_status"] = "FAILED"
        elif "Execution succeeded" in log_content:
            result["last_status"] = "RUNNING"
        
        # Log-Aktivität
        result["log_lines_total"] = len(all_lines)
        result["log_size_mb"] = round(os.path.getsize(log_path) / (1024 * 1024), 2)
        
    except Exception as e:
        result["error"] = str(e)
    
    return result


def get_memory_stats(memory_file: str) -> dict:
    """
    Holt Statistics aus dem Trade-Memory.
    """
    result = {
        "total_trades": 0,
        "win_rate": 0.0,
        "avg_return": 0.0,
        "total_pnl": 0.0
    }
    
    try:
        if not os.path.exists(memory_file):
            return result
        
        memory = EURUSDTradeMemory(memory_file)
        stats = memory.get_memory_stats()
        
        result["total_trades"] = stats.get("total_trades", 0)
        result["win_rate"] = round(stats.get("win_rate", 0) * 100, 1)
        result["avg_return"] = round(stats.get("avg_return", 0) * 100, 2)
        result["total_pnl"] = round(stats.get("total_pnl", 0) * 100, 2)
        result["sharpe_ratio"] = round(stats.get("sharpe_ratio", 0), 2)
        
    except Exception as e:
        result["error"] = str(e)
    
    return result


@app.route('/api/health', methods=['GET'])
def health_check():
    """Health Check Endpoint."""
    return jsonify({
        "status": "ok",
        "timestamp": datetime.now(timezone.utc).isoformat(),
        "modules_available": MODULES_AVAILABLE
    })


@app.route('/api/progress', methods=['GET'])
def get_progress():
    """
    Holt aktuellen Trading-Progress.
    
    Returns:
    - Aktueller Loop/Step
    - Fortschritts-Prozent
    - Letzter Faktor
    - Status
    """
    log_path = os.path.join(os.path.dirname(os.path.dirname(os.path.dirname(__file__))), 'fin_quant.log')
    progress = parse_fin_quant_log(log_path)
    
    return jsonify(progress)


@app.route('/api/macro', methods=['GET'])
def get_macro_data():
    """
    Holt Live-Macro-Daten.
    
    Returns:
    - EURUSD Preis
    - DXY (Dollar Index)
    - Realized Volatility
    - 24h Change
    """
    if not MODULES_AVAILABLE:
        return jsonify({"error": "Modules not available"}), 500
    
    live_data = get_live_fx_data()
    return jsonify(live_data)


@app.route('/api/session', methods=['GET'])
def get_session_data():
    """
    Holt aktuelle FX-Session Info.
    
    Returns:
    - Session Name
    - Hours
    - Characteristics
    - Recommended Strategy
    """
    if not MODULES_AVAILABLE:
        return jsonify({"error": "Modules not available"}), 500
    
    session = get_current_session_info()
    return jsonify(session)


@app.route('/api/memory', methods=['GET'])
def get_memory_data():
    """
    Holt Memory Statistics.
    
    Returns:
    - Total Trades
    - Win-Rate
    - Average Return
    - Sharpe Ratio
    """
    if not MODULES_AVAILABLE:
        return jsonify({"error": "Modules not available"}), 500
    
    config = get_fx_config()
    stats = get_memory_stats(config.memory_file)
    
    return jsonify(stats)


@app.route('/api/config', methods=['GET'])
def get_config_data():
    """
    Holt FX-Konfiguration.
    
    Returns:
    - Instrument
    - Target ARR
    - Max Drawdown
    - Spread
    """
    if not MODULES_AVAILABLE:
        return jsonify({"error": "Modules not available"}), 500
    
    config = get_fx_config()
    
    return jsonify({
        "instrument": config.instrument,
        "frequency": config.frequency,
        "target_arr": config.target_arr,
        "max_drawdown": config.max_drawdown,
        "spread_bps": config.spread_bps,
        "chat_model": config.chat_model
    })


@app.route('/api/dashboard', methods=['GET'])
def get_full_dashboard():
    """
    Holt alle Dashboard-Daten auf einmal.
    
    Kombiniert:
    - Progress
    - Macro
    - Session
    - Memory
    - Config
    """
    dashboard = {
        "timestamp": datetime.now(timezone.utc).isoformat(),
        "modules_available": MODULES_AVAILABLE
    }
    
    if MODULES_AVAILABLE:
        # Progress
        log_path = os.path.join(os.path.dirname(os.path.dirname(os.path.dirname(__file__))), 'fin_quant.log')
        dashboard["progress"] = parse_fin_quant_log(log_path)
        
        # Macro
        dashboard["macro"] = get_live_fx_data()
        
        # Session
        dashboard["session"] = get_current_session_info()
        
        # Memory
        config = get_fx_config()
        dashboard["memory"] = get_memory_stats(config.memory_file)
        
        # Config
        dashboard["config"] = {
            "instrument": config.instrument,
            "target_arr": config.target_arr,
            "max_drawdown": config.max_drawdown
        }
    else:
        dashboard["error"] = "Modules not available"
    
    return jsonify(dashboard)


@app.route('/', methods=['GET'])
def index():
    """Root Endpoint - zeigt API-Info."""
    return jsonify({
        "name": "NexQuant Dashboard API",
        "version": "1.0.0",
        "description": "COMPLETE Progress Visualisierung für EURUSD Trading-Agent",
        "endpoints": {
            "/api/health": "Health Check",
            "/api/progress": "Trading Progress",
            "/api/macro": "Live Macro Daten",
            "/api/session": "FX Session Info",
            "/api/memory": "Memory Statistics",
            "/api/config": "FX Konfiguration",
            "/api/dashboard": "Alle Daten kombiniert"
        }
    })


if __name__ == '__main__':
    print("="*60)
    print("NexQuant Dashboard API")
    print("="*60)
    print(f"Modules available: {MODULES_AVAILABLE}")
    print(f"Starting server on http://localhost:5000")
    print(f"API Docs: http://localhost:5000/")
    print("="*60)

    # Security fix: Disable debug mode in production
    # Debug mode allows arbitrary code execution via Werkzeug debugger
    # For development only: Set FLASK_DEBUG=1 environment variable
    import os
    debug_mode = os.getenv("FLASK_DEBUG", "0") == "1"
    
    if debug_mode:
        print("\n⚠️  WARNING: Running in DEBUG mode (development only!)")
        print("   Do NOT use in production - allows arbitrary code execution!\n")
    
    app.run(host='0.0.0.0', port=5000, debug=debug_mode)
