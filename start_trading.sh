#!/bin/bash
# Start EURUSD Trading mit automatischem Dashboard
# Verwendung: ./start_trading.sh

set -e

echo "============================================================"
echo "  Predix EURUSD Trading - Start mit Dashboard"
echo "============================================================"
echo ""

# Conda Environment aktivieren
if [ -f ~/miniconda3/etc/profile.d/conda.sh ]; then
    source ~/miniconda3/etc/profile.d/conda.sh
    conda activate rdagent
    echo "✓ Conda Environment 'rdagent' aktiviert"
else
    echo "⚠️  Conda nicht gefunden, versuche mit system Python..."
fi

# Dashboard API im Hintergrund starten
echo ""
echo "🚀 Starte Dashboard API..."
cd /home/nico/Predix
nohup python web/dashboard_api.py > /tmp/dashboard.log 2>&1 &
DASHBOARD_PID=$!
echo "✓ Dashboard API gestartet (PID: $DASHBOARD_PID)"
echo ""
echo "📊 Dashboard URL: http://localhost:5000/dashboard.html"
echo "   Dashboard Log:  /tmp/dashboard.log"
echo ""

# Cleanup Funktion
cleanup() {
    echo ""
    echo "⏹️  Stoppe Dashboard (PID: $DASHBOARD_PID)..."
    kill $DASHBOARD_PID 2>/dev/null || true
    echo "✓ Gestoppt"
    exit 0
}

# Trap für Ctrl+C
trap cleanup SIGINT SIGTERM

# RD-Agent fin_quant starten
echo "🔄 Starte EURUSD Trading-Agent..."
echo ""
dotenv run -- rdagent fin_quant

# Cleanup wenn fertig
cleanup
