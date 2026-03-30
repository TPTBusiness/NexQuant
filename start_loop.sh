#!/bin/bash
# Start EURUSD Trading in Endlosschleife mit Auto-Restart
# Verwendung: ./start_loop.sh

set -e

LOG_FILE=~/Predix/fin_quant.log

echo "============================================================"
echo "  Predix EURUSD Trading - Endlosschleife mit Auto-Restart"
echo "============================================================"
echo ""
echo "Log-Datei: $LOG_FILE"
echo ""
echo "Dashboard Optionen:"
echo "  Web:  rdagent fin_quant --with-dashboard"
echo "  CLI:  rdagent fin_quant --cli-dashboard"
echo ""
echo "Stoppen mit: Ctrl+C"
echo "============================================================"
echo ""

# Conda Environment aktivieren
if [ -f ~/miniconda3/etc/profile.d/conda.sh ]; then
    source ~/miniconda3/etc/profile.d/conda.sh
    conda activate rdagent
    echo "✓ Conda Environment 'rdagent' aktiviert"
else
    echo "⚠️  Conda nicht gefunden..."
fi

echo ""
echo "Starte Endlosschleife..."
echo ""

cd /home/nico/Predix

while true; do
    echo "=== START: $(date) ===" >> $LOG_FILE
    echo ""
    echo "╔════════════════════════════════════════════════════════╗"
    echo "║  🚀 START: $(date +"%Y-%m-%d %H:%M:%S")                      ║"
    echo "╚════════════════════════════════════════════════════════╝"
    echo ""
    
    dotenv run -- rdagent fin_quant 2>&1 | tee -a $LOG_FILE
    
    echo ""
    echo "╔════════════════════════════════════════════════════════╗"
    echo "║  ⏸️  RESTART: $(date +"%Y-%m-%d %H:%M:%S")                    ║"
    echo "╚════════════════════════════════════════════════════════╝"
    echo "=== RESTART: $(date) ===" >> $LOG_FILE
    echo ""
    echo "Warte 5 Sekunden vor Neustart..."
    echo ""
    sleep 5
done
