#!/bin/bash
# ============================================================================
# PREDIX Strategy Generator - Robust Loop
# Restarts automatically on crash, generates strategies continuously.
# ============================================================================

SCRIPT_DIR="/home/nico/Predix"
GENERATOR="python ${SCRIPT_DIR}/predix_smart_strategy_gen.py"
TARGET_COUNT=3
LOGFILE="${SCRIPT_DIR}/results/logs/generator_loop.log"
PIDFILE="/tmp/predix_loop.pid"

echo $$ > "$PIDFILE"
mkdir -p "${SCRIPT_DIR}/results/logs"

log() {
    echo "$(date '+%Y-%m-%d %H:%M:%S') - $1" | tee -a "$LOGFILE"
}

cleanup() {
    log "Received termination signal. Cleaning up..."
    pkill -f "predix_smart_strategy_gen.py" 2>/dev/null
    rm -f "$PIDFILE"
    log "Cleanup complete. Exiting."
    exit 0
}

trap cleanup SIGTERM SIGINT

log "========================================="
log "🚀 PREDIX Generator Loop Starting"
log "========================================="
log "Target: ${TARGET_COUNT} strategies per run"
log "Log: ${LOGFILE}"

ATTEMPT=0

while true; do
    ATTEMPT=$((ATTEMPT + 1))
    log ""
    log "=== Attempt #${ATTEMPT} ==================================="
    
    # Check disk space
    DISK_USAGE=$(df -h ${SCRIPT_DIR} | tail -1 | awk '{print $5}' | sed 's/%//')
    if [ "$DISK_USAGE" -gt 90 ]; then
        log "⚠️  Disk usage at ${DISK_USAGE}%. Pausing..."
        sleep 300
        continue
    fi
    
    # Check if we already have enough strategies
    STRAT_COUNT=$(ls ${SCRIPT_DIR}/results/strategies_new/*.json 2>/dev/null | wc -l)
    log "📁 Existing strategies: ${STRAT_COUNT}"
    
    # Kill any stale processes
    pkill -9 -f "predix_smart_strategy_gen.py" 2>/dev/null
    sleep 2
    
    # Start generator
    log "🤖 Starting generator..."
    cd "$SCRIPT_DIR"
    nohup $GENERATOR $TARGET_COUNT > /dev/null 2>&1 &
    GEN_PID=$!
    log "   PID: ${GEN_PID}"
    
    # Monitor progress
    ELAPSED=0
    MAX_WAIT=1800  # 30 minutes max per run
    
    while kill -0 $GEN_PID 2>/dev/null; do
        sleep 30
        ELAPSED=$((ELAPSED + 30))
        
        # Check latest log for progress
        LATEST_LOG=$(ls -t ${SCRIPT_DIR}/results/logs/smart_strategy_gen_*.log 2>/dev/null | head -1)
        if [ -n "$LATEST_LOG" ]; then
            LAST_LINE=$(tail -1 "$LATEST_LOG" 2>/dev/null)
            if [ $((ELAPSED % 120)) -eq 0 ]; then  # Every 2 min
                log "   ⏱️  ${ELAPSED}s elapsed - ${LAST_LINE:0:80}"
            fi
        fi
        
        # Timeout check
        if [ $ELAPSED -ge $MAX_WAIT ]; then
            log "   ⏰ Timeout after ${ELAPSED}s. Killing..."
            kill -9 $GEN_PID 2>/dev/null
            break
        fi
    done
    
    # Check results
    wait $GEN_PID 2>/dev/null
    EXIT_CODE=$?
    
    if [ $EXIT_CODE -eq 0 ]; then
        log "✅ Generator completed successfully"
    elif [ $EXIT_CODE -eq 137 ]; then
        log "❌ Generator killed (OOM? Exit 137)"
    else
        log "⚠️  Generator exited with code ${EXIT_CODE}"
    fi
    
    # Count new strategies
    NEW_STRATS=$(ls -t ${SCRIPT_DIR}/results/strategies_new/*.json 2>/dev/null | head -3)
    if [ -n "$NEW_STRATS" ]; then
        log "📊 Latest strategies:"
        echo "$NEW_STRATS" | while read f; do
            [ -f "$f" ] && log "   - $(basename $f)"
        done
    fi
    
    # Wait before next attempt
    log "⏳ Waiting 60s before next attempt..."
    sleep 60
done
