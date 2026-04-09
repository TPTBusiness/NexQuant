#!/bin/bash
# ============================================================================
# PREDIX Strategy Generator Watchdog
# Checks every 20min: is the generator running? If not, (re)start it.
# ============================================================================

SCRIPT_DIR="/home/nico/Predix"
GENERATOR="python ${SCRIPT_DIR}/predix_smart_strategy_gen.py"
TARGET_COUNT=3
LOGFILE="${SCRIPT_DIR}/results/logs/watchdog.log"
LOCKFILE="/tmp/predix_generator.lock"
MAX_ATTEMPTS=50  # Stop after this many attempts
PIDFILE="/tmp/predix_generator_attempt.pid"

mkdir -p "${SCRIPT_DIR}/results/logs"

log() {
    echo "$(date '+%Y-%m-%d %H:%M:%S') - $1" >> "$LOGFILE"
}

# Get current attempt count
get_attempt_count() {
    if [ -f "$PIDFILE" ]; then
        cat "$PIDFILE"
    else
        echo "0"
    fi
}

# Increment attempt count
increment_attempt() {
    local current=$(get_attempt_count)
    local next=$((current + 1))
    echo "$next" > "$PIDFILE"
    echo "$next"
}

# Check if generator is actually making progress
check_progress() {
    local latest_log=$(ls -t ${SCRIPT_DIR}/results/logs/smart_strategy_gen_*.log 2>/dev/null | head -1)
    if [ -n "$latest_log" ]; then
        # Check if log was updated in last 5 minutes
        local age=$(( $(date +%s) - $(stat -c %Y "$latest_log" 2>/dev/null || echo 0) ))
        if [ $age -gt 300 ]; then
            return 1  # Stale
        fi
        return 0  # Fresh
    fi
    return 1  # No log file
}

# Kill any existing generator processes
cleanup() {
    pkill -9 -f "predix_smart_strategy_gen.py" 2>/dev/null
    rm -f "$LOCKFILE"
    log "Cleaned up old processes"
}

# Check if we've hit max attempts
if [ "$(get_attempt_count)" -ge "$MAX_ATTEMPTS" ]; then
    log "MAX ATTEMPTS ($MAX_ATTEMPTS) reached. Stopping watchdog."
    exit 0
fi

# Check if generator is running
if pgrep -f "predix_smart_strategy_gen.py" > /dev/null 2>&1; then
    # Check if it's making progress
    if check_progress; then
        log "Generator is running and making progress. Exiting."
        exit 0
    else
        log "Generator is running but appears stalled. Restarting..."
        cleanup
    fi
else
    log "Generator is NOT running. Starting..."
    cleanup
fi

# Increment attempt counter
ATTEMPT=$(increment_attempt)
log "=== Attempt $ATTEMPT / $MAX_ATTEMPTS ==="

# Create lock file
echo $$ > "$LOCKFILE"

# Start generator in background, capture PID
cd "$SCRIPT_DIR"
nohup $GENERATOR $TARGET_COUNT > /dev/null 2>&1 &
GEN_PID=$!

log "Started generator with PID $GEN_PID"

# Wait for process to finish (up to 20 min)
WAIT=0
while kill -0 $GEN_PID 2>/dev/null; do
    sleep 10
    WAIT=$((WAIT + 10))
    if [ $WAIT -ge 1200 ]; then  # 20 min timeout
        log "Generator timed out after 20 min. Killing."
        kill -9 $GEN_PID 2>/dev/null
        break
    fi
done

# Cleanup lock
rm -f "$LOCKFILE"

log "Generator finished (or was killed). Exit code: $?"
