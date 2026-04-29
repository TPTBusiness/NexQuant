#!/bin/bash
# compress_old_logs.sh
# Komprimiert Log-Dateien, die älter als 3 Tage sind, mit zstd -19
# Ausführen via cron: 0 2 * * * /home/nico/Predix/scripts/compress_old_logs.sh

LOG_DIR="/home/nico/Predix"
LOG_DIRS=("$LOG_DIR/logs" "$LOG_DIR/log")

# Finde alle Log-Dateien (*.log, *.jsonl, *.pkl), die älter als 3 Tage sind und noch nicht komprimiert wurden
find "${LOG_DIRS[@]}" -type f \( -name "*.log" -o -name "*.jsonl" -o -name "*.pkl" \) -mtime +3 ! -name "*.zst" ! -name "*.gz" ! -name "*.xz" 2>/dev/null | while read -r file; do
    echo "Komprimiere: $file"
    zstd -19 -f "$file" -o "$file.zst" && rm -f "$file"
done

# Auch selector.log im Hauptverzeichnis behandeln, wenn es älter als 3 Tage ist
if [ -f "$LOG_DIR/selector.log" ]; then
    # Prüfe, ob Datei älter als 3 Tage ist
    if [ $(find "$LOG_DIR/selector.log" -mtime +3 | wc -l) -gt 0 ]; then
        echo "Komprimiere: $LOG_DIR/selector.log"
        zstd -19 -f "$LOG_DIR/selector.log" -o "$LOG_DIR/selector.log.zst" && rm -f "$LOG_DIR/selector.log"
    fi
fi

echo "Fertig mit Kompression der alten Logs."
