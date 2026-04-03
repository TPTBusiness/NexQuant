#!/bin/bash
# Bandit Security Scanner Wrapper for Pre-Commit
# This script runs Bandit with the correct configuration
# Usage: .pre-commit-hooks/run_bandit.sh [files...]

set -e

BANDIT_CONFIG=".bandit.yml"
SCAN_DIR="rdagent/"
EXCLUDE_DIRS="test/,.git/,.qwen/,results/,git_ignore_folder/"
EXCLUDE_FILES="rdagent/scenarios/qlib/proposal/bandit.py"

echo "🔒 Running Bandit Security Scanner..."
echo "   Config: ${BANDIT_CONFIG}"
echo "   Scan: ${SCAN_DIR}"
echo ""

# Run bandit with high severity threshold
# Exit code 1 if any HIGH severity issues found
bandit \
  --configfile "${BANDIT_CONFIG}" \
  --severity-level high \
  --confidence-level medium \
  --format txt \
  --recursive "${SCAN_DIR}" \
  --exclude "${EXCLUDE_DIRS},${EXCLUDE_FILES}" \
  "$@"

exit_code=$?

if [ $exit_code -eq 0 ]; then
    echo "✅ No HIGH severity security issues found"
else
    echo "⚠️  HIGH severity security issues detected!"
    echo "   Review issues above and fix before committing."
    echo "   To suppress false positives, add # nosec BXXX to the line."
fi

exit $exit_code
