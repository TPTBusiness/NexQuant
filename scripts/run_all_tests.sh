#!/bin/bash
# Run all Predix integration tests
# Usage:
#   ./scripts/run_all_tests.sh           # Full test suite
#   ./scripts/run_all_tests.sh --quick   # Skip slow tests
#   ./scripts/run_all_tests.sh -v        # Verbose output
#   ./scripts/run_all_tests.sh --cov     # With coverage

set -e

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
PROJECT_ROOT="$(dirname "$SCRIPT_DIR")"

echo "========================================="
echo "Predix Integration Test Suite"
echo "========================================="
echo "Project: $PROJECT_ROOT"
echo "Date: $(date '+%Y-%m-%d %H:%M:%S')"
echo "Python: $(python3 --version 2>&1)"
echo "========================================="
echo ""

cd "$PROJECT_ROOT"

# Parse arguments
EXTRA_ARGS="$@"
if [[ "$EXTRA_ARGS" == *"--cov"* ]]; then
    echo "Running with coverage..."
    pytest test/integration/test_all_features.py -v --cov=rdagent --cov-report=html --cov-report=term-missing $EXTRA_ARGS
else
    echo "Running full test suite..."
    pytest test/integration/test_all_features.py -v --tb=short $EXTRA_ARGS
fi

EXIT_CODE=$?

echo ""
echo "========================================="
echo "Tests completed! (Exit code: $EXIT_CODE)"
echo "========================================="

if [[ "$EXTRA_ARGS" == *"--cov"* ]]; then
    echo ""
    echo "Coverage report generated at: htmlcov/index.html"
    echo "Open with: python -m http.server --directory htmlcov"
fi

exit $EXIT_CODE
