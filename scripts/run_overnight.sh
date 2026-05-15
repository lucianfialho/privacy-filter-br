#!/usr/bin/env bash
# Run the dataset generation overnight via Haiku API + 20 parallel workers.
#
# Usage:
#   export ANTHROPIC_API_KEY=sk-ant-...
#   ./scripts/run_overnight.sh 100000   # target 100k examples
#
# Defaults: 100k examples, 20 workers, append to data/dataset_br_v2.jsonl

set -euo pipefail

TARGET="${1:-100000}"
WORKERS="${2:-20}"
OUTPUT="${OUTPUT:-data/dataset_br_v2.jsonl}"

if [[ -z "${ANTHROPIC_API_KEY:-}" ]]; then
    echo "ERROR: ANTHROPIC_API_KEY not set"
    exit 1
fi

cd "$(dirname "$0")/.."

# Activate venv if exists
if [[ -d venv ]]; then
    source venv/bin/activate
fi

mkdir -p data

LOG="data/run_$(date +%Y%m%d_%H%M%S).log"
echo "Target: $TARGET examples"
echo "Workers: $WORKERS"
echo "Output: $OUTPUT"
echo "Log: $LOG"
echo ""
echo "Starting nohup background process..."

nohup python3 scripts/generate_dataset.py \
    --n "$TARGET" \
    --output "$OUTPUT" \
    --workers "$WORKERS" \
    > "$LOG" 2>&1 < /dev/null &

PID=$!
echo "Started PID $PID"
echo ""
echo "Watch progress:"
echo "  tail -f $LOG"
echo ""
echo "Check stats:"
echo "  wc -l $OUTPUT data/$(basename "$OUTPUT" .jsonl)_holdout.jsonl"
echo ""
echo "Kill if needed:"
echo "  kill $PID"
