#!/usr/bin/env bash
set -euo pipefail

# Load environment file
if [ -f .env.test ]; then
  # shellcheck disable=SC1091
  source .env.test
else
  echo "ERROR: .env.test not found in $(pwd). Create it or run this script from the test/ directory." >&2
  exit 1
fi

# Number of repeats (default 10)
REPEATS=${REPEATS:-3}

OUTDIR="logs/step_${TAGS}_${MIN_SCALE_DOWN_ATTEMPTS}_${MAX_SCALE_DOWN_STEPS}"
mkdir -p "$OUTDIR"

for i in $(seq 1 "$REPEATS"); do
  echo "=== Run $i of $REPEATS starting at $(date) ==="

  LOGFILE="$OUTDIR/log_test_run_${MAX_REPLICAS}_${i}.log"

  if command -v unbuffer >/dev/null 2>&1; then
    unbuffer ./run-k6.sh --test train 2>&1 | tee "$LOGFILE"
  elif command -v stdbuf >/dev/null 2>&1; then
    stdbuf -oL ./run-k6.sh --test train 2>&1 | tee "$LOGFILE"
  else
    ./run-k6.sh --test train 2>&1 | tee "$LOGFILE"
  fi

  if [ "$i" -lt "$REPEATS" ]; then
    sleep 1100
  fi
done