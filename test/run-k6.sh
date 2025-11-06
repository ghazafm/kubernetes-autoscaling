#!/bin/bash

if [ -f .env ]; then
    export $(cat .env | grep -v '^#' | xargs)
fi

TEST_FILE=${1:-k6.js}

# Map friendly names to test files
case "$TEST_FILE" in
    "general")
        TEST_FILE="k6.js"
        ;;
    "cpu")
        TEST_FILE="k6-cpu-stress.js"
        ;;
    "memory")
        TEST_FILE="k6-memory-stress.js"
        ;;
    "spike")
        TEST_FILE="k6-spike.js"
        ;;
    "rl"|"autoscaler")
        TEST_FILE="k6-rl-autoscaler.js"
        ;;
esac

echo "Running k6 test: $TEST_FILE"
echo "BASE_URL: $BASE_URL"
echo ""

k6 run -e BASE_URL="$BASE_URL" "$TEST_FILE"
