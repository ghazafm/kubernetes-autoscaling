#!/bin/bash

# RL Autoscaler K6 Test Runner
# Usage: ./run-k6.sh [test-name|all]

set -e

# Colors for output
RED='\033[0;31m'
GREEN='\033[0;32m'
YELLOW='\033[1;33m'
BLUE='\033[0;34m'
NC='\033[0m' # No Color

# Optional: support a test env file when caller passes --test as first arg
# Usage: ./run-k6.sh --test training
TEST_ENV_FILE=""
if [ "$1" = "--test" ]; then
    TEST_ENV_FILE=".env.test"
    # shift so the next positional argument is the test name
    shift
fi

# Load environment variables (test env has priority when --test used)
if [ -n "$TEST_ENV_FILE" ]; then
    if [ -f "$TEST_ENV_FILE" ]; then
        # shellcheck disable=SC2046
        export $(grep -v '^#' "$TEST_ENV_FILE" | xargs)
        echo -e "${GREEN}Loaded test env from ${TEST_ENV_FILE}${NC}"
    else
        echo -e "${RED}Error: Test env file ${TEST_ENV_FILE} not found!${NC}"
        exit 1
    fi
else
    if [ -f .env ]; then
        # shellcheck disable=SC2046
        export $(grep -v '^#' .env | xargs)
    else
        echo -e "${YELLOW}Warning: .env file not found. Using defaults.${NC}"
    fi
fi

# Set default BASE_URL if not set
BASE_URL=${BASE_URL:-http://localhost:5000}
DURATION_MULTIPLIER=${DURATION_MULTIPLIER:-}
CYCLE_COUNT=${CYCLE_COUNT:-}
MAX_REPLICAS=${MAX_REPLICAS:-50}
MIN_REPLICAS=${MIN_REPLICAS:-1}
REQUESTS_PER_POD=${REQUESTS_PER_POD:-8}

TEST_FILE=${1:-k6.js}

# Function to run a test
run_test() {
    local test_file=$1
    local test_name=$2

    echo -e "\n${BLUE}═══════════════════════════════════════════════════════${NC}"
    echo -e "${GREEN}Running: ${test_name}${NC}"
    echo -e "${BLUE}File: ${test_file}${NC}"
    echo -e "${BLUE}BASE_URL: ${BASE_URL}${NC}"

    # Show duration config if set
    if [ -n "$DURATION_MULTIPLIER" ]; then
        echo -e "${BLUE}DURATION_MULTIPLIER: ${DURATION_MULTIPLIER}x${NC}"
    fi
    if [ -n "$CYCLE_COUNT" ]; then
        echo -e "${BLUE}CYCLE_COUNT: ${CYCLE_COUNT}${NC}"
    fi

    echo -e "${BLUE}═══════════════════════════════════════════════════════${NC}\n"

    if [ ! -f "$test_file" ]; then
        echo -e "${RED}Error: Test file '$test_file' not found!${NC}"
        return 1
    fi

    # Helper: find free k6 REST API port starting from a base (default 6565)
    find_free_k6_port() {
        local start_port=${1:-6565}
        local port=$start_port
        # avoid infinite loop: limit search to +100 ports
        local max_port=$((start_port + 100))
        while lsof -nP -iTCP:${port} -sTCP:LISTEN >/dev/null 2>&1; do
            port=$((port + 1))
            if [ "$port" -gt "$max_port" ]; then
                # give up and return start_port
                break
            fi
        done
        echo $port
    }

    # Build k6 command with optional env vars. We expose the k6 REST API on a free port
    # so multiple parallel runs can coexist. Starting port is taken from K6_API_PORT (env)
    # or defaults to 6565.
    local start_api_port=${K6_API_PORT:-6565}
    local chosen_port
    chosen_port=$(find_free_k6_port "$start_api_port")
    local k6_address="localhost:${chosen_port}"
    if [ "$chosen_port" != "$start_api_port" ]; then
        echo -e "${YELLOW}Note: port ${start_api_port} in use, using k6 REST API address ${k6_address}${NC}"
    else
        echo -e "${BLUE}k6 REST API address: ${k6_address}${NC}"
    fi

    # place the --address flag before the subcommand 'run' so it's applied to k6 itself
    local k6_cmd="k6 --address \"${k6_address}\" run --env BASE_URL=\"$BASE_URL\""

    if [ -n "$DURATION_MULTIPLIER" ]; then
        k6_cmd="$k6_cmd --env DURATION_MULTIPLIER=\"$DURATION_MULTIPLIER\""
    fi

    if [ -n "$CYCLE_COUNT" ]; then
        k6_cmd="$k6_cmd --env CYCLE_COUNT=\"$CYCLE_COUNT\""
    fi

    # Pass replica configuration for dynamic load scaling
    k6_cmd="$k6_cmd --env MAX_REPLICAS=\"$MAX_REPLICAS\""
    k6_cmd="$k6_cmd --env MIN_REPLICAS=\"$MIN_REPLICAS\""
    k6_cmd="$k6_cmd --env REQUESTS_PER_POD=\"$REQUESTS_PER_POD\""

    k6_cmd="$k6_cmd \"$test_file\""

    # Execute command
    eval $k6_cmd

    local exit_code=$?
    if [ $exit_code -eq 0 ]; then
        echo -e "\n${GREEN}✓ Test completed successfully: ${test_name}${NC}\n"
    else
        echo -e "\n${RED}✗ Test failed: ${test_name} (exit code: $exit_code)${NC}\n"
    fi

    return $exit_code
}

# Function to show help
show_help() {
    echo -e "${BLUE}RL Autoscaler K6 Test Runner${NC}"
    echo -e "${BLUE}═══════════════════════════════════════════════════════${NC}\n"
    echo "Usage: ./run-k6.sh [test-name]"
    echo ""
    echo "Available test names:"
    echo "  ${GREEN}training${NC}       - Comprehensive daily traffic patterns (60 min)"
    echo "  ${GREEN}edge${NC}           - Edge cases and stress scenarios (40 min)"
    echo "  ${GREEN}weekly${NC}         - Weekly simulation patterns (50 min)"
    echo "  ${GREEN}rl${NC}             - Original RL autoscaler test"
    echo "  ${GREEN}cpu${NC}            - CPU stress test (8 min)"
    echo "  ${GREEN}memory${NC}         - Memory stress test (8 min)"
    echo "  ${GREEN}spike${NC}          - Quick spike test (3 min)"
    echo "  ${GREEN}general${NC}        - General test (default)"
    echo ""
    echo "Special commands:"
    echo "  ${YELLOW}all${NC}            - Run all autoscaler training tests sequentially"
    echo "  ${YELLOW}full${NC}           - Run complete test suite (all tests)"
    echo "  ${YELLOW}quick${NC}          - Run quick validation (spike + cpu + memory)"
    echo "  ${YELLOW}help${NC}           - Show this help message"
    echo ""
    echo "Environment Variables:"
    echo "  ${YELLOW}BASE_URL${NC}              - Target URL (default: http://localhost:5000)"
    echo "  ${YELLOW}DURATION_MULTIPLIER${NC}   - Time scale multiplier (1=default, 24=1day, 168=1week)"
    echo "  ${YELLOW}CYCLE_COUNT${NC}           - Number of pattern repetitions (1=once, 7=weekly)"
    echo ""
    echo "Examples:"
    echo "  ${GRAY}# Run default 60-minute training${NC}"
    echo "  ./run-k6.sh training"
    echo ""
    echo "  ${GRAY}# Run all tests${NC}"
    echo "  ./run-k6.sh all"
    echo ""
    echo "  ${GRAY}# Quick validation test${NC}"
    echo "  ./run-k6.sh quick"
    echo ""
    echo "  ${GRAY}# Change target URL${NC}"
    echo "  BASE_URL=http://my-app:5000 ./run-k6.sh training"
    echo ""
    echo "  ${GRAY}# Run 1-day training (24x duration)${NC}"
    echo "  DURATION_MULTIPLIER=24 ./run-k6.sh training"
    echo ""
    echo "  ${GRAY}# Run 1-week training (24x duration, 7 cycles)${NC}"
    echo "  DURATION_MULTIPLIER=24 CYCLE_COUNT=7 ./run-k6.sh training"
    echo ""
    echo "  ${GRAY}# Run 1-week edge cases in background${NC}"
    echo "  nohup env DURATION_MULTIPLIER=24 CYCLE_COUNT=7 ./run-k6.sh edge > training.log 2>&1 &"
    echo ""
    echo "For more details on extended durations, see: ${YELLOW}EXTENDED_DURATION_GUIDE.md${NC}"
    echo ""
}

# Map friendly names to test files
case "$TEST_FILE" in
    "help"|"-h"|"--help")
        show_help
        exit 0
        ;;
    "training"|"train")
        run_test "k6-autoscaler-training.js" "Autoscaler Training (Comprehensive)"
        ;;
    "edge"|"edge-cases")
        run_test "k6-autoscaler-edge-cases.js" "Edge Cases & Stress Testing"
        ;;
    "weekly"|"week")
        run_test "k6-autoscaler-weekly.js" "Weekly Simulation"
        ;;
    "general")
        run_test "k6.js" "General Test"
        ;;
    "cpu")
        run_test "k6-cpu-stress.js" "CPU Stress Test"
        ;;
    "memory"|"mem")
        run_test "k6-memory-stress.js" "Memory Stress Test"
        ;;
    "spike")
        run_test "k6-spike.js" "Spike Test"
        ;;
    "rl"|"autoscaler")
        run_test "k6-rl-autoscaler.js" "RL Autoscaler Test"
        ;;
    "all")
        echo -e "${YELLOW}Running ALL autoscaler training tests...${NC}"

        # Calculate estimated duration
        base_duration=150  # ~2.5 hours in minutes (spike + training + edge + weekly)
        multiplier=${DURATION_MULTIPLIER:-1}
        cycles=${CYCLE_COUNT:-1}
        total_minutes=$(echo "$base_duration * $multiplier * $cycles" | bc)
        total_hours=$(echo "scale=1; $total_minutes / 60" | bc)

        if (( $(echo "$total_hours < 1" | bc -l) )); then
            echo -e "${YELLOW}This will take approximately ${total_minutes} minutes${NC}\n"
        elif (( $(echo "$total_hours < 24" | bc -l) )); then
            echo -e "${YELLOW}This will take approximately ${total_hours} hours${NC}\n"
        else
            local total_days=$(echo "scale=1; $total_hours / 24" | bc)
            echo -e "${YELLOW}This will take approximately ${total_days} days${NC}\n"
        fi

        failed_tests=()

        # Run all training tests
        run_test "k6-spike.js" "Quick Validation" || failed_tests+=("spike")
        sleep 10

        run_test "k6-autoscaler-training.js" "Comprehensive Training" || failed_tests+=("training")
        sleep 10

        run_test "k6-autoscaler-edge-cases.js" "Edge Cases" || failed_tests+=("edge-cases")
        sleep 10

        run_test "k6-autoscaler-weekly.js" "Weekly Patterns" || failed_tests+=("weekly")
        sleep 10

        run_test "k6-cpu-stress.js" "CPU Stress" || failed_tests+=("cpu")
        sleep 10

        run_test "k6-memory-stress.js" "Memory Stress" || failed_tests+=("memory")

        # Summary
        echo -e "\n${BLUE}═══════════════════════════════════════════════════════${NC}"
        echo -e "${BLUE}Test Suite Summary${NC}"
        echo -e "${BLUE}═══════════════════════════════════════════════════════${NC}"

        if [ ${#failed_tests[@]} -eq 0 ]; then
            echo -e "${GREEN}✓ All tests passed successfully!${NC}"
            exit 0
        else
            echo -e "${RED}✗ Some tests failed:${NC}"
            for test in "${failed_tests[@]}"; do
                echo -e "${RED}  - $test${NC}"
            done
            exit 1
        fi
        ;;
    "full")
        echo -e "${YELLOW}Running FULL test suite...${NC}"

        # Calculate estimated duration
        base_duration=180  # ~3 hours in minutes (all tests including general)
        multiplier=${DURATION_MULTIPLIER:-1}
        cycles=${CYCLE_COUNT:-1}
        total_minutes=$(echo "$base_duration * $multiplier * $cycles" | bc)
        total_hours=$(echo "scale=1; $total_minutes / 60" | bc)

        if (( $(echo "$total_hours < 1" | bc -l) )); then
            echo -e "${YELLOW}This will take approximately ${total_minutes} minutes${NC}\n"
        elif (( $(echo "$total_hours < 24" | bc -l) )); then
            echo -e "${YELLOW}This will take approximately ${total_hours} hours${NC}\n"
        else
            total_days=$(echo "scale=1; $total_hours / 24" | bc)
            echo -e "${YELLOW}This will take approximately ${total_days} days${NC}\n"
        fi

        failed_tests=()

        # Run all tests including general ones
        run_test "k6.js" "General Test" || failed_tests+=("general")
        sleep 10

        run_test "k6-spike.js" "Spike Test" || failed_tests+=("spike")
        sleep 10

        run_test "k6-rl-autoscaler.js" "RL Autoscaler" || failed_tests+=("rl")
        sleep 10

        run_test "k6-autoscaler-training.js" "Training" || failed_tests+=("training")
        sleep 10

        run_test "k6-autoscaler-edge-cases.js" "Edge Cases" || failed_tests+=("edge-cases")
        sleep 10

        run_test "k6-autoscaler-weekly.js" "Weekly" || failed_tests+=("weekly")
        sleep 10

        run_test "k6-cpu-stress.js" "CPU Stress" || failed_tests+=("cpu")
        sleep 10

        run_test "k6-memory-stress.js" "Memory Stress" || failed_tests+=("memory")

        # Summary
        echo -e "\n${BLUE}═══════════════════════════════════════════════════════${NC}"
        echo -e "${BLUE}Full Test Suite Summary${NC}"
        echo -e "${BLUE}═══════════════════════════════════════════════════════${NC}"

        if [ ${#failed_tests[@]} -eq 0 ]; then
            echo -e "${GREEN}✓ All tests passed successfully!${NC}"
            exit 0
        else
            echo -e "${RED}✗ Some tests failed:${NC}"
            for test in "${failed_tests[@]}"; do
                echo -e "${RED}  - $test${NC}"
            done
            exit 1
        fi
        ;;
    "quick")
        echo -e "${YELLOW}Running quick validation tests...${NC}"
        echo -e "${YELLOW}This will take approximately 15 minutes${NC}\n"

        failed_tests=()

        run_test "k6-spike.js" "Spike Test" || failed_tests+=("spike")
        sleep 5

        run_test "k6-cpu-stress.js" "CPU Stress" || failed_tests+=("cpu")
        sleep 5

        run_test "k6-memory-stress.js" "Memory Stress" || failed_tests+=("memory")

        # Summary
        echo -e "\n${BLUE}═══════════════════════════════════════════════════════${NC}"
        echo -e "${BLUE}Quick Validation Summary${NC}"
        echo -e "${BLUE}═══════════════════════════════════════════════════════${NC}"

        if [ ${#failed_tests[@]} -eq 0 ]; then
            echo -e "${GREEN}✓ All quick tests passed!${NC}"
            exit 0
        else
            echo -e "${RED}✗ Some tests failed:${NC}"
            for test in "${failed_tests[@]}"; do
                echo -e "${RED}  - $test${NC}"
            done
            exit 1
        fi
        ;;
    *)
        # Treat as direct filename
        if [ -f "$TEST_FILE" ]; then
            run_test "$TEST_FILE" "Custom Test"
        else
            echo -e "${RED}Error: Unknown test name or file not found: $TEST_FILE${NC}"
            echo ""
            show_help
            exit 1
        fi
        ;;
esac
