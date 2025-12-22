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
export PATH="$PWD:$PATH"

TEST_ENV_FILE=""
OVERRIDE_BASE_URL=""

while [[ $# -gt 0 ]]; do
    case "$1" in
        --test)
            TEST_ENV_FILE=".env.test"
            shift
            ;;
        --url)
            if [[ -z "${2:-}" ]]; then
                echo -e "${RED}Error: --url requires a value${NC}"
                exit 1
            fi
            OVERRIDE_BASE_URL="$2"
            shift 2
            ;;
        --url=*)
            OVERRIDE_BASE_URL="${1#*=}"
            shift
            ;;
        -h|--help|help)
            # Let the normal help handler deal with it later via TEST_FILE
            break
            ;;
        --*)
            echo -e "${RED}Error: Unknown option: $1${NC}"
            echo -e "${YELLOW}Try: ./run-k6.sh --help${NC}"
            exit 1
            ;;
        *)
            # First positional arg = test name/file
            break
            ;;
    esac
done

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

# Set final BASE_URL resolution (CLI > env > default)
if [ -n "$OVERRIDE_BASE_URL" ]; then
    BASE_URL="$OVERRIDE_BASE_URL"
else
    BASE_URL="${BASE_URL:-http://localhost:5000}"
fi

# Set default BASE_URL if not set
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

    # --- InfluxDB output support ---
    # Two modes supported:
    # 1) InfluxDB v1 (default k6 core): set INFLUXDB_URL and INFLUXDB_DB
    #    e.g. INFLUXDB_URL=http://localhost:8086 INFLUXDB_DB=myk6db
    #    The script will append: --out influxdb=http://localhost:8086/myk6db
    # 2) InfluxDB v2 using xk6-influxdb extension: set INFLUXDB_V2=true and provide
    #    K6_INFLUXDB_ADDR/K6_INFLUXDB_BUCKET/K6_INFLUXDB_TOKEN/K6_INFLUXDB_ORGANIZATION
    #    or INFLUXDB_URL/INFLUXDB_BUCKET/INFLUXDB_TOKEN/INFLUXDB_ORG. The script will
    #    set the K6_INFLUXDB_* env vars and append: -o xk6-influxdb=http://host:8086

    if [ -n "$INFLUXDB_URL" ] && [ -n "$INFLUXDB_DB" ] && [ -z "$INFLUXDB_V2" ]; then
        echo -e "${BLUE}Sending k6 metrics to InfluxDB v1 at ${INFLUXDB_URL}/${INFLUXDB_DB}${NC}"
        k6_cmd="$k6_cmd --out influxdb=${INFLUXDB_URL}/${INFLUXDB_DB}"
    elif [ "${INFLUXDB_V2:-}" = "true" ] || [ -n "${K6_INFLUXDB_TOKEN:-}" ] || [ -n "${INFLUXDB_TOKEN:-}" ]; then
        echo ${INFLUXDB_TOKEN}
        # Prefer explicit K6_ vars if present, otherwise fall back to INFLUXDB_* aliases
        export K6_INFLUXDB_ADDR=${K6_INFLUXDB_ADDR:-${INFLUXDB_URL:-http://localhost:8086}}
        export K6_INFLUXDB_BUCKET=${K6_INFLUXDB_BUCKET:-${INFLUXDB_BUCKET:-}}
        export K6_INFLUXDB_ORGANIZATION=${K6_INFLUXDB_ORGANIZATION:-${INFLUXDB_ORG:-}}
        export K6_INFLUXDB_TOKEN=${K6_INFLUXDB_TOKEN:-${INFLUXDB_TOKEN:-}}

        echo -e "${BLUE}Sending k6 metrics to InfluxDB v2 at ${K6_INFLUXDB_ADDR} using xk6-influxdb (ensure k6 was built with the extension)${NC}"

        # Preflight: verify we can write to the configured bucket (helps catch 403 permission errors)
        if [ -n "${K6_INFLUXDB_BUCKET:-}" ] && [ -n "${K6_INFLUXDB_ORGANIZATION:-}" ] && [ -n "${K6_INFLUXDB_TOKEN:-}" ]; then
            echo -e "${BLUE}Verifying InfluxDB write access to bucket '${K6_INFLUXDB_BUCKET}' (org: ${K6_INFLUXDB_ORGANIZATION})...${NC}"
            # Attempt a single lightweight line-protocol write (do not expose token in output)
            http_status=$(curl -s -o /dev/null -w "%{http_code}" -XPOST "${K6_INFLUXDB_ADDR}/api/v2/write?org=${K6_INFLUXDB_ORGANIZATION}&bucket=${K6_INFLUXDB_BUCKET}&precision=ns" -H "Authorization: Token ${K6_INFLUXDB_TOKEN}" --data-binary "k6_preflight value=1")

            if [ "${http_status}" != "204" ]; then
                echo -e "${RED}Error: InfluxDB write preflight failed (HTTP ${http_status}). This usually means the bucket does not exist or the token lacks WRITE permission for that bucket.${NC}"
                echo -e "${YELLOW}Suggestions:${NC}"
                echo "  - Ensure bucket '${K6_INFLUXDB_BUCKET}' exists in org '${K6_INFLUXDB_ORGANIZATION}'."
                echo "  - Ensure the token (K6_INFLUXDB_TOKEN / INFLUXDB_TOKEN) has WRITE permission scoped to that bucket."
                echo "  - Quick test (replace <TOKEN> if needed):"
                echo -e "    ${GREEN}curl -i -XPOST \"${K6_INFLUXDB_ADDR}/api/v2/write?org=${K6_INFLUXDB_ORGANIZATION}&bucket=${K6_INFLUXDB_BUCKET}&precision=ns\" -H \"Authorization: Token <TOKEN>\" --data-binary \"k6_check value=1\"${NC}"
                echo -e "${RED}Aborting test run to avoid generating lots of 403 errors.${NC}"
                return 1
            else
                echo -e "${GREEN}InfluxDB write preflight successful (HTTP 204). Continuing...${NC}"
            fi
        else
            echo -e "${YELLOW}Warning: InfluxDB v2 configured but bucket/org/token not fully set. Skipping preflight check.${NC}"
        fi

        # Use xk6-influxdb output identifier
        k6_cmd="$k6_cmd -o xk6-influxdb=${K6_INFLUXDB_ADDR}"
    else
        # No InfluxDB configuration detected; continue without outputs
        :
    fi

    k6_cmd="$k6_cmd \"$test_file\""

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
    echo "Usage: ./run-k6.sh [--test] [--url <BASE_URL>] [test-name]"
    echo ""
    echo "Options:"
    echo "  ${YELLOW}--test${NC}                 - Load .env.test instead of .env"
    echo "  ${YELLOW}--url <BASE_URL>${NC}       - Override BASE_URL from env files"
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
    echo "Examples:"
    echo "  ${GRAY}# Override URL without touching .env${NC}"
    echo "  ./run-k6.sh --url http://10.34.4.196 training"
    echo ""
    echo "  ${GRAY}# Use .env.test + override URL${NC}"
    echo "  ./run-k6.sh --test --url http://my-app:5000 training"
    echo ""
    echo "  ${GRAY}# Old style still works${NC}"
    echo "  ./run-k6.sh training"
    echo ""
}

# Map friendly names to test files
case "$TEST_FILE" in
    "help"|"-h"|"--help")
        show_help
        exit 0
        ;;
    "training"|"train")
        run_test "train.js" "Autoscaler Training"
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
