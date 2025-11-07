#!/usr/bin/env bash
# Response Time Testing by Replica Count (Improved)
# Usage:
#   ./test-response-time-by-replicas.sh [--phases "1,25,50"] [--rollout-timeout 900s] [--cooldown 60]
# Env (optional via .env in script dir):
#   NAMESPACE, DEPLOYMENT_NAME, BASE_URL, TEST_DURATION, TEST_VUS, COOLDOWN_TIME, ROLL_TIMEOUT

set -euo pipefail

# ── Colors ─────────────────────────────────────────────────────────────────────
RED='\033[0;31m'; GREEN='\033[0;32m'; YELLOW='\033[1;33m'; BLUE='\033[0;34m'; CYAN='\033[0;36m'; NC='\033[0m'

# ── Load .env (same dir as script) ─────────────────────────────────────────────
if [ -f "$(dirname "$0")/.env" ]; then
  # shellcheck disable=SC2046
  export $(grep -v '^#' "$(dirname "$0")/.env" | xargs)
fi

# ── Defaults ───────────────────────────────────────────────────────────────────
NAMESPACE="${NAMESPACE:-default}"
DEPLOYMENT_NAME="${DEPLOYMENT_NAME:-flask-app}"
BASE_URL="${BASE_URL:-http://localhost:8000}"

TEST_DURATION="${TEST_DURATION:-3m}"
TEST_VUS="${TEST_VUS:-20}"
COOLDOWN_TIME="${COOLDOWN_TIME:-60}"
ROLL_TIMEOUT="${ROLL_TIMEOUT:-600s}"

# default phases
PHASES_DEFAULT="1,25,50"

# ── Args ───────────────────────────────────────────────────────────────────────
PHASES="$PHASES_DEFAULT"
while [[ $# -gt 0 ]]; do
  case "$1" in
    --phases)
      PHASES="$2"; shift 2;;
    --rollout-timeout)
      ROLL_TIMEOUT="$2"; shift 2;;
    --cooldown)
      COOLDOWN_TIME="$2"; shift 2;;
    -h|--help)
      echo "Usage: $0 [--phases \"1,25,50\"] [--rollout-timeout 900s] [--cooldown 60]"
      exit 0;;
    *)
      echo "Unknown arg: $1"; exit 1;;
  esac
done

# ── Output files ───────────────────────────────────────────────────────────────
TIMESTAMP=$(date +%Y%m%d_%H%M%S)
OUTPUT_FILE="response_time_test_${TIMESTAMP}.txt"
CSV_FILE="response_time_test_${TIMESTAMP}.csv"
echo "Replicas,Test_Phase,Avg_RT(ms),p50(ms),p90(ms),p95(ms),p99(ms),Max_RT(ms),Error_Rate(%),Requests_Total,RPS" > "$CSV_FILE"

# ── K6 stable summary for parsing ──────────────────────────────────────────────
export K6_NO_COLOR=1
export K6_SUMMARY_TREND_STATS="avg,med,p(90),p(95),p(99),max"

# ── Header ─────────────────────────────────────────────────────────────────────
echo -e "${BLUE}════════════════════════════════════════════════════════${NC}" | tee -a "$OUTPUT_FILE"
echo -e "${GREEN}  RESPONSE TIME TESTING BY REPLICA COUNT (Improved)${NC}" | tee -a "$OUTPUT_FILE"
echo -e "${BLUE}════════════════════════════════════════════════════════${NC}" | tee -a "$OUTPUT_FILE"
echo "" | tee -a "$OUTPUT_FILE"
echo -e "${CYAN}Configuration:${NC}" | tee -a "$OUTPUT_FILE"
echo "  Namespace: $NAMESPACE" | tee -a "$OUTPUT_FILE"
echo "  Deployment: $DEPLOYMENT_NAME" | tee -a "$OUTPUT_FILE"
echo "  Base URL: $BASE_URL" | tee -a "$OUTPUT_FILE"
echo "  Test Duration: $TEST_DURATION per replica level" | tee -a "$OUTPUT_FILE"
echo "  Virtual Users: $TEST_VUS concurrent" | tee -a "$OUTPUT_FILE"
echo "  Rollout Timeout: $ROLL_TIMEOUT" | tee -a "$OUTPUT_FILE"
echo "  Cooldown: ${COOLDOWN_TIME}s" | tee -a "$OUTPUT_FILE"
echo "" | tee -a "$OUTPUT_FILE"
echo -e "${CYAN}Test Plan (replicas):${NC} $PHASES" | tee -a "$OUTPUT_FILE"
echo -e "${BLUE}════════════════════════════════════════════════════════${NC}" | tee -a "$OUTPUT_FILE"
echo "" | tee -a "$OUTPUT_FILE"

# ── Helpers ────────────────────────────────────────────────────────────────────
parse_ms () {
  # Accept "2.79ms" | "0.002s" | "883.65µs" | "2.79"
  local val="${1:-}"
  if [[ -z "$val" ]]; then echo "0"; return; fi
  if echo "$val" | grep -q "ms"; then
    echo "$val" | sed -E 's/ms//'
  elif echo "$val" | grep -q "µs"; then
    local us; us=$(echo "$val" | sed -E 's/µs//')
    awk "BEGIN {printf \"%.6f\", ($us+0)/1000}"
  elif echo "$val" | grep -q "s"; then
    local s; s=$(echo "$val" | sed -E 's/s//')
    awk "BEGIN {printf \"%.6f\", ($s+0)*1000}"
  else
    echo "$val"
  fi
}

scale_deployment () {
  local replicas="$1"
  local phase_name="$2"

  echo -e "\n${CYAN}━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━${NC}" | tee -a "$OUTPUT_FILE"
  echo -e "${GREEN}Phase: $phase_name${NC}" | tee -a "$OUTPUT_FILE"
  echo -e "${CYAN}━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━${NC}" | tee -a "$OUTPUT_FILE"

  echo -e "\n${YELLOW}Scaling to $replicas replicas...${NC}" | tee -a "$OUTPUT_FILE"
  kubectl scale deployment "$DEPLOYMENT_NAME" -n "$NAMESPACE" --replicas="$replicas"

  echo "Waiting for rollout to complete (timeout ${ROLL_TIMEOUT})..." | tee -a "$OUTPUT_FILE"
  if ! kubectl rollout status deployment/"$DEPLOYMENT_NAME" -n "$NAMESPACE" --timeout="$ROLL_TIMEOUT"; then
    echo -e "${RED}Rollout did not complete within ${ROLL_TIMEOUT}.${NC}" | tee -a "$OUTPUT_FILE"
  fi

  echo "Verifying ready replicas..." | tee -a "$OUTPUT_FILE"
  local READY_PODS="0"
  for _ in $(seq 1 60); do
    READY_PODS=$(kubectl get deploy "$DEPLOYMENT_NAME" -n "$NAMESPACE" -o jsonpath='{.status.readyReplicas}' 2>/dev/null || echo "0")
    READY_PODS=${READY_PODS:-0}
    if [[ "$READY_PODS" == "$replicas" ]]; then break; fi
    sleep 5
  done

  echo "Waiting for pods to stabilize (${COOLDOWN_TIME}s)..." | tee -a "$OUTPUT_FILE"
  sleep "$COOLDOWN_TIME"

  READY_PODS=$(kubectl get deploy "$DEPLOYMENT_NAME" -n "$NAMESPACE" -o jsonpath='{.status.readyReplicas}' 2>/dev/null || echo "0")
  echo -e "${GREEN}✓ Ready pods: ${READY_PODS:-0}/$replicas${NC}" | tee -a "$OUTPUT_FILE"

  echo "" | tee -a "$OUTPUT_FILE"
  kubectl get pods -n "$NAMESPACE" -l app="$DEPLOYMENT_NAME" | tee -a "$OUTPUT_FILE"
  echo "" | tee -a "$OUTPUT_FILE"
}

quick_curl_test () {
  local replicas="$1"
  echo -e "${YELLOW}Quick curl sanity check (10 requests)...${NC}" | tee -a "$OUTPUT_FILE"
  local TIMES=()
  for i in {1..10}; do
    local TIME; TIME=$(curl -w "%{time_total}" -o /dev/null -s "$BASE_URL/api/cpu?iterations=1000000" 2>&1)
    TIMES+=("$TIME")
    local MS; MS=$(awk "BEGIN {printf \"%.0f\", ($TIME+0)*1000}")
    echo "  Request $i: ${MS}ms" | tee -a "$OUTPUT_FILE"
    sleep 0.5
  done
  local AVG_TIME; AVG_TIME=$(printf '%s\n' "${TIMES[@]}" | awk '{sum+=$1} END {printf "%.0f", (sum/NR)*1000}')
  echo -e "${GREEN}  Average (curl): ${AVG_TIME}ms${NC}" | tee -a "$OUTPUT_FILE"
  echo "" | tee -a "$OUTPUT_FILE"
}

run_load_test () {
  local replicas="$1"
  local phase_name="$2"

  echo -e "${YELLOW}Running load test with k6...${NC}" | tee -a "$OUTPUT_FILE"
  echo "  Duration: $TEST_DURATION" | tee -a "$OUTPUT_FILE"
  echo "  Virtual Users: $TEST_VUS" | tee -a "$OUTPUT_FILE"
  echo "" | tee -a "$OUTPUT_FILE"

  # Temp k6 script
  cat > /tmp/k6_replica_test.js <<'EOFK6'
import http from 'k6/http';
import { check, sleep } from 'k6';
import { Trend } from 'k6/metrics';

const cpuDuration = new Trend('cpu_request_duration');
const memoryDuration = new Trend('memory_request_duration');
const basicDuration = new Trend('basic_request_duration');

export const options = {
  vus: parseInt(__ENV.TEST_VUS || '20'),
  duration: __ENV.TEST_DURATION || '3m',
  thresholds: { http_req_duration: ['p(95)<10000'] },
};

const BASE_URL = __ENV.BASE_URL || 'http://localhost:8000';

export default function () {
  const r = Math.random();

  if (r < 0.4) {
    const iterations = Math.floor(Math.random() * 2000000) + 1000000;
    const res = http.get(`${BASE_URL}/api/cpu?iterations=${iterations}`);
    cpuDuration.add(res.timings.duration);
    check(res, { 'cpu status 200': (r) => r.status === 200 });
  } else if (r < 0.7) {
    const sizeMb = Math.floor(Math.random() * 40) + 30;
    const res = http.get(`${BASE_URL}/api/memory?size_mb=${sizeMb}`);
    memoryDuration.add(res.timings.duration);
    check(res, { 'memory status 200': (r) => r.status === 200 });
  } else {
    const res = http.get(`${BASE_URL}/api`);
    basicDuration.add(res.timings.duration);
    check(res, { 'basic status 200': (r) => r.status === 200 });
  }

  sleep(Math.random() * 1 + 0.5);
}
EOFK6

  # Run k6
  local K6_JSON="/tmp/k6_summary_${replicas}.json"
  local K6_OUTPUT
  set +e
  K6_OUTPUT=$(k6 run \
      --summary-export="$K6_JSON" \
      --env BASE_URL="$BASE_URL" \
      --env TEST_VUS="$TEST_VUS" \
      --env TEST_DURATION="$TEST_DURATION" \
      /tmp/k6_replica_test.js 2>&1)
  local K6_EXIT=$?
  set -e
  echo "$K6_OUTPUT" | tee -a "$OUTPUT_FILE"

  echo "" | tee -a "$OUTPUT_FILE"
  echo -e "${GREEN}📊 Results Summary for $replicas replicas:${NC}" | tee -a "$OUTPUT_FILE"
  echo -e "${CYAN}────────────────────────────────────────────────────────${NC}" | tee -a "$OUTPUT_FILE"

  # Prefer JSON (seconds) → convert to ms
  local AVG="0" P50="0" P90="0" P95="0" P99="0" MAX="0" ERR_RATE="0" TOTAL_REQS="0" RPS="0"
  if [ -f "$K6_JSON" ] && command -v jq >/dev/null 2>&1; then
    AVG=$(jq -r '((.metrics.http_req_duration.values.avg // 0) * 1000)' "$K6_JSON")
    P50=$(jq -r '((.metrics.http_req_duration.values.med // 0) * 1000)' "$K6_JSON")
    P90=$(jq -r '((.metrics.http_req_duration.values["p(90)"] // 0) * 1000)' "$K6_JSON")
    P95=$(jq -r '((.metrics.http_req_duration.values["p(95)"] // 0) * 1000)' "$K6_JSON")
    P99=$(jq -r '((.metrics.http_req_duration.values["p(99)"] // 0) * 1000)' "$K6_JSON")
    MAX=$(jq -r '((.metrics.http_req_duration.values.max // 0) * 1000)' "$K6_JSON")
    ERR_RATE=$(jq -r '(.metrics.http_req_failed.values.rate // 0)' "$K6_JSON")
    TOTAL_REQS=$(jq -r '(.metrics.http_reqs.values.count // 0)' "$K6_JSON")
    RPS=$(jq -r '(.metrics.http_reqs.values.rate // 0)' "$K6_JSON")
  else
    # Fallback parse from text
    local line
    line=$(echo "$K6_OUTPUT" | grep -E "http_req_duration" | head -1 || true)
    AVG=$(parse_ms "$(echo "$line" | sed -nE 's/.*avg=([^ ]+).*/\1/p')")
    P50=$(parse_ms "$(echo "$line" | sed -nE 's/.*med=([^ ]+).*/\1/p')")
    P90=$(parse_ms "$(echo "$line" | sed -nE 's/.*p\(90\)=([^ ]+).*/\1/p')")
    P95=$(parse_ms "$(echo "$line" | sed -nE 's/.*p\(95\)=([^ ]+).*/\1/p')")
    P99=$(parse_ms "$(echo "$line" | sed -nE 's/.*p\(99\)=([^ ]+).*/\1/p')")
    MAX=$(parse_ms "$(echo "$line" | sed -nE 's/.*max=([^ ]+).*/\1/p')")
    # ERR/RPS/COUNT best-effort
    ERR_RATE=$(echo "$K6_OUTPUT" | awk '/http_req_failed/ {print $3}' | head -1 | sed 's/%//' | awk '{print ($1+0)/100}')
    TOTAL_REQS=$(echo "$K6_OUTPUT" | awk '/http_reqs/ {print $3}' | head -1)
    RPS=$(echo "$K6_OUTPUT" | awk '/http_reqs/ {print $(NF)}' | sed -E 's/.*\(([0-9.]+)\/s\).*/\1/' | head -1)
    AVG=${AVG:-0}; P50=${P50:-0}; P90=${P90:-0}; P95=${P95:-0}; P99=${P99:-0}; MAX=${MAX:-0}
    ERR_RATE=${ERR_RATE:-0}; TOTAL_REQS=${TOTAL_REQS:-0}; RPS=${RPS:-0}
  fi

  # Print
  printf "  %-20s %10.2f ms\n" "Average RT:" "$AVG" | tee -a "$OUTPUT_FILE"
  printf "  %-20s %10.2f ms\n" "Median (p50):" "$P50" | tee -a "$OUTPUT_FILE"
  printf "  %-20s %10.2f ms\n" "90th percentile:" "$P90" | tee -a "$OUTPUT_FILE"
  printf "  %-20s %10.2f ms\n" "95th percentile:" "$P95" | tee -a "$OUTPUT_FILE"
  printf "  %-20s %10.2f ms\n" "99th percentile:" "$P99" | tee -a "$OUTPUT_FILE"
  printf "  %-20s %10.2f ms\n" "Maximum:" "$MAX" | tee -a "$OUTPUT_FILE"
  printf "  %-20s %10.2f %%\n" "Error Rate:" "$(awk "BEGIN {printf \"%.4f\", ($ERR_RATE+0)*100}")" | tee -a "$OUTPUT_FILE"
  printf "  %-20s %10.0f\n" "Total Requests:" "$TOTAL_REQS" | tee -a "$OUTPUT_FILE"
  printf "  %-20s %10.2f req/s\n" "Throughput:" "$RPS" | tee -a "$OUTPUT_FILE"
  echo -e "${CYAN}────────────────────────────────────────────────────────${NC}" | tee -a "$OUTPUT_FILE"

  # Save to CSV
  local ERR_PCT; ERR_PCT=$(awk "BEGIN {printf \"%.6f\", ($ERR_RATE+0)*100}")
  echo "$replicas,$phase_name,$AVG,$P50,$P90,$P95,$P99,$MAX,$ERR_PCT,$TOTAL_REQS,$RPS" >> "$CSV_FILE"

  # Endpoint-specific averages from custom Trends (text only; unit-less → assume ms already)
  local CPU_AVG; CPU_AVG=$(echo "$K6_OUTPUT" | awk '/cpu_request_duration/ && /avg=/' | sed -E 's/.*avg=([0-9.]+).*/\1/' | head -1)
  local MEM_AVG; MEM_AVG=$(echo "$K6_OUTPUT" | awk '/memory_request_duration/ && /avg=/' | sed -E 's/.*avg=([0-9.]+).*/\1/' | head -1)
  local BASIC_AVG; BASIC_AVG=$(echo "$K6_OUTPUT" | awk '/basic_request_duration/ && /avg=/' | sed -E 's/.*avg=([0-9.]+).*/\1/' | head -1)
  [ -n "${CPU_AVG:-}" ] && echo -e "${YELLOW}  CPU Endpoint: avg=${CPU_AVG}ms${NC}" | tee -a "$OUTPUT_FILE"
  [ -n "${MEM_AVG:-}" ] && echo -e "${YELLOW}  Memory Endpoint: avg=${MEM_AVG}ms${NC}" | tee -a "$OUTPUT_FILE"
  [ -n "${BASIC_AVG:-}" ] && echo -e "${YELLOW}  Basic Endpoint: avg=${BASIC_AVG}ms${NC}" | tee -a "$OUTPUT_FILE"

  echo "" | tee -a "$OUTPUT_FILE"

  # return k6 exit code gracefully
  return "$K6_EXIT"
}

# ── Main ───────────────────────────────────────────────────────────────────────
main () {
  echo "Starting response time testing..." | tee -a "$OUTPUT_FILE"
  echo "Results will be saved to:" | tee -a "$OUTPUT_FILE"
  echo "  - $OUTPUT_FILE" | tee -a "$OUTPUT_FILE"
  echo "  - $CSV_FILE" | tee -a "$OUTPUT_FILE"
  echo "" | tee -a "$OUTPUT_FILE"

  IFS=',' read -r -a REPLS <<< "$PHASES"
  for r in "${REPLS[@]}"; do
    PHASE_NAME=$( [ "$r" -eq 1 ] && echo "MIN_REPLICAS" || ( [ "$r" -eq 50 ] && echo "MAX_REPLICAS" || echo "MEDIUM_REPLICAS" ) )
    scale_deployment "$r" "$PHASE_NAME"
    quick_curl_test "$r"
    run_load_test "$r" "$PHASE_NAME" || true
  done

  # Final comparison
  echo -e "\n${BLUE}════════════════════════════════════════════════════════${NC}" | tee -a "$OUTPUT_FILE"
  echo -e "${GREEN}  COMPARISON SUMMARY${NC}" | tee -a "$OUTPUT_FILE"
  echo -e "${BLUE}════════════════════════════════════════════════════════${NC}" | tee -a "$OUTPUT_FILE"
  echo "" | tee -a "$OUTPUT_FILE"

  echo -e "${CYAN}Response Time by Replica Count:${NC}" | tee -a "$OUTPUT_FILE"
  echo "" | tee -a "$OUTPUT_FILE"
  printf "%-10s | %-10s | %-10s | %-10s | %-10s | %-8s\n" \
      "Replicas" "Avg (ms)" "p50 (ms)" "p90 (ms)" "p95 (ms)" "Error %" | tee -a "$OUTPUT_FILE"
  echo "-----------|------------|------------|------------|------------|----------" | tee -a "$OUTPUT_FILE"

  tail -n +2 "$CSV_FILE" | while IFS=, read -r replicas phase avg p50 p90 p95 p99 max err total rps; do
    printf "%-10s | %10.0f | %10.0f | %10.0f | %10.0f | %7.2f%%\n" \
      "$replicas" "${avg:-0}" "${p50:-0}" "${p90:-0}" "${p95:-0}" "${err:-0}" | tee -a "$OUTPUT_FILE"
  done

  echo "" | tee -a "$OUTPUT_FILE"

  # Analysis & Recommendation (ignore p95=0 rows)
  echo -e "${CYAN}📊 Analysis & Recommendations:${NC}" | tee -a "$OUTPUT_FILE"
  echo "" | tee -a "$OUTPUT_FILE"

  # Extract p95 values > 0
  mapfile -t P95S < <(awk -F',' 'NR>1 && $6+0>0 {print $6}' "$CSV_FILE")
  if [ "${#P95S[@]}" -gt 0 ]; then
    # avg p95
    AVG_P95=$(printf "%s\n" "${P95S[@]}" | awk '{s+=$1} END {if(NR>0) printf "%.0f", s/NR; else print 0}')
    # min/max p95
    MIN_P95=$(printf "%s\n" "${P95S[@]}" | sort -n | head -1)
    MAX_P95=$(printf "%s\n" "${P95S[@]}" | sort -n | tail -1)

    CONSERVATIVE=$(awk "BEGIN {printf \"%.0f\", ($MAX_P95+0) * 1.25}")
    BALANCED=$(awk "BEGIN {printf \"%.0f\", ($AVG_P95+0) * 1.25}")
    AGGRESSIVE=$(awk "BEGIN {printf \"%.0f\", ($MIN_P95+0) * 1.10}")

    echo "Based on non-zero p95 values (ms):" | tee -a "$OUTPUT_FILE"
    echo "  Min p95: $MIN_P95 ms; Avg p95: $AVG_P95 ms; Max p95: $MAX_P95 ms" | tee -a "$OUTPUT_FILE"
    echo "" | tee -a "$OUTPUT_FILE"

    echo -e "${CYAN}━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━${NC}" | tee -a "$OUTPUT_FILE"
    echo -e "${GREEN}Recommended MAX_RESPONSE_TIME for .env:${NC}" | tee -a "$OUTPUT_FILE"
    echo -e "${CYAN}━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━${NC}" | tee -a "$OUTPUT_FILE"
    echo -e "  ${YELLOW}Conservative (max p95 × 1.25): ${CONSERVATIVE} ms${NC}" | tee -a "$OUTPUT_FILE"
    echo -e "  ${GREEN}Balanced (avg p95 × 1.25):     ${BALANCED} ms  ✅ RECOMMENDED${NC}" | tee -a "$OUTPUT_FILE"
    echo -e "  ${YELLOW}Aggressive (min p95 × 1.10):   ${AGGRESSIVE} ms${NC}" | tee -a "$OUTPUT_FILE"
  else
    echo -e "${RED}All p95 values are zero — check parsing or k6 run.${NC}" | tee -a "$OUTPUT_FILE"
  fi

  echo "" | tee -a "$OUTPUT_FILE"
  echo -e "${BLUE}════════════════════════════════════════════════════════${NC}" | tee -a "$OUTPUT_FILE"
  echo "" | tee -a "$OUTPUT_FILE"

  echo -e "${GREEN}✓ Testing complete!${NC}" | tee -a "$OUTPUT_FILE"
  echo "Full results: $OUTPUT_FILE" | tee -a "$OUTPUT_FILE"
  echo "CSV:          $CSV_FILE" | tee -a "$OUTPUT_FILE"

  # Cleanup temps
  rm -f /tmp/k6_replica_test.js /tmp/k6_summary_*.json || true
}

main
