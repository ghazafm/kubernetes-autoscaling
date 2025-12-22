#!/bin/bash
# test-memory-management-nocurl.sh
# Alternative test script that doesn't require curl in pods
# Uses Python's urllib instead

set -euo pipefail

# Colors
RED='\033[0;31m'
GREEN='\033[0;32m'
YELLOW='\033[1;33m'
BLUE='\033[0;34m'
NC='\033[0m' # No Color

# Configuration
DEPLOYMENT="${DEPLOYMENT:-test-flask-app}"
NAMESPACE="${NAMESPACE:-default}"
SERVICE_NAME="${SERVICE_NAME:-test-flask-app}"

echo -e "${BLUE}╔════════════════════════════════════════════════════════╗${NC}"
echo -e "${BLUE}║  Memory Management Testing Script (No curl required)  ║${NC}"
echo -e "${BLUE}╚════════════════════════════════════════════════════════╝${NC}"
echo ""

# Function to get pod names
get_pods() {
    kubectl get pods -n "$NAMESPACE" -l app="$DEPLOYMENT" -o jsonpath='{.items[*].metadata.name}'
}

# Function to get service IP
get_service_ip() {
    kubectl get svc "$SERVICE_NAME" -n "$NAMESPACE" -o jsonpath='{.status.loadBalancer.ingress[0].ip}'
}

# Function to make HTTP request using Python (available in all Python containers)
http_get() {
    local pod=$1
    local path=$2

    kubectl exec "$pod" -n "$NAMESPACE" -- python3 -c "
import urllib.request
import json
try:
    with urllib.request.urlopen('http://localhost:5000${path}', timeout=5) as response:
        data = response.read().decode('utf-8')
        try:
            print(json.dumps(json.loads(data), indent=2))
        except:
            print(data)
except Exception as e:
    print(f'Error: {e}')
" 2>/dev/null
}

# Function to check memory stats for a pod
check_pod_memory() {
    local pod=$1
    echo -e "${YELLOW}Memory stats for $pod:${NC}"

    # Get Kubernetes metrics
    kubectl top pod "$pod" -n "$NAMESPACE" 2>/dev/null || echo "  Metrics not available yet"

    # Get application-level stats using Python
    echo "  Application stats:"
    http_get "$pod" "/memory/stats" || echo "  Could not fetch stats"
    echo ""
}

# Test 1: Check initial memory
echo -e "${GREEN}Test 1: Checking initial pod memory${NC}"
echo "=================================="
pods=($(get_pods))
if [ ${#pods[@]} -eq 0 ]; then
    echo -e "${RED}No pods found for deployment $DEPLOYMENT${NC}"
    exit 1
fi

for pod in "${pods[@]}"; do
    check_pod_memory "$pod"
done

# Test 2: Allocate memory without caching (should release immediately)
echo -e "${GREEN}Test 2: Allocate 50MB without caching${NC}"
echo "=================================="
SERVICE_IP=$(get_service_ip)
if [ -z "$SERVICE_IP" ] || [ "$SERVICE_IP" = "null" ]; then
    echo -e "${YELLOW}LoadBalancer IP not ready, using port-forward${NC}"
    echo "Starting port-forward in background..."
    kubectl port-forward -n "$NAMESPACE" "svc/$SERVICE_NAME" 5000:80 &
    PF_PID=$!
    sleep 2
    SERVICE_IP="localhost:5000"
else
    SERVICE_IP="$SERVICE_IP:80"
fi

echo "Testing memory allocation without caching..."
for i in {1..10}; do
    curl -s "http://$SERVICE_IP/api/memory?size_mb=50&cache=false" > /dev/null 2>&1 || \
    wget -q -O- "http://$SERVICE_IP/api/memory?size_mb=50&cache=false" > /dev/null 2>&1 || \
    echo "Request $i (using external client)"
    echo -n "."
done
echo " Done!"
echo ""

sleep 5
echo "Memory after non-cached allocations:"
for pod in "${pods[@]}"; do
    check_pod_memory "$pod"
done

# Test 3: Allocate memory with caching (should retain)
echo -e "${GREEN}Test 3: Allocate 30MB with caching${NC}"
echo "=================================="
echo "Testing memory allocation with caching..."
for i in {1..5}; do
    curl -s "http://$SERVICE_IP/api/memory?size_mb=30&cache=true" > /dev/null 2>&1 || \
    wget -q -O- "http://$SERVICE_IP/api/memory?size_mb=30&cache=true" > /dev/null 2>&1 || \
    echo "Request $i (using external client)"
    echo -n "."
done
echo " Done!"
echo ""

sleep 5
echo "Memory after cached allocations:"
for pod in "${pods[@]}"; do
    check_pod_memory "$pod"
done

# Test 4: Manual cleanup
echo -e "${GREEN}Test 4: Manual cache cleanup${NC}"
echo "=================================="
echo "Triggering manual cleanup on all pods..."
for pod in "${pods[@]}"; do
    echo "Cleaning $pod..."
    result=$(http_get "$pod" "/clean")
    echo "  Result: $result"
done
echo ""

sleep 5
echo "Memory after cleanup:"
for pod in "${pods[@]}"; do
    check_pod_memory "$pod"
done

# Test 5: Stress test with malloc_trim
echo -e "${GREEN}Test 5: Stress test (50 allocations)${NC}"
echo "=================================="
echo "Running stress test..."
for i in {1..50}; do
    curl -s "http://$SERVICE_IP/api/memory?size_mb=30&cache=false" > /dev/null 2>&1 || \
    wget -q -O- "http://$SERVICE_IP/api/memory?size_mb=30&cache=false" > /dev/null 2>&1 || true
    if [ $((i % 10)) -eq 0 ]; then
        echo "  Completed $i/50 allocations"
    fi
done
echo "Stress test completed!"
echo ""

sleep 10
echo "Memory after stress test:"
for pod in "${pods[@]}"; do
    check_pod_memory "$pod"
done

# Test 6: Check malloc_trim availability
echo -e "${GREEN}Test 6: Verify malloc_trim availability${NC}"
echo "=================================="
for pod in "${pods[@]}"; do
    echo "Checking $pod..."
    kubectl exec "$pod" -n "$NAMESPACE" -- python3 -c "
import ctypes
import sys
try:
    libc = ctypes.CDLL('libc.so.6')
    result = libc.malloc_trim(0)
    print(f'  malloc_trim available: Yes (result={result})')
except Exception as e:
    print(f'  malloc_trim available: No ({e})')
    sys.exit(1)
" 2>/dev/null || echo "  Could not check malloc_trim"
done
echo ""

# Test 7: Monitor memory over time
echo -e "${GREEN}Test 7: Monitor memory for 60 seconds${NC}"
echo "=================================="
echo "Monitoring memory usage (will update every 10 seconds)..."
echo ""

for i in {1..6}; do
    echo "--- Time: ${i}0 seconds ---"
    kubectl top pods -n "$NAMESPACE" -l app="$DEPLOYMENT" 2>/dev/null || echo "Metrics not available"
    echo ""

    if [ $i -lt 6 ]; then
        # Generate some load in the middle
        if [ $i -eq 3 ]; then
            echo "Generating load..."
            for j in {1..20}; do
                curl -s "http://$SERVICE_IP/api/memory?size_mb=30&cache=false" > /dev/null 2>&1 &
            done
            wait
            echo "Load generation complete"
            echo ""
        fi
        sleep 10
    fi
done

# Cleanup
if [ -n "${PF_PID:-}" ]; then
    kill $PF_PID 2>/dev/null || true
fi

# Summary
echo ""
echo -e "${BLUE}╔════════════════════════════════════════════════════════╗${NC}"
echo -e "${BLUE}║  Test Summary                                          ║${NC}"
echo -e "${BLUE}╚════════════════════════════════════════════════════════╝${NC}"
echo ""
echo -e "${YELLOW}Expected Results:${NC}"
echo "  ✓ Non-cached allocations should release memory quickly"
echo "  ✓ Cached allocations should show increased memory usage"
echo "  ✓ Manual cleanup should reduce memory significantly"
echo "  ✓ Stress test should not cause runaway memory growth"
echo "  ✓ malloc_trim should be available on all pods"
echo "  ✓ Memory should stabilize after load"
echo ""
echo -e "${YELLOW}Compare your results against these benchmarks:${NC}"
echo "  • Fresh pod: ~30-40Mi"
echo "  • Under moderate load: ~80-150Mi"
echo "  • After cleanup: Returns to ~50-80Mi"
echo ""
echo -e "${GREEN}Testing complete!${NC}"
echo ""
echo "For continuous monitoring, run:"
echo "  watch kubectl top pods -n $NAMESPACE -l app=$DEPLOYMENT"