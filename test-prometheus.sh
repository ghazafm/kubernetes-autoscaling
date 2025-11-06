#!/bin/bash

# Prometheus Query Test Script
# This script tests all key Prometheus queries used by your RL autoscaler

PROM_URL="http://10.34.4.150:30080/monitoring"
NAMESPACE="default"
DEPLOYMENT="flask-app"

echo "🔍 Testing Prometheus Queries for RL Autoscaler"
echo "================================================"
echo "Prometheus: $PROM_URL"
echo "Namespace: $NAMESPACE"
echo "Deployment: $DEPLOYMENT"
echo ""

# Test 1: Pod Count
echo "📊 Test 1: Current Pod Count"
curl -sG "${PROM_URL}/api/v1/query" \
  --data-urlencode "query=count(kube_pod_info{namespace=\"${NAMESPACE}\",pod=~\"${DEPLOYMENT}-.*\"})" \
  | jq -r '.data.result[0].value[1] // "No data"' \
  | xargs -I {} echo "   Result: {} pods"
echo ""

# Test 2: Average CPU Usage
echo "🔥 Test 2: Average CPU Usage (cores)"
curl -sG "${PROM_URL}/api/v1/query" \
  --data-urlencode "query=avg(sum by (pod) (rate(container_cpu_usage_seconds_total{namespace=\"${NAMESPACE}\",pod=~\"${DEPLOYMENT}-.*\",container!=\"\",container!=\"POD\"}[1m])))" \
  | jq -r '.data.result[0].value[1] // "No data"' \
  | xargs -I {} echo "   Result: {} cores"
echo ""

# Test 3: Average Memory Usage
echo "💾 Test 3: Average Memory Usage (MiB)"
curl -sG "${PROM_URL}/api/v1/query" \
  --data-urlencode "query=avg(container_memory_working_set_bytes{namespace=\"${NAMESPACE}\",pod=~\"${DEPLOYMENT}-.*\",container!=\"\",container!=\"POD\"}) / 1024 / 1024" \
  | jq -r '.data.result[0].value[1] // "No data"' \
  | xargs -I {} echo "   Result: {} MiB"
echo ""

# Test 4: Request Rate
echo "🚀 Test 4: Request Rate (req/s)"
curl -sG "${PROM_URL}/api/v1/query" \
  --data-urlencode "query=sum(rate(http_requests_total{namespace=\"${NAMESPACE}\",pod=~\"${DEPLOYMENT}-.*\"}[1m]))" \
  | jq -r '.data.result[0].value[1] // "No data"' \
  | xargs -I {} echo "   Result: {} req/s"
echo ""

# Test 5: 95th Percentile Response Time
echo "⏱️  Test 5: Response Time p95 (ms)"
curl -sG "${PROM_URL}/api/v1/query" \
  --data-urlencode "query=histogram_quantile(0.95, sum by (le) (rate(http_request_duration_seconds_bucket{namespace=\"${NAMESPACE}\",pod=~\"${DEPLOYMENT}-.*\"}[1m]))) * 1000" \
  | jq -r '.data.result[0].value[1] // "No data"' \
  | xargs -I {} echo "   Result: {} ms"
echo ""

# Test 6: CPU Usage Percentage (vs limits)
echo "📈 Test 6: Average CPU Usage %"
curl -sG "${PROM_URL}/api/v1/query" \
  --data-urlencode "query=avg(sum by (pod) (rate(container_cpu_usage_seconds_total{namespace=\"${NAMESPACE}\",pod=~\"${DEPLOYMENT}-.*\",container!=\"\",container!=\"POD\"}[1m])) / on(pod) group_left() sum by (pod) (kube_pod_container_resource_limits{namespace=\"${NAMESPACE}\",pod=~\"${DEPLOYMENT}-.*\",resource=\"cpu\"}) * 100)" \
  | jq -r '.data.result[0].value[1] // "No data"' \
  | xargs -I {} echo "   Result: {} %"
echo ""

# Test 7: Memory Usage Percentage (vs limits)
echo "📈 Test 7: Average Memory Usage %"
curl -sG "${PROM_URL}/api/v1/query" \
  --data-urlencode "query=avg(sum by (pod) (container_memory_working_set_bytes{namespace=\"${NAMESPACE}\",pod=~\"${DEPLOYMENT}-.*\",container!=\"\",container!=\"POD\"}) / on(pod) group_left() sum by (pod) (kube_pod_container_resource_limits{namespace=\"${NAMESPACE}\",pod=~\"${DEPLOYMENT}-.*\",resource=\"memory\"}) * 100)" \
  | jq -r '.data.result[0].value[1] // "No data"' \
  | xargs -I {} echo "   Result: {} %"
echo ""

# Test 8: Pods by Status
echo "🔍 Test 8: Pods by Status"
curl -sG "${PROM_URL}/api/v1/query" \
  --data-urlencode "query=count by (phase) (kube_pod_status_phase{namespace=\"${NAMESPACE}\",pod=~\"${DEPLOYMENT}-.*\"})" \
  | jq -r '.data.result[] | "   \(.metric.phase): \(.value[1])"'
echo ""

# Test 9: Container Restarts
echo "🔄 Test 9: Total Container Restarts"
curl -sG "${PROM_URL}/api/v1/query" \
  --data-urlencode "query=sum(kube_pod_container_status_restarts_total{namespace=\"${NAMESPACE}\",pod=~\"${DEPLOYMENT}-.*\"})" \
  | jq -r '.data.result[0].value[1] // "0"' \
  | xargs -I {} echo "   Result: {} restarts"
echo ""

# Test 10: Check Prometheus Connection
echo "🔗 Test 10: Prometheus Connection"
if curl -s "${PROM_URL}/-/healthy" | grep -q "Prometheus"; then
  echo "   Result: ✅ Connected"
else
  echo "   Result: ❌ Connection failed"
fi
echo ""

echo "================================================"
echo "✅ Prometheus query tests completed!"
echo ""
echo "💡 To view in browser:"
echo "   ${PROM_URL}/graph"
echo ""
echo "📚 Full documentation:"
echo "   See PROMETHEUS_QUERIES.md"
