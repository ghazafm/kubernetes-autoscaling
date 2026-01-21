# Prometheus Queries for RL Autoscaler (Stable Version)

This document contains all Prometheus queries used by the **stable** RL autoscaler agent. Use these queries to manually test and debug metrics collection.

## 🔗 Connection Info

| Setting | Value |
|---------|-------|
| **Prometheus URL** | `http://10.34.4.150:30080/monitoring` |
| **Namespace** | `default` |
| **Deployment** | `flask-app` |
| **Metrics Interval** | `40s` |
| **Quantile** | `0.90` (P90) |
| **Wait Time** | `40s` |
| **Endpoints Monitored** | `/api/cpu`, `/api/memory` |

---

## 📊 Core Metrics Queries

The agent uses a `ready_filter` to only include pods that are Ready:

```promql
kube_pod_status_ready{
    namespace="default",
    pod=~"flask-app-.*",
    condition="true"
} == 1
```

### 1) CPU Usage P90 (scalar in cores)

Returns the **90th percentile** CPU usage across all ready pods as a single scalar value.

```promql
quantile(0.90,
    sum by (pod) (
        rate(container_cpu_usage_seconds_total{
            namespace="default",
            pod=~"flask-app-.*",
            container!="",
            container!="POD"
        }[40s])
    )
    * on(pod) group_left() (
        kube_pod_status_ready{
            namespace="default",
            pod=~"flask-app-.*",
            condition="true"
        } == 1
    )
)
```

**Expected output**: `0.05` to `0.4` cores (scalar value).

**Why P90**: Catches high-load pods early without overreacting to outliers.

---

### 2) Memory Usage P90 (scalar in bytes)

Returns the **90th percentile** memory usage across all ready pods.

```promql
quantile(0.90,
    sum by (pod) (
        container_memory_working_set_bytes{
            namespace="default",
            pod=~"flask-app-.*",
            container!="",
            container!="POD"
        }
    )
    * on(pod) group_left() (
        kube_pod_status_ready{
            namespace="default",
            pod=~"flask-app-.*",
            condition="true"
        } == 1
    )
)
```

**Expected output**: `50000000` to `400000000` bytes (50-400 MB) scalar.

---

### 3) CPU Limits P90 (scalar in cores)

Returns the **90th percentile** CPU limit across all ready pods.

```promql
quantile(0.90,
    sum by (pod) (
        kube_pod_container_resource_limits{
            namespace="default",
            pod=~"flask-app-.*",
            resource="cpu",
            unit="core"
        }
    )
    * on(pod) group_left() (
        kube_pod_status_ready{
            namespace="default",
            pod=~"flask-app-.*",
            condition="true"
        } == 1
    )
)
```

**Expected output**: `0.5` cores (500m) - typically same for all pods.

---

### 4) Memory Limits P90 (scalar in bytes)

Returns the **90th percentile** memory limit across all ready pods.

```promql
quantile(0.90,
    sum by (pod) (
        kube_pod_container_resource_limits{
            namespace="default",
            pod=~"flask-app-.*",
            resource="memory",
            unit="byte"
        }
    )
    * on(pod) group_left() (
        kube_pod_status_ready{
            namespace="default",
            pod=~"flask-app-.*",
            condition="true"
        } == 1
    )
)
```

**Expected output**: `536870912` bytes (512 Mi) - typically same for all pods.

---

### 5) Response Time - /api/cpu endpoint (P90 in milliseconds)

Returns the 90th percentile response time in **milliseconds**.

```promql
1000 * histogram_quantile(
    0.90,
    sum by (le) (
        rate(http_request_duration_seconds_bucket{
            namespace="default",
            pod=~"flask-app-.*",
            method="GET",
            path="/api/cpu"
        }[40s])
    )
)
```

**Expected output**: `100` to `500` ms under normal load, `NaN` if no traffic.

---

### 6) Response Time - /api/memory endpoint (P90 in milliseconds)

```promql
1000 * histogram_quantile(
    0.90,
    sum by (le) (
        rate(http_request_duration_seconds_bucket{
            namespace="default",
            pod=~"flask-app-.*",
            method="GET",
            path="/api/memory"
        }[40s])
    )
)
```

**Expected output**: `50` to `200` ms under normal load.

---

## 🎯 Pod Readiness Queries (used in `wait_for_pods_ready`)

### 7) Desired Replicas

```promql
scalar(
    sum(
        kube_deployment_spec_replicas{
            namespace="default",
            deployment="flask-app"
        }
    )
)
```

**Returns**: Integer count of desired replicas.

---

### 8) Ready Replicas (matches pods to Deployment via ReplicaSet)

```promql
scalar(
    sum(
        (kube_pod_status_ready{namespace="default", condition="true"} == 1)
        and on(pod)
        (
            label_replace(
                kube_pod_owner{namespace="default", owner_kind="ReplicaSet"},
                "replicaset", "$1", "owner_name", "(.*)"
            )
            * on(namespace, replicaset) group_left(owner_name)
            kube_replicaset_owner{
                namespace="default",
                owner_kind="Deployment",
                owner_name="flask-app"
            }
        )
    )
)
```

**Returns**: Integer count of ready replicas belonging to `flask-app` deployment.

---

## 🧮 How Metrics are Processed

The agent now uses **quantile aggregation in PromQL** (not Python), returning scalar values directly.

### CPU Percentage Calculation

```python
# cpu_value and cpu_limit are scalars from quantile() queries
cpu_percentage = (cpu_value / cpu_limit) * 100
# Example: 0.25 cores / 0.5 cores = 50%
```

### Memory Percentage Calculation

```python
# memory_value and memory_limit are scalars from quantile() queries
memory_percentage = (memory_value / memory_limit) * 100
# Example: 256MB / 512MB = 50%
```

### Response Time Percentage Calculation

```python
# Average of P90 response times from multiple endpoints
response_time_percentage = (avg_response_time_ms / MAX_RESPONSE_TIME) * 100
# Example: 150ms / 2500ms = 6%
```

---

## ⏱️ Timing Configuration

| Setting | Value | Rationale |
|---------|-------|-----------|
| **WAIT_TIME** | `40s` | Wait after pods ready before querying metrics |
| **METRICS_INTERVAL** | `40s` | Window for `rate()` calculations |
| **Scrape Interval** | `3s` | Prometheus scrapes flask-app every 3s |
| **cAdvisor Update** | `~15s` | Container metrics update frequency |

**Important**: `WAIT_TIME >= METRICS_INTERVAL` ensures 100% of metrics are post-scaling.

With `rate(...[40s])`, all 50 pods return valid data (100% coverage).

---

## 🔄 Aggregation Comparison

| Method | Query | Use Case |
|--------|-------|----------|
| `avg()` | Average across pods | General overview |
| `quantile(0.9, ...)` | 90th percentile | **Recommended** - catches pressure early |
| `max()` | Highest loaded pod | Detect worst-case |
| `min()` | Lowest loaded pod | Rarely useful |

**Why P90**: Gives earlier warning signal than `avg()` without overreacting to single-pod spikes like `max()`.

---

## 🧪 Quick Test Queries

### Current Pod Count

```promql
count(kube_pod_info{namespace="default", pod=~"flask-app-.*"})
```

### Ready Pod Count

```promql
count(kube_pod_status_ready{namespace="default", pod=~"flask-app-.*", condition="true"} == 1)
```

### Average CPU Usage (percentage) - Alternative

```promql
avg(
    sum by (pod) (
        rate(container_cpu_usage_seconds_total{
            namespace="default",
            pod=~"flask-app-.*",
            container!="",
            container!="POD"
        }[40s])
    )
    / on(pod)
    sum by (pod) (
        kube_pod_container_resource_limits{
            namespace="default",
            pod=~"flask-app-.*",
            resource="cpu",
            unit="core"
        }
    )
    * on(pod) group_left() (
        kube_pod_status_ready{
            namespace="default",
            pod=~"flask-app-.*",
            condition="true"
        } == 1
    )
) * 100
```

### P90 CPU Usage (percentage) - Used by Agent

```promql
quantile(0.90,
    sum by (pod) (
        rate(container_cpu_usage_seconds_total{
            namespace="default",
            pod=~"flask-app-.*",
            container!="",
            container!="POD"
        }[40s])
    )
    / on(pod)
    sum by (pod) (
        kube_pod_container_resource_limits{
            namespace="default",
            pod=~"flask-app-.*",
            resource="cpu",
            unit="core"
        }
    )
    * on(pod) group_left() (
        kube_pod_status_ready{
            namespace="default",
            pod=~"flask-app-.*",
            condition="true"
        } == 1
    )
) * 100
```

### Average Memory Usage (percentage) - Alternative

```promql
avg(
    sum by (pod) (
        container_memory_working_set_bytes{
            namespace="default",
            pod=~"flask-app-.*",
            container!="",
            container!="POD"
        }
    )
    / on(pod)
    sum by (pod) (
        kube_pod_container_resource_limits{
            namespace="default",
            pod=~"flask-app-.*",
            resource="memory",
            unit="byte"
        }
    )
    * on(pod) group_left() (
        kube_pod_status_ready{
            namespace="default",
            pod=~"flask-app-.*",
            condition="true"
        } == 1
    )
) * 100
```

### Total Request Rate (RPS)

```promql
sum(rate(http_requests_total{namespace="default", pod=~"flask-app-.*"}[40s]))
```

### 95th Percentile Response Time (ms)

```promql
histogram_quantile(0.95,
    sum by (le) (
        rate(http_request_duration_seconds_bucket{
            namespace="default",
            pod=~"flask-app-.*"
        }[40s])
    )
) * 1000
```

---

## 🔧 Testing from Command Line

### Using curl

```bash
# Test CPU P90 query (scalar)
curl -G 'http://10.34.4.150:30080/monitoring/api/v1/query' \
  --data-urlencode 'query=quantile(0.90, sum by (pod) (rate(container_cpu_usage_seconds_total{namespace="default",pod=~"flask-app-.*",container!="",container!="POD"}[40s])) * on(pod) group_left() (kube_pod_status_ready{namespace="default",pod=~"flask-app-.*",condition="true"} == 1))'

# Test pod count
curl -G 'http://10.34.4.150:30080/monitoring/api/v1/query' \
  --data-urlencode 'query=count(kube_pod_info{namespace="default",pod=~"flask-app-.*"})'

# Test response time
curl -G 'http://10.34.4.150:30080/monitoring/api/v1/query' \
  --data-urlencode 'query=1000 * histogram_quantile(0.90, sum by (le) (rate(http_request_duration_seconds_bucket{namespace="default",pod=~"flask-app-.*",method="GET",path="/api/cpu"}[40s])))'

# Check pod coverage with rate()[40s]
curl -G 'http://10.34.4.150:30080/monitoring/api/v1/query' \
  --data-urlencode 'query=count(sum by (pod) (rate(container_cpu_usage_seconds_total{namespace="default",pod=~"flask-app-.*",container!="",container!="POD"}[40s])))'
```

### Using Python

```python
from prometheus_api_client import PrometheusConnect

prom = PrometheusConnect(
    url="http://10.34.4.150:30080/monitoring",
    disable_ssl=True
)

# Test connection
print("Connected:", prom.check_prometheus_connection())

# Get ready pod count
query = 'count(kube_pod_status_ready{namespace="default",pod=~"flask-app-.*",condition="true"} == 1)'
result = prom.custom_query(query)
print(f"Ready pods: {result[0]['value'][1] if result else 'N/A'}")

# Get CPU P90 (scalar)
query = '''
quantile(0.90,
    sum by (pod) (
        rate(container_cpu_usage_seconds_total{
            namespace="default",
            pod=~"flask-app-.*",
            container!="",
            container!="POD"
        }[40s])
    )
    * on(pod) group_left() (
        kube_pod_status_ready{
            namespace="default",
            pod=~"flask-app-.*",
            condition="true"
        } == 1
    )
)
'''
result = prom.custom_query(query)
cpu_p90 = float(result[0]['value'][1]) if result else 0.0
print(f"CPU P90: {cpu_p90:.4f} cores")

# Get response time P90
query = '''
1000 * histogram_quantile(0.90,
    sum by (le) (
        rate(http_request_duration_seconds_bucket{
            namespace="default",
            pod=~"flask-app-.*",
            method="GET",
            path="/api/cpu"
        }[40s])
    )
)
'''
result = prom.custom_query(query)
rt = float(result[0]['value'][1]) if result else 'NaN'
print(f"Response time (P90): {rt} ms")
```

---

## 🐛 Troubleshooting

### Query Returns Empty Results

1. **Check if pods exist**:
   ```promql
   kube_pod_info{namespace="default", pod=~"flask-app-.*"}
   ```

2. **Check if pods are Ready**:
   ```promql
   kube_pod_status_ready{namespace="default", pod=~"flask-app-.*", condition="true"}
   ```

3. **Check if metrics are being scraped**:
   ```promql
   up{job="flask-app"}
   ```

### Response Time Returns NaN

This means **no HTTP traffic** is hitting the endpoints. Make sure k6 load test is running:

```bash
k6 run --vus 10 --duration 10m k6-autoscaler-training.js
```

### CPU/Memory Returns 0

1. **Check if container metrics exist**:
   ```promql
   container_cpu_usage_seconds_total{namespace="default", pod=~"flask-app-.*"}
   ```

2. **Check scrape interval** - metrics may not be available yet for new pods.

### Verify Metric Labels

```promql
# Check available labels for response time
http_request_duration_seconds_bucket{namespace="default", pod=~"flask-app-.*"}
```

Make sure `path` and `method` labels match your queries.

---

## 📈 Expected Values Summary

| Metric | Idle | Light Load | Heavy Load |
|--------|------|------------|------------|
| CPU % | 0-5% | 20-50% | 70-100% |
| Memory % | 20-40% | 40-60% | 60-90% |
| Response Time % | 0% (NaN) | 1-5% | 5-50%+ |
| Ready Pods | 1 | 1-5 | 5-20 |

---

## 🔄 Key Differences from Old Queries

| Aspect | Old (Per-Pod) | New (Quantile) |
|--------|---------------|----------------|
| **Aggregation** | Python `np.mean()` | PromQL `quantile(0.9)` |
| **Output** | Array of N pods | Single scalar |
| **Coverage Issue** | Wait for all N pods | Returns value if any pods have data |
| **Sensitivity** | Average (hides outliers) | P90 (catches pressure early) |
| **Network** | N values transferred | 1 value transferred |
| **NaN Handling** | Complex per-pod logic | Simple scalar check |

### Old Query (returned N pod values):
```promql
sum by (pod) (rate(...)) * on(pod) group_left() (ready_filter)
```

### New Query (returns 1 scalar):
```promql
quantile(0.90, sum by (pod) (rate(...)) * on(pod) group_left() (ready_filter))
```
