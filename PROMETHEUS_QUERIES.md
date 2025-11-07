# Prometheus Queries for RL Autoscaler

This document contains all Prometheus queries used by your RL autoscaler agent for monitoring the `flask-app` deployment.

## 🔗 Connection Info

```
Prometheus URL: http://10.34.4.150:30080/monitoring
Namespace: default
Deployment: flask-app
Metrics Interval: 35s
Quantile: 0.90 (90th percentile)
```

## 📊 Core Metrics Queries

### 1. **CPU Usage (Rate)**

Measures CPU usage rate per pod over the last 35 seconds. Uses OR logic to include both active and idle pods:

```promql
(
  sum by (pod) (
    rate(container_cpu_usage_seconds_total{
      namespace="default",
      pod=~"flask-app-.*",
      container!="",
      container!="POD"
    }[35s])
  )
) OR (
  count by (pod) (
    container_cpu_usage_seconds_total{
      namespace="default",
      pod=~"flask-app-.*",
      container!="",
      container!="POD"
    }
  ) * 0
)
```

**What it returns**: CPU usage in cores per pod (e.g., 0.204 = 204m). Idle pods return 0.

**Why OR logic?** The `rate()` function only returns values for pods with CPU activity changes. Idle pods would be missing from results. The OR clause adds idle pods with 0 CPU usage, ensuring all pods are included.

---

### 2. **Memory Usage (Working Set)**

Measures current memory usage per pod:

```promql
sum by (pod) (
  container_memory_working_set_bytes{
    namespace="default",
    pod=~"flask-app-.*",
    container!="",
    container!="POD"
  }
)
```

**What it returns**: Memory usage in bytes per pod (e.g., 25165824 = 24Mi)

---

### 3. **CPU Limits**

Gets CPU resource limits per pod:

```promql
sum by (pod) (
  kube_pod_container_resource_limits{
    namespace="default",
    pod=~"flask-app-.*",
    resource="cpu",
    unit="core"
  }
)
```

**What it returns**: CPU limit in cores (e.g., 0.5 = 500m)

---

### 4. **Memory Limits**

Gets memory resource limits per pod:

```promql
sum by (pod) (
  kube_pod_container_resource_limits{
    namespace="default",
    pod=~"flask-app-.*",
    resource="memory",
    unit="byte"
  }
)
```

**What it returns**: Memory limit in bytes (e.g., 536870912 = 512Mi)

---

### 5. **Response Time (90th Percentile)**

Your Flask app exposes metrics with the `path` label (changed from `endpoint` to avoid Prometheus label conflicts):

```promql
1000 *
histogram_quantile(
  0.90,
  sum by (le) (
    rate(http_request_duration_seconds_bucket{
      namespace="default",
      pod=~"flask-app-.*",
      method="GET",
      path="/api/cpu"
    }[35s])
  )
)
```

**What it returns**: 90th percentile response time in milliseconds

**Note**: The label is `path` (not `endpoint` or `exported_endpoint`) to avoid Prometheus renaming conflicts.

---

## 🎯 Calculated Metrics

Your RL agent calculates these from the raw metrics:

### CPU Usage Percentage
```python
cpu_percentage = (cpu_usage_cores / cpu_limit_cores) * 100
```

Example:
- CPU Usage: 0.204 cores (204m)
- CPU Limit: 0.5 cores (500m)
- **Result: 40.8%**

### Memory Usage Percentage
```python
memory_percentage = (memory_used_bytes / memory_limit_bytes) * 100
```

Example:
- Memory Used: 50331648 bytes (48Mi)
- Memory Limit: 536870912 bytes (512Mi)
- **Result: 9.4%**

---

## 🚀 Quick Test Queries (Prometheus UI)

### 1. **Current Pod Count**
```promql
count(kube_pod_info{
  namespace="default",
  pod=~"flask-app-.*",
  created_by_kind="ReplicaSet"
})
```

### 2. **Average CPU Usage Across All Pods**
```promql
avg(
  sum by (pod) (
    rate(container_cpu_usage_seconds_total{
      namespace="default",
      pod=~"flask-app-.*",
      container!="",
      container!="POD"
    }[1m])
  )
) * 100 / 0.5
```
*(Divide by 0.5 because your CPU limit is 500m)*

### 3. **Average Memory Usage Across All Pods**
```promql
avg(
  container_memory_working_set_bytes{
    namespace="default",
    pod=~"flask-app-.*",
    container!="",
    container!="POD"
  }
) / 1024 / 1024
```
*(Result in MiB)*

### 4. **Request Rate (Total RPS)**
```promql
sum(
  rate(http_requests_total{
    namespace="default",
    pod=~"flask-app-.*"
  }[1m])
)
```

### 5. **Request Rate by Path**
```promql
sum by (path) (
  rate(http_requests_total{
    namespace="default",
    pod=~"flask-app-.*"
  }[1m])
)
```

### 6. **95th Percentile Response Time (All Endpoints)**
```promql
histogram_quantile(0.95,
  sum by (le) (
    rate(http_request_duration_seconds_bucket{
      namespace="default",
      pod=~"flask-app-.*"
    }[1m])
  )
) * 1000
```
*(Result in milliseconds)*

### 7. **Error Rate**
```promql
sum(
  rate(http_requests_total{
    namespace="default",
    pod=~"flask-app-.*",
    http_status=~"5.."
  }[1m])
) / sum(
  rate(http_requests_total{
    namespace="default",
    pod=~"flask-app-.*"
  }[1m])
) * 100
```
*(Result as percentage)*

### 8. **Pods by Status**
```promql
count by (phase) (
  kube_pod_status_phase{
    namespace="default",
    pod=~"flask-app-.*"
  }
)
```

### 9. **Container Restarts**
```promql
sum by (pod) (
  kube_pod_container_status_restarts_total{
    namespace="default",
    pod=~"flask-app-.*"
  }
)
```

### 10. **Network I/O**
```promql
# Network receive rate
sum by (pod) (
  rate(container_network_receive_bytes_total{
    namespace="default",
    pod=~"flask-app-.*"
  }[1m])
) / 1024 / 1024

# Network transmit rate
sum by (pod) (
  rate(container_network_transmit_bytes_total{
    namespace="default",
    pod=~"flask-app-.*"
  }[1m])
) / 1024 / 1024
```
*(Result in MiB/s)*

---

## 📈 Grafana Dashboard Queries

### Panel 1: Pod Count Over Time
```promql
count(kube_pod_info{namespace="default", pod=~"flask-app-.*"})
```

### Panel 2: CPU Usage % (Per Pod)
```promql
sum by (pod) (
  rate(container_cpu_usage_seconds_total{
    namespace="default",
    pod=~"flask-app-.*",
    container!="",
    container!="POD"
  }[1m])
) / on(pod) group_left()
sum by (pod) (
  kube_pod_container_resource_limits{
    namespace="default",
    pod=~"flask-app-.*",
    resource="cpu"
  }
) * 100
```

### Panel 3: Memory Usage % (Per Pod)
```promql
sum by (pod) (
  container_memory_working_set_bytes{
    namespace="default",
    pod=~"flask-app-.*",
    container!="",
    container!="POD"
  }
) / on(pod) group_left()
sum by (pod) (
  kube_pod_container_resource_limits{
    namespace="default",
    pod=~"flask-app-.*",
    resource="memory"
  }
) * 100
```

### Panel 4: Request Rate & Response Time (Combined)
```promql
# Left Y-axis: Request Rate
sum(rate(http_requests_total{namespace="default", pod=~"flask-app-.*"}[1m]))

# Right Y-axis: Response Time (ms)
histogram_quantile(0.90,
  sum by (le) (
    rate(http_request_duration_seconds_bucket{
      namespace="default",
      pod=~"flask-app-.*"
    }[1m])
  )
) * 1000
```

### Panel 5: RL Agent Actions Over Time
```promql
# Query from InfluxDB (not Prometheus)
# This would be in your Grafana InfluxDB datasource
SELECT action FROM autoscaling_actions WHERE time > now() - 1h
```

---

## 🔧 Testing Queries Locally

### Using Prometheus UI
1. Open: `http://10.34.4.150:30080/monitoring/graph`
2. Paste any query above
3. Click "Execute"
4. Switch to "Graph" tab to visualize

### Using `curl`
```bash
# Test CPU usage query
curl -G 'http://10.34.4.150:30080/monitoring/api/v1/query' \
  --data-urlencode 'query=sum(rate(container_cpu_usage_seconds_total{namespace="default",pod=~"flask-app-.*",container!="",container!="POD"}[1m]))'

# Test pod count
curl -G 'http://10.34.4.150:30080/monitoring/api/v1/query' \
  --data-urlencode 'query=count(kube_pod_info{namespace="default",pod=~"flask-app-.*"})'
```

### Using Python (same as your agent)
```python
from prometheus_api_client import PrometheusConnect

prom = PrometheusConnect(
    url="http://10.34.4.150:30080/monitoring",
    disable_ssl=True
)

# Test connection
print(prom.check_prometheus_connection())

# Get current pod count
query = 'count(kube_pod_info{namespace="default",pod=~"flask-app-.*"})'
result = prom.custom_query(query)
print(f"Current pods: {result[0]['value'][1]}")

# Get average CPU usage
query = '''
avg(
  sum by (pod) (
    rate(container_cpu_usage_seconds_total{
      namespace="default",
      pod=~"flask-app-.*",
      container!="",
      container!="POD"
    }[1m])
  )
)
'''
result = prom.custom_query(query)
print(f"Avg CPU usage: {float(result[0]['value'][1]):.4f} cores")
```

---

## 🐛 Troubleshooting

### Query Returns Empty Results

1. **Check metric exists**:
   ```promql
   {__name__=~"container_cpu_usage_seconds_total"}
   ```

2. **Check namespace/pod labels**:
   ```promql
   kube_pod_info{namespace="default"}
   ```

3. **Verify ServiceMonitor is scraping**:
   ```bash
   kubectl get servicemonitor -n metrics flask-app-metrics -o yaml
   ```

### Response Time Metrics Not Available

Your Flask app exposes metrics on port 8000 with these labels:
- **Metric name**: `http_request_duration_seconds_bucket`
- **Labels**: `method`, `path` (not `endpoint` or `exported_endpoint`)

Check what labels are available:

```promql
# See all request duration metrics
http_request_duration_seconds_bucket{namespace="default", pod=~"flask-app-.*"}

# Or check all metrics from your app
{namespace="default", pod=~"flask-app-.*", __name__=~"http_.*"}
```

**Important**: The Flask app uses `path` as the label name (defined in `main.py`) to avoid Prometheus automatically renaming it to `exported_endpoint`.

---

## 📊 Expected Values for 50 Pods

Based on your load test results:

| Metric | Value | Query |
|--------|-------|-------|
| **Pod Count** | 50 | `count(kube_pod_info{...})` |
| **Avg CPU Usage** | ~100-150m | CPU usage query / 0.5 * 100 ≈ 20-30% |
| **Avg Memory** | ~25-30Mi | Memory query / 512Mi * 100 ≈ 5-6% |
| **Peak Memory** | ~200Mi | Some pods spike during `/memory` requests |
| **Request Rate** | ~10-15 req/s | With 5-10 concurrent users |
| **Response Time** | 600-2200ms | Depends on endpoint (CPU vs Memory) |

---

## 🎯 Next Steps

1. **Test queries in Prometheus UI** at `http://10.34.4.150:30080/monitoring`
2. **Verify metrics collection** before starting RL agent
3. **Create Grafana dashboard** with the panel queries above
4. **Run RL autoscaler test** and watch metrics in real-time
5. **Compare with InfluxDB data** to validate reward calculations

---

**Pro Tip**: Use Prometheus expression browser to test queries incrementally - start simple and add filters one by one!
