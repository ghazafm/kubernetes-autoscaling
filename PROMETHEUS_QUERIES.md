# Prometheus Queries for RL Autoscaler

This document contains all Prometheus queries used by your RL autoscaler agent for monitoring the `flask-app` deployment.

## 🔗 Connection Info

```
Prometheus URL: http://10.34.4.150:30080/monitoring
Namespace: default

```


The agent constructs a pod-scoped filter (called `pod_filter`) to cap results to the newest N ready pods. The full `pod_filter` expression the code appends is:

```promql
topk({pod_window},
  kube_pod_start_time{
    namespace="default",
    pod=~"flask-app-.*"
  }
  * on(pod) group_left()
    (kube_pod_status_ready{
      namespace="default",
      pod=~"flask-app-.*",
      condition="true"
    } == 1)
)
```

When you paste full queries into Prometheus UI, expand `{pod_window}` and `{interval}` to concrete numbers (e.g. `5` and `15s`). Below are the full copy/paste-ready queries that include the `pod_filter` inline.

### 1) CPU usage per pod (rate)
Used in `metrics._metrics_query` (interval is parameterised):

```promql
sum by (pod) (
  rate(container_cpu_usage_seconds_total{
    namespace="default",
    pod=~"flask-app-.*",
    container!="",
    container!="POD"
  }[{interval}s])
)
* on(pod) group_left() topk({pod_window},
  kube_pod_start_time{
    namespace="default",
    pod=~"flask-app-.*"
  }
  * on(pod) group_left()
    (kube_pod_status_ready{
      namespace="default",
      pod=~"flask-app-.*",
      condition="true"
    } == 1)
)
```

What it returns: CPU usage in cores per pod. The agent expects one entry per ready pod.

---

### 2) Memory usage per pod (working set)

```promql
sum by (pod) (
  container_memory_working_set_bytes{
    namespace="default",
    pod=~"flask-app-.*",
    container!="",
    container!="POD"
  }
)
* on(pod) group_left() topk({pod_window},
  kube_pod_start_time{
    namespace="default",
    pod=~"flask-app-.*"
  }
  * on(pod) group_left()
    (kube_pod_status_ready{
      namespace="default",
      pod=~"flask-app-.*",
      condition="true"
    } == 1)
)
```

What it returns: Memory used (working set) in bytes per pod.

---

### 3) CPU limits per pod

```promql
sum by (pod) (
  kube_pod_container_resource_limits{
    namespace="default",
    pod=~"flask-app-.*",
    resource="cpu",
    unit="core"
  }
)
* on(pod) group_left() topk({pod_window},
  kube_pod_start_time{
    namespace="default",
    pod=~"flask-app-.*"
  }
  * on(pod) group_left()
    (kube_pod_status_ready{
      namespace="default",
      pod=~"flask-app-.*",
      condition="true"
    } == 1)
)
```

What it returns: CPU limit in cores per pod.

---

### 4) Memory limits per pod

```promql
sum by (pod) (
  kube_pod_container_resource_limits{
    namespace="default",
    pod=~"flask-app-.*",
    resource="memory",
    unit="byte"
  }
)
* on(pod) group_left() topk({pod_window},
  kube_pod_start_time{
    namespace="default",
    pod=~"flask-app-.*"
  }
  * on(pod) group_left()
    (kube_pod_status_ready{
      namespace="default",
      pod=~"flask-app-.*",
      condition="true"
    } == 1)
)
```

What it returns: Memory limit in bytes per pod.

---

### 5) Request rate (RPS)

```promql
sum(
  rate(http_requests_total{
    namespace="default",
    pod=~"flask-app-.*"
  }[{interval}s])
)
```

The agent uses this scalar value as the current RPS. For breakdowns use `sum by (path)` or `sum by (pod)`.

---

### 6) Response time (quantile) — per endpoint
Used in `metrics._get_response_time`. The agent queries each `(path, method)` configured in `METRICS_ENDPOINTS_METHOD` and multiplies by 1000 to return milliseconds:

```promql
1000 * histogram_quantile(
  {quantile},
  sum by (le) (
    rate(http_request_duration_seconds_bucket{
      namespace="default",
      pod=~"flask-app-.*",
      method="GET",
      path="/api/cpu"
    }[{interval}s])
  )
)
```

The code will average the finite response-time values across configured endpoints. Important: the instrumentation exposes `path` and `method` labels; the agent relies on `path` (not `endpoint`).

---

### 7) Deployment desired and ready replicas (used in `wait_for_pods_ready`)

q_desired (desired replicas):

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

q_ready (ready replicas - matches pods to ReplicaSet -> Deployment owner):

```promql
scalar(
  sum(
    (kube_pod_status_ready{namespace="default", condition="true"} == 1)
    and on(pod) (
      label_replace(
        kube_pod_owner{namespace="default", owner_kind="ReplicaSet"},
        "replicaset", "$1", "owner_name", "(.*)"
      )
      * on(namespace, replicaset) group_left(owner_name)
        kube_replicaset_owner{
          namespace="default", owner_kind="Deployment", owner_name="flask-app"
        }
    )
  )
)
```

The agent extracts the scalar as an integer (see `_extract_scalar_value` in `agent/utils/cluster.py`).

---

## How the agent handles missing/stale data

- The metric scraping loop in `_scrape_metrics` retries for a short timeout until the number of returned entries matches the expected replicas; otherwise it falls back to last-known values or conservative defaults.
- Percentiles and fallbacks: when limits are missing, the code uses a percentile fallback (90th) across known limits; if no historical data exist, it returns conservative high values to force scale-up.

## Quick test queries

- Current pod count:

```promql
count(kube_pod_info{namespace="default", pod=~"flask-app-.*", created_by_kind="ReplicaSet"})
```

- Average CPU usage across pods (cores):

```promql
avg(sum by (pod) (rate(container_cpu_usage_seconds_total{namespace="default", pod=~"flask-app-.*", container!="", container!="POD"}[1m])))
```

- Total RPS (1m window):

```promql
sum(rate(http_requests_total{namespace="default", pod=~"flask-app-.*"}[1m]))
```

- 95th percentile response time (ms):

```promql
histogram_quantile(0.95, sum by (le) (rate(http_request_duration_seconds_bucket{namespace="default", pod=~"flask-app-.*"}[1m]))) * 1000
```

## Troubleshooting hints

- If a per-pod query returns fewer entries than expected, the agent will retry; check `container` label presence and `pod` naming pattern.
- If response time queries are empty, verify `http_request_duration_seconds_bucket` exists and that labels include `path` and `method`:

```promql
http_request_duration_seconds_bucket{namespace="default", pod=~"flask-app-.*"}
```

If instrumentation uses different labels, update the code or the instrumentation accordingly.

## Example: test from Python (same approach agent uses)

```python
from prometheus_api_client import PrometheusConnect

prom = PrometheusConnect(url="http://10.34.4.150:30080/monitoring", disable_ssl=True)
print(prom.check_prometheus_connection())

q = 'sum(rate(http_requests_total{namespace="default", pod=~"flask-app-.*"}[1m]))'
print(prom.custom_query(q))
```

---

If you want, I can now scan the repository for any other hard-coded PromQL expressions and add them here, or run the queries against your Prometheus and paste current results.

Updated to match the exact expressions generated in `agent/utils/metrics.py` and `agent/utils/cluster.py`.
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
curl -G 'http://10.34.4.150:30080/monitoring/api/v1/query' \container!="POD"}[1m]))'

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
