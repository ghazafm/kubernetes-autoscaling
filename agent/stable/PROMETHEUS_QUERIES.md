# Prometheus Queries for RL Autoscaler (Stable Version)

This document contains all Prometheus queries used by the **stable** RL autoscaler agent. Use these queries to manually test and debug metrics collection.

## 🔗 Connection Info

| Setting | Value |
|---------|-------|
| **Prometheus URL** | `http://10.34.4.150:30080/monitoring` |
| **Namespace** | `default` |
| **Deployment** | `flask-app` |
| **Metrics Interval** | `40s` |
| **Quantile** | `0.90` |
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

### 1) CPU Usage per Pod (rate in cores)

Returns CPU usage in **cores** per ready pod.

```promql
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
```

**Expected output**: `0.05` to `0.5` cores per pod under load.

---

### 2) Memory Usage per Pod (working set in bytes)

Returns memory used (working set) in **bytes** per ready pod.

```promql
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
```

**Expected output**: `50000000` to `500000000` bytes (50-500 MB) per pod.

---

### 3) CPU Limits per Pod (in cores)

Returns CPU limit in **cores** per ready pod.

```promql
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
```

**Expected output**: `0.5` cores (500m) per pod.

---

### 4) Memory Limits per Pod (in bytes)

Returns memory limit in **bytes** per ready pod.

```promql
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
```

**Expected output**: `536870912` bytes (512 Mi) per pod.

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

### CPU Percentage Calculation

```python
cpu_percentage = (cpu_usage_cores / cpu_limit_cores) * 100
# Example: 0.25 cores / 0.5 cores = 50%
```

### Memory Percentage Calculation

```python
memory_percentage = (memory_usage_bytes / memory_limit_bytes) * 100
# Example: 256MB / 512MB = 50%
```

### Response Time Percentage Calculation

```python
response_time_percentage = (response_time_ms / MAX_RESPONSE_TIME) * 100
# Example: 150ms / 3000ms = 5%
```

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

### Average CPU Usage (percentage)

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
    /
    sum by (pod) (
        kube_pod_container_resource_limits{
            namespace="default",
            pod=~"flask-app-.*",
            resource="cpu",
            unit="core"
        }
    )
) * 100
```

### Average Memory Usage (percentage)

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
    /
    sum by (pod) (
        kube_pod_container_resource_limits{
            namespace="default",
            pod=~"flask-app-.*",
            resource="memory",
            unit="byte"
        }
    )
) * 100
```

### Total Request Rate (RPS)

```promql
sum(rate(http_requests_total{namespace="default", pod=~"flask-app-.*"}[1m]))
```

### 95th Percentile Response Time (ms)

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

---

## 🔧 Testing from Command Line

### Using curl

```bash
# Test CPU usage query
curl -G 'http://10.34.4.150:30080/monitoring/api/v1/query' \
  --data-urlencode 'query=sum by (pod) (rate(container_cpu_usage_seconds_total{namespace="default",pod=~"flask-app-.*",container!="",container!="POD"}[40s]))'

# Test pod count
curl -G 'http://10.34.4.150:30080/monitoring/api/v1/query' \
  --data-urlencode 'query=count(kube_pod_info{namespace="default",pod=~"flask-app-.*"})'

# Test response time
curl -G 'http://10.34.4.150:30080/monitoring/api/v1/query' \
  --data-urlencode 'query=1000 * histogram_quantile(0.90, sum by (le) (rate(http_request_duration_seconds_bucket{namespace="default",pod=~"flask-app-.*",method="GET",path="/api/cpu"}[40s])))'
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

# Get CPU usage per pod
query = '''
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
'''
result = prom.custom_query(query)
for r in result:
    pod = r['metric']['pod']
    cpu = float(r['value'][1])
    print(f"  {pod}: {cpu:.4f} cores")

# Get response time
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

| Aspect | Old (Broken) | New (Fixed) |
|--------|--------------|-------------|
| Filter | `topk() * on(pod) ...` | `ready_filter == 1` |
| CPU value | Multiplied by timestamp | Correct cores |
| Memory value | Multiplied by timestamp | Correct bytes |
| Ready filter | Complex topk join | Simple equality |

The old queries multiplied metrics by Unix timestamps (1.7 billion+), causing garbage values. The new queries multiply by `1` (ready status) for correct filtering.
