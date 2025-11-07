# Flask App with RL Autoscaling Metrics

A production-ready Flask application instrumented with `rl-autoscale` for reinforcement learning-based autoscaling.

## 🚀 Features

- ✅ **Health & Readiness Probes** - Kubernetes-native health checks
- ✅ **Prometheus Metrics** - Automatic metrics via `rl-autoscale` library
- ✅ **ConfigMap Support** - Environment-based configuration
- ✅ **Multi-Architecture** - Supports AMD64 and ARM64
- ✅ **Load Testing Endpoints** - CPU and memory intensive endpoints for testing

## 📁 Files

| File | Description |
|------|-------------|
| `main.py` | Flask application with RL metrics |
| `Dockerfile` | Multi-stage Docker build |
| `requirements.txt` | Python dependencies |
| `configmap.yaml` | Kubernetes ConfigMap for environment variables |
| `deployment.yaml` | **Recommended** - Deployment with ConfigMap |
| `app.yaml` | Alternative - Deployment with inline env vars |
| `hpa.yaml` | Horizontal Pod Autoscaler configuration |
| `monitor.yaml` | Prometheus ServiceMonitor (optional) |
| `deploy.sh` | Automated deployment script |
| `ENV_VARS.md` | Complete environment variables documentation |

## 🔧 Environment Variables

| Variable | Default | Description |
|----------|---------|-------------|
| `APP_HOST` | `0.0.0.0` | Host address |
| `APP_PORT` | `5000` | Application port |
| `METRICS_PORT` | `8000` | Prometheus metrics port |
| `DEBUG` | `false` | Debug mode (dev only) |
| `DEFAULT_SLEEP_TIME` | `0.3` | Sleep time for `/api` |
| `MAX_MEMORY_MB` | `100` | Max memory allocation |

See [ENV_VARS.md](ENV_VARS.md) for complete documentation.

## 🚀 Quick Start

### Deploy to Kubernetes

```bash
# Using automated script (recommended)
./deploy.sh

# Or manually
kubectl apply -f configmap.yaml
kubectl apply -f deployment.yaml
kubectl apply -f hpa.yaml
```

### Local Development

```bash
# Install dependencies
pip install -r requirements.txt

# Run locally
export DEBUG=true
export APP_HOST=127.0.0.1
python main.py
```

### Test Endpoints

```bash
# Port forward
kubectl port-forward svc/flask-app 5000:5000

# Basic endpoint
curl http://localhost:5000/api

# Health check
curl http://localhost:5000/health

# Readiness check
curl http://localhost:5000/ready

# CPU intensive (for testing autoscaling)
curl http://localhost:5000/api/cpu?iterations=1000000

# Memory intensive (for testing autoscaling)
curl http://localhost:5000/api/memory?size_mb=50
```

### Check Metrics

```bash
# Port forward metrics
kubectl port-forward svc/flask-app 8000:8000

# View Prometheus metrics
curl http://localhost:8000/metrics
```

## 📊 Metrics Exposed

The app uses [`rl-autoscale`](https://pypi.org/project/rl-autoscale/) to automatically expose:

- `http_request_duration_seconds` - Request duration histogram
- `http_requests_total` - Total request counter

These metrics are used by the RL agent for autoscaling decisions.

## 🔄 Update Configuration

### Option 1: Edit ConfigMap (Recommended)

```bash
# Edit configuration
kubectl edit configmap flask-app-config

# Restart pods to apply changes
kubectl rollout restart deployment/flask-app
```

### Option 2: Direct Environment Variables

Edit `deployment.yaml` and reapply:
```bash
kubectl apply -f deployment.yaml
```

## 🧪 Load Testing

Run comprehensive load tests for RL training:

```bash
cd ../test
./run-k6.sh
```

See [test/README.md](../test/README.md) for more information.

## 📈 Monitoring

### View Pod Status

```bash
# Watch pods
kubectl get pods -l app=flask-app -w

# View logs
kubectl logs -l app=flask-app -f

# Check resource usage
kubectl top pods -l app=flask-app
```

### View HPA Status

```bash
# Watch HPA
kubectl get hpa -w

# Describe HPA
kubectl describe hpa flask-app
```

### Prometheus Queries

```promql
# Request rate
rate(http_requests_total{namespace="default",pod=~"flask-app-.*"}[1m])

# 95th percentile response time
histogram_quantile(0.95, rate(http_request_duration_seconds_bucket[1m]))

# CPU usage
rate(container_cpu_usage_seconds_total{namespace="default",pod=~"flask-app-.*"}[1m])
```

## 🏗️ Architecture

```
┌─────────────────┐
│   Load Balancer │
└────────┬────────┘
         │
    ┌────▼─────┐
    │ Service  │ :80 → :5000 (app)
    │          │ :8000 → :8000 (metrics)
    └────┬─────┘
         │
    ┌────▼──────────────┐
    │  Flask Pods       │
    │  ┌──────────────┐ │
    │  │ main.py      │ │ :5000
    │  │ rl-autoscale │ │ :8000
    │  └──────────────┘ │
    └────┬──────────────┘
         │
    ┌────▼─────┐
    │ConfigMap │ (flask-app-config)
    └──────────┘
```

## 🐛 Troubleshooting

### Pods Not Starting

```bash
# Check pod status
kubectl describe pod -l app=flask-app

# View logs
kubectl logs -l app=flask-app --tail=50
```

### Metrics Not Available

```bash
# Wait 1-2 minutes after deployment
kubectl top pods -l app=flask-app

# Check metrics endpoint directly
kubectl port-forward svc/flask-app 8000:8000
curl http://localhost:8000/metrics
```

### Health Checks Failing

```bash
# Check probe status
kubectl describe pod <pod-name> | grep -A10 "Liveness\|Readiness"

# Test endpoints manually
kubectl port-forward <pod-name> 5000:5000
curl http://localhost:5000/health
curl http://localhost:5000/ready
```

### ConfigMap Changes Not Applied

```bash
# Verify ConfigMap
kubectl get configmap flask-app-config -o yaml

# Force restart
kubectl rollout restart deployment/flask-app

# Check environment in pod
kubectl exec <pod-name> -- env | grep -E 'APP_|METRICS_|DEBUG'
```

## 🔐 Security

- ✅ Non-root container
- ✅ Resource limits defined
- ✅ Health probes configured
- ✅ Debug mode disabled in production
- ⚠️  Consider using Secrets for sensitive data

## 📚 Related Documentation

- [ENV_VARS.md](ENV_VARS.md) - Complete environment variables guide
- [../test/README.md](../test/README.md) - Load testing documentation
- [rl-autoscale](https://pypi.org/project/rl-autoscale/) - Metrics library

## 🎯 Next Steps

1. ✅ Deploy the application
2. ✅ Verify health checks
3. ✅ Confirm metrics are exposed
4. ✅ Deploy HPA
5. ✅ Run load tests
6. ✅ Monitor RL agent decisions

---

**Need help?** Check the troubleshooting section or view logs:
```bash
kubectl logs -l app=flask-app -f
```
