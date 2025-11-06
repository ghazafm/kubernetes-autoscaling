# K6 Load Tests for RL Autoscaler Training

This directory contains comprehensive k6 load testing scripts designed to train and validate your Reinforcement Learning (RL) autoscaler with realistic traffic patterns.

## 📋 Test Files Overview

### 1. **k6-autoscaler-training.js** (Primary Training Test)
**Duration:** ~60 minutes
**Purpose:** Comprehensive real-world daily traffic patterns

**Scenarios Covered:**
- ✅ Morning ramp-up (business hours start)
- ✅ Steady daytime load (normal operations)
- ✅ Lunch hour dip (realistic user behavior)
- ✅ Afternoon peak (highest daily load)
- ✅ Flash spike (viral content/promotions)
- ✅ Evening decline (gradual wind-down)
- ✅ Night-time low (maintenance window)
- ✅ Oscillating load (rapid adaptation testing)

**Use Case:** Primary training data collection for RL agent learning optimal scaling policies.

**Run Command:**
```bash
k6 run --env BASE_URL=http://your-app:5000 k6-autoscaler-training.js
```

---

### 2. **k6-autoscaler-edge-cases.js** (Edge Cases & Stress)
**Duration:** ~40 minutes
**Purpose:** Challenge RL agent with difficult scenarios

**Scenarios Covered:**
- ✅ Cold start (0 → moderate load)
- ✅ Thundering herd (extreme sudden spike)
- ✅ Sawtooth pattern (repeated spikes)
- ✅ Slow leak (gradual sustained increase)
- ✅ Staircase pattern (discrete load levels)
- ✅ Jitter pattern (noisy load)
- ✅ Sustained maximum (endurance test)
- ✅ Rapid oscillation (high-frequency changes)
- ✅ Asymmetric ramp (slow up, fast down)
- ✅ Dead zone (minimal load)

**Use Case:** Robust policy learning, handling outliers, and stress testing.

**Run Command:**
```bash
k6 run --env BASE_URL=http://your-app:5000 k6-autoscaler-edge-cases.js
```

---

### 3. **k6-autoscaler-weekly.js** (Weekly Simulation)
**Duration:** ~50 minutes
**Purpose:** Simulate entire week of traffic (compressed)

**Scenarios Covered:**
- ✅ Monday - Week start with gradual increase
- ✅ Tuesday - Regular business day
- ✅ Wednesday - Peak day with marketing spikes
- ✅ Thursday - Post-peak decline
- ✅ Friday - Early week-end decline
- ✅ Saturday - Weekend low activity
- ✅ Sunday - Minimal weekend activity

**Daily Patterns Per Day:**
- Morning rush, lunch dip, afternoon peak, evening decline
- Time-of-day specific workload distribution

**Use Case:** Weekly pattern recognition, time-series prediction, recurring pattern optimization.

**Run Command:**
```bash
k6 run --env BASE_URL=http://your-app:5000 k6-autoscaler-weekly.js
```

---

### 4. **k6-cpu-stress.js** (CPU-Focused)
**Duration:** ~8 minutes
**Purpose:** Pure CPU stress testing

**Use Case:** Validate CPU-based scaling decisions.

---

### 5. **k6-memory-stress.js** (Memory-Focused)
**Duration:** ~8 minutes
**Purpose:** Pure memory stress testing

**Use Case:** Validate memory-based scaling decisions.

---

### 6. **k6-spike.js** (Quick Spike Test)
**Duration:** ~3 minutes
**Purpose:** Fast spike testing for quick validation

**Use Case:** Quick validation after model updates.

---

## 🚀 Quick Start

### Prerequisites
1. Install k6:
   ```bash
   # macOS
   brew install k6

   # Linux
   sudo apt-get install k6

   # Or download from: https://k6.io/docs/getting-started/installation/
   ```

2. Set up environment variables:
   ```bash
   cp .env.example .env
   # Edit .env and set your BASE_URL
   ```

3. Ensure your application is running and accessible

### Running Tests

**Single Test:**
```bash
# Using environment variable
export BASE_URL=http://your-app:5000
k6 run k6-autoscaler-training.js

# Or inline
k6 run --env BASE_URL=http://your-app:5000 k6-autoscaler-training.js
```

**Complete Training Session (Recommended Order):**
```bash
# 1. Basic validation (quick)
k6 run --env BASE_URL=http://your-app:5000 k6-spike.js

# 2. Comprehensive training (main dataset)
k6 run --env BASE_URL=http://your-app:5000 k6-autoscaler-training.js

# 3. Edge cases (robustness)
k6 run --env BASE_URL=http://your-app:5000 k6-autoscaler-edge-cases.js

# 4. Weekly patterns (time-series learning)
k6 run --env BASE_URL=http://your-app:5000 k6-autoscaler-weekly.js

# 5. Specific stress tests (optional)
k6 run --env BASE_URL=http://your-app:5000 k6-cpu-stress.js
k6 run --env BASE_URL=http://your-app:5000 k6-memory-stress.js
```

**Using the Runner Script:**
```bash
# Make executable
chmod +x run-k6.sh

# Run specific test
./run-k6.sh k6-autoscaler-training.js

# Run all tests sequentially
./run-k6.sh all
```

---

## 📊 Understanding Test Results

### Key Metrics to Monitor

1. **Response Time:**
   - p50 (median): Typical user experience
   - p95: 95% of users experience this or better
   - p99: 99th percentile (outliers)

2. **Error Rate:**
   - Target: < 12% for training (allows learning from overload)
   - Production: < 5% recommended

3. **Throughput:**
   - Requests per second (RPS)
   - Total requests processed

### Output Files

Each test generates a JSON summary file:
- `training-summary-YYYY-MM-DD-HH-MM.json`
- `edge-cases-summary-YYYY-MM-DD-HH-MM.json`
- `weekly-simulation-summary-YYYY-MM-DD-HH-MM.json`

These files contain detailed metrics for analysis.

---

## 🧠 RL Agent Training Workflow

### Step-by-Step Training Process:

1. **Initial Baseline:**
   ```bash
   k6 run --env BASE_URL=http://your-app:5000 k6-spike.js
   ```
   - Verify basic functionality
   - Establish baseline metrics

2. **Main Training:**
   ```bash
   k6 run --env BASE_URL=http://your-app:5000 k6-autoscaler-training.js
   ```
   - Run during agent training phase
   - Monitor InfluxDB/Prometheus for metrics
   - Check agent logs in `agent/logs/`

3. **Edge Case Training:**
   ```bash
   k6 run --env BASE_URL=http://your-app:5000 k6-autoscaler-edge-cases.js
   ```
   - Train on challenging scenarios
   - Improve robustness

4. **Weekly Pattern Learning:**
   ```bash
   k6 run --env BASE_URL=http://your-app:5000 k6-autoscaler-weekly.js
   ```
   - Learn time-based patterns
   - Improve predictive scaling

5. **Validate Training:**
   - Review metrics in monitoring tools
   - Analyze agent reward patterns
   - Check scaling decision quality
   - Verify SLA compliance

---

## 🎯 Real-World Application Mapping

### What Makes These Tests Realistic?

1. **Traffic Patterns:**
   - Based on actual production patterns (morning peaks, lunch dips, etc.)
   - Variable load intensities
   - Realistic user behavior (random delays, mixed workloads)

2. **Request Distribution:**
   - 30-45% CPU-intensive operations
   - 20-35% Memory-intensive operations
   - 20-50% Basic/lightweight requests
   - Adjusts based on time and load

3. **Sleep Times:**
   - Variable based on load phase
   - Simulates realistic user think time
   - Accounts for different operation types

4. **Load Characteristics:**
   - Gradual ramps (not instant)
   - Natural fluctuations
   - Periodic patterns
   - Unexpected spikes

---

## 🔧 Customization

### Adjusting Test Parameters

**Increase Load:**
```javascript
// In any test file, modify stages
{ duration: '2m', target: 50 },  // Change 50 to higher value
```

**Change Request Distribution:**
```javascript
// In k6-autoscaler-training.js, modify REQUEST_PATTERNS
NORMAL: { cpu: 0.30, memory: 0.20, basic: 0.50 },
// Adjust percentages (must sum to 1.0)
```

**Adjust Thresholds:**
```javascript
export const options = {
  thresholds: {
    http_req_duration: ['p(95)<8000'], // Change 8000 to your SLA
    errors: ['rate<0.12'],              // Change error tolerance
  },
};
```

**Modify Endpoints:**
```javascript
// Change API endpoints to match your application
const res = http.get(`${BASE_URL}/your/custom/endpoint`);
```

---

## 📈 Monitoring During Tests

### InfluxDB Queries
```sql
-- Check autoscaling decisions
SELECT * FROM "autoscaling_metrics"
WHERE time > now() - 1h

-- View replica changes
SELECT "replica_state" FROM "autoscaling_metrics"
WHERE time > now() - 1h

-- Check rewards
SELECT "reward" FROM "autoscaling_metrics"
WHERE time > now() - 1h
```

### Prometheus Queries
```promql
# CPU usage
avg(rate(container_cpu_usage_seconds_total[5m])) by (pod)

# Memory usage
avg(container_memory_usage_bytes) by (pod)

# Request rate
rate(http_requests_total[5m])

# Response time
histogram_quantile(0.95, http_request_duration_seconds_bucket)
```

### Kubernetes Monitoring
```bash
# Watch pod scaling
watch kubectl get pods -n default

# Check HPA status
kubectl get hpa

# View pod metrics
kubectl top pods
```

---

## 🐛 Troubleshooting

### High Error Rates
- Check if pods are scaling appropriately
- Verify resource limits in deployment
- Check network connectivity
- Review application logs

### Slow Response Times
- Increase replica limits in environment
- Adjust CPU/memory requests
- Check for resource contention
- Review application performance

### Test Failures
```bash
# Verify connectivity
curl http://your-app:5000/api

# Check DNS resolution
nslookup your-app.default.svc.cluster.local

# Test with minimal load first
k6 run --vus 1 --duration 30s k6-spike.js
```

---

## 📚 Best Practices

1. **Always start with low load** - Validate before scaling up
2. **Monitor during tests** - Watch InfluxDB/Prometheus in real-time
3. **Run tests in sequence** - Allow cooldown between tests
4. **Save test results** - Keep JSON summaries for comparison
5. **Document changes** - Note any configuration changes
6. **Validate metrics** - Ensure data is being collected
7. **Use realistic load** - Don't exceed production expectations by too much

---

## 🎓 Training Tips for RL Agent

### Reward Function Tuning:
- Monitor reward values during tests
- Adjust weights in `environment.py`:
  - `response_time_weight`
  - `cpu_memory_weight`
  - `cost_weight`

### Training Iterations:
- Start with `iteration=100` for quick validation
- Increase to `iteration=1000+` for production training
- Use longer tests for better convergence

### Data Collection:
- Ensure InfluxDB is capturing all metrics
- Verify data quality before long training runs
- Check for any missing data points

---

## 📞 Support & Contribution

For issues or improvements:
1. Check logs in `agent/logs/`
2. Review test output and JSON summaries
3. Verify monitoring data in InfluxDB/Prometheus
4. Document any issues or suggestions

---

## 📝 Summary

| Test File | Duration | Purpose | Use Case |
|-----------|----------|---------|----------|
| `k6-autoscaler-training.js` | 60 min | Daily patterns | Primary training |
| `k6-autoscaler-edge-cases.js` | 40 min | Edge cases | Robustness |
| `k6-autoscaler-weekly.js` | 50 min | Weekly patterns | Time-series |
| `k6-cpu-stress.js` | 8 min | CPU stress | Validation |
| `k6-memory-stress.js` | 8 min | Memory stress | Validation |
| `k6-spike.js` | 3 min | Quick spike | Quick check |

**Total comprehensive training time:** ~2.5 hours

---

**Happy Training! 🚀🤖**
