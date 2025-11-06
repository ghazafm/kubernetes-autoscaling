# Extended Duration Training Guide

## 🚀 Overview

All three main training tests now support **customizable duration** through environment variables, allowing you to run tests for **hours, days, or even weeks** for comprehensive RL agent training.

## ⚙️ Configuration Variables

### 1. `DURATION_MULTIPLIER`
Multiplies the base duration of each test phase.

**Examples:**
```bash
DURATION_MULTIPLIER=1    # Default (1x speed)
DURATION_MULTIPLIER=24   # 24x longer (1 hour → 24 hours)
DURATION_MULTIPLIER=168  # 168x longer (1 hour → 1 week)
```

### 2. `CYCLE_COUNT`
Number of times to repeat the entire pattern.

**Examples:**
```bash
CYCLE_COUNT=1    # Run pattern once (default)
CYCLE_COUNT=10   # Repeat pattern 10 times
CYCLE_COUNT=100  # Repeat pattern 100 times
```

## 📊 Test Duration Reference

### k6-autoscaler-training.js (Base: 60 minutes)

| Multiplier | Cycle Count | Duration per Cycle | Total Duration | Use Case |
|------------|-------------|-------------------|----------------|----------|
| 1x | 1 | 60 min (1h) | 1 hour | Quick validation |
| 6x | 1 | 360 min (6h) | 6 hours | Extended training |
| 12x | 1 | 720 min (12h) | 12 hours | Half-day training |
| 24x | 1 | 1440 min (24h) | 1 day | Full day training |
| 24x | 2 | 1440 min (24h) | 2 days | Multi-day training |
| 24x | 7 | 1440 min (24h) | 1 week | Weekly training |
| 168x | 1 | 10080 min (168h) | 1 week | Direct week-long |

### k6-autoscaler-edge-cases.js (Base: 40 minutes)

| Multiplier | Cycle Count | Duration per Cycle | Total Duration | Use Case |
|------------|-------------|-------------------|----------------|----------|
| 1x | 1 | 40 min | 40 min | Quick stress test |
| 12x | 1 | 480 min (8h) | 8 hours | Day stress test |
| 24x | 1 | 960 min (16h) | 16 hours | Extended stress |
| 36x | 1 | 1440 min (24h) | 1 day | Full day stress |
| 36x | 7 | 1440 min (24h) | 1 week | Weekly stress |

### k6-autoscaler-weekly.js (Base: 50 minutes)

| Multiplier | Cycle Count | Duration per Cycle | Total Duration | Use Case |
|------------|-------------|-------------------|----------------|----------|
| 1x | 1 | 50 min | 50 min | Quick week sim |
| 12x | 1 | 600 min (10h) | 10 hours | Extended week |
| 20.16x | 1 | 1008 min (16.8h) | ~17 hours | Day per "day" |
| 20.16x | 4 | 1008 min (16.8h) | ~2.8 days | Month simulation |
| 201.6x | 1 | 10080 min (168h) | 1 week | Full week |

## 🎯 Usage Examples

### 1. Quick Validation (Default)
```bash
# Run with default settings (1x multiplier, 1 cycle)
k6 run --env BASE_URL=http://your-app:5000 k6-autoscaler-training.js
```

### 2. Full Day Training
```bash
# Training test: 24 hours
k6 run --env BASE_URL=http://your-app:5000 \
       --env DURATION_MULTIPLIER=24 \
       --env CYCLE_COUNT=1 \
       k6-autoscaler-training.js
```

### 3. Multi-Day Training (2 days)
```bash
# Run 2 full day cycles
k6 run --env BASE_URL=http://your-app:5000 \
       --env DURATION_MULTIPLIER=24 \
       --env CYCLE_COUNT=2 \
       k6-autoscaler-training.js
```

### 4. One Week Continuous Training
```bash
# Run 7 daily cycles
k6 run --env BASE_URL=http://your-app:5000 \
       --env DURATION_MULTIPLIER=24 \
       --env CYCLE_COUNT=7 \
       k6-autoscaler-training.js

# OR use direct weekly multiplier
k6 run --env BASE_URL=http://your-app:5000 \
       --env DURATION_MULTIPLIER=168 \
       --env CYCLE_COUNT=1 \
       k6-autoscaler-training.js
```

### 5. Weekly Pattern - Extended
```bash
# Each simulated "day" takes 2.4 hours (real day scale)
k6 run --env BASE_URL=http://your-app:5000 \
       --env DURATION_MULTIPLIER=20.16 \
       --env CYCLE_COUNT=1 \
       k6-autoscaler-weekly.js
```

### 6. Month-Long Training
```bash
# 30 days of continuous training
k6 run --env BASE_URL=http://your-app:5000 \
       --env DURATION_MULTIPLIER=24 \
       --env CYCLE_COUNT=30 \
       k6-autoscaler-training.js
```

### 7. Edge Cases - Week-Long Stress
```bash
# Repeat edge cases for 7 days
k6 run --env BASE_URL=http://your-app:5000 \
       --env DURATION_MULTIPLIER=36 \
       --env CYCLE_COUNT=7 \
       k6-autoscaler-edge-cases.js
```

## 🔧 Running in Background

For long-running tests, use `nohup` or `screen`:

### Using nohup
```bash
# Run in background, output to file
nohup k6 run --env BASE_URL=http://your-app:5000 \
             --env DURATION_MULTIPLIER=24 \
             --env CYCLE_COUNT=7 \
             k6-autoscaler-training.js > training-week.log 2>&1 &

# Check progress
tail -f training-week.log
```

### Using screen
```bash
# Create named screen session
screen -S k6-training

# Inside screen, run test
k6 run --env BASE_URL=http://your-app:5000 \
       --env DURATION_MULTIPLIER=24 \
       --env CYCLE_COUNT=7 \
       k6-autoscaler-training.js

# Detach: Ctrl+A, then D
# Reattach: screen -r k6-training
```

### Using tmux
```bash
# Create named tmux session
tmux new -s k6-training

# Inside tmux, run test
k6 run --env BASE_URL=http://your-app:5000 \
       --env DURATION_MULTIPLIER=24 \
       --env CYCLE_COUNT=7 \
       k6-autoscaler-training.js

# Detach: Ctrl+B, then D
# Reattach: tmux attach -t k6-training
```

## 📅 Recommended Training Schedules

### Week 1: Initial Training
```bash
# Day 1-2: Quick iterations (default duration)
./run-k6.sh training   # 1 hour
./run-k6.sh edge       # 40 min
./run-k6.sh weekly     # 50 min

# Day 3-4: Extended iterations (6x multiplier)
k6 run --env DURATION_MULTIPLIER=6 --env CYCLE_COUNT=1 k6-autoscaler-training.js

# Day 5-7: Full day iteration
k6 run --env DURATION_MULTIPLIER=24 --env CYCLE_COUNT=1 k6-autoscaler-training.js
```

### Week 2: Intensive Training
```bash
# Monday-Friday: Multi-day continuous training
k6 run --env DURATION_MULTIPLIER=24 --env CYCLE_COUNT=5 k6-autoscaler-training.js

# Weekend: Edge case training
k6 run --env DURATION_MULTIPLIER=36 --env CYCLE_COUNT=2 k6-autoscaler-edge-cases.js
```

### Week 3: Weekly Patterns
```bash
# Run full week simulation
k6 run --env DURATION_MULTIPLIER=20 --env CYCLE_COUNT=1 k6-autoscaler-weekly.js

# Or repeat weekly pattern 4 times (1 month)
k6 run --env DURATION_MULTIPLIER=20 --env CYCLE_COUNT=4 k6-autoscaler-weekly.js
```

### Week 4: Production-Like Training
```bash
# Continuous 7-day training
k6 run --env DURATION_MULTIPLIER=168 --env CYCLE_COUNT=1 k6-autoscaler-training.js

# OR
k6 run --env DURATION_MULTIPLIER=24 --env CYCLE_COUNT=7 k6-autoscaler-training.js
```

## 📊 Monitoring Extended Tests

### 1. Check Test Progress
```bash
# If using nohup
tail -f training-week.log | grep -E "stage|iteration|VUs"

# Watch k6 process
watch -n 5 'ps aux | grep k6'
```

### 2. Monitor Application
```bash
# Watch pod scaling
watch -n 10 'kubectl get pods -n default'

# Monitor HPA
watch -n 10 'kubectl get hpa'

# Check resource usage
watch -n 10 'kubectl top pods'
```

### 3. Monitor Metrics in InfluxDB
```bash
# Query recent metrics
curl -G 'http://localhost:8086/query?pretty=true' \
  --data-urlencode "db=mydb" \
  --data-urlencode "q=SELECT * FROM autoscaling_metrics WHERE time > now() - 1h"
```

### 4. Check Prometheus
```bash
# Open Prometheus UI
kubectl port-forward svc/prometheus 9090:9090

# Then visit: http://localhost:9090
```

## 💡 Best Practices for Extended Training

### 1. Start Small, Scale Up
```bash
# Week 1: Short iterations
DURATION_MULTIPLIER=1

# Week 2: Medium iterations
DURATION_MULTIPLIER=6

# Week 3: Long iterations
DURATION_MULTIPLIER=24

# Week 4: Very long iterations
DURATION_MULTIPLIER=168
```

### 2. Monitor Resource Usage
- Check CPU/Memory on k6 machine
- Monitor network bandwidth
- Watch for disk space (logs/summaries)
- Monitor database size (InfluxDB)

### 3. Use Separate Test Runs
Don't run all tests simultaneously:
```bash
# ❌ Bad: All at once
./run-k6.sh all

# ✅ Good: Sequential with extended durations
k6 run --env DURATION_MULTIPLIER=24 k6-autoscaler-training.js
# Wait for completion, then:
k6 run --env DURATION_MULTIPLIER=36 k6-autoscaler-edge-cases.js
```

### 4. Save Summaries
```bash
# Summaries are automatically saved with timestamps:
# - training-summary-2025-11-07-14-30.json
# - edge-cases-summary-2025-11-07-16-45.json
# - weekly-simulation-summary-2025-11-07-20-15.json

# Archive them after each run
mkdir -p results/week-1
mv *-summary-*.json results/week-1/
```

### 5. Checkpoint Your Agent
```bash
# Save model checkpoints periodically
# In your agent code (agent/train.py):
# - Save every N hours
# - Save after each cycle completion
# - Keep backup copies
```

## 🚨 Troubleshooting Long Tests

### Test Stopped Unexpectedly
```bash
# Check if k6 process is still running
ps aux | grep k6

# Check system resources
free -h
df -h

# Check for OOM kills
dmesg | grep -i "out of memory"
```

### High Resource Usage
```bash
# Reduce VU count in test files
# Or increase system resources

# Monitor k6 resource usage
top -p $(pgrep k6)
```

### Network Issues
```bash
# Check connection to app
curl http://your-app:5000/api

# Check DNS resolution
nslookup your-app.default.svc.cluster.local

# Increase timeout if needed (in test files)
timeout: '60s'
```

### InfluxDB Full
```bash
# Check database size
du -sh /var/lib/influxdb/data/

# Retention policy
influx -execute 'SHOW RETENTION POLICIES ON mydb'

# Create shorter retention for training data
influx -execute 'CREATE RETENTION POLICY "30days" ON "mydb" DURATION 30d REPLICATION 1 DEFAULT'
```

## 📈 Expected Results for Extended Training

### Short Tests (1x multiplier)
- **Data Points:** 5,000-10,000
- **Training Episodes:** 10-20
- **Use Case:** Quick validation

### Day-Long Tests (24x multiplier)
- **Data Points:** 120,000-240,000
- **Training Episodes:** 240-480
- **Use Case:** Solid training dataset

### Week-Long Tests (168x multiplier or 24x × 7 cycles)
- **Data Points:** 840,000-1,680,000
- **Training Episodes:** 1,680-3,360
- **Use Case:** Production-ready training

### Month-Long Tests (24x × 30 cycles)
- **Data Points:** 3,600,000-7,200,000
- **Training Episodes:** 7,200-14,400
- **Use Case:** Comprehensive training

## 🎓 Training Effectiveness

### Metrics to Track Over Time

```bash
# Average reward should increase
SELECT MEAN("reward") FROM "autoscaling_metrics"
WHERE time > now() - 7d
GROUP BY time(1h)

# Error rate should decrease
SELECT MEAN("errors") FROM "autoscaling_metrics"
WHERE time > now() - 7d
GROUP BY time(1h)

# Response time should stabilize
SELECT MEAN("response_time") FROM "autoscaling_metrics"
WHERE time > now() - 7d
GROUP BY time(1h)

# Replica efficiency
SELECT MEAN("replica_state"), MEAN("cpu_usage"), MEAN("memory_usage")
FROM "autoscaling_metrics"
WHERE time > now() - 7d
GROUP BY time(1h)
```

## 🎯 Quick Command Reference

```bash
# 1 day training
k6 run --env DURATION_MULTIPLIER=24 --env CYCLE_COUNT=1 k6-autoscaler-training.js

# 1 week training (7 daily cycles)
k6 run --env DURATION_MULTIPLIER=24 --env CYCLE_COUNT=7 k6-autoscaler-training.js

# 1 week training (direct)
k6 run --env DURATION_MULTIPLIER=168 --env CYCLE_COUNT=1 k6-autoscaler-training.js

# 1 month training (30 daily cycles)
k6 run --env DURATION_MULTIPLIER=24 --env CYCLE_COUNT=30 k6-autoscaler-training.js

# Background 1 week training
nohup k6 run --env DURATION_MULTIPLIER=24 --env CYCLE_COUNT=7 \
  k6-autoscaler-training.js > week-training.log 2>&1 &
```

## 🎉 Success Indicators

After extended training, you should see:

✅ **Increasing Reward Trend** - Agent learning optimal policies
✅ **Decreasing Error Rate** - Fewer failed requests over time
✅ **Stable Response Times** - Consistent performance under load
✅ **Efficient Resource Usage** - Lower average replicas with same SLA
✅ **Predictive Scaling** - Agent anticipates load changes
✅ **Faster Convergence** - Less exploration, more exploitation
✅ **Better Cost Optimization** - Aggressive scale-down when safe
✅ **Smooth Scaling Transitions** - Less oscillation in replica counts

---

**Now you can train your RL autoscaler for as long as needed!** 🚀🤖
