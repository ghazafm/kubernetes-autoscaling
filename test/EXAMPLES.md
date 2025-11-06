# k6 Test Examples - Quick Start

This guide provides ready-to-use commands for running k6 tests with different durations and configurations.

## 📋 Quick Command Reference

### Basic Tests (Default Duration)

```bash
# Training test - 60 minutes
./run-k6.sh training

# Edge cases test - 40 minutes
./run-k6.sh edge

# Weekly simulation - 50 minutes
./run-k6.sh weekly

# Quick validation - 5 minutes
./run-k6.sh quick

# All training tests - ~2.5 hours
./run-k6.sh all
```

### Short Duration Tests (For Quick Validation)

```bash
# 5-minute training test (0.083x multiplier = 1/12 of default duration)
DURATION_MULTIPLIER=0.083 ./run-k6.sh training

# 10-minute edge cases test
DURATION_MULTIPLIER=0.25 ./run-k6.sh edge

# 15-minute weekly simulation
DURATION_MULTIPLIER=0.3 ./run-k6.sh weekly
```

### Medium Duration Tests (Few Hours)

```bash
# 2-hour training test (2x multiplier)
DURATION_MULTIPLIER=2 ./run-k6.sh training

# 4-hour comprehensive test
DURATION_MULTIPLIER=4 ./run-k6.sh training

# 6-hour edge cases test
DURATION_MULTIPLIER=6 ./run-k6.sh edge
```

### Extended Duration Tests (Days)

```bash
# 1-day training (24x multiplier = 24 hours)
DURATION_MULTIPLIER=24 ./run-k6.sh training

# 2-day edge cases test
DURATION_MULTIPLIER=48 ./run-k6.sh edge

# 3-day weekly simulation
DURATION_MULTIPLIER=72 ./run-k6.sh weekly

# 1 week of daily patterns (7x repetition)
DURATION_MULTIPLIER=24 CYCLE_COUNT=7 ./run-k6.sh training
```

### Long-Term Tests (Weeks/Months)

```bash
# 1 week continuous training
DURATION_MULTIPLIER=168 ./run-k6.sh training

# 2 weeks of edge cases
DURATION_MULTIPLIER=336 ./run-k6.sh edge

# 1 month of daily patterns (30 days)
DURATION_MULTIPLIER=24 CYCLE_COUNT=30 ./run-k6.sh training

# 1 month of weekly patterns (4 weeks)
DURATION_MULTIPLIER=24 CYCLE_COUNT=4 ./run-k6.sh weekly
```

## 🚀 Background Execution

For long-running tests, use background execution to prevent interruption:

### Using nohup (Simple)

```bash
# Run in background, log to file
nohup env DURATION_MULTIPLIER=24 ./run-k6.sh training > training.log 2>&1 &

# Check status
tail -f training.log

# Get process ID
echo $!

# Kill if needed
kill <PID>
```

### Using screen (Recommended for SSH)

```bash
# Start new screen session
screen -S k6-training

# Run test inside screen
DURATION_MULTIPLIER=168 ./run-k6.sh training

# Detach: Press Ctrl+A, then D

# Reattach later
screen -r k6-training

# List all sessions
screen -ls
```

### Using tmux (Advanced)

```bash
# Start new tmux session
tmux new -s k6-training

# Run test inside tmux
DURATION_MULTIPLIER=168 CYCLE_COUNT=4 ./run-k6.sh training

# Detach: Press Ctrl+B, then D

# Reattach later
tmux attach -t k6-training

# List all sessions
tmux ls
```

## 🎯 Real-World Training Scenarios

### Quick Development Testing

```bash
# 5-minute smoke test before committing
DURATION_MULTIPLIER=0.083 ./run-k6.sh training

# 15-minute validation before deployment
DURATION_MULTIPLIER=0.25 ./run-k6.sh all
```

### Initial Model Training (1-3 Days)

```bash
# Day 1: Training patterns
nohup env DURATION_MULTIPLIER=24 ./run-k6.sh training > day1.log 2>&1 &

# Day 2: Edge cases (after Day 1 completes)
nohup env DURATION_MULTIPLIER=24 ./run-k6.sh edge > day2.log 2>&1 &

# Day 3: Weekly patterns
nohup env DURATION_MULTIPLIER=24 ./run-k6.sh weekly > day3.log 2>&1 &
```

### Extended Model Training (1 Week)

```bash
# Screen session for 1-week training
screen -S training-week

# Run 7 days of training patterns
DURATION_MULTIPLIER=24 CYCLE_COUNT=7 ./run-k6.sh training

# Detach and let it run (Ctrl+A, D)
```

### Production-Like Training (1 Month)

```bash
# Tmux session for month-long training
tmux new -s training-month

# Run 30 days of training data
DURATION_MULTIPLIER=24 CYCLE_COUNT=30 ./run-k6.sh training

# Detach and monitor periodically (Ctrl+B, D)
```

## 📊 Monitoring Long-Running Tests

### Check Test Progress

```bash
# Monitor live logs
tail -f training.log

# Check last 100 lines
tail -n 100 training.log

# Search for errors
grep -i error training.log
grep -i fail training.log

# Check summary statistics
grep "http_req_duration" training.log
```

### Monitor System Resources

```bash
# Watch CPU and Memory
watch -n 5 kubectl top nodes
watch -n 5 kubectl top pods

# Monitor autoscaling
kubectl get hpa -w

# Check pod count
watch -n 10 "kubectl get pods | grep running | wc -l"
```

### Monitor InfluxDB Data

```bash
# Check data points collected
influx -database k6 -execute "SELECT COUNT(*) FROM http_reqs"

# Check latest timestamp
influx -database k6 -execute "SELECT * FROM http_reqs ORDER BY time DESC LIMIT 1"

# Check data rate (points per minute)
influx -database k6 -execute "SELECT COUNT(*) FROM http_reqs WHERE time > now() - 1m"
```

### Monitor with Grafana

```bash
# Port forward to Grafana
kubectl port-forward svc/grafana 3000:3000

# Open browser to http://localhost:3000
# Look for k6 dashboard
# Monitor:
# - Request rate (RPS)
# - Response time (p95, p99)
# - Error rate
# - Active VUs
```

## 🔧 Customization Examples

### Custom Target URL

```bash
# Test against staging environment
BASE_URL=http://staging.example.com:5000 ./run-k6.sh training

# Test against production
BASE_URL=https://api.example.com ./run-k6.sh edge
```

### Combined Parameters

```bash
# Custom URL + extended duration
BASE_URL=http://staging:5000 DURATION_MULTIPLIER=24 ./run-k6.sh training

# Custom URL + cycles + background execution
nohup env BASE_URL=http://prod:5000 DURATION_MULTIPLIER=24 CYCLE_COUNT=7 \
  ./run-k6.sh training > prod-training.log 2>&1 &
```

### Running Multiple Tests Sequentially

```bash
#!/bin/bash
# multi-day-training.sh

echo "Starting 3-day training sequence..."

echo "Day 1: Training patterns"
DURATION_MULTIPLIER=24 ./run-k6.sh training
sleep 300  # 5-minute cooldown

echo "Day 2: Edge cases"
DURATION_MULTIPLIER=24 ./run-k6.sh edge
sleep 300

echo "Day 3: Weekly patterns"
DURATION_MULTIPLIER=24 ./run-k6.sh weekly

echo "Training sequence complete!"
```

### Running Multiple Tests in Parallel (Advanced)

```bash
#!/bin/bash
# parallel-training.sh

echo "Starting parallel training..."

# Start all tests in background
DURATION_MULTIPLIER=24 ./run-k6.sh training > training.log 2>&1 &
PID1=$!

DURATION_MULTIPLIER=24 ./run-k6.sh edge > edge.log 2>&1 &
PID2=$!

DURATION_MULTIPLIER=24 ./run-k6.sh weekly > weekly.log 2>&1 &
PID3=$!

echo "Started 3 tests in parallel"
echo "Training PID: $PID1"
echo "Edge Cases PID: $PID2"
echo "Weekly PID: $PID3"

# Wait for all to complete
wait $PID1 $PID2 $PID3

echo "All tests complete!"
```

## 📈 Expected Data Collection

### Default Duration Tests

| Test | Duration | Data Points* | Storage |
|------|----------|-------------|---------|
| Training | 60 min | ~360K | ~50 MB |
| Edge Cases | 40 min | ~240K | ~35 MB |
| Weekly | 50 min | ~300K | ~45 MB |
| Quick | 5 min | ~30K | ~5 MB |

\* Approximate, depends on VU count and request rate

### Extended Duration Tests (1 Week)

| Test | Duration | Data Points* | Storage |
|------|----------|-------------|---------|
| Training (24x) | 24 hours | ~8.6M | ~1.2 GB |
| Training (168x) | 7 days | ~60M | ~8.4 GB |
| Training (24x, 7 cycles) | 7 days | ~60M | ~8.4 GB |

## ⚠️ Important Notes

1. **InfluxDB Retention**: Ensure your InfluxDB has adequate storage and retention policy:
   ```bash
   # Check retention policy
   influx -database k6 -execute "SHOW RETENTION POLICIES"

   # Update retention (example: 30 days)
   influx -database k6 -execute "ALTER RETENTION POLICY autogen ON k6 DURATION 30d"
   ```

2. **Disk Space**: Monitor disk space during long tests:
   ```bash
   # Check disk usage
   df -h

   # Monitor InfluxDB data directory
   du -sh /var/lib/influxdb/
   ```

3. **Network Stability**: For multi-day tests, ensure stable network connection. Use screen/tmux to prevent disconnection issues.

4. **Cooldown Periods**: When running multiple tests sequentially, add cooldown periods:
   ```bash
   DURATION_MULTIPLIER=24 ./run-k6.sh training
   sleep 600  # 10-minute cooldown
   DURATION_MULTIPLIER=24 ./run-k6.sh edge
   ```

5. **Resource Monitoring**: Set up alerts for:
   - High CPU/Memory usage
   - Disk space warnings
   - Pod crash loops
   - High error rates

## 🎓 Best Practices

1. **Start Small**: Always validate with a short test first
   ```bash
   DURATION_MULTIPLIER=0.083 ./run-k6.sh training  # 5 minutes
   ```

2. **Use Screen/Tmux**: For tests > 1 hour, always use screen or tmux

3. **Monitor Initially**: Stay connected for first 15-30 minutes to ensure test starts correctly

4. **Log Everything**: Always redirect output to log files for long tests

5. **Document**: Keep notes on test configurations and any observed issues

6. **Backup Data**: Periodically backup InfluxDB data during long tests:
   ```bash
   influxd backup -database k6 /backup/k6-$(date +%Y%m%d)
   ```

## 🆘 Troubleshooting

### Test Won't Start

```bash
# Check k6 installation
k6 version

# Verify environment variables
echo $BASE_URL
echo $DURATION_MULTIPLIER
echo $CYCLE_COUNT

# Check file permissions
ls -la *.js
chmod +x run-k6.sh
```

### High Error Rate

```bash
# Check application health
curl $BASE_URL/api

# Check pod status
kubectl get pods
kubectl describe pod <pod-name>

# Check logs
kubectl logs <pod-name>
```

### Out of Memory

```bash
# Check resource usage
kubectl top nodes
kubectl top pods

# Scale up if needed
kubectl scale deployment <app> --replicas=<count>

# Adjust test load
DURATION_MULTIPLIER=0.5 ./run-k6.sh training  # Reduce load
```

## 📚 Additional Resources

- [EXTENDED_DURATION_GUIDE.md](./EXTENDED_DURATION_GUIDE.md) - Detailed duration configuration guide
- [README.md](./README.md) - Complete test documentation
- [TESTS_SUMMARY.md](./TESTS_SUMMARY.md) - Test scenarios and capabilities
- [QUICK_REFERENCE.md](./QUICK_REFERENCE.md) - Visual comparison and quick reference

---

**Need Help?** Check the troubleshooting section in [README.md](./README.md) or [EXTENDED_DURATION_GUIDE.md](./EXTENDED_DURATION_GUIDE.md).
