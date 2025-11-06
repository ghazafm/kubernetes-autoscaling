# K6 Test Files - Quick Reference

## 📊 Visual Comparison

```
┌─────────────────────────────────────────────────────────────────────┐
│                    TEST FILE COMPARISON                              │
├─────────────────────────────────────────────────────────────────────┤
│                                                                      │
│  k6-autoscaler-training.js (⭐ RECOMMENDED FOR TRAINING)            │
│  ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━  │
│  Duration: 60 minutes                                                │
│  Load: 0 → 50 VUs                                                   │
│  Focus: Daily traffic patterns                                       │
│  Scenarios: 10 phases (morning → peak → night)                      │
│                                                                      │
│  Load Pattern:                                                       │
│   50 VUs │     ╭─────╮  ╭────╮                                     │
│          │    ╱       ╰──╯    ╲                                     │
│   25 VUs │   ╱                 ╲  ╱╲                               │
│          │  ╱                   ╰╯  ╲                               │
│    0 VUs ╰─╯                        ╰──                             │
│          └─────────────────────────────────→ Time                   │
│                                                                      │
├─────────────────────────────────────────────────────────────────────┤
│                                                                      │
│  k6-autoscaler-edge-cases.js (🔥 STRESS & EDGE CASES)              │
│  ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━  │
│  Duration: 40 minutes                                                │
│  Load: 0 → 80 VUs (extreme)                                         │
│  Focus: Challenging scenarios                                        │
│  Scenarios: 10 edge cases (spikes, oscillations, extremes)         │
│                                                                      │
│  Load Pattern:                                                       │
│   80 VUs │        ╭─────╮                                           │
│          │       ╱       ╲    ╱╲  ╱╲  ╱╲      ╭────╮              │
│   40 VUs │  ╱╲  ╱         ╰──╯  ╰╯  ╰╯  ╰────╯    ╲              │
│          │ ╱  ╰╯                                     ╲              │
│    0 VUs ╰╯                                           ╰─            │
│          └─────────────────────────────────→ Time                   │
│                                                                      │
├─────────────────────────────────────────────────────────────────────┤
│                                                                      │
│  k6-autoscaler-weekly.js (📅 WEEKLY PATTERNS)                      │
│  ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━  │
│  Duration: 50 minutes (~7 min per day)                              │
│  Load: 0 → 40 VUs                                                   │
│  Focus: Weekly business patterns                                     │
│  Scenarios: 7 days (Mon → Sun)                                      │
│                                                                      │
│  Load Pattern:                                                       │
│   40 VUs │           ╭────╮                                         │
│          │      ╭───╯    ╰──╮                                       │
│   20 VUs │  ╭──╯            ╰──╮  ╭─╮                              │
│          │ ╱                   ╰──╯ ╰─╮                             │
│    0 VUs ╰╯                          ╰───                           │
│          └───────────────────────────────→ Time                     │
│          Mon Tue Wed Thu Fri Sat Sun                                │
│                                                                      │
└─────────────────────────────────────────────────────────────────────┘
```

## 🎯 When to Use Each Test

### 🟢 Use `k6-autoscaler-training.js` when:
- ✅ Initial training of RL agent
- ✅ Need realistic daily patterns
- ✅ Testing gradual scaling behavior
- ✅ Learning cost optimization
- ✅ Collecting primary training data
- ✅ Validating general performance

**Best for:** Foundation training, daily operation simulation

---

### 🔴 Use `k6-autoscaler-edge-cases.js` when:
- ✅ Testing robustness
- ✅ Training for extreme scenarios
- ✅ Validating failure handling
- ✅ Testing rapid scale-up/down
- ✅ Stress testing limits
- ✅ Outlier scenario learning

**Best for:** Robustness training, stress testing, edge learning

---

### 🔵 Use `k6-autoscaler-weekly.js` when:
- ✅ Learning time-based patterns
- ✅ Weekly cycle recognition
- ✅ Predictive scaling training
- ✅ Business hour optimization
- ✅ Weekend vs weekday learning
- ✅ Long-term pattern recognition

**Best for:** Time-series learning, pattern prediction, weekly optimization

---

## 📈 Training Progression

```
┌──────────────────────────────────────────────────────────────┐
│               RECOMMENDED TRAINING SEQUENCE                   │
└──────────────────────────────────────────────────────────────┘

Week 1: Foundation
├─ Day 1-2: Quick validation
│  └─> ./run-k6.sh quick (15 min)
│
├─ Day 3-5: Basic training
│  └─> ./run-k6.sh training (60 min) × 3 runs
│
└─ Day 6-7: First analysis
   └─> Review metrics, adjust parameters

Week 2: Robustness
├─ Day 1-3: Edge case training
│  └─> ./run-k6.sh edge (40 min) × 3 runs
│
├─ Day 4-5: Combined training
│  └─> ./run-k6.sh training + edge alternating
│
└─ Day 6-7: Second analysis
   └─> Evaluate robustness improvements

Week 3: Patterns
├─ Day 1-3: Weekly pattern training
│  └─> ./run-k6.sh weekly (50 min) × 3 runs
│
├─ Day 4-5: Full suite
│  └─> ./run-k6.sh all (2.5 hours)
│
└─ Day 6-7: Final optimization
   └─> Fine-tune based on all data

Week 4: Validation
└─> Production-like extended tests
    └─> Validate learned policies
```

## 🔍 Test Characteristics Comparison

| Feature | Training | Edge Cases | Weekly |
|---------|----------|------------|--------|
| **Duration** | 60 min | 40 min | 50 min |
| **Max VUs** | 50 | 80 | 40 |
| **Scenario Count** | 10 | 10 | 7 days × multiple times |
| **Pattern Type** | Daily cycle | Extreme cases | Weekly cycle |
| **Request Mix** | Dynamic | Intensive | Time-based |
| **Error Tolerance** | 12% | 25% | 10% |
| **Focus** | General | Stress | Prediction |
| **Realism** | Very High | Moderate | Very High |
| **Difficulty** | Medium | High | Medium |
| **Training Value** | ⭐⭐⭐⭐⭐ | ⭐⭐⭐⭐ | ⭐⭐⭐⭐⭐ |

## 💡 Quick Decision Guide

```
┌─────────────────────────────────────────────────────────────┐
│  "Which test should I run?"                                  │
├─────────────────────────────────────────────────────────────┤
│                                                              │
│  Just starting?                                              │
│  └─> ./run-k6.sh quick                                      │
│                                                              │
│  First training session?                                     │
│  └─> ./run-k6.sh training                                   │
│                                                              │
│  Agent keeps failing on spikes?                              │
│  └─> ./run-k6.sh edge                                       │
│                                                              │
│  Need time-based predictions?                                │
│  └─> ./run-k6.sh weekly                                     │
│                                                              │
│  Want comprehensive training?                                │
│  └─> ./run-k6.sh all                                        │
│                                                              │
│  Need to validate everything?                                │
│  └─> ./run-k6.sh full                                       │
│                                                              │
│  Quick sanity check?                                         │
│  └─> ./run-k6.sh spike                                      │
│                                                              │
└─────────────────────────────────────────────────────────────┘
```

## 🎓 Learning Objectives by Test

### k6-autoscaler-training.js learns:
```
✓ Gradual scaling patterns
✓ Cost optimization during low load
✓ Handling lunch-time dips
✓ Flash spike response
✓ Evening scale-down timing
✓ Resource efficiency
✓ SLA maintenance
✓ Daily recurring patterns
```

### k6-autoscaler-edge-cases.js learns:
```
✓ Extreme spike handling (thundering herd)
✓ Rapid oscillation stability
✓ Cold start optimization
✓ Sustained maximum load
✓ Aggressive scale-down
✓ Recovery from extremes
✓ Dead zone efficiency
✓ Failure mode handling
```

### k6-autoscaler-weekly.js learns:
```
✓ Day-of-week patterns
✓ Weekend vs weekday behavior
✓ Mid-week peak handling (Wednesday)
✓ Friday early decline
✓ Weekly cycle prediction
✓ Time-of-day optimization
✓ Business hours patterns
✓ Recurring event handling
```

## 📊 Expected Training Data Output

```
After running ./run-k6.sh all, you'll collect:

InfluxDB Metrics:
├─ ~15,000-20,000 data points
├─ CPU, Memory, Response Time per action
├─ Reward values per state-action pair
├─ Replica counts and scaling events
└─ Tagged by scenario, phase, day

Agent Logs:
├─ Scaling decision history
├─ State-action-reward sequences
├─ Learning progression
└─ Error and success patterns

K6 Summaries:
├─ training-summary-*.json
├─ edge-cases-summary-*.json
└─ weekly-simulation-summary-*.json
```

## 🚀 Performance Expectations

```
┌──────────────────────────────────────────────────────────────┐
│                  EXPECTED RESULTS                             │
├──────────────────────────────────────────────────────────────┤
│                                                               │
│  Training Test:                                               │
│  ├─ Total Requests: 8,000-12,000                            │
│  ├─ Success Rate: >88%                                       │
│  ├─ p95 Response: <8s                                        │
│  └─ Throughput: 2-3 req/s avg                               │
│                                                               │
│  Edge Cases Test:                                             │
│  ├─ Total Requests: 6,000-10,000                            │
│  ├─ Success Rate: >75%                                       │
│  ├─ p95 Response: <15s                                       │
│  └─ Throughput: 2-4 req/s avg                               │
│                                                               │
│  Weekly Test:                                                 │
│  ├─ Total Requests: 7,000-11,000                            │
│  ├─ Success Rate: >90%                                       │
│  ├─ p95 Response: <7s                                        │
│  └─ Throughput: 2-3 req/s avg                               │
│                                                               │
└──────────────────────────────────────────────────────────────┘
```

## 🎯 Success Indicators

After successful training, monitor for:

```
✅ Increasing average reward over time
✅ Decreasing error rates
✅ Faster response to load changes
✅ More stable replica counts
✅ Better cost efficiency (lower avg replicas with same SLA)
✅ Predictive scaling (preemptive actions)
✅ Reduced SLA violations
✅ Smoother scaling transitions
```

## 📝 Quick Command Reference

```bash
# Individual tests
./run-k6.sh training    # 60 min - Daily patterns
./run-k6.sh edge        # 40 min - Edge cases
./run-k6.sh weekly      # 50 min - Weekly patterns

# Test suites
./run-k6.sh quick       # 15 min - Quick validation
./run-k6.sh all         # 2.5 hrs - All training tests
./run-k6.sh full        # 3 hrs - Complete suite

# Legacy tests
./run-k6.sh spike       # 3 min - Quick spike
./run-k6.sh cpu         # 8 min - CPU stress
./run-k6.sh memory      # 8 min - Memory stress
./run-k6.sh rl          # Original RL test

# Help
./run-k6.sh help        # Show all options
```

---

**Ready to start training your RL autoscaler!** 🚀🤖
