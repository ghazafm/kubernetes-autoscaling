# 🎯 Kubernetes Autoscaling RL Reward System

## Overview

This document explains the multi-component reward calculation system used in our Kubernetes autoscaling reinforcement learning agent. The system is designed to train an intelligent autoscaler that maintains optimal performance while being resource-efficient and stable.

## 🏗️ Architecture

The reward system consists of **6 main components** that work together to guide the RL agent toward optimal autoscaling decisions:

```
Total Reward = Performance Score
             + Resource Efficiency
             + Scaling Appropriateness
             + Cluster Health
             + Stability Bonus
             + Metrics Reliability
```

## 📊 Reward Components

### 1. 🔥 Performance Score (Primary Objective)

**Purpose**: Ensures optimal resource utilization and prevents performance degradation.

**Target Range**: 20-85% CPU and Memory usage

#### CPU Performance Scoring
```python
if cpu_usage > 100%:           # CRITICAL - Pod exceeding limits
    penalty = -(cpu_usage - 100) * 5    # Heavy penalty: -25 for 105% CPU
elif cpu_usage > 95%:          # HIGH - Near limit
    penalty = -(cpu_usage - 95) * 3     # Moderate penalty: -15 for 100% CPU
elif cpu_usage > 85%:          # Above target
    penalty = -(cpu_usage - 85) * 1.5   # Light penalty: -7.5 for 90% CPU
```

#### Memory Performance Scoring
```python
if memory_usage > 100%:        # CRITICAL - Pod exceeding limits
    penalty = -(memory_usage - 100) * 4   # Heavy penalty: -20 for 105% memory
elif memory_usage > 95%:       # HIGH - Near limit
    penalty = -(memory_usage - 95) * 2    # Moderate penalty: -10 for 100% memory
elif memory_usage > 85%:       # Above target
    penalty = -(memory_usage - 85) * 1    # Light penalty: -5 for 90% memory
```

#### Optimal Performance Rewards
```python
# Sweet spot: 20-85% CPU and 20-85% memory
if cpu_optimal AND memory_optimal:
    bonus = +25               # Best case bonus
elif cpu_optimal OR memory_optimal:
    bonus = +10               # Partial bonus

# Resource balance bonus (up to +5)
resource_balance = 1 - abs(cpu_usage - memory_usage) / 100
bonus += resource_balance * 5
```

### 2. ⚡ Resource Efficiency Score

**Purpose**: Prevents resource waste and encourages efficient resource utilization.

```python
# Waste penalty for over-provisioning
if replicas > min_replicas:
    if cpu_usage < 20%:  # Below target
        waste_factor = (20 - cpu_usage) / 20
        penalty = replicas * waste_factor * 2

    if memory_usage < 20%:  # Below target
        waste_factor = (20 - memory_usage) / 20
        penalty = replicas * waste_factor * 1.5

# Efficiency bonus for single replica handling reasonable load
if replicas == 1 AND cpu_usage > 30% AND memory_usage > 20%:
    bonus = +5
```

### 3. 🚫 Scaling Appropriateness (Action Penalty)

**Purpose**: Prevents inappropriate scaling decisions based on current system state.

#### Inappropriate Scale Down
```python
if action < 0:  # Scaling down
    if cpu_usage > 85% OR memory_usage > 85%:
        penalty = abs(action) * 2  # Heavy penalty for scaling down under load
    elif cpu_usage > 80% OR memory_usage > 80%:
        penalty = abs(action) * 1  # Light penalty for moderate load
```

#### Inappropriate Scale Up
```python
if action > 0:  # Scaling up
    if cpu_usage < 10% AND memory_usage < 10% AND replicas > 2:
        penalty = action * 1.5  # Penalty for scaling up with low usage
```

#### Large Action Penalty
```python
if abs(action) > 20:           # Large jumps (>20 replicas)
    penalty = abs(action) * 0.5
elif abs(action) > 10:         # Medium jumps (>10 replicas)
    penalty = abs(action) * 0.2
```

### 4. 🏥 Cluster Health Score

**Purpose**: Ensures system stability and prevents resource exhaustion.

#### Unschedulable Pod Penalty
```python
if unschedulable_replicas > 0:
    # Progressive penalty (not linear)
    cluster_penalty = min(50, unschedulable_replicas * 5 + unschedulable_replicas^1.5)
    penalty = -cluster_penalty
```

#### Healthy Cluster Bonus
```python
if all_pods_ready:
    bonus = +5  # Small bonus for healthy cluster
```

### 5. 🎯 Stability Bonus

**Purpose**: Encourages gradual, stable scaling and reduces oscillation.

```python
if action == 0:              # No scaling
    bonus = +2
elif 0 < abs(action) <= 5:   # Small adjustments
    bonus = +3
elif 5 < abs(action) <= 10:  # Medium adjustments
    bonus = +1
# Large adjustments get no bonus
```

### 6. 📈 Metrics Reliability

**Purpose**: Accounts for metrics API reliability and data quality.

```python
metrics_reliability = actual_metrics_fetched / ready_replicas

# Light penalty for metrics issues (not agent's fault)
if not_fetchable_replicas > 0:
    metrics_penalty = min(5, not_fetchable_replicas * 1)
    penalty = -metrics_penalty

# Bonus for high metrics reliability
if metrics_reliability >= 0.9:
    bonus = +1
```

## 🎮 Action Space

The agent operates with **5 discrete actions**:

```python
action_step = 25  # Configured in environment
actions = [-2, -1, 0, +1, +2]  # Relative replica changes

# Raw action mapping:
# 23 -> -2 replicas
# 24 -> -1 replica
# 25 -> 0 replicas (no change)
# 26 -> +1 replica
# 27 -> +2 replicas
```

## 📋 Observation Space

The agent observes **9-dimensional state**:

```python
observation = [
    cpu_usage,              # 0-100% (current CPU utilization)
    memory_usage,           # 0-100% (current memory utilization)
    cpu_available,          # 0-100% (cluster CPU availability)
    memory_available,       # 0-100% (cluster memory availability)
    current_replicas,       # 1-500 (current number of replicas)
    unschedulable_replicas, # 0-100 (pods that cannot be scheduled)
    replica_trend,          # -1 to +1 (recent scaling trend)
    time_since_last_scale,  # 0-1 (normalized time since last action)
    resource_pressure_score # 0-1 (combined resource pressure)
]
```

## 📊 Example Reward Scenarios

### Scenario 1: Optimal Performance ✅
```
State: CPU=60%, Memory=55%, Replicas=5, Action=0
- Performance: +25 (optimal) +3 (balance) = +28
- Efficiency: 0 (no waste)
- Scaling: 0 (appropriate)
- Cluster: +5 (healthy)
- Stability: +2 (no action)
- Metrics: +1 (reliable)
Total Reward: +36
```

### Scenario 2: Under Load, Appropriate Scale Up ⚡
```
State: CPU=90%, Memory=85%, Replicas=3, Action=+2
- Performance: -7.5 (above target)
- Efficiency: 0
- Scaling: 0 (appropriate scale up)
- Cluster: +5 (healthy)
- Stability: +3 (small action)
- Metrics: +1 (reliable)
Total Reward: +1.5
```

### Scenario 3: Over-provisioned, Scale Down 📉
```
State: CPU=15%, Memory=20%, Replicas=10, Action=-3
- Performance: +10 (memory optimal)
- Efficiency: -10 (CPU waste with many replicas)
- Scaling: 0 (appropriate scale down)
- Cluster: +5 (healthy)
- Stability: +1 (medium action)
- Metrics: +1 (reliable)
Total Reward: +7
```

### Scenario 4: Bad Decision - Scale Down Under Load ❌
```
State: CPU=95%, Memory=90%, Replicas=2, Action=-1
- Performance: -15 (high CPU) -10 (high memory) = -25
- Efficiency: 0
- Scaling: -2 (inappropriate scale down)
- Cluster: +5 (healthy)
- Stability: +3 (small action)
- Metrics: +1 (reliable)
Total Reward: -18
```

### Scenario 5: Cluster Resource Exhaustion 🚨
```
State: CPU=80%, Memory=75%, Replicas=15, Action=+5, Unschedulable=3
- Performance: +10 (partial optimal)
- Efficiency: 0
- Scaling: -1 (large action penalty)
- Cluster: -22.4 (unschedulable penalty)
- Stability: 0 (large action)
- Metrics: -2 (some metrics unavailable)
Total Reward: -15.4
```

## 🎓 Design Principles

### 1. Hierarchical Priorities
- **Performance > Efficiency > Stability**
- Critical performance issues get heaviest penalties
- Resource efficiency is secondary concern
- Stability encourages gradual learning

### 2. Anti-Oscillation
- Stability bonuses for small/no actions
- Progressive penalties for large actions
- Time-based considerations in observations

### 3. Contextual Actions
- Same action gets different rewards based on system state
- Scaling up under high load = good
- Scaling up under low load = bad

### 4. Progressive Learning
- Graduated penalties instead of binary rewards
- Allows agent to learn nuanced policies
- Prevents cliff effects in learning

### 5. Multi-objective Optimization
- Balances performance, efficiency, and stability
- No single metric dominates decision making
- Encourages well-rounded policies

## 🔧 Configuration Constants

```python
# Reward calculation thresholds
CPU_LIMIT_THRESHOLD = 100.0       # Critical CPU threshold
MEMORY_LIMIT_THRESHOLD = 100.0    # Critical memory threshold
CPU_WARNING_THRESHOLD = 95.0      # High CPU warning
MEMORY_WARNING_THRESHOLD = 95.0   # High memory warning
MODERATE_CPU_THRESHOLD = 80.0     # Moderate CPU usage
MODERATE_MEMORY_THRESHOLD = 80.0  # Moderate memory usage

# Efficiency thresholds
MIN_EFFICIENT_CPU = 30.0          # Minimum efficient CPU for single replica
MIN_EFFICIENT_MEMORY = 20.0       # Minimum efficient memory for single replica
LOW_USAGE_CPU = 10.0              # Low CPU usage threshold
LOW_USAGE_MEMORY = 10.0           # Low memory usage threshold

# Action thresholds
LARGE_ACTION_THRESHOLD = 20       # Large scaling action
MEDIUM_ACTION_THRESHOLD = 10      # Medium scaling action
SMALL_ACTION_THRESHOLD = 5        # Small scaling action

# Metrics reliability
MIN_METRICS_RELIABILITY = 0.9    # Minimum acceptable metrics reliability
```

## 🚀 Training Integration

### Environment Setup
```python
env = K8sAutoscalerEnv(
    min_replicas=1,
    max_replicas=50,
    iteration=100,
    action_step=25,  # 5 discrete actions
    verbose=True,    # Enable reward breakdown logging
    timeout=120,     # Stable measurements
)
```

### Model Configuration
```python
model = DQN(
    policy="MlpPolicy",
    env=env,
    learning_rate=1e-4,
    verbose=1,
)
```

### Training Process
```python
# Train with keyboard interrupt handling
model.learn(total_timesteps=50000, progress_bar=True)
```

## 📈 Monitoring & Debugging

### Reward Breakdown Logging
```python
reward_breakdown = {
    "performance_score": performance_score,
    "resource_efficiency": efficiency_score,
    "scaling_appropriateness": -scaling_penalty,
    "cluster_health": cluster_health_score,
    "stability_bonus": stability_bonus,
    "metrics_penalty": metrics_penalty,
}
logging.info(f"Reward breakdown: {reward_breakdown}")
```

### Step Information
```python
info = {
    "cpu_usage": cpu_usage,
    "memory_usage": memory_usage,
    "current_replicas": replica_state,
    "actual_replicas": replica,
    "action": mapped_action,
    "unschedulable_replicas": unschedulable_replicas,
    "not_fetchable_replicas": not_fetchable_replicas,
    "metrics_reliability": metrics_reliability,
    # ... additional metrics
}
```

## 🎯 Expected Learning Outcomes

With this reward system, the agent should learn to:

1. **Maintain Optimal Performance**: Keep CPU/Memory in 20-85% range
2. **Prevent Oscillation**: Make gradual, measured scaling decisions
3. **Resource Efficiency**: Avoid over-provisioning while ensuring performance
4. **Cluster Awareness**: Respect cluster resource constraints
5. **Stability**: Prefer small adjustments over large jumps
6. **Context Sensitivity**: Scale appropriately based on current system state

## 🔗 Related Files

- **Environment**: [`simulation/simulation_environment.py`](simulation/simulation_environment.py)
- **Training**: [`simulation/train.py`](simulation/train.py)
- **Load Testing**: [`simulation/load.py`](simulation/load.py)
- **Deployment**: [`demo/nodejs-deployment.yaml`](demo/nodejs-deployment.yaml)

---

*This reward system is designed to train robust, intelligent autoscaling policies that balance performance, efficiency, and stability in real Kubernetes environments.*
