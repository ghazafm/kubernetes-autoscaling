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
             + Critical Zone Override (NEW)
```

## 📊 Reward Components

The reward system consists of **7 main components** that work together to guide the RL agent toward optimal autoscaling decisions:

### 1. 🔥 Performance Score (Primary Objective)

**Purpose**: Ensures optimal resource utilization and prevents performance degradation using an incremental threshold system.

**Target Range**: 20-70% CPU and Memory usage

#### Incremental Threshold System

The new system applies **cumulative penalties** that stack as usage increases through multiple thresholds:

#### CPU Performance Scoring (Incremental)
```python
# CPU incremental penalties - penalties STACK as thresholds are crossed
if cpu_usage > 70%:            # Above target range
    penalty += (excess_usage) * 1.0     # Base penalty: -1.0 per %

if cpu_usage > 80%:            # Moderate zone
    penalty += (excess_usage) * 2.0     # Additional: -2.0 per %

if cpu_usage > 95%:            # Warning zone
    penalty += (excess_usage) * 5.0     # Additional: -5.0 per %

if cpu_usage > 100%:           # CRITICAL zone
    penalty += (excess_usage) * 10.0    # Additional: -10.0 per %

# Example: 100.5% CPU = -(30*1.0 + 15*2.0 + 5*5.0 + 0.5*10.0) = -95 penalty!
```

#### Memory Performance Scoring (Incremental)
```python
# Memory incremental penalties - penalties STACK as thresholds are crossed
if memory_usage > 70%:         # Above target range
    penalty += (excess_usage) * 0.8     # Base penalty: -0.8 per %

if memory_usage > 80%:         # Moderate zone
    penalty += (excess_usage) * 1.5     # Additional: -1.5 per %

if memory_usage > 95%:         # Warning zone
    penalty += (excess_usage) * 4.0     # Additional: -4.0 per %

if memory_usage > 100%:        # CRITICAL zone
    penalty += (excess_usage) * 8.0     # Additional: -8.0 per %

# Example: 105% Memory = -(30*0.8 + 15*1.5 + 5*4.0 + 5*8.0) = -108.5 penalty!
```

#### Critical Zone Override
```python
# NEW: Critical zone override prevents positive rewards when resources are critical
if cpu_usage > 100% OR memory_usage > 100%:
    if total_reward > 0:
        total_reward = min(total_reward, -10)  # Cap at -10 minimum
        log("CRITICAL ZONE OVERRIDE: Positive reward blocked")
```

#### Optimal Performance Rewards (Only When Safe)
```python
# Sweet spot: 20-70% CPU and 20-70% memory
# Bonuses only given if NOT in critical zone (>100%)
if NOT in_critical_zone:
    if cpu_optimal AND memory_optimal:
        bonus = +25               # Best case bonus
    elif cpu_optimal OR memory_optimal:
        bonus = +12               # Partial bonus (increased from +10)

    # Resource balance bonus (reduced from +5 to +3)
    resource_balance = 1 - abs(cpu_usage - memory_usage) / 100
    bonus += resource_balance * 3
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

**Purpose**: Prevents inappropriate scaling decisions with much stricter penalties based on current system state.

#### Inappropriate Scale Down (Much Stricter)
```python
if action < 0:  # Scaling down
    if cpu_usage > 100% OR memory_usage > 100%:
        penalty = abs(action) * 10     # SEVERE: -10x penalty in critical zone
    elif cpu_usage > 95% OR memory_usage > 95%:
        penalty = abs(action) * 5      # HIGH: -5x penalty in warning zone
    elif cpu_usage > 70% OR memory_usage > 70%:
        penalty = abs(action) * 3      # MODERATE: -3x penalty above target
```

#### Inappropriate Scale Up (Unchanged)
```python
if action > 0:  # Scaling up
    if cpu_usage < 10% AND memory_usage < 10% AND replicas > 2:
        penalty = action * 3  # Increased penalty for scaling up with low usage
```

#### Large Action Penalty (Increased)
```python
if abs(action) > 20:           # Large jumps (>20 replicas)
    penalty = abs(action) * 1.0    # Increased from 0.5
elif abs(action) > 10:         # Medium jumps (>10 replicas)
    penalty = abs(action) * 0.5    # Increased from 0.2
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

#### Healthy Cluster Bonus (Modified)
```python
# Only give cluster health bonus if NOT in critical zone
if all_pods_ready AND cpu_usage <= 100% AND memory_usage <= 100%:
    bonus = +5  # Bonus only when truly healthy
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

### 7. 🚨 Critical Zone Override (NEW)

**Purpose**: Prevents positive rewards when system is in critical state, ensuring safety-first learning.

```python
# Critical zone detection
critical_cpu = cpu_usage > 100%
critical_memory = memory_usage > 100%

# Override positive rewards in critical zones
if (critical_cpu OR critical_memory) AND total_reward > 0:
    critical_override = -total_reward      # Negate the positive reward
    total_reward = min(total_reward, -10)  # Cap at -10 minimum

    log(f"CRITICAL ZONE OVERRIDE: Reward capped at {total_reward} "
        f"(was {total_reward - critical_override}) - "
        f"CPU: {cpu_usage}%, Memory: {memory_usage}%")
```

**Key Features**:
- **Safety First**: No positive rewards when resources exceed 100%
- **Prevents Gaming**: Agent cannot get positive rewards by ignoring critical states
- **Minimum Penalty**: Always ensures at least -10 reward in critical zones
- **Clear Logging**: Alerts when override is applied for debugging

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
- Performance: +25 (optimal) +2.4 (balance) = +27.4
- Efficiency: 0 (no waste)
- Scaling: 0 (appropriate)
- Cluster: +5 (healthy)
- Stability: +2 (no action)
- Metrics: +1 (reliable)
- Critical Override: 0 (not in critical zone)
Total Reward: +35.4
```

### Scenario 2: Critical CPU - Should Be Heavily Penalized 🚨
```
State: CPU=100.5%, Memory=45%, Replicas=1, Action=0 (no scaling)
- Performance: -95.5 (incremental penalties: 30*1 + 15*2 + 5*5 + 0.5*10)
- Efficiency: +5 (single replica handling load)
- Scaling: 0 (no inappropriate action)
- Cluster: 0 (no bonus in critical zone)
- Stability: +2 (no action)
- Metrics: +1 (reliable)
- Critical Override: -13 (blocks positive rewards, caps at -10)
Total Reward: -10 (capped by critical override)
```

### Scenario 3: Near Critical - High Penalties �
```
State: CPU=99%, Memory=85%, Replicas=3, Action=+2 (appropriate scale up)
- Performance: -66.0 (incremental: 29*1 + 14*2 + 4*5) + memory penalties
- Efficiency: 0
- Scaling: 0 (appropriate scale up under load)
- Cluster: +5 (healthy)
- Stability: +3 (small action)
- Metrics: +1 (reliable)
- Critical Override: 0 (not >100%)
Total Reward: ~-57 (heavily negative due to high resource usage)
```

### Scenario 4: Bad Decision - Scale Down Under Load ❌
```
State: CPU=95%, Memory=90%, Replicas=2, Action=-1
- Performance: -40 (incremental penalties for both CPU and memory)
- Efficiency: 0
- Scaling: -5 (inappropriate scale down in warning zone)
- Cluster: +5 (healthy)
- Stability: +3 (small action)
- Metrics: +1 (reliable)
- Critical Override: 0 (not >100%)
Total Reward: -36 (much worse than before)
```

### Scenario 5: Over-provisioned, Appropriate Scale Down 📉
```
State: CPU=15%, Memory=20%, Replicas=10, Action=-3
- Performance: +12 (memory optimal, no CPU optimal due to waste)
- Efficiency: -20 (high waste with many replicas at low usage)
- Scaling: 0 (appropriate scale down)
- Cluster: +5 (healthy)
- Stability: +1 (medium action)
- Metrics: +1 (reliable)
- Critical Override: 0 (not in critical zone)
Total Reward: -1 (slightly negative due to waste)
```

### Scenario 6: Cluster Resource Exhaustion 🚨
```
State: CPU=80%, Memory=75%, Replicas=15, Action=+5, Unschedulable=3
- Performance: -10 (above target) + partial bonus = -5
- Efficiency: 0
- Scaling: -5 (large action penalty increased)
- Cluster: -22.4 (unschedulable penalty)
- Stability: 0 (large action)
- Metrics: -2 (some metrics unavailable)
- Critical Override: 0 (not >100%)
Total Reward: -34.4 (much worse penalty for poor decisions)
```

## 🎓 Design Principles

### 1. Hierarchical Priorities
- **Safety > Performance > Efficiency > Stability**
- Critical resource violations (>100%) get immediate override
- Resource efficiency is secondary to performance
- Stability encourages gradual learning

### 2. Incremental Penalty System
- **Graduated penalties** that stack as thresholds are exceeded
- 99% CPU gets much less penalty than 100.5% CPU
- Prevents cliff effects while maintaining strong signals
- More realistic representation of system stress

### 3. Anti-Oscillation
- Stability bonuses for small/no actions
- Progressive penalties for large actions
- Time-based considerations in observations
- Critical zone override prevents reward gaming

### 4. Contextual Actions
- Same action gets different rewards based on system state
- Scaling up under high load = good
- Scaling up under low load = bad
- Scaling down in critical zone = severely penalized

### 5. Safety-First Learning
- **NEW**: Critical zone override ensures safety
- No positive rewards when resources exceed limits
- Agent cannot learn to ignore critical states
- Always enforces minimum penalty in dangerous situations

### 6. Multi-objective Optimization
- Balances performance, efficiency, and stability
- No single metric dominates decision making
- Encourages well-rounded policies
- Prevents exploitation of reward system loopholes

## 🔧 Configuration Constants

```python
# Reward calculation thresholds (UPDATED)
CPU_LIMIT_THRESHOLD = 100.0       # Critical CPU threshold
MEMORY_LIMIT_THRESHOLD = 100.0    # Critical memory threshold
CPU_WARNING_THRESHOLD = 95.0      # High CPU warning
MEMORY_WARNING_THRESHOLD = 95.0   # High memory warning
MODERATE_CPU_THRESHOLD = 80.0     # Moderate CPU usage
MODERATE_MEMORY_THRESHOLD = 80.0  # Moderate memory usage

# Target ranges (CHANGED from 85% to 70%)
target_cpu = [20, 70]             # Optimal CPU range (was 85%)
target_memory = [20, 70]          # Optimal memory range (was 85%)

# Efficiency thresholds
MIN_EFFICIENT_CPU = 30.0          # Minimum efficient CPU for single replica
MIN_EFFICIENT_MEMORY = 20.0       # Minimum efficient memory for single replica
LOW_USAGE_CPU = 10.0              # Low CPU usage threshold
LOW_USAGE_MEMORY = 10.0           # Low memory usage threshold

# Action thresholds (INCREASED penalties)
LARGE_ACTION_THRESHOLD = 20       # Large scaling action (penalty: 1.0x)
MEDIUM_ACTION_THRESHOLD = 10      # Medium scaling action (penalty: 0.5x)
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
    "critical_override": critical_override,  # NEW
}
logging.info(f"Reward breakdown: {reward_breakdown}")

# NEW: Critical zone logging
if cpu_usage > 100% or memory_usage > 100%:
    logging.error(f"CRITICAL ZONE OVERRIDE: Reward capped at {reward}")
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

With this **enhanced incremental reward system**, the agent should learn to:

1. **Maintain Optimal Performance**: Keep CPU/Memory in 20-70% range (tightened from 85%)
2. **Avoid Critical Zones**: Strong aversion to exceeding 100% resource usage
3. **Prevent Oscillation**: Make gradual, measured scaling decisions
4. **Resource Efficiency**: Avoid over-provisioning while ensuring performance
5. **Cluster Awareness**: Respect cluster resource constraints
6. **Safety-First**: Never ignore critical resource states for rewards
7. **Context Sensitivity**: Scale appropriately based on current system state
8. **Incremental Understanding**: Learn nuanced differences between 99% and 100.5% usage

## 🔄 Key Improvements from Previous System

### ✅ **Fixes Applied**
- **Incremental Penalties**: 100.5% CPU now gets ~95 penalty vs 99% getting ~59 penalty
- **Critical Zone Override**: No positive rewards when >100% resource usage
- **Stricter Scaling Penalties**: 10x penalty for scaling down in critical zones
- **Tighter Target Range**: 20-70% instead of 20-85% for more conservative policies
- **Balanced Rewards**: Reduced positive bonuses to balance with increased penalties

### 🚨 **Problem Solved**
- **Before**: CPU=100.03% could get +25 reward (inappropriate!)
- **After**: CPU=100.03% gets -10 reward minimum (appropriate!)
- **Incremental Logic**: 99% CPU gets less penalty than 100.5% CPU (as requested)

### 📊 **Expected Behavior Changes**
- Agent will strongly avoid >100% resource usage
- More conservative scaling policies (targeting 20-70% range)
- Better understanding of resource pressure gradients
- No exploitation of reward system loopholes in critical states

## 🔗 Related Files

- **Environment**: [`simulation/simulation_environment.py`](simulation/simulation_environment.py)
- **Training**: [`simulation/train.py`](simulation/train.py)
- **Load Testing**: [`simulation/load.py`](simulation/load.py)
- **Deployment**: [`demo/nodejs-deployment.yaml`](demo/nodejs-deployment.yaml)

---

*This reward system is designed to train robust, intelligent autoscaling policies that balance performance, efficiency, and stability in real Kubernetes environments.*
