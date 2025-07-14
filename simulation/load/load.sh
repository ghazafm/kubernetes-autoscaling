#!/bin/bash

echo "Starting RL Training Data Collection"
echo "Total Duration: ~5 hours"
echo "======================================"

echo "Pre-training cluster state:"
kubectl top nodes
kubectl get pods -l app=nodejs

echo "Phase 1: Exploration (2 hours)"
./phase1.sh

sleep 60

echo "Phase 2: Exploitation (2 hours)"
./phase2.sh

sleep 60

echo "Phase 3: Stress Testing (2 hour)"
./phase3.sh

echo "Phase 4: Stress Testing (1 hour)"
./phase4.sh

echo "Phase 5: Stress Testing (1 hour)"
./phase5.sh

echo "Phase 6: Stress Testing (1 hour)"
./phase6.sh

echo "RL Training Data Collection Complete"
echo "Check logs and metrics for training data quality"
