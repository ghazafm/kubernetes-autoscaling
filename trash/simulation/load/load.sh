#!/bin/bash

echo "Starting RL Training Data Collection"
echo "Total Duration: ~18 hours"
echo "======================================"

echo "Pre-training cluster state:"
kubectl top nodes
kubectl get pods -l app=nodejs

echo "Phase 1: Exploration (2 hours)"
./phase1.sh

sleep 3600

echo "Phase 2: Exploitation (2 hours)"
./phase2.sh

sleep 3600

echo "Phase 3: Stress Testing (2 hour)"
./phase3.sh

sleep 3600

echo "Phase 4: Stress Testing (2 hour)"
./phase4.sh

sleep 3600

echo "Phase 5: Stress Testing (2 hour)"
./phase5.sh

sleep 3600

echo "Phase 6: Stress Testing (2 hour)"
./phase6.sh

sleep 3600

echo "RL Training Data Collection Complete"
echo "Check logs and metrics for training data quality"
