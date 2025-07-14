#!/bin/bash
echo "RL Training Phase 3: Stress Testing"

uv run load.py \
  --url http://10.34.4.248:30002 \
  --pattern burst \
  --duration 7200 \
  --base-rps 10000 \
  --max-rps 100000 \
  --max-concurrent 50 \
  --stress-probability 0.4 \
  --verbose
