#!/bin/bash
echo "RL Training Phase 3: Stress Testing"

uv run load.py \
  --url http://10.34.4.248:30002 \
  --pattern burst \
  --duration 7200 \
  --base-rps 100 \
  --max-rps 500 \
  --max-concurrent 30 \
  --stress-probability 0.4 \
  --verbose
