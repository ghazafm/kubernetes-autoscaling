#!/bin/bash
echo "RL Training Phase 2: Exploitation"

uv run load.py \
  --url http://10.34.4.248:30002 \
  --adaptive \
  --pattern-duration 240 \
  --duration 7200 \
  --base-rps 80 \
  --max-rps 350 \
  --max-concurrent 20 \
  --stress-probability 0.3 \
  --dynamic-stress \
  --verbose
