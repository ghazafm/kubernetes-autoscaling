#!/bin/bash
echo "RL Training Phase 1: Exploration"

uv run load.py \
  --url http://10.34.4.248:30002 \
  --adaptive \
  --pattern-duration 360 \
  --duration 7200 \
  --base-rps 30 \
  --max-rps 250 \
  --max-concurrent 15 \
  --stress-probability 0.2 \
  --dynamic-stress \
  --verbose
