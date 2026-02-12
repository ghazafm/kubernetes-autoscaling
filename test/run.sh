source .env.test
for i in {1..10}; do
  echo "=== Run $i of 10 starting at $(date) ==="
  unbuffer ./run-k6.sh --test train 2>&1 | tee "logs/log_test_run_${MAX_REPLICAS}_${i}.log"
  if [ $i -lt 10 ]; then
    sleep 1100
  fi
done