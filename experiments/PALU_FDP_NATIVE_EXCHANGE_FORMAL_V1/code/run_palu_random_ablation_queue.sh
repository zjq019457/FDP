#!/usr/bin/env bash
set -u
BRANCH=/diskE/jiaqizhang/PALU_FDP_NATIVE_EXCHANGE_FORMAL_V1
PYTHON=/home/bai-project/jiaqizhang/.venv/bin/python
SCRIPT="$BRANCH/code/run_palu_formal_quality.py"
first_pid="${1:-}"
if [ -n "$first_pid" ]; then
  while kill -0 "$first_pid" 2>/dev/null; do sleep 20; done
fi
for seed in 1 2; do
  log="$BRANCH/logs/FORMAL_QUALITY_0P5_RANDOM_${seed}.log"
  echo "START random_$seed $(date -Is)" >> "$BRANCH/logs/FORMAL_RANDOM_QUEUE.log"
  env CUDA_VISIBLE_DEVICES=0 "$PYTHON" "$SCRIPT" --ratio 0.5 --method "random_$seed" > "$log" 2>&1
  rc=$?
  echo "END random_$seed rc=$rc $(date -Is)" >> "$BRANCH/logs/FORMAL_RANDOM_QUEUE.log"
  [ "$rc" -eq 0 ] || exit "$rc"
done
echo "RANDOM_QUEUE_COMPLETE $(date -Is)" >> "$BRANCH/logs/FORMAL_RANDOM_QUEUE.log"
