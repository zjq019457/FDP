#!/usr/bin/env bash
set -u
BRANCH=/diskE/jiaqizhang/PALU_FDP_NATIVE_EXCHANGE_FORMAL_V1
PYTHON=/home/bai-project/jiaqizhang/.venv/bin/python
SCRIPT="$BRANCH/code/profile_palu_formal_system.py"
for method in fisher fdp_native; do
  log="$BRANCH/logs/SYSTEM_PROFILE_0P5_${method^^}_RETRY3.log"
  echo "START ratio=0.5 method=$method $(date -Is)" >> "$BRANCH/logs/SYSTEM_PROFILE_QUEUE.log"
  env CUDA_VISIBLE_DEVICES=0 "$PYTHON" "$SCRIPT" --ratio 0.5 --method "$method" > "$log" 2>&1
  rc=$?
  echo "END ratio=0.5 method=$method rc=$rc $(date -Is)" >> "$BRANCH/logs/SYSTEM_PROFILE_QUEUE.log"
  [ "$rc" -eq 0 ] || exit "$rc"
done
echo "SYSTEM_PROFILE_QUEUE_COMPLETE $(date -Is)" >> "$BRANCH/logs/SYSTEM_PROFILE_QUEUE.log"
