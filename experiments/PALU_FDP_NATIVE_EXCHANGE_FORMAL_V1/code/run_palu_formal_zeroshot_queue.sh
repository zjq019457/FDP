#!/usr/bin/env bash
set -u
BRANCH=/diskE/jiaqizhang/PALU_FDP_NATIVE_EXCHANGE_FORMAL_V1
PYTHON=/home/bai-project/jiaqizhang/.venv/bin/python
SCRIPT="$BRANCH/code/run_palu_formal_zeroshot.py"
first_pid="${1:-}"
if [ -n "$first_pid" ]; then
  while kill -0 "$first_pid" 2>/dev/null; do sleep 20; done
fi
for method in uniform fisher fdp_prior fdp_native; do
  log="$BRANCH/logs/FORMAL_ZS_0P5_${method^^}.log"
  echo "START ratio=0.5 method=$method $(date -Is)" >> "$BRANCH/logs/FORMAL_ZS_QUEUE.log"
  env CUDA_VISIBLE_DEVICES=0 "$PYTHON" "$SCRIPT" --ratio 0.5 --method "$method" > "$log" 2>&1
  rc=$?
  echo "END ratio=0.5 method=$method rc=$rc $(date -Is)" >> "$BRANCH/logs/FORMAL_ZS_QUEUE.log"
  [ "$rc" -eq 0 ] || exit "$rc"
done
echo "ZS_QUEUE_COMPLETE $(date -Is)" >> "$BRANCH/logs/FORMAL_ZS_QUEUE.log"
