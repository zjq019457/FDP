#!/usr/bin/env bash
set -u

BRANCH=/diskE/jiaqizhang/PALU_FDP_NATIVE_EXCHANGE_FORMAL_V1
PYTHON=/home/bai-project/jiaqizhang/.venv/bin/python
SCRIPT="$BRANCH/code/run_palu_formal_quality.py"

wait_for_pid() {
  pid="$1"
  while kill -0 "$pid" 2>/dev/null; do
    sleep 20
  done
}

run_cell() {
  ratio="$1"
  method="$2"
  tag="${ratio/./P}"
  log="$BRANCH/logs/FORMAL_QUALITY_${tag}_${method^^}.log"
  echo "START ratio=$ratio method=$method $(date -Is)" >> "$BRANCH/logs/FORMAL_QUALITY_QUEUE.log"
  env CUDA_VISIBLE_DEVICES=0 "$PYTHON" "$SCRIPT" --ratio "$ratio" --method "$method" > "$log" 2>&1
  rc=$?
  echo "END ratio=$ratio method=$method rc=$rc $(date -Is)" >> "$BRANCH/logs/FORMAL_QUALITY_QUEUE.log"
  return "$rc"
}

# The first cell is already running when this queue is launched.
wait_for_pid "${1:?running_first_pid_required}"

for pair in \
  "0.7 fisher" \
  "0.7 fdp_prior" \
  "0.7 fdp_native" \
  "0.3 uniform" \
  "0.3 fisher" \
  "0.3 fdp_prior" \
  "0.3 fdp_native"; do
  set -- $pair
  run_cell "$1" "$2" || exit $?
done

echo "QUEUE_COMPLETE $(date -Is)" >> "$BRANCH/logs/FORMAL_QUALITY_QUEUE.log"
