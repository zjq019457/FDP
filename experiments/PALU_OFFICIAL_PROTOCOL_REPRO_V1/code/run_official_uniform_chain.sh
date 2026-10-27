#!/usr/bin/env bash
set -euo pipefail

BRANCH=/diskE/jiaqizhang/PALU_OFFICIAL_PROTOCOL_REPRO_V1
PYTHON=/home/bai-project/anaconda3/envs/eigen/bin/python
RUNNER="$BRANCH/code/run_palu_official_repro_and_formal.py"
mkdir -p "$BRANCH/logs"
cd "$BRANCH"

for spec in "0.7 uniform_0p7" "0.5 uniform_0p5" "0.3 uniform_0p3"; do
  ratio="${spec%% *}"
  tag="${spec##* }"
  result="$BRANCH/results/${tag}_ratio_${ratio}_uniform.json"
  while ps -eo args= | grep -F -- "code/run_palu_official_repro_and_formal.py --ratio $ratio --search_method uniform --device cuda --tag $tag" | grep -v grep >/dev/null; do
    sleep 20
  done
  if [[ ! -f "$result" ]]; then
    echo "UNIFORM_CHAIN_BLOCKED ratio=$ratio missing_result=$result" >&2
    exit 2
  fi
  if [[ "$ratio" == "0.7" ]]; then
    next_tag="uniform_0p5"
  elif [[ "$ratio" == "0.5" ]]; then
    next_tag="uniform_0p3"
  else
    next_tag=""
  fi
  if [[ -n "$next_tag" ]]; then
    next_ratio="0.5"
    [[ "$ratio" == "0.5" ]] && next_ratio="0.3"
    nohup env CUDA_VISIBLE_DEVICES=2 \
      PYTHONPATH=/diskF/jiaqizhang/PALU_PAPER_REPRO_AND_FDP_V2/official/Palu/3rdparty/fast-hadamard-transform \
      "$PYTHON" -u "$RUNNER" --ratio "$next_ratio" --search_method uniform --device cuda --tag "$next_tag" \
      > "$BRANCH/logs/${next_tag}_gpu2.log" 2>&1 &
  fi
done
echo "UNIFORM_CHAIN_COMPLETE"
