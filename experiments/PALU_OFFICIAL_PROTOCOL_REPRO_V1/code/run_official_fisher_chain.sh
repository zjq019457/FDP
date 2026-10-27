#!/usr/bin/env bash
set -euo pipefail

BRANCH=/diskE/jiaqizhang/PALU_OFFICIAL_PROTOCOL_REPRO_V1
PYTHON=/home/bai-project/anaconda3/envs/eigen/bin/python
RUNNER="$BRANCH/code/run_palu_official_repro_and_formal.py"
cd "$BRANCH"

# Do not start Fisher until all official Uniform cells exist. This is a
# dependency gate, not a quality-driven choice.
for ratio in 0.7 0.5 0.3; do
  uniform="$BRANCH/results/uniform_0p${ratio#0.}_ratio_${ratio}_uniform.json"
  while [[ ! -f "$uniform" ]]; do sleep 20; done
done

for spec in "0.7 fisher_0p7" "0.5 fisher_0p5" "0.3 fisher_0p3"; do
  ratio="${spec%% *}"
  tag="${spec##* }"
  result="$BRANCH/results/${tag}_ratio_${ratio}_fisher_uniform.json"
  if [[ -f "$result" ]]; then continue; fi
  rebuild_arg=""
  fisher_cache="$BRANCH/cache/_diskF_jiaqizhang_PALU_PAPER_REPRO_AND_FDP_V2_models_Llama-2-7b-hf_calib_fisher_info.pt"
  [[ ! -f "$fisher_cache" && "$ratio" == "0.7" ]] && rebuild_arg="--rebuild-fisher-cache"
  nohup env CUDA_VISIBLE_DEVICES=2 \
    PYTHONPATH=/diskF/jiaqizhang/PALU_PAPER_REPRO_AND_FDP_V2/official/Palu/3rdparty/fast-hadamard-transform \
    "$PYTHON" -u "$RUNNER" --ratio "$ratio" --search_method fisher_uniform --device cuda --tag "$tag" $rebuild_arg \
    > "$BRANCH/logs/${tag}_gpu2.log" 2>&1 &
  while ps -eo args= | grep -F -- "code/run_palu_official_repro_and_formal.py --ratio $ratio --search_method fisher_uniform --device cuda --tag $tag" | grep -v grep >/dev/null; do
    sleep 20
  done
  [[ -f "$result" ]] || { echo "FISHER_CHAIN_BLOCKED ratio=$ratio missing_result=$result" >&2; exit 2; }
done
echo "FISHER_CHAIN_COMPLETE"
