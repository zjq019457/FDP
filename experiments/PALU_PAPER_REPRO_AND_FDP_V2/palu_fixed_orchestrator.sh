#!/usr/bin/env bash
set -u

root=/diskF/jiaqizhang/PALU_PAPER_REPRO_AND_FDP_V2
python=/home/bai-project/anaconda3/envs/eigen/bin/python
export CUDA_VISIBLE_DEVICES=3
export PYTHONPATH=$root/official/compat:$root/official/Palu/3rdparty/fast-hadamard-transform:$root/official/Palu
mkdir -p "$root/official/Palu/cache" "$root/official/Palu/cache/whiten"

for ratio in 0.7 0.5 0.3; do
  out=$root/official/results/fixed_hfexact_v3_ratio_$ratio
  log=$root/logs/OFFICIAL_FIXED_PPL_hfexact_v3_ratio_$ratio.log
  if [ -f "$out/quality.json" ]; then
    echo "SKIP ratio=$ratio quality exists"
    continue
  fi
  mkdir -p "$out"
  while true; do
    free_mib=$(nvidia-smi --query-gpu=memory.free --format=csv,noheader,nounits -i 3 | tr -d " ")
    echo "WAIT ratio=$ratio gpu3_free_mib=$free_mib"
    if [ "$free_mib" -ge 40000 ]; then
      break
    fi
    sleep 60
  done
  cd "$root/official/Palu"
  "$python" -u "$root/official/run_official_palu_offline.py" \
    --mode fixed --ratio "$ratio" --use-cache --device cuda \
    --wiki-train "$root/official/data/wiki-train.hf_joined.raw" \
    --wiki-test "$root/official/data/wiki-test.hf_joined.raw" \
    --out "$out" >> "$log" 2>&1
  status=$?
  if [ "$status" -ne 0 ] || [ ! -f "$out/quality.json" ]; then
    echo "STOP ratio=$ratio status=$status quality_missing=1"
    exit "$status"
  fi
  echo "DONE ratio=$ratio"
done
echo "ALL_FIXED_RATIOS_DONE"
