#!/usr/bin/env bash
set -u
ROOT=/diskE/jiaqizhang/PALU_OFFICIAL_PROTOCOL_REPRO_V1
PY=/home/bai-project/anaconda3/envs/eigen/bin/python

# Recovery of the original chain's dependency check:
# wait for the frozen official Fisher baseline artifact, not for a
# downstream status.json that this same chain is supposed to create.
for ratio in 0.5 0.3; do
  tag=${ratio/./p}
  fisher="$ROOT/results/fisher_${tag}_ratio_${ratio}_fisher_uniform.json"
  while [[ ! -f "$fisher" ]]; do
    sleep 60
  done
  status="$ROOT/official_native_marginal/ratio_${tag}/status.json"
  while [[ ! -f "$status" ]]; do
    if ps -eo args | grep -F "profile_official_native_phi.py --ratio $ratio" | grep -v grep >/dev/null; then
      sleep 60
      continue
    fi
    free_mib=$(nvidia-smi --query-gpu=index,memory.free --format=csv,noheader,nounits | awk -F',' '$1+0==2 {gsub(/ /,"",$2); print $2}')
    if [[ -z "$free_mib" || "$free_mib" -lt 40960 ]]; then
      sleep 60
      continue
    fi
    break
  done
  if [[ -f "$status" ]]; then continue; fi
  CUDA_VISIBLE_DEVICES=2 "$PY" -u "$ROOT/code/profile_official_native_phi.py" --ratio "$ratio" --device cuda \
    > "$ROOT/logs/official_native_phi_${tag}_gpu2.log" 2>&1
done
