#!/usr/bin/env bash
ROOT=/diskF/jiaqizhang/fdp125_final_campaign_v1
for attempt in 1 2 3 4 5; do
  while true; do
    free=$(nvidia-smi --query-gpu=memory.free --format=csv,noheader,nounits -i 2 | tr -d ' ')
    if [ "$free" -ge 45000 ]; then break; fi
    echo "$(date -Is) WAIT attempt=$attempt gpu=2 free=$free"
    sleep 60
  done
  echo "$(date -Is) START attempt=$attempt"
  CUDA_VISIBLE_DEVICES=2 /home/bai-project/anaconda3/envs/eigen/bin/python -u "$ROOT/official_uniform_llama_gapfill_0571.py" --tier medium --gpu 2
  rc=$?
  echo "$(date -Is) EXIT=$rc"
  if [ "$rc" -eq 0 ]; then exit 0; fi
  sleep 60
done
exit 1
