#!/usr/bin/env bash
# FDP_OFFICIAL_NO45_V2 stage 0. usage: launch_trace.sh <name> <gpu> <upstream: official|no45> <error_budget>
# Records every evaluated candidate (ranks, threshold, exact error) and the upstream eigenvalues; saves no model.
set -u
N=/diskF/jiaqizhang/FDP_OFFICIAL_NO45_V2; PY=/home/bai-project/anaconda3/envs/eigen/bin/python
name=$1; gpu=$2; up=$3; eb=$4
run=$N/runs/$name; mkdir -p $run/cache
U=/diskF/jiaqizhang/OFFICIAL_EIGENATTN_REPRO_V1/upstream; [ "$up" = no45 ] && U=$N/code/upstream_no45
export CUDA_VISIBLE_DEVICES=$gpu PYTHONHASHSEED=2 TOKENIZERS_PARALLELISM=false HF_HUB_OFFLINE=1 TRANSFORMERS_OFFLINE=1 HF_DATASETS_OFFLINE=1
export EIGENATTN_UPSTREAM=$U EIGENATTN_TRACE_OUT=$run/official_trace.json EIGENATTN_EVALS_OUT=$run/evals.pt
echo "START $name $(date -Is) gpu=$gpu upstream=$U eb=$eb" >> $N/logs/driver.log
$PY -u $N/code/run_shim_v2.py --model /diskD/jiaqizhang/Llama-3 --calib_dataset wikitext2 --nsamples 512 --seed 2 --avg_dim 4 \
  --error_budget $eb --attn_implementation eager --output_dir $run --cache_dir $run/cache > $run/stdout.log 2>&1
echo "END $name rc=$? $(date -Is)" >> $N/logs/driver.log
