#!/usr/bin/env bash
# Parallel part of the no45 full trace. usage: launch_split.sh <name> <gpu> <lo> <hi>
# Layers lo..hi (inclusive) get eb=1e9 (full search to the threshold floor); all other layers get eb=1e-9 (one candidate,
# no search). Inputs are dense-propagated upstream, so the per-layer curves equal those of a single full run.
set -u
N=/diskF/jiaqizhang/FDP_OFFICIAL_NO45_V2; PY=/home/bai-project/anaconda3/envs/eigen/bin/python
name=$1; gpu=$2; lo=$3; hi=$4
run=$N/runs/$name; mkdir -p $run/cache
$PY -c "import json; json.dump({'error_budget_by_layer': [1e9 if $lo <= i <= $hi else 1e-9 for i in range(32)]}, open('$run/per_layer_eb.json', 'w'))"
export CUDA_VISIBLE_DEVICES=$gpu PYTHONHASHSEED=2 TOKENIZERS_PARALLELISM=false HF_HUB_OFFLINE=1 TRANSFORMERS_OFFLINE=1 HF_DATASETS_OFFLINE=1
export EIGENATTN_UPSTREAM=$N/code/upstream_no45 EIGENATTN_TRACE_OUT=$run/official_trace.json EIGENATTN_PER_LAYER_EB_JSON=$run/per_layer_eb.json
echo "START $name $(date -Is) gpu=$gpu layers=$lo..$hi" >> $N/logs/driver.log
$PY -u $N/code/run_shim_v2.py --model /diskD/jiaqizhang/Llama-3 --calib_dataset wikitext2 --nsamples 512 --seed 2 --avg_dim 4 \
  --error_budget 1000000000 --attn_implementation eager --output_dir $run --cache_dir $run/cache > $run/stdout.log 2>&1
echo "END $name rc=$? $(date -Is)" >> $N/logs/driver.log
