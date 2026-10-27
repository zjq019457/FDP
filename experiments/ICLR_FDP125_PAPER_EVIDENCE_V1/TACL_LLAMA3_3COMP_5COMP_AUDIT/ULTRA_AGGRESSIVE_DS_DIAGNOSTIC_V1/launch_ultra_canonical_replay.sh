#!/usr/bin/env bash
set -euo pipefail
ARM="$1"
GPU="$2"
BASE="$3"
EPS_JSON="$4"
OUT_DIR="$5"
PREFIX="$6"
PY=/home/bai-project/anaconda3/envs/eigen/bin/python
BUILDER=/home/bai-project/jiaqizhang/EigenAttn/EigenAttn/tools/build_eigen_basis_llama_error_budget_offline.py
MODEL=/diskD/jiaqizhang/Llama-3
DATA=/home/bai-project/jiaqizhang/wikitext2
mkdir -p "$OUT_DIR"
export CUDA_VISIBLE_DEVICES="$GPU" OMP_NUM_THREADS=4 HF_HUB_OFFLINE=1 TRANSFORMERS_OFFLINE=1 HF_DATASETS_OFFLINE=1
set +e
"$PY" -u "$BUILDER" \
  --model "$MODEL" --tokenizer "$MODEL" --data_dir "$DATA" \
  --out_dir "$OUT_DIR" --out_prefix "$PREFIX" \
  --per_layer_eb_json "$EPS_JSON" --freeze_layer0_eb --layer0_eb "$BASE" \
  --nsamples 512 --seqlen 2048 --token_budget 1500000 --dtype bf16 \
  --attn_implementation eager --device cuda:0 --seed 2 \
  >"$OUT_DIR/builder.stdout.log" 2>"$OUT_DIR/builder.stderr.log"
RC=$?
set -e
printf '%s\n' "$RC" >"$OUT_DIR/builder.rc"
exit "$RC"
