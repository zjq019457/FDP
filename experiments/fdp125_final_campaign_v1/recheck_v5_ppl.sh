#!/usr/bin/env bash
set -e
ROOT=/diskF/jiaqizhang/fdp125_final_campaign_v1
MODEL=/diskE/jiaqizhang/fdp125_final_campaign_v1_scratch/official_uniform_llama_v5/medium/final_model
OUT=$ROOT/results/llama/medium/official_uniform_more_resource_v5_recheck
REPO=/home/bai-project/jiaqizhang/EigenAttn
PY=/home/bai-project/anaconda3/envs/eigen/bin/python
WIKI=/home/bai-project/jiaqizhang/wikitext2
C4=/diskD/jiaqizhang/llama3_8b_eb045_eval_minimal-20260502T170035Z-3-001/llama3_8b_eb045_eval_minimal/data/c4_ppl_selected_texts.jsonl
LOCK=$REPO/rebuttal_legacy_reproduction/LEGACY_EVALUATION_PROTOCOL_LOCK.json
mkdir -p "$OUT/wiki" "$OUT/c4"
export PYTHONPATH=$REPO:$REPO/EigenAttn
export HF_HUB_OFFLINE=1 TRANSFORMERS_OFFLINE=1 HF_DATASETS_OFFLINE=1
export C4_DATA_DIR=$(dirname "$C4")
CUDA_VISIBLE_DEVICES=3 "$PY" -u "$REPO/rebuttal_legacy_reproduction/runners/ppl_instrumented.py" --dataset wikitext2 --wikitext2_dir "$WIKI" --model "$MODEL" --tok /diskD/jiaqizhang/Llama-3 --out_dir "$OUT/wiki" --dtype bf16 --method OFFICIAL_UNIFORM_LLAMA_medium_EB0055_V5_RECHECK --evaluation_protocol_hash "$(sha256sum "$LOCK" | awk '{print $1}')"
CUDA_VISIBLE_DEVICES=3 "$PY" -u "$REPO/rebuttal_legacy_reproduction/runners/ppl_instrumented.py" --dataset c4 --c4_texts "$C4" --model "$MODEL" --tok /diskD/jiaqizhang/Llama-3 --out_dir "$OUT/c4" --dtype bf16 --method OFFICIAL_UNIFORM_LLAMA_medium_EB0055_V5_RECHECK --evaluation_protocol_hash "$(sha256sum "$LOCK" | awk '{print $1}')"
echo DONE
