#!/usr/bin/env bash
# 分层 error_budget 生成 OPT-30B eigen basis；单卡 GPU0；产物写到 F 盘。
# 依赖：per_layer_eb_from_step1.py 生成的 JSON（含 error_budget_by_layer，长度 48）。

set -euo pipefail

REPO_ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
cd "$REPO_ROOT"

MODEL="${MODEL:-/diskE/opt-30b}"
DATA_DIR="${DATA_DIR:-/diskE/jiaqizhang/c4-en}"
OUT_DIR="${OUT_DIR:-/diskF/jiaqizhang}"
OUT_PREFIX="${OUT_PREFIX:-eigen_basis_opt30b_paper_perlayer_eb}"
EB_JSON="${EB_JSON:-/home/bai-project/jiaqizhang/step1_out/eb024_per_layer_error_budget.json}"
LOG="${LOG:-/diskF/jiaqizhang/${OUT_PREFIX}_nohup.log}"
PID_FILE="${PID_FILE:-/diskF/jiaqizhang/${OUT_PREFIX}.pid}"

export CUDA_VISIBLE_DEVICES=0
export OMP_NUM_THREADS=1
export PYTHONUNBUFFERED=1
export PYTORCH_CUDA_ALLOC_CONF="${PYTORCH_CUDA_ALLOC_CONF:-expandable_segments:True,max_split_size_mb:256}"

mkdir -p "$OUT_DIR"

# 注意：不要传 --device_map auto，否则可能用多段显存；单卡用 --device cuda:0
# 与 --low_vram 同用时必须显式传入 --nsamples/--seqlen，否则脚本会降为 256×1536。
nohup /home/bai-project/anaconda3/envs/eigen/bin/python -u tools/build_eigen_basis_opt_error_budget_offline.py \
  --model "$MODEL" \
  --data_dir "$DATA_DIR" \
  --out_dir "$OUT_DIR" \
  --out_prefix "$OUT_PREFIX" \
  --per_layer_eb_json "$EB_JSON" \
  --nsamples 512 --seqlen 2048 \
  --device cuda:0 \
  --paper_defaults \
  --low_vram \
  --verbose 1 \
  >"$LOG" 2>&1 &

echo $! >"$PID_FILE"
echo "PID $(cat "$PID_FILE")  log=$LOG  basis -> ${OUT_DIR}/${OUT_PREFIX}_perlayer_eb.pt"
