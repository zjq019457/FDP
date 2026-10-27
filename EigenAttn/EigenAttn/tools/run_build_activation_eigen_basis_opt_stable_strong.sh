#!/usr/bin/env bash
# 加强版：更大 C4 抽样 + 更多前向 batch + refine 采样/步数，并打开小幅 branch coupling。
# 默认 4 卡：CUDA_VISIBLE_DEVICES=0,1,2,3 + --device_map auto（模型按层切到多卡，需安装 accelerate）。
# 单卡跑：CUDA_VISIBLE_DEVICES=0 DEVICE_MAP=cuda:0 bash tools/...
#
# 用法（在 EigenAttn/EigenAttn 目录下）:
#   bash tools/run_build_activation_eigen_basis_opt_stable_strong.sh
# 或覆盖环境变量:
#   MODEL=/diskE/opt-30b REF_BASIS=/diskD/jiaqizhang/eigen_basis.pt \
#   OUT_BASIS=/diskF/.../eigen_basis_strong_v1.pt \
#   bash tools/run_build_activation_eigen_basis_opt_stable_strong.sh

set -euo pipefail

ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
cd "$ROOT"
export PYTHONPATH="${ROOT}:${PYTHONPATH:-}"

MODEL="${MODEL:-/diskE/opt-30b}"
REF_BASIS="${REF_BASIS:-/diskD/jiaqizhang/eigen_basis.pt}"
DATA_DIR="${DATA_DIR:-/diskE/jiaqizhang/c4-en/en}"
OUT_DIR="${OUT_DIR:-${ROOT}/qkv_outputs}"
OUT_BASIS="${OUT_BASIS:-${OUT_DIR}/eigen_basis_stable_strong_v1.pt}"
COMPARE_JSON="${COMPARE_JSON:-${OUT_DIR}/eigen_basis_stable_strong_v1_compare.json}"

CUDA_VISIBLE_DEVICES="${CUDA_VISIBLE_DEVICES:-0,1,2}"
DEVICE_MAP="${DEVICE_MAP:-auto}"
OMP_NUM_THREADS="${OMP_NUM_THREADS:-1}"

mkdir -p "$OUT_DIR"

echo "[strong] ROOT=$ROOT"
echo "[strong] CUDA_VISIBLE_DEVICES=$CUDA_VISIBLE_DEVICES DEVICE_MAP=$DEVICE_MAP"
echo "[strong] MODEL=$MODEL REF_BASIS=$REF_BASIS"
echo "[strong] OUT_BASIS=$OUT_BASIS"

PYTORCH_CUDA_ALLOC_CONF="${PYTORCH_CUDA_ALLOC_CONF:-max_split_size_mb:256}" \
CUDA_VISIBLE_DEVICES="$CUDA_VISIBLE_DEVICES" \
OMP_NUM_THREADS="$OMP_NUM_THREADS" \
python3 tools/build_activation_eigen_basis_opt_stable.py \
  --model "$MODEL" \
  --ref_basis "$REF_BASIS" \
  --out_basis "$OUT_BASIS" \
  --compare_json "$COMPARE_JSON" \
  --data_dir "$DATA_DIR" \
  --token_budget "${TOKEN_BUDGET:-3000000}" \
  --max_data_files "${MAX_DATA_FILES:-500}" \
  --max_data_lines "${MAX_DATA_LINES:-5000000}" \
  --layers all \
  --max_batches "${MAX_BATCHES:-96}" \
  --batch_size "${BATCH_SIZE:-1}" \
  --seqlen "${SEQLEN:-512}" \
  --device_map "$DEVICE_MAP" \
  --dtype bf16 \
  --repair_keqk 0 \
  --cov_residual_to_ref 0 \
  --weight_mode nll \
  --nll_power 1.0 \
  --nll_clip_percentile 99.5 \
  --nll_mix_lambda 0.9 \
  --nll_mix_lambda_qk 0.95 \
  --nll_mix_lambda_v 0.25 \
  --nll_clip_percentile_qk 99.5 \
  --nll_clip_percentile_v 100.0 \
  --nll_power_qk 1.0 \
  --nll_power_v 1.0 \
  --adaptive_refine 1 \
  --adaptive_front_scale "${ADAPTIVE_FRONT_SCALE:-0.40}" \
  --refine_steps "${REFINE_STEPS:-5}" \
  --refine_lr "${REFINE_LR:-0.004}" \
  --refine_batches "${REFINE_BATCHES:-6}" \
  --refine_sample_m "${REFINE_SAMPLE_M:-72}" \
  --refine_sample_n "${REFINE_SAMPLE_N:-72}" \
  --loss_score 1.0 \
  --loss_value 0.6 \
  --loss_couple "${LOSS_COUPLE:-0.08}" \
  --loss_drift 0.2 \
  --loss_resnorm "${LOSS_RESNORM:-0.10}" \
  --loss_stream_ratio "${LOSS_STREAM_RATIO:-0.08}" \
  --loss_anchor "${LOSS_ANCHOR:-0.06}" \
  --depth_boost "${DEPTH_BOOST:-0.30}" \
  --loss_wo "${LOSS_WO:-0.0}" \
  --loss_prox 0.1 \
  --accept_bad_layer 0 \
  --accept_mode composite \
  --accept_comp_w_score 0.65 \
  --accept_comp_w_value 0.35 \
  --accept_comp_max_delta "${ACCEPT_COMP_MAX_DELTA:--0.022}" \
  --accept_comp_metric "${ACCEPT_COMP_METRIC:-delta_over_floor}" \
  --accept_baseline_floor "${ACCEPT_BASELINE_FLOOR:-1e-4}"

echo "[strong] done. basis=$OUT_BASIS compare=$COMPARE_JSON"
