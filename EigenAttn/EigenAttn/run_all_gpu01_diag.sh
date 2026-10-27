#!/usr/bin/env bash
set -euo pipefail

export CUDA_VISIBLE_DEVICES=0,1
export HF_HOME=/diskF/hf
export HF_DATASETS_CACHE=/diskF/jiaqi_hf_cache/datasets
export HF_DATASETS_OFFLINE=1
export PYTORCH_CUDA_ALLOC_CONF=max_split_size_mb:256

MODEL=/diskE/opt-30b
TOKENIZER=/diskE/opt-30b
CALIB=/diskE/c4-en/en

OUT=/diskD/eigen_outputs_opt30b_r448_gpu01_diag
EIGEN_BASIS=/home/bai-project/jiaqizhang/EigenAttn/EigenAttn/qkv_outputs/eigen_basis.pt
LAYER=46
RANK=448

mkdir -p "$OUT"

echo "[1/2] Run main_eigen_attn (2 GPUs) to export qkv mats + baseline analysis..."
torchrun --standalone --nproc_per_node=2 --master_port=29511 \
  main_eigen_attn.py \
    --model "$MODEL" \
    --tokenizer "$TOKENIZER" \
    --net opt-30b \
    --calib_dataset c4 \
    --calib_data_path "$CALIB" \
    --nsamples 128 \
    --batch_size 1 \
    --seed 2 \
    --attn_implementation eager \
    --avg_dim "$RANK" \
    --error_budget 0.025 \
    --dtype bf16 \
    --multigpu \
    --output_dir "$OUT" \
    --qkv_export --qkv_export_layers "$LAYER" \
    --qkv_analyze --qkv_analyze_stage baseline \
    --qkv_layers "$LAYER" --qkv_ranks "$RANK" \
    --qkv_do_svd \
    --qkv_eigen_basis "$EIGEN_BASIS"

echo "[2/2] Diagnose basis side + orthonormalize if needed..."
python diag_basis_side_fix.py \
  --out_dir "$OUT" \
  --layer "$LAYER" \
  --rank "$RANK" \
  --eigen_basis "$EIGEN_BASIS"

echo "DONE. Check: $OUT/diag_basis_side_layer${LAYER}.txt and fixed basis (if generated)."