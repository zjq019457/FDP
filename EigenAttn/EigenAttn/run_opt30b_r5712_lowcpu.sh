#!/bin/bash
# OPT-30B r5712 校准，限制 CPU 线程避免占满
export CUDA_VISIBLE_DEVICES=0,1,2,3
export OMP_NUM_THREADS=1
export MKL_NUM_THREADS=1
export OPENBLAS_NUM_THREADS=1
export HF_HOME=/diskF/hf
export HF_DATASETS_CACHE=/diskF/jiaqi_hf_cache/datasets
export HF_DATASETS_OFFLINE=1
export PYTORCH_CUDA_ALLOC_CONF=max_split_size_mb:256

torchrun --nproc_per_node=4 --master_port=29511 main.py \
  --model /diskE/opt-30b \
  --net opt-30b \
  --calib_dataset c4 \
  --nsamples 128 \
  --batch_size 1 \
  --seed 2 \
  --attn_implementation eager \
  --avg_dim 5712 \
  --error_budget 0.025 \
  --output_dir /diskD/eigen_outputs_opt30b_r5712
