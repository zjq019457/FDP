#!/usr/bin/env bash
# Dense (uncompressed) baselines for the NAACL tables. Sequential, one GPU, our jobs only.
set -u
R=/diskF/jiaqizhang/DENSE_BASELINES_V1; PY=/home/bai-project/anaconda3/envs/eigen/bin/python
GPU=${1:-1}; REPO=/home/bai-project/jiaqizhang/EigenAttn
export CUDA_VISIBLE_DEVICES=$GPU PYTHONHASHSEED=2 TOKENIZERS_PARALLELISM=false
echo "START $(date -Is) gpu=$GPU" >> $R/logs/driver.log
# 1) Llama-3-8B dense five-task zero-shot (same runner/tasks/args as all Llama-3 arms; --dense_base)
C=$R/cache_llama; mkdir -p $C/datasets $C/tmp
( export PYTHONPATH=$REPO/EigenAttn:$REPO HF_HOME=$C TRANSFORMERS_CACHE=$C HF_DATASETS_CACHE=$C/datasets TMPDIR=$C/tmp
  $PY -u $REPO/rebuttal_legacy_reproduction/runners/zero_shot_instrumented.py --dense_base --model /diskD/jiaqizhang/Llama-3 \
    --tokenizer_model /diskD/jiaqizhang/Llama-3 --task_yaml_dir /diskF/jiaqizhang/a1804298693/offline_datasets \
    --tasks piqa_localjsonl,winogrande_localjsonl,arc_easy_localjsonl,arc_challenge_localjsonl,hellaswag_localjsonl \
    --out_dir $R/llama3_dense_zs/output --net Llama-3-8b --dtype bf16 --method LLAMA3_DENSE --cache_dir $C \
    --attn_implementation eager --batch_size 8 ) > $R/logs/llama3_dense_zs.log 2>&1 && echo "DONE llama3_dense_zs $(date -Is)" >> $R/logs/driver.log || echo "FAIL llama3_dense_zs" >> $R/logs/driver.log
# 2) MPT-7B dense PPL (frozen MPT window binding)
$PY -u $R/code/mpt_dense_ppl.py > $R/logs/mpt_dense_ppl.log 2>&1 && echo "DONE mpt_dense_ppl $(date -Is)" >> $R/logs/driver.log || echo "FAIL mpt_dense_ppl" >> $R/logs/driver.log
# 3) MPT-7B dense five-task accuracy (frozen MPT manifest)
$PY -u $R/code/mpt_dense_acc.py > $R/logs/mpt_dense_acc.log 2>&1 && echo "DONE mpt_dense_acc $(date -Is)" >> $R/logs/driver.log || echo "FAIL mpt_dense_acc" >> $R/logs/driver.log
echo "ALL_DONE $(date -Is)" >> $R/logs/driver.log
