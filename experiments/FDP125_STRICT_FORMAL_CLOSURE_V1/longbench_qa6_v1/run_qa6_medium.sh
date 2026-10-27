#!/usr/bin/env bash
set -euo pipefail
export CUDA_VISIBLE_DEVICES=1
export PYTHONHASHSEED=2
PY=/home/bai-project/anaconda3/envs/eigen/bin/python
RUNNER=/home/bai-project/jiaqizhang/EigenAttn/rebuttal_completion_pipeline/runners/wave2_longbench_eigen.py
BASE=/diskE/jiaqizhang/FDP125_STRICT_FORMAL_CLOSURE_V1
QA=$BASE/longbench_qa6_v1
TASKS=narrativeqa
$PY -u $RUNNER --method official_uniform --model $BASE/materialized_formal_checkpoints/medium/official_uniform/final_model --tok /diskD/jiaqizhang/Llama-3 --data_dir $QA/data --max_len 8192 --limit_per_task 100000 --tasks $TASKS --out_dir $QA/medium/official_uniform
$PY -u $RUNNER --method fdp125_direct --model $BASE/materialized_formal_checkpoints/medium/fdp125_direct/final_model --tok /diskD/jiaqizhang/Llama-3 --data_dir $QA/data --max_len 8192 --limit_per_task 100000 --tasks $TASKS --out_dir $QA/medium/fdp125_direct
$PY -u $QA/score_qa6.py --pred_root $QA/medium --manifest $QA/FDP125_LONGBENCH_QA6_FORMAL_MANIFEST.json --out_dir $QA/scores
