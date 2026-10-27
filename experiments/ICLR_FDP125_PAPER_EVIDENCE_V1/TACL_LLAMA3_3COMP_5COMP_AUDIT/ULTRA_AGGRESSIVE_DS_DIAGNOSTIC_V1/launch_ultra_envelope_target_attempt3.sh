#!/usr/bin/env bash
set -euo pipefail
F=/diskF/jiaqizhang/llama3_full_resource_robustness_v1/ultra_aggressive_v1/envelope_target_attempt3
P=/home/bai-project/anaconda3/envs/eigen/bin/python
B=/home/bai-project/jiaqizhang/EigenAttn/EigenAttn/tools/build_eigen_basis_llama_error_budget_offline.py
mkdir -p "$F"
cmd="$P -u $B --model /diskD/jiaqizhang/Llama-3 --tokenizer /diskD/jiaqizhang/Llama-3 --data_dir /home/bai-project/jiaqizhang/wikitext2 --out_dir $F --out_prefix ultra_aggressive_curve_full_q_target_attempt3 --per_layer_eb_json $F/ultra_curve_envelope_epsilon.json --freeze_layer0_eb --layer0_eb 0.12 --nsamples 512 --seqlen 2048 --token_budget 1500000 --dtype bf16 --attn_implementation eager --device cuda:1 --seed 2 > $F/builder.stdout.log 2> $F/builder.stderr.log; rc=\$?; printf '%s\\n' \"\$rc\" > $F/builder.rc"
tmux new-session -d -s fdp125_ultra_curve_target_attempt3 bash -lc "$cmd"
