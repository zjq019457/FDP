#!/usr/bin/env bash
set -uo pipefail
cd /home/bai-project/jiaqizhang/EigenAttn/EigenAttn
export CUDA_VISIBLE_DEVICES=0
export PYTHONUNBUFFERED=1
/home/bai-project/jiaqizhang/.venv/bin/python /diskF/jiaqizhang/fdp125_tacl_llama2_exact_3comp_pass1_v1_20260930/LLAMA2_UNIFORM_EB059_STRICT_MATCH_V1/engineering_fix/run_main_with_opt_import_overlay.py \
  --model /diskF/jiaqizhang/PALU_PAPER_REPRO_AND_FDP_V2/models/Llama-2-7b-hf \
  --cache_dir /diskF/jiaqizhang/fdp125_tacl_llama2_exact_3comp_pass1_v1_20260930/LLAMA2_UNIFORM_EB059_STRICT_MATCH_V1/attempt5/hf_cache \
  --calib_dataset wikitext2 \
  --wikitext2_data_dir /home/bai-project/jiaqizhang/wikitext2 \
  --nsamples 512 --batch_size 1 --seed 2 \
  --attn_implementation eager --net Llama-2-7b \
  --save_dir /diskF/jiaqizhang/fdp125_tacl_llama2_exact_3comp_pass1_v1_20260930/LLAMA2_UNIFORM_EB059_STRICT_MATCH_V1/attempt5/final_model \
  --output_dir /diskF/jiaqizhang/fdp125_tacl_llama2_exact_3comp_pass1_v1_20260930/LLAMA2_UNIFORM_EB059_STRICT_MATCH_V1/attempt5 \
  --avg_dim 4096 --error_budget 0.059 --compress_only \
  --dtype bf16 --seqlen 2048 \
  --eigen_basis_path /diskF/jiaqizhang/fdp125_tacl_llama2_exact_3comp_pass1_v1_20260930/llama2_e060_full_capacity_basis_only_v1/llama2_e060_full_basis_capacity_only_eb0p06.pt
rc=$?
printf '%s\n' "$rc" > /diskF/jiaqizhang/fdp125_tacl_llama2_exact_3comp_pass1_v1_20260930/LLAMA2_UNIFORM_EB059_STRICT_MATCH_V1/attempt5/materialize.exit_code
exit "$rc"
