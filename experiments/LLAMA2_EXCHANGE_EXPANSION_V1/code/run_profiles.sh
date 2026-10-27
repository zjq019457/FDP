#!/bin/bash
# Frozen Llama-2 d1/d2/d5 profile: same runner, args and Python env (.venv, torch 2.11 / transformers 4.53) as the e068 profile.
R=/diskF/jiaqizhang/LLAMA2_EXCHANGE_EXPANSION_V1; PY=/home/bai-project/anaconda3/envs/eigen/bin/python
M4=/diskE/jiaqizhang/ICLR_FDP125_PAPER_EVIDENCE_V1/13_MODEL_MATRIX_4X3
export PYTHONPATH=/home/bai-project/jiaqizhang/EigenAttn/EigenAttn:/home/bai-project/jiaqizhang/EigenAttn HF_HUB_OFFLINE=1 TRANSFORMERS_OFFLINE=1 PYTHONHASHSEED=2
for op in "$@"; do
  $PY $R/code/l2x_profile_inputs.py $op
  B=$($PY -c "import sys;sys.path.insert(0,'$R/code');import l2x_lib as X;print(X.B_of(X.uniform_pairs('$op')))")
  /home/bai-project/jiaqizhang/.venv/bin/python -u $M4/unified_protocol_closure_v1/llama2_clean_redesign_profile_uniform068_v5.py --manifest $M4/unified_protocol_closure_v1/FDP125_SAMPLE_MANIFEST_V1.json \
    --model-key llama2_7b --base /diskF/jiaqizhang/PALU_PAPER_REPRO_AND_FDP_V2/models/Llama-2-7b-hf --data-dir /home/bai-project/jiaqizhang/wikitext2 \
    --basis /diskF/jiaqizhang/fdp125_tacl_llama2_exact_3comp_pass1_v1_20260930/llama2_e060_full_capacity_basis_only_v1/llama2_e060_full_basis_capacity_only_eb0p06.pt \
    --rank-csv $R/profiles/$op/inputs/UNIFORM_${op}_PER_LAYER.csv --resource-manifest $R/profiles/$op/inputs/UNIFORM_${op}_RESOURCE_MANIFEST.json \
    --expected-b-live-bytes $B --expected-basis-sha256 196863f8ac17c17b276a8dc74a9dee9e30ee8e540f960c0638dccf86095f01d0 \
    --out-dir $R/profiles/$op/run --gpu 0 --memory-fraction 0.34 --torch-threads 8 || { echo "PROFILE $op FAILED"; exit 1; }
done
echo PROFILES_DONE
