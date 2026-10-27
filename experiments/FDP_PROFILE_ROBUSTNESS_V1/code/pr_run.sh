#!/bin/bash
# FDP_PROFILE_ROBUSTNESS_V1: rebuild the 0.74x (Medium) EigenAttention model at eps_base 0.06 (frozen builder, no floor),
# gate its ranks against the frozen record, then profile it with (a) the original two-passage probe and
# (b) 8 real WikiText-2 *train* passages of 512 tokens (no padding). No evaluation data is used.
set -u
R=/diskF/jiaqizhang/FDP_PROFILE_ROBUSTNESS_V1
PY=/home/bai-project/anaconda3/envs/eigen/bin/python
EIGEN=/home/bai-project/jiaqizhang/EigenAttn/EigenAttn
REPO=/home/bai-project/jiaqizhang/EigenAttn
MODEL=/diskD/jiaqizhang/Llama-3
WIKI=/home/bai-project/jiaqizhang/wikitext2
BUILDER=/diskF/jiaqizhang/FDP_FROZEN_PROTOCOL_CONSTRAINED45_V1/code/build_eigen_basis_llama_error_budget_offline_c45.py
SCR=/diskF/jiaqizhang/FDP_PROFILE_ROBUSTNESS_V1_SCRATCH
mkdir -p $R/logs $SCR
log(){ echo "$(date -Iseconds) $*" >> $R/logs/run.log; }
pick_gpu(){ nvidia-smi --query-gpu=index,memory.free --format=csv,noheader,nounits | sort -t, -k2 -nr | head -1 | cut -d, -f1; }
G=$(pick_gpu); log "start gpu=$G"
export CUDA_VISIBLE_DEVICES=$G PYTHONPATH=$EIGEN:$REPO PYTHONHASHSEED=2
echo '{"error_budget_by_layer": ['"$(python3 -c 'print(",".join(["0.06"]*32))')"']}' > $R/per_layer_eb.json
if [ ! -f $R/BASIS_DONE ]; then
  HF_HUB_OFFLINE=1 TRANSFORMERS_OFFLINE=1 HF_DATASETS_OFFLINE=1 $PY -u $BUILDER --model $MODEL --data_dir $WIKI --out_dir $SCR/basis --out_prefix pr_uniform_medium \
    --per_layer_eb_json $R/per_layer_eb.json --freeze_layer0_eb --layer0_eb 0.06 --nsamples 512 --seqlen 2048 --dtype bf16 \
    --attn_implementation eager --device cuda:0 --seed 2 --official_min_thresh 0.0 > $R/logs/basis.log 2>&1 || { log "BASIS FAILED"; exit 1; }
  touch $R/BASIS_DONE; log "basis done"
fi
if [ ! -f $R/MAT_DONE ]; then
  HF_HUB_OFFLINE=1 TRANSFORMERS_OFFLINE=1 HF_DATASETS_OFFLINE=1 $PY -u $EIGEN/main_eigen_attn.py --model $MODEL --net Llama-3-8b --dtype bf16 --error_budget 0.06 --avg_dim 4096 \
    --seqlen 2048 --nsamples 64 --seed 2 --calib_dataset c4 --compress_only --attn_implementation eager \
    --eigen_basis_path $SCR/basis/pr_uniform_medium_perlayer_eb.pt --save_dir $SCR/final_model --output_dir $SCR --cache_dir $EIGEN/HF_cache > $R/logs/materialize.log 2>&1 || { log "MAT FAILED"; exit 1; }
  cp $SCR/final_model/low_rank_config.json $R/low_rank_config.json
  python3 -c "
import json,sys
a=json.load(open('$R/low_rank_config.json')); b=json.load(open('/diskF/jiaqizhang/fdp125_final_campaign_v1/results/llama/medium/uniform/low_rank_config.json'))
ok=a==b; json.dump({'ranks_equal_frozen_uniform':ok,'B_live':sum(sum(x) for x in a)*4096},open('$R/RANK_GATE.json','w')); sys.exit(0 if ok else 3)" || { log "RANK GATE FAILED"; exit 1; }
  rm -rf $SCR/basis; touch $R/MAT_DONE; log "materialized, rank gate PASS"
fi
cd $EIGEN
$PY -u step1_diagnose_llama.py --base $MODEL --cmp $SCR/final_model --tok $MODEL --dtype bf16 --seqlen 512 --num_seqs 2 \
   --out $R/step1_default_probe.json --gpu 0 --attn_implementation eager > $R/logs/step1_default.log 2>&1 || { log "STEP1 default FAILED"; exit 1; }
log "step1 default done"
$PY -u step1_diagnose_llama.py --base $MODEL --cmp $SCR/final_model --tok $MODEL --dtype bf16 --seqlen 512 --num_seqs 8 \
   --prompts_file $R/wikitext2_train_prompts.txt --out $R/step1_wikitext2_train8.json --gpu 0 --attn_implementation eager > $R/logs/step1_wiki.log 2>&1 || { log "STEP1 wiki FAILED"; exit 1; }
log "step1 wiki done"
touch $R/PROFILES_DONE; log "all done (checkpoint kept until analysis is complete)"
