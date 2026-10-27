#!/usr/bin/env bash
# FDP_MEDIUM_LATENCY_V1 driver: re-materialize Medium FDP + Uniform from the verified cost-benchmark bases (frozen
# main_eigen_attn command), verify rank-config SHA, then time Dense / Uniform / FDP sequentially on one exclusive GPU.
set -u
R=/diskF/jiaqizhang/FDP_MEDIUM_LATENCY_V1; C=/diskF/jiaqizhang/FDP_MEDIUM_OFFLINE_COST_V1/work
PY=/home/bai-project/anaconda3/envs/eigen/bin/python; EIGEN=/home/bai-project/jiaqizhang/EigenAttn/EigenAttn
GPU=${1:-1}
export PYTHONPATH=$EIGEN:/home/bai-project/jiaqizhang/EigenAttn HF_HUB_OFFLINE=1 TRANSFORMERS_OFFLINE=1 HF_DATASETS_OFFLINE=1
mat() {  # name basis eb
  mkdir -p $R/models/$1
  CUDA_VISIBLE_DEVICES=$GPU $PY -u $EIGEN/main_eigen_attn.py --model /diskD/jiaqizhang/Llama-3 --net Llama-3-8b --dtype bf16 --error_budget $3 \
    --avg_dim 4096 --seqlen 2048 --nsamples 64 --seed 2 --calib_dataset c4 --compress_only --attn_implementation eager --eigen_basis_path $2 \
    --save_dir $R/models/$1/final_model --output_dir $R/models/$1 --cache_dir $EIGEN/HF_cache > $R/logs/mat_$1.log 2>&1
}
mkdir -p $R/logs $R/results
echo "START $(date -Is)" >> $R/logs/driver.log
mat fdp $C/fdp_chain/fdp125/basis/fdp125_llama_medium_fdp125_perlayer_eb.pt 0.06 || { echo MAT_FDP_FAIL >> $R/logs/driver.log; exit 1; }
mat uniform $C/uniform_comparator/basis/official_uniform_medium_eb_0p0567_retry_gpu2_perlayer_eb.pt 0.0567 || { echo MAT_UNI_FAIL >> $R/logs/driver.log; exit 1; }
$PY - <<EOF >> $R/logs/driver.log
import hashlib, json
want = {"fdp": "f0c462759f53f77214eb7fdce65ea311b5890fd0cd7c6c3eaa061f4f07bf211b"}
r = json.load(open("$R/models/fdp/final_model/low_rank_config.json"))
print("FDP rank sha", hashlib.sha256(json.dumps(r, separators=(",", ":")).encode()).hexdigest() == want["fdp"], sum((a+b)*4096 for a, b in r))
u = open("$R/models/uniform/final_model/low_rank_config.json", "rb").read()
print("UNIFORM file sha", hashlib.sha256(u).hexdigest() == "db207077467e658317a1972240a42b211212563f38b3e819210b90006a1615cb", sum((a+b)*4096 for a, b in json.loads(u)))
EOF
nvidia-smi --query-compute-apps=pid,used_memory,gpu_uuid --format=csv,noheader >> $R/logs/driver.log
for m in dense uniform fdp; do
  src=$R/models/$m/final_model; [ $m = dense ] && src=dense
  echo "BENCH $m $(date -Is) gpu_procs_before: $(nvidia-smi --query-compute-apps=pid --format=csv,noheader -i $GPU | wc -l)" >> $R/logs/driver.log
  CUDA_VISIBLE_DEVICES=$GPU $PY -u $R/code/latency_bench.py $m $src $R/results/latency_$m.json 2048 8064 > $R/logs/bench_$m.log 2>&1 || echo "BENCH_FAIL $m" >> $R/logs/driver.log
done
echo "ALL_DONE $(date -Is)" >> $R/logs/driver.log
