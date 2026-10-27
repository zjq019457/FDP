#!/usr/bin/env bash
set -euo pipefail

GPU="$XKV_SELECTED_GPU"
case "$GPU" in 0|1|3) ;; *) echo "forbidden GPU: $GPU" >&2; exit 2 ;; esac

AUTH=/diskF/jiaqizhang/XKV_PAPER_REPRO_FDP_V1
EXP=/home/bai-project/jiaqizhang/XKV_PAPER_REPRO_FDP_V1_RUNTIME_V2
REPO=/diskF/jiaqizhang/fdp_lowrank_operator_zoo_v1/repos/xkv_upstream
MODEL=/diskF/jiaqizhang/Llama-3.1
PY=/home/bai-project/jiaqizhang/.venv/bin/python
RUNNER="$AUTH/code/xkv_paper_entry.py"
RULER_DATA="$EXP/ruler_data/llama-3/65536"
TASKS_RULER='ruler/niah_single_1,ruler/niah_single_2,ruler/niah_multikey_1,ruler/niah_multikey_2,ruler/niah_multiquery,ruler/niah_multivalue,ruler/qa_1,ruler/qa_2,ruler/vt,ruler/fwe'
TASKS_LONG='long_bench/narrativeqa,long_bench/qasper,long_bench/multifieldqa_en,long_bench/hotpotqa,long_bench/2wikimqa,long_bench/musique,long_bench/gov_report,long_bench/qmsum,long_bench/multi_news,long_bench/trec,long_bench/triviaqa,long_bench/samsum,long_bench/passage_count,long_bench/passage_retrieval_en,long_bench/lcc,long_bench/repobench-p'
COMMON=(--model_name_or_path "$MODEL" --batch_size 1 --method xkv --merge_k --merge_v --rank_k 384 --rank_v 576 --layer_group_size 4 --start_layer_idx 0 --end_layer_idx -1 --result_dir "$EXP/results")

if [ ! -f "$MODEL/config.json" ] || [ ! -f "$MODEL/model.safetensors.index.json" ]; then
  echo "MODEL_NOT_READY $MODEL" >&2
  exit 2
fi
for task in niah_single_1 niah_single_2 niah_multikey_1 niah_multikey_2 niah_multiquery niah_multivalue qa_1 qa_2 vt fwe; do
  [ -s "$RULER_DATA/$task/validation.jsonl" ] || { echo "RULER_DATA_MISSING $task" >&2; exit 3; }
done

mkdir -p "$EXP/logs/ruler" "$EXP/logs/longbench" "$EXP/results"
export CUDA_VISIBLE_DEVICES="$GPU"
export OMP_NUM_THREADS=48
export PYTHONPATH="$AUTH/code:$REPO:${PYTHONPATH:-}"
cd "$REPO"
echo "SINGLE_GPU_OFFICIAL_START $(date -Is) physical_gpu=$GPU"

"$PY" -m torch.distributed.run --standalone --nnodes=1 --nproc_per_node=1 "$RUNNER" --flash2 --dataset_name "$TASKS_RULER" --datalen 65536 "${COMMON[@]}" 2>&1 | tee "$EXP/logs/ruler/official_xkv_baseline_single_gpu.log"
touch "$EXP/results/OFFICIAL_XKV_RULER_SINGLE_GPU_DONE"

"$PY" -m torch.distributed.run --standalone --nnodes=1 --nproc_per_node=1 "$RUNNER" --flash2 --dataset_name "$TASKS_LONG" --datalen 65536 "${COMMON[@]}" 2>&1 | tee "$EXP/logs/longbench/official_xkv_baseline_single_gpu.log"
touch "$EXP/results/OFFICIAL_XKV_LONGBENCH_SINGLE_GPU_DONE"

echo "SINGLE_GPU_OFFICIAL_COMPLETE $(date -Is) physical_gpu=$GPU"
