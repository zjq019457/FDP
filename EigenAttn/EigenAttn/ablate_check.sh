#!/usr/bin/env bash
set -euo pipefail

MODEL="${1:?need model_path}"
TOK="${2:?need tokenizer_path}"

run_case () {
  local name="$1"
  shift
  echo "=============================="
  echo "[CASE] $name"
  env "$@" python3 check_model_cli.py \
    --model_path "$MODEL" \
    --tokenizer_path "$TOK" \
    --device cuda \
    --dtype fp16 \
    --max_len 64 \
    --text "Hello world. This is a quick sanity check."
}

run_case "BASELINE (no disable flags)" \
  EIGEN_DISABLE_ALL=0 EIGEN_DISABLE_Q=0 EIGEN_DISABLE_K=0 EIGEN_DISABLE_V=0

run_case "DISABLE_ALL (bypass eigen)" \
  EIGEN_DISABLE_ALL=1

run_case "DISABLE_Q" \
  EIGEN_DISABLE_ALL=0 EIGEN_DISABLE_Q=1

run_case "DISABLE_K" \
  EIGEN_DISABLE_ALL=0 EIGEN_DISABLE_K=1

run_case "DISABLE_V" \
  EIGEN_DISABLE_ALL=0 EIGEN_DISABLE_V=1

echo "=============================="
echo "Done."
