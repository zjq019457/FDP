#!/usr/bin/env bash
set -euo pipefail

ROOT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
PYTHON_BIN="/home/bai-project/anaconda3/envs/eigen/bin/python"
STATE_DIR="$ROOT_DIR/rebuttal_legacy_reproduction"

if [[ ! -f "$STATE_DIR/APPROVE_GPU_START" ]]; then
  "$PYTHON_BIN" -u "$STATE_DIR/legacy_rebuttal_orchestrator.py" bootstrap --project-root "$ROOT_DIR"
fi
if [[ -f "$STATE_DIR/APPROVE_GPU_START" ]]; then
  "$PYTHON_BIN" -u "$STATE_DIR/legacy_rebuttal_orchestrator.py" start --project-root "$ROOT_DIR"
  echo "Started session eigenattn_legacy_rebuttal"
else
  echo "GPU start blocked; see $STATE_DIR/START_BLOCKED_REPORT.md" >&2
  exit 1
fi
