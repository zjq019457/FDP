#!/usr/bin/env bash
set -euo pipefail

ROOT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
PYTHON_BIN="/home/bai-project/anaconda3/envs/eigen/bin/python"
STATE_DIR="$ROOT_DIR/rebuttal_legacy_reproduction"

rm -f "$STATE_DIR/STOP_REQUESTED"
if tmux has-session -t eigenattn_legacy_rebuttal 2>/dev/null; then
  echo "session already running"
  exit 0
fi
if [[ ! -f "$STATE_DIR/APPROVE_GPU_START" ]]; then
  echo "APPROVE_GPU_START missing; run start_legacy_rebuttal_tmux.sh / bootstrap first" >&2
  exit 1
fi
"$PYTHON_BIN" -u "$STATE_DIR/legacy_rebuttal_orchestrator.py" start --project-root "$ROOT_DIR"
echo "resumed eigenattn_legacy_rebuttal"
