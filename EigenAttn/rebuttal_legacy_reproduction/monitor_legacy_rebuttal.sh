#!/usr/bin/env bash
set -euo pipefail

ROOT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
PYTHON_BIN="/home/bai-project/anaconda3/envs/eigen/bin/python"
STATE_DIR="$ROOT_DIR/rebuttal_legacy_reproduction"

"$PYTHON_BIN" -u "$STATE_DIR/legacy_rebuttal_orchestrator.py" monitor --project-root "$ROOT_DIR"
echo "--- session ---"
tmux has-session -t eigenattn_legacy_rebuttal 2>/dev/null && echo "session=UP" || echo "session=DOWN"
echo "--- heartbeats ---"
ls -lt "$STATE_DIR"/job_outputs/*/heartbeat.json 2>/dev/null | head -10 || true
echo "--- STOP ---"
[[ -f "$STATE_DIR/STOP_REQUESTED" ]] && echo STOP_REQUESTED=yes || echo STOP_REQUESTED=no
