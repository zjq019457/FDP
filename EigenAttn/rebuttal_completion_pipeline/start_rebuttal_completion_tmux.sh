#!/usr/bin/env bash
# Start detached tmux session for rebuttal completion pipeline.
set -euo pipefail
ROOT="$(cd "$(dirname "$0")/.." && pwd)"
PIPE="$ROOT/rebuttal_completion_pipeline"
PY="/home/bai-project/anaconda3/envs/eigen/bin/python"
LOG="$PIPE/background_logs/orchestrator.log"
mkdir -p "$PIPE/background_logs"

if tmux has-session -t fdp_rebuttal_completion 2>/dev/null; then
  echo "Session fdp_rebuttal_completion already running"
  exit 0
fi

cd "$PIPE"
# Soft bootstrap: preserve DONE markers (orchestrator.bootstrap no longer wipes).
"$PY" -u orchestrator.py bootstrap
tmux new-session -d -s fdp_rebuttal_completion \
  "$PY -u $PIPE/orchestrator.py run-loop >> $LOG 2>&1"
echo "Started tmux session: fdp_rebuttal_completion"
echo "Log: $LOG"
