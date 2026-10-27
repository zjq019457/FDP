#!/usr/bin/env bash
set -euo pipefail
PIPE="$(cd "$(dirname "$0")" && pwd)"
PY="/home/bai-project/anaconda3/envs/eigen/bin/python"
cd "$PIPE"
echo "=== orchestrator_state.json ==="
cat orchestrator_state.json 2>/dev/null || echo '{}'
echo ""
echo "=== tmux ==="
tmux ls 2>/dev/null || echo 'no sessions'
echo ""
echo "=== job counts ==="
"$PY" orchestrator.py monitor 2>/dev/null || true
echo ""
echo "=== recent log ==="
tail -20 background_logs/orchestrator.log 2>/dev/null || true
