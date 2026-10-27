#!/usr/bin/env bash
# Graceful stop: write STOP_REQUESTED only (no kill -9).
set -euo pipefail
PIPE="$(cd "$(dirname "$0")" && pwd)"
touch "$PIPE/STOP_REQUESTED"
echo "STOP_REQUESTED written. Orchestrator will exit gracefully after current jobs."
