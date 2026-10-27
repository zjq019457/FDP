#!/usr/bin/env bash
# Resume after stop or server restart.
set -euo pipefail
PIPE="$(cd "$(dirname "$0")" && pwd)"
rm -f "$PIPE/STOP_REQUESTED"
exec "$PIPE/start_rebuttal_completion_tmux.sh"
