#!/usr/bin/env bash
set -euo pipefail
exec "$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)/rebuttal_legacy_reproduction/stop_legacy_rebuttal.sh"
