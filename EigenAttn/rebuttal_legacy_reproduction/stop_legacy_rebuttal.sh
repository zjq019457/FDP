#!/usr/bin/env bash
set -euo pipefail

ROOT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
STATE_DIR="$ROOT_DIR/rebuttal_legacy_reproduction"
date -Is > "$STATE_DIR/STOP_REQUESTED"
echo "Wrote $STATE_DIR/STOP_REQUESTED (no kill -9). Workers will stop after current atomic units."
