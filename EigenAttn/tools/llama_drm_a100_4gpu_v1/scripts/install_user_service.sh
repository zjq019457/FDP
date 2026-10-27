#!/usr/bin/env bash
# Install the LLAMA_DRM_A100_4GPU_V1 orchestrator as a *user* systemd unit.
# No sudo is required or used. Linger is enabled best-effort so the unit survives logout.
set -euo pipefail

PROTOCOL_ID="LLAMA_DRM_A100_4GPU_V1"
UNIT_NAME="llama-drm-a100-4gpu-v1.service"
SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
TOOL_ROOT="$(cd "${SCRIPT_DIR}/.." && pwd)"
UNIT_SRC="${SCRIPT_DIR}/${UNIT_NAME}"
UNIT_DIR="${HOME}/.config/systemd/user"
UNIT_DST="${UNIT_DIR}/${UNIT_NAME}"

echo "[install] protocol: ${PROTOCOL_ID}"
echo "[install] tool root: ${TOOL_ROOT}"

if [[ ! -f "${UNIT_SRC}" ]]; then
  echo "[install] ERROR: unit file not found at ${UNIT_SRC}" >&2
  exit 1
fi

mkdir -p "${UNIT_DIR}" "${TOOL_ROOT}/logs" "${TOOL_ROOT}/locks"

# Rewrite the hard-coded paths in case the tool tree was relocated.
PYTHON_BIN="${LLAMA_DRM_PYTHON:-/home/bai-project/a1804298693/.conda/envs/llamaqkv/bin/python}"
if [[ ! -x "${PYTHON_BIN}" ]]; then
  echo "[install] ERROR: interpreter not executable: ${PYTHON_BIN}" >&2
  exit 1
fi

sed \
  -e "s|/home/bai-project/jiaqizhang/EigenAttn/tools/llama_drm_a100_4gpu_v1|${TOOL_ROOT}|g" \
  -e "s|/home/bai-project/a1804298693/.conda/envs/llamaqkv/bin/python|${PYTHON_BIN}|g" \
  "${UNIT_SRC}" > "${UNIT_DST}"
echo "[install] wrote ${UNIT_DST}"

# Linger lets a user unit keep running after logout. It may be denied on a locked-down
# host, which is not fatal: the unit still runs inside an active session.
if command -v loginctl >/dev/null 2>&1; then
  if loginctl enable-linger "${USER}" 2>/dev/null; then
    echo "[install] linger enabled for ${USER}"
  else
    echo "[install] WARNING: could not enable linger for ${USER} (no sudo); the unit will"
    echo "[install]          stop when your last session ends."
  fi
else
  echo "[install] WARNING: loginctl unavailable; skipping linger"
fi

if ! systemctl --user daemon-reload 2>/dev/null; then
  echo "[install] ERROR: 'systemctl --user' is unavailable (no user D-Bus session?)." >&2
  echo "[install] Start the orchestrator manually instead:" >&2
  echo "[install]   ${PYTHON_BIN} ${TOOL_ROOT}/orchestrator.py --resume" >&2
  exit 2
fi
echo "[install] daemon-reload done"

systemctl --user enable --now "${UNIT_NAME}"
echo "[install] enabled and started ${UNIT_NAME}"
echo
systemctl --user status "${UNIT_NAME}" --no-pager --lines=20 || true
echo
echo "[install] follow logs:   journalctl --user -u ${UNIT_NAME} -f"
echo "[install] pipeline state: ${TOOL_ROOT}/RESULTS_ROOT.txt -> <result_dir>/status/PIPELINE_STATUS.md"
echo "[install] stop:          systemctl --user stop ${UNIT_NAME}"
