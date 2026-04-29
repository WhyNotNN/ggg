#!/usr/bin/env bash
# Install JAX + ROCm runtime for AMD MI50 (gfx906) from the AMD nightly index.
#
# Usage:
#   scripts/install_rocm_jax.sh                     # latest nightly compatible with current Python
#   scripts/install_rocm_jax.sh 20260428            # pin all wheels to one nightly date
#   ROCM_DATE=20260428 scripts/install_rocm_jax.sh  # same, via env var
#
# Index: https://rocm.nightlies.amd.com/v2-staging/gfx906/
#
# Prereqs:
#   - Linux x86_64 host with /dev/kfd and /dev/dri exposed (or run inside a
#     container started with --device=/dev/kfd --device=/dev/dri --group-add video).
#   - Python 3.11, 3.12, 3.13 or 3.14. The gfx906 index does not publish wheels
#     for older interpreters. Python 3.12 is recommended.
#   - pip >= 24.

set -euo pipefail

INDEX_URL="${ROCM_NIGHTLY_INDEX:-https://rocm.nightlies.amd.com/v2-staging/gfx906/}"
ROCM_DATE="${1:-${ROCM_DATE:-}}"
PYTHON_BIN="${PYTHON_BIN:-python3}"

if ! command -v "${PYTHON_BIN}" >/dev/null 2>&1; then
  echo "error: ${PYTHON_BIN} not found on PATH" >&2
  exit 2
fi

PY_VER="$("${PYTHON_BIN}" -c 'import sys; print(f"{sys.version_info.major}.{sys.version_info.minor}")')"
case "${PY_VER}" in
  3.11|3.12|3.13|3.14) ;;
  *)
    echo "error: gfx906 nightlies require Python 3.11/3.12/3.13/3.14, got ${PY_VER}" >&2
    echo "       set PYTHON_BIN to a supported interpreter, for example PYTHON_BIN=python3.12" >&2
    exit 2
    ;;
esac

echo "==> using ${PYTHON_BIN} (Python ${PY_VER})"
echo "==> nightly index: ${INDEX_URL}"
if [[ -n "${ROCM_DATE}" ]]; then
  echo "==> pinning all gfx906/JAX wheels to nightly date: ${ROCM_DATE}"
fi

"${PYTHON_BIN}" -m pip install --upgrade pip

# Pinning policy: when ROCM_DATE is set we keep jaxlib, jax-rocm7-plugin,
# jax-rocm7-pjrt, and rocm-sdk-libraries-gfx906 on the SAME nightly date so the
# JAX/PJRT plugin ABI matches the bundled ROCm libraries.
if [[ -n "${ROCM_DATE}" ]]; then
  ROCM_VERSION="7.13.0a${ROCM_DATE}"
  "${PYTHON_BIN}" -m pip install \
    --extra-index-url "${INDEX_URL}" \
    "rocm-sdk-libraries-gfx906==${ROCM_VERSION}" \
    "jaxlib==0.9.0+${ROCM_VERSION}" \
    "jax-rocm7-plugin==0.9.1+${ROCM_VERSION}" \
    "jax-rocm7-pjrt==0.9.1+${ROCM_VERSION}"
  "${PYTHON_BIN}" -m pip install "jax==0.9.0"
else
  "${PYTHON_BIN}" -m pip install \
    --extra-index-url "${INDEX_URL}" \
    rocm-sdk-libraries-gfx906 \
    jaxlib \
    jax-rocm7-plugin \
    jax-rocm7-pjrt
  "${PYTHON_BIN}" -m pip install "jax"
fi

# SDXL/Diffusers stack from PyPI. Do NOT pull jaxlib/PJRT from PyPI; the gfx906
# build above must win.
SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
REQ_FILE="${SCRIPT_DIR}/../requirements-mi50-jax.txt"
"${PYTHON_BIN}" -m pip install -r "${REQ_FILE}"

echo
echo "==> installed packages:"
"${PYTHON_BIN}" -m pip list 2>/dev/null \
  | grep -Ei '^(jax|jaxlib|jax-rocm7-plugin|jax-rocm7-pjrt|rocm-sdk-libraries-gfx906|diffusers|flax|transformers)\b' \
  || true

echo
echo "==> done. Next: scripts/run_sdxl_jax.sh --check-only --compile-smoke-test"
