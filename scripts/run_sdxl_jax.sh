#!/usr/bin/env bash
# Convenience wrapper for scripts/generate_sdxl_jax.py with sensible defaults
# for 2x AMD MI50 16GB (gfx906).
#
# All extra arguments are forwarded to the Python runner.
#
# Examples:
#   scripts/run_sdxl_jax.sh --check-only --compile-smoke-test
#   scripts/run_sdxl_jax.sh --prompt "moody portrait, cinematic"
#   HIP_VISIBLE_DEVICES=0 scripts/run_sdxl_jax.sh --require-devices 1 --prompt "..."

set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
PYTHON_BIN="${PYTHON_BIN:-python3}"

# Devices: default to both MI50s. Override with HIP_VISIBLE_DEVICES=0 for a
# single card.
export HIP_VISIBLE_DEVICES="${HIP_VISIBLE_DEVICES:-0,1}"

# JAX/XLA tuning that matters on 16GB gfx906:
export JAX_PLATFORMS="${JAX_PLATFORMS:-rocm}"
# Do not preallocate the whole VRAM; SDXL allocates spikily and 16GB is tight.
export XLA_PYTHON_CLIENT_PREALLOCATE="${XLA_PYTHON_CLIENT_PREALLOCATE:-false}"
# Cap the per-process VRAM ceiling well below 16GB so the allocator does not
# fragment to OOM during UNet attention.
export XLA_PYTHON_CLIENT_MEM_FRACTION="${XLA_PYTHON_CLIENT_MEM_FRACTION:-0.90}"
# gfx906 has no MFMA; this skips MIOpen tuning passes that hang on Vega20.
export MIOPEN_FIND_MODE="${MIOPEN_FIND_MODE:-FAST}"
export MIOPEN_LOG_LEVEL="${MIOPEN_LOG_LEVEL:-3}"
# Use the ROCm libs bundled with rocm-sdk-libraries-gfx906 if /opt/rocm is empty.
if [[ -d /opt/rocm ]]; then
  export ROCM_PATH="${ROCM_PATH:-/opt/rocm}"
fi

echo "==> HIP_VISIBLE_DEVICES=${HIP_VISIBLE_DEVICES}"
echo "==> JAX_PLATFORMS=${JAX_PLATFORMS}"
echo "==> XLA_PYTHON_CLIENT_MEM_FRACTION=${XLA_PYTHON_CLIENT_MEM_FRACTION}"

exec "${PYTHON_BIN}" "${SCRIPT_DIR}/generate_sdxl_jax.py" "$@"
