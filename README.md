# SDXL generation with JAX on 2× AMD MI50 (gfx906)

This repository runs Stable Diffusion XL text-to-image generation through the
Hugging Face Diffusers Flax/JAX pipeline on AMD Instinct MI50 16GB cards.
MI50 uses the `gfx906` architecture, so we install AMD's `gfx906` ROCm and
JAX nightly Python wheels from:

```text
https://rocm.nightlies.amd.com/v2-staging/gfx906/
```

> Note: AMD has stopped shipping ROCm support for MI50 in stable releases.
> The `v2-staging/gfx906` index is the only supported way to get a working
> JAX/PJRT stack on `gfx906` today. Treat the wheels as a coupled set: pin
> `jaxlib`, `jax-rocm7-plugin`, `jax-rocm7-pjrt`, and
> `rocm-sdk-libraries-gfx906` to the **same nightly date**.

## Quick start (host with MI50)

The MI50 host needs `/dev/kfd` and `/dev/dri` exposed and a kernel/`amdgpu`
module recent enough to support `gfx906`. Everything else (the ROCm 7
userspace) is bundled in the `rocm-sdk-libraries-gfx906` wheel.

### Option A: bare metal / venv

```bash
# Python 3.11, 3.12, 3.13 or 3.14 — the gfx906 index does not publish
# wheels for older interpreters. 3.12 is recommended.
python3.12 -m venv .venv && source .venv/bin/activate

# Install ROCm + JAX wheels for gfx906, plus Diffusers/Flax stack.
PYTHON_BIN=python scripts/install_rocm_jax.sh

# Or pin every wheel to the same nightly date for reproducibility:
PYTHON_BIN=python scripts/install_rocm_jax.sh 20260428
```

### Option B: Docker

```bash
docker build -f Dockerfile.mi50-jax -t sdxl-mi50-jax .
# or pinned:
docker build -f Dockerfile.mi50-jax \
  --build-arg ROCM_DATE=20260428 -t sdxl-mi50-jax:20260428 .

docker run --rm -it \
  --device=/dev/kfd --device=/dev/dri --group-add video \
  --ipc=host --shm-size 64G \
  --cap-add=SYS_PTRACE --security-opt seccomp=unconfined \
  -v "$PWD/outputs:/workspace/outputs" \
  -v "$HOME/.cache/huggingface:/root/.cache/huggingface" \
  sdxl-mi50-jax \
  scripts/run_sdxl_jax.sh --check-only --compile-smoke-test
```

If you use a private or gated Hugging Face model, log in once before running:

```bash
huggingface-cli login
```

## 1. Verify both MI50 cards

```bash
scripts/run_sdxl_jax.sh \
  --check-only \
  --compile-smoke-test \
  --devices 0,1 \
  --require-devices 2
```

Expected output: two ROCm devices (e.g. `rocm:0` and `rocm:1`) and
`compile smoke test ok`. If the smoke test fails, the ROCm/JAX nightly set is
not compiling on `gfx906` correctly — try pinning to an earlier nightly date
via `scripts/install_rocm_jax.sh <YYYYMMDD>`.

## 2. Generate images

One 1024×1024 image per MI50 (two images total):

```bash
scripts/run_sdxl_jax.sh \
  --devices 0,1 \
  --require-devices 2 \
  --prompt "cinematic photo of a futuristic research station on Mars, detailed, sharp focus" \
  --negative-prompt "low quality, blurry, distorted" \
  --steps 30 \
  --guidance-scale 5.0 \
  --images-per-device 1 \
  --dtype fp16 \
  --output-dir outputs/sdxl-mi50
```

Single-card run:

```bash
HIP_VISIBLE_DEVICES=0 scripts/run_sdxl_jax.sh \
  --require-devices 1 \
  --prompt "studio portrait of a snow leopard, detailed fur" \
  --steps 30 --dtype fp16 \
  --output-dir outputs/sdxl-mi50
```

The first run compiles the JAX program and downloads model weights, so it is
much slower than later runs with the same tensor shapes.

## What `scripts/run_sdxl_jax.sh` sets for you

The wrapper exports values that matter on `gfx906` 16GB:

- `JAX_PLATFORMS=rocm`
- `XLA_PYTHON_CLIENT_PREALLOCATE=false` — do not reserve all VRAM up front.
- `XLA_PYTHON_CLIENT_MEM_FRACTION=0.90` — soft cap below 16GB, headroom for
  SDXL UNet attention spikes.
- `MIOPEN_FIND_MODE=FAST` — skip MIOpen autotuning passes that can hang on
  Vega20.
- `HIP_VISIBLE_DEVICES=0,1` — both MI50s by default.

Override any of these by exporting them yourself before invoking the script.

## Notes for MI50 16GB

- Keep `--images-per-device 1` for 1024×1024 SDXL on 16GB GPUs.
- Use `--dtype fp16`; SDXL in fp32 will usually exceed 16GB per card.
- The script keeps scheduler parameters in fp32 even when model weights are
  cast to fp16. This avoids common SDXL scheduler precision failures.
- If the latest JAX ROCm wheels fail with Triton or `gfx906` compiler errors,
  install an earlier date from the same `v2-staging/gfx906` index by running
  `scripts/install_rocm_jax.sh <YYYYMMDD>` with a known-good date. The script
  pins `jaxlib`, `jax-rocm7-plugin`, `jax-rocm7-pjrt`, and
  `rocm-sdk-libraries-gfx906` to that same date.
- For diagnostics only, use `--no-jit`; normal generation should keep JIT on.
- If you see `XLA_PYTHON_CLIENT_MEM_FRACTION` related OOMs, lower it with
  `--mem-fraction 0.85` (or even `0.80`).
