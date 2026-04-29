# SDXL generation with JAX on 2x AMD MI50

This repository contains a small runner for Stable Diffusion XL text-to-image
generation through the Hugging Face Diffusers Flax/JAX pipeline.

The target machine is 2x AMD Instinct MI50 16GB. MI50 uses the `gfx906`
architecture. The setup below installs AMD ROCm nightly Python wheels from:

```text
https://rocm.nightlies.amd.com/v2-staging/gfx906/
```

## 1. Start a Python container with host ROCm devices

On the MI50 host, start from a plain Python image and mount the repository:

```bash
docker run --rm -it \
  --device=/dev/kfd \
  --device=/dev/dri \
  --group-add video \
  --ipc=host \
  --shm-size 64G \
  --cap-add=SYS_PTRACE \
  --security-opt seccomp=unconfined \
  -v "$PWD:/workspace" \
  -w /workspace \
  python:3.12-bookworm \
  bash
```

Inside the container, install the ROCm/JAX stack first. Keep these packages on
the same nightly date; do not let a later `pip install` replace them with PyPI
CPU wheels.

```bash
export HIP_VISIBLE_DEVICES=0,1
export JAX_PLATFORM_NAME=rocm
export XLA_PYTHON_CLIENT_PREALLOCATE=false

python3 -m pip install --upgrade pip
python3 -m pip install \
  --index-url https://rocm.nightlies.amd.com/v2-staging/gfx906/ \
  "rocm-sdk-libraries-gfx906==7.13.0a20260428" \
  "jaxlib==0.9.0+rocm7.13.0a20260428" \
  "jax-rocm7-plugin==0.9.1+rocm7.13.0a20260428" \
  "jax-rocm7-pjrt==0.9.1+rocm7.13.0a20260428"
python3 -m pip install "jax==0.9.0"

python3 -m pip install \
  --constraint constraints-mi50-jax.txt \
  -r requirements-mi50-jax.txt
```

`requirements-mi50-jax.txt` intentionally does not include `jax`, `jaxlib`,
`jax_rocm7_plugin`, `jax_rocm7_pjrt`, `accelerate`, `torch`, or `torchvision`.
The constraints file pins the already-installed JAX/ROCm stack so installing
Diffusers/Flax dependencies does not replace it.

If you use a private or gated Hugging Face model, log in first:

```bash
huggingface-cli login
```

## 2. Verify that JAX sees both MI50 cards

```bash
python3 scripts/generate_sdxl_jax.py \
  --check-only \
  --compile-smoke-test \
  --devices 0,1 \
  --require-devices 2
```

Expected output should include two ROCm devices, for example `rocm:0` and
`rocm:1`, plus `compile smoke test ok`. If the smoke test fails, the ROCm/JAX
nightly set is not compiling on `gfx906` correctly.

## 3. Generate images

This command generates one 1024x1024 image on each MI50, so two images total:

```bash
python3 scripts/generate_sdxl_jax.py \
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

The first run compiles the JAX program and downloads model weights, so it is
much slower than later runs with the same tensor shapes.

## Notes for MI50 16GB

- Keep `--images-per-device 1` for 1024x1024 SDXL on 16GB GPUs.
- Use `--dtype fp16`; SDXL in fp32 will usually exceed 16GB per card.
- The script keeps scheduler parameters in fp32 even when model weights are
  cast to fp16. This avoids common SDXL scheduler precision failures.
- If the latest JAX ROCm wheels fail with Triton or `gfx906` compiler errors,
  install an earlier date from the same `v2-staging/gfx906` index by pinning
  matching `jaxlib`, `jax_rocm7_plugin`, `jax_rocm7_pjrt`, and
  `rocm-sdk-libraries-gfx906` wheel versions from the same date.
- For diagnostics only, use `--no-jit`; normal generation should keep JIT on.
