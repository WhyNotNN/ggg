#!/usr/bin/env python3
"""Generate SDXL images with the Diffusers Flax/JAX pipeline on ROCm GPUs."""

from __future__ import annotations

import argparse
import os
import sys
import time
from pathlib import Path
from typing import Any


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Run Stable Diffusion XL text-to-image generation with JAX/Flax."
    )
    parser.add_argument(
        "--model-id",
        default="stabilityai/stable-diffusion-xl-base-1.0",
        help="Hugging Face model id or local model directory.",
    )
    parser.add_argument(
        "--prompt",
        action="append",
        required=False,
        default=None,
        help=(
            "Prompt to render. Pass once to repeat it on every GPU, or pass exactly "
            "device_count * images_per_device prompts."
        ),
    )
    parser.add_argument(
        "--negative-prompt",
        action="append",
        default=None,
        help=(
            "Negative prompt. Pass once to repeat it for every image, or pass exactly "
            "device_count * images_per_device values."
        ),
    )
    parser.add_argument("--output-dir", default="outputs/sdxl-jax", help="Directory for PNG outputs.")
    parser.add_argument("--seed", type=int, default=33, help="Base random seed.")
    parser.add_argument("--steps", type=int, default=30, help="Number of inference steps.")
    parser.add_argument("--guidance-scale", type=float, default=5.0, help="Classifier-free guidance scale.")
    parser.add_argument("--height", type=int, default=1024, help="Output image height.")
    parser.add_argument("--width", type=int, default=1024, help="Output image width.")
    parser.add_argument(
        "--dtype",
        choices=("fp16", "fp32", "bf16"),
        default="fp16",
        help="Model parameter dtype. MI50 should normally use fp16.",
    )
    parser.add_argument(
        "--images-per-device",
        type=int,
        default=1,
        help="Per-GPU batch size. Keep this at 1 on 16GB cards unless you have memory headroom.",
    )
    parser.add_argument(
        "--devices",
        default=None,
        help="Comma-separated HIP_VISIBLE_DEVICES value, for example '0,1'. Must be set before JAX loads.",
    )
    parser.add_argument(
        "--require-devices",
        type=int,
        default=2,
        help="Fail unless JAX sees this many devices. Set to 0 to disable the check.",
    )
    parser.add_argument(
        "--platform",
        default="rocm",
        help="JAX platform name. Use 'rocm' on AMD GPUs or 'cpu' only for diagnostics.",
    )
    parser.add_argument(
        "--check-only",
        action="store_true",
        help="Print JAX devices and exit before loading SDXL.",
    )
    parser.add_argument(
        "--compile-smoke-test",
        action="store_true",
        help="During --check-only, compile a tiny pmap workload on all visible devices.",
    )
    parser.add_argument(
        "--no-jit",
        action="store_true",
        help="Disable JIT compilation. Useful only for debugging; generation will be much slower.",
    )
    parser.add_argument(
        "--split-head-dim",
        action=argparse.BooleanOptionalAction,
        default=True,
        help="Use split attention head dimension when loading the SDXL Flax pipeline.",
    )
    return parser.parse_args()


def configure_environment(args: argparse.Namespace) -> None:
    if args.devices:
        os.environ["HIP_VISIBLE_DEVICES"] = args.devices

    if args.platform:
        os.environ.setdefault("JAX_PLATFORM_NAME", args.platform)

    if Path("/opt/rocm").exists():
        os.environ.setdefault("ROCM_PATH", "/opt/rocm")

    # Avoid reserving all VRAM up front; this makes OOM behavior easier to diagnose on 16GB cards.
    os.environ.setdefault("XLA_PYTHON_CLIENT_PREALLOCATE", "false")


def dtype_from_name(name: str) -> Any:
    import jax.numpy as jnp

    if name == "fp16":
        return jnp.float16
    if name == "bf16":
        return jnp.bfloat16
    return jnp.float32


def cast_params_for_inference(params: dict[str, Any], dtype: Any) -> dict[str, Any]:
    import jax
    import jax.numpy as jnp

    if dtype == jnp.float32:
        return params

    scheduler_state = params.get("scheduler")
    model_params = {key: value for key, value in params.items() if key != "scheduler"}

    def maybe_cast(value: Any) -> Any:
        if hasattr(value, "dtype") and jnp.issubdtype(value.dtype, jnp.floating):
            return value.astype(dtype)
        return value

    casted = jax.tree_util.tree_map(maybe_cast, model_params)
    if scheduler_state is not None:
        # Scheduler sigmas are numerically sensitive for SDXL; keep them in fp32.
        casted["scheduler"] = scheduler_state
    return casted


def expand_values(values: list[str] | None, total: int, default: str, label: str) -> list[str]:
    if not values:
        return [default] * total
    if len(values) == 1:
        return values * total
    if len(values) != total:
        raise ValueError(f"{label} count must be 1 or {total}, got {len(values)}")
    return values


def save_images(pipeline: Any, images: Any, output_dir: Path, seed: int) -> None:
    import numpy as np

    images = images.reshape((images.shape[0] * images.shape[1],) + images.shape[-3:])
    pil_images = pipeline.numpy_to_pil(np.asarray(images))
    output_dir.mkdir(parents=True, exist_ok=True)
    for index, image in enumerate(pil_images):
        path = output_dir / f"sdxl_seed-{seed}_image-{index:02d}.png"
        image.save(path)
        print(f"saved {path}")


def run_compile_smoke_test(device_count: int) -> None:
    import jax
    import jax.numpy as jnp

    @jax.pmap
    def smoke(x: Any) -> Any:
        return jax.nn.softmax(x @ jnp.swapaxes(x, -1, -2), axis=-1)

    x = jnp.ones((device_count, 8, 8), dtype=jnp.float32)
    y = smoke(x)
    y.block_until_ready()
    print(f"compile smoke test ok: shape={y.shape}, dtype={y.dtype}")


def main() -> int:
    args = parse_args()
    configure_environment(args)

    import jax

    devices = jax.devices()
    print(f"jax backend: {jax.default_backend()}")
    print(f"jax devices ({len(devices)}): {devices}")

    if args.require_devices and len(devices) != args.require_devices:
        print(
            f"expected {args.require_devices} JAX devices, but found {len(devices)}",
            file=sys.stderr,
        )
        return 2

    if args.check_only:
        if args.compile_smoke_test:
            run_compile_smoke_test(len(devices))
        return 0

    if args.prompt is None:
        print("--prompt is required unless --check-only is used", file=sys.stderr)
        return 2

    from diffusers import FlaxStableDiffusionXLPipeline
    import numpy as np
    from flax.jax_utils import replicate
    from flax.training.common_utils import shard

    dtype = dtype_from_name(args.dtype)
    total_images = len(devices) * args.images_per_device
    prompts = expand_values(args.prompt, total_images, default="", label="prompt")
    negative_prompts = expand_values(
        args.negative_prompt,
        total_images,
        default="",
        label="negative prompt",
    )

    print(f"loading {args.model_id} with dtype={args.dtype}")
    pipeline, params = FlaxStableDiffusionXLPipeline.from_pretrained(
        args.model_id,
        dtype=dtype,
        split_head_dim=args.split_head_dim,
    )
    params = cast_params_for_inference(params, dtype)
    p_params = replicate(params)

    prompt_ids = shard(pipeline.prepare_inputs(prompts))
    negative_prompt_ids = shard(pipeline.prepare_inputs(negative_prompts))
    rng = shard(jax.random.split(jax.random.PRNGKey(args.seed), total_images))

    print(
        "generating "
        f"{total_images} image(s): {args.width}x{args.height}, "
        f"steps={args.steps}, guidance={args.guidance_scale}, jit={not args.no_jit}"
    )
    started = time.time()
    output = pipeline(
        prompt_ids,
        p_params,
        rng,
        num_inference_steps=args.steps,
        height=args.height,
        width=args.width,
        guidance_scale=args.guidance_scale,
        neg_prompt_ids=negative_prompt_ids,
        jit=not args.no_jit,
    )
    images = output.images
    if hasattr(images, "block_until_ready"):
        images.block_until_ready()
    print(f"inference finished in {time.time() - started:.2f}s")

    save_images(pipeline, np.asarray(images), Path(args.output_dir), args.seed)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
