from __future__ import annotations

import argparse
import importlib.metadata
import json
import os
import re
import resource
import subprocess
import time
from pathlib import Path
from typing import Sequence

import mlx
import mlx.core as mx
import numpy as np

from .config import SenseNovaConfig
from .download import download_all, load_download_manifest, remaining_download_bytes
from .pipeline import SenseNovaT2IPipeline, required_t2i_weight_keys
from .prompt import build_t2i_query
from .weights import (
    FileRecord,
    ShardManifest,
    load_lora_adapters,
    load_sharded_weights,
    verify_file,
)


MODEL_NAME = "SenseNova-U1.5-8B-MoT"
BASE_REVISION = "1c75534f02a734f7bb6319a16fd82a67769d12bb"
LORA_REVISION = "da0b25652ee8f8b447bf817e4bfdbb4221d521ec"
LORA_RELATIVE_PATH = "lora/SenseNova-U1.5-8B-MoT-LoRA-8step.safetensors"


def default_model_dir() -> Path:
    explicit = os.environ.get("SENSENOVA_MODEL_DIR")
    if explicit:
        return Path(explicit).expanduser()
    cache_root = os.environ.get("XDG_CACHE_HOME")
    if cache_root:
        return Path(cache_root).expanduser() / "sensenova-mlx" / MODEL_NAME
    return Path.home() / ".cache" / "sensenova-mlx" / MODEL_NAME


def _mlx_version() -> str:
    version = getattr(mlx, "__version__", None)
    return str(version) if version is not None else importlib.metadata.version("mlx")


def _manifest_path() -> Path:
    project_copy = Path(__file__).parents[2] / "docs" / "download-manifest.json"
    if project_copy.is_file():
        return project_copy
    packaged = Path(__file__).parent / "data" / "download-manifest.json"
    if packaged.is_file():
        return packaged
    raise FileNotFoundError("packaged download-manifest.json is missing")


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        prog="sensenova-mlx",
        description="Formal SenseNova-U1.5-8B-MoT text-to-image inference on MLX GPU",
    )
    subcommands = parser.add_subparsers(dest="command", required=True)

    download = subcommands.add_parser("download", help="download pinned formal weights with resume and SHA-256")
    download.add_argument("--model-dir", type=Path, default=default_model_dir())
    download.add_argument("--manifest", type=Path, default=None)

    verify = subcommands.add_parser("verify", help="verify every formal model and LoRA file")
    verify.add_argument("--model-dir", type=Path, default=default_model_dir())
    verify.add_argument("--manifest", type=Path, default=None)

    generate = subcommands.add_parser("generate", help="generate one PNG from a prompt")
    generate.add_argument("--model-dir", type=Path, default=default_model_dir())
    generate.add_argument("--prompt", required=True)
    generate.add_argument("--output", type=Path, default=Path("output.png"))
    generate.add_argument("--profile", type=Path, default=None)
    generate.add_argument("--width", type=int, default=1024)
    generate.add_argument("--height", type=int, default=1024)
    generate.add_argument("--steps", type=int, default=50)
    generate.add_argument("--seed", type=int, default=42)
    generate.add_argument(
        "--cfg-scale",
        type=float,
        default=None,
        help="classifier-free guidance scale (default: 4 for base, official 1 for 8-step LoRA)",
    )
    generate.add_argument("--cfg-norm", choices=("none", "global", "channel"), default="none")
    generate.add_argument("--timestep-shift", type=float, default=3.0)
    generate.add_argument("--lora-8step", action="store_true")

    serve = subcommands.add_parser(
        "serve",
        help="serve one persistent MLX image model through an OpenAI-compatible local API",
    )
    serve.add_argument("--model-dir", type=Path, default=default_model_dir())
    serve.add_argument("--host", default="127.0.0.1")
    serve.add_argument("--port", type=int, default=11235)
    serve.add_argument("--mode", choices=("fast", "quality"), default="fast")
    serve.add_argument("--api-key-env", default="SENSENOVA_API_KEY")
    serve.add_argument(
        "--allow-remote",
        action="store_true",
        help="allow a non-loopback bind; requires a firewall and a strong API key",
    )
    return parser


def validate_args(args: argparse.Namespace) -> None:
    if args.command == "serve":
        if not 1 <= args.port <= 65535:
            raise ValueError("--port must be between 1 and 65535")
        if args.host not in {"127.0.0.1", "::1", "localhost"} and not args.allow_remote:
            raise ValueError("a non-loopback --host requires explicit --allow-remote")
        if not args.api_key_env.strip():
            raise ValueError("--api-key-env must not be empty")
        return
    if args.command != "generate":
        return
    if not args.prompt.strip():
        raise ValueError("--prompt must not be empty")
    if args.width <= 0 or args.height <= 0:
        raise ValueError("--width and --height must be positive")
    if args.steps <= 0:
        raise ValueError("--steps must be positive")
    if args.cfg_scale is not None and args.cfg_scale < 1.0:
        raise ValueError("--cfg-scale must be at least 1")
    if args.lora_8step and args.steps != 8:
        raise ValueError("the formal 8-step LoRA must run with exactly --steps 8")


def _effective_cfg_scale(args: argparse.Namespace) -> float:
    if args.cfg_scale is not None:
        return float(args.cfg_scale)
    return 1.0 if args.lora_8step else 4.0


def _shard_manifest() -> ShardManifest:
    records = load_download_manifest(_manifest_path())
    shard_records = tuple(
        FileRecord(Path(record.relative_path).name, record.size, record.sha256)
        for record in records
        if "/" not in record.relative_path and record.relative_path.endswith(".safetensors")
    )
    return ShardManifest(revision=BASE_REVISION, files=shard_records)


def _swap_used_bytes() -> int | None:
    try:
        result = subprocess.run(
            ["/usr/sbin/sysctl", "-n", "vm.swapusage"],
            check=True,
            capture_output=True,
            text=True,
            timeout=5,
        )
    except (FileNotFoundError, subprocess.SubprocessError):
        return None
    match = re.search(r"used = ([0-9.]+)([KMGTP])", result.stdout)
    if match is None:
        return None
    multiplier = {"K": 1024, "M": 1024**2, "G": 1024**3, "T": 1024**4, "P": 1024**5}
    return round(float(match.group(1)) * multiplier[match.group(2)])


def _load_pipeline(model_dir: Path, *, use_lora: bool) -> tuple[SenseNovaT2IPipeline, SenseNovaConfig]:
    config = SenseNovaConfig.from_dict(json.loads((model_dir / "config.json").read_text(encoding="utf-8")))
    required = required_t2i_weight_keys(config)
    weights = load_sharded_weights(
        model_dir,
        index_path=model_dir / "model.safetensors.index.json",
        manifest=_shard_manifest(),
        include=required,
        target_dtype=mx.bfloat16,
    )
    adapters = None
    if use_lora:
        lora_record = next(
            record
            for record in load_download_manifest(_manifest_path())
            if record.relative_path == LORA_RELATIVE_PATH
        )
        lora_path = model_dir / lora_record.relative_path
        verify_file(
            lora_path,
            FileRecord(lora_record.relative_path, lora_record.size, lora_record.sha256),
        )
        adapters = load_lora_adapters(lora_path, target_dtype=mx.bfloat16)
    return SenseNovaT2IPipeline(config, weights, adapters=adapters), config


def _token_ids(model_dir: Path, prompt: str, *, conditional: bool) -> mx.array:
    from transformers import AutoTokenizer

    tokenizer = AutoTokenizer.from_pretrained(
        model_dir,
        local_files_only=True,
        trust_remote_code=False,
    )
    query = build_t2i_query(prompt, conditional=conditional)
    token_ids = tokenizer.encode(query, add_special_tokens=True)
    if not token_ids:
        raise ValueError("tokenizer produced an empty prefix")
    return mx.array([token_ids], dtype=mx.int32)


def _save_normalized_png(image: mx.array, output: Path) -> dict[str, float | int]:
    from PIL import Image

    mx.eval(image)
    array = np.asarray(image.astype(mx.float32))
    if array.shape[0] != 1 or array.shape[1] != 3:
        raise ValueError(f"generated image must be [1,3,H,W], got {array.shape}")
    if not np.isfinite(array).all():
        raise FloatingPointError("generated image contains NaN or infinity")
    pixels = np.clip(array[0] * 0.5 + 0.5, 0.0, 1.0)
    uint8 = np.rint(pixels.transpose(1, 2, 0) * 255.0).astype(np.uint8)
    if int(uint8.min()) == int(uint8.max()):
        raise ValueError("generated image is a single flat color")
    output.parent.mkdir(parents=True, exist_ok=True)
    Image.fromarray(uint8).save(output)
    return {
        "pixel_min": int(uint8.min()),
        "pixel_max": int(uint8.max()),
        "pixel_mean": float(uint8.mean()),
        "pixel_std": float(uint8.std()),
    }


def _generate(args: argparse.Namespace) -> None:
    started = time.monotonic()
    cfg_scale = _effective_cfg_scale(args)
    swap_before = _swap_used_bytes()
    mx.reset_peak_memory()
    pipeline, config = _load_pipeline(args.model_dir, use_lora=args.lora_8step)
    if mx.get_active_memory() > 90 * 1024**3:
        raise MemoryError("formal weights alone exceeded the 90 GiB MLX active-memory stop limit")
    swap_after_load = _swap_used_bytes()
    if (
        swap_before is not None
        and swap_after_load is not None
        and swap_after_load - swap_before > 4 * 1024**3
    ):
        raise MemoryError("swap increased by more than 4 GiB while loading formal weights")
    config.validate_image_size(args.width, args.height)
    condition_ids = _token_ids(args.model_dir, args.prompt, conditional=True)
    uncondition_ids = (
        _token_ids(args.model_dir, "", conditional=False)
        if cfg_scale > 1.0
        else None
    )

    def report_step(completed: int, total: int) -> None:
        active = mx.get_active_memory()
        peak = mx.get_peak_memory()
        print(f"[step] {completed}/{total} active={active / 1024**3:.2f}GiB peak={peak / 1024**3:.2f}GiB")
        current_swap = _swap_used_bytes()
        if swap_before is not None and current_swap is not None and current_swap - swap_before > 4 * 1024**3:
            raise MemoryError(
                f"swap increased by {current_swap - swap_before} bytes, above the 4 GiB stop limit"
            )

    image = pipeline.generate_from_ids(
        condition_ids,
        uncondition_ids,
        width=args.width,
        height=args.height,
        num_steps=args.steps,
        seed=args.seed,
        cfg_scale=cfg_scale,
        cfg_norm=args.cfg_norm,
        timestep_shift=args.timestep_shift,
        max_active_bytes=90 * 1024**3,
        step_callback=report_step,
    )
    pixel_stats = _save_normalized_png(image, args.output)
    swap_after = _swap_used_bytes()
    profile = {
        "model": "SenseNova-U1.5-8B-MoT",
        "base_revision": BASE_REVISION,
        "lora_revision": LORA_REVISION if args.lora_8step else None,
        "mlx_version": _mlx_version(),
        "device": str(mx.default_device()),
        "prompt": args.prompt,
        "width": args.width,
        "height": args.height,
        "steps": args.steps,
        "seed": args.seed,
        "cfg_scale": cfg_scale,
        "cfg_norm": args.cfg_norm,
        "timestep_shift": args.timestep_shift,
        "elapsed_seconds": time.monotonic() - started,
        "mlx_active_bytes": mx.get_active_memory(),
        "mlx_cache_bytes": mx.get_cache_memory(),
        "mlx_peak_bytes": mx.get_peak_memory(),
        "process_max_rss_bytes": resource.getrusage(resource.RUSAGE_SELF).ru_maxrss,
        "swap_before_bytes": swap_before,
        "swap_after_bytes": swap_after,
        "swap_delta_bytes": (
            swap_after - swap_before
            if swap_before is not None and swap_after is not None
            else None
        ),
        "output": str(args.output.resolve()),
        **pixel_stats,
    }
    profile_path = args.profile or args.output.with_suffix(".profile.json")
    profile_path.parent.mkdir(parents=True, exist_ok=True)
    profile_path.write_text(json.dumps(profile, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
    if args.width == 1024 and args.height == 1024 and profile["mlx_peak_bytes"] > 70 * 1024**3:
        raise MemoryError(
            f"1024 MLX peak {profile['mlx_peak_bytes']} exceeded the 70 GiB acceptance limit"
        )
    print(f"[saved] {args.output}")
    print(f"[profile] {profile_path}")


def main(argv: Sequence[str] | None = None) -> int:
    parser = build_parser()
    args = parser.parse_args(argv)
    validate_args(args)
    manifest = None
    if hasattr(args, "manifest"):
        manifest = args.manifest or _manifest_path()
    if args.command == "download":
        download_all(manifest, args.model_dir)
    elif args.command == "verify":
        remaining = remaining_download_bytes(args.model_dir, load_download_manifest(manifest))
        if remaining:
            raise FileNotFoundError(f"formal download is incomplete by {remaining} bytes")
        print("[verified] all formal base and 8-step LoRA files")
    elif args.command == "generate":
        _generate(args)
    elif args.command == "serve":
        from .server import serve_from_args

        serve_from_args(args)
    else:
        parser.error(f"unknown command {args.command}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
