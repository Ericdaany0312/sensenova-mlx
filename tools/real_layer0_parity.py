from __future__ import annotations

import argparse
import json
import math
from dataclasses import replace
from pathlib import Path
from typing import Sequence

import mlx.core as mx
import numpy as np
import torch
import torch.nn.functional as torch_f
from safetensors import safe_open

from sensenova_mlx.attention import KVCache
from sensenova_mlx.config import SenseNovaConfig
from sensenova_mlx.generate import default_model_dir
from sensenova_mlx.qwen3_mot import Qwen3MoTModel
from sensenova_mlx.rope import build_image_indexes, build_text_indexes, create_block_causal_mask


DEFAULT_OUTPUT = Path(__file__).parents[1] / "artifacts" / "real-layer0-parity.json"


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="Compare formal layer 0 between PyTorch CPU and MLX")
    parser.add_argument("--model-dir", type=Path, default=default_model_dir())
    parser.add_argument("--output", type=Path, default=DEFAULT_OUTPUT)
    return parser


def _layer_keys() -> list[str]:
    root = "language_model.model.layers.0"
    keys: list[str] = []
    for suffix, mlp in (("", "mlp"), ("_mot_gen", "mlp_mot_gen")):
        keys.extend(
            (
                f"{root}.input_layernorm{suffix}.weight",
                f"{root}.post_attention_layernorm{suffix}.weight",
            )
        )
        keys.extend(f"{root}.self_attn.{name}_proj{suffix}.weight" for name in ("q", "k", "v", "o"))
        keys.extend(
            f"{root}.self_attn.{name}{suffix}.weight"
            for name in ("q_norm", "q_norm_hw", "k_norm", "k_norm_hw")
        )
        keys.extend(f"{root}.{mlp}.{name}_proj.weight" for name in ("gate", "up", "down"))
    return keys


def _rms(x: torch.Tensor, weight: torch.Tensor, eps: float) -> torch.Tensor:
    normalized = x.float() * torch.rsqrt(x.float().square().mean(-1, keepdim=True) + eps)
    return weight * normalized.to(x.dtype)


def _rope(x: torch.Tensor, positions: np.ndarray, theta: float) -> torch.Tensor:
    dimension = x.shape[-1]
    inverse = 1.0 / (theta ** (torch.arange(0, dimension, 2).float() / dimension))
    angles = torch.from_numpy(positions.astype(np.float32))[:, None] * inverse[None]
    frequencies = torch.cat((angles, angles), dim=-1)[None, None]
    half = dimension // 2
    rotated = torch.cat((-x[..., half:], x[..., :half]), dim=-1)
    return x * frequencies.cos().to(x.dtype) + rotated * frequencies.sin().to(x.dtype)


def _reference_layer(
    hidden: torch.Tensor,
    weights: dict[str, torch.Tensor],
    indexes: np.ndarray,
    config: SenseNovaConfig,
    *,
    expert: str,
    prefix: tuple[torch.Tensor, torch.Tensor] | None,
    mask: np.ndarray | None,
) -> tuple[torch.Tensor, tuple[torch.Tensor, torch.Tensor]]:
    root = "language_model.model.layers.0"
    suffix = "" if expert == "und" else "_mot_gen"
    mlp = "mlp" if expert == "und" else "mlp_mot_gen"

    residual = hidden
    normalized = _rms(hidden, weights[f"{root}.input_layernorm{suffix}.weight"], config.rms_norm_eps)
    q = torch_f.linear(normalized, weights[f"{root}.self_attn.q_proj{suffix}.weight"])
    k = torch_f.linear(normalized, weights[f"{root}.self_attn.k_proj{suffix}.weight"])
    v = torch_f.linear(normalized, weights[f"{root}.self_attn.v_proj{suffix}.weight"])
    q = q.reshape(1, hidden.shape[1], config.num_attention_heads, config.head_dim)
    k = k.reshape(1, hidden.shape[1], config.num_key_value_heads, config.head_dim)
    v = v.reshape(1, hidden.shape[1], config.num_key_value_heads, config.head_dim)
    half = config.head_dim // 2
    q_t = _rms(q[..., :half], weights[f"{root}.self_attn.q_norm{suffix}.weight"], config.rms_norm_eps)
    q_hw = _rms(q[..., half:], weights[f"{root}.self_attn.q_norm_hw{suffix}.weight"], config.rms_norm_eps)
    k_t = _rms(k[..., :half], weights[f"{root}.self_attn.k_norm{suffix}.weight"], config.rms_norm_eps)
    k_hw = _rms(k[..., half:], weights[f"{root}.self_attn.k_norm_hw{suffix}.weight"], config.rms_norm_eps)
    q_t, q_hw = q_t.transpose(1, 2), q_hw.transpose(1, 2)
    k_t, k_hw = k_t.transpose(1, 2), k_hw.transpose(1, 2)
    quarter = config.head_dim // 4
    q = torch.cat(
        (
            _rope(q_t, indexes[0], config.rope_theta),
            _rope(q_hw[..., :quarter], indexes[1], config.rope_theta_hw),
            _rope(q_hw[..., quarter:], indexes[2], config.rope_theta_hw),
        ),
        dim=-1,
    )
    k = torch.cat(
        (
            _rope(k_t, indexes[0], config.rope_theta),
            _rope(k_hw[..., :quarter], indexes[1], config.rope_theta_hw),
            _rope(k_hw[..., quarter:], indexes[2], config.rope_theta_hw),
        ),
        dim=-1,
    )
    v = v.transpose(1, 2)
    current = (k, v)
    if prefix is not None:
        k = torch.cat((prefix[0], k), dim=2)
        v = torch.cat((prefix[1], v), dim=2)
    repeats = config.num_attention_heads // config.num_key_value_heads
    k = torch.repeat_interleave(k, repeats, dim=1)
    v = torch.repeat_interleave(v, repeats, dim=1)
    scores = torch.matmul(q.float(), k.float().transpose(-1, -2)) / math.sqrt(config.head_dim)
    if mask is not None:
        scores = scores + torch.from_numpy(mask)
    probabilities = torch.softmax(scores, dim=-1)
    attended = torch.matmul(probabilities, v.float()).to(hidden.dtype)
    attended = attended.transpose(1, 2).reshape(1, hidden.shape[1], config.hidden_size)
    hidden = residual + torch_f.linear(attended, weights[f"{root}.self_attn.o_proj{suffix}.weight"])
    residual = hidden
    normalized = _rms(
        hidden,
        weights[f"{root}.post_attention_layernorm{suffix}.weight"],
        config.rms_norm_eps,
    )
    gate = torch_f.silu(torch_f.linear(normalized, weights[f"{root}.{mlp}.gate_proj.weight"]))
    up = torch_f.linear(normalized, weights[f"{root}.{mlp}.up_proj.weight"])
    hidden = residual + torch_f.linear(gate * up, weights[f"{root}.{mlp}.down_proj.weight"])
    return hidden, current


def _metrics(actual: np.ndarray, expected: np.ndarray) -> dict[str, float]:
    left = actual.astype(np.float64).reshape(-1)
    right = expected.astype(np.float64).reshape(-1)
    cosine = float(np.dot(left, right) / (np.linalg.norm(left) * np.linalg.norm(right)))
    return {
        "cosine": cosine,
        "max_abs": float(np.max(np.abs(left - right))),
        "mean_abs": float(np.mean(np.abs(left - right))),
    }


def main(argv: Sequence[str] | None = None) -> int:
    args = build_parser().parse_args(argv)
    shard = args.model_dir / "model-00006-of-00016.safetensors"
    if shard.stat().st_size != 7_120_064_448:
        raise ValueError(f"formal layer-0 shard has wrong size: {shard.stat().st_size}")
    config = SenseNovaConfig.from_dict(
        json.loads((args.model_dir / "config.json").read_text(encoding="utf-8"))
    )
    one_layer = replace(config, num_hidden_layers=1)
    keys = _layer_keys()

    raw_mlx = mx.load(str(shard))
    mlx_weights = {key: raw_mlx[key].astype(mx.bfloat16) for key in keys}
    mlx_weights["language_model.model.norm.weight"] = mx.ones((config.hidden_size,), dtype=mx.bfloat16)
    mlx_weights["language_model.model.norm_mot_gen.weight"] = mx.ones(
        (config.hidden_size,), dtype=mx.bfloat16
    )
    mx.eval(*mlx_weights.values())
    model = Qwen3MoTModel(one_layer, mlx_weights)

    with safe_open(shard, framework="pt", device="cpu") as handle:
        torch_weights = {key: handle.get_tensor(key).to(torch.bfloat16) for key in keys}

    rng = np.random.default_rng(101)
    prefix_np = (rng.normal(size=(1, 3, config.hidden_size)) * 0.1).astype(np.float32)
    generation_np = (rng.normal(size=(1, 4, config.hidden_size)) * 0.1).astype(np.float32)
    text_indexes = build_text_indexes(3)
    image_indexes = build_image_indexes(2, 2, text_len=3)
    mask = create_block_causal_mask(text_indexes[0])

    cache = KVCache()
    mlx_prefix = model._layer(
        mx.array(prefix_np).astype(mx.bfloat16),
        text_indexes,
        cache,
        layer_idx=0,
        expert="und",
        mask=mask,
        update_cache=True,
    )
    mlx_generation = model._layer(
        mx.array(generation_np).astype(mx.bfloat16),
        image_indexes,
        cache,
        layer_idx=0,
        expert="gen",
        mask=None,
        update_cache=False,
    )
    mx.eval(mlx_prefix, mlx_generation)

    torch_prefix, prefix_kv = _reference_layer(
        torch.from_numpy(prefix_np).to(torch.bfloat16),
        torch_weights,
        np.asarray(text_indexes),
        config,
        expert="und",
        prefix=None,
        mask=np.asarray(mask),
    )
    torch_generation, _ = _reference_layer(
        torch.from_numpy(generation_np).to(torch.bfloat16),
        torch_weights,
        np.asarray(image_indexes),
        config,
        expert="gen",
        prefix=prefix_kv,
        mask=None,
    )
    results = {
        "model": "SenseNova-U1.5-8B-MoT",
        "revision": "1c75534f02a734f7bb6319a16fd82a67769d12bb",
        "shard": shard.name,
        "dtype": "BF16 (formal F32 generation expert cast to BF16 on both sides)",
        "prefix": _metrics(np.asarray(mlx_prefix.astype(mx.float32)), torch_prefix.float().numpy()),
        "generation_with_prefix_cache": _metrics(
            np.asarray(mlx_generation.astype(mx.float32)),
            torch_generation.float().numpy(),
        ),
        "threshold": {"cosine_min": 0.999},
    }
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(results, indent=2) + "\n", encoding="utf-8")
    print(json.dumps(results, indent=2))
    if results["prefix"]["cosine"] < 0.999 or results["generation_with_prefix_cache"]["cosine"] < 0.999:
        raise AssertionError("real layer-0 parity cosine fell below 0.999")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
