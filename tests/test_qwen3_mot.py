from __future__ import annotations

import math
import unittest

import mlx.core as mx
import numpy as np
import torch

from sensenova_mlx.config import SenseNovaConfig
from sensenova_mlx.qwen3_mot import Qwen3MoTModel
from sensenova_mlx.rope import build_image_indexes, build_text_indexes, create_block_causal_mask


def _config() -> SenseNovaConfig:
    return SenseNovaConfig(
        hidden_size=32,
        intermediate_size=64,
        num_hidden_layers=1,
        num_attention_heads=4,
        num_key_value_heads=2,
        head_dim=8,
        vocab_size=32,
        rms_norm_eps=1e-6,
        rope_theta=5_000_000.0,
        rope_theta_hw=10_000.0,
        max_position_embeddings=1024,
        max_position_embeddings_hw=128,
        patch_size=16,
        downsample_ratio=0.5,
        max_image_seq_len=4096,
        noise_scale=1.0,
        noise_scale_base_image_seq_len=64,
        noise_scale_max_value=16.0,
        t_eps=0.05,
        template="neo1_0",
    )


def _make_weights() -> dict[str, np.ndarray]:
    rng = np.random.default_rng(19)

    def normal(shape: tuple[int, ...], scale: float = 0.08) -> np.ndarray:
        return (rng.normal(size=shape) * scale).astype(np.float32)

    weights: dict[str, np.ndarray] = {
        "language_model.model.norm.weight": 1.0 + normal((32,), 0.02),
        "language_model.model.norm_mot_gen.weight": 1.0 + normal((32,), 0.02),
    }
    root = "language_model.model.layers.0"
    for suffix in ("", "_mot_gen"):
        weights[f"{root}.input_layernorm{suffix}.weight"] = 1.0 + normal((32,), 0.02)
        weights[f"{root}.post_attention_layernorm{suffix}.weight"] = 1.0 + normal((32,), 0.02)
    for name, out_dim in (("q", 32), ("k", 16), ("v", 16), ("o", 32)):
        in_dim = 32
        weights[f"{root}.self_attn.{name}_proj.weight"] = normal((out_dim, in_dim))
        weights[f"{root}.self_attn.{name}_proj_mot_gen.weight"] = normal((out_dim, in_dim))
    for name in ("q_norm", "q_norm_hw", "k_norm", "k_norm_hw"):
        weights[f"{root}.self_attn.{name}.weight"] = 1.0 + normal((4,), 0.02)
        weights[f"{root}.self_attn.{name}_mot_gen.weight"] = 1.0 + normal((4,), 0.02)
    for expert in ("mlp", "mlp_mot_gen"):
        weights[f"{root}.{expert}.gate_proj.weight"] = normal((64, 32))
        weights[f"{root}.{expert}.up_proj.weight"] = normal((64, 32))
        weights[f"{root}.{expert}.down_proj.weight"] = normal((32, 64))
    return weights


def _rms(x: torch.Tensor, weight: np.ndarray) -> torch.Tensor:
    normalized = x.float() * torch.rsqrt(x.float().square().mean(-1, keepdim=True) + 1e-6)
    return normalized * torch.from_numpy(weight)


def _rope(x: torch.Tensor, positions: np.ndarray, theta: float) -> torch.Tensor:
    dim = x.shape[-1]
    inv = 1.0 / (theta ** (torch.arange(0, dim, 2, dtype=torch.float32) / dim))
    angles = torch.from_numpy(positions.astype(np.float32))[:, None] * inv[None]
    cos = torch.cat((angles.cos(), angles.cos()), dim=-1)[None, None]
    sin = torch.cat((angles.sin(), angles.sin()), dim=-1)[None, None]
    half = dim // 2
    rotated = torch.cat((-x[..., half:], x[..., :half]), dim=-1)
    return x * cos + rotated * sin


def _reference_layer(
    x_np: np.ndarray,
    weights: dict[str, np.ndarray],
    indexes: np.ndarray,
    *,
    expert: str,
    prefix: tuple[torch.Tensor, torch.Tensor] | None,
    mask: np.ndarray | None,
) -> tuple[torch.Tensor, tuple[torch.Tensor, torch.Tensor]]:
    root = "language_model.model.layers.0"
    suffix = "" if expert == "und" else "_mot_gen"
    mlp = "mlp" if expert == "und" else "mlp_mot_gen"
    x = torch.from_numpy(x_np)
    residual = x
    normed = _rms(x, weights[f"{root}.input_layernorm{suffix}.weight"])

    def linear(name: str, value: torch.Tensor) -> torch.Tensor:
        return value @ torch.from_numpy(weights[name]).T

    q = linear(f"{root}.self_attn.q_proj{suffix}.weight", normed).reshape(1, x.shape[1], 4, 8)
    k = linear(f"{root}.self_attn.k_proj{suffix}.weight", normed).reshape(1, x.shape[1], 2, 8)
    v = linear(f"{root}.self_attn.v_proj{suffix}.weight", normed).reshape(1, x.shape[1], 2, 8)
    q_t = _rms(q[..., :4], weights[f"{root}.self_attn.q_norm{suffix}.weight"]).transpose(1, 2)
    q_hw = _rms(q[..., 4:], weights[f"{root}.self_attn.q_norm_hw{suffix}.weight"]).transpose(1, 2)
    k_t = _rms(k[..., :4], weights[f"{root}.self_attn.k_norm{suffix}.weight"]).transpose(1, 2)
    k_hw = _rms(k[..., 4:], weights[f"{root}.self_attn.k_norm_hw{suffix}.weight"]).transpose(1, 2)
    q_t = _rope(q_t, indexes[0], 5_000_000.0)
    k_t = _rope(k_t, indexes[0], 5_000_000.0)
    q_h = _rope(q_hw[..., :2], indexes[1], 10_000.0)
    k_h = _rope(k_hw[..., :2], indexes[1], 10_000.0)
    q_w = _rope(q_hw[..., 2:], indexes[2], 10_000.0)
    k_w = _rope(k_hw[..., 2:], indexes[2], 10_000.0)
    q = torch.cat((q_t, q_h, q_w), dim=-1)
    k = torch.cat((k_t, k_h, k_w), dim=-1)
    v = v.transpose(1, 2)
    current = (k, v)
    if prefix is not None:
        k = torch.cat((prefix[0], k), dim=2)
        v = torch.cat((prefix[1], v), dim=2)
    k_attn = torch.repeat_interleave(k, 2, dim=1)
    v_attn = torch.repeat_interleave(v, 2, dim=1)
    scores = (q @ k_attn.transpose(-1, -2)) / math.sqrt(8)
    if mask is not None:
        scores = scores + torch.from_numpy(mask)
    probs = torch.softmax(scores.float(), dim=-1)
    attended = (probs @ v_attn).transpose(1, 2).reshape(1, x.shape[1], 32)
    x = residual + linear(f"{root}.self_attn.o_proj{suffix}.weight", attended)
    residual = x
    normed = _rms(x, weights[f"{root}.post_attention_layernorm{suffix}.weight"])
    gated = torch.nn.functional.silu(linear(f"{root}.{mlp}.gate_proj.weight", normed))
    gated = gated * linear(f"{root}.{mlp}.up_proj.weight", normed)
    x = residual + linear(f"{root}.{mlp}.down_proj.weight", gated)
    return x, current


class Qwen3MoTLayerTest(unittest.TestCase):
    def test_prefix_cache_and_generation_expert_match_independent_reference(self) -> None:
        rng = np.random.default_rng(23)
        weights_np = _make_weights()
        model = Qwen3MoTModel(_config(), {key: mx.array(value) for key, value in weights_np.items()})
        prefix_np = (rng.normal(size=(1, 3, 32)) * 0.1).astype(np.float32)
        gen_np = (rng.normal(size=(1, 4, 32)) * 0.1).astype(np.float32)
        text_indexes = build_text_indexes(3)
        image_indexes = build_image_indexes(2, 2, text_len=3)
        mask = create_block_causal_mask(text_indexes[0])

        prefix_out, caches = model.forward_prefix_embeddings(mx.array(prefix_np), text_indexes, mask)
        gen_out = model.forward_generation(mx.array(gen_np), image_indexes, caches)
        mx.eval(prefix_out, gen_out)

        prefix_ref, prefix_kv = _reference_layer(
            prefix_np,
            weights_np,
            np.asarray(text_indexes),
            expert="und",
            prefix=None,
            mask=np.asarray(mask),
        )
        prefix_ref = _rms(prefix_ref, weights_np["language_model.model.norm.weight"])
        gen_ref, _ = _reference_layer(
            gen_np,
            weights_np,
            np.asarray(image_indexes),
            expert="gen",
            prefix=prefix_kv,
            mask=None,
        )
        gen_ref = _rms(gen_ref, weights_np["language_model.model.norm_mot_gen.weight"])

        np.testing.assert_allclose(np.asarray(prefix_out), prefix_ref.numpy(), atol=1e-4, rtol=1e-4)
        np.testing.assert_allclose(np.asarray(gen_out), gen_ref.numpy(), atol=1e-4, rtol=1e-4)
        self.assertEqual(caches[0].length, 3)


if __name__ == "__main__":
    unittest.main()
