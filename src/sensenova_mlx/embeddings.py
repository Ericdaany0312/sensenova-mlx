from __future__ import annotations

import math

import mlx.core as mx

from .weights import linear


def _gelu_exact(x: mx.array) -> mx.array:
    return 0.5 * x * (1.0 + mx.erf(x / math.sqrt(2.0)))


def _interleaved_rope(x: mx.array, positions: mx.array, theta: float) -> mx.array:
    """Apply the pairwise RoPE convention used by the official vision embedder."""
    dimension = x.shape[-1]
    if dimension % 2:
        raise ValueError(f"vision RoPE dimension must be even, got {dimension}")
    inverse = 1.0 / (theta ** (mx.arange(0, dimension, 2, dtype=mx.float32) / dimension))
    angles = positions.astype(mx.float32)[:, None] * inverse[None, :]
    cosine = mx.cos(angles)[None, :, :]
    sine = mx.sin(angles)[None, :, :]
    even = x[..., 0::2].astype(mx.float32)
    odd = x[..., 1::2].astype(mx.float32)
    paired = mx.stack((even * cosine - odd * sine, even * sine + odd * cosine), axis=-1)
    return paired.reshape(x.shape).astype(x.dtype)


def generation_vision_embeddings(
    patches: mx.array,
    *,
    grid_h: int,
    grid_w: int,
    patch_weight_oihw: mx.array,
    patch_bias: mx.array,
    dense_weight_oihw: mx.array,
    dense_bias: mx.array,
    rope_theta: float = 10_000.0,
) -> mx.array:
    """Port ``NEOVisionEmbeddings.forward`` for the generation branch.

    Source: OpenSenseNova/SenseNova-U1 ``modeling_neo_vit.py`` lines 859-1010.
    Input patches use the official channel-first flattened layout [B,L,3*P*P].
    """
    if patches.ndim != 3:
        raise ValueError(f"patches must be [B,L,C*P*P], got {patches.shape}")
    batch, length, patch_features = patches.shape
    if length != grid_h * grid_w:
        raise ValueError(f"grid {grid_h}x{grid_w} does not match {length} patches")
    out_channels, in_channels, patch_h, patch_w = patch_weight_oihw.shape
    if patch_h != patch_w or patch_features != in_channels * patch_h * patch_w:
        raise ValueError(
            f"patch weight {patch_weight_oihw.shape} does not match flattened feature {patch_features}"
        )
    if out_channels % 4:
        raise ValueError(f"vision embedding width must split into even x/y halves, got {out_channels}")

    # Kernel and stride equal the patch size, so each independent patch produces
    # one value and is exactly equivalent to a flattened OIHW linear projection.
    hidden = linear(patches, patch_weight_oihw.reshape(out_channels, -1), patch_bias)
    hidden = _gelu_exact(hidden)
    flat = mx.arange(length, dtype=mx.int32)
    columns = flat % grid_w
    rows = flat // grid_w
    half = out_channels // 2
    hidden = mx.concatenate(
        (
            _interleaved_rope(hidden[..., :half], columns, rope_theta),
            _interleaved_rope(hidden[..., half:], rows, rope_theta),
        ),
        axis=-1,
    )

    dense_out, dense_in, merge_h, merge_w = dense_weight_oihw.shape
    if dense_in != out_channels or merge_h != merge_w:
        raise ValueError(f"unsupported dense embedding weight shape {dense_weight_oihw.shape}")
    if grid_h % merge_h or grid_w % merge_w:
        raise ValueError(f"grid {grid_h}x{grid_w} is not divisible by merge size {merge_h}")
    hidden = hidden.reshape(batch, grid_h, grid_w, out_channels)
    dense_weight_ohwi = dense_weight_oihw.transpose(0, 2, 3, 1)
    hidden = mx.conv2d(hidden, dense_weight_ohwi, stride=merge_h, padding=0) + dense_bias
    return hidden.reshape(batch, (grid_h // merge_h) * (grid_w // merge_w), dense_out)


def timestep_embedding(values: mx.array, dimension: int, *, max_period: float = 10_000.0) -> mx.array:
    """Official GLIDE-style scalar sinusoidal embedding in FP32."""
    if values.ndim != 1:
        raise ValueError(f"timestep values must be rank 1, got {values.shape}")
    half = dimension // 2
    frequencies = mx.exp(
        -math.log(max_period) * mx.arange(half, dtype=mx.float32) / half
    )
    arguments = values.astype(mx.float32)[:, None] * frequencies[None, :]
    embedded = mx.concatenate((mx.cos(arguments), mx.sin(arguments)), axis=-1)
    if dimension % 2:
        embedded = mx.concatenate((embedded, mx.zeros_like(embedded[:, :1])), axis=-1)
    return embedded


def timestep_mlp(
    values: mx.array,
    *,
    first_weight: mx.array,
    first_bias: mx.array,
    second_weight: mx.array,
    second_bias: mx.array,
) -> mx.array:
    """Official ``TimestepEmbedder``: sinusoid -> Linear -> SiLU -> Linear."""
    embedded = timestep_embedding(values, int(first_weight.shape[1])).astype(first_weight.dtype)
    hidden = linear(embedded, first_weight, first_bias)
    hidden = hidden * mx.sigmoid(hidden)
    return linear(hidden, second_weight, second_bias)
