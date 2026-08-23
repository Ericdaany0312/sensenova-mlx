from __future__ import annotations

import mlx.core as mx


def build_text_indexes(length: int) -> mx.array:
    if length < 0:
        raise ValueError(f"length must be non-negative, got {length}")
    t = mx.arange(length, dtype=mx.int32)
    zeros = mx.zeros_like(t)
    return mx.stack((t, zeros, zeros), axis=0)


def build_image_indexes(token_h: int, token_w: int, text_len: int) -> mx.array:
    if token_h <= 0 or token_w <= 0:
        raise ValueError(f"token grid must be positive, got {token_h}x{token_w}")
    flat = mx.arange(token_h * token_w, dtype=mx.int32)
    t = mx.full((token_h * token_w,), text_len, dtype=mx.int32)
    return mx.stack((t, flat // token_w, flat % token_w), axis=0)


def create_block_causal_mask(index: mx.array) -> mx.array:
    if index.ndim != 1:
        raise ValueError(f"index must be rank 1, got shape {index.shape}")
    length = index.shape[0]
    same_block = index[None, :] == index[:, None]
    positions = mx.arange(length, dtype=mx.int32)
    causal = positions[None, :] <= positions[:, None]
    allowed = mx.logical_or(same_block, causal)
    zeros = mx.zeros((length, length), dtype=mx.float32)
    negative_inf = mx.full((length, length), -float("inf"), dtype=mx.float32)
    return mx.where(allowed, zeros, negative_inf)[None, None, :, :]


def _rotate_half(x: mx.array) -> mx.array:
    half = x.shape[-1] // 2
    return mx.concatenate((-x[..., half:], x[..., :half]), axis=-1)


def _apply_rope(x: mx.array, positions: mx.array, theta: float) -> mx.array:
    dim = x.shape[-1]
    if dim % 2:
        raise ValueError(f"RoPE dimension must be even, got {dim}")
    inv_freq = 1.0 / (theta ** (mx.arange(0, dim, 2, dtype=mx.float32) / dim))
    angles = positions.astype(mx.float32)[:, None] * inv_freq[None, :]
    frequencies = mx.concatenate((angles, angles), axis=-1)
    cos = mx.cos(frequencies)[None, None, :, :].astype(x.dtype)
    sin = mx.sin(frequencies)[None, None, :, :].astype(x.dtype)
    return x * cos + _rotate_half(x) * sin


def apply_thw_rope(
    query: mx.array,
    key: mx.array,
    indexes: mx.array,
    *,
    theta_t: float = 5_000_000.0,
    theta_hw: float = 10_000.0,
) -> tuple[mx.array, mx.array]:
    if query.ndim != 4 or key.ndim != 4:
        raise ValueError(f"query/key must be BHSD, got {query.shape} and {key.shape}")
    if query.shape[-1] != key.shape[-1] or query.shape[-2] != key.shape[-2]:
        raise ValueError(f"query/key RoPE shape mismatch: {query.shape} vs {key.shape}")
    if indexes.shape != (3, query.shape[-2]):
        raise ValueError(f"indexes must be [3,S]={3, query.shape[-2]}, got {indexes.shape}")
    head_dim = query.shape[-1]
    if head_dim % 4:
        raise ValueError(f"head_dim must be divisible by 4, got {head_dim}")
    t_dim = head_dim // 2
    h_end = t_dim + head_dim // 4

    def apply_one(x: mx.array) -> mx.array:
        return mx.concatenate(
            (
                _apply_rope(x[..., :t_dim], indexes[0], theta_t),
                _apply_rope(x[..., t_dim:h_end], indexes[1], theta_hw),
                _apply_rope(x[..., h_end:], indexes[2], theta_hw),
            ),
            axis=-1,
        )

    return apply_one(query), apply_one(key)
