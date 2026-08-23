from __future__ import annotations

import math

import mlx.core as mx


def make_timesteps(num_steps: int, *, shift: float = 1.0) -> mx.array:
    """Official standard schedule: shift sigma, then return ``1 - sigma``."""
    if num_steps <= 0:
        raise ValueError(f"num_steps must be positive, got {num_steps}")
    if shift <= 0:
        raise ValueError(f"shift must be positive, got {shift}")
    timesteps = mx.linspace(0.0, 1.0, num_steps + 1, dtype=mx.float32)
    sigma = 1.0 - timesteps
    sigma = shift * sigma / (1.0 + (shift - 1.0) * sigma)
    return 1.0 - sigma


def resolution_noise_scale(
    width: int,
    height: int,
    *,
    patch_size: int = 16,
    merge_size: int = 2,
    base_image_seq_len: int = 64,
    base_scale: float = 1.0,
    max_value: float = 16.0,
) -> float:
    grid_h = height // patch_size
    grid_w = width // patch_size
    image_seq_len = (grid_h * grid_w) / (merge_size**2)
    return min(math.sqrt(image_seq_len / float(base_image_seq_len)) * base_scale, max_value)


def patchify(images: mx.array, patch_size: int, *, channel_first: bool = False) -> mx.array:
    if images.ndim != 4:
        raise ValueError(f"images must be NCHW, got shape {images.shape}")
    n, channels, height, width = images.shape
    if height % patch_size or width % patch_size:
        raise ValueError(f"image shape {images.shape} is not divisible by patch_size={patch_size}")
    grid_h, grid_w = height // patch_size, width // patch_size
    patches = images.reshape(n, channels, grid_h, patch_size, grid_w, patch_size)
    if channel_first:
        patches = patches.transpose(0, 2, 4, 1, 3, 5)
    else:
        patches = patches.transpose(0, 2, 4, 3, 5, 1)
    return patches.reshape(n, grid_h * grid_w, channels * patch_size * patch_size)


def unpatchify(tokens: mx.array, patch_size: int, height: int, width: int) -> mx.array:
    if tokens.ndim != 3:
        raise ValueError(f"tokens must be [N,L,C], got shape {tokens.shape}")
    n, length, feature_dim = tokens.shape
    grid_h, grid_w = height // patch_size, width // patch_size
    if length != grid_h * grid_w:
        raise ValueError(f"token length mismatch: expected {grid_h * grid_w}, got {length}")
    expected_dim = 3 * patch_size * patch_size
    if feature_dim != expected_dim:
        raise ValueError(f"token feature mismatch: expected {expected_dim}, got {feature_dim}")
    image = tokens.reshape(n, grid_h, grid_w, patch_size, patch_size, 3)
    image = image.transpose(0, 5, 1, 3, 2, 4)
    return image.reshape(n, 3, height, width)


def apply_cfg(cond: mx.array, uncond: mx.array, *, scale: float, norm: str = "none") -> mx.array:
    if cond.shape != uncond.shape:
        raise ValueError(f"CFG shape mismatch: condition {cond.shape}, uncondition {uncond.shape}")
    guided = uncond + scale * (cond - uncond)
    if norm == "none":
        return guided
    if norm == "global":
        axes = tuple(range(1, cond.ndim))
    elif norm == "channel":
        axes = (-1,)
    else:
        raise ValueError(f"unsupported CFG norm {norm!r}")
    cond_norm = mx.sqrt(mx.sum(mx.square(cond.astype(mx.float32)), axis=axes, keepdims=True))
    cfg_norm = mx.sqrt(mx.sum(mx.square(guided.astype(mx.float32)), axis=axes, keepdims=True))
    factor = mx.clip(cond_norm / (cfg_norm + 1e-8), 0.0, 1.0).astype(guided.dtype)
    return guided * factor


def euler_step(z: mx.array, velocity: mx.array, t: mx.array | float, t_next: mx.array | float) -> mx.array:
    return z + (t_next - t) * velocity
