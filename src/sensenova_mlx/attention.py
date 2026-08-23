from __future__ import annotations

from dataclasses import dataclass

import mlx.core as mx


@dataclass
class KVCache:
    """Per-layer cache in the official BHSD layout."""

    keys: mx.array | None = None
    values: mx.array | None = None

    @property
    def length(self) -> int:
        return 0 if self.keys is None else int(self.keys.shape[2])

    def combine(
        self,
        current_keys: mx.array,
        current_values: mx.array,
        *,
        update: bool,
    ) -> tuple[mx.array, mx.array]:
        if current_keys.shape != current_values.shape:
            raise ValueError(f"key/value shape mismatch: {current_keys.shape} vs {current_values.shape}")
        if self.keys is None:
            combined_keys, combined_values = current_keys, current_values
        else:
            if self.values is None:
                raise ValueError("cache has keys without values")
            combined_keys = mx.concatenate((self.keys, current_keys), axis=2)
            combined_values = mx.concatenate((self.values, current_values), axis=2)
        if update:
            self.keys, self.values = combined_keys, combined_values
        return combined_keys, combined_values


def scaled_dot_product_gqa(
    query: mx.array,
    key: mx.array,
    value: mx.array,
    *,
    mask: mx.array | None = None,
) -> mx.array:
    """MLX SDPA supports the checkpoint's 32Q/8KV grouped-query layout."""
    if query.ndim != 4 or key.ndim != 4 or value.ndim != 4:
        raise ValueError(f"Q/K/V must be BHSD, got {query.shape}, {key.shape}, {value.shape}")
    if key.shape != value.shape:
        raise ValueError(f"key/value shape mismatch: {key.shape} vs {value.shape}")
    if query.shape[0] != key.shape[0] or query.shape[-1] != key.shape[-1]:
        raise ValueError(f"query/key batch or head_dim mismatch: {query.shape} vs {key.shape}")
    if query.shape[1] % key.shape[1]:
        raise ValueError(f"query heads {query.shape[1]} are not divisible by KV heads {key.shape[1]}")
    if mask is not None and mask.dtype != query.dtype:
        # MLX fused SDPA rejects an FP32 additive mask for BF16 Q/K/V instead
        # of applying NumPy-style promotion. Preserve -inf while matching Q.
        mask = mask.astype(query.dtype)
    return mx.fast.scaled_dot_product_attention(
        query,
        key,
        value,
        scale=query.shape[-1] ** -0.5,
        mask=mask,
    )
