from __future__ import annotations

import math
import unittest

import mlx.core as mx
import numpy as np

from sensenova_mlx.attention import KVCache, scaled_dot_product_gqa
from sensenova_mlx.rope import (
    apply_thw_rope,
    build_image_indexes,
    build_text_indexes,
    create_block_causal_mask,
)


def _rotate_half(x: np.ndarray) -> np.ndarray:
    half = x.shape[-1] // 2
    return np.concatenate((-x[..., half:], x[..., :half]), axis=-1)


def _rope_reference(x: np.ndarray, positions: np.ndarray, theta: float) -> np.ndarray:
    dim = x.shape[-1]
    inv_freq = 1.0 / (theta ** (np.arange(0, dim, 2, dtype=np.float32) / dim))
    angles = positions.astype(np.float32)[:, None] * inv_freq[None, :]
    cos = np.concatenate((np.cos(angles), np.cos(angles)), axis=-1)[None, None]
    sin = np.concatenate((np.sin(angles), np.sin(angles)), axis=-1)[None, None]
    return x * cos + _rotate_half(x) * sin


class RopeAndAttentionTest(unittest.TestCase):
    def test_bfloat16_attention_accepts_float32_causal_mask(self) -> None:
        query = mx.ones((1, 4, 2, 8), dtype=mx.bfloat16)
        key = mx.ones((1, 2, 2, 8), dtype=mx.bfloat16)
        value = mx.ones((1, 2, 2, 8), dtype=mx.bfloat16)
        mask = create_block_causal_mask(mx.array([0, 1], dtype=mx.int32))

        output = scaled_dot_product_gqa(query, key, value, mask=mask)
        mx.eval(output)

        self.assertEqual(output.dtype, mx.bfloat16)
        self.assertEqual(output.shape, query.shape)

    def test_text_image_indexes_and_block_causal_mask(self) -> None:
        text = build_text_indexes(3)
        image = build_image_indexes(token_h=2, token_w=3, text_len=3)

        np.testing.assert_array_equal(np.asarray(text), [[0, 1, 2], [0, 0, 0], [0, 0, 0]])
        np.testing.assert_array_equal(
            np.asarray(image),
            [[3, 3, 3, 3, 3, 3], [0, 0, 0, 1, 1, 1], [0, 1, 2, 0, 1, 2]],
        )

        mask = create_block_causal_mask(mx.array([0, 1, 2, 2, 2], dtype=mx.int32))
        got = np.asarray(mask)
        self.assertEqual(got.shape, (1, 1, 5, 5))
        self.assertTrue(np.isneginf(got[0, 0, 0, 1]))
        self.assertEqual(got[0, 0, 3, 4], 0.0)
        self.assertEqual(got[0, 0, 4, 3], 0.0)

    def test_thw_rope_matches_independent_numpy_reference(self) -> None:
        rng = np.random.default_rng(7)
        q_np = rng.normal(size=(1, 4, 6, 8)).astype(np.float32)
        k_np = rng.normal(size=(1, 2, 6, 8)).astype(np.float32)
        indexes = np.array(
            [[5, 5, 5, 5, 5, 5], [0, 0, 0, 1, 1, 1], [0, 1, 2, 0, 1, 2]],
            dtype=np.int32,
        )

        q, k = apply_thw_rope(mx.array(q_np), mx.array(k_np), mx.array(indexes))
        q_expected = np.concatenate(
            (
                _rope_reference(q_np[..., :4], indexes[0], 5_000_000.0),
                _rope_reference(q_np[..., 4:6], indexes[1], 10_000.0),
                _rope_reference(q_np[..., 6:], indexes[2], 10_000.0),
            ),
            axis=-1,
        )
        k_expected = np.concatenate(
            (
                _rope_reference(k_np[..., :4], indexes[0], 5_000_000.0),
                _rope_reference(k_np[..., 4:6], indexes[1], 10_000.0),
                _rope_reference(k_np[..., 6:], indexes[2], 10_000.0),
            ),
            axis=-1,
        )

        mx.eval(q, k)
        np.testing.assert_allclose(np.asarray(q), q_expected, atol=2e-6, rtol=2e-6)
        np.testing.assert_allclose(np.asarray(k), k_expected, atol=2e-6, rtol=2e-6)

    def test_gqa_and_read_only_prefix_cache_match_explicit_reference(self) -> None:
        rng = np.random.default_rng(9)
        q = rng.normal(size=(1, 4, 3, 8)).astype(np.float32)
        prefix_k = rng.normal(size=(1, 2, 2, 8)).astype(np.float32)
        prefix_v = rng.normal(size=(1, 2, 2, 8)).astype(np.float32)
        current_k = rng.normal(size=(1, 2, 3, 8)).astype(np.float32)
        current_v = rng.normal(size=(1, 2, 3, 8)).astype(np.float32)

        cache = KVCache(mx.array(prefix_k), mx.array(prefix_v))
        k, v = cache.combine(mx.array(current_k), mx.array(current_v), update=False)
        got = scaled_dot_product_gqa(mx.array(q), k, v)

        full_k = np.concatenate((prefix_k, current_k), axis=2)
        full_v = np.concatenate((prefix_v, current_v), axis=2)
        full_k = np.repeat(full_k, 2, axis=1)
        full_v = np.repeat(full_v, 2, axis=1)
        tiled = mx.fast.scaled_dot_product_attention(
            mx.array(q),
            mx.array(full_k),
            mx.array(full_v),
            scale=8**-0.5,
        )
        mx.eval(got, tiled)
        scores = np.matmul(q, np.swapaxes(full_k, -1, -2)) / math.sqrt(8)
        scores = scores - scores.max(axis=-1, keepdims=True)
        probs = np.exp(scores) / np.exp(scores).sum(axis=-1, keepdims=True)
        expected = np.matmul(probs, full_v)

        # The Metal fused FP32 matmul path differs from NumPy CPU by about 1e-3,
        # while MLX's supported GQA and explicitly tiled MHA paths must agree.
        np.testing.assert_allclose(np.asarray(got), np.asarray(tiled), atol=2e-6, rtol=2e-6)
        np.testing.assert_allclose(np.asarray(got), expected, atol=2e-3, rtol=2e-3)
        self.assertEqual(cache.length, 2)


if __name__ == "__main__":
    unittest.main()
