from __future__ import annotations

import unittest

import mlx.core as mx
import numpy as np
import torch
import torch.nn.functional as torch_f

from sensenova_mlx.embeddings import (
    generation_vision_embeddings,
    timestep_mlp,
)
from sensenova_mlx.prompt import SYSTEM_MESSAGE_FOR_GEN, build_t2i_query


def _torch_vision_reference(
    patches: np.ndarray,
    patch_weight: np.ndarray,
    patch_bias: np.ndarray,
    dense_weight: np.ndarray,
    dense_bias: np.ndarray,
) -> np.ndarray:
    patch = torch.from_numpy(patches).reshape(-1, 3, 2, 2)
    hidden = torch_f.gelu(
        torch_f.conv2d(
            patch,
            torch.from_numpy(patch_weight),
            torch.from_numpy(patch_bias),
            stride=2,
        )
    ).reshape(-1, 8)

    rows = torch.arange(4).repeat_interleave(4)
    cols = torch.arange(4).repeat(4)
    parts = []
    for value, positions in ((hidden[:, :4], cols), (hidden[:, 4:], rows)):
        frequencies = 1.0 / (10_000.0 ** (torch.arange(0, 4, 2).float() / 4))
        angles = positions.float()[:, None] * frequencies[None]
        even, odd = value[:, 0::2], value[:, 1::2]
        rotated = torch.empty_like(value)
        rotated[:, 0::2] = even * angles.cos() - odd * angles.sin()
        rotated[:, 1::2] = even * angles.sin() + odd * angles.cos()
        parts.append(rotated)
    hidden = torch.cat(parts, dim=-1).reshape(1, 4, 4, 8).permute(0, 3, 1, 2)
    output = torch_f.conv2d(
        hidden,
        torch.from_numpy(dense_weight),
        torch.from_numpy(dense_bias),
        stride=2,
    )
    return output.permute(0, 2, 3, 1).reshape(1, 4, 12).numpy()


class EmbeddingsAndPromptTest(unittest.TestCase):
    def test_generation_vision_embeddings_match_pytorch(self) -> None:
        rng = np.random.default_rng(31)
        patches = rng.normal(size=(1, 16, 12)).astype(np.float32)
        patch_weight = rng.normal(size=(8, 3, 2, 2)).astype(np.float32) * 0.1
        patch_bias = rng.normal(size=(8,)).astype(np.float32) * 0.1
        dense_weight = rng.normal(size=(12, 8, 2, 2)).astype(np.float32) * 0.1
        dense_bias = rng.normal(size=(12,)).astype(np.float32) * 0.1

        got = generation_vision_embeddings(
            mx.array(patches),
            grid_h=4,
            grid_w=4,
            patch_weight_oihw=mx.array(patch_weight),
            patch_bias=mx.array(patch_bias),
            dense_weight_oihw=mx.array(dense_weight),
            dense_bias=mx.array(dense_bias),
            rope_theta=10_000.0,
        )
        expected = _torch_vision_reference(
            patches,
            patch_weight,
            patch_bias,
            dense_weight,
            dense_bias,
        )
        mx.eval(got)
        np.testing.assert_allclose(np.asarray(got), expected, atol=2e-5, rtol=2e-5)

    def test_timestep_mlp_matches_pytorch(self) -> None:
        rng = np.random.default_rng(37)
        values = np.array([0.0, 0.25, 1.0], dtype=np.float32)
        first_weight = rng.normal(size=(10, 6)).astype(np.float32) * 0.1
        first_bias = rng.normal(size=(10,)).astype(np.float32) * 0.1
        second_weight = rng.normal(size=(10, 10)).astype(np.float32) * 0.1
        second_bias = rng.normal(size=(10,)).astype(np.float32) * 0.1

        got = timestep_mlp(
            mx.array(values),
            first_weight=mx.array(first_weight),
            first_bias=mx.array(first_bias),
            second_weight=mx.array(second_weight),
            second_bias=mx.array(second_bias),
        )
        half = 3
        frequencies = torch.exp(-np.log(10_000.0) * torch.arange(half).float() / half)
        arguments = torch.from_numpy(values)[:, None] * frequencies[None]
        embedded = torch.cat((arguments.cos(), arguments.sin()), dim=-1)
        expected = torch_f.linear(embedded, torch.from_numpy(first_weight), torch.from_numpy(first_bias))
        expected = torch_f.silu(expected)
        expected = torch_f.linear(expected, torch.from_numpy(second_weight), torch.from_numpy(second_bias))

        mx.eval(got)
        np.testing.assert_allclose(np.asarray(got), expected.numpy(), atol=2e-6, rtol=2e-6)

    def test_formal_non_think_prompt_is_byte_exact(self) -> None:
        prompt = "一只戴着红围巾的猫"
        expected = (
            f"<|im_start|>system\n{SYSTEM_MESSAGE_FOR_GEN}<|im_end|>\n"
            f"<|im_start|>user\n{prompt}<|im_end|>\n"
            "<|im_start|>assistant\n<think>\n\n</think>\n\n<img>"
        )
        self.assertEqual(build_t2i_query(prompt, conditional=True), expected)
        self.assertEqual(
            build_t2i_query("", conditional=False),
            "<|im_start|>user\n<|im_end|>\n<|im_start|>assistant\n<img>",
        )


if __name__ == "__main__":
    unittest.main()
