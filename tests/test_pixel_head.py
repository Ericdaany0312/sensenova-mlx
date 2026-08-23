from __future__ import annotations

import unittest

import mlx.core as mx
import numpy as np
import torch
import torch.nn.functional as torch_f

from sensenova_mlx.pixel_head import conv_decoder, pixel_shuffle_nhwc


class PixelHeadTest(unittest.TestCase):
    def test_pixel_shuffle_layout(self) -> None:
        x = mx.arange(1 * 2 * 3 * 12, dtype=mx.float32).reshape(1, 2, 3, 12)
        got = pixel_shuffle_nhwc(x, upscale_factor=2)

        ref = torch_f.pixel_shuffle(
            torch.from_numpy(np.asarray(x)).permute(0, 3, 1, 2), 2
        ).permute(0, 2, 3, 1)
        np.testing.assert_array_equal(np.asarray(got), ref.numpy())

    def test_conv_decoder_matches_pytorch(self) -> None:
        rng = np.random.default_rng(11)
        hidden = (rng.normal(size=(1, 2, 2, 16)) * 0.1).astype(np.float32)
        conv1_w = (rng.normal(size=(8, 4, 3, 3)) * 0.1).astype(np.float32)
        conv1_b = (rng.normal(size=(8,)) * 0.1).astype(np.float32)
        conv2_w = (rng.normal(size=(192, 2, 3, 3)) * 0.1).astype(np.float32)
        conv2_b = (rng.normal(size=(192,)) * 0.1).astype(np.float32)

        got = conv_decoder(
            mx.array(hidden),
            mx.array(conv1_w),
            mx.array(conv1_b),
            mx.array(conv2_w),
            mx.array(conv2_b),
        )

        ref = torch.from_numpy(hidden).permute(0, 3, 1, 2)
        ref = torch_f.pixel_shuffle(ref, 2)
        ref = torch_f.conv2d(ref, torch.from_numpy(conv1_w), torch.from_numpy(conv1_b), padding=1)
        ref = torch_f.gelu(ref, approximate="none")
        ref = torch_f.pixel_shuffle(ref, 2)
        ref = torch_f.conv2d(ref, torch.from_numpy(conv2_w), torch.from_numpy(conv2_b), padding=1)
        ref = torch_f.pixel_shuffle(ref, 8).permute(0, 2, 3, 1)

        mx.eval(got)
        self.assertEqual(tuple(got.shape), (1, 64, 64, 3))
        np.testing.assert_allclose(np.asarray(got), ref.numpy(), atol=1e-4, rtol=1e-4)


if __name__ == "__main__":
    unittest.main()
