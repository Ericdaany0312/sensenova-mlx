from __future__ import annotations

import unittest

import mlx.core as mx
import numpy as np
import torch
import torch.nn.functional as torch_f

from sensenova_mlx.config import SenseNovaConfig
from sensenova_mlx.pipeline import SenseNovaT2IPipeline, decode_velocity


class PipelineTest(unittest.TestCase):
    def test_one_step_orchestration_is_reproducible_on_mlx(self) -> None:
        rng = np.random.default_rng(43)
        config = SenseNovaConfig(
            hidden_size=32,
            intermediate_size=64,
            num_hidden_layers=1,
            num_attention_heads=4,
            num_key_value_heads=2,
            head_dim=8,
            vocab_size=64,
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

        def normal(shape: tuple[int, ...], scale: float = 0.02) -> mx.array:
            return mx.array((rng.normal(size=shape) * scale).astype(np.float32))

        weights = {
            "language_model.model.norm.weight": mx.ones((32,), dtype=mx.float32),
            "fm_modules.vision_model_mot_gen.embeddings.patch_embedding.weight": normal((8, 3, 16, 16)),
            "fm_modules.vision_model_mot_gen.embeddings.patch_embedding.bias": normal((8,)),
            "fm_modules.vision_model_mot_gen.embeddings.dense_embedding.weight": normal((32, 8, 2, 2)),
            "fm_modules.vision_model_mot_gen.embeddings.dense_embedding.bias": normal((32,)),
            "fm_modules.fm_head.conv1.weight": normal((16, 8, 3, 3)),
            "fm_modules.fm_head.conv1.bias": normal((16,)),
            "fm_modules.fm_head.conv2.weight": normal((192, 4, 3, 3)),
            "fm_modules.fm_head.conv2.bias": normal((192,)),
        }
        for prefix in ("timestep_embedder", "noise_scale_embedder"):
            weights[f"fm_modules.{prefix}.mlp.0.weight"] = normal((32, 6))
            weights[f"fm_modules.{prefix}.mlp.0.bias"] = normal((32,))
            weights[f"fm_modules.{prefix}.mlp.2.weight"] = normal((32, 32))
            weights[f"fm_modules.{prefix}.mlp.2.bias"] = normal((32,))

        class IdentityBackbone:
            def forward_prefix_ids(self, input_ids, indexes, mask):
                return mx.zeros((1, input_ids.shape[1], 32)), []

            def forward_generation(self, embeddings, indexes, caches):
                return embeddings

        pipeline = SenseNovaT2IPipeline.__new__(SenseNovaT2IPipeline)
        pipeline.config = config
        pipeline.weights = weights
        pipeline.backbone = IdentityBackbone()
        ids = mx.array([[1, 2, 3]], dtype=mx.int32)

        first = pipeline.generate_from_ids(
            ids,
            None,
            width=64,
            height=64,
            num_steps=1,
            seed=123,
            cfg_scale=1.0,
        )
        second = pipeline.generate_from_ids(
            ids,
            None,
            width=64,
            height=64,
            num_steps=1,
            seed=123,
            cfg_scale=1.0,
        )
        mx.eval(first, second)
        self.assertEqual(first.shape, (1, 3, 64, 64))
        self.assertTrue(np.isfinite(np.asarray(first)).all())
        np.testing.assert_array_equal(np.asarray(first), np.asarray(second))

    def test_pixel_head_velocity_matches_independent_pytorch_layout(self) -> None:
        rng = np.random.default_rng(41)
        hidden = rng.normal(size=(1, 2, 2, 32)).astype(np.float32) * 0.05
        z = rng.normal(size=(1, 4, 3 * 32 * 32)).astype(np.float32) * 0.05
        conv1_weight = rng.normal(size=(16, 8, 3, 3)).astype(np.float32) * 0.02
        conv1_bias = rng.normal(size=(16,)).astype(np.float32) * 0.02
        conv2_weight = rng.normal(size=(192, 4, 3, 3)).astype(np.float32) * 0.02
        conv2_bias = rng.normal(size=(192,)).astype(np.float32) * 0.02

        got = decode_velocity(
            mx.array(hidden),
            mx.array(z),
            t=0.25,
            t_eps=0.05,
            patch_size=32,
            conv1_weight_oihw=mx.array(conv1_weight),
            conv1_bias=mx.array(conv1_bias),
            conv2_weight_oihw=mx.array(conv2_weight),
            conv2_bias=mx.array(conv2_bias),
        )

        value = torch.from_numpy(hidden).permute(0, 3, 1, 2)
        value = torch_f.pixel_shuffle(value, 2)
        value = torch_f.gelu(
            torch_f.conv2d(
                value,
                torch.from_numpy(conv1_weight),
                torch.from_numpy(conv1_bias),
                padding=1,
            )
        )
        value = torch_f.pixel_shuffle(value, 2)
        value = torch_f.conv2d(
            value,
            torch.from_numpy(conv2_weight),
            torch.from_numpy(conv2_bias),
            padding=1,
        )
        value = torch_f.pixel_shuffle(value, 8)
        value = value.reshape(1, 3, 2, 32, 2, 32)
        value = torch.einsum("nchpwq->nhwpqc", value).reshape(1, 4, 3 * 32 * 32)
        expected = (value - torch.from_numpy(z)) / 0.75

        mx.eval(got)
        np.testing.assert_allclose(np.asarray(got), expected.numpy(), atol=3e-5, rtol=3e-5)


if __name__ == "__main__":
    unittest.main()
