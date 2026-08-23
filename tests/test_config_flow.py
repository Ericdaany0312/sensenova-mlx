from __future__ import annotations

import json
import unittest
from pathlib import Path

import mlx.core as mx
import numpy as np

from sensenova_mlx.config import SenseNovaConfig
from sensenova_mlx.flow import (
    apply_cfg,
    make_timesteps,
    patchify,
    resolution_noise_scale,
    unpatchify,
)


AUDITED_CONFIG = Path(__file__).parent / "fixtures" / "formal-config.json"


class ConfigAndFlowTest(unittest.TestCase):
    def test_head_dimension_must_leave_even_t_h_w_rope_slices(self) -> None:
        raw = json.loads(AUDITED_CONFIG.read_text())
        raw["llm_config"]["hidden_size"] = 128
        raw["llm_config"]["num_attention_heads"] = 32
        raw["llm_config"]["head_dim"] = 4

        with self.assertRaisesRegex(ValueError, "divisible by 8"):
            SenseNovaConfig.from_dict(raw)

    def test_official_config_is_parsed_without_transformers(self) -> None:
        config = SenseNovaConfig.from_dict(json.loads(AUDITED_CONFIG.read_text()))

        self.assertEqual(config.hidden_size, 4096)
        self.assertEqual(config.intermediate_size, 12288)
        self.assertEqual(config.num_hidden_layers, 42)
        self.assertEqual(config.num_attention_heads, 32)
        self.assertEqual(config.num_key_value_heads, 8)
        self.assertEqual(config.head_dim, 128)
        self.assertEqual(config.effective_patch_size, 32)
        self.assertEqual(config.image_tokens(1024, 1024), 1024)
        self.assertEqual(config.image_tokens(2048, 2048), 4096)

    def test_invalid_image_geometry_fails_early(self) -> None:
        config = SenseNovaConfig.from_dict(json.loads(AUDITED_CONFIG.read_text()))

        with self.assertRaisesRegex(ValueError, "divisible by 32"):
            config.validate_image_size(1000, 1024)
        with self.assertRaisesRegex(ValueError, "4096"):
            config.validate_image_size(4096, 4096)

    def test_standard_timestep_shift_matches_official_formula(self) -> None:
        got = make_timesteps(4, shift=3.0)
        base = np.linspace(0.0, 1.0, 5, dtype=np.float32)
        sigma = 1.0 - base
        expected = 1.0 - 3.0 * sigma / (1.0 + 2.0 * sigma)

        mx.eval(got)
        np.testing.assert_allclose(np.asarray(got), expected, atol=1e-7, rtol=1e-7)

    def test_resolution_noise_scale_matches_official_1024_and_2048_values(self) -> None:
        self.assertEqual(resolution_noise_scale(1024, 1024), 4.0)
        self.assertEqual(resolution_noise_scale(2048, 2048), 8.0)

    def test_patchify_round_trip_and_cfg(self) -> None:
        image = mx.arange(1 * 3 * 8 * 12, dtype=mx.float32).reshape(1, 3, 8, 12)
        tokens = patchify(image, patch_size=4)
        restored = unpatchify(tokens, patch_size=4, height=8, width=12)
        mx.eval(restored)

        np.testing.assert_array_equal(np.asarray(restored), np.asarray(image))

        cond = mx.array([[[3.0, 5.0]]], dtype=mx.float32)
        uncond = mx.array([[[1.0, 2.0]]], dtype=mx.float32)
        guided = apply_cfg(cond, uncond, scale=2.5, norm="none")
        np.testing.assert_allclose(
            np.asarray(guided),
            np.asarray(uncond + 2.5 * (cond - uncond)),
            atol=0,
            rtol=0,
        )


if __name__ == "__main__":
    unittest.main()
