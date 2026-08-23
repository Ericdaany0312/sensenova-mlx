from __future__ import annotations

import hashlib
import json
import tempfile
import unittest
from pathlib import Path

import mlx.core as mx
import numpy as np
from safetensors.numpy import save_file

from sensenova_mlx.config import SenseNovaConfig
from sensenova_mlx.pipeline import required_t2i_weight_keys
from sensenova_mlx.weights import (
    FileRecord,
    LoRAWeights,
    ShardManifest,
    load_lora_adapters,
    load_sharded_weights,
    lora_linear,
    verify_file,
)


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


class WeightLoadingTest(unittest.TestCase):
    def test_formal_runtime_key_set_omits_only_lm_head_and_understanding_vision(self) -> None:
        root = Path(__file__).parents[1]
        raw_config = json.loads((root / "tests" / "fixtures" / "formal-config.json").read_text())
        audited = set(json.loads((root / "docs" / "weights-audit.json").read_text())["tensors"])
        required = required_t2i_weight_keys(SenseNovaConfig.from_dict(raw_config))
        omitted = audited - required

        self.assertEqual(len(required), 1111)
        self.assertEqual(
            omitted,
            {
                "language_model.lm_head.weight",
                "vision_model.embeddings.dense_embedding.bias",
                "vision_model.embeddings.dense_embedding.weight",
                "vision_model.embeddings.patch_embedding.bias",
                "vision_model.embeddings.patch_embedding.weight",
            },
        )

    def test_formal_lora_triplets_map_to_base_weight_keys(self) -> None:
        with tempfile.TemporaryDirectory(prefix="sensenova-lora-") as raw:
            path = Path(raw) / "adapter.safetensors"
            prefix = "language_model.model.layers.0.self_attn.q_proj_mot_gen"
            save_file(
                {
                    f"{prefix}.alpha": np.array(6.0, dtype=np.float32),
                    f"{prefix}.lora_down.weight": np.arange(8, dtype=np.float32).reshape(2, 4),
                    f"{prefix}.lora_up.weight": np.arange(10, dtype=np.float32).reshape(5, 2),
                },
                path,
                metadata={"tensor_kind": "neo_hf_lora", "lora_alpha": "8.0"},
            )

            adapters = load_lora_adapters(path, target_dtype=mx.float32)

            self.assertEqual(set(adapters), {f"{prefix}.weight"})
            self.assertEqual(adapters[f"{prefix}.weight"].alpha, 6.0)
            self.assertEqual(adapters[f"{prefix}.weight"].rank, 2)

    def test_hash_size_and_sharded_load_are_strict(self) -> None:
        with tempfile.TemporaryDirectory(prefix="sensenova-weights-") as raw:
            root = Path(raw)
            first = root / "a.safetensors"
            second = root / "b.safetensors"
            save_file({"a": np.arange(12, dtype=np.float32).reshape(3, 4)}, first)
            save_file({"b": np.arange(8, dtype=np.float32).reshape(2, 4)}, second)
            index = root / "model.safetensors.index.json"
            index.write_text(json.dumps({"weight_map": {"a": first.name, "b": second.name}}))
            manifest = ShardManifest(
                revision="fixture",
                files=(
                    FileRecord(first.name, first.stat().st_size, _sha256(first)),
                    FileRecord(second.name, second.stat().st_size, _sha256(second)),
                ),
            )

            verify_file(first, manifest.files[0])
            weights = load_sharded_weights(
                root,
                index_path=index,
                manifest=manifest,
                include={"a"},
                target_dtype=mx.bfloat16,
            )
            self.assertEqual(set(weights), {"a"})
            self.assertEqual(weights["a"].dtype, mx.bfloat16)

            first.write_bytes(first.read_bytes() + b"corrupt")
            with self.assertRaisesRegex(ValueError, "size mismatch"):
                verify_file(first, manifest.files[0])

    def test_lora_linear_matches_explicit_delta(self) -> None:
        rng = np.random.default_rng(13)
        x = mx.array(rng.normal(size=(2, 3, 4)).astype(np.float32))
        base = mx.array(rng.normal(size=(5, 4)).astype(np.float32))
        down = mx.array(rng.normal(size=(2, 4)).astype(np.float32))
        up = mx.array(rng.normal(size=(5, 2)).astype(np.float32))
        adapter = LoRAWeights(down=down, up=up, alpha=8.0)

        got = lora_linear(x, base, adapter)
        expected = x @ base.T + ((x @ down.T) @ up.T) * (8.0 / 2.0)
        mx.eval(got, expected)
        np.testing.assert_allclose(np.asarray(got), np.asarray(expected), atol=1e-6, rtol=1e-6)


if __name__ == "__main__":
    unittest.main()
