from __future__ import annotations

import hashlib
import json
from dataclasses import dataclass
from pathlib import Path
from typing import Iterable, Mapping

import mlx.core as mx
from safetensors import safe_open


@dataclass(frozen=True)
class FileRecord:
    name: str
    size: int
    sha256: str


@dataclass(frozen=True)
class ShardManifest:
    revision: str
    files: tuple[FileRecord, ...]

    @property
    def by_name(self) -> dict[str, FileRecord]:
        return {record.name: record for record in self.files}


@dataclass(frozen=True)
class LoRAWeights:
    down: mx.array
    up: mx.array
    alpha: float

    @property
    def rank(self) -> int:
        return int(self.down.shape[0])


def verify_file(path: Path, record: FileRecord, *, chunk_size: int = 8 * 1024 * 1024) -> None:
    if not path.is_file():
        raise FileNotFoundError(path)
    actual_size = path.stat().st_size
    if actual_size != record.size:
        raise ValueError(f"{path.name}: size mismatch, expected {record.size}, got {actual_size}")
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(chunk_size), b""):
            digest.update(block)
    actual_hash = digest.hexdigest()
    if actual_hash != record.sha256:
        raise ValueError(f"{path.name}: SHA-256 mismatch, expected {record.sha256}, got {actual_hash}")


def load_sharded_weights(
    model_dir: Path,
    *,
    index_path: Path,
    manifest: ShardManifest,
    include: set[str] | None = None,
    target_dtype: mx.Dtype = mx.bfloat16,
) -> dict[str, mx.array]:
    index = json.loads(index_path.read_text(encoding="utf-8"))
    weight_map: dict[str, str] = index["weight_map"]
    selected_keys = set(weight_map) if include is None else set(include)
    unknown = selected_keys - set(weight_map)
    if unknown:
        raise KeyError(f"requested weights absent from index: {sorted(unknown)}")

    keys_by_shard: dict[str, set[str]] = {}
    for key in selected_keys:
        keys_by_shard.setdefault(weight_map[key], set()).add(key)

    records = manifest.by_name
    output: dict[str, mx.array] = {}
    for shard_name in sorted(keys_by_shard):
        if shard_name not in records:
            raise ValueError(f"index references shard missing from manifest: {shard_name}")
        shard_path = model_dir / shard_name
        verify_file(shard_path, records[shard_name])
        shard = mx.load(str(shard_path))
        if not isinstance(shard, dict):
            raise TypeError(f"expected safetensors mapping from {shard_name}, got {type(shard).__name__}")
        for key in keys_by_shard[shard_name]:
            if key not in shard:
                raise KeyError(f"{shard_name}: index key missing from payload: {key}")
            value = shard[key]
            output[key] = value if value.dtype == target_dtype else value.astype(target_dtype)
        if keys_by_shard[shard_name]:
            mx.eval(*[output[key] for key in keys_by_shard[shard_name]])
        del shard
    return output


def load_lora_adapters(
    path: Path,
    *,
    target_dtype: mx.Dtype = mx.bfloat16,
) -> dict[str, LoRAWeights]:
    """Load formal ``neo_hf_lora`` alpha/down/up triplets without merging."""
    with safe_open(path, framework="numpy") as handle:
        metadata = handle.metadata() or {}
    tensor_kind = metadata.get("tensor_kind")
    if tensor_kind != "neo_hf_lora":
        raise ValueError(f"unsupported LoRA tensor_kind {tensor_kind!r}")

    tensors = mx.load(str(path))
    if not isinstance(tensors, dict):
        raise TypeError(f"expected LoRA safetensors mapping, got {type(tensors).__name__}")
    prefixes = sorted(key[: -len(".alpha")] for key in tensors if key.endswith(".alpha"))
    expected_keys: set[str] = set()
    adapters: dict[str, LoRAWeights] = {}
    for prefix in prefixes:
        alpha_key = f"{prefix}.alpha"
        down_key = f"{prefix}.lora_down.weight"
        up_key = f"{prefix}.lora_up.weight"
        expected_keys.update((alpha_key, down_key, up_key))
        missing = [key for key in (alpha_key, down_key, up_key) if key not in tensors]
        if missing:
            raise KeyError(f"incomplete LoRA group {prefix}: missing {missing}")
        down = tensors[down_key]
        up = tensors[up_key]
        down = down if down.dtype == target_dtype else down.astype(target_dtype)
        up = up if up.dtype == target_dtype else up.astype(target_dtype)
        if down.ndim != 2 or up.ndim != 2 or up.shape[1] != down.shape[0]:
            raise ValueError(f"invalid LoRA shapes for {prefix}: down={down.shape}, up={up.shape}")
        adapters[f"{prefix}.weight"] = LoRAWeights(
            down=down,
            up=up,
            alpha=float(tensors[alpha_key].item()),
        )
    unexpected = set(tensors) - expected_keys
    if unexpected:
        raise ValueError(f"unexpected tensors in LoRA checkpoint: {sorted(unexpected)[:8]}")
    if not adapters:
        raise ValueError("LoRA checkpoint contains no adapter groups")
    mx.eval(*[array for adapter in adapters.values() for array in (adapter.down, adapter.up)])
    return adapters


def linear(x: mx.array, weight: mx.array, bias: mx.array | None = None) -> mx.array:
    result = x @ weight.T
    return result if bias is None else result + bias


def lora_linear(
    x: mx.array,
    weight: mx.array,
    adapter: LoRAWeights | None = None,
    bias: mx.array | None = None,
) -> mx.array:
    result = linear(x, weight, bias)
    if adapter is None:
        return result
    if adapter.down.shape[1] != weight.shape[1] or adapter.up.shape != (weight.shape[0], adapter.rank):
        raise ValueError(
            f"LoRA shape mismatch for base {weight.shape}: down={adapter.down.shape}, up={adapter.up.shape}"
        )
    scale = adapter.alpha / adapter.rank
    return result + linear(linear(x, adapter.down), adapter.up) * scale


def require_keys(weights: Mapping[str, mx.array], names: Iterable[str]) -> None:
    missing = [name for name in names if name not in weights]
    if missing:
        raise KeyError(f"missing required checkpoint weights: {missing}")
