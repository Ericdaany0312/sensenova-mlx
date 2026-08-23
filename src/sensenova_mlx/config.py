from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Mapping


@dataclass(frozen=True)
class SenseNovaConfig:
    """The subset of the formal checkpoint config required by the MLX runtime."""

    hidden_size: int
    intermediate_size: int
    num_hidden_layers: int
    num_attention_heads: int
    num_key_value_heads: int
    head_dim: int
    vocab_size: int
    rms_norm_eps: float
    rope_theta: float
    rope_theta_hw: float
    max_position_embeddings: int
    max_position_embeddings_hw: int
    patch_size: int
    downsample_ratio: float
    max_image_seq_len: int
    noise_scale: float
    noise_scale_base_image_seq_len: int
    noise_scale_max_value: float
    t_eps: float
    template: str

    @classmethod
    def from_dict(cls, raw: Mapping[str, Any]) -> "SenseNovaConfig":
        llm = raw["llm_config"]
        config = cls(
            hidden_size=int(llm["hidden_size"]),
            intermediate_size=int(llm["intermediate_size"]),
            num_hidden_layers=int(llm["num_hidden_layers"]),
            num_attention_heads=int(llm["num_attention_heads"]),
            num_key_value_heads=int(llm["num_key_value_heads"]),
            head_dim=int(llm["head_dim"]),
            vocab_size=int(llm["vocab_size"]),
            rms_norm_eps=float(llm["rms_norm_eps"]),
            rope_theta=float(llm["rope_theta"]),
            rope_theta_hw=float(llm["rope_theta_hw"]),
            max_position_embeddings=int(llm["max_position_embeddings"]),
            max_position_embeddings_hw=int(llm["max_position_embeddings_hw"]),
            patch_size=int(raw["patch_size"]),
            downsample_ratio=float(raw["downsample_ratio"]),
            max_image_seq_len=int(raw["max_image_seq_len"]),
            noise_scale=float(raw["noise_scale"]),
            noise_scale_base_image_seq_len=int(raw["noise_scale_base_image_seq_len"]),
            noise_scale_max_value=float(raw["noise_scale_max_value"]),
            t_eps=float(raw["t_eps"]),
            template=str(raw["template"]),
        )
        config.validate_architecture()
        return config

    @property
    def merge_size(self) -> int:
        reciprocal = 1.0 / self.downsample_ratio
        rounded = round(reciprocal)
        if abs(reciprocal - rounded) > 1e-9:
            raise ValueError(f"downsample_ratio must have an integer reciprocal, got {self.downsample_ratio}")
        return int(rounded)

    @property
    def effective_patch_size(self) -> int:
        return self.patch_size * self.merge_size

    def image_tokens(self, width: int, height: int) -> int:
        self.validate_image_size(width, height)
        return (width // self.effective_patch_size) * (height // self.effective_patch_size)

    def validate_image_size(self, width: int, height: int) -> None:
        patch = self.effective_patch_size
        if width <= 0 or height <= 0:
            raise ValueError(f"image dimensions must be positive, got {width}x{height}")
        if width % patch or height % patch:
            raise ValueError(f"width and height must be divisible by {patch}, got {width}x{height}")
        tokens = (width // patch) * (height // patch)
        if tokens > self.max_image_seq_len:
            raise ValueError(
                f"image has {tokens} tokens, above checkpoint max_image_seq_len={self.max_image_seq_len}"
            )

    def validate_architecture(self) -> None:
        if self.head_dim % 8:
            raise ValueError(
                "head_dim must be divisible by 8 so every t/h/w RoPE slice is even, "
                f"got {self.head_dim}"
            )
        if self.num_attention_heads % self.num_key_value_heads:
            raise ValueError(
                "num_attention_heads must be divisible by num_key_value_heads: "
                f"{self.num_attention_heads} vs {self.num_key_value_heads}"
            )
        if self.hidden_size != self.num_attention_heads * self.head_dim:
            raise ValueError(
                f"hidden_size must equal num_attention_heads * head_dim, got {self.hidden_size}"
            )
        _ = self.merge_size
