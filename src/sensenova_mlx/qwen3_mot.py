from __future__ import annotations

from typing import Mapping, Sequence

import mlx.core as mx

from .attention import KVCache, scaled_dot_product_gqa
from .config import SenseNovaConfig
from .rope import apply_thw_rope
from .weights import LoRAWeights, lora_linear, require_keys


def rms_norm(x: mx.array, weight: mx.array, eps: float) -> mx.array:
    """Qwen3 RMSNorm keeps its variance calculation in FP32."""
    input_dtype = x.dtype
    x_fp32 = x.astype(mx.float32)
    normalized = x_fp32 * mx.rsqrt(mx.mean(mx.square(x_fp32), axis=-1, keepdims=True) + eps)
    return weight * normalized.astype(input_dtype)


def silu(x: mx.array) -> mx.array:
    return x * mx.sigmoid(x)


class Qwen3MoTModel:
    """Weight-allocation-free Qwen3 dual-expert backbone.

    The official runtime calls the understanding expert for the text prefix and
    the generation expert for every denoising block. Mixed token-type forwards
    are deliberately out of scope because upstream itself does not validate
    that path.
    """

    def __init__(
        self,
        config: SenseNovaConfig,
        weights: Mapping[str, mx.array],
        adapters: Mapping[str, LoRAWeights] | None = None,
    ) -> None:
        self.config = config
        self.weights = weights
        self.adapters = {} if adapters is None else adapters
        require_keys(
            weights,
            (
                "language_model.model.norm.weight",
                "language_model.model.norm_mot_gen.weight",
            ),
        )

    def _weight(self, key: str) -> mx.array:
        try:
            return self.weights[key]
        except KeyError as exc:
            raise KeyError(f"missing required checkpoint weight: {key}") from exc

    def _linear(self, x: mx.array, key: str) -> mx.array:
        return lora_linear(x, self._weight(key), self.adapters.get(key))

    def embed(self, input_ids: mx.array) -> mx.array:
        if input_ids.ndim != 2:
            raise ValueError(f"input_ids must be [B,S], got {input_ids.shape}")
        return self._weight("language_model.model.embed_tokens.weight")[input_ids]

    def _attention(
        self,
        hidden: mx.array,
        indexes: mx.array,
        cache: KVCache,
        *,
        layer_idx: int,
        expert: str,
        mask: mx.array | None,
        update_cache: bool,
    ) -> mx.array:
        config = self.config
        root = f"language_model.model.layers.{layer_idx}.self_attn"
        suffix = "" if expert == "und" else "_mot_gen"
        batch, sequence, _ = hidden.shape
        q = self._linear(hidden, f"{root}.q_proj{suffix}.weight").reshape(
            batch, sequence, config.num_attention_heads, config.head_dim
        )
        k = self._linear(hidden, f"{root}.k_proj{suffix}.weight").reshape(
            batch, sequence, config.num_key_value_heads, config.head_dim
        )
        v = self._linear(hidden, f"{root}.v_proj{suffix}.weight").reshape(
            batch, sequence, config.num_key_value_heads, config.head_dim
        )

        half = config.head_dim // 2
        q_t = rms_norm(q[..., :half], self._weight(f"{root}.q_norm{suffix}.weight"), config.rms_norm_eps)
        q_hw = rms_norm(q[..., half:], self._weight(f"{root}.q_norm_hw{suffix}.weight"), config.rms_norm_eps)
        k_t = rms_norm(k[..., :half], self._weight(f"{root}.k_norm{suffix}.weight"), config.rms_norm_eps)
        k_hw = rms_norm(k[..., half:], self._weight(f"{root}.k_norm_hw{suffix}.weight"), config.rms_norm_eps)
        q = mx.concatenate((q_t, q_hw), axis=-1).transpose(0, 2, 1, 3)
        k = mx.concatenate((k_t, k_hw), axis=-1).transpose(0, 2, 1, 3)
        v = v.transpose(0, 2, 1, 3)
        q, k = apply_thw_rope(
            q,
            k,
            indexes,
            theta_t=config.rope_theta,
            theta_hw=config.rope_theta_hw,
        )
        attended_k, attended_v = cache.combine(k, v, update=update_cache)
        output = scaled_dot_product_gqa(q, attended_k, attended_v, mask=mask)
        output = output.transpose(0, 2, 1, 3).reshape(batch, sequence, config.hidden_size)
        return self._linear(output, f"{root}.o_proj{suffix}.weight")

    def _layer(
        self,
        hidden: mx.array,
        indexes: mx.array,
        cache: KVCache,
        *,
        layer_idx: int,
        expert: str,
        mask: mx.array | None,
        update_cache: bool,
    ) -> mx.array:
        root = f"language_model.model.layers.{layer_idx}"
        suffix = "" if expert == "und" else "_mot_gen"
        mlp_name = "mlp" if expert == "und" else "mlp_mot_gen"
        residual = hidden
        hidden = rms_norm(
            hidden,
            self._weight(f"{root}.input_layernorm{suffix}.weight"),
            self.config.rms_norm_eps,
        )
        hidden = residual + self._attention(
            hidden,
            indexes,
            cache,
            layer_idx=layer_idx,
            expert=expert,
            mask=mask,
            update_cache=update_cache,
        )

        residual = hidden
        hidden = rms_norm(
            hidden,
            self._weight(f"{root}.post_attention_layernorm{suffix}.weight"),
            self.config.rms_norm_eps,
        )
        gate = silu(self._linear(hidden, f"{root}.{mlp_name}.gate_proj.weight"))
        up = self._linear(hidden, f"{root}.{mlp_name}.up_proj.weight")
        hidden = residual + self._linear(gate * up, f"{root}.{mlp_name}.down_proj.weight")
        return hidden

    def forward_prefix_embeddings(
        self,
        embeddings: mx.array,
        indexes: mx.array,
        mask: mx.array,
    ) -> tuple[mx.array, list[KVCache]]:
        caches = [KVCache() for _ in range(self.config.num_hidden_layers)]
        hidden = embeddings
        for layer_idx, cache in enumerate(caches):
            hidden = self._layer(
                hidden,
                indexes,
                cache,
                layer_idx=layer_idx,
                expert="und",
                mask=mask,
                update_cache=True,
            )
            # Materialize each layer so a 42-layer lazy graph and every prefix
            # cache producer do not remain live at the same time.
            if cache.keys is None or cache.values is None:
                raise RuntimeError(f"layer {layer_idx} did not populate its prefix cache")
            mx.eval(hidden, cache.keys, cache.values)
        hidden = rms_norm(
            hidden,
            self._weight("language_model.model.norm.weight"),
            self.config.rms_norm_eps,
        )
        return hidden, caches

    def forward_prefix_ids(
        self,
        input_ids: mx.array,
        indexes: mx.array,
        mask: mx.array,
    ) -> tuple[mx.array, list[KVCache]]:
        return self.forward_prefix_embeddings(self.embed(input_ids), indexes, mask)

    def forward_generation(
        self,
        embeddings: mx.array,
        indexes: mx.array,
        caches: Sequence[KVCache],
    ) -> mx.array:
        if len(caches) != self.config.num_hidden_layers:
            raise ValueError(
                f"expected {self.config.num_hidden_layers} layer caches, got {len(caches)}"
            )
        hidden = embeddings
        for layer_idx, cache in enumerate(caches):
            hidden = self._layer(
                hidden,
                indexes,
                cache,
                layer_idx=layer_idx,
                expert="gen",
                mask=None,
                update_cache=False,
            )
            mx.eval(hidden)
        return rms_norm(
            hidden,
            self._weight("language_model.model.norm_mot_gen.weight"),
            self.config.rms_norm_eps,
        )
