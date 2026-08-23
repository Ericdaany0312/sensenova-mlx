from __future__ import annotations

from collections.abc import Callable, Mapping, Sequence

import mlx.core as mx

from .attention import KVCache
from .config import SenseNovaConfig
from .embeddings import generation_vision_embeddings, timestep_mlp
from .flow import (
    apply_cfg,
    euler_step,
    make_timesteps,
    patchify,
    resolution_noise_scale,
    unpatchify,
)
from .pixel_head import conv_decoder
from .qwen3_mot import Qwen3MoTModel
from .rope import build_image_indexes, build_text_indexes, create_block_causal_mask
from .weights import require_keys


COMPONENT_KEYS = (
    "fm_modules.vision_model_mot_gen.embeddings.patch_embedding.weight",
    "fm_modules.vision_model_mot_gen.embeddings.patch_embedding.bias",
    "fm_modules.vision_model_mot_gen.embeddings.dense_embedding.weight",
    "fm_modules.vision_model_mot_gen.embeddings.dense_embedding.bias",
    "fm_modules.timestep_embedder.mlp.0.weight",
    "fm_modules.timestep_embedder.mlp.0.bias",
    "fm_modules.timestep_embedder.mlp.2.weight",
    "fm_modules.timestep_embedder.mlp.2.bias",
    "fm_modules.noise_scale_embedder.mlp.0.weight",
    "fm_modules.noise_scale_embedder.mlp.0.bias",
    "fm_modules.noise_scale_embedder.mlp.2.weight",
    "fm_modules.noise_scale_embedder.mlp.2.bias",
    "fm_modules.fm_head.conv1.weight",
    "fm_modules.fm_head.conv1.bias",
    "fm_modules.fm_head.conv2.weight",
    "fm_modules.fm_head.conv2.bias",
)


def required_t2i_weight_keys(config: SenseNovaConfig) -> set[str]:
    """Return the exact formal non-thinking T2I subset (1111 checkpoint tensors)."""
    keys = {
        "language_model.model.embed_tokens.weight",
        "language_model.model.norm.weight",
        "language_model.model.norm_mot_gen.weight",
        *COMPONENT_KEYS,
    }
    for layer_index in range(config.num_hidden_layers):
        root = f"language_model.model.layers.{layer_index}"
        for suffix, mlp_name in (("", "mlp"), ("_mot_gen", "mlp_mot_gen")):
            keys.add(f"{root}.input_layernorm{suffix}.weight")
            keys.add(f"{root}.post_attention_layernorm{suffix}.weight")
            for projection in ("q", "k", "v", "o"):
                keys.add(f"{root}.self_attn.{projection}_proj{suffix}.weight")
            for norm in ("q_norm", "q_norm_hw", "k_norm", "k_norm_hw"):
                keys.add(f"{root}.self_attn.{norm}{suffix}.weight")
            for projection in ("gate", "up", "down"):
                keys.add(f"{root}.{mlp_name}.{projection}_proj.weight")
    return keys


def decode_velocity(
    hidden_nhwc: mx.array,
    z: mx.array,
    *,
    t: mx.array | float,
    t_eps: float,
    patch_size: int,
    conv1_weight_oihw: mx.array,
    conv1_bias: mx.array,
    conv2_weight_oihw: mx.array,
    conv2_bias: mx.array,
) -> mx.array:
    """Run the formal pixel head and convert x-prediction to flow velocity."""
    predicted_nhwc = conv_decoder(
        hidden_nhwc,
        conv1_weight_oihw,
        conv1_bias,
        conv2_weight_oihw,
        conv2_bias,
    )
    predicted_nchw = predicted_nhwc.transpose(0, 3, 1, 2)
    predicted_tokens = patchify(predicted_nchw, patch_size)
    if predicted_tokens.shape != z.shape:
        raise ValueError(f"pixel head tokens {predicted_tokens.shape} do not match z {z.shape}")
    denominator = mx.maximum(1.0 - t, t_eps)
    return (predicted_tokens - z) / denominator


class SenseNovaT2IPipeline:
    """Formal non-thinking T2I inference composed entirely from MLX forwards."""

    def __init__(
        self,
        config: SenseNovaConfig,
        weights: Mapping[str, mx.array],
        *,
        adapters=None,
    ) -> None:
        require_keys(weights, COMPONENT_KEYS)
        self.config = config
        self.weights = weights
        self.backbone = Qwen3MoTModel(config, weights, adapters=adapters)

    def _weight(self, key: str) -> mx.array:
        return self.weights[key]

    def build_prefix(self, input_ids: mx.array) -> list[KVCache]:
        if input_ids.ndim != 2 or input_ids.shape[0] != 1:
            raise ValueError(f"formal first-phase runtime expects input_ids [1,S], got {input_ids.shape}")
        indexes = build_text_indexes(int(input_ids.shape[1]))
        mask = create_block_causal_mask(indexes[0])
        _, caches = self.backbone.forward_prefix_ids(input_ids, indexes, mask)
        return caches

    def _step_embeddings(
        self,
        image_prediction: mx.array,
        *,
        t: mx.array | float,
        noise_scale: float,
    ) -> tuple[mx.array, mx.array, int, int]:
        batch, channels, height, width = image_prediction.shape
        if channels != 3:
            raise ValueError(f"expected three image channels, got {image_prediction.shape}")
        patch = self.config.patch_size
        grid_h, grid_w = height // patch, width // patch
        token_h, token_w = grid_h // self.config.merge_size, grid_w // self.config.merge_size
        z = patchify(image_prediction, self.config.effective_patch_size)
        image_patches = patchify(image_prediction, patch, channel_first=True)
        image_embeddings = generation_vision_embeddings(
            image_patches,
            grid_h=grid_h,
            grid_w=grid_w,
            patch_weight_oihw=self._weight(
                "fm_modules.vision_model_mot_gen.embeddings.patch_embedding.weight"
            ),
            patch_bias=self._weight(
                "fm_modules.vision_model_mot_gen.embeddings.patch_embedding.bias"
            ),
            dense_weight_oihw=self._weight(
                "fm_modules.vision_model_mot_gen.embeddings.dense_embedding.weight"
            ),
            dense_bias=self._weight(
                "fm_modules.vision_model_mot_gen.embeddings.dense_embedding.bias"
            ),
            rope_theta=10_000.0,
        )
        count = batch * token_h * token_w
        t_values = mx.full((count,), t, dtype=mx.float32)
        time_embeddings = timestep_mlp(
            t_values,
            first_weight=self._weight("fm_modules.timestep_embedder.mlp.0.weight"),
            first_bias=self._weight("fm_modules.timestep_embedder.mlp.0.bias"),
            second_weight=self._weight("fm_modules.timestep_embedder.mlp.2.weight"),
            second_bias=self._weight("fm_modules.timestep_embedder.mlp.2.bias"),
        )
        noise_values = mx.full(
            (count,),
            noise_scale / self.config.noise_scale_max_value,
            dtype=mx.float32,
        )
        noise_embeddings = timestep_mlp(
            noise_values,
            first_weight=self._weight("fm_modules.noise_scale_embedder.mlp.0.weight"),
            first_bias=self._weight("fm_modules.noise_scale_embedder.mlp.0.bias"),
            second_weight=self._weight("fm_modules.noise_scale_embedder.mlp.2.weight"),
            second_bias=self._weight("fm_modules.noise_scale_embedder.mlp.2.bias"),
        )
        time_embeddings = (time_embeddings + noise_embeddings).reshape(
            batch, token_h * token_w, self.config.hidden_size
        )
        image_embeddings = image_embeddings + time_embeddings
        return z, image_embeddings, token_h, token_w

    def _predict_velocity(
        self,
        image_embeddings: mx.array,
        z: mx.array,
        image_indexes: mx.array,
        caches: Sequence[KVCache],
        *,
        t: mx.array | float,
        token_h: int,
        token_w: int,
    ) -> mx.array:
        hidden = self.backbone.forward_generation(image_embeddings, image_indexes, caches)
        hidden = hidden.reshape(hidden.shape[0], token_h, token_w, self.config.hidden_size)
        velocity = decode_velocity(
            hidden,
            z,
            t=t,
            t_eps=self.config.t_eps,
            patch_size=self.config.effective_patch_size,
            conv1_weight_oihw=self._weight("fm_modules.fm_head.conv1.weight"),
            conv1_bias=self._weight("fm_modules.fm_head.conv1.bias"),
            conv2_weight_oihw=self._weight("fm_modules.fm_head.conv2.weight"),
            conv2_bias=self._weight("fm_modules.fm_head.conv2.bias"),
        )
        mx.eval(velocity)
        return velocity

    def generate_from_ids(
        self,
        condition_ids: mx.array,
        uncondition_ids: mx.array | None,
        *,
        width: int,
        height: int,
        num_steps: int,
        seed: int,
        cfg_scale: float = 4.0,
        cfg_norm: str = "none",
        timestep_shift: float = 3.0,
        cfg_interval: tuple[float, float] = (0.0, 1.0),
        max_active_bytes: int | None = None,
        step_callback: Callable[[int, int], None] | None = None,
    ) -> mx.array:
        """Generate normalized NCHW pixels; tokenization remains a CPU concern."""
        self.config.validate_image_size(width, height)
        if cfg_scale > 1.0 and uncondition_ids is None:
            raise ValueError("uncondition_ids are required when cfg_scale > 1")
        condition_caches = self.build_prefix(condition_ids)
        uncondition_caches = self.build_prefix(uncondition_ids) if uncondition_ids is not None else None
        token_h = height // self.config.effective_patch_size
        token_w = width // self.config.effective_patch_size
        condition_indexes = build_image_indexes(token_h, token_w, int(condition_ids.shape[1]))
        uncondition_indexes = (
            build_image_indexes(token_h, token_w, int(uncondition_ids.shape[1]))
            if uncondition_ids is not None
            else None
        )
        noise_scale = resolution_noise_scale(
            width,
            height,
            patch_size=self.config.patch_size,
            merge_size=self.config.merge_size,
            base_image_seq_len=self.config.noise_scale_base_image_seq_len,
            base_scale=self.config.noise_scale,
            max_value=self.config.noise_scale_max_value,
        )
        key = mx.random.key(seed)
        image_prediction = noise_scale * mx.random.normal(
            (1, 3, height, width),
            dtype=self._weight("language_model.model.norm.weight").dtype,
            key=key,
        )
        timesteps = make_timesteps(num_steps, shift=timestep_shift)
        for step_index in range(num_steps):
            t = timesteps[step_index]
            t_next = timesteps[step_index + 1]
            z, image_embeddings, actual_h, actual_w = self._step_embeddings(
                image_prediction,
                t=t,
                noise_scale=noise_scale,
            )
            if (actual_h, actual_w) != (token_h, token_w):
                raise RuntimeError(
                    f"step token grid changed from {token_h}x{token_w} to {actual_h}x{actual_w}"
                )
            condition_velocity = self._predict_velocity(
                image_embeddings,
                z,
                condition_indexes,
                condition_caches,
                t=t,
                token_h=token_h,
                token_w=token_w,
            )
            use_cfg = cfg_scale > 1.0 and bool(
                cfg_interval[0] <= float(t.item()) <= cfg_interval[1]
            )
            if use_cfg:
                if uncondition_caches is None or uncondition_indexes is None:
                    raise RuntimeError("CFG cache was not initialized")
                uncondition_velocity = self._predict_velocity(
                    image_embeddings,
                    z,
                    uncondition_indexes,
                    uncondition_caches,
                    t=t,
                    token_h=token_h,
                    token_w=token_w,
                )
                velocity = apply_cfg(
                    condition_velocity,
                    uncondition_velocity,
                    scale=cfg_scale,
                    norm=cfg_norm,
                )
            else:
                velocity = condition_velocity
            z = euler_step(z, velocity, t, t_next)
            image_prediction = unpatchify(
                z,
                self.config.effective_patch_size,
                height,
                width,
            )
            mx.eval(image_prediction)
            active = mx.get_active_memory()
            if max_active_bytes is not None and active > max_active_bytes:
                raise MemoryError(
                    f"MLX active memory {active} exceeded stop limit {max_active_bytes} bytes"
                )
            if step_callback is not None:
                step_callback(step_index + 1, num_steps)
        return image_prediction
