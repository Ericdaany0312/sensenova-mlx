from __future__ import annotations

import math

import mlx.core as mx


def pixel_shuffle_nhwc(x: mx.array, upscale_factor: int) -> mx.array:
    if x.ndim != 4:
        raise ValueError(f"pixel shuffle input must be NHWC, got {x.shape}")
    batch, height, width, channels = x.shape
    divisor = upscale_factor**2
    if channels % divisor:
        raise ValueError(f"channels {channels} not divisible by upscale_factor^2={divisor}")
    output_channels = channels // divisor
    x = x.reshape(batch, height, width, output_channels, upscale_factor, upscale_factor)
    x = x.transpose(0, 1, 4, 2, 5, 3)
    return x.reshape(batch, height * upscale_factor, width * upscale_factor, output_channels)


def _conv2d_oihw(x: mx.array, weight: mx.array, bias: mx.array) -> mx.array:
    if weight.ndim != 4:
        raise ValueError(f"conv weight must be OIHW, got {weight.shape}")
    # MLX Conv2d uses OHWI weights and NHWC activations.
    weight_ohwi = weight.transpose(0, 2, 3, 1)
    return mx.conv2d(x, weight_ohwi, stride=1, padding=1) + bias


def _gelu_exact(x: mx.array) -> mx.array:
    return 0.5 * x * (1.0 + mx.erf(x / math.sqrt(2.0)))


def conv_decoder(
    hidden_nhwc: mx.array,
    conv1_weight_oihw: mx.array,
    conv1_bias: mx.array,
    conv2_weight_oihw: mx.array,
    conv2_bias: mx.array,
) -> mx.array:
    """Official U1.5 ConvDecoder: PS2 -> Conv/GELU -> PS2 -> Conv -> PS8."""
    x = pixel_shuffle_nhwc(hidden_nhwc, 2)
    x = _gelu_exact(_conv2d_oihw(x, conv1_weight_oihw, conv1_bias))
    x = pixel_shuffle_nhwc(x, 2)
    x = _conv2d_oihw(x, conv2_weight_oihw, conv2_bias)
    return pixel_shuffle_nhwc(x, 8)
