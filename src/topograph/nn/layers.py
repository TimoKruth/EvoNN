"""Quantized and ternary linear layers with straight-through estimators."""

import math

import mlx.core as mx
import mlx.nn as nn


class QuantizedLinear(nn.Module):
    """Simulated INT4/INT8 linear layer with STE for quantization-aware training."""

    def __init__(self, input_dim: int, output_dim: int, bits: int = 8):
        super().__init__()
        self.bits = bits
        self.qmax = (1 << (bits - 1)) - 1
        scale = (2 / (input_dim + output_dim)) ** 0.5
        self.weight = mx.random.normal((output_dim, input_dim)) * scale
        self.bias = mx.zeros((output_dim,))

    def __call__(self, x: mx.array) -> mx.array:
        w_scale = mx.max(mx.abs(self.weight)) / self.qmax + 1e-8
        w_q_int = mx.clip(mx.round(self.weight / w_scale), -self.qmax, self.qmax)
        w_q = mx.stop_gradient(w_q_int * w_scale - self.weight) + self.weight
        return (x @ w_q.T) + self.bias


class BitLinear(nn.Module):
    """Ternary-weight {-1, 0, +1} linear layer with STE."""

    def __init__(self, input_dim: int, output_dim: int):
        super().__init__()
        scale = (2 / (input_dim + output_dim)) ** 0.5
        self.weight = mx.random.normal((output_dim, input_dim)) * scale
        self.bias = mx.zeros((output_dim,))

    def __call__(self, x: mx.array) -> mx.array:
        # Weight quantization to ternary
        w_scale = mx.mean(mx.abs(self.weight)) + 1e-8
        w_ternary = mx.clip(mx.round(self.weight / w_scale), -1, 1)
        w_q = mx.stop_gradient(w_ternary - self.weight) + self.weight

        # Activation quantization to INT8
        x_scale = mx.max(mx.abs(x), axis=-1, keepdims=True) / 127.0 + 1e-8
        x_q_int = mx.clip(mx.round(x / x_scale), -128, 127)
        x_q = mx.stop_gradient(x_q_int * x_scale - x) + x

        return (x_q @ (w_q * w_scale).T) + self.bias


def hadamard_smooth(x: mx.array, bits: int) -> mx.array:
    """Hadamard rotation + fake activation quantization for smoothing.

    Applies Hadamard rotation (pads to power-of-2 if needed), quantize-dequantizes
    activations at the given bit width, then returns the smoothed result.
    """
    if bits >= 16:
        return x
    dim = x.shape[-1]
    padded_dim = 1 << math.ceil(math.log2(max(dim, 2)))

    # Build normalized Hadamard matrix
    H = _hadamard_matrix(padded_dim)

    # Pad if necessary
    if dim < padded_dim:
        padding = mx.zeros((*x.shape[:-1], padded_dim - dim))
        x_padded = mx.concatenate([x, padding], axis=-1)
    else:
        x_padded = x

    # Rotate
    rotated = x_padded @ H
    rotated = rotated[..., :dim]

    # Quantize-dequantize activations
    qmax = (1 << (bits - 1)) - 1
    scale = mx.stop_gradient(mx.max(mx.abs(rotated), axis=-1, keepdims=True) / qmax + 1e-8)
    q = mx.clip(mx.round(rotated / scale), -qmax, qmax)
    return mx.stop_gradient(q * scale - rotated) + rotated


def _hadamard_matrix(n: int) -> mx.array:
    """Normalized Hadamard matrix of size n (must be power of 2)."""
    if n == 1:
        return mx.array([[1.0]])
    half = _hadamard_matrix(n // 2)
    top = mx.concatenate([half, half], axis=1)
    bot = mx.concatenate([half, -half], axis=1)
    return mx.concatenate([top, bot], axis=0) / math.sqrt(2)
