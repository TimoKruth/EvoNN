"""Mixture of Experts module with sparse top-k routing."""

import mlx.core as mx
import mlx.nn as nn


class GateNetwork(nn.Module):
    """Lightweight gating network: Linear -> ReLU -> Linear for routing logits."""

    def __init__(self, input_dim: int, num_experts: int, hidden_dim: int = 32):
        super().__init__()
        self.hidden = nn.Linear(input_dim, hidden_dim)
        self.gate = nn.Linear(hidden_dim, num_experts)

    def __call__(self, x: mx.array) -> mx.array:
        return self.gate(nn.relu(self.hidden(x)))


class ExpertNetwork(nn.Module):
    """Single expert MLP with configurable hidden dimensions."""

    def __init__(self, input_dim: int, output_dim: int, hidden_dims: list[int]):
        super().__init__()
        self.layers: list[nn.Linear] = []
        self.activations: list[nn.Module] = []
        prev = input_dim
        for dim in hidden_dims:
            self.layers.append(nn.Linear(prev, dim))
            self.activations.append(nn.ReLU())
            prev = dim
        self.output_proj = nn.Linear(prev, output_dim)

    def __call__(self, x: mx.array) -> mx.array:
        for layer, act in zip(self.layers, self.activations):
            x = act(layer(x))
        return self.output_proj(x)


class MixtureOfExperts(nn.Module):
    """Sparse Mixture of Experts with top-k routing and load balance loss.

    Routes inputs through a gate network to select top-k experts,
    combines expert outputs via weighted sum. Stores load_balance_loss
    as an attribute after each forward pass.
    """

    def __init__(
        self,
        input_dim: int,
        output_dim: int,
        num_experts: int = 4,
        top_k: int = 2,
        gate_hidden_dim: int = 32,
        expert_hidden_dims: list[int] | None = None,
    ):
        super().__init__()
        self.num_experts = num_experts
        self.top_k = min(top_k, num_experts)
        self.output_dim = output_dim

        self.gate = GateNetwork(input_dim, num_experts, gate_hidden_dim)

        if expert_hidden_dims is None:
            expert_hidden_dims = [max(16, output_dim)]
        self.experts: list[ExpertNetwork] = [
            ExpertNetwork(input_dim, output_dim, expert_hidden_dims)
            for _ in range(num_experts)
        ]

        # Pre-allocate as array to avoid None -> array mutation during traced grad pass
        self._last_gate_weights: mx.array = mx.zeros((1, num_experts))

    def __call__(self, x: mx.array) -> mx.array:
        gate_logits = self.gate(x)  # (batch, num_experts)

        if self.top_k < self.num_experts:
            # Sparse routing: mask from detached logits so topk stays off gradient tape
            detached = mx.stop_gradient(gate_logits)
            topk_vals = mx.topk(detached, self.top_k, axis=-1)
            threshold = mx.min(topk_vals, axis=-1, keepdims=True)
            mask = (detached >= threshold).astype(mx.float32)
            masked_logits = gate_logits * mask + (1 - mask) * (-1e9)
            gate_weights = mx.softmax(masked_logits, axis=-1)
        else:
            gate_weights = mx.softmax(gate_logits, axis=-1)

        self._last_gate_weights = mx.stop_gradient(gate_weights)

        combined = mx.zeros((x.shape[0], self.output_dim))
        for i, expert in enumerate(self.experts):
            combined = combined + expert(x) * gate_weights[:, i : i + 1]

        return combined

    def load_balance_loss(self) -> mx.array:
        """Switch Transformer load balance loss: N * sum(f_i^2)."""
        expert_fraction = mx.mean(self._last_gate_weights, axis=0)
        return self.num_experts * mx.sum(expert_fraction * expert_fraction)
