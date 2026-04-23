"""Model families for Primordia MLX search — 11 MLX nn.Module subclasses.

Families by modality:
  TABULAR:  FlexMLP, SparseMLP, MoEMLP
  IMAGE:    ImageConvNet, LiteImageConvNet
  SEQUENCE: SequenceConvNet, LiteSequenceConvNet, SequenceGRUNet
  TEXT:     TextEmbeddingModel, AttentionEncoderNet, SparseAttentionNet
"""

from __future__ import annotations

import mlx.core as mx
import mlx.nn as nn

from evonn_primordia.genome import ModelGenome

# ---------------------------------------------------------------------------
# Shared helpers
# ---------------------------------------------------------------------------

_ACTIVATIONS: dict[str, callable] = {
    "relu": nn.relu,
    "gelu": nn.gelu,
    "tanh": nn.tanh,
    "silu": nn.silu,
}


def _get_activation(name: str):
    """Return activation function by name."""
    if name not in _ACTIVATIONS:
        raise ValueError(f"Unsupported activation: {name!r}. Choose from {list(_ACTIVATIONS)}")
    return _ACTIVATIONS[name]


def _make_norm(norm_type: str, dims: int) -> nn.Module | None:
    """Create a normalization layer, or None for 'none'."""
    if norm_type == "layer":
        return nn.LayerNorm(dims)
    if norm_type == "rms":
        return nn.RMSNorm(dims)
    if norm_type == "batch":
        return nn.BatchNorm(dims)
    return None


def _make_norms(norm_type: str, dims_list: list[int]) -> list[nn.Module]:
    """Create normalization layers for each dimension in the list."""
    if norm_type == "none":
        return []
    return [_make_norm(norm_type, d) for d in dims_list]


def _apply_norm(x, norm_layer: nn.Module | None):
    """Apply normalization if the layer exists."""
    return norm_layer(x) if norm_layer is not None else x


def _apply_sparsity(x, ratio: float):
    """Zero the smallest `ratio` fraction of activations by magnitude."""
    if ratio <= 0.0:
        return x
    keep = max(0.0, min(1.0, 1.0 - ratio))
    width = x.shape[-1]
    if width <= 0:
        return x
    keep_count = max(1, int(round(width * keep)))
    if keep_count >= width:
        return x
    threshold = mx.sort(mx.abs(x), axis=-1)[..., -keep_count][..., None]
    mask = (mx.abs(x) >= threshold).astype(x.dtype)
    return x * mask


def _resolve_heads(dims: int, requested: int) -> int:
    """Pick largest valid head count <= requested that divides dims evenly."""
    valid = [h for h in [1, 2, 4, 8] if h <= requested and dims % h == 0]
    if valid:
        return max(valid)
    divisors = [h for h in [1, 2, 4, 8] if dims % h == 0]
    return max(divisors) if divisors else 1


def _flat_dim(input_shape: list[int]) -> int:
    """Product of all dimensions in input_shape."""
    result = 1
    for d in input_shape:
        result *= d
    return result


# ===========================================================================
# TABULAR families
# ===========================================================================


class FlexMLP(nn.Module):
    """Linear layers with optional residual + normalization + dropout."""

    def __init__(self, genome: ModelGenome, input_shape: list[int], output_dim: int) -> None:
        super().__init__()
        in_dim = _flat_dim(input_shape)
        hl = genome.hidden_layers

        self.layers = [
            nn.Linear(in_dim if i == 0 else hl[i - 1], w) for i, w in enumerate(hl)
        ]
        self.norms = _make_norms(genome.norm_type, hl)
        self.head = nn.Linear(hl[-1] if hl else in_dim, output_dim)
        self.act = _get_activation(genome.activation)
        self.drop = nn.Dropout(genome.dropout) if genome.dropout > 0 else None
        self.residual = genome.residual

    def __call__(self, x):
        h = x.reshape(x.shape[0], -1) if x.ndim > 2 else x
        for i, layer in enumerate(self.layers):
            prev = h
            h = self.act(layer(h))
            if i < len(self.norms):
                h = self.norms[i](h)
            if self.drop is not None:
                h = self.drop(h)
            if self.residual and h.shape == prev.shape:
                h = h + prev
        return self.head(h)


class SparseMLP(nn.Module):
    """MLP with activation sparsity (top-k zeroing per layer)."""

    def __init__(self, genome: ModelGenome, input_shape: list[int], output_dim: int) -> None:
        super().__init__()
        in_dim = _flat_dim(input_shape)
        hl = genome.hidden_layers

        self.layers = [
            nn.Linear(in_dim if i == 0 else hl[i - 1], w) for i, w in enumerate(hl)
        ]
        self.norms = _make_norms(genome.norm_type, hl)
        self.head = nn.Linear(hl[-1] if hl else in_dim, output_dim)
        self.act = _get_activation(genome.activation)
        self.drop = nn.Dropout(genome.dropout) if genome.dropout > 0 else None
        self.residual = genome.residual
        self.sparsity = genome.activation_sparsity

    def __call__(self, x):
        h = x.reshape(x.shape[0], -1) if x.ndim > 2 else x
        for i, layer in enumerate(self.layers):
            prev = h
            h = self.act(layer(h))
            if i < len(self.norms):
                h = self.norms[i](h)
            if self.sparsity > 0:
                h = _apply_sparsity(h, self.sparsity)
            if self.drop is not None:
                h = self.drop(h)
            if self.residual and h.shape == prev.shape:
                h = h + prev
        return self.head(h)


class MoEMLP(nn.Module):
    """Gated Mixture-of-Experts with top-k expert routing."""

    def __init__(self, genome: ModelGenome, input_shape: list[int], output_dim: int) -> None:
        super().__init__()
        in_dim = _flat_dim(input_shape)
        expert_width = genome.hidden_layers[0] if genome.hidden_layers else 128
        n_experts = max(2, genome.num_experts)
        self.n_experts = n_experts
        self.top_k = min(genome.moe_top_k, n_experts)

        self.gate = nn.Linear(in_dim, n_experts)
        self.expert_in = [nn.Linear(in_dim, expert_width) for _ in range(n_experts)]
        self.expert_out = [nn.Linear(expert_width, expert_width) for _ in range(n_experts)]
        self.norm = _make_norm(genome.norm_type, expert_width)
        self.act = _get_activation(genome.activation)
        self.drop = nn.Dropout(genome.dropout) if genome.dropout > 0 else None
        self.head = nn.Linear(expert_width, output_dim)

    def __call__(self, x):
        h = x.reshape(x.shape[0], -1) if x.ndim > 2 else x

        # Gating with top-k masking
        logits = self.gate(h)  # (B, n_experts)
        if self.top_k < self.n_experts:
            sorted_l = mx.sort(logits, axis=-1)
            threshold = sorted_l[:, -self.top_k].reshape(-1, 1)
            mask = (logits >= threshold).astype(logits.dtype)
            weights = mx.softmax(logits * mask + (1.0 - mask) * -1e9, axis=-1)
        else:
            weights = mx.softmax(logits, axis=-1)

        # Weighted expert outputs
        combined = mx.zeros((h.shape[0], self.expert_out[0].weight.shape[0]))
        for i in range(self.n_experts):
            out = self.act(self.expert_in[i](h))
            out = self.act(self.expert_out[i](out))
            combined = combined + weights[:, i : i + 1] * out

        if self.norm is not None:
            combined = self.norm(combined)
        if self.drop is not None:
            combined = self.drop(combined)
        return self.head(combined)


# ===========================================================================
# IMAGE families
# ===========================================================================


class ImageConvNet(nn.Module):
    """Conv2d layers + global average pooling + FC head."""

    def __init__(self, genome: ModelGenome, input_shape: list[int], output_dim: int) -> None:
        super().__init__()
        if len(input_shape) != 3:
            raise ValueError(f"ImageConvNet expects [H, W, C] input, got {input_shape}")
        in_ch = input_shape[-1]
        channels = genome.hidden_layers
        ks = genome.kernel_size

        self.convs = [
            nn.Conv2d(in_ch if i == 0 else channels[i - 1], c, kernel_size=ks, padding=ks // 2)
            for i, c in enumerate(channels)
        ]
        self.norms = _make_norms(genome.norm_type, channels)
        self.act = _get_activation(genome.activation)
        self.drop = nn.Dropout(genome.dropout) if genome.dropout > 0 else None
        self.head = nn.Linear(channels[-1], output_dim)

    def __call__(self, x):
        h = x
        for i, conv in enumerate(self.convs):
            h = self.act(conv(h))
            if i < len(self.norms):
                h = self.norms[i](h)
            if self.drop is not None:
                h = self.drop(h)
        # Global average pool over spatial dims (H, W)
        pooled = mx.mean(h, axis=(1, 2))
        return self.head(pooled)


class LiteImageConvNet(nn.Module):
    """Depthwise-separable Conv2d (bottleneck) + global average pooling + FC head."""

    def __init__(self, genome: ModelGenome, input_shape: list[int], output_dim: int) -> None:
        super().__init__()
        if len(input_shape) != 3:
            raise ValueError(f"LiteImageConvNet expects [H, W, C] input, got {input_shape}")
        in_ch = input_shape[-1]
        channels = genome.hidden_layers
        ks = genome.kernel_size

        self.reduce = []
        self.spatial = []
        self.expand = []
        for i, c in enumerate(channels):
            src = in_ch if i == 0 else channels[i - 1]
            neck = max(8, c // 4)
            self.reduce.append(nn.Conv2d(src, neck, kernel_size=1, padding=0))
            self.spatial.append(nn.Conv2d(neck, neck, kernel_size=ks, padding=ks // 2))
            self.expand.append(nn.Conv2d(neck, c, kernel_size=1, padding=0))

        self.norms = _make_norms(genome.norm_type, channels)
        self.act = _get_activation(genome.activation)
        self.drop = nn.Dropout(genome.dropout) if genome.dropout > 0 else None
        self.head = nn.Linear(channels[-1], output_dim)

    def __call__(self, x):
        h = x
        for i, (red, spa, exp) in enumerate(zip(self.reduce, self.spatial, self.expand)):
            h = self.act(red(h))
            h = self.act(spa(h))
            h = self.act(exp(h))
            if i < len(self.norms):
                h = self.norms[i](h)
            if self.drop is not None:
                h = self.drop(h)
        pooled = mx.mean(h, axis=(1, 2))
        return self.head(pooled)


# ===========================================================================
# SEQUENCE families
# ===========================================================================


class SequenceConvNet(nn.Module):
    """Conv1d layers + global average pooling + FC head."""

    def __init__(self, genome: ModelGenome, input_shape: list[int], output_dim: int) -> None:
        super().__init__()
        if len(input_shape) != 2:
            raise ValueError(f"SequenceConvNet expects [length, channels] input, got {input_shape}")
        in_ch = input_shape[-1]
        channels = genome.hidden_layers
        ks = genome.kernel_size

        self.convs = [
            nn.Conv1d(in_ch if i == 0 else channels[i - 1], c, kernel_size=ks, padding=ks // 2)
            for i, c in enumerate(channels)
        ]
        self.norms = _make_norms(genome.norm_type, channels)
        self.act = _get_activation(genome.activation)
        self.drop = nn.Dropout(genome.dropout) if genome.dropout > 0 else None
        self.head = nn.Linear(channels[-1], output_dim)

    def __call__(self, x):
        h = x
        for i, conv in enumerate(self.convs):
            h = self.act(conv(h))
            if i < len(self.norms):
                h = self.norms[i](h)
            if self.drop is not None:
                h = self.drop(h)
        pooled = mx.mean(h, axis=1)
        return self.head(pooled)


class LiteSequenceConvNet(nn.Module):
    """Depthwise-separable Conv1d (bottleneck) + global average pooling + FC head."""

    def __init__(self, genome: ModelGenome, input_shape: list[int], output_dim: int) -> None:
        super().__init__()
        if len(input_shape) != 2:
            raise ValueError(f"LiteSequenceConvNet expects [length, channels] input, got {input_shape}")
        in_ch = input_shape[-1]
        channels = genome.hidden_layers
        ks = genome.kernel_size

        self.reduce = []
        self.spatial = []
        self.expand = []
        for i, c in enumerate(channels):
            src = in_ch if i == 0 else channels[i - 1]
            neck = max(8, c // 4)
            self.reduce.append(nn.Conv1d(src, neck, kernel_size=1, padding=0))
            self.spatial.append(nn.Conv1d(neck, neck, kernel_size=ks, padding=ks // 2))
            self.expand.append(nn.Conv1d(neck, c, kernel_size=1, padding=0))

        self.norms = _make_norms(genome.norm_type, channels)
        self.act = _get_activation(genome.activation)
        self.drop = nn.Dropout(genome.dropout) if genome.dropout > 0 else None
        self.head = nn.Linear(channels[-1], output_dim)

    def __call__(self, x):
        h = x
        for i, (red, spa, exp) in enumerate(zip(self.reduce, self.spatial, self.expand)):
            h = self.act(red(h))
            h = self.act(spa(h))
            h = self.act(exp(h))
            if i < len(self.norms):
                h = self.norms[i](h)
            if self.drop is not None:
                h = self.drop(h)
        pooled = mx.mean(h, axis=1)
        return self.head(pooled)


class SequenceGRUNet(nn.Module):
    """GRU layers + projection + FC head. Uses last hidden state."""

    def __init__(self, genome: ModelGenome, input_shape: list[int], output_dim: int) -> None:
        super().__init__()
        if len(input_shape) != 2:
            raise ValueError(f"SequenceGRUNet expects [length, channels] input, got {input_shape}")
        in_ch = input_shape[-1]
        hidden = genome.hidden_layers[-1]

        self.gru = nn.GRU(in_ch, hidden)
        self.project = nn.Linear(hidden, hidden)
        self.norm = _make_norm(genome.norm_type, hidden)
        self.act = _get_activation(genome.activation)
        self.drop = nn.Dropout(genome.dropout) if genome.dropout > 0 else None
        self.head = nn.Linear(hidden, output_dim)

    def __call__(self, x):
        h = self.gru(x)
        h = h[:, -1, :]  # last time step
        h = self.act(self.project(h))
        if self.norm is not None:
            h = self.norm(h)
        if self.drop is not None:
            h = self.drop(h)
        return self.head(h)


# ===========================================================================
# TEXT families
# ===========================================================================

_DEFAULT_VOCAB_SIZE = 64


class TextEmbeddingModel(nn.Module):
    """Token embedding -> mean pool -> MLP head."""

    def __init__(
        self,
        genome: ModelGenome,
        input_shape: list[int],
        output_dim: int,
        task: str = "classification",
    ) -> None:
        super().__init__()
        edim = genome.embedding_dim
        hl = genome.hidden_layers
        self.task = task

        vocab_size = max(_DEFAULT_VOCAB_SIZE, output_dim)
        self.embedding = nn.Embedding(vocab_size, edim)
        self.layers = [
            nn.Linear(edim if i == 0 else hl[i - 1], w) for i, w in enumerate(hl)
        ]
        self.norms = _make_norms(genome.norm_type, hl)
        self.head = nn.Linear(hl[-1] if hl else edim, output_dim)
        self.act = _get_activation(genome.activation)
        self.drop = nn.Dropout(genome.dropout) if genome.dropout > 0 else None

    def __call__(self, x):
        h = self.embedding(x)
        if self.task != "language_modeling":
            h = mx.mean(h, axis=1)
        for i, layer in enumerate(self.layers):
            h = self.act(layer(h))
            if i < len(self.norms):
                h = self.norms[i](h)
            if self.drop is not None:
                h = self.drop(h)
        return self.head(h)


class _TransformerBlock(nn.Module):
    """Pre-norm transformer encoder block with RoPE on Q/K."""

    def __init__(
        self,
        *,
        dims: int,
        num_heads: int,
        mlp_dims: int,
        dropout: float,
        activation,
        causal: bool = False,
    ) -> None:
        super().__init__()
        self.dims = dims
        self.num_heads = num_heads
        self.head_dim = dims // num_heads
        self.scale = self.head_dim ** -0.5
        self.causal = causal

        self.q_proj = nn.Linear(dims, dims)
        self.k_proj = nn.Linear(dims, dims)
        self.v_proj = nn.Linear(dims, dims)
        self.out_proj = nn.Linear(dims, dims)
        self.rope = nn.RoPE(self.head_dim)

        self.norm1 = nn.LayerNorm(dims)
        self.norm2 = nn.LayerNorm(dims)
        self.ff1 = nn.Linear(dims, mlp_dims)
        self.ff2 = nn.Linear(mlp_dims, dims)
        self.act = activation
        self.drop = nn.Dropout(dropout) if dropout > 0 else None

    def __call__(self, x):
        B, T, _ = x.shape

        # Self-attention with RoPE
        h = self.norm1(x)
        q = self.q_proj(h).reshape(B, T, self.num_heads, self.head_dim).transpose(0, 2, 1, 3)
        k = self.k_proj(h).reshape(B, T, self.num_heads, self.head_dim).transpose(0, 2, 1, 3)
        v = self.v_proj(h).reshape(B, T, self.num_heads, self.head_dim).transpose(0, 2, 1, 3)
        q, k = self.rope(q), self.rope(k)

        attn = mx.fast.scaled_dot_product_attention(
            q,
            k,
            v,
            scale=self.scale,
            mask="causal" if self.causal else None,
        )
        attn = attn.transpose(0, 2, 1, 3).reshape(B, T, self.dims)
        attn = self.out_proj(attn)
        if self.drop is not None:
            attn = self.drop(attn)
        x = x + attn

        # Feed-forward
        h = self.norm2(x)
        ffn = self.ff2(self.act(self.ff1(h)))
        if self.drop is not None:
            ffn = self.drop(ffn)
        return x + ffn


class AttentionEncoderNet(nn.Module):
    """Multi-head transformer encoder with RoPE. Supports sequence and text input."""

    def __init__(
        self,
        genome: ModelGenome,
        input_shape: list[int],
        output_dim: int,
        task: str = "classification",
    ) -> None:
        super().__init__()
        is_text = len(input_shape) == 1
        depth = len(genome.hidden_layers)
        act_fn = _get_activation(genome.activation)
        self.task = task

        if is_text:
            dims = max(genome.embedding_dim, 8)
            num_heads = _resolve_heads(dims, genome.num_heads)
            mlp_dims = max(max(genome.hidden_layers), genome.embedding_dim)
            vocab_size = max(_DEFAULT_VOCAB_SIZE, output_dim)
            self.embedding = nn.Embedding(vocab_size, dims)
            self.input_proj = None
        else:
            if len(input_shape) != 2:
                raise ValueError(f"AttentionEncoderNet expects [length, channels] or [seq_len], got {input_shape}")
            dims = genome.hidden_layers[0]
            num_heads = _resolve_heads(dims, genome.num_heads)
            mlp_dims = max(genome.hidden_layers)
            self.embedding = None
            self.input_proj = nn.Linear(input_shape[-1], dims)

        self.blocks = [
            _TransformerBlock(
                dims=dims, num_heads=num_heads, mlp_dims=mlp_dims,
                dropout=genome.dropout, activation=act_fn, causal=task == "language_modeling",
            )
            for _ in range(depth)
        ]
        self.final_norm = _make_norm(genome.norm_type, dims)
        self.drop = nn.Dropout(genome.dropout) if genome.dropout > 0 else None
        self.head = nn.Linear(dims, output_dim)

    def __call__(self, x):
        if self.embedding is not None:
            h = self.embedding(x)
        else:
            h = self.input_proj(x)
        for block in self.blocks:
            h = block(h)
        if self.final_norm is not None:
            h = self.final_norm(h)
        if self.task == "language_modeling":
            if self.drop is not None:
                h = self.drop(h)
            return self.head(h)
        pooled = mx.mean(h, axis=1)
        if self.drop is not None:
            pooled = self.drop(pooled)
        return self.head(pooled)


class SparseAttentionNet(nn.Module):
    """Transformer encoder + activation sparsity after each block."""

    def __init__(
        self,
        genome: ModelGenome,
        input_shape: list[int],
        output_dim: int,
        task: str = "classification",
    ) -> None:
        super().__init__()
        is_text = len(input_shape) == 1
        depth = len(genome.hidden_layers)
        act_fn = _get_activation(genome.activation)
        self.task = task

        if is_text:
            dims = max(genome.embedding_dim, 8)
            num_heads = _resolve_heads(dims, genome.num_heads)
            mlp_dims = max(max(genome.hidden_layers), genome.embedding_dim)
            vocab_size = max(_DEFAULT_VOCAB_SIZE, output_dim)
            self.embedding = nn.Embedding(vocab_size, dims)
            self.input_proj = None
        else:
            if len(input_shape) != 2:
                raise ValueError(
                    f"SparseAttentionNet expects [length, channels] or [seq_len], got {input_shape}"
                )
            dims = genome.hidden_layers[0]
            num_heads = _resolve_heads(dims, genome.num_heads)
            mlp_dims = max(genome.hidden_layers)
            self.embedding = None
            self.input_proj = nn.Linear(input_shape[-1], dims)

        self.blocks = [
            _TransformerBlock(
                dims=dims, num_heads=num_heads, mlp_dims=mlp_dims,
                dropout=genome.dropout, activation=act_fn, causal=task == "language_modeling",
            )
            for _ in range(depth)
        ]
        self.final_norm = _make_norm(genome.norm_type, dims)
        self.drop = nn.Dropout(genome.dropout) if genome.dropout > 0 else None
        self.sparsity = genome.activation_sparsity
        self.head = nn.Linear(dims, output_dim)

    def __call__(self, x):
        if self.embedding is not None:
            h = self.embedding(x)
        else:
            h = self.input_proj(x)
        for block in self.blocks:
            h = block(h)
            if self.sparsity > 0:
                h = _apply_sparsity(h, self.sparsity)
        if self.final_norm is not None:
            h = self.final_norm(h)
        if self.task == "language_modeling":
            if self.drop is not None:
                h = self.drop(h)
            return self.head(h)
        pooled = mx.mean(h, axis=1)
        if self.drop is not None:
            pooled = self.drop(pooled)
        return self.head(pooled)


# ---------------------------------------------------------------------------
# Registry: family name -> model class
# ---------------------------------------------------------------------------

FAMILY_CLASSES: dict[str, type[nn.Module]] = {
    "mlp": FlexMLP,
    "sparse_mlp": SparseMLP,
    "moe_mlp": MoEMLP,
    "conv2d": ImageConvNet,
    "lite_conv2d": LiteImageConvNet,
    "conv1d": SequenceConvNet,
    "lite_conv1d": LiteSequenceConvNet,
    "gru": SequenceGRUNet,
    "embedding": TextEmbeddingModel,
    "attention": AttentionEncoderNet,
    "sparse_attention": SparseAttentionNet,
}
