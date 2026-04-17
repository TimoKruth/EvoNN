"""Compile a HybridGenome into an MLX nn.Module."""
from __future__ import annotations

import math

import mlx.core as mx
import mlx.nn as nn

from evonn_compare.hybrid.genome import (
    HybridFamily,
    HybridGenome,
    HybridNodeGene,
    INPUT_INNOVATION,
    OUTPUT_INNOVATION,
)


def _get_activation(name: str):
    return {"relu": nn.ReLU(), "gelu": nn.GELU(), "silu": nn.SiLU(), "tanh": nn.Tanh()}.get(name, nn.ReLU())


class FamilyBlock(nn.Module):
    """A macro block implementing one family type."""

    def __init__(self, input_dim: int, node: HybridNodeGene):
        super().__init__()
        self.node = node
        self.family = node.family
        self._activation = _get_activation(node.activation)
        self.dropout = nn.Dropout(node.dropout) if node.dropout > 0 else None
        self.norms = [nn.LayerNorm(node.width) for _ in range(node.internal_layers)] if node.norm_type == "layer" else []

        if self.family == HybridFamily.ATTENTION:
            self._init_attention_layers(input_dim, node)
        else:
            self.layers = []
            current_dim = input_dim
            for _ in range(node.internal_layers):
                self.layers.append(nn.Linear(current_dim, node.width))
                current_dim = node.width

    def _init_attention_layers(self, input_dim: int, node: HybridNodeGene) -> None:
        self.q_layers = []
        self.k_layers = []
        self.v_layers = []
        self.o_layers = []
        current_dim = input_dim
        for _ in range(node.internal_layers):
            self.q_layers.append(nn.Linear(current_dim, node.width))
            self.k_layers.append(nn.Linear(current_dim, node.width))
            self.v_layers.append(nn.Linear(current_dim, node.width))
            self.o_layers.append(nn.Linear(node.width, node.width))
            current_dim = node.width

    def __call__(self, x: mx.array) -> mx.array:
        if self.family == HybridFamily.ATTENTION:
            x = self._forward_attention(x)
        elif self.family == HybridFamily.CONV2D:
            x = self._forward_conv_like(x)
        else:
            x = self._forward_dense(x)
        if self.dropout is not None:
            x = self.dropout(x)
        return x

    def _forward_dense(self, x: mx.array) -> mx.array:
        for index, layer in enumerate(self.layers):
            x = self._activation(layer(x))
            if self.node.family == HybridFamily.SPARSE_MLP:
                x = _apply_activation_sparsity(x)
            if self.node.norm_type == "layer":
                x = self.norms[index](x)
        return x

    def _forward_conv_like(self, x: mx.array) -> mx.array:
        radius = max(1, self.node.kernel_size // 2)
        # For 3D input (batch, seq, features): mix along the sequence dimension
        # For 2D input (batch, features): mix along the feature dimension (original)
        roll_axis = 1 if x.ndim == 3 else -1
        for index, layer in enumerate(self.layers):
            local = x
            for shift in range(1, radius + 1):
                local = local + mx.roll(x, shift=shift, axis=roll_axis) + mx.roll(x, shift=-shift, axis=roll_axis)
            local = local / float((radius * 2) + 1)
            x = self._activation(layer(local))
            if self.node.norm_type == "layer":
                x = self.norms[index](x)
        return x

    def _forward_attention(self, x: mx.array) -> mx.array:
        # 3D input (batch, seq, features) -> cross-position causal attention
        # 2D input (batch, features) -> per-sample feature attention (original)
        use_causal = x.ndim == 3
        for index, (q_layer, k_layer, v_layer, o_layer) in enumerate(
            zip(self.q_layers, self.k_layers, self.v_layers, self.o_layers, strict=True)
        ):
            q = q_layer(x)
            k = k_layer(x)
            v = v_layer(x)
            if use_causal:
                attn = _causal_sequence_attention(q, k, v, self.node.num_heads)
            else:
                attn = _feature_attention(q, k, v, self.node.num_heads)
            x = self._activation(o_layer(attn))
            if self.node.norm_type == "layer":
                x = self.norms[index](x)
        return x


def _apply_activation_sparsity(x: mx.array) -> mx.array:
    threshold = mx.mean(mx.abs(x), axis=-1, keepdims=True)
    mask = mx.abs(x) >= threshold
    return x * mask.astype(x.dtype)


def _feature_attention(q: mx.array, k: mx.array, v: mx.array, num_heads: int) -> mx.array:
    width = q.shape[-1]
    if width == 0:
        return v
    heads = max(1, min(num_heads, width))
    if width % heads != 0:
        heads = 1
    head_dim = width // heads
    scale = 1.0 / math.sqrt(max(head_dim, 1))

    qh = q.reshape(q.shape[0], heads, head_dim)
    kh = k.reshape(k.shape[0], heads, head_dim)
    vh = v.reshape(v.shape[0], heads, head_dim)

    scores = mx.expand_dims(qh, -1) * mx.expand_dims(kh, -2) * scale
    weights = mx.softmax(scores, axis=-1)
    attended = mx.matmul(weights, mx.expand_dims(vh, -1)).squeeze(-1)
    return attended.reshape(q.shape[0], width)


def _causal_sequence_attention(q: mx.array, k: mx.array, v: mx.array, num_heads: int) -> mx.array:
    """Scaled dot-product attention over sequence positions with a causal mask.

    Args:
        q, k, v: (batch, seq, width) tensors after linear projection.
        num_heads: desired number of attention heads (adjusted to divide width).

    Returns:
        (batch, seq, width) attended output.
    """
    batch, seq, width = q.shape
    if width == 0:
        return v
    heads = max(1, min(num_heads, width))
    if width % heads != 0:
        heads = 1
    head_dim = width // heads
    scale = 1.0 / math.sqrt(max(head_dim, 1))

    # Reshape to (batch, seq, heads, head_dim) then transpose to (batch, heads, seq, head_dim)
    q = q.reshape(batch, seq, heads, head_dim).transpose(0, 2, 1, 3)
    k = k.reshape(batch, seq, heads, head_dim).transpose(0, 2, 1, 3)
    v = v.reshape(batch, seq, heads, head_dim).transpose(0, 2, 1, 3)

    # Scaled dot-product: (batch, heads, seq, seq)
    scores = (q @ k.transpose(0, 1, 3, 2)) * scale

    # Causal mask: prevent attending to future positions
    mask = mx.triu(mx.full((seq, seq), -1e9), k=1)
    scores = scores + mask

    weights = mx.softmax(scores, axis=-1)
    out = weights @ v  # (batch, heads, seq, head_dim)
    out = out.transpose(0, 2, 1, 3).reshape(batch, seq, width)
    return out


class HybridModel(nn.Module):
    """Compiled hybrid DAG model."""

    def __init__(self, genome: HybridGenome, input_dim: int, num_classes: int, task: str = "classification"):
        super().__init__()
        self.task = task
        self.input_dim = input_dim
        self.num_classes = num_classes

        sorted_nodes = sorted(genome.enabled_nodes, key=lambda n: n.order)
        connections = genome.enabled_connections

        dims = {INPUT_INNOVATION: input_dim}
        for node in sorted_nodes:
            dims[node.innovation_number] = node.width

        valid_ids = set(dims.keys()) | {OUTPUT_INNOVATION}
        connections = [
            conn
            for conn in connections
            if conn.source_innovation in valid_ids and conn.target_innovation in valid_ids
        ]

        self.blocks: dict[int, FamilyBlock] = {}
        for node in sorted_nodes:
            self.blocks[node.innovation_number] = FamilyBlock(node.width, node)

        self.projections: dict[str, nn.Linear] = {}
        for conn in connections:
            src_dim = dims.get(conn.source_innovation, input_dim)
            tgt_dim = dims[conn.target_innovation] if conn.target_innovation != OUTPUT_INNOVATION else num_classes
            key = f"proj_{conn.innovation_number}"
            proj = nn.Linear(src_dim, tgt_dim)
            new_w = mx.random.normal(proj.weight.shape) * (2.0 / proj.weight.shape[1]) ** 0.5
            proj.weight = new_w
            self.projections[key] = proj

        self._node_inns = [node.innovation_number for node in sorted_nodes]
        self._conn_routing: dict[int, list[tuple[str, int]]] = {}
        for inn in self._node_inns:
            self._conn_routing[inn] = [
                (f"proj_{conn.innovation_number}", conn.source_innovation)
                for conn in connections
                if conn.target_innovation == inn
            ]
        self._output_routing = [
            (f"proj_{conn.innovation_number}", conn.source_innovation)
            for conn in connections
            if conn.target_innovation == OUTPUT_INNOVATION
        ]

    def __call__(self, x: mx.array) -> mx.array:
        outputs = {INPUT_INNOVATION: x}

        for inn in self._node_inns:
            incoming = [
                self.projections[proj_key](outputs[src_inn])
                for proj_key, src_inn in self._conn_routing[inn]
                if src_inn in outputs
            ]
            if not incoming:
                continue
            summed = incoming[0]
            for tensor in incoming[1:]:
                summed = summed + tensor
            outputs[inn] = self.blocks[inn](summed)

        out_incoming = [
            self.projections[proj_key](outputs[src_inn])
            for proj_key, src_inn in self._output_routing
            if src_inn in outputs
        ]
        if not out_incoming:
            last_inn = self._node_inns[-1] if self._node_inns else INPUT_INNOVATION
            if last_inn in outputs:
                fallback_proj = nn.Linear(outputs[last_inn].shape[-1], self.num_classes)
                out_incoming.append(fallback_proj(outputs[last_inn]))

        logits = out_incoming[0]
        for tensor in out_incoming[1:]:
            logits = logits + tensor

        if self.task == "regression":
            return logits
        return mx.softmax(logits, axis=-1)


class CausalHybridModel(nn.Module):
    """Hybrid model wrapped for language modeling with token embeddings and causal attention.

    Wraps a HybridModel (which operates on feature vectors) by:
    1. Embedding input token IDs into dense vectors
    2. Adding positional encoding
    3. Running the DAG on the embedded sequence (each position independently)
    4. Projecting to vocabulary logits via an LM head
    """

    def __init__(self, genome: HybridGenome, vocab_size: int, embed_dim: int = 128, seq_len: int = 128):
        super().__init__()
        self.vocab_size = vocab_size
        self.embed_dim = embed_dim
        self.seq_len = seq_len
        self.task = "language_modeling"
        self.num_classes = vocab_size

        # Token + position embeddings
        self.token_embed = nn.Embedding(vocab_size, embed_dim)
        self.pos_embed = nn.Embedding(seq_len, embed_dim)

        # Inner DAG model operates on embed_dim features per position
        self.inner = HybridModel(genome, input_dim=embed_dim, num_classes=embed_dim, task="regression")

        # LM head projects back to vocabulary
        self.lm_head = nn.Linear(embed_dim, vocab_size)
        self.norm = nn.RMSNorm(embed_dim)

    def __call__(self, x: mx.array) -> mx.array:
        # x: (batch, seq_len) integer token IDs
        batch_size, seq_len = x.shape

        # Embed tokens + positions
        positions = mx.arange(seq_len)
        tok_emb = self.token_embed(x.astype(mx.int32))        # (batch, seq, embed)
        pos_emb = self.pos_embed(positions)                     # (seq, embed)
        h = tok_emb + pos_emb                                   # (batch, seq, embed)

        # Run DAG on full 3D tensor so attention blocks attend across positions.
        # nn.Linear handles arbitrary leading dims, so MLP/dense paths work on 3D.
        # Attention blocks detect ndim==3 and use causal sequence attention.
        h_out = self.inner(h)                                   # (batch, seq, embed)

        # Norm + LM head
        h_out = self.norm(h_out)
        logits = self.lm_head(h_out)                            # (batch, seq, vocab)

        return logits


def compile_hybrid(genome: HybridGenome, input_dim: int, num_classes: int, task: str = "classification") -> HybridModel:
    """Compile a HybridGenome into an MLX model."""
    if task == "language_modeling" or (task == "classification" and num_classes > 200):
        return CausalHybridModel(genome, vocab_size=num_classes, embed_dim=min(128, input_dim))
    return HybridModel(genome, input_dim, num_classes, task)
