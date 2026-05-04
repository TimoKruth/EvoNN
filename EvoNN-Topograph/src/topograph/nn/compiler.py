"""Compile a Genome into a trainable MLX nn.Module."""

from __future__ import annotations

import math

try:  # pragma: no cover - depends on host runtime
    import mlx.core as mx
    import mlx.nn as nn

    MLX_AVAILABLE = True
except ImportError:  # pragma: no cover - covered by fallback-only hosts
    mx = None

    class _UnavailableModule:
        pass

    class _UnavailableLayer:
        def __init__(self, *args, **kwargs) -> None:
            raise RuntimeError("MLX compiler backend is unavailable.")

    class _UnavailableNN:
        Module = _UnavailableModule
        ReLU = _UnavailableLayer
        Sigmoid = _UnavailableLayer
        Tanh = _UnavailableLayer
        GELU = _UnavailableLayer
        SiLU = _UnavailableLayer
        Linear = _UnavailableLayer
        Conv2d = _UnavailableLayer
        LayerNorm = _UnavailableLayer
        Embedding = _UnavailableLayer

    nn = _UnavailableNN()
    MLX_AVAILABLE = False

from topograph.genome.genes import (
    Activation,
    OperatorType,
    WeightBits,
)
from topograph.genome.genome import INPUT_INNOVATION, OUTPUT_INNOVATION, Genome
if MLX_AVAILABLE:
    from topograph.nn.layers import BitLinear, QuantizedLinear, hadamard_smooth
    from topograph.nn.moe import MixtureOfExperts
else:  # pragma: no cover - fallback-only import path
    BitLinear = QuantizedLinear = _UnavailableLayer

    def hadamard_smooth(x, bits):
        return x

    MixtureOfExperts = _UnavailableLayer

_ACTIVATION_MAP = {
    Activation.RELU: nn.ReLU,
    Activation.SIGMOID: nn.Sigmoid,
    Activation.TANH: nn.Tanh,
    Activation.GELU: nn.GELU,
    Activation.SILU: nn.SiLU,
}


def _make_projection(src_dim: int, tgt_dim: int, weight_bits: WeightBits) -> nn.Module:
    """Create a projection layer with appropriate precision and Kaiming init."""
    if weight_bits == WeightBits.TERNARY:
        layer = BitLinear(src_dim, tgt_dim)
    elif weight_bits in (WeightBits.INT4, WeightBits.INT8):
        layer = QuantizedLinear(src_dim, tgt_dim, bits=weight_bits.value)
    else:
        layer = nn.Linear(src_dim, tgt_dim)
    layer.weight = mx.random.normal(layer.weight.shape) * (2.0 / layer.weight.shape[1]) ** 0.5
    return layer


def _fake_quantize_activation(x: mx.array, bits: int) -> mx.array:
    """Simulate reduced-precision activations via quantize-dequantize round-trip."""
    if bits >= 16:
        return x
    qmax = (1 << (bits - 1)) - 1
    scale = mx.stop_gradient(mx.max(mx.abs(x), axis=-1, keepdims=True) / qmax + 1e-8)
    x_q = mx.clip(mx.round(x / scale), -qmax, qmax)
    return mx.stop_gradient(x_q * scale - x) + x


def _apply_sparsity(x: mx.array, ratio: float) -> mx.array:
    """Zero out the smallest `ratio` fraction of elements by magnitude (STE)."""
    if ratio <= 0.0:
        return x
    flat = mx.abs(x).reshape(-1)
    k = int(flat.size * ratio)
    if k == 0:
        return x
    threshold = mx.sort(flat)[k]
    mask = (mx.abs(x) >= threshold).astype(x.dtype)
    return x * mx.stop_gradient(mask)


class EvolvedModel(nn.Module):
    """Neural network compiled from an evolved genome."""

    def __init__(
        self,
        genome: Genome,
        input_dim: int,
        num_classes: int,
        task: str = "classification",
        image_shape: tuple[int, int, int] | None = None,
        layer_norm: bool = True,
    ):
        super().__init__()
        self.task = task
        self.input_dim = input_dim
        self.num_classes = num_classes
        self.image_shape = image_shape

        layers = genome.enabled_layers
        connections = genome.enabled_connections
        conv_layers_raw = [c for c in genome.conv_layers if c.enabled] if genome.conv_layers else []
        gate_config = genome.gate_config

        # Sort layers by topological order
        layer_order = sorted(layers, key=lambda g: g.order)

        # ---- Conv stem ----
        self.conv_layers_list: list[nn.Conv2d] = []
        self.conv_activations: list[nn.Module] = []
        self._conv_output_dim: int | None = None
        if conv_layers_raw and image_shape:
            sorted_conv = sorted(conv_layers_raw, key=lambda g: g.order)
            in_ch = image_shape[0]
            h, w = image_shape[1], image_shape[2]
            for cg in sorted_conv:
                pad = cg.kernel_size // 2
                self.conv_layers_list.append(
                    nn.Conv2d(in_ch, cg.channels, cg.kernel_size, stride=cg.stride, padding=pad)
                )
                self.conv_activations.append(_ACTIVATION_MAP[cg.activation]())
                in_ch = cg.channels
                h = (h + 2 * pad - cg.kernel_size) // cg.stride + 1
                w = (w + 2 * pad - cg.kernel_size) // cg.stride + 1
            self._conv_output_dim = in_ch * h * w

        # ---- Dimension map ----
        effective_input = self._conv_output_dim or input_dim
        self._dims: dict[int, int] = {INPUT_INNOVATION: effective_input}
        for lg in layer_order:
            self._dims[lg.innovation] = lg.width

        # Filter connections to known layers
        valid_ids = set(self._dims.keys()) | {OUTPUT_INNOVATION}
        connections = [c for c in connections if c.source in valid_ids and c.target in valid_ids]

        # ---- Reachability (forward walk from INPUT) ----
        reachable = {INPUT_INNOVATION}
        changed = True
        while changed:
            changed = False
            for c in connections:
                if c.source in reachable and c.target not in reachable:
                    reachable.add(c.target)
                    changed = True
        connections = [c for c in connections if c.source in reachable and c.target in reachable]
        layer_order = [lg for lg in layer_order if lg.innovation in reachable]

        layer_map = {lg.innovation: lg for lg in layer_order}

        # ---- Per-connection projections ----
        self.projections: dict[str, nn.Module] = {}
        for conn in connections:
            src_dim = self._dims.get(conn.source, effective_input)
            if conn.target == OUTPUT_INNOVATION:
                tgt_dim, wb = num_classes, WeightBits.FP16
            else:
                tgt_dim = self._dims[conn.target]
                wb = layer_map[conn.target].weight_bits
            self.projections[f"proj_{conn.innovation}"] = _make_projection(src_dim, tgt_dim, wb)

        # ---- Activation functions ----
        self.activations: dict[int, nn.Module] = {
            lg.innovation: _ACTIVATION_MAP[lg.activation]() for lg in layer_order
        }

        # ---- Operator-specific modules ----
        self._operator_types: dict[int, str] = {}
        self._num_heads: dict[int, int] = {}
        self._sparsity: dict[int, float] = {}
        self.q_projs: dict[str, nn.Module] = {}
        self.k_projs: dict[str, nn.Module] = {}
        self.v_projs: dict[str, nn.Module] = {}
        self.o_projs: dict[str, nn.Module] = {}
        self.spatial_projs: dict[str, nn.Module] = {}
        self.residual_skip_projs: dict[str, nn.Module] = {}

        for lg in layer_order:
            inn = lg.innovation
            dim = lg.width
            self._sparsity[inn] = lg.sparsity
            op = lg.operator

            # Spatial needs reasonable feature count
            if op == OperatorType.SPATIAL and dim < 8:
                op = OperatorType.DENSE
            self._operator_types[inn] = op.value

            # Ensure num_heads divides width for attention
            nh = max(1, lg.num_heads)
            if op in (OperatorType.ATTENTION_LITE, OperatorType.TRANSFORMER_LITE):
                while nh > 1 and dim % nh != 0:
                    nh -= 1
            self._num_heads[inn] = nh

            if op in (OperatorType.ATTENTION_LITE, OperatorType.TRANSFORMER_LITE):
                key = str(inn)
                self.q_projs[key] = nn.Linear(dim, dim)
                self.k_projs[key] = nn.Linear(dim, dim)
                self.v_projs[key] = nn.Linear(dim, dim)
                self.o_projs[key] = nn.Linear(dim, dim)
            elif op == OperatorType.SPATIAL and dim >= 8:
                self.spatial_projs[str(inn)] = nn.Linear(dim, dim)
            elif op == OperatorType.RESIDUAL:
                self.residual_skip_projs[str(inn)] = nn.Linear(dim, dim)

        # ---- LayerNorm ----
        self.layer_norms: dict[str, nn.LayerNorm] = {}
        if layer_norm:
            for lg in layer_order:
                self.layer_norms[str(lg.innovation)] = nn.LayerNorm(lg.width)

        # ---- MoE ----
        self.moe: MixtureOfExperts | None = None
        self._load_balance_weight = 0.0
        if gate_config and gate_config.num_experts > 0:
            moe_input = layer_order[-1].width if layer_order else effective_input
            self.moe = MixtureOfExperts(
                input_dim=moe_input,
                output_dim=num_classes,
                num_experts=gate_config.num_experts,
                top_k=gate_config.top_k,
                gate_hidden_dim=gate_config.gate_hidden_dim,
                expert_hidden_dims=[max(16, num_classes * 2)],
            )
            self._load_balance_weight = gate_config.load_balance_weight

        # ---- Activation precision + Hadamard smoothing bits ----
        self._activation_bits: dict[int, int] = {}
        for lg in layer_order:
            self._activation_bits[lg.innovation] = lg.activation_bits.value

        # ---- Pre-compute routing tables (plain Python for mx.compile compat) ----
        self._layer_inns: list[int] = [lg.innovation for lg in layer_order]

        self._conn_routing: dict[int, list[tuple[str, int]]] = {}
        for inn in self._layer_inns:
            self._conn_routing[inn] = [
                (f"proj_{c.innovation}", c.source) for c in connections if c.target == inn
            ]

        self._output_routing: list[tuple[str, int]] = [
            (f"proj_{c.innovation}", c.source) for c in connections if c.target == OUTPUT_INNOVATION
        ]

        self._last_layer_inn: int | None = self._layer_inns[-1] if self._layer_inns else None
        self._last_layer_dim: int | None = (
            self._dims[self._last_layer_inn] if self._last_layer_inn else None
        )
        self.output_fallback: nn.Module | None = None
        if not self._output_routing and self._last_layer_dim is not None:
            self.output_fallback = nn.Linear(self._last_layer_dim, self.num_classes)

    def __call__(self, x: mx.array) -> mx.array:
        # Conv stem (NHWC)
        if self.conv_layers_list and self.image_shape:
            c, h, w = self.image_shape
            x = x.reshape(-1, h, w, c)
            for conv, act in zip(self.conv_layers_list, self.conv_activations):
                x = act(conv(x))
            x = x.reshape(x.shape[0], -1)

        outputs: dict[int, mx.array] = {INPUT_INNOVATION: x}

        for inn in self._layer_inns:
            # Gather incoming projections
            incoming = [
                self.projections[pk](outputs[src])
                for pk, src in self._conn_routing[inn]
                if src in outputs
            ]
            if not incoming:
                continue

            summed = incoming[0]
            for t in incoming[1:]:
                summed = summed + t

            # Operator dispatch
            op = self._operator_types.get(inn, "dense")
            if op == "sparse_dense":
                activated = self.activations[inn](summed)
                sp = self._sparsity.get(inn, 0.0)
                if sp > 0.0:
                    k = max(1, int(activated.shape[-1] * (1.0 - sp)))
                    if k < activated.shape[-1]:
                        threshold = mx.sort(mx.abs(activated), axis=-1)[..., -k:][..., 0:1]
                        mask = (mx.abs(activated) >= threshold).astype(activated.dtype)
                        activated = activated * mx.stop_gradient(mask)

            elif op == "residual":
                activated = self.activations[inn](summed)
                key = str(inn)
                if key in self.residual_skip_projs:
                    activated = activated + self.residual_skip_projs[key](summed)
                else:
                    activated = activated + summed

            elif op in ("attention_lite", "transformer_lite"):
                key = str(inn)
                q = self.q_projs[key](summed)
                k_ = self.k_projs[key](summed)
                v = self.v_projs[key](summed)
                num_heads = self._num_heads.get(inn, 1)
                head_dim = max(1, summed.shape[-1] // num_heads)

                if summed.ndim == 3:
                    batch_size, seq_len, dim = q.shape
                    q = q.reshape(batch_size, seq_len, num_heads, head_dim).transpose(0, 2, 1, 3)
                    k_ = k_.reshape(batch_size, seq_len, num_heads, head_dim).transpose(0, 2, 1, 3)
                    v = v.reshape(batch_size, seq_len, num_heads, head_dim).transpose(0, 2, 1, 3)

                    scale = 1.0 / math.sqrt(head_dim)
                    logits = mx.matmul(q, k_.transpose(0, 1, 3, 2)) * scale
                    mask = mx.tril(mx.ones((seq_len, seq_len), dtype=logits.dtype))
                    masked_logits = mx.where(mask[None, None, :, :] > 0, logits, -1e9)
                    weights = mx.softmax(masked_logits, axis=-1)
                    attended = mx.matmul(weights, v)
                    attended = attended.transpose(0, 2, 1, 3).reshape(batch_size, seq_len, dim)
                    activated = self.activations[inn](self.o_projs[key](attended))
                else:
                    scale = 1.0 / math.sqrt(head_dim)
                    weights = mx.softmax((q * k_) * scale, axis=-1)
                    activated = self.activations[inn](self.o_projs[key](weights * v))

            elif op == "spatial":
                kernel = 2
                local = summed
                for shift in range(1, kernel + 1):
                    local = local + mx.roll(summed, shift=shift, axis=-1) + mx.roll(summed, shift=-shift, axis=-1)
                local = local / float(kernel * 2 + 1)
                activated = self.activations[inn](self.spatial_projs[str(inn)](local))

            else:  # dense
                activated = self.activations[inn](summed)

            # LayerNorm
            ln_key = str(inn)
            if ln_key in self.layer_norms:
                activated = self.layer_norms[ln_key](activated)

            # Activation quantization + Hadamard smoothing
            bits = self._activation_bits.get(inn, 16)
            if bits < 16:
                activated = hadamard_smooth(activated, bits)
            else:
                activated = _fake_quantize_activation(activated, bits)

            outputs[inn] = activated

        # ---- MoE routing ----
        if self.moe is not None and self._last_layer_inn in outputs:
            logits = self.moe(outputs[self._last_layer_inn])
            return logits if self.task == "regression" else mx.softmax(logits, axis=-1)

        # ---- Standard output ----
        out_incoming = [
            self.projections[pk](outputs[src])
            for pk, src in self._output_routing
            if src in outputs
        ]
        if not out_incoming and self._last_layer_inn in outputs and self.output_fallback is not None:
            out_incoming.append(self.output_fallback(outputs[self._last_layer_inn]))

        logits = out_incoming[0]
        for t in out_incoming[1:]:
            logits = logits + t

        return logits if self.task == "regression" else mx.softmax(logits, axis=-1)


class CausalLanguageModel(nn.Module):
    """Wrap an evolved DAG with token/position embeddings and an LM head."""

    def __init__(
        self,
        genome: Genome,
        vocab_size: int,
        context_length: int,
        *,
        embed_dim: int = 128,
        layer_norm: bool = True,
    ) -> None:
        super().__init__()
        self.task = "language_modeling"
        self.vocab_size = vocab_size
        self.context_length = context_length
        self.embed_dim = embed_dim

        self.token_embed = nn.Embedding(vocab_size, embed_dim)
        self.pos_embed = nn.Embedding(context_length, embed_dim)
        self.inner = EvolvedModel(
            genome,
            input_dim=embed_dim,
            num_classes=embed_dim,
            task="regression",
            layer_norm=layer_norm,
        )
        self.output_norm = nn.LayerNorm(embed_dim)
        self.lm_head = nn.Linear(embed_dim, vocab_size)

    def __call__(self, x: mx.array) -> mx.array:
        x = x.astype(mx.int32)
        _, seq_len = x.shape
        if seq_len > self.context_length:
            raise ValueError(
                f"Sequence length {seq_len} exceeds context length {self.context_length}"
            )

        positions = mx.arange(seq_len, dtype=mx.int32)
        h = self.token_embed(x) + self.pos_embed(positions)
        h = self.inner(h)
        h = self.output_norm(h)
        logits = self.lm_head(h)
        return mx.softmax(logits, axis=-1)


def compile_genome(
    genome: Genome,
    input_dim: int,
    num_classes: int,
    task: str = "classification",
    image_shape: tuple[int, int, int] | None = None,
    layer_norm: bool = True,
) -> EvolvedModel:
    """Factory: compile a Genome into a trainable EvolvedModel."""
    if not MLX_AVAILABLE:
        raise RuntimeError("MLX compiler backend is unavailable; use runtime.backend='numpy-fallback'.")
    if task == "language_modeling":
        return CausalLanguageModel(
            genome,
            vocab_size=num_classes,
            context_length=input_dim,
            embed_dim=min(max(32, input_dim), 128),
            layer_norm=layer_norm,
        )
    return EvolvedModel(
        genome, input_dim, num_classes, task,
        image_shape=image_shape, layer_norm=layer_norm,
    )


def estimate_model_bytes(genome: Genome) -> int:
    """Quick precision-aware size estimate from genome structure alone.

    Does not require input_dim/num_classes -- uses only layer widths and
    connection topology. For the full estimate including I/O dimensions,
    use effective_model_bytes in train.py.
    """
    layer_map = {lg.innovation: lg for lg in genome.enabled_layers}
    total = 0
    for conn in genome.enabled_connections:
        src_lg = layer_map.get(conn.source)
        tgt_lg = layer_map.get(conn.target)
        src_w = src_lg.width if src_lg else 0
        tgt_w = tgt_lg.width if tgt_lg else 0
        if src_w == 0 or tgt_w == 0:
            continue
        bits = 1.58 if tgt_lg.weight_bits.value == 2 else float(tgt_lg.weight_bits.value)
        density = 1.0 - tgt_lg.sparsity
        total += int(src_w * tgt_w * bits * density / 8)
    return total
