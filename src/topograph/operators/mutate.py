"""Mutation operators for topology-based genome evolution.

Each function takes a genome (and optionally an innovation counter and RNG)
and returns a NEW genome with the mutation applied. Never mutates in place.
"""

from __future__ import annotations

import math
import random

from topograph.genome import (
    Activation,
    ActivationBits,
    ConnectionGene,
    Genome,
    InnovationCounter,
    LayerGene,
    OperatorType,
    WeightBits,
)
from topograph.genome.genome import INPUT_INNOVATION, OUTPUT_INNOVATION


def _copy_genome(genome: Genome) -> Genome:
    """Shallow-copy a genome with fresh gene lists."""
    g = Genome(layers=list(genome.layers), connections=list(genome.connections))
    g.conv_layers = list(genome.conv_layers)
    g.experts = list(genome.experts)
    g.expert_connections = list(genome.expert_connections)
    g.gate_config = genome.gate_config
    g.fitness = genome.fitness
    g.param_count = genome.param_count
    g.model_bytes = genome.model_bytes
    g.learning_rate = genome.learning_rate
    g.batch_size = genome.batch_size
    return g


def _pick_enabled_layer_idx(genome: Genome, rng: random.Random) -> int | None:
    """Return the index of a random enabled layer, or None."""
    indices = [i for i, g in enumerate(genome.layers) if g.enabled]
    return rng.choice(indices) if indices else None


# ---------------------------------------------------------------------------
# Width
# ---------------------------------------------------------------------------


def mutate_width(
    genome: Genome,
    rng: random.Random,
    min_width: int = 4,
    max_width: int = 512,
    max_delta: int = 32,
) -> Genome:
    g = _copy_genome(genome)
    idx = _pick_enabled_layer_idx(g, rng)
    if idx is None:
        return g
    gene = g.layers[idx]
    delta = rng.randint(-max_delta, max_delta)
    new_width = max(min_width, min(max_width, gene.width + delta))
    if new_width == gene.width:
        new_width = gene.width + (1 if gene.width < max_width else -1)
    g.layers[idx] = gene.model_copy(update={"width": new_width})
    return g


# ---------------------------------------------------------------------------
# Activation
# ---------------------------------------------------------------------------


def mutate_activation(genome: Genome, rng: random.Random) -> Genome:
    g = _copy_genome(genome)
    idx = _pick_enabled_layer_idx(g, rng)
    if idx is None:
        return g
    gene = g.layers[idx]
    others = [a for a in Activation if a != gene.activation]
    g.layers[idx] = gene.model_copy(update={"activation": rng.choice(others)})
    return g


# ---------------------------------------------------------------------------
# Add layer
# ---------------------------------------------------------------------------


def mutate_add_layer(
    genome: Genome,
    counter: InnovationCounter,
    rng: random.Random,
) -> Genome:
    g = _copy_genome(genome)
    enabled_conns = g.enabled_connections
    if not enabled_conns:
        return g

    conn = rng.choice(enabled_conns)
    conn_idx = g.connections.index(conn)

    # Determine order for new layer
    layer_orders: dict[int, float] = {lg.innovation: lg.order for lg in g.layers}
    layer_orders[INPUT_INNOVATION] = 0.0
    layer_orders[OUTPUT_INNOVATION] = float("inf")
    src_order = layer_orders.get(conn.source, 0.0)
    tgt_order = layer_orders.get(conn.target, float("inf"))
    if tgt_order == float("inf"):
        max_hidden = max((lg.order for lg in g.enabled_layers), default=0.0)
        new_order = max_hidden + 0.5
    else:
        new_order = (src_order + tgt_order) / 2.0

    new_layer = LayerGene(
        innovation=counter.next(),
        width=rng.randint(16, 256),
        activation=rng.choice(list(Activation)),
        order=new_order,
    )
    conn_in = ConnectionGene(
        innovation=counter.next(),
        source=conn.source,
        target=new_layer.innovation,
    )
    conn_out = ConnectionGene(
        innovation=counter.next(),
        source=new_layer.innovation,
        target=conn.target,
    )

    g.connections[conn_idx] = conn.model_copy(update={"enabled": False})
    g.layers.append(new_layer)
    g.connections.append(conn_in)
    g.connections.append(conn_out)
    return g


# ---------------------------------------------------------------------------
# Remove layer
# ---------------------------------------------------------------------------


def mutate_remove_layer(
    genome: Genome,
    counter: InnovationCounter,
    rng: random.Random,
) -> Genome:
    g = _copy_genome(genome)
    enabled = g.enabled_layers
    if not enabled:
        return g

    layer = rng.choice(enabled)
    layer_idx = g.layers.index(layer)
    inn = layer.innovation

    incoming_sources: list[int] = []
    outgoing_targets: list[int] = []
    for i, c in enumerate(g.connections):
        if not c.enabled:
            continue
        if c.target == inn:
            incoming_sources.append(c.source)
            g.connections[i] = c.model_copy(update={"enabled": False})
        elif c.source == inn:
            outgoing_targets.append(c.target)
            g.connections[i] = c.model_copy(update={"enabled": False})

    g.layers[layer_idx] = layer.model_copy(update={"enabled": False})

    existing = {(c.source, c.target) for c in g.enabled_connections}
    for src in incoming_sources:
        for tgt in outgoing_targets:
            if (src, tgt) not in existing:
                g.connections.append(
                    ConnectionGene(
                        innovation=counter.next(),
                        source=src,
                        target=tgt,
                    )
                )
    return g


# ---------------------------------------------------------------------------
# Add connection
# ---------------------------------------------------------------------------


def mutate_add_connection(
    genome: Genome,
    counter: InnovationCounter,
    rng: random.Random,
) -> Genome:
    g = _copy_genome(genome)

    layer_orders: dict[int, float] = {
        lg.innovation: lg.order for lg in g.layers if lg.enabled
    }
    layer_orders[INPUT_INNOVATION] = 0.0
    layer_orders[OUTPUT_INNOVATION] = float("inf")

    existing = {(c.source, c.target) for c in g.enabled_connections}
    all_ids = list(layer_orders.keys())
    candidates = [
        (s, t)
        for s in all_ids
        for t in all_ids
        if s != t and layer_orders[s] < layer_orders[t] and (s, t) not in existing
    ]
    if not candidates:
        return g

    src, tgt = rng.choice(candidates)
    g.connections.append(
        ConnectionGene(innovation=counter.next(), source=src, target=tgt)
    )
    return g


# ---------------------------------------------------------------------------
# Remove connection
# ---------------------------------------------------------------------------


def mutate_remove_connection(genome: Genome, rng: random.Random) -> Genome:
    g = _copy_genome(genome)
    enabled = g.enabled_connections
    if len(enabled) <= 1:
        return g

    removable: list[ConnectionGene] = []
    for conn in enabled:
        remaining = [c for c in enabled if c is not conn]
        connected: set[int] = set()
        for c in remaining:
            connected.add(c.source)
            connected.add(c.target)
        if not any(lg.innovation not in connected for lg in g.enabled_layers):
            removable.append(conn)

    if not removable:
        return g

    conn = rng.choice(removable)
    idx = g.connections.index(conn)
    g.connections[idx] = conn.model_copy(update={"enabled": False})
    return g


# ---------------------------------------------------------------------------
# Add residual (skip connection)
# ---------------------------------------------------------------------------


def mutate_add_residual(
    genome: Genome,
    counter: InnovationCounter,
    rng: random.Random,
) -> Genome:
    g = _copy_genome(genome)

    layer_orders: dict[int, float] = {
        lg.innovation: lg.order for lg in g.layers if lg.enabled
    }
    layer_orders[INPUT_INNOVATION] = 0.0
    layer_orders[OUTPUT_INNOVATION] = float("inf")

    existing = {(c.source, c.target) for c in g.enabled_connections}
    sorted_ids = sorted(layer_orders, key=lambda k: layer_orders[k])
    candidates = [
        (sorted_ids[i], sorted_ids[j])
        for i in range(len(sorted_ids))
        for j in range(i + 2, len(sorted_ids))
        if (sorted_ids[i], sorted_ids[j]) not in existing
    ]
    if not candidates:
        return g

    src, tgt = rng.choice(candidates)
    g.connections.append(
        ConnectionGene(innovation=counter.next(), source=src, target=tgt)
    )
    return g


# ---------------------------------------------------------------------------
# Weight bits
# ---------------------------------------------------------------------------

_WEIGHT_PRECISION_ORDER = [
    WeightBits.TERNARY,
    WeightBits.INT4,
    WeightBits.INT8,
    WeightBits.FP16,
]


def mutate_weight_bits(
    genome: Genome,
    rng: random.Random,
    allowed_bits: list[WeightBits] | None = None,
) -> Genome:
    g = _copy_genome(genome)
    idx = _pick_enabled_layer_idx(g, rng)
    if idx is None:
        return g

    gene = g.layers[idx]
    order = [p for p in _WEIGHT_PRECISION_ORDER if p in allowed_bits] if allowed_bits else list(_WEIGHT_PRECISION_ORDER)
    if len(order) <= 1:
        return g

    if gene.weight_bits not in order:
        new_bits = rng.choice(order)
    else:
        cur = order.index(gene.weight_bits)
        if rng.random() < 0.7:
            new_idx = max(0, min(len(order) - 1, cur + rng.choice([-1, 1])))
        else:
            new_idx = rng.choice([i for i in range(len(order)) if i != cur])
        new_bits = order[new_idx]

    g.layers[idx] = gene.model_copy(update={"weight_bits": new_bits})
    return g


# ---------------------------------------------------------------------------
# Activation bits
# ---------------------------------------------------------------------------


def mutate_activation_bits(
    genome: Genome,
    rng: random.Random,
    allowed_bits: list[ActivationBits] | None = None,
) -> Genome:
    g = _copy_genome(genome)
    idx = _pick_enabled_layer_idx(g, rng)
    if idx is None:
        return g

    gene = g.layers[idx]
    pool = [b for b in (allowed_bits or list(ActivationBits)) if b != gene.activation_bits]
    if not pool:
        return g

    g.layers[idx] = gene.model_copy(update={"activation_bits": rng.choice(pool)})
    return g


# ---------------------------------------------------------------------------
# Sparsity
# ---------------------------------------------------------------------------


def mutate_sparsity(genome: Genome, rng: random.Random) -> Genome:
    g = _copy_genome(genome)
    idx = _pick_enabled_layer_idx(g, rng)
    if idx is None:
        return g

    gene = g.layers[idx]
    new_sparsity = max(0.0, min(0.75, gene.sparsity + rng.gauss(0, 0.1)))
    g.layers[idx] = gene.model_copy(update={"sparsity": round(new_sparsity, 2)})
    return g


# ---------------------------------------------------------------------------
# Operator type
# ---------------------------------------------------------------------------


def mutate_operator_type(genome: Genome, rng: random.Random) -> Genome:
    g = _copy_genome(genome)
    idx = _pick_enabled_layer_idx(g, rng)
    if idx is None:
        return g

    gene = g.layers[idx]
    others = [op for op in OperatorType if op != gene.operator]
    new_op = rng.choice(others)
    update: dict = {"operator": new_op}
    if new_op in (OperatorType.ATTENTION_LITE, OperatorType.TRANSFORMER_LITE):
        nh = max(1, gene.num_heads)
        while nh > 1 and gene.width % nh != 0:
            nh -= 1
        update["num_heads"] = nh
    g.layers[idx] = gene.model_copy(update=update)
    return g


# ---------------------------------------------------------------------------
# Learning rate
# ---------------------------------------------------------------------------


def mutate_learning_rate(genome: Genome, rng: random.Random) -> Genome:
    g = _copy_genome(genome)
    if g.learning_rate is None:
        return g
    log_lr = math.log10(g.learning_rate) + rng.gauss(0, 0.2)
    g.learning_rate = max(1e-5, min(0.1, 10**log_lr))
    return g


# ---------------------------------------------------------------------------
# Batch size
# ---------------------------------------------------------------------------


def mutate_batch_size(genome: Genome, rng: random.Random) -> Genome:
    g = _copy_genome(genome)
    g.batch_size = rng.choice([16, 32, 64, 128])
    return g
