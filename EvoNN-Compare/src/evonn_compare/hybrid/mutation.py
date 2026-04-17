"""Mutation operators for the hybrid genome."""
from __future__ import annotations

import random

from evonn_compare.hybrid.genome import (
    HybridConnectionGene,
    HybridFamily,
    HybridGenome,
    HybridNodeGene,
    INPUT_INNOVATION,
    OUTPUT_INNOVATION,
)


def mutate_add_node(genome: HybridGenome, rng: random.Random, counter: int) -> int:
    """Add a new node by splitting an existing connection."""
    enabled = genome.enabled_connections
    if not enabled:
        return counter
    conn = rng.choice(enabled)
    idx = genome.connections.index(conn)

    # Compute order
    node_orders = {n.innovation_number: n.order for n in genome.nodes}
    node_orders[INPUT_INNOVATION] = 0.0
    node_orders[OUTPUT_INNOVATION] = float("inf")
    src_order = node_orders.get(conn.source_innovation, 0.0)
    tgt_order = node_orders.get(conn.target_innovation, float("inf"))
    new_order = (src_order + tgt_order) / 2.0

    counter += 1
    new_node = HybridNodeGene(
        innovation_number=counter,
        order=new_order,
        family=rng.choice(list(HybridFamily)),
        width=rng.choice([32, 64, 128, 256]),
        activation=rng.choice(["relu", "gelu", "silu"]),
    )
    counter += 1
    conn_in = HybridConnectionGene(
        innovation_number=counter,
        source_innovation=conn.source_innovation,
        target_innovation=new_node.innovation_number,
    )
    counter += 1
    conn_out = HybridConnectionGene(
        innovation_number=counter,
        source_innovation=new_node.innovation_number,
        target_innovation=conn.target_innovation,
    )

    genome.connections[idx] = conn.model_copy(update={"enabled": False})
    genome.nodes.append(new_node)
    genome.connections.append(conn_in)
    genome.connections.append(conn_out)
    return counter


def mutate_add_connection(genome: HybridGenome, rng: random.Random, counter: int) -> int:
    """Add a new connection between non-connected nodes."""
    node_orders = {n.innovation_number: n.order for n in genome.enabled_nodes}
    node_orders[INPUT_INNOVATION] = 0.0
    node_orders[OUTPUT_INNOVATION] = float("inf")
    existing = {(c.source_innovation, c.target_innovation) for c in genome.enabled_connections}
    candidates = []
    all_ids = list(node_orders.keys())
    for src in all_ids:
        for tgt in all_ids:
            if src == tgt or node_orders[src] >= node_orders[tgt] or (src, tgt) in existing:
                continue
            candidates.append((src, tgt))
    if not candidates:
        return counter
    src, tgt = rng.choice(candidates)
    counter += 1
    genome.connections.append(HybridConnectionGene(
        innovation_number=counter,
        source_innovation=src,
        target_innovation=tgt,
    ))
    return counter


def mutate_node_family(genome: HybridGenome, rng: random.Random) -> None:
    """Change the family type of a random node."""
    enabled = genome.enabled_nodes
    if not enabled:
        return
    idx = rng.randrange(len(genome.nodes))
    while not genome.nodes[idx].enabled:
        idx = rng.randrange(len(genome.nodes))
    node = genome.nodes[idx]
    others = [f for f in HybridFamily if f != node.family]
    genome.nodes[idx] = node.model_copy(update={"family": rng.choice(others)})


def mutate_node_width(genome: HybridGenome, rng: random.Random) -> None:
    """Change the width of a random node."""
    enabled = genome.enabled_nodes
    if not enabled:
        return
    idx = rng.randrange(len(genome.nodes))
    while not genome.nodes[idx].enabled:
        idx = rng.randrange(len(genome.nodes))
    node = genome.nodes[idx]
    delta = rng.randint(-32, 32)
    new_width = max(8, min(512, node.width + delta))
    genome.nodes[idx] = node.model_copy(update={"width": new_width})


def mutate_node_activation(genome: HybridGenome, rng: random.Random) -> None:
    """Change the activation of a random node."""
    enabled = genome.enabled_nodes
    if not enabled:
        return
    idx = rng.randrange(len(genome.nodes))
    while not genome.nodes[idx].enabled:
        idx = rng.randrange(len(genome.nodes))
    node = genome.nodes[idx]
    others = [a for a in ["relu", "gelu", "silu", "tanh"] if a != node.activation]
    genome.nodes[idx] = node.model_copy(update={"activation": rng.choice(others)})


def mutate(genome: HybridGenome, rng: random.Random, counter: int) -> int:
    """Apply one random mutation to the genome."""
    roll = rng.random()
    if roll < 0.15:
        counter = mutate_add_node(genome, rng, counter)
    elif roll < 0.25:
        counter = mutate_add_connection(genome, rng, counter)
    elif roll < 0.45:
        mutate_node_family(genome, rng)
    elif roll < 0.65:
        mutate_node_width(genome, rng)
    elif roll < 0.80:
        mutate_node_activation(genome, rng)
    # else: no mutation (15% chance)
    return counter
