"""Hierarchy-aware mutation and crossover operators."""

from __future__ import annotations

import random

from stratograph.genome.models import (
    ActivationKind,
    CellEdgeGene,
    CellGene,
    CellNodeGene,
    HierarchicalGenome,
    MacroEdgeGene,
    MacroNodeGene,
    PrimitiveKind,
)

MOTIF_LIBRARY: tuple[tuple[tuple[PrimitiveKind, ActivationKind], ...], ...] = (
    (
        (PrimitiveKind.LINEAR, ActivationKind.IDENTITY),
        (PrimitiveKind.MIX, ActivationKind.IDENTITY),
    ),
    (
        (PrimitiveKind.GATE, ActivationKind.IDENTITY),
        (PrimitiveKind.NORM, ActivationKind.IDENTITY),
        (PrimitiveKind.MIX, ActivationKind.RELU),
    ),
    (
        (PrimitiveKind.NORM, ActivationKind.GELU),
        (PrimitiveKind.RESIDUAL, ActivationKind.GELU),
        (PrimitiveKind.LINEAR, ActivationKind.TANH),
        (PrimitiveKind.MIX, ActivationKind.TANH),
    ),
    (
        (PrimitiveKind.GATE, ActivationKind.RELU),
        (PrimitiveKind.RESIDUAL, ActivationKind.TANH),
        (PrimitiveKind.GATE, ActivationKind.IDENTITY),
    ),
)
TASK_MOTIFS: dict[str, tuple[tuple[tuple[PrimitiveKind, ActivationKind], ...], ...]] = {
    "classification": MOTIF_LIBRARY
    + (
        (
            (PrimitiveKind.NORM, ActivationKind.RELU),
            (PrimitiveKind.MIX, ActivationKind.GELU),
            (PrimitiveKind.LINEAR, ActivationKind.IDENTITY),
        ),
    ),
    "language_modeling": (
        (
            (PrimitiveKind.GATE, ActivationKind.IDENTITY),
            (PrimitiveKind.NORM, ActivationKind.IDENTITY),
            (PrimitiveKind.MIX, ActivationKind.RELU),
        ),
        (
            (PrimitiveKind.NORM, ActivationKind.GELU),
            (PrimitiveKind.RESIDUAL, ActivationKind.GELU),
            (PrimitiveKind.LINEAR, ActivationKind.TANH),
            (PrimitiveKind.MIX, ActivationKind.TANH),
        ),
        (
            (PrimitiveKind.LINEAR, ActivationKind.IDENTITY),
            (PrimitiveKind.GATE, ActivationKind.RELU),
            (PrimitiveKind.GATE, ActivationKind.TANH),
        ),
    ),
}


def mutate_genome(
    genome: HierarchicalGenome,
    *,
    rng: random.Random,
    candidate_id: str,
    allow_clone_mutation: bool = True,
    motif_bias: bool = True,
) -> HierarchicalGenome:
    """Return mutated copy of a hierarchical genome."""
    macro_nodes = [node.model_copy() for node in genome.macro_nodes]
    macro_edges = [edge.model_copy() for edge in genome.macro_edges]
    cell_library = {cell_id: cell.model_copy(deep=True) for cell_id, cell in genome.cell_library.items()}
    modes = ["width", "activation", "add_macro", "specialize_cell", "add_skip_edge", "rewire_macro"]
    if allow_clone_mutation:
        modes.append("clone_cell")
    if motif_bias and genome.task != "regression":
        modes.append("motif_rewrite")
    mode = rng.choice(modes)

    if mode == "width":
        target = rng.choice(macro_nodes)
        new_width = _clamp_width(target.output_width + rng.choice([-16, -8, 8, 16]))
        target = target.model_copy(update={"input_width": new_width, "output_width": new_width})
        macro_nodes = [target if node.node_id == target.node_id else node for node in macro_nodes]
        cell = cell_library[target.cell_id]
        cell_library[target.cell_id] = _resize_cell(cell, new_width)

    elif mode == "activation":
        cell_id = rng.choice(list(cell_library))
        cell = cell_library[cell_id]
        target_node = rng.choice(cell.nodes)
        new_activation = (
            rng.choice([activation for _, activation in _task_motif(genome.task, rng)])
            if motif_bias
            else rng.choice(list(ActivationKind))
        )
        replaced = [
            node.model_copy(update={"activation": new_activation})
            if node.node_id == target_node.node_id
            else node
            for node in cell.nodes
        ]
        cell_library[cell_id] = cell.model_copy(update={"nodes": replaced})

    elif mode == "clone_cell":
        target = rng.choice(macro_nodes)
        old_cell = cell_library[target.cell_id]
        new_cell_id = f"{target.cell_id}_clone_{rng.randint(0, 9999)}"
        new_nodes = old_cell.nodes[:]
        if motif_bias:
            new_nodes = _apply_motif(old_cell.nodes, _task_motif(genome.task, rng))
        elif new_nodes:
            chosen = rng.randrange(len(new_nodes))
            new_nodes[chosen] = new_nodes[chosen].model_copy(
                update={"activation": rng.choice(list(ActivationKind))}
            )
        new_cell = old_cell.model_copy(
            update={"cell_id": new_cell_id, "shared": False, "nodes": new_nodes},
            deep=True,
        )
        cell_library[new_cell_id] = new_cell
        macro_nodes = [
            node.model_copy(update={"cell_id": new_cell_id}) if node.node_id == target.node_id else node
            for node in macro_nodes
        ]

    elif mode == "add_macro":
        anchor = rng.choice(macro_nodes)
        new_cell_id = rng.choice(list(cell_library))
        new_node = MacroNodeGene(
            node_id=f"macro_{len(macro_nodes)}",
            cell_id=new_cell_id,
            input_width=anchor.output_width,
            output_width=anchor.output_width,
            role="body",
        )
        macro_nodes.append(new_node)
        macro_edges.append(MacroEdgeGene(source=anchor.node_id, target=new_node.node_id))
        if len(macro_nodes) > 2:
            parent_choices = [node for node in macro_nodes[:-1] if node.node_id != anchor.node_id]
            if parent_choices and rng.random() < 0.65:
                extra_parent = rng.choice(parent_choices)
                if not _edge_exists(macro_edges, extra_parent.node_id, new_node.node_id):
                    macro_edges.append(MacroEdgeGene(source=extra_parent.node_id, target=new_node.node_id))
        if rng.random() < 0.55:
            macro_edges.append(MacroEdgeGene(source=new_node.node_id, target="output"))
        macro_edges = _normalize_macro_edges(macro_nodes, macro_edges)

    elif mode == "specialize_cell":
        shared_nodes = [node for node in macro_nodes if cell_library[node.cell_id].shared]
        if shared_nodes:
            target = rng.choice(shared_nodes)
            old_cell = cell_library[target.cell_id]
            new_cell_id = f"{target.cell_id}_spec_{rng.randint(0, 9999)}"
            nodes = old_cell.nodes[:]
            if motif_bias:
                motif = _task_motif(genome.task, rng)
                nodes = _apply_motif(old_cell.nodes, motif)
            else:
                new_kind = rng.choice(list(PrimitiveKind))
                nodes[0] = nodes[0].model_copy(update={"kind": new_kind})
            cell_library[new_cell_id] = old_cell.model_copy(
                update={"cell_id": new_cell_id, "shared": False, "nodes": nodes},
                deep=True,
            )
            macro_nodes = [
                node.model_copy(update={"cell_id": new_cell_id}) if node.node_id == target.node_id else node
                for node in macro_nodes
            ]

    elif mode == "motif_rewrite":
        cell_id = rng.choice(list(cell_library))
        cell = cell_library[cell_id]
        motif = _task_motif(genome.task, rng)
        cell_library[cell_id] = cell.model_copy(update={"nodes": _apply_motif(cell.nodes, motif)})

    elif mode == "add_skip_edge":
        edge = _sample_valid_macro_edge(macro_nodes, macro_edges, rng)
        if edge is not None:
            macro_edges.append(edge)
        macro_edges = _normalize_macro_edges(macro_nodes, macro_edges)

    elif mode == "rewire_macro":
        removable = [
            edge
            for edge in macro_edges
            if edge.target != "output" and edge.source != "input"
        ]
        if removable:
            drop = rng.choice(removable)
            macro_edges = [edge for edge in macro_edges if edge != drop]
        replacement = _sample_valid_macro_edge(macro_nodes, macro_edges, rng)
        if replacement is not None:
            macro_edges.append(replacement)
        if rng.random() < 0.35 and len(macro_nodes) > 2:
            branch = rng.choice(macro_nodes[:-1])
            macro_edges.append(MacroEdgeGene(source=branch.node_id, target="output"))
        macro_edges = _normalize_macro_edges(macro_nodes, macro_edges)

    return HierarchicalGenome(
        genome_id=candidate_id,
        task=genome.task,
        input_dim=genome.input_dim,
        output_dim=genome.output_dim,
        macro_nodes=macro_nodes,
        macro_edges=macro_edges,
        cell_library=cell_library,
    )


def crossover_genomes(
    left: HierarchicalGenome,
    right: HierarchicalGenome,
    *,
    rng: random.Random,
    candidate_id: str,
    allow_clone_mutation: bool = True,
    motif_bias: bool = True,
) -> HierarchicalGenome:
    """Combine two parent genomes into one child."""
    take_left_prefix = max(1, min(len(left.macro_nodes), rng.randint(1, len(left.macro_nodes))))
    take_right_suffix = max(1, min(len(right.macro_nodes), rng.randint(1, len(right.macro_nodes))))
    left_part = left.macro_nodes[:take_left_prefix]
    right_part = right.macro_nodes[-take_right_suffix:]

    macro_nodes: list[MacroNodeGene] = []
    cell_library: dict[str, CellGene] = {}
    source_index: dict[tuple[str, str], str] = {}
    for index, (side, node) in enumerate([("left", node) for node in left_part] + [("right", node) for node in right_part]):
        source_library = left.cell_library if side == "left" else right.cell_library
        source_cell = source_library[node.cell_id]
        child_cell_id = f"cell_x_{index}_{source_cell.cell_id}"
        cell_library[child_cell_id] = source_cell.model_copy(update={"cell_id": child_cell_id}, deep=True)
        width = max(node.input_width, node.output_width)
        child_node_id = f"macro_{index}"
        source_index[(side, node.node_id)] = child_node_id
        macro_nodes.append(
            MacroNodeGene(
                node_id=child_node_id,
                cell_id=child_cell_id,
                input_width=width,
                output_width=width,
                role="stem" if index == 0 else "body",
            )
        )

    macro_edges = _inherit_macro_edges(
        left_edges=left.macro_edges,
        right_edges=right.macro_edges,
        source_index=source_index,
        macro_nodes=macro_nodes,
        rng=rng,
    )
    child = HierarchicalGenome(
        genome_id=candidate_id,
        task=left.task,
        input_dim=left.input_dim,
        output_dim=left.output_dim,
        macro_nodes=macro_nodes,
        macro_edges=macro_edges,
        cell_library=cell_library,
    )
    return mutate_genome(
        child,
        rng=rng,
        candidate_id=candidate_id,
        allow_clone_mutation=allow_clone_mutation,
        motif_bias=motif_bias,
    )


def _resize_cell(cell: CellGene, width: int) -> CellGene:
    nodes = [node.model_copy(update={"width": width}) for node in cell.nodes]
    return cell.model_copy(update={"input_width": width, "output_width": width, "nodes": nodes})


def _clamp_width(width: int) -> int:
    return max(8, min(256, width))


def _apply_motif(
    nodes: list[CellNodeGene],
    motif: tuple[tuple[PrimitiveKind, ActivationKind], ...],
) -> list[CellNodeGene]:
    updated: list[CellNodeGene] = []
    for index, node in enumerate(nodes):
        kind, activation = motif[index % len(motif)]
        updated.append(node.model_copy(update={"kind": kind, "activation": activation}))
    return updated


def _task_motif(
    task: str,
    rng: random.Random,
) -> tuple[tuple[PrimitiveKind, ActivationKind], ...]:
    motifs = TASK_MOTIFS.get(task, MOTIF_LIBRARY)
    return rng.choice(motifs)


def _sample_valid_macro_edge(
    macro_nodes: list[MacroNodeGene],
    macro_edges: list[MacroEdgeGene],
    rng: random.Random,
) -> MacroEdgeGene | None:
    if len(macro_nodes) < 2:
        return None
    id_order = {node.node_id: index for index, node in enumerate(macro_nodes)}
    candidates: list[MacroEdgeGene] = []
    for source in macro_nodes:
        for target in macro_nodes:
            if id_order[source.node_id] >= id_order[target.node_id]:
                continue
            if _edge_exists(macro_edges, source.node_id, target.node_id):
                continue
            candidates.append(MacroEdgeGene(source=source.node_id, target=target.node_id))
    if not candidates:
        return None
    return rng.choice(candidates)


def _inherit_macro_edges(
    *,
    left_edges: list[MacroEdgeGene],
    right_edges: list[MacroEdgeGene],
    source_index: dict[tuple[str, str], str],
    macro_nodes: list[MacroNodeGene],
    rng: random.Random,
) -> list[MacroEdgeGene]:
    inherited: list[MacroEdgeGene] = []
    for side, edges in (("left", left_edges), ("right", right_edges)):
        for edge in edges:
            mapped_source = "input" if edge.source == "input" else source_index.get((side, edge.source))
            mapped_target = "output" if edge.target == "output" else source_index.get((side, edge.target))
            if mapped_source is None or mapped_target is None:
                continue
            inherited.append(MacroEdgeGene(source=mapped_source, target=mapped_target, enabled=edge.enabled))
    return _normalize_macro_edges(macro_nodes, inherited, rng=rng)


def _normalize_macro_edges(
    macro_nodes: list[MacroNodeGene],
    macro_edges: list[MacroEdgeGene],
    *,
    rng: random.Random | None = None,
) -> list[MacroEdgeGene]:
    node_ids = [node.node_id for node in macro_nodes]
    order = {node_id: index for index, node_id in enumerate(node_ids)}
    seen: set[tuple[str, str]] = set()
    normalized: list[MacroEdgeGene] = []
    for edge in macro_edges:
        if not edge.enabled:
            continue
        if edge.source != "input" and edge.source not in order:
            continue
        if edge.target != "output" and edge.target not in order:
            continue
        if edge.source != "input" and edge.target != "output":
            if order[edge.source] >= order[edge.target]:
                continue
        key = (edge.source, edge.target)
        if key in seen:
            continue
        seen.add(key)
        normalized.append(MacroEdgeGene(source=edge.source, target=edge.target))

    if not normalized:
        normalized.append(MacroEdgeGene(source="input", target=node_ids[0]))
        normalized.append(MacroEdgeGene(source=node_ids[-1], target="output"))

    outgoing: dict[str, int] = {node_id: 0 for node_id in node_ids}
    incoming: dict[str, int] = {node_id: 0 for node_id in node_ids}
    for edge in normalized:
        if edge.source in outgoing:
            outgoing[edge.source] += 1
        if edge.target in incoming:
            incoming[edge.target] += 1

    if incoming[node_ids[0]] == 0:
        normalized.append(MacroEdgeGene(source="input", target=node_ids[0]))
        incoming[node_ids[0]] += 1

    for index in range(1, len(node_ids)):
        node_id = node_ids[index]
        if incoming[node_id] == 0:
            parent_id = node_ids[index - 1]
            normalized.append(MacroEdgeGene(source=parent_id, target=node_id))
            outgoing[parent_id] += 1
            incoming[node_id] += 1
        if rng is not None and index >= 2 and outgoing[node_ids[index - 2]] == 0 and rng.random() < 0.4:
            normalized.append(MacroEdgeGene(source=node_ids[index - 2], target=node_id))
            outgoing[node_ids[index - 2]] += 1
            incoming[node_id] += 1

    if not any(edge.target == "output" for edge in normalized):
        normalized.append(MacroEdgeGene(source=node_ids[-1], target="output"))
        outgoing[node_ids[-1]] += 1
    else:
        for node_id in node_ids:
            if outgoing[node_id] == 0 and not _edge_exists(normalized, node_id, "output"):
                normalized.append(MacroEdgeGene(source=node_id, target="output"))

    normalized.sort(key=lambda edge: (_edge_sort_key(edge, order), edge.source, edge.target))
    return normalized


def _edge_exists(edges: list[MacroEdgeGene], source: str, target: str) -> bool:
    return any(edge.enabled and edge.source == source and edge.target == target for edge in edges)


def _edge_sort_key(edge: MacroEdgeGene, order: dict[str, int]) -> tuple[int, int]:
    source_index = -1 if edge.source == "input" else order.get(edge.source, 10_000)
    target_index = 10_001 if edge.target == "output" else order.get(edge.target, 10_000)
    return (source_index, target_index)
