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


def mutate_genome(
    genome: HierarchicalGenome,
    *,
    rng: random.Random,
    candidate_id: str,
) -> HierarchicalGenome:
    """Return mutated copy of a hierarchical genome."""
    macro_nodes = [node.model_copy() for node in genome.macro_nodes]
    macro_edges = [edge.model_copy() for edge in genome.macro_edges]
    cell_library = {cell_id: cell.model_copy(deep=True) for cell_id, cell in genome.cell_library.items()}
    mode = rng.choice(["width", "activation", "clone_cell", "add_macro", "specialize_cell"])

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
        new_activation = rng.choice(list(ActivationKind))
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
        new_cell = old_cell.model_copy(update={"cell_id": new_cell_id, "shared": False}, deep=True)
        cell_library[new_cell_id] = new_cell
        macro_nodes = [
            node.model_copy(update={"cell_id": new_cell_id}) if node.node_id == target.node_id else node
            for node in macro_nodes
        ]

    elif mode == "add_macro":
        tail_node = macro_nodes[-1]
        new_cell_id = rng.choice(list(cell_library))
        new_node = MacroNodeGene(
            node_id=f"macro_{len(macro_nodes)}",
            cell_id=new_cell_id,
            input_width=tail_node.output_width,
            output_width=tail_node.output_width,
            role="body",
        )
        macro_nodes.append(new_node)
        macro_edges = [edge for edge in macro_edges if not (edge.source == tail_node.node_id and edge.target == "output")]
        macro_edges.extend(
            [
                MacroEdgeGene(source=tail_node.node_id, target=new_node.node_id),
                MacroEdgeGene(source=new_node.node_id, target="output"),
            ]
        )

    elif mode == "specialize_cell":
        shared_nodes = [node for node in macro_nodes if cell_library[node.cell_id].shared]
        if shared_nodes:
            target = rng.choice(shared_nodes)
            old_cell = cell_library[target.cell_id]
            new_cell_id = f"{target.cell_id}_spec_{rng.randint(0, 9999)}"
            new_kind = rng.choice(list(PrimitiveKind))
            nodes = old_cell.nodes[:]
            nodes[0] = nodes[0].model_copy(update={"kind": new_kind})
            cell_library[new_cell_id] = old_cell.model_copy(
                update={"cell_id": new_cell_id, "shared": False, "nodes": nodes},
                deep=True,
            )
            macro_nodes = [
                node.model_copy(update={"cell_id": new_cell_id}) if node.node_id == target.node_id else node
                for node in macro_nodes
            ]

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
) -> HierarchicalGenome:
    """Combine two parent genomes into one child."""
    take_left_prefix = max(1, min(len(left.macro_nodes), rng.randint(1, len(left.macro_nodes))))
    take_right_suffix = max(1, min(len(right.macro_nodes), rng.randint(1, len(right.macro_nodes))))
    left_part = left.macro_nodes[:take_left_prefix]
    right_part = right.macro_nodes[-take_right_suffix:]

    macro_nodes: list[MacroNodeGene] = []
    cell_library: dict[str, CellGene] = {}
    for index, node in enumerate(left_part + right_part):
        source_cell = (left.cell_library if node.cell_id in left.cell_library else right.cell_library)[node.cell_id]
        child_cell_id = f"cell_x_{index}_{source_cell.cell_id}"
        cell_library[child_cell_id] = source_cell.model_copy(update={"cell_id": child_cell_id}, deep=True)
        width = max(node.input_width, node.output_width)
        macro_nodes.append(
            MacroNodeGene(
                node_id=f"macro_{index}",
                cell_id=child_cell_id,
                input_width=width,
                output_width=width,
                role="stem" if index == 0 else "body",
            )
        )

    macro_edges = [MacroEdgeGene(source="input", target="macro_0")]
    for index in range(1, len(macro_nodes)):
        macro_edges.append(MacroEdgeGene(source=f"macro_{index-1}", target=f"macro_{index}"))
    macro_edges.append(MacroEdgeGene(source=f"macro_{len(macro_nodes)-1}", target="output"))
    child = HierarchicalGenome(
        genome_id=candidate_id,
        task=left.task,
        input_dim=left.input_dim,
        output_dim=left.output_dim,
        macro_nodes=macro_nodes,
        macro_edges=macro_edges,
        cell_library=cell_library,
    )
    return mutate_genome(child, rng=rng, candidate_id=candidate_id)


def _resize_cell(cell: CellGene, width: int) -> CellGene:
    nodes = [node.model_copy(update={"width": width}) for node in cell.nodes]
    return cell.model_copy(update={"input_width": width, "output_width": width, "nodes": nodes})


def _clamp_width(width: int) -> int:
    return max(8, min(256, width))
