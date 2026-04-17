"""Hierarchical genome models for Stratograph."""

from __future__ import annotations

from enum import Enum
from typing import Self

from pydantic import BaseModel, ConfigDict, Field, model_validator


class PrimitiveKind(str, Enum):
    LINEAR = "linear"
    RESIDUAL = "residual"
    GATE = "gate"
    MIX = "mix"
    NORM = "norm"


class ActivationKind(str, Enum):
    RELU = "relu"
    TANH = "tanh"
    GELU = "gelu"
    IDENTITY = "identity"


class CellNodeGene(BaseModel):
    """Single operator inside a cell."""

    model_config = ConfigDict(frozen=True)

    node_id: str
    kind: PrimitiveKind = PrimitiveKind.LINEAR
    width: int
    activation: ActivationKind = ActivationKind.RELU


class CellEdgeGene(BaseModel):
    """Directed connection inside a cell."""

    model_config = ConfigDict(frozen=True)

    source: str
    target: str
    enabled: bool = True


class CellGene(BaseModel):
    """Reusable micro-graph."""

    model_config = ConfigDict(frozen=True)

    cell_id: str
    input_width: int
    output_width: int
    shared: bool = True
    nodes: list[CellNodeGene] = Field(default_factory=list)
    edges: list[CellEdgeGene] = Field(default_factory=list)

    @model_validator(mode="after")
    def validate_cell(self) -> Self:
        node_ids = {node.node_id for node in self.nodes}
        if len(node_ids) != len(self.nodes):
            raise ValueError(f"cell {self.cell_id} has duplicate node ids")
        for edge in self.edges:
            if edge.source not in node_ids | {"input"}:
                raise ValueError(f"cell {self.cell_id} edge source missing: {edge.source}")
            if edge.target not in node_ids | {"output"}:
                raise ValueError(f"cell {self.cell_id} edge target missing: {edge.target}")
        _assert_acyclic([(edge.source, edge.target) for edge in self.edges if edge.enabled], node_ids)
        return self


class MacroNodeGene(BaseModel):
    """Node in macro graph referencing a cell."""

    model_config = ConfigDict(frozen=True)

    node_id: str
    cell_id: str
    input_width: int
    output_width: int
    role: str = "hidden"


class MacroEdgeGene(BaseModel):
    """Directed connection between macro nodes."""

    model_config = ConfigDict(frozen=True)

    source: str
    target: str
    enabled: bool = True


class HierarchicalGenome(BaseModel):
    """Two-level genome: macro DAG over reusable cell library."""

    model_config = ConfigDict(frozen=True)

    genome_id: str
    task: str
    input_dim: int
    output_dim: int
    macro_nodes: list[MacroNodeGene]
    macro_edges: list[MacroEdgeGene]
    cell_library: dict[str, CellGene]

    @model_validator(mode="after")
    def validate_hierarchy(self) -> Self:
        macro_ids = {node.node_id for node in self.macro_nodes}
        if len(macro_ids) != len(self.macro_nodes):
            raise ValueError("duplicate macro node ids")
        for node in self.macro_nodes:
            if node.cell_id not in self.cell_library:
                raise ValueError(f"macro node {node.node_id} references missing cell {node.cell_id}")
        for edge in self.macro_edges:
            if edge.source not in macro_ids | {"input"}:
                raise ValueError(f"macro edge source missing: {edge.source}")
            if edge.target not in macro_ids | {"output"}:
                raise ValueError(f"macro edge target missing: {edge.target}")
        _assert_acyclic([(edge.source, edge.target) for edge in self.macro_edges if edge.enabled], macro_ids)
        if not any(edge.target == "output" and edge.enabled for edge in self.macro_edges):
            raise ValueError("macro graph must reach output")
        return self

    @property
    def macro_depth(self) -> int:
        return _graph_depth([(edge.source, edge.target) for edge in self.macro_edges if edge.enabled])

    @property
    def average_cell_depth(self) -> float:
        depths = [_graph_depth([(edge.source, edge.target) for edge in cell.edges if edge.enabled]) for cell in self.cell_library.values()]
        return float(sum(depths) / len(depths)) if depths else 0.0

    @property
    def reuse_ratio(self) -> float:
        if not self.macro_nodes:
            return 0.0
        distinct = len({node.cell_id for node in self.macro_nodes})
        return 1.0 - (distinct / len(self.macro_nodes))

    @classmethod
    def create_seed(
        cls,
        *,
        benchmark_name: str,
        task: str,
        input_dim: int,
        output_dim: int,
        seed: int,
    ) -> "HierarchicalGenome":
        width = max(16, min(128, max(input_dim, min(output_dim, 64))))
        cell_id = "cell_shared_stem"
        cell = CellGene(
            cell_id=cell_id,
            input_width=width,
            output_width=width,
            shared=True,
            nodes=[
                CellNodeGene(node_id="mix_0", kind=PrimitiveKind.MIX, width=width, activation=ActivationKind.GELU),
                CellNodeGene(node_id="gate_0", kind=PrimitiveKind.GATE, width=width, activation=ActivationKind.TANH),
                CellNodeGene(node_id="res_0", kind=PrimitiveKind.RESIDUAL, width=width, activation=ActivationKind.IDENTITY),
            ],
            edges=[
                CellEdgeGene(source="input", target="mix_0"),
                CellEdgeGene(source="mix_0", target="gate_0"),
                CellEdgeGene(source="gate_0", target="res_0"),
                CellEdgeGene(source="res_0", target="output"),
            ],
        )
        return cls(
            genome_id=f"{benchmark_name}_seed_{seed}",
            task=task,
            input_dim=input_dim,
            output_dim=output_dim,
            macro_nodes=[
                MacroNodeGene(node_id="macro_0", cell_id=cell_id, input_width=width, output_width=width, role="stem"),
                MacroNodeGene(node_id="macro_1", cell_id=cell_id, input_width=width, output_width=width, role="body"),
            ],
            macro_edges=[
                MacroEdgeGene(source="input", target="macro_0"),
                MacroEdgeGene(source="macro_0", target="macro_1"),
                MacroEdgeGene(source="macro_1", target="output"),
            ],
            cell_library={cell_id: cell},
        )


def _assert_acyclic(edges: list[tuple[str, str]], nodes: set[str]) -> None:
    adjacency: dict[str, list[str]] = {node: [] for node in nodes}
    for source, target in edges:
        if source == "input" or target == "output":
            continue
        adjacency.setdefault(source, []).append(target)
        adjacency.setdefault(target, [])

    visiting: set[str] = set()
    visited: set[str] = set()

    def visit(node: str) -> None:
        if node in visited:
            return
        if node in visiting:
            raise ValueError("graph must be acyclic")
        visiting.add(node)
        for target in adjacency.get(node, []):
            visit(target)
        visiting.remove(node)
        visited.add(node)

    for node in adjacency:
        visit(node)


def _graph_depth(edges: list[tuple[str, str]]) -> int:
    preds: dict[str, list[str]] = {}
    nodes: set[str] = {"input", "output"}
    for source, target in edges:
        preds.setdefault(target, []).append(source)
        nodes.add(source)
        nodes.add(target)

    cache: dict[str, int] = {"input": 0}

    def depth(node: str) -> int:
        if node in cache:
            return cache[node]
        parents = preds.get(node, [])
        if not parents:
            cache[node] = 0
        else:
            cache[node] = 1 + max(depth(parent) for parent in parents)
        return cache[node]

    return max(depth(node) for node in nodes if node != "input")
