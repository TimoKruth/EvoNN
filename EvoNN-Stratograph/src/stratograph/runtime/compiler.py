"""Deterministic prototype compiler for hierarchical genomes."""

from __future__ import annotations

import hashlib
from dataclasses import dataclass

import numpy as np

from stratograph.genome.codec import genome_digest
from stratograph.genome.models import (
    ActivationKind,
    CellGene,
    HierarchicalGenome,
    PrimitiveKind,
)


@dataclass(frozen=True)
class CompiledCell:
    """Executable micro-graph."""

    cell: CellGene
    digest: str

    def forward(self, x: np.ndarray) -> np.ndarray:
        states: dict[str, np.ndarray] = {"input": x}
        node_order = _topological_nodes(self.cell.edges, [node.node_id for node in self.cell.nodes])
        node_map = {node.node_id: node for node in self.cell.nodes}

        for node_id in node_order:
            node = node_map[node_id]
            incoming = [edge.source for edge in self.cell.edges if edge.enabled and edge.target == node_id]
            merged = _merge_inputs(
                [states[source] for source in incoming],
                out_dim=node.width,
                tag=f"{self.digest}:{self.cell.cell_id}:{node_id}",
            )
            states[node_id] = _node_transform(
                merged,
                kind=node.kind,
                activation=node.activation,
                width=node.width,
                tag=f"{self.digest}:{self.cell.cell_id}:{node_id}",
            )

        outputs = [edge.source for edge in self.cell.edges if edge.enabled and edge.target == "output"]
        if not outputs:
            tail = states[node_order[-1]] if node_order else x
            return _project(tail, self.cell.output_width, f"{self.digest}:{self.cell.cell_id}:output")
        return _merge_inputs(
            [states[source] for source in outputs],
            out_dim=self.cell.output_width,
            tag=f"{self.digest}:{self.cell.cell_id}:output",
        )


@dataclass(frozen=True)
class CompiledHierarchy:
    """Executable macro graph with shared compiled cells."""

    genome: HierarchicalGenome
    compiled_cells: dict[str, CompiledCell]
    digest: str

    def encode(self, x: np.ndarray) -> np.ndarray:
        macro_nodes = {node.node_id: node for node in self.genome.macro_nodes}
        states: dict[str, np.ndarray] = {"input": self._prepare_input(x)}
        for node_id in _topological_nodes(self.genome.macro_edges, list(macro_nodes)):
            node = macro_nodes[node_id]
            incoming = [edge.source for edge in self.genome.macro_edges if edge.enabled and edge.target == node_id]
            merged = _merge_inputs(
                [states[source] for source in incoming],
                out_dim=node.input_width,
                tag=f"{self.digest}:macro:{node_id}:input",
            )
            states[node_id] = self.compiled_cells[node.cell_id].forward(merged)

        output_sources = [edge.source for edge in self.genome.macro_edges if edge.enabled and edge.target == "output"]
        return _merge_inputs(
            [states[source] for source in output_sources],
            out_dim=macro_nodes[output_sources[-1]].output_width if output_sources and output_sources[-1] in macro_nodes else self.genome.macro_nodes[-1].output_width,
            tag=f"{self.digest}:macro:encoded_output",
        )

    def forward(self, x: np.ndarray) -> np.ndarray:
        encoded = self.encode(x)
        return _merge_inputs(
            [encoded],
            out_dim=self.genome.output_dim,
            tag=f"{self.digest}:macro:output",
        )

    def architecture_summary(self) -> str:
        return (
            f"macro_nodes={len(self.genome.macro_nodes)} "
            f"cells={len(self.genome.cell_library)} "
            f"macro_depth={self.genome.macro_depth} "
            f"avg_cell_depth={self.genome.average_cell_depth:.1f} "
            f"reuse_ratio={self.genome.reuse_ratio:.2f}"
        )

    def parameter_count(self) -> int:
        total = 0
        for cell in self.genome.cell_library.values():
            prev_width = cell.input_width
            for node in cell.nodes:
                total += prev_width * node.width + node.width
                prev_width = node.width
            total += prev_width * cell.output_width + cell.output_width
        total += sum(node.output_width for node in self.genome.macro_nodes) * self.genome.output_dim
        if self.genome.task == "language_modeling":
            total += self.genome.output_dim * self.genome.macro_nodes[0].input_width
        return total

    def _prepare_input(self, x: np.ndarray) -> np.ndarray:
        if self.genome.task != "language_modeling":
            array = np.asarray(x, dtype=np.float32)
            if array.ndim > 2:
                return array.reshape(array.shape[0], -1)
            return array

        tokens = np.asarray(x, dtype=np.int32)
        first_width = self.genome.macro_nodes[0].input_width
        embedding = _deterministic_matrix(
            self.digest + ":embedding",
            self.genome.output_dim,
            first_width,
        )
        return embedding[tokens]


def compile_genome(genome: HierarchicalGenome) -> CompiledHierarchy:
    """Compile hierarchical genome into deterministic NumPy executor."""
    digest = genome_digest(genome)
    return CompiledHierarchy(
        genome=genome,
        compiled_cells={
            cell_id: CompiledCell(cell=cell, digest=digest)
            for cell_id, cell in genome.cell_library.items()
        },
        digest=digest,
    )


def _merge_inputs(inputs: list[np.ndarray], out_dim: int, tag: str) -> np.ndarray:
    if not inputs:
        raise ValueError("node must have at least one input")
    projected = [_project(array, out_dim, f"{tag}:{index}") for index, array in enumerate(inputs)]
    merged = projected[0]
    for array in projected[1:]:
        merged = merged + array
    return merged / len(projected)


def _project(x: np.ndarray, out_dim: int, tag: str) -> np.ndarray:
    in_dim = int(x.shape[-1])
    weights = _deterministic_matrix(tag, in_dim, out_dim)
    bias = _deterministic_vector(tag + ":bias", out_dim)
    return np.einsum("...i,ij->...j", x, weights, optimize=True) + bias


def _deterministic_matrix(tag: str, rows: int, cols: int) -> np.ndarray:
    seed = int.from_bytes(hashlib.sha256(tag.encode("utf-8")).digest()[:8], "big") % (2**32)
    rng = np.random.default_rng(seed)
    scale = 1.0 / max(1, rows) ** 0.5
    return rng.normal(loc=0.0, scale=scale, size=(rows, cols)).astype(np.float32)


def _deterministic_vector(tag: str, size: int) -> np.ndarray:
    seed = int.from_bytes(hashlib.sha256(tag.encode("utf-8")).digest()[:8], "big") % (2**32)
    rng = np.random.default_rng(seed)
    return rng.normal(loc=0.0, scale=0.05, size=size).astype(np.float32)


def _apply_activation(x: np.ndarray, activation: ActivationKind) -> np.ndarray:
    if activation == ActivationKind.IDENTITY:
        return x
    if activation == ActivationKind.RELU:
        return np.maximum(x, 0.0)
    if activation == ActivationKind.TANH:
        return np.tanh(x)
    if activation == ActivationKind.GELU:
        return 0.5 * x * (1.0 + np.tanh(np.sqrt(2.0 / np.pi) * (x + 0.044715 * (x**3))))
    raise ValueError(f"Unsupported activation: {activation}")


def _node_transform(
    x: np.ndarray,
    *,
    kind: PrimitiveKind,
    activation: ActivationKind,
    width: int,
    tag: str,
) -> np.ndarray:
    if kind == PrimitiveKind.LINEAR:
        base = _project(x, width, f"{tag}:linear")
    elif kind == PrimitiveKind.RESIDUAL:
        residual = _project(x, width, f"{tag}:residual")
        skip = _project(x, width, f"{tag}:skip")
        base = (residual + skip) / 2.0
    elif kind == PrimitiveKind.GATE:
        value = _project(x, width, f"{tag}:value")
        gate = _sigmoid(_project(x, width, f"{tag}:gate"))
        base = value * gate
    elif kind == PrimitiveKind.MIX:
        left = _project(x, width, f"{tag}:mix_left")
        right = _project(np.tanh(x), width, f"{tag}:mix_right")
        base = (left + right) / 2.0
    elif kind == PrimitiveKind.NORM:
        normalized = _layer_norm(x)
        base = _project(normalized, width, f"{tag}:norm")
    else:
        raise ValueError(f"Unsupported primitive kind: {kind}")
    return _apply_activation(base, activation)


def _sigmoid(x: np.ndarray) -> np.ndarray:
    return 1.0 / (1.0 + np.exp(-np.clip(x, -40.0, 40.0)))


def _layer_norm(x: np.ndarray, eps: float = 1e-5) -> np.ndarray:
    mean = np.mean(x, axis=-1, keepdims=True)
    var = np.var(x, axis=-1, keepdims=True)
    return (x - mean) / np.sqrt(var + eps)


def _topological_nodes(edges: list, nodes: list[str]) -> list[str]:
    enabled_edges = [(edge.source, edge.target) for edge in edges if edge.enabled]
    preds: dict[str, set[str]] = {node: set() for node in nodes}
    succs: dict[str, set[str]] = {node: set() for node in nodes}
    for source, target in enabled_edges:
        if source == "input" or target == "output":
            if target in preds and source != "input":
                preds[target].add(source)
            continue
        preds.setdefault(target, set()).add(source)
        succs.setdefault(source, set()).add(target)
        succs.setdefault(target, set())

    ready = sorted([node for node in nodes if not preds.get(node)])
    order: list[str] = []
    while ready:
        node = ready.pop(0)
        order.append(node)
        for target in sorted(succs.get(node, ())):
            preds[target].discard(node)
            if not preds[target] and target not in order and target not in ready:
                ready.append(target)
        ready.sort()
    if len(order) != len(nodes):
        raise ValueError("graph is not a DAG")
    return order
