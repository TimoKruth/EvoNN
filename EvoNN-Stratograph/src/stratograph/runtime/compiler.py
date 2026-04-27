"""Deterministic compiler for hierarchical genomes.

Prefers MLX when available and falls back to NumPy on non-MLX hosts so the
package still imports and tests outside Apple Silicon.
"""

from __future__ import annotations

import hashlib
from dataclasses import dataclass
from typing import Any

import numpy as np

try:  # pragma: no cover - exercised implicitly when MLX is installed
    import mlx.core as _mlx_core

    MLX_AVAILABLE = True
    _MLX_ARRAY_TYPE = type(_mlx_core.array([0.0]))
except ImportError:  # pragma: no cover - covered by Linux CI / non-MLX hosts
    MLX_AVAILABLE = False
    _mlx_core = None

from stratograph.runtime.backends import resolve_runtime_backend


class _NumpyBackend:
    float32 = np.float32
    int32 = np.int32

    @staticmethod
    def array(value):
        return np.asarray(value)

    @staticmethod
    def maximum(a, b):
        return np.maximum(a, b)

    @staticmethod
    def tanh(x):
        return np.tanh(x)

    @staticmethod
    def matmul(a, b):
        return np.matmul(a, b)

    @staticmethod
    def mean(x, axis=None, keepdims=False):
        return np.mean(x, axis=axis, keepdims=keepdims)

    @staticmethod
    def var(x, axis=None, keepdims=False):
        return np.var(x, axis=axis, keepdims=keepdims)

    @staticmethod
    def sqrt(x):
        return np.sqrt(x)

    @staticmethod
    def exp(x):
        return np.exp(x)

    @staticmethod
    def clip(x, a_min, a_max):
        return np.clip(x, a_min, a_max)


_NUMPY_BACKEND = _NumpyBackend()

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
    runtime_backend: str
    use_mlx: bool

    def forward(self, x: Any):
        dtype = _runtime_ops(self.use_mlx).float32
        states: dict[str, Any] = {"input": _as_array(x, dtype=dtype, use_mlx=self.use_mlx)}
        node_order = _topological_nodes(self.cell.edges, [node.node_id for node in self.cell.nodes])
        node_map = {node.node_id: node for node in self.cell.nodes}

        for node_id in node_order:
            node = node_map[node_id]
            incoming = [edge.source for edge in self.cell.edges if edge.enabled and edge.target == node_id]
            merged = _merge_inputs(
                [states[source] for source in incoming],
                out_dim=node.width,
                tag=f"{self.digest}:{self.cell.cell_id}:{node_id}",
                use_mlx=self.use_mlx,
            )
            states[node_id] = _node_transform(
                merged,
                kind=node.kind,
                activation=node.activation,
                width=node.width,
                tag=f"{self.digest}:{self.cell.cell_id}:{node_id}",
                use_mlx=self.use_mlx,
            )

        outputs = [edge.source for edge in self.cell.edges if edge.enabled and edge.target == "output"]
        if not outputs:
            tail = states[node_order[-1]] if node_order else _as_array(x, dtype=dtype, use_mlx=self.use_mlx)
            return _project(tail, self.cell.output_width, f"{self.digest}:{self.cell.cell_id}:output", use_mlx=self.use_mlx)
        return _merge_inputs(
            [states[source] for source in outputs],
            out_dim=self.cell.output_width,
            tag=f"{self.digest}:{self.cell.cell_id}:output",
            use_mlx=self.use_mlx,
        )


@dataclass(frozen=True)
class CompiledHierarchy:
    """Executable macro graph with shared compiled cells."""

    genome: HierarchicalGenome
    compiled_cells: dict[str, CompiledCell]
    digest: str
    runtime_backend: str
    use_mlx: bool

    def encode(self, x: Any):
        macro_nodes = {node.node_id: node for node in self.genome.macro_nodes}
        states: dict[str, Any] = {"input": self._prepare_input(x)}
        for node_id in _topological_nodes(self.genome.macro_edges, list(macro_nodes)):
            node = macro_nodes[node_id]
            incoming = [edge.source for edge in self.genome.macro_edges if edge.enabled and edge.target == node_id]
            merged = _merge_inputs(
                [states[source] for source in incoming],
                out_dim=node.input_width,
                tag=f"{self.digest}:macro:{node_id}:input",
                use_mlx=self.use_mlx,
            )
            states[node_id] = self.compiled_cells[node.cell_id].forward(merged)

        output_sources = [edge.source for edge in self.genome.macro_edges if edge.enabled and edge.target == "output"]
        output_dim = (
            macro_nodes[output_sources[-1]].output_width
            if output_sources and output_sources[-1] in macro_nodes
            else self.genome.macro_nodes[-1].output_width
        )
        return _merge_inputs(
            [states[source] for source in output_sources],
            out_dim=output_dim,
            tag=f"{self.digest}:macro:encoded_output",
            use_mlx=self.use_mlx,
        )

    def forward(self, x: Any):
        encoded = self.encode(x)
        return _merge_inputs(
            [encoded],
            out_dim=self.genome.output_dim,
            tag=f"{self.digest}:macro:output",
            use_mlx=self.use_mlx,
        )

    def architecture_summary(self) -> str:
        macro_edge_count = sum(1 for edge in self.genome.macro_edges if edge.enabled)
        macro_branch_factor = macro_edge_count / max(1, len(self.genome.macro_nodes))
        return (
            f"macro_nodes={len(self.genome.macro_nodes)} "
            f"macro_edges={macro_edge_count} "
            f"cells={len(self.genome.cell_library)} "
            f"macro_depth={self.genome.macro_depth} "
            f"avg_cell_depth={self.genome.average_cell_depth:.1f} "
            f"branch_factor={macro_branch_factor:.2f} "
            f"reuse_ratio={self.genome.reuse_ratio:.2f} "
            f"backend={self.runtime_backend}"
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

    def _prepare_input(self, x: Any):
        ops = _runtime_ops(self.use_mlx)
        if self.genome.task != "language_modeling":
            array = _as_array(x, dtype=ops.float32, use_mlx=self.use_mlx)
            if array.ndim > 2:
                return array.reshape(array.shape[0], -1)
            return array

        tokens = _as_array(x, dtype=ops.int32, use_mlx=self.use_mlx)
        first_width = self.genome.macro_nodes[0].input_width
        embedding = _deterministic_matrix(
            self.digest + ":embedding",
            self.genome.output_dim,
            first_width,
            use_mlx=self.use_mlx,
        )
        return embedding[tokens]


def compile_genome(genome: HierarchicalGenome, *, runtime_backend: str = "auto") -> CompiledHierarchy:
    """Compile hierarchical genome into a deterministic executor."""
    selection = resolve_runtime_backend(runtime_backend)
    use_mlx = selection.resolved_backend == "mlx"
    digest = genome_digest(genome)
    return CompiledHierarchy(
        genome=genome,
        compiled_cells={
            cell_id: CompiledCell(
                cell=cell,
                digest=digest,
                runtime_backend=selection.resolved_backend,
                use_mlx=use_mlx,
            )
            for cell_id, cell in genome.cell_library.items()
        },
        digest=digest,
        runtime_backend=selection.resolved_backend,
        use_mlx=use_mlx,
    )


def _runtime_ops(use_mlx: bool):
    return _mlx_core if use_mlx else _NUMPY_BACKEND


def _array_type(use_mlx: bool):
    return _MLX_ARRAY_TYPE if use_mlx else np.ndarray


def _as_array(x: Any, *, dtype=None, use_mlx: bool):
    if isinstance(x, _array_type(use_mlx)):
        array = x
    else:
        array = np.asarray(x)
        if use_mlx:
            array = _mlx_core.array(array)
    if dtype is not None:
        if use_mlx:
            return array.astype(dtype)
        return np.asarray(array, dtype=dtype)
    return array


def _merge_inputs(inputs: list[Any], out_dim: int, tag: str, *, use_mlx: bool):
    if not inputs:
        raise ValueError("node must have at least one input")
    projected = [_project(array, out_dim, f"{tag}:{index}", use_mlx=use_mlx) for index, array in enumerate(inputs)]
    merged = projected[0]
    for array in projected[1:]:
        merged = merged + array
    return merged / len(projected)


def _project(x: Any, out_dim: int, tag: str, *, use_mlx: bool):
    ops = _runtime_ops(use_mlx)
    array = _as_array(x, dtype=ops.float32, use_mlx=use_mlx)
    in_dim = int(array.shape[-1])
    weights = _deterministic_matrix(tag, in_dim, out_dim, use_mlx=use_mlx)
    bias = _deterministic_vector(tag + ":bias", out_dim, use_mlx=use_mlx)
    return ops.matmul(array, weights) + bias


def _deterministic_matrix(tag: str, rows: int, cols: int, *, use_mlx: bool):
    seed = int.from_bytes(hashlib.sha256(tag.encode("utf-8")).digest()[:8], "big") % (2**32)
    rng = np.random.default_rng(seed)
    scale = 1.0 / max(1, rows) ** 0.5
    array = rng.normal(loc=0.0, scale=scale, size=(rows, cols)).astype(np.float32)
    return _mlx_core.array(array) if use_mlx else array


def _deterministic_vector(tag: str, size: int, *, use_mlx: bool):
    seed = int.from_bytes(hashlib.sha256(tag.encode("utf-8")).digest()[:8], "big") % (2**32)
    rng = np.random.default_rng(seed)
    array = rng.normal(loc=0.0, scale=0.05, size=size).astype(np.float32)
    return _mlx_core.array(array) if use_mlx else array


def _apply_activation(x, activation: ActivationKind, *, use_mlx: bool):
    ops = _runtime_ops(use_mlx)
    if activation == ActivationKind.IDENTITY:
        return x
    if activation == ActivationKind.RELU:
        return ops.maximum(x, 0.0)
    if activation == ActivationKind.TANH:
        return ops.tanh(x)
    if activation == ActivationKind.GELU:
        return 0.5 * x * (1.0 + ops.tanh(np.sqrt(2.0 / np.pi) * (x + 0.044715 * (x**3))))
    raise ValueError(f"Unsupported activation: {activation}")


def _node_transform(
    x: Any,
    *,
    kind: PrimitiveKind,
    activation: ActivationKind,
    width: int,
    tag: str,
    use_mlx: bool,
):
    ops = _runtime_ops(use_mlx)
    array = _as_array(x, dtype=ops.float32, use_mlx=use_mlx)
    if kind == PrimitiveKind.LINEAR:
        base = _project(array, width, f"{tag}:linear", use_mlx=use_mlx)
    elif kind == PrimitiveKind.RESIDUAL:
        residual = _project(array, width, f"{tag}:residual", use_mlx=use_mlx)
        skip = _project(array, width, f"{tag}:skip", use_mlx=use_mlx)
        base = (residual + skip) / 2.0
    elif kind == PrimitiveKind.GATE:
        value = _project(array, width, f"{tag}:value", use_mlx=use_mlx)
        gate = _sigmoid(_project(array, width, f"{tag}:gate", use_mlx=use_mlx), use_mlx=use_mlx)
        base = value * gate
    elif kind == PrimitiveKind.MIX:
        left = _project(array, width, f"{tag}:mix_left", use_mlx=use_mlx)
        right = _project(ops.tanh(array), width, f"{tag}:mix_right", use_mlx=use_mlx)
        base = (left + right) / 2.0
    elif kind == PrimitiveKind.NORM:
        normalized = _layer_norm(array, use_mlx=use_mlx)
        base = _project(normalized, width, f"{tag}:norm", use_mlx=use_mlx)
    else:
        raise ValueError(f"Unsupported primitive kind: {kind}")
    return _apply_activation(base, activation, use_mlx=use_mlx)


def _sigmoid(x, *, use_mlx: bool):
    ops = _runtime_ops(use_mlx)
    return 1.0 / (1.0 + ops.exp(-ops.clip(x, -40.0, 40.0)))


def _layer_norm(x, eps: float = 1e-5, *, use_mlx: bool):
    ops = _runtime_ops(use_mlx)
    mean = ops.mean(x, axis=-1, keepdims=True)
    var = ops.var(x, axis=-1, keepdims=True)
    return (x - mean) / ops.sqrt(var + eps)


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
