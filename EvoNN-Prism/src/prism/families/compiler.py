"""Compile a ModelGenome into an MLX model with family-modality validation."""

from __future__ import annotations

from collections.abc import Mapping, Sequence
from dataclasses import dataclass

import mlx.core as mx
import mlx.nn as nn

from prism.genome import ModelGenome, _sanitize_for_family
from prism.families.models import FAMILY_CLASSES

# ---------------------------------------------------------------------------
# Family -> supported modalities
# ---------------------------------------------------------------------------

FAMILY_MODALITY: dict[str, list[str]] = {
    "mlp": ["tabular", "image", "sequence", "text"],
    "sparse_mlp": ["tabular", "image", "sequence", "text"],
    "moe_mlp": ["tabular"],
    "conv2d": ["image"],
    "lite_conv2d": ["image"],
    "conv1d": ["sequence"],
    "lite_conv1d": ["sequence"],
    "gru": ["sequence"],
    "embedding": ["text"],
    "attention": ["text", "sequence"],
    "sparse_attention": ["text", "sequence"],
}


# ---------------------------------------------------------------------------
# Compiled output
# ---------------------------------------------------------------------------

@dataclass
class CompiledModel:
    """Result of compiling a genome: the model, its family name, and parameter count."""

    model: nn.Module
    family: str
    parameter_count: int


# ---------------------------------------------------------------------------
# Parameter counting
# ---------------------------------------------------------------------------

def count_parameters(params) -> int:
    """Count total scalar parameters in a (possibly nested) parameter tree."""
    if hasattr(params, "shape"):
        result = 1
        for d in params.shape:
            result *= d
        return int(result)
    if isinstance(params, Mapping):
        return sum(count_parameters(v) for v in params.values())
    if isinstance(params, Sequence) and not isinstance(params, (str, bytes)):
        return sum(count_parameters(v) for v in params)
    return 0


# ---------------------------------------------------------------------------
# Public API
# ---------------------------------------------------------------------------

def compile_genome(
    genome: ModelGenome,
    input_shape: list[int],
    output_dim: int,
    modality: str,
    task: str = "classification",
) -> CompiledModel:
    """Compile a genome into an MLX model.

    Args:
        genome: Architecture genome to compile.
        input_shape: Shape of a single input sample (excluding batch dim).
        output_dim: Number of output units (classes or regression targets).
        modality: One of "tabular", "image", "sequence", "text".

    Returns:
        CompiledModel with the instantiated model, family name, and parameter count.

    Raises:
        ValueError: If the family is unknown, incompatible with the modality,
                     or the genome is invalid.
    """
    genome = prepare_genome_for_compile(genome, input_shape, output_dim, modality, task)
    family = genome.family

    cls = FAMILY_CLASSES[family]
    try:
        model = cls(genome, input_shape, output_dim, task=task)
    except TypeError as exc:
        if "unexpected keyword argument 'task'" not in str(exc):
            raise
        model = cls(genome, input_shape, output_dim)

    _dry_run_guard(model, input_shape, output_dim, modality, task)

    # Count parameters
    param_count = count_parameters(model.parameters())

    return CompiledModel(model=model, family=family, parameter_count=param_count)


def compatible_families(modality: str) -> list[str]:
    """Return list of family names compatible with the given modality.

    Args:
        modality: One of "tabular", "image", "sequence", "text".

    Returns:
        Sorted list of compatible family name strings.
    """
    return sorted(f for f, modalities in FAMILY_MODALITY.items() if modality in modalities)


def is_genome_compatible(
    genome: ModelGenome,
    modality: str,
    task: str = "classification",
) -> bool:
    family = genome.family
    if family not in FAMILY_CLASSES:
        return False
    if task == "language_modeling" and family not in {"embedding", "attention", "sparse_attention"}:
        return False
    return modality in FAMILY_MODALITY.get(family, [])


def prepare_genome_for_compile(
    genome: ModelGenome,
    input_shape: list[int],
    output_dim: int,
    modality: str,
    task: str = "classification",
) -> ModelGenome:
    family = genome.family
    if family not in FAMILY_CLASSES:
        raise ValueError(f"Unknown model family: {family!r}")
    if task == "language_modeling" and family not in {"embedding", "attention", "sparse_attention"}:
        raise ValueError(f"Family {family!r} does not support language_modeling.")

    allowed = FAMILY_MODALITY.get(family, [])
    if modality not in allowed:
        raise ValueError(
            f"Family {family!r} is not compatible with modality {modality!r}. "
            f"Allowed modalities: {allowed}"
        )
    if not input_shape or any(int(dim) <= 0 for dim in input_shape):
        raise ValueError(f"Invalid input_shape: {input_shape!r}")
    if output_dim <= 0:
        raise ValueError("output_dim must be positive.")

    payload = _sanitize_for_family(genome.model_dump(mode="python"))
    payload["hidden_layers"] = [max(8, int(width)) for width in payload["hidden_layers"]]
    payload["dropout"] = max(0.0, min(0.9, float(payload["dropout"])))
    payload["kernel_size"] = max(1, int(payload["kernel_size"]))
    if payload["kernel_size"] % 2 == 0:
        payload["kernel_size"] += 1
    if task == "language_modeling" and payload["norm_type"] == "batch":
        payload["norm_type"] = "layer"
    if family in {"attention", "sparse_attention", "embedding"}:
        payload["embedding_dim"] = max(8, int(payload["embedding_dim"]))
    if family == "moe_mlp":
        payload["num_experts"] = max(2, int(payload["num_experts"] or 2))
        payload["moe_top_k"] = min(max(1, int(payload["moe_top_k"])), payload["num_experts"])
    return ModelGenome.model_validate(payload)


def _dry_run_guard(
    model: nn.Module,
    input_shape: list[int],
    output_dim: int,
    modality: str,
    task: str,
) -> None:
    sample = _dummy_input(input_shape, modality, task)
    output = model(sample)
    mx.eval(output)
    shape = tuple(int(dim) for dim in output.shape)
    if task == "language_modeling":
        if len(shape) != 3 or shape[-1] != output_dim:
            raise ValueError(f"Invalid LM output shape: {shape!r}")
        return
    if len(shape) != 2 or shape[-1] != output_dim:
        raise ValueError(f"Invalid output shape: {shape!r}")


def _dummy_input(input_shape: list[int], modality: str, task: str):
    batch = 2
    shape = (batch, *input_shape)
    if modality == "text" or task == "language_modeling":
        return mx.zeros(shape, dtype=mx.int32)
    return mx.zeros(shape, dtype=mx.float32)
