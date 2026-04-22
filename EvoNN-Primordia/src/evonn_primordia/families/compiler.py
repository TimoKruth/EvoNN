"""Compile a ModelGenome into an MLX model with family-modality validation."""

from __future__ import annotations

from collections.abc import Mapping, Sequence
from dataclasses import dataclass
from typing import Any

from evonn_primordia.genome import ModelGenome

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

    model: Any
    family: str
    parameter_count: int


def _load_family_classes() -> dict[str, type[Any]]:
    """Import MLX-backed family implementations lazily.

    This keeps Primordia importable on non-Darwin hosts until the actual MLX
    runtime path is requested.
    """
    try:
        from evonn_primordia.families.models import FAMILY_CLASSES
    except Exception as exc:  # pragma: no cover - exercised via runtime error path
        raise RuntimeError(
            "Primordia family compilation requires MLX-backed model families. "
            "Install Primordia on an MLX-capable host or use the exported run artifacts instead."
        ) from exc
    return FAMILY_CLASSES


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
    family = genome.family
    family_classes = _load_family_classes()

    # Validate family exists
    if family not in family_classes:
        raise ValueError(f"Unknown model family: {family!r}")

    if task == "language_modeling" and family not in {"embedding", "attention", "sparse_attention"}:
        raise ValueError(f"Family {family!r} does not support language_modeling.")

    # Validate family-modality compatibility
    allowed = FAMILY_MODALITY.get(family, [])
    if modality not in allowed:
        raise ValueError(
            f"Family {family!r} is not compatible with modality {modality!r}. "
            f"Allowed modalities: {allowed}"
        )

    # Validate genome has hidden layers
    if not genome.hidden_layers:
        raise ValueError("Genome must have at least one hidden layer.")

    if output_dim <= 0:
        raise ValueError("output_dim must be positive.")

    # Instantiate the model
    cls = family_classes[family]
    try:
        model = cls(genome, input_shape, output_dim, task=task)
    except TypeError as exc:
        if "unexpected keyword argument 'task'" not in str(exc):
            raise
        model = cls(genome, input_shape, output_dim)

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
