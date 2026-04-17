"""Genome representation and mutation operators for Prism NAS."""

from __future__ import annotations

import hashlib
import json
from enum import Enum
from random import Random
from typing import TYPE_CHECKING

from pydantic import BaseModel

if TYPE_CHECKING:
    from prism.config import EvolutionConfig

NORM_CHOICES = ["none", "layer", "rms", "batch"]
KERNEL_CHOICES = [3, 5]
HEAD_CHOICES = [1, 2, 4, 8]
WEIGHT_DECAY_CHOICES = [0.0, 1e-5, 1e-4, 1e-3, 1e-2]
SPARSITY_CHOICES = [0.1, 0.25, 0.5, 0.75]
EXPERT_CHOICES = [0, 2, 4, 8]
MOE_TOP_K_CHOICES = [1, 2]
LR_SCALES = [0.5, 0.8, 1.2, 1.5]
WIDTH_DELTAS = [-32, -16, 16, 32]


class Modality(str, Enum):
    TABULAR = "tabular"
    IMAGE = "image"
    SEQUENCE = "sequence"
    TEXT = "text"


class ModelGenome(BaseModel, frozen=True):
    """Immutable genome describing a neural architecture candidate."""

    family: str
    hidden_layers: list[int]
    activation: str = "relu"
    dropout: float = 0.0
    residual: bool = False
    activation_sparsity: float = 0.0
    learning_rate: float = 1e-3
    kernel_size: int = 3
    embedding_dim: int = 64
    num_heads: int = 4
    norm_type: str = "none"
    weight_decay: float = 0.0
    num_experts: int = 0
    moe_top_k: int = 2

    @property
    def genome_id(self) -> str:
        payload = json.dumps(self.model_dump(), sort_keys=True)
        return hashlib.sha1(payload.encode("utf-8")).hexdigest()[:12]

    @property
    def parameter_estimate(self) -> int:
        """Rough parameter count from hidden layer dimensions."""
        if not self.hidden_layers:
            return 0
        total = 0
        prev = self.hidden_layers[0]  # assume input ~= first hidden
        for width in self.hidden_layers:
            total += prev * width + width  # weights + bias
            prev = width
        total += prev  # output layer (single output estimate)
        return total


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def _pick_other(current, choices, rng: Random):
    """Pick a random value from choices that differs from current."""
    candidates = [c for c in choices if c != current]
    return rng.choice(candidates) if candidates else current


def _clamp(value, lo, hi):
    return max(lo, min(hi, value))


# ---------------------------------------------------------------------------
# Mutation operators
# ---------------------------------------------------------------------------

def mutate_family(genome: ModelGenome, allowed: list[str], rng: Random) -> ModelGenome:
    new_family = _pick_other(genome.family, allowed, rng)
    return genome.model_copy(update={"family": new_family})


def mutate_width(genome: ModelGenome, max_width: int, rng: Random) -> ModelGenome:
    if not genome.hidden_layers:
        return genome
    idx = rng.randrange(len(genome.hidden_layers))
    delta = rng.choice(WIDTH_DELTAS)
    layers = list(genome.hidden_layers)
    layers[idx] = _clamp(layers[idx] + delta, 16, max_width)
    return genome.model_copy(update={"hidden_layers": layers})


def mutate_depth_add(genome: ModelGenome, max_layers: int, max_width: int, rng: Random) -> ModelGenome:
    if len(genome.hidden_layers) >= max_layers:
        return genome
    width = rng.choice(genome.hidden_layers) if genome.hidden_layers else _clamp(128, 16, max_width)
    layers = list(genome.hidden_layers)
    layers.insert(rng.randrange(len(layers) + 1), width)
    return genome.model_copy(update={"hidden_layers": layers})


def mutate_depth_remove(genome: ModelGenome, rng: Random) -> ModelGenome:
    if len(genome.hidden_layers) <= 1:
        return genome
    layers = list(genome.hidden_layers)
    layers.pop(rng.randrange(len(layers)))
    return genome.model_copy(update={"hidden_layers": layers})


def mutate_activation(genome: ModelGenome, choices: list[str], rng: Random) -> ModelGenome:
    return genome.model_copy(update={"activation": _pick_other(genome.activation, choices, rng)})


def mutate_dropout(genome: ModelGenome, choices: list[float], rng: Random) -> ModelGenome:
    return genome.model_copy(update={"dropout": _pick_other(genome.dropout, choices, rng)})


def mutate_residual(genome: ModelGenome, rng: Random) -> ModelGenome:
    return genome.model_copy(update={"residual": not genome.residual})


def mutate_learning_rate(genome: ModelGenome, rng: Random) -> ModelGenome:
    scale = rng.choice(LR_SCALES)
    new_lr = _clamp(genome.learning_rate * scale, 1e-5, 1e-1)
    return genome.model_copy(update={"learning_rate": new_lr})


def mutate_norm_type(genome: ModelGenome, rng: Random) -> ModelGenome:
    return genome.model_copy(update={"norm_type": _pick_other(genome.norm_type, NORM_CHOICES, rng)})


def mutate_weight_decay(genome: ModelGenome, rng: Random) -> ModelGenome:
    return genome.model_copy(update={
        "weight_decay": _pick_other(genome.weight_decay, WEIGHT_DECAY_CHOICES, rng),
    })


def mutate_kernel_size(genome: ModelGenome, rng: Random) -> ModelGenome:
    return genome.model_copy(update={
        "kernel_size": _pick_other(genome.kernel_size, KERNEL_CHOICES, rng),
    })


def mutate_sparsity(genome: ModelGenome, rng: Random) -> ModelGenome:
    return genome.model_copy(update={
        "activation_sparsity": _pick_other(genome.activation_sparsity, SPARSITY_CHOICES, rng),
    })


def mutate_num_heads(genome: ModelGenome, rng: Random) -> ModelGenome:
    return genome.model_copy(update={
        "num_heads": _pick_other(genome.num_heads, HEAD_CHOICES, rng),
    })


def mutate_experts(genome: ModelGenome, rng: Random) -> ModelGenome:
    new_experts = _pick_other(genome.num_experts, EXPERT_CHOICES, rng)
    top_k = min(genome.moe_top_k, max(1, new_experts)) if new_experts > 0 else genome.moe_top_k
    return genome.model_copy(update={"num_experts": new_experts, "moe_top_k": top_k})


_OPERATORS: list[tuple[str, str]] = [
    ("family", "family"),
    ("width", "width"),
    ("depth_add", "depth_add"),
    ("depth_remove", "depth_remove"),
    ("activation", "activation"),
    ("dropout", "dropout"),
    ("residual", "residual"),
    ("learning_rate", "lr"),
    ("norm_type", "norm_type"),
    ("weight_decay", "weight_decay"),
    ("kernel_size", "kernel_size"),
    ("sparsity", "sparsity"),
    ("num_heads", "num_heads"),
    ("experts", "experts"),
]


def apply_random_mutation(
    genome: ModelGenome,
    config: EvolutionConfig,
    rng: Random,
) -> tuple[ModelGenome, str]:
    """Apply a single random mutation, returning (child, operator_name)."""
    op_name, label = rng.choice(_OPERATORS)
    allowed_families = config.allowed_families or ["mlp", "conv2d", "attention"]

    match op_name:
        case "family":
            child = mutate_family(genome, allowed_families, rng)
        case "width":
            child = mutate_width(genome, config.max_hidden_width, rng)
        case "depth_add":
            child = mutate_depth_add(genome, config.max_hidden_layers, config.max_hidden_width, rng)
        case "depth_remove":
            child = mutate_depth_remove(genome, rng)
        case "activation":
            child = mutate_activation(genome, config.activation_choices, rng)
        case "dropout":
            child = mutate_dropout(genome, config.dropout_choices, rng)
        case "residual":
            child = mutate_residual(genome, rng)
        case "learning_rate":
            child = mutate_learning_rate(genome, rng)
        case "norm_type":
            child = mutate_norm_type(genome, rng)
        case "weight_decay":
            child = mutate_weight_decay(genome, rng)
        case "kernel_size":
            child = mutate_kernel_size(genome, rng)
        case "sparsity":
            child = mutate_sparsity(genome, rng)
        case "num_heads":
            child = mutate_num_heads(genome, rng)
        case "experts":
            child = mutate_experts(genome, rng)
        case _:
            child = genome

    return child, label


# ---------------------------------------------------------------------------
# Crossover
# ---------------------------------------------------------------------------

def _splice_hidden_layers(left: list[int], right: list[int], rng: Random) -> list[int]:
    left_cut = rng.randrange(len(left) + 1)
    right_cut = rng.randrange(len(right) + 1)
    result = left[:left_cut] + right[right_cut:]
    return result if result else [rng.choice(left + right)]


def _blend_lr(lr_a: float, lr_b: float, rng: Random) -> float:
    mix = rng.uniform(0.35, 0.65)
    return _clamp(lr_a * mix + lr_b * (1.0 - mix), 1e-5, 1e-1)


def crossover(parent_a: ModelGenome, parent_b: ModelGenome, rng: Random) -> ModelGenome:
    """Produce a child from two parents. 50% uniform, 50% splice."""
    a, b = parent_a.model_dump(), parent_b.model_dump()
    pick = rng.choice

    if rng.random() < 0.5:
        # Uniform crossover: each gene independently from either parent
        child = {key: pick([a[key], b[key]]) for key in a}
        child["hidden_layers"] = pick([list(a["hidden_layers"]), list(b["hidden_layers"])])
    else:
        # Splice crossover: hidden_layers spliced, other genes from either parent
        child = {key: pick([a[key], b[key]]) for key in a}
        child["hidden_layers"] = _splice_hidden_layers(a["hidden_layers"], b["hidden_layers"], rng)

    # Blended learning rate for both modes
    child["learning_rate"] = _blend_lr(a["learning_rate"], b["learning_rate"], rng)

    return ModelGenome.model_validate(child)


# ---------------------------------------------------------------------------
# Seed creation
# ---------------------------------------------------------------------------

def create_seed_genome(
    family: str,
    config: EvolutionConfig,
    rng: Random,
) -> ModelGenome:
    """Create an initial genome for the given family."""
    return ModelGenome(
        family=family,
        hidden_layers=[config.seed_hidden_width] * config.seed_hidden_layers,
        activation=config.activation_choices[0] if config.activation_choices else "relu",
        dropout=0.0,
        residual=False,
        learning_rate=1e-3,
        kernel_size=3,
        embedding_dim=64,
        num_heads=4,
        norm_type="none",
        weight_decay=0.0,
        num_experts=0,
        moe_top_k=2,
    )
