"""Sklearn-backed fallback runtime for non-MLX Primordia hosts."""

from __future__ import annotations

from dataclasses import dataclass
from hashlib import sha1
from random import Random
from time import perf_counter
from typing import Any
import warnings

import numpy as np
from sklearn import __version__ as SKLEARN_VERSION
from sklearn.exceptions import ConvergenceWarning
from sklearn.metrics import accuracy_score, mean_squared_error
from sklearn.neural_network import MLPClassifier, MLPRegressor
from sklearn.preprocessing import StandardScaler

from evonn_primordia.config import EvolutionConfig, RunConfig
from evonn_primordia.families.compiler import FAMILY_MODALITY
from evonn_primordia.genome import ModelGenome, apply_random_mutation


FALLBACK_RUNTIME_BACKEND = "numpy-fallback"


@dataclass(frozen=True)
class FallbackCompiledModel:
    """Compiled fallback candidate placeholder."""

    model: ModelGenome
    family: str
    parameter_count: int


@dataclass(frozen=True)
class FallbackEvalResult:
    """Normalized benchmark result payload for the fallback runtime."""

    metric_name: str
    metric_value: float
    quality: float
    parameter_count: int
    train_seconds: float
    failure_reason: str | None = None


def runtime_version() -> str:
    """Return the fallback runtime version string."""

    return SKLEARN_VERSION


def compatible_families(modality: str) -> list[str]:
    """Expose non-text families that the local fallback can execute."""

    if modality == "text":
        return []
    return sorted(
        family
        for family, modalities in FAMILY_MODALITY.items()
        if modality in modalities and family not in {"embedding", "attention", "sparse_attention"}
    )


def create_seed_genome(family: str, width: int, depth: int) -> ModelGenome:
    """Create a deterministic seed genome for the fallback runtime."""

    resolved_width = max(16, int(width))
    resolved_depth = max(1, int(depth))
    updates: dict[str, Any] = {
        "family": family,
        "hidden_layers": [resolved_width] * resolved_depth,
    }
    if family == "moe_mlp":
        updates["num_experts"] = 2
        updates["moe_top_k"] = 1
    if family in {"conv2d", "lite_conv2d", "conv1d", "lite_conv1d"}:
        updates["kernel_size"] = 3
    if family in {"attention", "sparse_attention"}:
        updates["embedding_dim"] = resolved_width
        updates["num_heads"] = 1
    return ModelGenome(**updates)


def mutate_genome(
    genome: ModelGenome,
    slot_index: int,
    allowed_families: list[str],
    config: RunConfig,
) -> ModelGenome:
    """Apply Primordia-style mutation without requiring MLX."""

    evo = EvolutionConfig(
        seed_hidden_width=config.search.seed_hidden_width,
        seed_hidden_layers=config.search.seed_hidden_layers,
        max_hidden_width=config.search.max_hidden_width,
        max_hidden_layers=config.search.max_hidden_layers,
        allowed_families=allowed_families,
    )
    child, _label = apply_random_mutation(genome, evo, Random(slot_index))
    return child


def compile_genome(
    genome: ModelGenome,
    input_shape: list[int],
    output_dim: int,
    modality: str,
    task: str = "classification",
) -> FallbackCompiledModel:
    """Validate a genome and return a sklearn-backed compiled placeholder."""

    if task == "language_modeling" or modality == "text":
        raise RuntimeError(
            "Primordia numpy fallback does not support language_modeling. "
            "Use an MLX-capable host for text-family validation."
        )
    allowed = FAMILY_MODALITY.get(genome.family, [])
    if modality not in allowed:
        raise ValueError(
            f"Family {genome.family!r} is not compatible with modality {modality!r}. "
            f"Allowed modalities: {allowed}"
        )
    if not genome.hidden_layers:
        raise ValueError("Genome must have at least one hidden layer.")
    if output_dim <= 0:
        raise ValueError("output_dim must be positive.")
    return FallbackCompiledModel(
        model=genome,
        family=genome.family,
        parameter_count=_parameter_count(input_shape=input_shape, hidden_layers=genome.hidden_layers, output_dim=output_dim),
    )


def train_and_evaluate(
    model: ModelGenome,
    x_train,
    y_train,
    x_val,
    y_val,
    *,
    task: str,
    epochs: int,
    lr: float,
    batch_size: int,
    parameter_count: int,
) -> FallbackEvalResult:
    """Train a small sklearn MLP as a contract-compatible local fallback."""

    del batch_size
    if task == "language_modeling":
        raise RuntimeError(
            "Primordia numpy fallback does not support language_modeling. "
            "Use an MLX-capable host for text-family validation."
        )

    started = perf_counter()
    x_train_np = _flatten_features(x_train)
    x_val_np = _flatten_features(x_val)
    scaler = StandardScaler()
    x_train_np = scaler.fit_transform(x_train_np)
    x_val_np = scaler.transform(x_val_np)
    y_train_np = np.asarray(y_train)
    y_val_np = np.asarray(y_val)

    hidden_layers = tuple(int(width) for width in model.hidden_layers) or (16,)
    random_state = _stable_seed(model.genome_id)
    activation = _activation_name(model.activation)
    alpha = max(float(model.weight_decay), 1e-6)
    max_iter = max(1, int(epochs))

    with warnings.catch_warnings():
        warnings.simplefilter("ignore", ConvergenceWarning)
        if task == "classification":
            estimator = MLPClassifier(
                hidden_layer_sizes=hidden_layers,
                activation=activation,
                alpha=alpha,
                learning_rate_init=max(float(lr), 1e-5),
                max_iter=max_iter,
                random_state=random_state,
            )
            estimator.fit(x_train_np, y_train_np)
            predictions = estimator.predict(x_val_np)
            metric_name = "accuracy"
            metric_value = float(accuracy_score(y_val_np, predictions))
            quality = metric_value
        elif task == "regression":
            estimator = MLPRegressor(
                hidden_layer_sizes=hidden_layers,
                activation=activation,
                alpha=alpha,
                learning_rate_init=max(float(lr), 1e-5),
                max_iter=max_iter,
                random_state=random_state,
            )
            estimator.fit(x_train_np, y_train_np)
            predictions = estimator.predict(x_val_np)
            metric_name = "mse"
            metric_value = float(mean_squared_error(y_val_np, predictions))
            quality = -metric_value
        else:
            raise RuntimeError(f"Primordia numpy fallback does not support task {task!r}.")

    return FallbackEvalResult(
        metric_name=metric_name,
        metric_value=metric_value,
        quality=quality,
        parameter_count=_fitted_parameter_count(estimator) or parameter_count,
        train_seconds=perf_counter() - started,
    )


def _flatten_features(values: Any) -> np.ndarray:
    array = np.asarray(values, dtype=np.float32)
    if array.ndim <= 2:
        return array
    return array.reshape((array.shape[0], -1))


def _activation_name(name: str) -> str:
    if name == "tanh":
        return "tanh"
    return "relu"


def _parameter_count(*, input_shape: list[int], hidden_layers: list[int], output_dim: int) -> int:
    previous = 1
    for dim in input_shape:
        previous *= max(1, int(dim))
    total = 0
    for width in hidden_layers:
        resolved_width = max(1, int(width))
        total += previous * resolved_width + resolved_width
        previous = resolved_width
    total += previous * int(output_dim) + int(output_dim)
    return int(total)


def _fitted_parameter_count(estimator: Any) -> int | None:
    coefs = getattr(estimator, "coefs_", None)
    intercepts = getattr(estimator, "intercepts_", None)
    if coefs is None or intercepts is None:
        return None
    return int(sum(array.size for array in coefs) + sum(array.size for array in intercepts))


def _stable_seed(value: str) -> int:
    return int(sha1(value.encode("utf-8")).hexdigest()[:8], 16)
