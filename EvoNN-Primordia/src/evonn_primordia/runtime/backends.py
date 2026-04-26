"""Runtime backend selection and fallback implementations for Primordia."""
from __future__ import annotations

from dataclasses import dataclass
from random import Random
from types import SimpleNamespace
from typing import Any, Callable

import numpy as np
from sklearn import __version__ as sklearn_version
from sklearn.ensemble import RandomForestClassifier, RandomForestRegressor
from sklearn.linear_model import LogisticRegression, Ridge
from sklearn.metrics import mean_squared_error
from sklearn.neural_network import MLPClassifier, MLPRegressor
from sklearn.preprocessing import StandardScaler

from evonn_primordia.config import EvolutionConfig, RunConfig
from evonn_primordia.runtime.training import EvaluationResult, _compute_metric

try:  # pragma: no cover - exercised on MLX-capable hosts
    import mlx
except Exception:  # pragma: no cover - covered on non-MLX hosts
    mlx = None


@dataclass(frozen=True)
class RuntimeBindings:
    get_benchmark: Callable[[str], Any]
    benchmark_group: Callable[[Any], str]
    compatible_families: Callable[[str], list[str]]
    create_seed_genome: Callable[[str, int, int], Any]
    mutate_genome: Callable[..., tuple[Any, str]]
    compile_genome: Callable[[Any, list[int], int, str, str], Any]
    train_and_evaluate: Callable[..., Any]
    runtime_backend: str
    runtime_version: str | None
    precision_mode: str = "fp32"


def resolve_runtime_bindings(config: RunConfig) -> RuntimeBindings:
    requested = config.runtime.backend
    if requested == "mlx":
        return _load_mlx_runtime_bindings()
    if requested == "numpy-fallback":
        return _load_numpy_fallback_runtime_bindings()

    try:
        return _load_mlx_runtime_bindings()
    except RuntimeError as exc:
        if not config.runtime.allow_fallback:
            raise
        message = str(exc)
        portable_failures = (
            "mlx is not importable",
            "requires local MLX dependencies",
        )
        if any(fragment in message for fragment in portable_failures):
            return _load_numpy_fallback_runtime_bindings()
        raise


def _load_common_runtime_dependencies():
    from evonn_primordia.benchmarks import benchmark_group, get_benchmark
    from evonn_primordia.families.compiler import compatible_families
    from evonn_primordia.genome import apply_random_mutation, create_seed_genome

    def _seed_genome(family: str, width: int, depth: int):
        evo = EvolutionConfig(
            seed_hidden_width=width,
            seed_hidden_layers=depth,
            allowed_families=[family],
        )
        return create_seed_genome(family, evo, Random(0))

    def _mutate(genome, slot_index: int, allowed_families: list[str], config: RunConfig):
        evo = EvolutionConfig(
            seed_hidden_width=config.search.seed_hidden_width,
            seed_hidden_layers=config.search.seed_hidden_layers,
            max_hidden_width=config.search.max_hidden_width,
            max_hidden_layers=config.search.max_hidden_layers,
            allowed_families=allowed_families,
        )
        child, label = apply_random_mutation(genome, evo, Random(slot_index))
        return child, label

    return benchmark_group, compatible_families, get_benchmark, _mutate, _seed_genome


def _load_mlx_runtime_bindings() -> RuntimeBindings:
    if mlx is None:
        raise RuntimeError("mlx is not importable in this environment")
    try:
        benchmark_group, compatible_families, get_benchmark, mutate, seed_genome = _load_common_runtime_dependencies()
        from evonn_primordia.families.compiler import _load_family_classes, compile_genome
        from evonn_primordia.runtime.training import train_and_evaluate

        _load_family_classes()
    except Exception as exc:  # pragma: no cover - exercised on MLX-capable hosts
        raise RuntimeError(
            "Primordia MLX runtime requires local MLX dependencies and Primordia's own benchmark/model modules. "
            "Run this on your Apple Silicon workspace with `uv sync --package evonn-primordia`."
        ) from exc

    return RuntimeBindings(
        get_benchmark=get_benchmark,
        benchmark_group=benchmark_group,
        compatible_families=compatible_families,
        create_seed_genome=seed_genome,
        mutate_genome=mutate,
        compile_genome=lambda genome, input_shape, output_dim, modality, task: compile_genome(
            genome, input_shape, output_dim, modality, task=task
        ),
        train_and_evaluate=train_and_evaluate,
        runtime_backend="mlx",
        runtime_version=getattr(mlx, "__version__", None),
        precision_mode="fp32",
    )


def _load_numpy_fallback_runtime_bindings() -> RuntimeBindings:
    benchmark_group, compatible_families, get_benchmark, mutate, seed_genome = _load_common_runtime_dependencies()

    def _compile(genome, input_shape: list[int], output_dim: int, modality: str, task: str):
        return SimpleNamespace(
            model={
                "family": genome.family,
                "genome": genome,
                "input_shape": list(input_shape),
                "output_dim": int(output_dim),
                "modality": modality,
                "task": task,
            },
            family=genome.family,
            parameter_count=int(getattr(genome, "parameter_estimate", 0)),
        )

    return RuntimeBindings(
        get_benchmark=get_benchmark,
        benchmark_group=benchmark_group,
        compatible_families=compatible_families,
        create_seed_genome=seed_genome,
        mutate_genome=mutate,
        compile_genome=_compile,
        train_and_evaluate=_train_and_evaluate_numpy_fallback,
        runtime_backend="numpy-fallback",
        runtime_version=f"sklearn-{sklearn_version}",
        precision_mode="fp32",
    )


def _train_and_evaluate_numpy_fallback(
    model,
    X_train: np.ndarray,
    y_train: np.ndarray,
    X_val: np.ndarray,
    y_val: np.ndarray,
    *,
    task: str,
    epochs: int,
    lr: float,
    batch_size: int,
    lr_schedule: str = "cosine",
    grad_clip_norm: float = 1.0,
    weight_decay: float = 0.0,
    early_stopping_patience: int = 3,
    parameter_count: int = 0,
) -> EvaluationResult:
    del batch_size, lr_schedule, grad_clip_norm, early_stopping_patience

    import time

    start = time.perf_counter()
    family = str(model.get("family", "mlp"))
    genome = model.get("genome")

    try:
        if task == "language_modeling":
            metric_name, metric_value, quality = _evaluate_language_modeling_fallback(y_train, y_val)
            return EvaluationResult(
                metric_name=metric_name,
                metric_value=metric_value,
                quality=quality,
                parameter_count=parameter_count,
                train_seconds=time.perf_counter() - start,
            )

        x_train = _flatten_inputs(X_train)
        x_val = _flatten_inputs(X_val)
        scaler = StandardScaler()
        x_train = scaler.fit_transform(x_train)
        x_val = scaler.transform(x_val)

        if task == "classification":
            estimator = _build_classification_estimator(family, genome, epochs, lr, weight_decay)
            estimator.fit(x_train, y_train)
            if hasattr(estimator, "predict_proba"):
                y_pred = estimator.predict_proba(x_val)
            else:
                preds = estimator.predict(x_val)
                class_count = max(int(np.max(y_train)) + 1, 2)
                y_pred = np.eye(class_count, dtype=np.float32)[preds.astype(int)]
        else:
            estimator = _build_regression_estimator(family, genome, epochs, lr, weight_decay)
            estimator.fit(x_train, y_train)
            y_pred = estimator.predict(x_val)
            metric_value = float(mean_squared_error(y_val, y_pred))
            return EvaluationResult(
                metric_name="mse",
                metric_value=metric_value,
                quality=-metric_value,
                parameter_count=parameter_count,
                train_seconds=time.perf_counter() - start,
            )

        metric_name, metric_value, quality = _compute_metric(task, np.asarray(y_val), np.asarray(y_pred))
        return EvaluationResult(
            metric_name=metric_name,
            metric_value=metric_value,
            quality=quality,
            parameter_count=parameter_count,
            train_seconds=time.perf_counter() - start,
        )
    except Exception as exc:
        return EvaluationResult(
            metric_name="perplexity" if task == "language_modeling" else ("mse" if task == "regression" else "accuracy"),
            metric_value=float("nan"),
            quality=float("-inf"),
            parameter_count=parameter_count,
            train_seconds=time.perf_counter() - start,
            failure_reason=f"numpy_fallback_error: {exc}",
        )


def _flatten_inputs(values: np.ndarray) -> np.ndarray:
    values = np.asarray(values)
    if values.ndim <= 2:
        return values.astype(np.float32)
    return values.reshape(values.shape[0], -1).astype(np.float32)


def _build_classification_estimator(family: str, genome: Any, epochs: int, lr: float, weight_decay: float):
    hidden_layers = tuple(getattr(genome, "hidden_layers", [64])[:3]) or (64,)
    activation = str(getattr(genome, "activation", "relu"))
    activation_map = {"relu": "relu", "tanh": "tanh", "gelu": "relu", "silu": "relu"}
    max_iter = max(30, int(epochs) * 25)
    if family == "moe_mlp":
        return RandomForestClassifier(n_estimators=max(32, min(128, sum(hidden_layers))), random_state=42)
    if family in {"conv2d", "lite_conv2d"}:
        return LogisticRegression(max_iter=max_iter, multi_class="auto")
    return MLPClassifier(
        hidden_layer_sizes=hidden_layers,
        activation=activation_map.get(activation, "relu"),
        learning_rate_init=max(float(lr), 1e-4),
        alpha=max(float(weight_decay), 1e-6),
        max_iter=max_iter,
        random_state=42,
    )


def _build_regression_estimator(family: str, genome: Any, epochs: int, lr: float, weight_decay: float):
    hidden_layers = tuple(getattr(genome, "hidden_layers", [64])[:3]) or (64,)
    activation = str(getattr(genome, "activation", "relu"))
    activation_map = {"relu": "relu", "tanh": "tanh", "gelu": "relu", "silu": "relu"}
    max_iter = max(40, int(epochs) * 30)
    if family == "moe_mlp":
        return RandomForestRegressor(n_estimators=max(32, min(128, sum(hidden_layers))), random_state=42)
    if family in {"conv2d", "lite_conv2d"}:
        return Ridge(alpha=max(float(weight_decay), 1e-6))
    return MLPRegressor(
        hidden_layer_sizes=hidden_layers,
        activation=activation_map.get(activation, "relu"),
        learning_rate_init=max(float(lr), 1e-4),
        alpha=max(float(weight_decay), 1e-6),
        max_iter=max_iter,
        random_state=42,
    )


def _evaluate_language_modeling_fallback(y_train: np.ndarray, y_val: np.ndarray) -> tuple[str, float, float]:
    targets = np.asarray(y_train).reshape(-1).astype(int)
    if targets.size == 0:
        return "perplexity", float("inf"), float("-inf")
    vocab = max(int(np.max(targets)), int(np.max(np.asarray(y_val).reshape(-1).astype(int))), 0) + 1
    counts = np.bincount(targets, minlength=vocab).astype(np.float64)
    probs = (counts + 1.0) / (counts.sum() + vocab)
    val_targets = np.asarray(y_val).reshape(-1).astype(int)
    cross_entropy = float(-np.mean(np.log(probs[val_targets])))
    perplexity = float(np.exp(np.clip(cross_entropy, -20.0, 20.0)))
    return "perplexity", perplexity, -perplexity
