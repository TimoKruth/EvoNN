"""Evaluation stage: compile genomes, train models, collect results."""

from __future__ import annotations

from contextvars import ContextVar
from dataclasses import dataclass, field
from math import log1p
from random import Random

import numpy as np

from prism.genome import ModelGenome
from prism.storage import RunStore
from prism.runtime.cache import WeightCache
from prism.runtime.backends import resolve_runtime_backend_with_policy
from prism.runtime.training import EvaluationResult, train_and_evaluate
from prism.families.compiler import is_genome_compatible

_RUNTIME_CONFIG: ContextVar[object | None] = ContextVar("prism_runtime_config", default=None)


@dataclass
class GenerationState:
    """Mutable state passed between pipeline stages."""

    generation: int
    population: list[ModelGenome]
    results: dict[str, dict[str, EvaluationResult]] = field(
        default_factory=dict,
    )  # genome_id -> {benchmark_id -> EvaluationResult}
    archives: dict = field(default_factory=dict)
    total_evaluations: int = 0
    parent_ids: dict[str, list[str]] = field(default_factory=dict)
    benchmark_history: dict[str, list[float]] = field(default_factory=dict)
    benchmark_failures: dict[str, int] = field(default_factory=dict)
    benchmark_evaluations: dict[str, int] = field(default_factory=dict)
    lineage_ops: dict[str, str] = field(default_factory=dict)
    operator_stats: dict[str, dict[str, float]] = field(default_factory=dict)
    family_stats: dict[str, dict[str, float]] = field(default_factory=dict)


def evaluate(
    state: GenerationState,
    config,
    benchmark_specs: list,
    cache: WeightCache | None = None,
    store: RunStore | None = None,
    run_id: str | None = None,
) -> GenerationState:
    """Evaluate all genomes on selected benchmarks.

    For each genome/benchmark pair:
      1. Check eval cache (same genome_id + benchmark_id = skip)
      2. Compile genome
      3. Weight inheritance from parent if available
      4. Apply multi-fidelity epoch scaling
      5. Load + preprocess data
      6. Train and evaluate
      7. Cache weights

    Args:
        state: Current GenerationState.
        config: RunConfig with training and evolution settings.
        benchmark_specs: List of benchmark specifications to evaluate.
        cache: Optional WeightCache for weight inheritance.

    Returns:
        Updated GenerationState with results populated.
    """

    training = config.training
    evolution = config.evolution

    # Multi-fidelity epoch scaling
    epoch_scale = _multi_fidelity_scale(
        generation=state.generation,
        total_generations=evolution.num_generations,
        schedule=training.multi_fidelity_schedule if training.multi_fidelity else None,
    )

    priority_scores = _benchmark_priority_scores(state, benchmark_specs)
    genome_profiles = _genome_profiles(state)

    runtime_token = _RUNTIME_CONFIG.set(getattr(config, "runtime", None))
    try:
        for genome in state.population:
            genome_results = state.results.get(genome.genome_id, {})
            parent_ids = state.parent_ids.get(genome.genome_id, [])

            if store is not None and run_id is not None:
                store.save_genome(run_id, genome)

            for spec in benchmark_specs:
                benchmark_id = spec.id if hasattr(spec, "id") else spec.name

                if benchmark_id in genome_results:
                    continue

                benchmark_scale = _benchmark_epoch_multiplier(
                    benchmark_id,
                    priority_scores,
                    training.benchmark_epoch_min_scale,
                    training.benchmark_epoch_max_scale,
                )
                genome_scale = _genome_epoch_multiplier(
                    genome,
                    genome_profiles,
                    state.generation,
                    evolution,
                    training.efficiency_epoch_min_scale,
                    training.efficiency_epoch_max_scale,
                )

                result = _evaluate_single(
                    genome=genome,
                    spec=spec,
                    training=training,
                    epoch_scale=epoch_scale * benchmark_scale * genome_scale,
                    cache=cache,
                    parent_ids=parent_ids,
                )

                genome_results[benchmark_id] = result
                if result.failure_reason == "unsupported_benchmark":
                    continue
                state.total_evaluations += 1
                _record_benchmark_result(state, benchmark_id, result)

                if store is not None and run_id is not None:
                    store.save_evaluation(
                        run_id,
                        genome.genome_id,
                        state.generation,
                        benchmark_id,
                        result.metric_name,
                        result.metric_value,
                        result.quality,
                        result.parameter_count,
                        result.train_seconds,
                        result.failure_reason,
                        result.inherited_from,
                    )

            state.results[genome.genome_id] = genome_results
    finally:
        _RUNTIME_CONFIG.reset(runtime_token)

    return state


def select_benchmarks(
    state: GenerationState,
    config,
    all_benchmarks: list,
    rng: Random,
) -> list:
    """Select which benchmarks to evaluate this generation.

    Uses undercovered focus: identifies benchmarks where the population
    performs worst (relative to others) and prioritizes them.
    """
    if not all_benchmarks:
        return []

    # Keep evaluation count stable for fair-budget runs; change order, not set size.
    if len(all_benchmarks) <= 4:
        return list(all_benchmarks)

    if not state.benchmark_evaluations:
        return list(all_benchmarks)

    priority_scores = _benchmark_priority_scores(state, all_benchmarks)
    ranked = sorted(
        all_benchmarks,
        key=lambda spec: priority_scores.get(spec.id if hasattr(spec, "id") else spec.name, 0.0),
        reverse=True,
    )
    return ranked


def _evaluate_single(
    genome: ModelGenome,
    spec,
    training,
    epoch_scale: float,
    cache: WeightCache | None,
    parent_ids: list[str] | None = None,
) -> EvaluationResult:
    """Compile, optionally inherit weights, train, cache, return result."""
    from prism.families.compiler import compile_genome

    # Determine modality and task from spec
    modality = spec.modality if hasattr(spec, "modality") else "tabular"
    task = spec.task if hasattr(spec, "task") else "classification"
    input_shape = spec.input_shape if hasattr(spec, "input_shape") else None
    if not is_genome_compatible(genome, modality, task):
        return EvaluationResult(
            metric_name=_default_metric_name(task),
            metric_value=float("nan"),
            quality=float("-inf"),
            parameter_count=0,
            train_seconds=0.0,
            failure_reason="unsupported_benchmark",
        )
    # Load data before compile so LM output_dim can expand to observed token range.
    X_train, y_train, X_val, y_val = _load_data(spec, seed=42)
    output_dim = _resolve_output_dim(spec, X_train, y_train)
    runtime_config = _RUNTIME_CONFIG.get()
    runtime_selection = resolve_runtime_backend_with_policy(
        getattr(runtime_config, "backend", "auto"),
        allow_fallback=getattr(runtime_config, "allow_fallback", True),
    )
    if runtime_selection.resolved_backend == "numpy-fallback":
        return _evaluate_single_numpy_fallback(
            genome=genome,
            X_train=X_train,
            y_train=y_train,
            X_val=X_val,
            y_val=y_val,
            task=task,
            output_dim=output_dim,
        )

    try:
        compiled = compile_genome(genome, input_shape, output_dim, modality, task=task)
    except Exception as exc:
        return EvaluationResult(
            metric_name="accuracy" if task == "classification" else "mse",
            metric_value=float("nan"),
            quality=float("-inf"),
            parameter_count=0,
            train_seconds=0.0,
            failure_reason=f"compile_error:{type(exc).__name__}",
        )

    model = compiled.model
    param_count = compiled.parameter_count

    # Weight inheritance
    inherited_from = None
    if cache is not None:
        for parent_id in parent_ids or []:
            if cache.transfer_weights(parent_id, model):
                inherited_from = parent_id
                break
        if inherited_from is None and hasattr(cache, "transfer_best_available"):
            inherited_from = cache.transfer_best_available(
                model,
                family=genome.family,
                preferred_ids=parent_ids or [],
                exclude_ids={genome.genome_id},
            )

    # Load data from spec
    # Apply multi-fidelity epoch scaling
    epochs = max(1, int(training.epochs * epoch_scale))

    result = train_and_evaluate(
        model,
        X_train,
        y_train,
        X_val,
        y_val,
        task=task,
        epochs=epochs,
        lr=genome.learning_rate,
        batch_size=training.batch_size,
        lr_schedule=training.lr_schedule,
        grad_clip_norm=training.grad_clip_norm,
        weight_decay=genome.weight_decay,
        early_stopping_patience=training.early_stopping_patience,
        parameter_count=param_count,
    )

    # Cache weights on success
    if cache is not None and result.failure_reason is None:
        cache.store(genome.genome_id, model, family=genome.family)
    result.inherited_from = inherited_from

    return result


def _evaluate_single_numpy_fallback(
    *,
    genome: ModelGenome,
    X_train: np.ndarray,
    y_train: np.ndarray,
    X_val: np.ndarray,
    y_val: np.ndarray,
    task: str,
    output_dim: int,
) -> EvaluationResult:
    import time

    start = time.perf_counter()
    try:
        if task == "regression":
            predictions = _fallback_regression(X_train, y_train, X_val)
        elif task == "language_modeling":
            predictions = _fallback_language_probs(y_train, X_val, output_dim)
        else:
            predictions = _fallback_classification(X_train, y_train, X_val, output_dim)
        metric_name, metric_value, quality = _compute_fallback_metric(task, y_val, predictions)
        return EvaluationResult(
            metric_name=metric_name,
            metric_value=metric_value,
            quality=quality,
            parameter_count=max(1, int(genome.parameter_estimate)),
            train_seconds=time.perf_counter() - start,
        )
    except Exception as exc:
        return EvaluationResult(
            metric_name=_default_metric_name(task),
            metric_value=float("nan"),
            quality=float("-inf"),
            parameter_count=0,
            train_seconds=time.perf_counter() - start,
            failure_reason=f"numpy_fallback_error:{type(exc).__name__}",
        )


def _fallback_classification(
    X_train: np.ndarray,
    y_train: np.ndarray,
    X_val: np.ndarray,
    output_dim: int,
) -> np.ndarray:
    train, val = _standardize_pair(X_train, X_val)
    labels = y_train.ravel().astype(int)
    classes = np.unique(labels)
    centroids = []
    for cls in classes:
        rows = train[labels == cls]
        centroids.append(np.mean(rows, axis=0) if rows.size else np.zeros(train.shape[1]))
    centroid_array = np.vstack(centroids)
    distances = np.sum((val[:, None, :] - centroid_array[None, :, :]) ** 2, axis=2)
    nearest = classes[np.argmin(distances, axis=1)]
    logits = np.full((val.shape[0], max(output_dim, int(classes.max()) + 1)), -8.0)
    logits[np.arange(val.shape[0]), nearest] = 8.0
    return logits


def _fallback_regression(X_train: np.ndarray, y_train: np.ndarray, X_val: np.ndarray) -> np.ndarray:
    train, val = _standardize_pair(X_train, X_val)
    design = np.c_[train, np.ones(train.shape[0])]
    reg = 1e-3 * np.eye(design.shape[1])
    weights = np.linalg.pinv(design.T @ design + reg) @ design.T @ y_train.reshape(-1, 1)
    return (np.c_[val, np.ones(val.shape[0])] @ weights).reshape(-1)


def _fallback_language_probs(
    y_train: np.ndarray,
    X_val: np.ndarray,
    output_dim: int,
) -> np.ndarray:
    tokens = y_train.reshape(-1).astype(int)
    vocab = max(int(output_dim), int(tokens.max()) + 1 if tokens.size else 1)
    counts = np.bincount(np.clip(tokens, 0, vocab - 1), minlength=vocab).astype(float) + 1.0
    probs = counts / counts.sum()
    return np.tile(probs, (X_val.size, 1)).reshape(*X_val.shape, vocab)


def _standardize_pair(X_train: np.ndarray, X_val: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    train = X_train.reshape(X_train.shape[0], -1).astype(float)
    val = X_val.reshape(X_val.shape[0], -1).astype(float)
    mean = np.mean(train, axis=0, keepdims=True)
    std = np.std(train, axis=0, keepdims=True)
    std = np.where(std < 1e-8, 1.0, std)
    return (train - mean) / std, (val - mean) / std


def _compute_fallback_metric(
    task: str,
    y_true: np.ndarray,
    predictions: np.ndarray,
) -> tuple[str, float, float]:
    if task == "classification":
        labels = y_true.ravel().astype(int)
        pred_labels = np.argmax(predictions, axis=1)
        accuracy = float(np.mean(pred_labels == labels))
        return "accuracy", accuracy, accuracy
    if task == "language_modeling":
        probs = np.clip(predictions.reshape(-1, predictions.shape[-1]), 1e-8, 1.0)
        targets = y_true.reshape(-1).astype(int)
        targets = np.clip(targets, 0, probs.shape[-1] - 1)
        cross_entropy = float(-np.mean(np.log(probs[np.arange(targets.shape[0]), targets])))
        perplexity = float(np.exp(np.clip(cross_entropy, -20.0, 20.0)))
        return "perplexity", perplexity, -perplexity
    predicted = predictions.reshape(-1)
    actual = y_true.reshape(-1)
    mse = float(np.mean((predicted - actual) ** 2))
    return "mse", mse, -mse


def _resolve_output_dim(spec, X_train: np.ndarray, y_train: np.ndarray) -> int:
    """Resolve model output dim, expanding LM vocab when cached tokens exceed catalog cap."""
    declared = spec.output_dim if hasattr(spec, "output_dim") else getattr(spec, "num_classes", 1)
    if getattr(spec, "task", None) != "language_modeling":
        return declared
    observed_max = max(
        int(np.max(X_train)) if X_train.size else 0,
        int(np.max(y_train)) if y_train.size else 0,
    )
    return max(int(declared), observed_max + 1)


def _multi_fidelity_scale(
    generation: int,
    total_generations: int,
    schedule: list[float] | None,
) -> float:
    """Return epoch multiplier for the current generation.

    With a schedule like [0.35, 0.65, 1.0], each entry covers an equal
    fraction of the total generations.
    """
    if schedule is None:
        return 1.0

    if not schedule:
        return 1.0

    # Map generation to schedule bucket
    if total_generations <= 1:
        return schedule[-1]

    progress = generation / (total_generations - 1)
    bucket = min(int(progress * len(schedule)), len(schedule) - 1)
    return schedule[bucket]


def _load_data(spec, seed: int = 42):
    """Load train/val data from a benchmark spec.

    Supports specs with load_data() method or x_train/y_train attributes.
    """
    if hasattr(spec, "load_data"):
        return spec.load_data(seed=seed)

    # Fallback: assume spec has data attributes
    return spec.x_train, spec.y_train, spec.x_val, spec.y_val


def _default_metric_name(task: str) -> str:
    if task == "classification":
        return "accuracy"
    if task == "language_modeling":
        return "perplexity"
    return "mse"


def _record_benchmark_result(
    state: GenerationState,
    benchmark_id: str,
    result: EvaluationResult,
) -> None:
    state.benchmark_evaluations[benchmark_id] = state.benchmark_evaluations.get(benchmark_id, 0) + 1
    if result.failure_reason is None:
        state.benchmark_history.setdefault(benchmark_id, []).append(float(result.quality))
    else:
        state.benchmark_failures[benchmark_id] = state.benchmark_failures.get(benchmark_id, 0) + 1


def update_search_memory(
    state: GenerationState,
    config,
) -> None:
    """Fold latest genome outcomes back into operator and family priors."""
    adaptation_rate = config.training.operator_adaptation_rate
    bias = _efficiency_bias(
        state.generation,
        config.evolution.num_generations,
        config.evolution.efficiency_bias_start,
        config.evolution.efficiency_bias_end,
        config.evolution.efficiency_warmup_generations,
    )
    for genome in state.population:
        genome_results = state.results.get(genome.genome_id, {})
        valid_results = [result for result in genome_results.values() if result.failure_reason is None]
        failures = sum(1 for result in genome_results.values() if result.failure_reason not in {None, "unsupported_benchmark"})
        avg_quality = (
            sum(result.quality for result in valid_results) / len(valid_results)
            if valid_results else float("-inf")
        )
        avg_time = (
            sum(result.train_seconds for result in valid_results) / len(valid_results)
            if valid_results else 0.0
        )
        avg_params = (
            sum(result.parameter_count for result in valid_results) / len(valid_results)
            if valid_results else float(genome.parameter_estimate)
        )
        efficiency_score = _efficiency_adjusted_value(
            avg_quality,
            avg_time,
            avg_params,
            bias,
            config.evolution.time_penalty_weight,
            config.evolution.param_penalty_weight,
        )

        family_bucket = state.family_stats.setdefault(
            genome.family,
            {
                "count": 0.0,
                "quality_sum": 0.0,
                "time_sum": 0.0,
                "param_sum": 0.0,
                "efficiency_sum": 0.0,
                "failures": 0.0,
            },
        )
        family_bucket["count"] += 1.0
        if valid_results:
            family_bucket["quality_sum"] += float(avg_quality)
            family_bucket["time_sum"] += float(avg_time)
            family_bucket["param_sum"] += float(avg_params)
            family_bucket["efficiency_sum"] += float(efficiency_score)
        family_bucket["failures"] += float(failures)

        operator = state.lineage_ops.get(genome.genome_id)
        if operator is None:
            continue
        operator_bucket = state.operator_stats.setdefault(
            operator,
            {
                "count": 0.0,
                "quality_sum": 0.0,
                "time_sum": 0.0,
                "param_sum": 0.0,
                "efficiency_sum": 0.0,
                "failures": 0.0,
            },
        )
        operator_bucket["count"] += 1.0
        if valid_results:
            operator_bucket["quality_sum"] += float(avg_quality) * adaptation_rate
            operator_bucket["time_sum"] += float(avg_time) * adaptation_rate
            operator_bucket["param_sum"] += float(avg_params) * adaptation_rate
            operator_bucket["efficiency_sum"] += float(efficiency_score) * adaptation_rate
        operator_bucket["failures"] += float(failures)


def _benchmark_priority_scores(state: GenerationState, benchmark_specs: list) -> dict[str, float]:
    scores: dict[str, float] = {}
    if not benchmark_specs:
        return scores

    all_ids = [spec.id if hasattr(spec, "id") else spec.name for spec in benchmark_specs]
    max_evals = max((state.benchmark_evaluations.get(bid, 0) for bid in all_ids), default=0)
    spreads = {}
    for bid in all_ids:
        history = state.benchmark_history.get(bid, [])
        spreads[bid] = float(np.std(history)) if len(history) >= 2 else 0.0
    max_spread = max(spreads.values(), default=0.0)

    for benchmark_id in all_ids:
        evals = state.benchmark_evaluations.get(benchmark_id, 0)
        failures = state.benchmark_failures.get(benchmark_id, 0)
        failure_rate = failures / evals if evals else 0.0
        coverage_gap = 1.0 - (evals / max_evals) if max_evals else 1.0
        spread_score = (spreads[benchmark_id] / max_spread) if max_spread > 1e-9 else 0.0
        scores[benchmark_id] = (coverage_gap * 1.25) + (failure_rate * 1.5) + (spread_score * 0.75)

    return scores


def _benchmark_epoch_multiplier(
    benchmark_id: str,
    priority_scores: dict[str, float],
    min_scale: float,
    max_scale: float,
) -> float:
    if not priority_scores:
        return 1.0
    score = priority_scores.get(benchmark_id, 0.0)
    max_score = max(priority_scores.values(), default=0.0)
    min_score = min(priority_scores.values(), default=0.0)
    if max_score <= min_score + 1e-9:
        return 1.0
    normalized = (score - min_score) / (max_score - min_score)
    return min_scale + normalized * (max_scale - min_scale)


def _genome_profiles(state: GenerationState) -> dict[str, dict[str, float]]:
    profiles: dict[str, dict[str, float]] = {}
    for genome in state.population:
        valid = [
            result for result in state.results.get(genome.genome_id, {}).values()
            if result.failure_reason is None
        ]
        if valid:
            profiles[genome.genome_id] = {
                "quality": sum(result.quality for result in valid) / len(valid),
                "time": sum(result.train_seconds for result in valid) / len(valid),
                "params": sum(result.parameter_count for result in valid) / len(valid),
            }
        else:
            profiles[genome.genome_id] = {
                "quality": 0.0,
                "time": 0.0,
                "params": float(genome.parameter_estimate),
            }
    return profiles


def _genome_epoch_multiplier(
    genome: ModelGenome,
    profiles: dict[str, dict[str, float]],
    generation: int,
    evolution,
    min_scale: float,
    max_scale: float,
) -> float:
    if not profiles:
        return 1.0
    bias = _efficiency_bias(
        generation,
        evolution.num_generations,
        evolution.efficiency_bias_start,
        evolution.efficiency_bias_end,
        evolution.efficiency_warmup_generations,
    )
    time_logs = [log1p(profile["time"]) for profile in profiles.values()]
    param_logs = [log1p(profile["params"]) for profile in profiles.values()]
    score_values = []
    for profile in profiles.values():
        time_penalty = _normalized_range(log1p(profile["time"]), time_logs)
        param_penalty = _normalized_range(log1p(profile["params"]), param_logs)
        total_weight = max(1e-9, evolution.time_penalty_weight + evolution.param_penalty_weight)
        efficiency_penalty = (
            (evolution.time_penalty_weight * time_penalty)
            + (evolution.param_penalty_weight * param_penalty)
        ) / total_weight
        score_values.append(profile["quality"] - (bias * efficiency_penalty))

    profile = profiles[genome.genome_id]
    time_penalty = _normalized_range(log1p(profile["time"]), time_logs)
    param_penalty = _normalized_range(log1p(profile["params"]), param_logs)
    total_weight = max(1e-9, evolution.time_penalty_weight + evolution.param_penalty_weight)
    efficiency_penalty = (
        (evolution.time_penalty_weight * time_penalty)
        + (evolution.param_penalty_weight * param_penalty)
    ) / total_weight
    score = profile["quality"] - (bias * efficiency_penalty)
    normalized = _normalized_range(score, score_values)
    return min_scale + (normalized * (max_scale - min_scale))


def _efficiency_adjusted_value(
    quality: float,
    train_seconds: float,
    parameter_count: float,
    bias: float,
    time_weight: float,
    param_weight: float,
) -> float:
    total_weight = max(1e-9, time_weight + param_weight)
    penalty = (
        (time_weight * log1p(max(0.0, train_seconds)))
        + (param_weight * (log1p(max(1.0, parameter_count)) / 10.0))
    ) / total_weight
    return ((1.0 - bias) * quality) - (bias * penalty)


def _efficiency_bias(
    generation: int,
    total_generations: int,
    start: float,
    end: float,
    warmup_generations: int,
) -> float:
    if total_generations <= 1:
        return end
    if generation < warmup_generations:
        return start
    active_total = max(1, total_generations - 1 - warmup_generations)
    progress = min(1.0, max(0.0, (generation - warmup_generations) / active_total))
    return start + ((end - start) * progress)


def _normalized_range(value: float, values: list[float]) -> float:
    if not values:
        return 0.5
    lo = min(values)
    hi = max(values)
    if hi <= lo + 1e-9:
        return 1.0
    return (value - lo) / (hi - lo)
