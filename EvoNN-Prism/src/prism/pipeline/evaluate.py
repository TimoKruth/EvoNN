"""Evaluation stage: compile genomes, train models, collect results."""

from __future__ import annotations

from dataclasses import dataclass, field
from random import Random

import numpy as np

from prism.genome import ModelGenome
from prism.storage import RunStore
from prism.runtime.cache import WeightCache
from prism.runtime.training import EvaluationResult, train_and_evaluate


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

            result = _evaluate_single(
                genome=genome,
                spec=spec,
                training=training,
                epoch_scale=epoch_scale * benchmark_scale,
                cache=cache,
                parent_ids=parent_ids,
            )

            genome_results[benchmark_id] = result
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
    # Load data before compile so LM output_dim can expand to observed token range.
    X_train, y_train, X_val, y_val = _load_data(spec, seed=42)
    output_dim = _resolve_output_dim(spec, X_train, y_train)

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
        cache.store(genome.genome_id, model)
    result.inherited_from = inherited_from

    return result


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
    adaptation_rate: float,
) -> None:
    """Fold latest genome outcomes back into operator and family priors."""
    for genome in state.population:
        genome_results = state.results.get(genome.genome_id, {})
        valid = [result.quality for result in genome_results.values() if result.failure_reason is None]
        failures = sum(1 for result in genome_results.values() if result.failure_reason is not None)
        avg_quality = sum(valid) / len(valid) if valid else float("-inf")

        family_bucket = state.family_stats.setdefault(
            genome.family, {"count": 0.0, "quality_sum": 0.0, "failures": 0.0},
        )
        family_bucket["count"] += 1.0
        if valid:
            family_bucket["quality_sum"] += float(avg_quality)
        family_bucket["failures"] += float(failures)

        operator = state.lineage_ops.get(genome.genome_id)
        if operator is None:
            continue
        operator_bucket = state.operator_stats.setdefault(
            operator, {"count": 0.0, "quality_sum": 0.0, "failures": 0.0},
        )
        operator_bucket["count"] += 1.0
        if valid:
            operator_bucket["quality_sum"] += float(avg_quality) * adaptation_rate
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
