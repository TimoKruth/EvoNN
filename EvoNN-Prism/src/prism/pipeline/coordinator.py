"""Main evolution loop: setup, evaluate, archive, reproduce, checkpoint."""

from __future__ import annotations

from collections import Counter
import importlib.metadata
import json
import os
import time
from pathlib import Path
from random import Random
from uuid import uuid4

from prism.config import RunConfig
from prism.genome import ModelGenome, apply_random_mutation, create_seed_genome
from prism.families.compiler import compatible_families
from prism.monitor import TerminalMonitor
from prism.pipeline.archive import (
    build_archives,
    summaries_from_state_results,
)
from prism.pipeline.evaluate import GenerationState, evaluate, select_benchmarks, update_search_memory
from prism.pipeline.reproduce import reproduce
from prism.runtime.cache import WeightCache
from prism.storage import RunStore


def run_evolution(
    config: RunConfig,
    benchmark_specs: list,
    run_dir: str | None = None,
    resume: bool = False,
) -> GenerationState:
    """Run full evolution. Returns final GenerationState.

    Args:
        config: RunConfig with training, evolution, and benchmark settings.
        benchmark_specs: List of benchmark specifications to evaluate against.
        run_dir: Optional directory for checkpoints and logs.
        resume: If True, attempt to resume from checkpoint in run_dir.

    Returns:
        Final GenerationState with results and archives.
    """
    config = _resolved_run_config(config, benchmark_specs)
    rng = Random(config.seed)
    evolution = config.evolution
    training = config.training

    # Setup run directory
    if run_dir is None:
        run_dir = f"runs/prism-{uuid4().hex[:8]}"
    os.makedirs(run_dir, exist_ok=True)
    os.makedirs(os.path.join(run_dir, "checkpoints"), exist_ok=True)
    run_id = Path(run_dir).name

    # Save config
    config_path = os.path.join(run_dir, "config.json")
    Path(config_path).write_text(
        json.dumps(config.model_dump(), indent=2),
        encoding="utf-8",
    )

    store = RunStore(Path(run_dir) / "metrics.duckdb")
    store.save_run(run_id, config.model_dump(mode="json"))

    # Weight cache
    cache = WeightCache() if training.weight_inheritance else None

    # Monitor
    monitor = TerminalMonitor()

    # Resume from checkpoint
    start_gen = 0
    state: GenerationState | None = None

    if resume:
        state, start_gen = _try_resume(run_dir)
        if state is not None:
            monitor.on_info(f"Resuming from generation {start_gen}")

    # Create seed population
    if state is None:
        prior_memory = _load_prior_run_memory(config.prior_run_dirs)
        population = _create_seed_population(evolution, rng, prior_genomes=prior_memory["genomes"])
        state = GenerationState(
            generation=0,
            population=population,
            parent_ids={g.genome_id: [] for g in population},
            operator_stats=prior_memory["operator_stats"],
            family_stats=prior_memory["family_stats"],
        )

    run_start = time.time()

    try:
        # Main evolution loop
        total_gens = evolution.num_generations
        for gen in range(start_gen, total_gens):
            state.generation = gen

            # 1. Select benchmarks (undercovered focus)
            gen_benchmarks = select_benchmarks(state, config, benchmark_specs, rng)

            # 2. Evaluate (with cache + multi-fidelity)
            state = evaluate(
                state,
                config,
                gen_benchmarks,
                cache,
                store=store,
                run_id=run_id,
            )

            # 3. Build archives (elite + pareto + niche)
            summaries = summaries_from_state_results(
                state.population, state.results, gen,
            )
            state.archives = build_archives(
                summaries,
                elite_per_benchmark=evolution.elite_per_benchmark,
            )
            _persist_archives(store, run_id, gen, state.archives)
            update_search_memory(state, config)

            # 4. Monitor
            elapsed = time.time() - run_start
            qualities = [s.aggregate_quality for s in summaries if s.qualities]
            best_q = max(qualities) if qualities else float("-inf")
            avg_q = sum(qualities) / len(qualities) if qualities else float("-inf")
            families = sorted({g.family for g in state.population})

            monitor.on_generation(
                gen=gen,
                total=total_gens,
                best_quality=best_q,
                avg_quality=avg_q,
                families_active=families,
                population_size=len(state.population),
                elapsed=elapsed,
            )

            # 5. Checkpoint
            _checkpoint(run_dir, gen, state)

            # 6. Reproduce (skip on last generation)
            if gen < total_gens - 1:
                offspring, lineage = reproduce(state, config, rng)
                _persist_lineage(store, run_id, gen + 1, lineage)

                # Replace population with offspring
                state.population = offspring
                state.parent_ids = {
                    record["genome_id"]: list(record.get("parent_ids", []))
                    for record in lineage
                }
                state.lineage_ops = {
                    record["genome_id"]: str(record.get("operator", "mutation"))
                    for record in lineage
                }

                # Clear per-genome results for the new population
                active_ids = {g.genome_id for g in state.population}
                state.results = {
                    gid: res for gid, res in state.results.items()
                    if gid in active_ids
                }

        # Completion
        elapsed = time.time() - run_start
        qualities = [
            s.aggregate_quality
            for s in summaries_from_state_results(state.population, state.results, total_gens - 1)
            if s.qualities
        ]
        best_q = max(qualities) if qualities else float("-inf")
        monitor.on_complete(best_q, total_gens, elapsed)

        # Write final summary
        _write_summary(run_dir, state, elapsed)

        return state
    finally:
        store.close()


def _create_seed_population(
    evolution,
    rng: Random,
    prior_genomes: list[ModelGenome] | None = None,
) -> list[ModelGenome]:
    """Create initial population: one genome per allowed family, then fill randomly."""
    allowed = evolution.allowed_families or ["mlp", "conv2d", "attention"]

    population: list[ModelGenome] = []
    seen_ids: set[str] = set()

    for genome in prior_genomes or []:
        if genome.family not in allowed or genome.genome_id in seen_ids:
            continue
        population.append(genome)
        seen_ids.add(genome.genome_id)
        if len(population) >= evolution.population_size:
            return population[: evolution.population_size]

    # One seed per family
    for family in allowed:
        genome = create_seed_genome(family, evolution, rng)
        if genome.genome_id in seen_ids:
            continue
        population.append(genome)
        seen_ids.add(genome.genome_id)

    # Fill remaining slots via mutation from diverse seeds
    while len(population) < evolution.population_size:
        parent = rng.choice(population)
        child = None
        for _ in range(32):
            candidate, _ = apply_random_mutation(parent, evolution, rng)
            if candidate.genome_id not in seen_ids:
                child = candidate
                break
            parent = rng.choice(population)
        if child is None:
            break
        population.append(child)
        seen_ids.add(child.genome_id)

    return population[: evolution.population_size]


def _checkpoint(run_dir: str, generation: int, state: GenerationState) -> None:
    """Save generation state to checkpoint file."""
    checkpoint = {
        "generation": generation,
        "population": [g.model_dump() for g in state.population],
        "parent_ids": state.parent_ids,
        "results": {
            genome_id: {
                bid: {
                    "metric_name": r.metric_name,
                    "metric_value": r.metric_value,
                    "quality": r.quality,
                    "parameter_count": r.parameter_count,
                    "train_seconds": r.train_seconds,
                    "failure_reason": r.failure_reason,
                    "inherited_from": r.inherited_from,
                }
                for bid, r in benchmark_results.items()
            }
            for genome_id, benchmark_results in state.results.items()
        },
        "total_evaluations": state.total_evaluations,
        "benchmark_history": state.benchmark_history,
        "benchmark_failures": state.benchmark_failures,
        "benchmark_evaluations": state.benchmark_evaluations,
        "lineage_ops": state.lineage_ops,
        "operator_stats": state.operator_stats,
        "family_stats": state.family_stats,
    }
    path = os.path.join(run_dir, "checkpoints", f"gen_{generation:04d}.json")
    Path(path).write_text(json.dumps(checkpoint, indent=2), encoding="utf-8")

    # Also write latest state
    latest = os.path.join(run_dir, "state.json")
    Path(latest).write_text(json.dumps(checkpoint, indent=2), encoding="utf-8")


def _try_resume(run_dir: str) -> tuple[GenerationState | None, int]:
    """Attempt to load state from checkpoint. Returns (state, next_gen)."""
    from prism.runtime.training import EvaluationResult

    state_path = os.path.join(run_dir, "state.json")
    if not os.path.exists(state_path):
        return None, 0

    try:
        data = json.loads(Path(state_path).read_text(encoding="utf-8"))
    except (json.JSONDecodeError, OSError):
        return None, 0

    population = [ModelGenome.model_validate(g) for g in data["population"]]

    results: dict[str, dict[str, EvaluationResult]] = {}
    for genome_id, benchmark_results in data.get("results", {}).items():
        results[genome_id] = {
            bid: EvaluationResult(**r_data)
            for bid, r_data in benchmark_results.items()
        }

    state = GenerationState(
        generation=data["generation"],
        population=population,
        results=results,
        total_evaluations=data.get("total_evaluations", 0),
        benchmark_history={
            benchmark_id: [float(value) for value in values]
            for benchmark_id, values in data.get("benchmark_history", {}).items()
        },
        benchmark_failures={
            benchmark_id: int(value)
            for benchmark_id, value in data.get("benchmark_failures", {}).items()
        },
        benchmark_evaluations={
            benchmark_id: int(value)
            for benchmark_id, value in data.get("benchmark_evaluations", {}).items()
        },
        lineage_ops={
            genome_id: str(operator)
            for genome_id, operator in data.get("lineage_ops", {}).items()
        },
        operator_stats={
            operator: {key: float(value) for key, value in payload.items()}
            for operator, payload in data.get("operator_stats", {}).items()
        },
        family_stats={
            family: {key: float(value) for key, value in payload.items()}
            for family, payload in data.get("family_stats", {}).items()
        },
        parent_ids={
            genome_id: list(parent_ids)
            for genome_id, parent_ids in data.get("parent_ids", {}).items()
        },
    )

    return state, data["generation"] + 1


def _write_summary(run_dir: str, state: GenerationState, elapsed: float) -> None:
    """Write a final summary.json with best results."""
    summaries = summaries_from_state_results(
        state.population, state.results, state.generation,
    )
    best = max(summaries, key=lambda s: s.aggregate_quality) if summaries else None
    runtime_backend, runtime_version, precision_mode = _runtime_metadata()
    evaluation_rows = [
        result
        for benchmark_results in state.results.values()
        for result in benchmark_results.values()
    ]
    failure_patterns = Counter(
        str(result.failure_reason)
        for result in evaluation_rows
        if result.failure_reason
    )

    summary = {
        "elapsed_seconds": elapsed,
        "total_evaluations": state.total_evaluations,
        "benchmarks_evaluated": len({benchmark_id for benchmark_results in state.results.values() for benchmark_id in benchmark_results}),
        "population_size": len(state.population),
        "best_genome_id": best.genome_id if best else None,
        "best_family": best.family if best else None,
        "best_quality": best.aggregate_quality if best else None,
        "best_parameter_count": best.parameter_count if best else None,
        "families_active": sorted({g.family for g in state.population}),
        "failure_count": sum(1 for result in evaluation_rows if result.failure_reason),
        "failure_patterns": dict(failure_patterns.most_common()),
        "runtime_backend": runtime_backend,
        "runtime_version": runtime_version,
        "precision_mode": precision_mode,
    }

    path = os.path.join(run_dir, "summary.json")
    Path(path).write_text(json.dumps(summary, indent=2), encoding="utf-8")


def _runtime_metadata() -> tuple[str, str | None, str]:
    """Return persisted runtime metadata for Prism run artifacts."""
    try:
        version = importlib.metadata.version("mlx")
    except importlib.metadata.PackageNotFoundError:
        version = None
    return "mlx", version, "fp32"


def _resolved_run_config(config: RunConfig, benchmark_specs: list) -> RunConfig:
    if config.evolution.allowed_families:
        return config

    allowed_sets: list[set[str]] = []
    for spec in benchmark_specs:
        modality = getattr(spec, "modality", "tabular")
        task = getattr(spec, "task", "classification")
        families = set(compatible_families(modality))
        if task == "language_modeling":
            families &= {"embedding", "attention", "sparse_attention"}
        if families:
            allowed_sets.append(families)

    if not allowed_sets:
        return config

    allowed = sorted(set.intersection(*allowed_sets))
    if not allowed:
        return config

    return config.model_copy(
        update={
            "evolution": config.evolution.model_copy(update={"allowed_families": allowed}),
        }
    )


def _persist_lineage(store: RunStore, run_id: str, generation: int, lineage: list[dict]) -> None:
    for record in lineage:
        parent_ids = record.get("parent_ids", [])
        operator = record.get("operator", "mutation")
        for parent_id in parent_ids or [None]:
            store.save_lineage(
                run_id,
                record["genome_id"],
                parent_id,
                generation,
                operator,
                operator_kind=operator.split(":", 1)[0],
            )


def _persist_archives(store: RunStore, run_id: str, generation: int, archives: dict) -> None:
    for summary in archives.get("pareto", []):
        store.save_archive(
            run_id, generation, "pareto", None, summary.genome_id, summary.aggregate_quality,
        )

    for benchmark_id, elites in archives.get("elite", {}).items():
        for summary in elites:
            store.save_archive(
                run_id, generation, "elite", benchmark_id, summary.genome_id, summary.aggregate_quality,
            )

    for family, summary in archives.get("niche", {}).items():
        store.save_archive(
            run_id, generation, f"niche:{family}", None, summary.genome_id, summary.aggregate_quality,
        )

    efficient_archive = archives.get("efficient", {})
    for family, summary in efficient_archive.get("family", {}).items():
        store.save_archive(
            run_id, generation, f"efficient:family:{family}", None, summary.genome_id, summary.aggregate_quality,
        )
    for benchmark_id, summaries in efficient_archive.get("benchmark", {}).items():
        for summary in summaries:
            store.save_archive(
                run_id, generation, f"efficient:{benchmark_id}", benchmark_id, summary.genome_id, summary.aggregate_quality,
            )

    for benchmark_id, specialists in archives.get("specialist", {}).items():
        for family, summary in specialists.items():
            store.save_archive(
                run_id,
                generation,
                f"specialist:{benchmark_id}:{family}",
                benchmark_id,
                summary.genome_id,
                summary.aggregate_quality,
            )


def _load_prior_run_memory(prior_run_dirs: list[str]) -> dict:
    genomes: list[ModelGenome] = []
    operator_stats: dict[str, dict[str, float]] = {}
    family_stats: dict[str, dict[str, float]] = {}

    for run_dir_str in prior_run_dirs:
        run_dir = Path(run_dir_str)
        db_path = run_dir / "metrics.duckdb"
        if not db_path.exists():
            continue
        store = RunStore(db_path)
        try:
            run_id = _resolve_store_run_id(store)
            genome_rows = store.load_genomes(run_id)
            evaluations = store.load_evaluations(run_id)
            lineage = store.load_lineage(run_id)
        finally:
            store.close()

        genome_map: dict[str, ModelGenome] = {}
        for row in genome_rows:
            try:
                genome = ModelGenome.model_validate(row)
            except Exception:
                continue
            genome_map[genome.genome_id] = genome
            genomes.append(genome)

        for row in evaluations:
            genome_id = row.get("genome_id", "")
            quality = row.get("quality")
            failure = row.get("failure_reason")
            genome = genome_map.get(genome_id)
            if genome is None:
                continue
            family_bucket = family_stats.setdefault(
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
            if quality is not None and failure is None:
                family_bucket["quality_sum"] += float(quality)
                family_bucket["time_sum"] += float(row.get("train_seconds") or 0.0)
                family_bucket["param_sum"] += float(row.get("parameter_count") or 0.0)
                family_bucket["efficiency_sum"] += float(quality)
            if failure is not None:
                family_bucket["failures"] += 1.0

        lineage_ops = {row["genome_id"]: str(row.get("mutation_summary") or "mutation") for row in lineage}
        for row in evaluations:
            operator = lineage_ops.get(row.get("genome_id", ""))
            if not operator:
                continue
            bucket = operator_stats.setdefault(
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
            bucket["count"] += 1.0
            if row.get("failure_reason") is None and row.get("quality") is not None:
                bucket["quality_sum"] += float(row["quality"])
                bucket["time_sum"] += float(row.get("train_seconds") or 0.0)
                bucket["param_sum"] += float(row.get("parameter_count") or 0.0)
                bucket["efficiency_sum"] += float(row["quality"])
            elif row.get("failure_reason") is not None:
                bucket["failures"] += 1.0

    unique_genomes: dict[str, ModelGenome] = {}
    for genome in genomes:
        unique_genomes.setdefault(genome.genome_id, genome)
    return {
        "genomes": list(unique_genomes.values()),
        "operator_stats": operator_stats,
        "family_stats": family_stats,
    }


def _resolve_store_run_id(store: RunStore) -> str:
    row = store.conn.execute(
        "SELECT run_id FROM runs ORDER BY created_at DESC LIMIT 1"
    ).fetchone()
    if row:
        return row[0]
    return "default"
