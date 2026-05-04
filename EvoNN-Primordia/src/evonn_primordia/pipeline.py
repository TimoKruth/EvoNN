"""MLX-backed primitive-first search runtime for Primordia."""
from __future__ import annotations

from datetime import datetime, timezone
import json
from pathlib import Path
import shutil
from random import Random
from types import SimpleNamespace
from time import perf_counter
from typing import Any

import numpy as np

from evonn_primordia.config import RunConfig
from evonn_primordia.export.report import build_primitive_bank_summary, write_report
from evonn_primordia.export.seeding import build_seed_candidates
from evonn_primordia.genome import ModelGenome
from evonn_primordia.objectives import candidate_signature, search_score
from evonn_primordia.runtime import backends as runtime_backends
from evonn_primordia.runtime.backends import (
    FALLBACK_LIMITATIONS,
    RuntimeBindings,
    resolve_runtime_bindings,
    runtime_execution_policy,
)
from evonn_primordia.search_state import CandidateSeed, EliteArchive
from evonn_primordia.status import load_checkpoint, write_checkpoint, write_status

BUDGET_POLICY_NAME = "prototype_equal_budget"
PRECISION_MODE = "fp32"
mlx = runtime_backends.mlx
_MLX_VERSION = getattr(mlx, "__version__", None) if mlx is not None else None


def run_search(
    config: RunConfig,
    *,
    run_dir: str | Path,
    config_path: str | Path | None = None,
) -> Path:
    """Run Primordia search with MLX-backed family evaluation."""

    runtime = _load_runtime_bindings(config)
    runtime_backend = getattr(runtime, "runtime_backend", "unknown")
    runtime_version = getattr(runtime, "runtime_version", None)
    precision_mode = getattr(runtime, "precision_mode", PRECISION_MODE)
    runtime_policy = runtime_execution_policy()
    run_dir = Path(run_dir)
    run_dir.mkdir(parents=True, exist_ok=True)
    if config_path is not None:
        shutil.copy2(config_path, run_dir / "config.yaml")

    run_id = run_dir.name
    run_name = config.run_name or run_id
    benchmarks = list(config.benchmark_pool.benchmarks)
    if not benchmarks:
        raise ValueError("Primordia requires at least one benchmark")

    target_evals = _target_evaluation_count(config)
    benchmark_total = len(benchmarks)
    executed_records: list[dict[str, Any]] = []
    best_results: list[dict[str, Any]] = []
    primitive_usage: dict[str, int] = {}
    benchmark_slot_plan: list[dict[str, Any]] = []
    group_counts = {"tabular": 0, "synthetic": 0, "image": 0, "language_modeling": 0}
    started_at = datetime.now(timezone.utc).isoformat()
    started_clock = perf_counter()
    checkpoint = load_checkpoint(run_dir)
    resumed = checkpoint is not None
    completed_benchmark_names: list[str] = []
    if checkpoint:
        executed_records = list(checkpoint.get("executed_records") or [])
        best_results = list(checkpoint.get("best_results") or [])
        primitive_usage = dict(checkpoint.get("primitive_usage") or {})
        benchmark_slot_plan = list(checkpoint.get("benchmark_slot_plan") or [])
        group_counts = dict(checkpoint.get("group_counts") or group_counts)
        started_at = str(checkpoint.get("started_at") or started_at)
        completed_benchmark_names = list(checkpoint.get("completed_benchmark_names") or [])
    write_status(
        run_dir,
        run_id=run_id,
        run_name=run_name,
        state="running",
        total_benchmarks=benchmark_total,
        completed_benchmarks=completed_benchmark_names,
        target_evaluation_count=target_evals,
        evaluation_count=len(executed_records),
        runtime_backend=runtime_backend,
    )

    _emit_progress(
        f"start run_id={run_id} benchmarks={benchmark_total} target_evals={target_evals} mode={config.search.mode} runtime={runtime_backend} resumed={resumed}"
    )
    for benchmark_index, benchmark_name in enumerate(benchmarks, start=1):
        if benchmark_name in completed_benchmark_names:
            _emit_progress(f"[{benchmark_index}/{benchmark_total}] skip benchmark={benchmark_name} reason=checkpoint")
            continue
        spec = runtime.get_benchmark(benchmark_name)
        group = runtime.benchmark_group(spec)
        group_counts[group] += 1
        modality = _modality_for_group(group)
        allowed_families = _allowed_families(runtime, config, group, modality)
        slots = _slots_for_benchmark(
            config,
            benchmark_index=benchmark_index - 1,
            benchmark_total=benchmark_total,
            primitive_count=len(allowed_families),
        )
        raw_slots = slots
        if config.search.max_candidates_per_benchmark is not None:
            slots = min(slots, max(1, int(config.search.max_candidates_per_benchmark)))
        benchmark_slot_plan.append(
            {
                "benchmark_name": benchmark_name,
                "benchmark_group": group,
                "raw_slots": raw_slots,
                "effective_slots": slots,
                "family_count": len(allowed_families),
            }
        )
        if slots <= 0:
            raise ValueError(f"No evaluation slots assigned to benchmark '{benchmark_name}'")

        _emit_progress(
            f"[{benchmark_index}/{benchmark_total}] benchmark={benchmark_name} group={group} families={len(allowed_families)} slots={slots}"
        )
        try:
            x_train, y_train, x_val, y_val = spec.load_data(seed=config.seed)
            input_shape = _input_shape_for_spec(spec, group)
            x_train_np, y_train_np, x_val_np, y_val_np = _prepare_arrays(
                spec=spec,
                group=group,
                input_shape=input_shape,
                x_train=x_train,
                y_train=y_train,
                x_val=x_val,
                y_val=y_val,
            )
        except Exception as exc:
            failed = _failed_record(
                spec=spec,
                benchmark_name=benchmark_name,
                benchmark_group=group,
                reason=str(exc),
                runtime_backend=runtime_backend,
                runtime_version=runtime_version,
                precision_mode=precision_mode,
            )
            executed_records.append(failed)
            best_results.append(dict(failed))
            _emit_progress(f"[{benchmark_index}/{benchmark_total}] load-failed benchmark={benchmark_name} reason={exc}")
            continue

        benchmark_records = _run_benchmark_search(
            runtime=runtime,
            config=config,
            spec=spec,
            benchmark_name=benchmark_name,
            benchmark_group=group,
            benchmark_index=benchmark_index,
            benchmark_total=benchmark_total,
            allowed_families=allowed_families,
            slots=slots,
            input_shape=input_shape,
            x_train_np=x_train_np,
            y_train_np=y_train_np,
            x_val_np=x_val_np,
            y_val_np=y_val_np,
            runtime_backend=runtime_backend,
            runtime_version=runtime_version,
            precision_mode=precision_mode,
        )
        executed_records.extend(benchmark_records)
        for record in benchmark_records:
            family = str(record.get("primitive_family") or "unknown")
            primitive_usage[family] = primitive_usage.get(family, 0) + 1
        best = _choose_best(spec.metric_direction, benchmark_records)
        best_results.append(dict(best))
        _emit_progress(
            f"[{benchmark_index}/{benchmark_total}] done benchmark={benchmark_name} best={best['primitive_name']} status={best['status']} metric={best['metric_value']}"
        )
        completed_benchmark_names.append(benchmark_name)
        write_checkpoint(
            run_dir,
            payload={
                "started_at": started_at,
                "executed_records": executed_records,
                "best_results": best_results,
                "primitive_usage": primitive_usage,
                "benchmark_slot_plan": benchmark_slot_plan,
                "group_counts": group_counts,
                "completed_benchmark_names": completed_benchmark_names,
            },
        )
        write_status(
            run_dir,
            run_id=run_id,
            run_name=run_name,
            state="running",
            total_benchmarks=benchmark_total,
            completed_benchmarks=completed_benchmark_names,
            target_evaluation_count=target_evals,
            evaluation_count=len(executed_records),
            runtime_backend=runtime_backend,
        )

    wall_clock_seconds = perf_counter() - started_clock
    failure_count = sum(1 for record in executed_records if record.get("status") != "ok")
    success_count = sum(1 for record in executed_records if record.get("status") == "ok")
    benchmark_leaders, family_leaders = _build_search_leader_surfaces(best_results, executed_records)
    summary = {
        "system": "primordia",
        "runtime": runtime_backend,
        "runtime_backend_requested": config.runtime.backend,
        "runtime_version": runtime_version,
        "runtime_backend_limitations": (
            FALLBACK_LIMITATIONS if runtime_backend == "numpy-fallback" else None
        ),
        "runtime_execution_policy": runtime_policy.as_metadata(resolved_backend=runtime_backend),
        "precision_mode": precision_mode,
        "run_id": run_id,
        "run_name": run_name,
        "status": "complete",
        "created_at": started_at,
        "seed": config.seed,
        "benchmark_count": benchmark_total,
        "completed_benchmarks": completed_benchmark_names,
        "evaluation_count": len(executed_records),
        "attempted_evaluations": len(executed_records),
        "successful_evaluations": success_count,
        "failed_evaluations": failure_count,
        "skipped_evaluations": 0,
        "resumed": resumed,
        "resumed_benchmark_count": len(completed_benchmark_names) if resumed else 0,
        "target_evaluation_count": target_evals,
        "epochs_per_candidate": config.training.epochs_per_candidate,
        "budget_policy_name": BUDGET_POLICY_NAME,
        "selection_mode": config.search.selection_mode,
        "primitive_search_policy": runtime_policy.primitive_search_policy,
        "seed_selection_policy": runtime_policy.seed_selection_policy,
        "search_policy": {
            "population_size": config.search.population_size,
            "elite_fraction": config.search.elite_fraction,
            "mutation_rounds_per_parent": config.search.mutation_rounds_per_parent,
            "family_exploration_floor": config.search.family_exploration_floor,
            "novelty_weight": config.search.novelty_weight,
            "complexity_penalty_weight": config.search.complexity_penalty_weight,
            "max_candidates_per_benchmark": config.search.max_candidates_per_benchmark,
        },
        "benchmark_slot_plan": benchmark_slot_plan,
        "benchmark_slot_integrity": _benchmark_slot_integrity(
            benchmark_slot_plan=benchmark_slot_plan,
            actual_evaluations=len(executed_records),
            target_evaluation_count=target_evals,
            failed_evaluations=failure_count,
        ),
        "benchmark_leaders": benchmark_leaders,
        "family_leaders": family_leaders,
        "primitive_usage": dict(sorted(primitive_usage.items(), key=lambda item: (-item[1], item[0]))),
        "group_counts": group_counts,
        "failure_count": failure_count,
        "wall_clock_seconds": wall_clock_seconds,
        "best_results": best_results,
    }
    primitive_bank_summary = build_primitive_bank_summary(
        summary=summary,
        best_results=best_results,
        trial_records=executed_records,
    )
    seed_candidates = build_seed_candidates(
        summary=summary,
        best_results=best_results,
        trial_records=executed_records,
        primitive_bank=primitive_bank_summary,
    )
    (run_dir / "trial_records.json").write_text(json.dumps(executed_records, indent=2), encoding="utf-8")
    (run_dir / "best_results.json").write_text(json.dumps(best_results, indent=2), encoding="utf-8")
    (run_dir / "summary.json").write_text(json.dumps(summary, indent=2), encoding="utf-8")
    (run_dir / "search_leaders.json").write_text(
        json.dumps({"benchmark_leaders": benchmark_leaders, "family_leaders": family_leaders}, indent=2),
        encoding="utf-8",
    )
    (run_dir / "primitive_bank_summary.json").write_text(json.dumps(primitive_bank_summary, indent=2), encoding="utf-8")
    (run_dir / "seed_candidates.json").write_text(json.dumps(seed_candidates, indent=2), encoding="utf-8")
    write_status(
        run_dir,
        run_id=run_id,
        run_name=run_name,
        state="complete",
        total_benchmarks=benchmark_total,
        completed_benchmarks=completed_benchmark_names,
        target_evaluation_count=target_evals,
        evaluation_count=len(executed_records),
        runtime_backend=runtime_backend,
    )
    write_checkpoint(
        run_dir,
        payload={
            "started_at": started_at,
            "executed_records": executed_records,
            "best_results": best_results,
            "primitive_usage": primitive_usage,
            "group_counts": group_counts,
            "completed_benchmark_names": completed_benchmark_names,
        },
    )
    write_report(run_dir)
    _emit_progress(
        f"finished run_id={run_id} evaluation_count={len(executed_records)} wall_clock_seconds={wall_clock_seconds:.3f}"
    )
    return run_dir


def _run_benchmark_search(
    *,
    runtime: RuntimeBindings,
    config: RunConfig,
    spec: Any,
    benchmark_name: str,
    benchmark_group: str,
    benchmark_index: int,
    benchmark_total: int,
    allowed_families: list[str],
    slots: int,
    input_shape: list[int],
    x_train_np: np.ndarray,
    y_train_np: np.ndarray,
    x_val_np: np.ndarray,
    y_val_np: np.ndarray,
    runtime_backend: str,
    runtime_version: str | None,
    precision_mode: str,
) -> list[dict[str, Any]]:
    population_size = config.search.population_size or len(allowed_families)
    population_size = max(1, min(population_size, slots))
    rng = Random(config.seed + benchmark_index * 1009)
    archive = EliteArchive(config.search.elite_fraction)
    benchmark_records: list[dict[str, Any]] = []
    seen_signatures: set[str] = set()
    generation = 0
    pending: list[CandidateSeed] = _spawn_initial_candidates(
        runtime=runtime,
        config=config,
        allowed_families=allowed_families,
        count=population_size,
    )

    while len(benchmark_records) < slots and pending:
        current_batch = pending[: max(1, min(population_size, slots - len(benchmark_records)))]
        pending = []
        for local_index, seed in enumerate(current_batch):
            record = _evaluate_candidate(
                runtime=runtime,
                config=config,
                spec=spec,
                benchmark_name=benchmark_name,
                benchmark_group=benchmark_group,
                benchmark_index=benchmark_index,
                input_shape=input_shape,
                x_train_np=x_train_np,
                y_train_np=y_train_np,
                x_val_np=x_val_np,
                y_val_np=y_val_np,
                runtime_backend=runtime_backend,
                runtime_version=runtime_version,
                precision_mode=precision_mode,
                genome=seed.genome,
                generation=seed.generation,
                parent_genome_id=seed.parent_genome_id,
                mutation_operator=seed.mutation_operator,
                slot_index=len(benchmark_records),
                seed_value=config.seed + benchmark_index * 1009 + len(benchmark_records),
                seen_signatures=seen_signatures,
                selection_mode=config.search.selection_mode,
                novelty_weight=config.search.novelty_weight,
                complexity_penalty_weight=config.search.complexity_penalty_weight,
            )
            benchmark_records.append(record)
            archive.update(record)
            seen_signatures.add(candidate_signature(record))
            if len(benchmark_records) >= slots:
                break

        generation += 1
        remaining = slots - len(benchmark_records)
        if remaining <= 0:
            break
        parents = archive.sample_parent_records(
            count=max(1, min(population_size, remaining)),
            total_budget=slots,
            rng=rng,
            family_exploration_floor=config.search.family_exploration_floor,
        )
        pending = _spawn_offspring(
            runtime=runtime,
            config=config,
            allowed_families=allowed_families,
            parents=parents,
            generation=generation,
            count=max(1, min(population_size, remaining)),
        )

    return benchmark_records


def _spawn_initial_candidates(
    *,
    runtime: RuntimeBindings,
    config: RunConfig,
    allowed_families: list[str],
    count: int,
) -> list[CandidateSeed]:
    seeds: list[CandidateSeed] = []
    for index in range(count):
        family = allowed_families[index % len(allowed_families)]
        genome = runtime.create_seed_genome(
            family,
            config.search.seed_hidden_width,
            config.search.seed_hidden_layers,
        )
        seeds.append(CandidateSeed(genome=genome, generation=0, parent_genome_id=None, mutation_operator=None))
    return seeds


def _spawn_offspring(
    *,
    runtime: RuntimeBindings,
    config: RunConfig,
    allowed_families: list[str],
    parents: list[dict[str, Any]],
    generation: int,
    count: int,
) -> list[CandidateSeed]:
    offspring: list[CandidateSeed] = []
    for index in range(count):
        parent = parents[index % len(parents)]
        genome = _rebuild_parent_genome(
            parent,
            fallback_family=allowed_families[index % len(allowed_families)],
            config=config,
        )
        parent_id = str(parent.get("genome_id")) if parent.get("genome_id") is not None else None
        mutation_label = None
        for mutation_round in range(max(1, config.search.mutation_rounds_per_parent)):
            mutated = runtime.mutate_genome(genome, generation * 1000 + index * 10 + mutation_round + 1, allowed_families, config)
            if isinstance(mutated, tuple):
                genome, mutation_label = mutated
            else:
                genome = mutated
                mutation_label = None
        offspring.append(
            CandidateSeed(
                genome=genome,
                generation=generation,
                parent_genome_id=parent_id,
                mutation_operator=mutation_label,
            )
        )
    return offspring


def _evaluate_candidate(
    *,
    runtime: RuntimeBindings,
    config: RunConfig,
    spec: Any,
    benchmark_name: str,
    benchmark_group: str,
    benchmark_index: int,
    input_shape: list[int],
    x_train_np: np.ndarray,
    y_train_np: np.ndarray,
    x_val_np: np.ndarray,
    y_val_np: np.ndarray,
    runtime_backend: str,
    runtime_version: str | None,
    precision_mode: str,
    genome: Any,
    generation: int,
    parent_genome_id: str | None,
    mutation_operator: str | None,
    slot_index: int,
    seed_value: int,
    seen_signatures: set[str],
    selection_mode: str,
    novelty_weight: float,
    complexity_penalty_weight: float,
) -> dict[str, Any]:
    family = str(getattr(genome, "family", "unknown"))
    primitive_label = family if generation == 0 else f"{family}@r{generation + 1}"

    try:
        compiled = runtime.compile_genome(
            genome,
            input_shape,
            _resolved_output_dim_for_spec(
                spec,
                x_train=x_train_np,
                y_train=y_train_np,
                x_val=x_val_np,
                y_val=y_val_np,
            ),
            _modality_for_group(benchmark_group),
            spec.task,
        )
        try:
            result = runtime.train_and_evaluate(
                compiled.model,
                x_train_np,
                y_train_np,
                x_val_np,
                y_val_np,
                task=spec.task,
                epochs=config.training.epochs_per_candidate,
                lr=getattr(genome, "learning_rate", config.training.learning_rate),
                batch_size=config.training.batch_size,
                parameter_count=compiled.parameter_count,
                weight_decay=getattr(genome, "weight_decay", config.training.weight_decay),
            )
        except TypeError as exc:
            if "weight_decay" not in str(exc):
                raise
            result = runtime.train_and_evaluate(
                compiled.model,
                x_train_np,
                y_train_np,
                x_val_np,
                y_val_np,
                task=spec.task,
                epochs=config.training.epochs_per_candidate,
                lr=getattr(genome, "learning_rate", config.training.learning_rate),
                batch_size=config.training.batch_size,
                parameter_count=compiled.parameter_count,
            )
        record = {
            "benchmark_name": benchmark_name,
            "benchmark_group": benchmark_group,
            "primitive_name": primitive_label,
            "primitive_family": family,
            "metric_name": result.metric_name,
            "metric_direction": spec.metric_direction,
            "metric_value": result.metric_value,
            "quality": result.quality,
            "parameter_count": result.parameter_count,
            "train_seconds": result.train_seconds,
            "architecture_summary": _architecture_summary(genome),
            "genome_id": getattr(genome, "genome_id", primitive_label),
            "genome_payload": _serialize_genome_payload(genome),
            "status": "ok" if result.failure_reason is None else "failed",
            "failure_reason": result.failure_reason,
            "seed": seed_value,
            "slot_index": slot_index,
            "runtime": runtime_backend,
            "runtime_version": runtime_version,
            "precision_mode": precision_mode,
            "generation": generation,
            "parent_genome_id": parent_genome_id,
            "mutation_operator": mutation_operator,
        }
    except Exception as exc:
        record = {
            "benchmark_name": benchmark_name,
            "benchmark_group": benchmark_group,
            "primitive_name": primitive_label,
            "primitive_family": family,
            "metric_name": spec.metric_name,
            "metric_direction": spec.metric_direction,
            "metric_value": None,
            "quality": float("-inf") if spec.metric_direction == "max" else float("inf"),
            "parameter_count": getattr(genome, "parameter_estimate", 0),
            "train_seconds": 0.0,
            "architecture_summary": _architecture_summary(genome),
            "genome_id": getattr(genome, "genome_id", primitive_label),
            "genome_payload": _serialize_genome_payload(genome),
            "status": "failed",
            "failure_reason": str(exc),
            "seed": seed_value,
            "slot_index": slot_index,
            "runtime": runtime_backend,
            "runtime_version": runtime_version,
            "precision_mode": precision_mode,
            "generation": generation,
            "parent_genome_id": parent_genome_id,
            "mutation_operator": mutation_operator,
        }

    score_fields = search_score(
        record,
        benchmark_group=benchmark_group,
        novelty_weight=novelty_weight,
        complexity_penalty_weight=complexity_penalty_weight,
        seen_signatures=seen_signatures,
        selection_mode=selection_mode,
    )
    record.update(score_fields)
    return record


def _load_runtime_bindings(config: RunConfig | None = None) -> RuntimeBindings:
    runtime_backends.mlx = mlx
    if config is None:
        config = SimpleNamespace(runtime=SimpleNamespace(backend="auto", allow_fallback=True))
    return resolve_runtime_bindings(config)


def _benchmark_slot_integrity(
    *,
    benchmark_slot_plan: list[dict[str, Any]],
    actual_evaluations: int,
    target_evaluation_count: int,
    failed_evaluations: int,
) -> dict[str, int | bool | str]:
    planned_slots = sum(int(row.get("effective_slots", 0)) for row in benchmark_slot_plan)
    matches = planned_slots == int(actual_evaluations) == int(target_evaluation_count)
    return {
        "status": "complete" if matches else "mismatch",
        "planned_slots": planned_slots,
        "completed_slots": int(actual_evaluations),
        "failed_slots": int(failed_evaluations),
        "matches_evaluation_count": matches,
    }


def _serialize_genome_payload(genome: Any) -> dict[str, Any]:
    if hasattr(genome, "model_dump"):
        return dict(genome.model_dump())
    payload = {
        "family": getattr(genome, "family", "mlp"),
        "hidden_layers": list(getattr(genome, "hidden_layers", [64])),
        "activation": getattr(genome, "activation", "relu"),
        "dropout": getattr(genome, "dropout", 0.0),
        "residual": getattr(genome, "residual", False),
        "activation_sparsity": getattr(genome, "activation_sparsity", 0.0),
        "learning_rate": getattr(genome, "learning_rate", 1e-3),
        "kernel_size": getattr(genome, "kernel_size", 3),
        "embedding_dim": getattr(genome, "embedding_dim", 64),
        "num_heads": getattr(genome, "num_heads", 4),
        "norm_type": getattr(genome, "norm_type", "none"),
        "weight_decay": getattr(genome, "weight_decay", 0.0),
        "num_experts": getattr(genome, "num_experts", 0),
        "moe_top_k": getattr(genome, "moe_top_k", 2),
    }
    return payload


def _rebuild_parent_genome(parent: dict[str, Any], *, fallback_family: str, config: RunConfig) -> ModelGenome:
    payload = dict(parent.get("genome_payload") or {})
    if not payload:
        payload = {
            "family": str(parent.get("primitive_family") or fallback_family),
            "hidden_layers": [config.search.seed_hidden_width] * config.search.seed_hidden_layers,
        }
    payload.setdefault("family", str(parent.get("primitive_family") or fallback_family))
    payload.setdefault("hidden_layers", [config.search.seed_hidden_width] * config.search.seed_hidden_layers)
    return ModelGenome.model_validate(payload)


def _build_search_leader_surfaces(
    best_results: list[dict[str, Any]], trial_records: list[dict[str, Any]]
) -> tuple[list[dict[str, Any]], list[dict[str, Any]]]:
    benchmark_leaders: list[dict[str, Any]] = []
    for record in best_results:
        benchmark_leaders.append(
            {
                "benchmark_name": record.get("benchmark_name"),
                "benchmark_group": record.get("benchmark_group"),
                "leader_family": record.get("primitive_family"),
                "leader_genome_id": record.get("genome_id"),
                "leader_generation": record.get("generation"),
                "leader_search_score": record.get("search_score"),
                "metric_name": record.get("metric_name"),
                "metric_value": record.get("metric_value"),
                "status": record.get("status"),
                "parent_genome_id": record.get("parent_genome_id"),
            }
        )

    wins_by_family: dict[str, set[str]] = {}
    for record in best_results:
        if record.get("status") != "ok":
            continue
        family = str(record.get("primitive_family") or "unknown")
        benchmark_name = str(record.get("benchmark_name") or "")
        if benchmark_name:
            wins_by_family.setdefault(family, set()).add(benchmark_name)

    family_records: dict[str, list[dict[str, Any]]] = {}
    for record in trial_records:
        family = str(record.get("primitive_family") or "unknown")
        family_records.setdefault(family, []).append(record)

    family_leaders: list[dict[str, Any]] = []
    for family, records in family_records.items():
        representative = max(
            records,
            key=lambda row: (
                float(row.get("search_score")) if row.get("search_score") is not None else float("-inf"),
                float(row.get("quality")) if row.get("quality") is not None else float("-inf"),
                int(row.get("generation") or 0),
            ),
        )
        wins = sorted(wins_by_family.get(family, set()))
        family_leaders.append(
            {
                "family": family,
                "evaluation_count": len(records),
                "benchmark_wins": len(wins),
                "supporting_benchmarks": wins,
                "benchmark_groups": sorted({str(row.get("benchmark_group") or "unknown") for row in records if row.get("benchmark_group")}),
                "best_generation": representative.get("generation"),
                "best_search_score": representative.get("search_score"),
                "best_metric_name": representative.get("metric_name"),
                "best_metric_value": representative.get("metric_value"),
                "representative_genome_id": representative.get("genome_id"),
                "representative_architecture_summary": representative.get("architecture_summary"),
            }
        )

    family_leaders.sort(
        key=lambda row: (
            int(row.get("benchmark_wins") or 0),
            float(row.get("best_search_score")) if row.get("best_search_score") is not None else float("-inf"),
            -int(row.get("evaluation_count") or 0),
            str(row.get("family") or "unknown"),
        ),
        reverse=True,
    )
    return benchmark_leaders, family_leaders


def _allowed_families(runtime: RuntimeBindings, config: RunConfig, group: str, modality: str) -> list[str]:
    configured = _primitive_names_for_group(config, group)
    compatible = set(runtime.compatible_families(modality))
    allowed = [family for family in configured if family in compatible]
    if not allowed:
        raise ValueError(f"No compatible Primordia families configured for group '{group}' and modality '{modality}'")
    return allowed


def _modality_for_group(group: str) -> str:
    if group == "image":
        return "image"
    if group == "language_modeling":
        return "text"
    return "tabular"


def _input_shape_for_spec(spec: Any, group: str) -> list[int]:
    if group == "image":
        return list(spec.resolved_image_shape)
    if group == "language_modeling":
        return [int(spec.model_input_dim)]
    return [int(spec.model_input_dim)]


def _output_dim_for_spec(spec: Any) -> int:
    return int(spec.model_output_dim)


def _resolved_output_dim_for_spec(
    spec: Any,
    *,
    x_train: np.ndarray,
    y_train: np.ndarray,
    x_val: np.ndarray,
    y_val: np.ndarray,
) -> int:
    output_dim = _output_dim_for_spec(spec)
    if spec.task != "language_modeling":
        return output_dim

    max_token_id = max(
        int(np.max(x_train)),
        int(np.max(y_train)),
        int(np.max(x_val)),
        int(np.max(y_val)),
    )
    return max(output_dim, max_token_id + 1)


def _prepare_arrays(
    *,
    spec: Any,
    group: str,
    input_shape: list[int],
    x_train,
    y_train,
    x_val,
    y_val,
) -> tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    x_train_np = np.asarray(x_train)
    y_train_np = np.asarray(y_train)
    x_val_np = np.asarray(x_val)
    y_val_np = np.asarray(y_val)
    if group == "image":
        x_train_np = x_train_np.reshape((x_train_np.shape[0], *input_shape)).astype(np.float32)
        x_val_np = x_val_np.reshape((x_val_np.shape[0], *input_shape)).astype(np.float32)
    elif spec.task == "language_modeling":
        x_train_np = x_train_np.astype(np.int32)
        y_train_np = y_train_np.astype(np.int32)
        x_val_np = x_val_np.astype(np.int32)
        y_val_np = y_val_np.astype(np.int32)
    else:
        x_train_np = x_train_np.astype(np.float32)
        y_train_np = y_train_np.astype(np.float32 if spec.task == "regression" else np.int32)
        x_val_np = x_val_np.astype(np.float32)
        y_val_np = y_val_np.astype(np.float32 if spec.task == "regression" else np.int32)
    return x_train_np, y_train_np, x_val_np, y_val_np


def _target_evaluation_count(config: RunConfig) -> int:
    family_count = sum(
        len(names)
        for names in (
            config.primitive_pool.tabular,
            config.primitive_pool.synthetic,
            config.primitive_pool.image,
            config.primitive_pool.language_modeling,
        )
    )
    default_budget = max(len(config.benchmark_pool.benchmarks), family_count)
    return config.search.target_evaluation_count or default_budget


def _slots_for_benchmark(
    config: RunConfig,
    *,
    benchmark_index: int,
    benchmark_total: int,
    primitive_count: int,
) -> int:
    if primitive_count <= 0:
        raise ValueError("No primitive candidates configured")
    if config.search.mode == "fixed_pool":
        return primitive_count
    target = _target_evaluation_count(config)
    if target < benchmark_total:
        raise ValueError(
            "Primordia budget_matched mode requires target_evaluation_count >= benchmark_count "
            f"({target} < {benchmark_total})"
        )
    base_slots = target // benchmark_total
    remainder = target % benchmark_total
    return base_slots + (1 if benchmark_index < remainder else 0)


def _primitive_names_for_group(config: RunConfig, group: str) -> list[str]:
    if group == "synthetic":
        return config.primitive_pool.synthetic
    if group == "image":
        return config.primitive_pool.image
    if group == "language_modeling":
        return config.primitive_pool.language_modeling
    return config.primitive_pool.tabular


def _failed_record(
    *,
    spec: Any,
    benchmark_name: str,
    benchmark_group: str,
    reason: str,
    runtime_backend: str,
    runtime_version: str | None,
    precision_mode: str,
) -> dict[str, Any]:
    return {
        "benchmark_name": benchmark_name,
        "benchmark_group": benchmark_group,
        "primitive_name": "load_failed",
        "primitive_family": "load_failed",
        "metric_name": spec.metric_name,
        "metric_direction": spec.metric_direction,
        "metric_value": None,
        "quality": None,
        "parameter_count": 0,
        "train_seconds": 0.0,
        "architecture_summary": "load_failed",
        "genome_id": f"{benchmark_name}:load_failed",
        "status": "failed",
        "failure_reason": reason,
        "seed": 0,
        "slot_index": 0,
        "runtime": runtime_backend,
        "runtime_version": runtime_version,
        "precision_mode": precision_mode,
    }


def _choose_best(metric_direction: str, records: list[dict[str, Any]]) -> dict[str, Any]:
    ok_records = [record for record in records if record["status"] == "ok" and record["metric_value"] is not None]
    if not ok_records:
        return records[0]
    reverse = metric_direction == "max"
    return sorted(ok_records, key=lambda record: record["metric_value"], reverse=reverse)[0]


def _architecture_summary(genome: Any) -> str:
    hidden_layers = getattr(genome, "hidden_layers", [])
    widths = "x".join(str(width) for width in hidden_layers) if hidden_layers else "none"
    return f"{getattr(genome, 'family', 'unknown')}[{widths}]"




def _emit_progress(message: str) -> None:
    print(f"[primordia] {message}", flush=True)
