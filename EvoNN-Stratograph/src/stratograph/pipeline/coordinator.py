"""Prototype run coordinator for Stratograph."""

from __future__ import annotations

from datetime import datetime, timezone
import json
from pathlib import Path
import random
import shutil
import time
from collections import Counter

from stratograph.benchmarks import get_benchmark
from stratograph.config import BenchmarkPoolConfig, EvolutionConfig, RunConfig, TrainingConfig
from stratograph.export.report import write_report
from stratograph.genome import HierarchicalGenome, genome_to_dict
from stratograph.genome.models import (
    ActivationKind,
    CellEdgeGene,
    CellGene,
    CellNodeGene,
    MacroEdgeGene,
    MacroNodeGene,
    PrimitiveKind,
)
from stratograph.pipeline.evaluator import (
    EvaluationOutcome,
    EvaluationRecord,
    TrainingArtifact,
    evaluate_candidate_with_state,
)
from stratograph.runtime.backends import resolve_runtime_backend_with_policy
from stratograph.search import crossover_genomes, descriptor, mutate_genome, niche_key, novelty_score
from stratograph.search.operators import MOTIF_LIBRARY
from stratograph.storage import RunStore


PRECISION_MODE = "fp32"


def run_evolution(
    config: RunConfig,
    *,
    run_dir: str | Path,
    config_path: str | Path | None = None,
    resume: bool = False,
    variant: str | None = None,
) -> Path:
    """Run Stratograph search/evaluation."""
    config = _normalize_config(config)
    started = time.perf_counter()
    run_dir = Path(run_dir)
    run_dir.mkdir(parents=True, exist_ok=True)
    checkpoint_path = run_dir / "checkpoint.json"
    status_path = run_dir / "status.json"
    if (run_dir / "metrics.duckdb").exists() and not resume:
        raise FileExistsError(f"Run directory already has metrics; use --resume or new run dir: {run_dir}")
    if config_path is not None:
        shutil.copy2(config_path, run_dir / "config.yaml")

    run_id = run_dir.name
    created_at = datetime.now(timezone.utc).isoformat()
    architecture_mode = variant or config.evolution.architecture_mode
    runtime_selection = resolve_runtime_backend_with_policy(
        config.runtime.backend,
        allow_fallback=config.runtime.allow_fallback,
    )
    variant_policy = _variant_policy(architecture_mode)
    completed_benchmarks: set[str] = set()
    if resume and checkpoint_path.exists():
        checkpoint = json.loads(checkpoint_path.read_text(encoding="utf-8"))
        completed_benchmarks = set(checkpoint.get("completed_benchmarks", []))
    store = RunStore(run_dir / "metrics.duckdb")
    store.record_run(
        run_id=run_id,
        run_name=config.run_name or run_id,
        created_at=created_at,
        seed=config.seed,
        config=config.model_dump(mode="json"),
    )

    benchmark_names = config.benchmark_pool.benchmarks
    evaluations_per_benchmark = config.evolution.population_size * config.evolution.generations
    novelty_scores: list[float] = []
    occupied_niches: set[tuple[int, int, int, int]] = set()
    _write_status(
        status_path,
        run_id=run_id,
        architecture_mode=architecture_mode,
        total_benchmarks=len(benchmark_names),
        completed_benchmarks=sorted(completed_benchmarks),
        state="running",
    )
    for benchmark_name in benchmark_names:
        if benchmark_name in completed_benchmarks:
            continue
        spec = get_benchmark(benchmark_name)
        best_genome: HierarchicalGenome | None = None
        best_result: EvaluationRecord | None = None
        try:
            data = spec.load_data(seed=config.seed)
        except Exception as exc:
            failed_genome = _make_candidate(
                benchmark_name=benchmark_name,
                task=spec.task,
                input_dim=spec.model_input_dim,
                output_dim=spec.model_output_dim,
                seed=config.seed,
                candidate_index=0,
                architecture_mode=architecture_mode,
            )
            failed_genome = _enforce_architecture_mode(failed_genome, architecture_mode)
            store.record_genome(
                run_id=run_id,
                generation=0,
                genome_id=failed_genome.genome_id,
                benchmark_name=benchmark_name,
                payload=genome_to_dict(failed_genome),
                architecture_summary="load_failed",
                parameter_count=0,
            )
            store.record_result(
                run_id=run_id,
                benchmark_name=benchmark_name,
                record={
                    "metric_name": spec.metric_name,
                    "metric_direction": spec.metric_direction,
                    "metric_value": None,
                    "quality": None,
                    "parameter_count": 0,
                    "train_seconds": 0.0,
                    "architecture_summary": "load_failed",
                    "genome_id": failed_genome.genome_id,
                    "status": "failed",
                    "failure_reason": str(exc),
                },
            )
            continue
        benchmark_archive: list[tuple[float, float, float, float]] = []
        population = [
            _enforce_architecture_mode(
                _make_candidate(
                    benchmark_name=benchmark_name,
                    task=spec.task,
                    input_dim=spec.model_input_dim,
                    output_dim=spec.model_output_dim,
                    seed=config.seed,
                    candidate_index=index,
                    architecture_mode=architecture_mode,
                ),
                architecture_mode,
            )
            for index in range(config.evolution.population_size)
        ]
        inherited_states: dict[str, TrainingArtifact | None] = {genome.genome_id: None for genome in population}
        for generation in range(config.evolution.generations):
            evaluated: list[tuple[HierarchicalGenome, EvaluationRecord, float]] = []
            trained_states: dict[str, TrainingArtifact | None] = {}
            for genome in population:
                try:
                    outcome = evaluate_candidate_with_state(
                        genome,
                        spec,
                        data=data,
                        inherited_state=inherited_states.get(genome.genome_id),
                        epochs=config.training.epochs,
                        batch_size=config.training.batch_size,
                        learning_rate=config.training.learning_rate,
                        runtime_backend=runtime_selection.resolved_backend,
                    )
                    result = outcome.record
                    trained_states[genome.genome_id] = outcome.training_artifact
                except Exception as exc:
                    result = EvaluationRecord(
                        metric_value=float("nan"),
                        quality=float("-inf") if spec.metric_direction == "max" else float("inf"),
                        parameter_count=0,
                        train_seconds=0.0,
                        architecture_summary="",
                        genome_id=genome.genome_id,
                        status="failed",
                        failure_reason=str(exc),
                    )
                    trained_states[genome.genome_id] = None
                desc = descriptor(genome)
                score = novelty_score(desc, benchmark_archive)
                benchmark_archive.append(desc)
                novelty_scores.append(score)
                occupied_niches.add(niche_key(desc))
                evaluated.append((genome, result, score))
                if _is_better(spec.metric_direction, result, best_result):
                    best_genome = genome
                    best_result = result

            if best_genome is not None and best_result is not None:
                store.record_genome(
                    run_id=run_id,
                    generation=generation,
                    genome_id=best_genome.genome_id,
                    benchmark_name=benchmark_name,
                    payload=genome_to_dict(best_genome),
                    architecture_summary=best_result.architecture_summary,
                    parameter_count=best_result.parameter_count,
                )
            if generation == config.evolution.generations - 1:
                continue
            population, inherited_states = _next_population(
                evaluated=evaluated,
                benchmark_name=benchmark_name,
                task=spec.task,
                input_dim=spec.model_input_dim,
                output_dim=spec.model_output_dim,
                seed=config.seed,
                generation=generation,
                population_size=config.evolution.population_size,
                architecture_mode=architecture_mode,
                allow_clone_mutation=bool(variant_policy["allow_clone_mutation"]),
                motif_bias=bool(variant_policy["motif_bias"]),
                trained_states=trained_states,
            )

        if best_genome is None or best_result is None:
            continue
        store.record_result(
            run_id=run_id,
            benchmark_name=benchmark_name,
            record={
                "metric_name": spec.metric_name,
                "metric_direction": spec.metric_direction,
                "metric_value": None if best_result.status != "ok" else best_result.metric_value,
                "quality": None if best_result.status != "ok" else best_result.quality,
                "parameter_count": best_result.parameter_count,
                "train_seconds": best_result.train_seconds,
                "architecture_summary": best_result.architecture_summary,
                "genome_id": best_result.genome_id,
                "status": best_result.status,
                "failure_reason": best_result.failure_reason,
            },
        )
        _write_best_genome(run_dir, benchmark_name, best_genome, best_result, architecture_mode)
        completed_benchmarks.add(benchmark_name)
        _write_checkpoint(
            checkpoint_path,
            run_id=run_id,
            architecture_mode=architecture_mode,
            completed_benchmarks=sorted(completed_benchmarks),
            total_benchmarks=len(benchmark_names),
        )
        _write_status(
            status_path,
            run_id=run_id,
            architecture_mode=architecture_mode,
            total_benchmarks=len(benchmark_names),
            completed_benchmarks=sorted(completed_benchmarks),
            state="running",
        )

    evaluation_count = config.evolution.population_size * config.evolution.generations * len(benchmark_names)
    wall_clock_seconds = time.perf_counter() - started
    store.save_budget_metadata(
        run_id=run_id,
        payload={
            "evaluation_count": evaluation_count,
            "effective_training_epochs": config.training.epochs,
            "wall_clock_seconds": wall_clock_seconds,
            "created_at": created_at,
            "runtime_backend_requested": runtime_selection.requested_backend,
            "runtime_backend": runtime_selection.resolved_backend,
            "runtime_version": runtime_selection.runtime_version,
            "runtime_backend_limitations": runtime_selection.backend_limitations,
            "precision_mode": PRECISION_MODE,
            "architecture_mode": architecture_mode,
            "allow_clone_mutation": variant_policy["allow_clone_mutation"],
            "motif_bias": variant_policy["motif_bias"],
            "novelty_archive_final_size": len(novelty_scores),
            "novelty_score_mean": (sum(novelty_scores) / len(novelty_scores)) if novelty_scores else 0.0,
            "novelty_score_max": max(novelty_scores, default=0.0),
            "map_elites_occupied_niches": len(occupied_niches),
            "map_elites_total_niches": max(1, len(occupied_niches)),
            "map_elites_fill_ratio": 1.0 if occupied_niches else 0.0,
            "qd_enabled": True,
        },
    )
    store.close()
    write_report(run_dir)
    _write_status(
        status_path,
        run_id=run_id,
        architecture_mode=architecture_mode,
        total_benchmarks=len(benchmark_names),
        completed_benchmarks=sorted(completed_benchmarks),
        state="completed",
    )
    return run_dir


def _make_candidate(
    *,
    benchmark_name: str,
    task: str,
    input_dim: int,
    output_dim: int,
    seed: int,
    candidate_index: int,
    architecture_mode: str = "two_level_shared",
) -> HierarchicalGenome:
    rng = random.Random(seed * 1000 + candidate_index * 17 + len(benchmark_name))
    variant_policy = _variant_policy(architecture_mode)
    base_mode = str(variant_policy["base_mode"])
    motif_bias = bool(variant_policy["motif_bias"])
    depth = 3 + (candidate_index % 3)
    width_choices = [16, 24, 32, 48, 64, 96, 128]
    width = max(8, min(max(input_dim, min(output_dim, 128)), rng.choice(width_choices)))
    shared = base_mode == "two_level_shared" and candidate_index % 2 == 0
    cells: dict[str, CellGene] = {}
    macro_nodes: list[MacroNodeGene] = []
    macro_edges: list[MacroEdgeGene] = []

    for macro_index in range(depth):
        cell_id = "cell_shared" if shared else f"cell_{macro_index}"
        if base_mode in {"two_level_unshared", "flat_macro"}:
            cell_id = f"cell_{macro_index}"
        if cell_id not in cells:
            inner_depth = 1 if base_mode == "flat_macro" else 2 + ((candidate_index + macro_index) % 3)
            node_ids = [f"node_{macro_index}_{node_index}" for node_index in range(inner_depth)]
            if base_mode == "flat_macro":
                nodes = [
                    CellNodeGene(
                        node_id=node_id,
                        kind=PrimitiveKind.LINEAR,
                        width=width,
                        activation=ActivationKind.IDENTITY,
                    )
                    for node_id in node_ids
                ]
            elif motif_bias and base_mode == "two_level_shared":
                motif = MOTIF_LIBRARY[(candidate_index + macro_index + len(benchmark_name)) % len(MOTIF_LIBRARY)]
                nodes = [
                    CellNodeGene(
                        node_id=node_id,
                        kind=motif[node_index % len(motif)][0],
                        width=width,
                        activation=motif[node_index % len(motif)][1],
                    )
                    for node_index, node_id in enumerate(node_ids)
                ]
            else:
                nodes = [
                    CellNodeGene(
                        node_id=node_id,
                        kind=rng.choice(list(PrimitiveKind)),
                        width=width,
                        activation=rng.choice(list(ActivationKind)),
                    )
                    for node_id in node_ids
                ]
            edges = [CellEdgeGene(source="input", target=node_ids[0])]
            for src, dst in zip(node_ids, node_ids[1:]):
                edges.append(CellEdgeGene(source=src, target=dst))
            if len(node_ids) >= 3 and base_mode != "flat_macro":
                edges.append(CellEdgeGene(source=node_ids[0], target=node_ids[-1]))
            edges.append(CellEdgeGene(source=node_ids[-1], target="output"))
            cells[cell_id] = CellGene(
                cell_id=cell_id,
                input_width=width,
                output_width=width,
                shared=shared and base_mode == "two_level_shared",
                nodes=nodes,
                edges=edges,
            )
        macro_nodes.append(
            MacroNodeGene(
                node_id=f"macro_{macro_index}",
                cell_id=cell_id,
                input_width=width,
                output_width=width,
                role="body" if macro_index else "stem",
            )
        )
    macro_edges = _seed_macro_edges(macro_nodes, rng=rng, shared=shared and base_mode == "two_level_shared")
    return HierarchicalGenome(
        genome_id=f"{benchmark_name}_seed_{seed}_cand_{candidate_index}",
        task=task,
        input_dim=input_dim,
        output_dim=output_dim,
        macro_nodes=macro_nodes,
        macro_edges=macro_edges,
        cell_library=cells,
    )


def _is_better(metric_direction: str, candidate: EvaluationRecord, incumbent: EvaluationRecord | None) -> bool:
    if incumbent is None:
        return True
    if candidate.status == "ok" and incumbent.status != "ok":
        return True
    if candidate.status != "ok":
        return False
    if incumbent.status != "ok":
        return True
    if metric_direction == "max":
        return candidate.metric_value > incumbent.metric_value
    return candidate.metric_value < incumbent.metric_value


def _next_population(
    *,
    evaluated: list[tuple[HierarchicalGenome, EvaluationRecord, float]],
    benchmark_name: str,
    task: str,
    input_dim: int,
    output_dim: int,
    seed: int,
    generation: int,
    population_size: int,
    architecture_mode: str,
    allow_clone_mutation: bool,
    motif_bias: bool,
    trained_states: dict[str, TrainingArtifact | None],
) -> tuple[list[HierarchicalGenome], dict[str, TrainingArtifact | None]]:
    scored = sorted(
        evaluated,
        key=lambda item: _selection_key(item[1], item[2]),
        reverse=True,
    )
    elites = [item[0] for item in scored[: max(2, min(len(scored), population_size // 2 or 1))]]
    rng = random.Random(seed * 100_000 + generation * 97 + len(benchmark_name))
    next_population: list[HierarchicalGenome] = []
    next_states: dict[str, TrainingArtifact | None] = {}
    for index in range(population_size):
        candidate_id = f"{benchmark_name}_seed_{seed}_gen_{generation+1}_cand_{index}"
        if len(elites) >= 2 and index % 3 == 0:
            left = elites[index % len(elites)]
            right = elites[(index + 1) % len(elites)]
            child = crossover_genomes(
                left,
                right,
                rng=rng,
                candidate_id=candidate_id,
                allow_clone_mutation=allow_clone_mutation,
                motif_bias=motif_bias,
            )
            inherited_state = trained_states.get(left.genome_id) or trained_states.get(right.genome_id)
        else:
            parent = elites[index % len(elites)] if elites else _make_candidate(
                benchmark_name=benchmark_name,
                task=task,
                input_dim=input_dim,
                output_dim=output_dim,
                seed=seed,
                candidate_index=index,
                architecture_mode=architecture_mode,
            )
            child = mutate_genome(
                parent,
                rng=rng,
                candidate_id=candidate_id,
                allow_clone_mutation=allow_clone_mutation,
                motif_bias=motif_bias,
            )
            inherited_state = trained_states.get(parent.genome_id)
        child = _enforce_architecture_mode(child, architecture_mode)
        next_population.append(child)
        next_states[child.genome_id] = inherited_state if _artifact_compatible(child, inherited_state) else None
    return next_population, next_states


def _selection_key(result: EvaluationRecord, novelty: float) -> float:
    if result.status != "ok":
        return float("-inf")
    return result.quality + novelty * 0.05


def _artifact_compatible(genome: HierarchicalGenome, artifact: TrainingArtifact | None) -> bool:
    if artifact is None:
        return False
    if artifact.task != genome.task:
        return False
    if artifact.model_name == "neural_classifier":
        return int(artifact.payload.get("feature_dim", -1)) > 0 and int(artifact.payload.get("num_classes", -1)) == genome.output_dim
    if artifact.model_name == "neural_lm_head":
        return int(artifact.payload.get("feature_dim", -1)) > 0 and int(artifact.payload.get("vocab_size", -1)) == genome.output_dim
    return True


def _normalize_config(config: RunConfig) -> RunConfig:
    benchmark_pool = config.benchmark_pool
    runtime = config.runtime
    evolution = config.evolution
    training = config.training
    if not isinstance(benchmark_pool, BenchmarkPoolConfig):
        benchmark_pool = BenchmarkPoolConfig.model_validate(benchmark_pool)
    if not hasattr(runtime, "backend"):
        from stratograph.config import RuntimeConfig

        runtime = RuntimeConfig.model_validate(runtime)
    if not isinstance(evolution, EvolutionConfig):
        evolution = EvolutionConfig.model_validate(evolution)
    if not isinstance(training, TrainingConfig):
        training = TrainingConfig.model_validate(training)
    if (
        benchmark_pool is config.benchmark_pool
        and runtime is config.runtime
        and evolution is config.evolution
        and training is config.training
    ):
        return config
    return RunConfig(
        seed=config.seed,
        run_name=config.run_name,
        benchmark_pool=benchmark_pool,
        runtime=runtime,
        training=training,
        evolution=evolution,
    )


def _variant_policy(architecture_mode: str) -> dict[str, str | bool]:
    if architecture_mode == "flat_macro":
        return {"base_mode": "flat_macro", "allow_clone_mutation": False, "motif_bias": False}
    if architecture_mode == "two_level_unshared":
        return {"base_mode": "two_level_unshared", "allow_clone_mutation": False, "motif_bias": False}
    if architecture_mode == "two_level_shared":
        return {"base_mode": "two_level_shared", "allow_clone_mutation": True, "motif_bias": True}
    if architecture_mode == "two_level_shared_no_clone":
        return {"base_mode": "two_level_shared", "allow_clone_mutation": False, "motif_bias": True}
    if architecture_mode == "two_level_shared_no_motif_bias":
        return {"base_mode": "two_level_shared", "allow_clone_mutation": True, "motif_bias": False}
    raise ValueError(f"Unknown architecture mode: {architecture_mode}")


def _enforce_architecture_mode(genome: HierarchicalGenome, architecture_mode: str) -> HierarchicalGenome:
    base_mode = str(_variant_policy(architecture_mode)["base_mode"])
    if base_mode == "two_level_shared":
        if len(genome.macro_nodes) < 2:
            return genome
        counts = Counter(node.cell_id for node in genome.macro_nodes)
        if not any(count > 1 for count in counts.values()):
            base_node = genome.macro_nodes[0]
            base_cell = genome.cell_library[base_node.cell_id]
            target_index = len(genome.macro_nodes) - 1
            macro_nodes = [
                node.model_copy(
                    update={
                        "cell_id": base_cell.cell_id,
                        "input_width": base_cell.input_width,
                        "output_width": base_cell.output_width,
                    }
                )
                if index == target_index
                else node
                for index, node in enumerate(genome.macro_nodes)
            ]
            counts = Counter(node.cell_id for node in macro_nodes)
            used_ids = {node.cell_id for node in macro_nodes}
            cell_library = {
                cell_id: genome.cell_library[cell_id].model_copy(update={"shared": counts[cell_id] > 1}, deep=True)
                for cell_id in used_ids
            }
            return genome.model_copy(update={"macro_nodes": macro_nodes, "cell_library": cell_library})
        used_ids = {node.cell_id for node in genome.macro_nodes}
        cell_library = {
            cell_id: genome.cell_library[cell_id].model_copy(update={"shared": counts[cell_id] > 1}, deep=True)
            for cell_id in used_ids
        }
        return genome.model_copy(update={"cell_library": cell_library})
    if base_mode == "two_level_unshared":
        macro_nodes = []
        cell_library = {}
        for index, node in enumerate(genome.macro_nodes):
            source_cell = genome.cell_library[node.cell_id]
            cell_id = f"cell_u_{index}"
            cell_library[cell_id] = source_cell.model_copy(update={"cell_id": cell_id, "shared": False}, deep=True)
            macro_nodes.append(node.model_copy(update={"cell_id": cell_id}))
        return genome.model_copy(update={"macro_nodes": macro_nodes, "cell_library": cell_library})
    if base_mode == "flat_macro":
        macro_nodes = []
        cell_library = {}
        for index, node in enumerate(genome.macro_nodes):
            width = node.output_width
            cell_id = f"cell_f_{index}"
            cell_library[cell_id] = CellGene(
                cell_id=cell_id,
                input_width=width,
                output_width=width,
                shared=False,
                nodes=[
                    CellNodeGene(
                        node_id=f"flat_{index}",
                        kind=PrimitiveKind.LINEAR,
                        width=width,
                        activation=ActivationKind.IDENTITY,
                    )
                ],
                edges=[
                    CellEdgeGene(source="input", target=f"flat_{index}"),
                    CellEdgeGene(source=f"flat_{index}", target="output"),
                ],
            )
            macro_nodes.append(node.model_copy(update={"cell_id": cell_id}))
        return genome.model_copy(update={"macro_nodes": macro_nodes, "cell_library": cell_library})
    return genome


def _write_checkpoint(
    path: Path,
    *,
    run_id: str,
    architecture_mode: str,
    completed_benchmarks: list[str],
    total_benchmarks: int,
) -> None:
    path.write_text(
        json.dumps(
            {
                "run_id": run_id,
                "architecture_mode": architecture_mode,
                "completed_benchmarks": completed_benchmarks,
                "completed_count": len(completed_benchmarks),
                "remaining_count": max(0, total_benchmarks - len(completed_benchmarks)),
                "total_benchmarks": total_benchmarks,
            },
            indent=2,
        ),
        encoding="utf-8",
    )


def _write_status(
    path: Path,
    *,
    run_id: str,
    architecture_mode: str,
    total_benchmarks: int,
    completed_benchmarks: list[str],
    state: str,
) -> None:
    path.write_text(
        json.dumps(
            {
                "run_id": run_id,
                "architecture_mode": architecture_mode,
                "state": state,
                "total_benchmarks": total_benchmarks,
                "completed_benchmarks": completed_benchmarks,
                "completed_count": len(completed_benchmarks),
                "remaining_count": max(0, total_benchmarks - len(completed_benchmarks)),
            },
            indent=2,
        ),
        encoding="utf-8",
    )


def _write_best_genome(
    run_dir: Path,
    benchmark_name: str,
    genome: HierarchicalGenome,
    result: EvaluationRecord,
    architecture_mode: str,
) -> None:
    target_dir = run_dir / "best_genomes"
    target_dir.mkdir(parents=True, exist_ok=True)
    (target_dir / f"{benchmark_name}.json").write_text(
        json.dumps(
            {
                "benchmark_name": benchmark_name,
                "architecture_mode": architecture_mode,
                "metric_value": result.metric_value,
                "quality": result.quality,
                "architecture_summary": result.architecture_summary,
                "genome": genome_to_dict(genome),
            },
            indent=2,
        ),
        encoding="utf-8",
    )


def _seed_macro_edges(
    macro_nodes: list[MacroNodeGene],
    *,
    rng: random.Random,
    shared: bool,
) -> list[MacroEdgeGene]:
    if not macro_nodes:
        return []
    edges: list[MacroEdgeGene] = [MacroEdgeGene(source="input", target=macro_nodes[0].node_id)]
    for index, node in enumerate(macro_nodes[1:], start=1):
        parent = macro_nodes[index - 1]
        edges.append(MacroEdgeGene(source=parent.node_id, target=node.node_id))
        if index >= 2 and (shared or rng.random() < 0.7):
            skip_parent = macro_nodes[max(0, index - 2)]
            if skip_parent.node_id != parent.node_id:
                edges.append(MacroEdgeGene(source=skip_parent.node_id, target=node.node_id))
        if index >= 3 and rng.random() < 0.35:
            long_parent = macro_nodes[rng.randrange(0, index - 1)]
            if long_parent.node_id not in {parent.node_id, node.node_id}:
                edges.append(MacroEdgeGene(source=long_parent.node_id, target=node.node_id))

    sink_sources = {edge.source for edge in edges if edge.target != "output"}
    for node in macro_nodes:
        if node.node_id not in sink_sources or rng.random() < 0.25:
            edges.append(MacroEdgeGene(source=node.node_id, target="output"))
    return _dedupe_macro_edges(edges)


def _dedupe_macro_edges(edges: list[MacroEdgeGene]) -> list[MacroEdgeGene]:
    seen: set[tuple[str, str]] = set()
    unique: list[MacroEdgeGene] = []
    for edge in edges:
        key = (edge.source, edge.target)
        if key in seen:
            continue
        seen.add(key)
        unique.append(edge)
    return unique
