"""Symbiosis manifest/results export for Topograph.

Produces manifest.json + results.json for cross-system comparison with
EvoNN and EvoNN-2 via the EvoNN-Symbiosis layer.
"""

from __future__ import annotations

import hashlib
import importlib.metadata
import json
import math
import platform
import shutil
import subprocess
from collections import Counter
from datetime import datetime, timezone
from pathlib import Path
from statistics import median as stat_median
from typing import Any

try:
    _MLX_VERSION = importlib.metadata.version("mlx")
except importlib.metadata.PackageNotFoundError:
    try:
        import mlx

        _MLX_VERSION = getattr(mlx, "__version__", None)
    except ImportError:
        mlx = None
        _MLX_VERSION = None

import topograph
from evonn_shared.manifests import benchmark_signature, fairness_manifest
from topograph.benchmarks.parity import (
    get_canonical_id,
    load_parity_pack,
    resolve_benchmark_pool_names,
)
from topograph.benchmarks.spec import BenchmarkSpec
from topograph.config import RunConfig, load_config
from topograph.genome.codec import dict_to_genome
from topograph.genome.genome import Genome
from topograph.storage import RunStore


# ---------------------------------------------------------------------------
# Public API
# ---------------------------------------------------------------------------


def export_symbiosis_contract(
    run_dir: str | Path,
    pack_path: str | Path,
    output_dir: str | Path | None = None,
) -> tuple[Path, Path]:
    """Export symbiosis contract from a completed Topograph run.

    Returns (manifest_path, results_path).
    """
    run_dir = Path(run_dir)
    if not run_dir.exists():
        raise FileNotFoundError(f"Run directory not found: {run_dir}")

    output_dir = Path(output_dir) if output_dir else run_dir
    output_dir.mkdir(parents=True, exist_ok=True)

    # 1. Load run config
    store = RunStore(run_dir / "metrics.duckdb")
    config = _load_run_config(run_dir, store)

    # 2. Load latest generation
    run_id = _resolve_run_id(store)
    latest_gen = store.load_latest_generation(run_id)
    if latest_gen is None:
        store.close()
        raise ValueError(f"No generations found in {run_dir}")

    genome_dicts = store.load_genomes(run_id, latest_gen)
    population = [dict_to_genome(d) for d in genome_dicts]

    # 3. Load parity pack benchmarks
    pack_specs = load_parity_pack(pack_path)
    benchmark_names = [spec.name for spec in pack_specs]

    # 4. Select representative genome (best fitness = lowest)
    representative = _select_representative(population)

    # 5. Load existing benchmark results or leave empty
    existing_results = store.load_best_benchmark_results(run_id)
    existing_by_name = {r["benchmark_name"]: r for r in existing_results}

    # 6. Build results list
    results: list[dict[str, Any]] = []
    benchmark_entries: list[dict[str, Any]] = []

    for spec in pack_specs:
        native_name = spec.name
        metric_name = _benchmark_metric_name(spec.task)
        metric_direction = _benchmark_metric_direction(spec.task)

        existing = existing_by_name.get(native_name)
        if existing:
            metric_value = existing["metric_value"]
            quality = existing["quality"]
            parameter_count = existing["parameter_count"]
            train_seconds = existing["train_seconds"]
            architecture_summary = existing["architecture_summary"]
            status = existing.get("status", "missing")
            failure_reason = existing.get("failure_reason")
        else:
            # Use representative genome's fitness as fallback for the primary benchmark
            if representative and representative.fitness is not None:
                metric_value = representative.fitness
                quality = representative.fitness
                parameter_count = representative.param_count
                train_seconds = None
                architecture_summary = _architecture_summary_str(representative)
                status = "ok"
                failure_reason = None
            else:
                metric_value = None
                quality = None
                parameter_count = None
                train_seconds = None
                architecture_summary = None
                status = "missing"
                failure_reason = "no_evaluated_genome"

        results.append({
            "system": "topograph",
            "run_id": run_dir.name,
            "benchmark_id": get_canonical_id(native_name),
            "metric_name": metric_name,
            "metric_direction": metric_direction,
            "metric_value": metric_value,
            "quality": quality,
            "parameter_count": parameter_count,
            "train_seconds": train_seconds,
            "peak_memory_mb": None,
            "architecture_summary": architecture_summary,
            "genome_id": None,
            "residual_connections_count": (
                _skip_connection_count(representative) if representative else None
            ),
            "estimated_model_bytes": (
                _estimate_model_bytes(representative) if representative else None
            ),
            "status": status,
            "failure_reason": failure_reason,
        })

        benchmark_entries.append({
            "benchmark_id": get_canonical_id(native_name),
            "task_kind": spec.task,
            "metric_name": metric_name,
            "metric_direction": metric_direction,
            "status": status,
        })

    # 7. Load budget metadata
    budget_meta = store.load_budget_metadata(run_id) or {}
    store.close()

    # 8. Build manifest
    pack_name = Path(pack_path).stem
    runtime_meta = _runtime_metadata_from_budget(budget_meta)

    budget_manifest = _budget_manifest(config, budget_meta, latest_gen, len(population))
    manifest = {
        "schema_version": "1.0",
        "system": "topograph",
        "version": topograph.__version__,
        "run_id": run_dir.name,
        "run_name": run_dir.name,
        "created_at": datetime.now(timezone.utc).isoformat(),
        "pack_name": pack_name,
        "seed": config.seed,
        "benchmarks": benchmark_entries,
        "budget": budget_manifest,
        "device": {
            "device_name": _detect_device(),
            "precision_mode": runtime_meta["precision_mode"],
            "framework": runtime_meta["runtime_backend"],
            "framework_version": runtime_meta["runtime_version"],
        },
        "config_snapshot": config.model_dump(mode="json"),
        "artifacts": _build_artifacts_section(
            representative, benchmark_names, config, pack_name, run_dir,
        ),
        "search_telemetry": _search_telemetry(config, budget_meta),
        "fairness": fairness_manifest(
            pack_name=pack_name,
            seed=config.seed,
            evaluation_count=int(budget_manifest["evaluation_count"]),
            budget_policy_name="prototype_equal_budget",
            benchmark_entries=benchmark_entries,
            data_signature=benchmark_signature(pack_name, benchmark_entries),
            code_version=_code_version(),
        ),
    }

    # 9. Write manifest.json and results.json
    manifest_path = output_dir / "manifest.json"
    results_path = output_dir / "results.json"
    manifest_path.write_text(json.dumps(manifest, indent=2), encoding="utf-8")
    results_path.write_text(json.dumps(results, indent=2), encoding="utf-8")

    # 10. Copy config, write report and summary
    config_src = run_dir / "config.yaml"
    if config_src.exists() and output_dir != run_dir:
        shutil.copy2(config_src, output_dir / "config.yaml")

    _write_report_md(output_dir, run_dir)
    _write_summary_json(output_dir, manifest, results, population, latest_gen, config)

    return manifest_path, results_path


# ---------------------------------------------------------------------------
# Helpers — genome analysis
# ---------------------------------------------------------------------------


def _genome_summary(genome: Genome) -> dict[str, Any]:
    """Compact topology + precision summary of a genome."""
    layers = sorted(genome.enabled_layers, key=lambda lg: lg.order)
    depths = _longest_path_depths(genome)
    max_depth = max(depths.values(), default=0)
    width_profile = [0] * (max_depth + 1) if max_depth >= 0 else []
    for d in depths.values():
        if 0 <= d < len(width_profile):
            width_profile[d] += 1

    return {
        "num_layers": len(layers),
        "num_connections": len(genome.enabled_connections),
        "layer_widths": [lg.width for lg in layers],
        "activations": [lg.activation.value for lg in layers],
        "weight_bits": [lg.weight_bits.value for lg in layers],
        "activation_bits": [lg.activation_bits.value for lg in layers],
        "dag_depth": _dag_depth(genome),
        "skip_connections": _skip_connection_count(genome),
        "precision_distribution": _precision_distribution(genome),
        "has_experts": len(genome.experts) > 0,
        "num_experts": len([e for e in genome.experts if e.enabled]) if genome.experts else 0,
        "total_params": genome.param_count or sum(lg.width for lg in layers),
    }


def _model_summary(genome: Genome, benchmark_spec: BenchmarkSpec) -> dict[str, Any]:
    """Compiled model summary."""
    params = genome.param_count or sum(lg.width for lg in genome.enabled_layers)
    return {
        "total_params": params,
        "estimated_bytes": _estimate_model_bytes(genome),
        "task": benchmark_spec.task,
        "input_dim": benchmark_spec.input_dim,
        "num_classes": benchmark_spec.num_classes,
        "fitness": genome.fitness if genome.fitness is not None else None,
        "validation_metric": (
            "perplexity"
            if benchmark_spec.task == "language_modeling"
            else ("mse" if benchmark_spec.task == "regression" else "cross_entropy")
        ),
    }


def _budget_manifest(
    config: RunConfig,
    budget_meta: dict[str, Any],
    latest_gen: int,
    population_size: int,
) -> dict[str, Any]:
    """Budget section of the manifest."""
    generations = latest_gen + 1
    evaluation_count = budget_meta.get("evaluation_count")
    if evaluation_count is None:
        benchmark_count = 1
        if config.benchmark_pool is not None:
            benchmark_count = min(
                config.benchmark_pool.sample_k,
                len(resolve_benchmark_pool_names(config.benchmark_pool)),
            )
        evaluation_count = int(population_size * generations * benchmark_count)
    return {
        "evaluation_count": int(evaluation_count),
        "epochs_per_candidate": config.training.epochs,
        "effective_training_epochs": budget_meta.get("effective_training_epochs"),
        "wall_clock_seconds": budget_meta.get("wall_clock_seconds"),
        "generations": generations,
        "population_size": int(
            budget_meta.get("population_size", config.evolution.population_size)
        ),
        "budget_policy_name": "prototype_equal_budget",
    }


def _search_telemetry(
    config: RunConfig, budget_meta: dict[str, Any],
) -> dict[str, Any] | None:
    """Search telemetry section of the manifest."""
    qd_enabled = config.novelty_weight > 0 or config.map_elites
    multi_fidelity = config.training.multi_fidelity
    family_search = config.benchmark_pool is not None

    if not qd_enabled and not multi_fidelity and not family_search:
        return None

    occupied = budget_meta.get("map_elites_occupied_niches")
    total = budget_meta.get("map_elites_total_niches")
    fill = budget_meta.get("map_elites_fill_ratio")
    if fill is None and total and occupied is not None:
        fill = occupied / total

    return {
        "qd_enabled": qd_enabled,
        "multi_fidelity": multi_fidelity,
        "multi_fidelity_schedule": config.training.multi_fidelity_schedule,
        "effective_training_epochs": budget_meta.get("effective_training_epochs"),
        "novelty_weight": config.novelty_weight,
        "novelty_k": config.novelty_k,
        "novelty_archive_limit": (
            config.novelty_archive_size if config.novelty_weight > 0 else None
        ),
        "novelty_archive_final_size": budget_meta.get("novelty_archive_final_size"),
        "novelty_score_mean": budget_meta.get("novelty_score_mean"),
        "novelty_score_max": budget_meta.get("novelty_score_max"),
        "map_elites_enabled": config.map_elites,
        "map_elites_occupied_niches": occupied,
        "map_elites_total_niches": total,
        "map_elites_fill_ratio": fill,
        "map_elites_insertions": budget_meta.get("map_elites_insertions"),
        "benchmark_elite_archive": config.benchmark_elite_archive,
        "benchmark_elites": budget_meta.get("benchmark_elites"),
        "benchmark_pool_aggregation": (
            config.benchmark_pool.aggregation if config.benchmark_pool is not None else None
        ),
        "benchmark_pool_family_stage_generations": (
            config.benchmark_pool.family_stage_generations
            if config.benchmark_pool is not None
            else None
        ),
        "benchmark_pool_family_transfer": (
            config.benchmark_pool.family_transfer
            if config.benchmark_pool is not None
            else None
        ),
        "family_stage_history": budget_meta.get("family_stage_history"),
        "benchmark_elite_families": budget_meta.get("benchmark_elite_families"),
        "topology_atlas_motif_counts": budget_meta.get("topology_atlas_motif_counts"),
        "primordia_seeding": budget_meta.get("primordia_seeding"),
    }


def _runtime_metadata_from_budget(budget_meta: dict[str, Any]) -> dict[str, str]:
    """Normalize recorded runtime metadata for export artifacts.

    Export must describe the runtime that produced the run artifacts, not the
    exporter host. Missing recorded fields therefore degrade to ``unknown``.
    """

    return {
        "runtime_backend": str(budget_meta.get("runtime_backend") or "unknown"),
        "runtime_version": str(budget_meta.get("runtime_version") or "unknown"),
        "precision_mode": str(budget_meta.get("precision_mode") or "unknown"),
    }


def _estimate_model_bytes(genome: Genome) -> int:
    """Precision-aware byte estimate from genome structure."""
    layers = genome.enabled_layers
    if not layers:
        return 0
    weight_bits_vals = [lg.weight_bits.value for lg in layers]
    avg_bits = sum(weight_bits_vals) / len(weight_bits_vals)
    params = genome.param_count or sum(lg.width for lg in layers)
    return math.ceil(params * avg_bits / 8)


# ---------------------------------------------------------------------------
# Helpers — DAG analysis (local copies, independent of archive.py)
# ---------------------------------------------------------------------------


def _build_adjacency(genome: Genome) -> dict[int, list[int]]:
    from collections import defaultdict

    adj: dict[int, list[int]] = defaultdict(list)
    for c in genome.enabled_connections:
        adj[c.source].append(c.target)
    return dict(adj)


def _all_nodes(adj: dict[int, list[int]]) -> set[int]:
    nodes: set[int] = set()
    for src, targets in adj.items():
        nodes.add(src)
        nodes.update(targets)
    return nodes


def _longest_path_depths(genome: Genome) -> dict[int, int]:
    from collections import defaultdict, deque
    from topograph.genome.genome import INPUT_INNOVATION

    adj = _build_adjacency(genome)
    nodes = _all_nodes(adj)
    if INPUT_INNOVATION not in nodes:
        return {}

    in_degree: dict[int, int] = defaultdict(int)
    for node in nodes:
        in_degree.setdefault(node, 0)
    for targets in adj.values():
        for tgt in targets:
            in_degree[tgt] += 1

    dist: dict[int, int] = {node: -1 for node in nodes}
    dist[INPUT_INNOVATION] = 0

    queue: deque[int] = deque(n for n in nodes if in_degree[n] == 0)
    while queue:
        node = queue.popleft()
        for neighbor in adj.get(node, []):
            if dist[node] >= 0:
                dist[neighbor] = max(dist[neighbor], dist[node] + 1)
            in_degree[neighbor] -= 1
            if in_degree[neighbor] == 0:
                queue.append(neighbor)

    return {n: d for n, d in dist.items() if d >= 0}


def _dag_depth(genome: Genome) -> int:
    from topograph.genome.genome import OUTPUT_INNOVATION
    return _longest_path_depths(genome).get(OUTPUT_INNOVATION, 0)


def _skip_connection_count(genome: Genome) -> int:
    depths = _longest_path_depths(genome)
    count = 0
    for c in genome.enabled_connections:
        sd = depths.get(c.source)
        td = depths.get(c.target)
        if sd is not None and td is not None and td - sd > 1:
            count += 1
    return count


def _precision_distribution(genome: Genome) -> dict[str, dict[str, int]]:
    from collections import defaultdict

    weight_bits: dict[str, int] = defaultdict(int)
    activation_bits: dict[str, int] = defaultdict(int)
    for lg in genome.enabled_layers:
        weight_bits[str(lg.weight_bits.value)] += 1
        activation_bits[str(lg.activation_bits.value)] += 1
    return {
        "weight_bits": dict(weight_bits),
        "activation_bits": dict(activation_bits),
    }


# ---------------------------------------------------------------------------
# Helpers — misc
# ---------------------------------------------------------------------------


def _select_representative(population: list[Genome]) -> Genome | None:
    if not population:
        return None
    return min(
        population,
        key=lambda g: g.fitness if g.fitness is not None else float("inf"),
    )


def _architecture_summary_str(genome: Genome) -> str:
    layers = genome.enabled_layers
    conns = genome.enabled_connections
    return f"{len(layers)}L/{len(conns)}C depth={_dag_depth(genome)}"


def _benchmark_metric_name(task: str) -> str:
    if task == "language_modeling":
        return "perplexity"
    return "mse" if task == "regression" else "accuracy"


def _benchmark_metric_direction(task: str) -> str:
    return "min" if task in {"regression", "language_modeling"} else "max"


def _detect_device() -> str:
    machine = platform.machine()
    system = platform.system()
    if system == "Darwin":
        return "apple_silicon" if "arm" in machine.lower() else "apple_intel"
    return f"{system.lower()}_{machine}"


def _load_run_config(run_dir: Path, store: RunStore) -> RunConfig:
    config_path = run_dir / "config.yaml"
    if config_path.exists():
        return load_config(config_path)
    # Fallback: try loading from DB
    try:
        run_id = _resolve_run_id(store)
        config_dict = store.load_run(run_id)
        return RunConfig.model_validate(config_dict)
    except (ValueError, KeyError):
        return RunConfig()


def _resolve_run_id(store: RunStore) -> str:
    """Resolve the run_id from the store. Tries 'current', then first available."""
    try:
        store.load_run("current")
        return "current"
    except ValueError:
        pass
    row = store.conn.execute(
        "SELECT run_id FROM runs ORDER BY created_at DESC LIMIT 1"
    ).fetchone()
    if row:
        return row[0]
    return "current"


def _compute_dataset_hash(benchmark_names: list[str]) -> str:
    key = "|".join(sorted(benchmark_names))
    return hashlib.sha256(key.encode()).hexdigest()[:16]


def _code_version() -> str | None:
    try:
        return subprocess.check_output(
            ["git", "rev-parse", "HEAD"],
            cwd=Path(__file__).resolve().parents[3],
            text=True,
        ).strip()
    except Exception:
        return None


def _build_artifacts_section(
    representative: Genome | None,
    benchmark_names: list[str],
    config: RunConfig,
    pack_name: str,
    run_dir: Path | None = None,
) -> dict[str, Any]:
    """Artifacts metadata for the manifest."""
    genome_summary = _genome_summary(representative) if representative else {"status": "missing"}
    payload = {
        "genome_summary": genome_summary,
        "dataset_manifest_hash": _compute_dataset_hash(benchmark_names),
        "pack_name": pack_name,
        "benchmarks": benchmark_names,
        "canonical_benchmarks": [get_canonical_id(n) for n in benchmark_names],
    }
    if run_dir is not None:
        archive_files = {}
        for name in ["benchmark_elites.json", "topology_atlas_summary.json", "map_elites_archive.json"]:
            path = run_dir / name
            if path.exists():
                archive_files[name.removesuffix(".json")] = str(path)
        if archive_files:
            payload["archive_artifacts"] = archive_files
    return payload


def _write_report_md(output_dir: Path, run_dir: Path) -> None:
    """Generate and write report.md if not already present."""
    report_path = output_dir / "report.md"
    if report_path.exists():
        return
    try:
        from topograph.export.report import generate_report
        text = generate_report(run_dir)
        report_path.write_text(text, encoding="utf-8")
    except Exception:
        pass  # report generation is best-effort


def _write_summary_json(
    output_dir: Path,
    manifest: dict[str, Any],
    results: list[dict[str, Any]],
    population: list[Genome],
    latest_gen: int,
    config: RunConfig,
) -> None:
    """Write summary.json with cross-system durability contract fields."""
    # Best metric per benchmark
    best_fitness: dict[str, float] = {}
    for record in results:
        bid = record.get("benchmark_id", "")
        mv = record.get("metric_value")
        status = record.get("status", "")
        if status == "ok" and mv is not None:
            best_fitness[bid] = float(mv)

    # Median parameter count
    param_counts = []
    for genome in population:
        pc = genome.param_count or sum(lg.width for lg in genome.enabled_layers)
        param_counts.append(pc)
    median_param_count = int(stat_median(param_counts)) if param_counts else 0

    # Median quality
    qualities = [v for v in best_fitness.values()]
    median_quality = float(stat_median(qualities)) if qualities else None

    failure_count = sum(1 for r in results if r.get("status") != "ok")
    non_ok_results = [r for r in results if r.get("status") != "ok"]
    failure_patterns = dict(
        Counter(str(r.get("failure_reason") or r.get("status") or "unknown") for r in non_ok_results).most_common()
    )

    budget = manifest.get("budget", {})
    device = manifest.get("device", {})
    seeding = budget.get("primordia_seeding") if isinstance(budget.get("primordia_seeding"), dict) else None
    summary = {
        "system": "topograph",
        "run_id": manifest["run_id"],
        "status": "complete",
        "total_evaluations": budget.get("evaluation_count", 0),
        "wall_clock_seconds": budget.get("wall_clock_seconds"),
        "generations_completed": latest_gen + 1,
        "epochs_per_candidate": config.training.epochs,
        "population_size": budget.get("population_size", config.evolution.population_size),
        "best_fitness": best_fitness,
        "median_parameter_count": median_param_count,
        "median_benchmark_quality": median_quality,
        "failure_count": failure_count,
        "failure_patterns": failure_patterns,
        "benchmarks_evaluated": len(best_fitness),
        "runtime_backend": device.get("framework", "unknown"),
        "runtime_version": device.get("framework_version", "unknown"),
        "precision_mode": device.get("precision_mode", "unknown"),
        "seed_source_system": "primordia" if seeding else None,
        "seed_source_path": seeding.get("seed_path") if seeding else None,
        "seed_target_family": seeding.get("target_family") if seeding else None,
        "seed_selected_family": seeding.get("selected_family") if seeding else None,
        "seed_selected_rank": seeding.get("selected_rank") if seeding else None,
        "seed_representative_genome_id": seeding.get("representative_genome_id") if seeding else None,
        "seed_representative_architecture_summary": (
            seeding.get("representative_architecture_summary") if seeding else None
        ),
    }
    (output_dir / "summary.json").write_text(
        json.dumps(summary, indent=2), encoding="utf-8",
    )
