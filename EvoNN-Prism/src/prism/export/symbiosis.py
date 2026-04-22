"""Symbiosis manifest/results export for Prism.

Produces manifest.json + results.json for cross-system comparison with
EvoNN, EvoNN-2, and Topograph via the EvoNN-Symbiosis layer.
"""

from __future__ import annotations

import hashlib
import importlib.metadata
import json
import platform
import subprocess
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

import prism
from prism.benchmarks.parity import get_canonical_id, load_parity_pack
from prism.config import RunConfig, load_config
from prism.genome import ModelGenome
from prism.storage import RunStore


# ---------------------------------------------------------------------------
# Public API
# ---------------------------------------------------------------------------


def export_symbiosis_contract(
    run_dir: str | Path,
    pack_path: str | Path,
    output_dir: str | Path | None = None,
) -> tuple[Path, Path]:
    """Export symbiosis contract from a completed Prism run.

    Returns (manifest_path, results_path).
    """
    run_dir = Path(run_dir)
    if not run_dir.exists():
        raise FileNotFoundError(f"Run directory not found: {run_dir}")

    output_dir = Path(output_dir) if output_dir else run_dir
    output_dir.mkdir(parents=True, exist_ok=True)

    # 1. Load run config + DuckDB store
    store = RunStore(run_dir / "metrics.duckdb")
    config = _load_run_config(run_dir)
    run_id = _resolve_run_id(store)

    # 2. Load genome data
    genome_rows = store.load_genomes(run_id)
    genomes: list[ModelGenome] = []
    for row in genome_rows:
        try:
            genomes.append(ModelGenome.model_validate(row))
        except Exception:
            pass

    # 3. Load evaluation results
    evaluations = store.load_evaluations(run_id)
    best_per_benchmark = store.load_best_per_benchmark(run_id)
    latest_gen = store.latest_generation(run_id)
    lineage_records = store.load_lineage(run_id)

    # 4. Load parity pack benchmarks
    pack_specs = load_parity_pack(pack_path)
    benchmark_names = [spec.id for spec in pack_specs]

    # 5. Select representative genome (highest average quality)
    representative = _select_representative(genomes, evaluations)

    # 6. Build results list
    results: list[dict[str, Any]] = []
    benchmark_entries: list[dict[str, Any]] = []

    for spec in pack_specs:
        native_name = spec.id
        metric_name = _benchmark_metric_name(spec.task)
        metric_direction = _benchmark_metric_direction(spec.task)

        best = best_per_benchmark.get(native_name)
        if best and best.get("quality") is not None:
            metric_value = best["metric_value"]
            quality = best["quality"]
            parameter_count = best.get("parameter_count", 0)
            train_seconds = best.get("train_seconds")
            genome_id = best.get("genome_id")
            status = "ok"
            failure_reason = None
        elif representative:
            metric_value = None
            quality = None
            parameter_count = representative.parameter_estimate
            train_seconds = None
            genome_id = representative.genome_id
            status = "missing"
            failure_reason = "no_benchmark_result"
        else:
            metric_value = None
            quality = None
            parameter_count = None
            train_seconds = None
            genome_id = None
            status = "missing"
            failure_reason = "no_evaluated_genome"

        results.append({
            "system": "prism",
            "run_id": run_dir.name,
            "benchmark_id": get_canonical_id(native_name),
            "metric_name": metric_name,
            "metric_direction": metric_direction,
            "metric_value": metric_value,
            "quality": quality,
            "parameter_count": parameter_count,
            "train_seconds": train_seconds,
            "peak_memory_mb": None,
            "architecture_summary": _architecture_summary(representative) if representative else None,
            "genome_id": genome_id,
            "family": representative.family if representative else None,
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

    store.close()

    # 7. Build manifest
    runtime_meta = _load_runtime_metadata(run_dir)
    pack_name = Path(pack_path).stem
    generations = (latest_gen + 1) if latest_gen is not None else 0
    total_evaluations = _intended_evaluation_count(
        config=config,
        generations=generations,
        benchmark_count=len(pack_specs),
        fallback=len(evaluations),
    )
    config_snapshot_name = "config.yaml" if (output_dir / "config.yaml").exists() else "config.json"
    report_name = "report.md"

    manifest = {
        "schema_version": "1.0",
        "system": "prism",
        "version": prism.__version__,
        "run_id": run_dir.name,
        "run_name": run_dir.name,
        "created_at": datetime.now(timezone.utc).isoformat(),
        "pack_name": pack_name,
        "seed": config.seed,
        "benchmarks": benchmark_entries,
        "budget": {
            "evaluation_count": total_evaluations,
            "epochs_per_candidate": config.training.epochs,
            "generations": generations,
            "population_size": config.evolution.population_size,
            "budget_policy_name": "prototype_equal_budget",
        },
        "device": {
            "device_name": _detect_device(),
            "precision_mode": runtime_meta["precision_mode"],
            "framework": runtime_meta["runtime_backend"],
            "framework_version": runtime_meta["runtime_version"],
        },
        "config_snapshot": config.model_dump(mode="json"),
        "artifacts": {
            "config_snapshot": config_snapshot_name,
            "report_markdown": report_name,
            "genome_summary": _genome_summary(representative) if representative else {"status": "missing"},
            "dataset_manifest_hash": _compute_dataset_hash(benchmark_names),
            "pack_name": pack_name,
            "benchmarks": benchmark_names,
            "canonical_benchmarks": [get_canonical_id(n) for n in benchmark_names],
        },
        "fairness": _fairness_manifest(
            pack_name=pack_name,
            seed=config.seed,
            evaluation_count=total_evaluations,
            budget_policy_name="prototype_equal_budget",
            benchmark_entries=benchmark_entries,
            data_signature=_benchmark_signature(pack_name, benchmark_entries),
        ),
    }

    # 8. Write manifest.json and results.json
    manifest_path = output_dir / "manifest.json"
    results_path = output_dir / "results.json"
    manifest_path.write_text(json.dumps(manifest, indent=2), encoding="utf-8")
    results_path.write_text(json.dumps(results, indent=2), encoding="utf-8")
    (output_dir / report_name).write_text(_render_report_markdown(manifest, results), encoding="utf-8")

    # 9. Write summary.json
    _write_summary_json(
        output_dir,
        manifest,
        results,
        genomes,
        latest_gen,
        config,
        best_per_benchmark=best_per_benchmark,
        lineage_records=lineage_records,
    )

    return manifest_path, results_path


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------


def _select_representative(
    genomes: list[ModelGenome],
    evaluations: list[dict],
) -> ModelGenome | None:
    """Select the genome with the highest average quality across benchmarks."""
    if not genomes:
        return None

    genome_qualities: dict[str, list[float]] = {}
    for ev in evaluations:
        gid = ev.get("genome_id", "")
        q = ev.get("quality")
        if gid and q is not None and ev.get("failure_reason") is None:
            genome_qualities.setdefault(gid, []).append(float(q))

    if not genome_qualities:
        return genomes[0] if genomes else None

    best_id = max(
        genome_qualities,
        key=lambda gid: sum(genome_qualities[gid]) / len(genome_qualities[gid]),
    )
    for g in genomes:
        if g.genome_id == best_id:
            return g
    return genomes[0]


def _genome_summary(genome: ModelGenome) -> dict[str, Any]:
    """Compact summary of a Prism genome."""
    return {
        "family": genome.family,
        "hidden_layers": genome.hidden_layers,
        "activation": genome.activation,
        "dropout": genome.dropout,
        "residual": genome.residual,
        "norm_type": genome.norm_type,
        "parameter_estimate": genome.parameter_estimate,
        "num_experts": genome.num_experts,
    }


def _architecture_summary(genome: ModelGenome) -> str:
    """One-line architecture description."""
    layers_str = "x".join(str(w) for w in genome.hidden_layers)
    return f"{genome.family} [{layers_str}] {genome.activation}"


def _benchmark_metric_name(task: str) -> str:
    if task == "regression":
        return "mse"
    if task == "language_modeling":
        return "perplexity"
    return "accuracy"


def _benchmark_metric_direction(task: str) -> str:
    if task in {"regression", "language_modeling"}:
        return "min"
    return "max"


def _render_report_markdown(manifest: dict[str, Any], results: list[dict[str, Any]]) -> str:
    lines = [
        "# Prism Export Report",
        "",
        f"- Run ID: `{manifest['run_id']}`",
        f"- Pack: `{manifest['pack_name']}`",
        f"- Seed: `{manifest['seed']}`",
        f"- Evaluations: `{manifest['budget']['evaluation_count']}`",
        "",
        "## Results",
        "",
        "| Benchmark | Metric | Value | Status |",
        "|---|---|---:|---|",
    ]
    for row in results:
        value = "---" if row["metric_value"] is None else f"{float(row['metric_value']):.6f}"
        lines.append(
            f"| {row['benchmark_id']} | {row['metric_name']} | {value} | {row['status']} |"
        )
    return "\n".join(lines) + "\n"


def _detect_device() -> str:
    machine = platform.machine()
    system = platform.system()
    if system == "Darwin":
        return "apple_silicon" if "arm" in machine.lower() else "apple_intel"
    return f"{system.lower()}_{machine}"


def _load_runtime_metadata(run_dir: Path) -> dict[str, str | None]:
    summary_path = run_dir / "summary.json"
    if summary_path.exists():
        try:
            summary = json.loads(summary_path.read_text(encoding="utf-8"))
        except json.JSONDecodeError:
            summary = {}
    else:
        summary = {}
    return {
        "runtime_backend": summary.get("runtime_backend") or "unknown",
        "runtime_version": summary.get("runtime_version") or "unknown",
        "precision_mode": summary.get("precision_mode") or "fp32",
    }


def _load_run_config(run_dir: Path) -> RunConfig:
    config_path = run_dir / "config.yaml"
    if config_path.exists():
        return load_config(config_path)
    config_json = run_dir / "config.json"
    if config_json.exists():
        data = json.loads(config_json.read_text(encoding="utf-8"))
        return RunConfig.model_validate(data)
    return RunConfig()


def _resolve_run_id(store: RunStore) -> str:
    """Resolve the run_id from the store (most recent)."""
    row = store.conn.execute(
        "SELECT run_id FROM runs ORDER BY created_at DESC LIMIT 1"
    ).fetchone()
    if row:
        return row[0]
    return "default"


def _compute_dataset_hash(benchmark_names: list[str]) -> str:
    key = "|".join(sorted(benchmark_names))
    return hashlib.sha256(key.encode()).hexdigest()[:16]


def _fairness_manifest(
    *,
    pack_name: str,
    seed: int,
    evaluation_count: int,
    budget_policy_name: str,
    benchmark_entries: list[dict[str, Any]],
    data_signature: str,
) -> dict[str, Any]:
    return {
        "benchmark_pack_id": pack_name,
        "seed": seed,
        "evaluation_count": evaluation_count,
        "budget_policy_name": budget_policy_name,
        "data_signature": data_signature or _benchmark_signature(pack_name, benchmark_entries),
        "code_version": _code_version(),
    }


def _benchmark_signature(pack_name: str, benchmark_entries: list[dict[str, Any]]) -> str:
    payload = json.dumps(
        {
            "pack_name": pack_name,
            "benchmarks": benchmark_entries,
        },
        sort_keys=True,
        separators=(",", ":"),
    )
    return hashlib.sha256(payload.encode("utf-8")).hexdigest()[:16]


def _code_version() -> str | None:
    try:
        return subprocess.check_output(
            ["git", "rev-parse", "HEAD"],
            cwd=Path(__file__).resolve().parents[3],
            text=True,
        ).strip()
    except Exception:
        return None


def _intended_evaluation_count(
    *,
    config: RunConfig,
    generations: int,
    benchmark_count: int,
    fallback: int,
) -> int:
    if generations <= 0 or benchmark_count <= 0:
        return fallback
    return config.evolution.population_size * generations * benchmark_count


def _write_summary_json(
    output_dir: Path,
    manifest: dict[str, Any],
    results: list[dict[str, Any]],
    genomes: list[ModelGenome],
    latest_gen: int | None,
    config: RunConfig,
    best_per_benchmark: dict[str, dict[str, Any]] | None = None,
    lineage_records: list[dict[str, Any]] | None = None,
) -> None:
    """Write summary.json with cross-system durability contract fields."""
    best_fitness: dict[str, float] = {}
    for record in results:
        bid = record.get("benchmark_id", "")
        mv = record.get("metric_value")
        status = record.get("status", "")
        if status == "ok" and mv is not None:
            best_fitness[bid] = float(mv)

    param_counts = [g.parameter_estimate for g in genomes if g.parameter_estimate > 0]
    median_param_count = int(stat_median(param_counts)) if param_counts else 0

    qualities = list(best_fitness.values())
    median_quality = float(stat_median(qualities)) if qualities else None

    failure_count = sum(1 for r in results if r.get("status") != "ok")
    budget = manifest.get("budget", {})
    device = manifest.get("device", {})

    summary = {
        "system": "prism",
        "run_id": manifest["run_id"],
        "status": "complete",
        "total_evaluations": budget.get("evaluation_count", 0),
        "generations_completed": (latest_gen + 1) if latest_gen is not None else 0,
        "epochs_per_candidate": config.training.epochs,
        "population_size": config.evolution.population_size,
        "runtime_backend": device.get("framework", "unknown"),
        "runtime_version": device.get("framework_version") or "unknown",
        "precision_mode": device.get("precision_mode", "fp32"),
        "best_fitness": best_fitness,
        "median_parameter_count": median_param_count,
        "median_benchmark_quality": median_quality,
        "failure_count": failure_count,
        "benchmarks_evaluated": len(best_fitness),
        "operator_mix": _operator_mix(lineage_records or []),
        "family_benchmark_wins": _family_benchmark_wins(best_per_benchmark or {}, genomes),
        "failure_patterns": _failure_patterns(results),
    }
    (output_dir / "summary.json").write_text(
        json.dumps(summary, indent=2), encoding="utf-8",
    )


def _operator_mix(lineage_records: list[dict[str, Any]]) -> dict[str, int]:
    counts: dict[str, int] = {}
    for row in lineage_records:
        label = str(row.get("mutation_summary") or row.get("operator_kind") or "")
        if not label:
            continue
        counts[label] = counts.get(label, 0) + 1
    return dict(sorted(counts.items(), key=lambda item: (-item[1], item[0])))


def _family_benchmark_wins(
    best_per_benchmark: dict[str, dict[str, Any]],
    genomes: list[ModelGenome],
) -> dict[str, int]:
    genome_families = {genome.genome_id: genome.family for genome in genomes}
    counts: dict[str, int] = {}
    for best in best_per_benchmark.values():
        family = genome_families.get(best.get("genome_id", ""))
        if family is None:
            continue
        counts[family] = counts.get(family, 0) + 1
    return dict(sorted(counts.items(), key=lambda item: (-item[1], item[0])))


def _failure_patterns(results: list[dict[str, Any]]) -> dict[str, int]:
    counts: dict[str, int] = {}
    for row in results:
        reason = row.get("failure_reason") or (
            row.get("status") if row.get("status") not in {None, "ok"} else None
        )
        if reason is None:
            continue
        key = str(reason)
        counts[key] = counts.get(key, 0) + 1
    return dict(sorted(counts.items(), key=lambda item: (-item[1], item[0])))
