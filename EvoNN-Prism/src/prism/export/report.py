"""Markdown report generation for completed Prism runs."""

from __future__ import annotations

import json
from collections import Counter
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

from prism.config import RunConfig, load_config
from prism.genome import ModelGenome
from prism.storage import RunStore


def generate_report(run_dir: str | Path, output_path: str | Path | None = None) -> str:
    """Generate a markdown report for a completed Prism run.

    Returns the markdown text. Optionally writes to output_path.
    """
    run_dir = Path(run_dir)
    config = _load_config(run_dir)
    store = RunStore(run_dir / "metrics.duckdb")
    run_id = _resolve_run_id(store)

    # Load data
    genome_rows = store.load_genomes(run_id)
    genomes: list[ModelGenome] = []
    for row in genome_rows:
        try:
            genomes.append(ModelGenome.model_validate(row))
        except Exception:
            pass

    evaluations = store.load_evaluations(run_id)
    best_per_benchmark = store.load_best_per_benchmark(run_id)
    latest_gen = store.latest_generation(run_id)
    lineage = store.load_lineage(run_id)
    archives = store.load_archives(run_id)
    store.close()
    runtime_meta = _load_runtime_metadata(run_dir)

    # Build sections
    sections: list[str] = []
    sections.append("# Prism Evolution Report")
    sections.append("")
    sections.append(f"**Run:** `{run_dir.name}`")
    sections.append(f"**Generated:** {datetime.now(timezone.utc).strftime('%Y-%m-%d %H:%M UTC')}")
    sections.append("")

    # Summary
    sections.append("## Summary")
    sections.append("")
    sections.append("| Metric | Value |")
    sections.append("|--------|-------|")
    sections.append(f"| Seed | {config.seed} |")
    sections.append(f"| Population Size | {config.evolution.population_size} |")
    sections.append(f"| Generations | {(latest_gen + 1) if latest_gen is not None else 0} |")
    sections.append(f"| Total Evaluations | {len(evaluations)} |")
    sections.append(f"| Epochs/Candidate | {config.training.epochs} |")
    sections.append(f"| Genomes Evolved | {len(genomes)} |")
    sections.append(f"| Benchmarks | {len(best_per_benchmark)} |")
    sections.append(f"| Runtime | {runtime_meta['runtime_backend']} |")
    sections.append(f"| Runtime Version | {runtime_meta['runtime_version']} |")
    sections.append(f"| Precision Mode | {runtime_meta['precision_mode']} |")
    if runtime_meta.get("wall_clock_seconds") is not None:
        sections.append(f"| Wall Clock Seconds | {float(runtime_meta['wall_clock_seconds']):.3f} |")
    sections.append("")

    # Best genome
    sections.append("## Best Genome")
    sections.append("")
    representative = _select_best(genomes, evaluations)
    if representative:
        sections.append("| Property | Value |")
        sections.append("|----------|-------|")
        sections.append(f"| Family | {representative.family} |")
        sections.append(f"| Hidden Layers | {representative.hidden_layers} |")
        sections.append(f"| Activation | {representative.activation} |")
        sections.append(f"| Dropout | {representative.dropout} |")
        sections.append(f"| Residual | {representative.residual} |")
        sections.append(f"| Norm Type | {representative.norm_type} |")
        sections.append(f"| Parameter Estimate | {representative.parameter_estimate:,} |")
        if representative.num_experts > 0:
            sections.append(f"| MoE Experts | {representative.num_experts} (top-{representative.moe_top_k}) |")
        sections.append("")
    else:
        sections.append("No genomes found.")
        sections.append("")

    # Evolution progress
    sections.append("## Evolution Progress")
    sections.append("")
    if latest_gen is not None and evaluations:
        gen_stats = _compute_generation_stats(evaluations, latest_gen)
        sections.append("| Generation | Best Quality | Avg Quality | Evaluations |")
        sections.append("|------------|-------------|-------------|-------------|")
        for gen, stats in sorted(gen_stats.items()):
            sections.append(
                f"| {gen} | {stats['best']:.6f} | {stats['avg']:.6f} | {stats['count']} |"
            )
        sections.append("")
    else:
        sections.append("No evolution data available.")
        sections.append("")

    # Family distribution
    sections.append("## Family Distribution")
    sections.append("")
    if genomes:
        family_counts = Counter(g.family for g in genomes)
        sections.append("| Family | Count | Share |")
        sections.append("|--------|-------|-------|")
        for family, count in family_counts.most_common():
            share = count / len(genomes) * 100
            sections.append(f"| {family} | {count} | {share:.1f}% |")
        sections.append("")
    else:
        sections.append("No genomes found.")
        sections.append("")

    # Family benchmark wins
    sections.append("## Family Benchmark Wins")
    sections.append("")
    family_wins = _compute_family_benchmark_wins(best_per_benchmark, genomes)
    if family_wins:
        sections.append("| Family | Benchmark Wins | Share |")
        sections.append("|--------|----------------|-------|")
        total_wins = sum(family_wins.values())
        for family, wins in family_wins.items():
            share = (wins / total_wins * 100.0) if total_wins else 0.0
            sections.append(f"| {family} | {wins} | {share:.1f}% |")
        sections.append("")
    else:
        sections.append("No benchmark leaders available.")
        sections.append("")

    # Operator mix
    sections.append("## Operator Mix")
    sections.append("")
    operator_mix = _compute_operator_mix(lineage)
    if operator_mix:
        sections.append("| Operator | Count | Share |")
        sections.append("|----------|-------|-------|")
        total_ops = sum(operator_mix.values())
        for operator, count in operator_mix.items():
            share = (count / total_ops * 100.0) if total_ops else 0.0
            sections.append(f"| {operator} | {count} | {share:.1f}% |")
        sections.append("")
    else:
        sections.append("No lineage data available.")
        sections.append("")

    sections.append("## Operator Success")
    sections.append("")
    operator_success = _compute_operator_success(evaluations, genomes, lineage)
    if operator_success:
        sections.append("| Operator | Family | Benchmark | Avg Quality | Evaluations |")
        sections.append("|----------|--------|-----------|-------------|-------------|")
        for row in operator_success:
            sections.append(
                f"| {row['operator']} | {row['family']} | {row['benchmark']} | "
                f"{row['avg_quality']:.6f} | {row['count']} |"
            )
        sections.append("")
    else:
        sections.append("No operator/evaluation overlap available.")
        sections.append("")

    sections.append("## Weight Inheritance")
    sections.append("")
    inheritance_summary = _compute_inheritance_summary(evaluations)
    if inheritance_summary:
        sections.append("| Metric | Value |")
        sections.append("|--------|-------|")
        sections.append(f"| Inheritance Hits | {inheritance_summary['hits']} |")
        sections.append(f"| Inheritance Rate | {inheritance_summary['rate']:.1f}% |")
        sections.append(f"| Avg Quality (Hit) | {inheritance_summary['avg_quality_hit']:.6f} |")
        sections.append(f"| Avg Quality (Miss) | {inheritance_summary['avg_quality_miss']:.6f} |")
        sections.append("")
    else:
        sections.append("No inheritance metadata available.")
        sections.append("")

    sections.append("## Efficiency")
    sections.append("")
    efficiency_summary = _compute_efficiency_summary(evaluations)
    if efficiency_summary:
        sections.append("| Metric | Value |")
        sections.append("|--------|-------|")
        sections.append(f"| Avg Train Seconds | {efficiency_summary['avg_train_seconds']:.4f} |")
        sections.append(f"| Avg Parameter Count | {efficiency_summary['avg_parameter_count']:.1f} |")
        sections.append(f"| Avg Quality / Second | {efficiency_summary['quality_per_second']:.6f} |")
        sections.append(f"| Avg Quality / 1k Params | {efficiency_summary['quality_per_kparam']:.6f} |")
        sections.append("")
    else:
        sections.append("No efficiency data available.")
        sections.append("")

    sections.append("## Family Efficiency")
    sections.append("")
    family_efficiency = _compute_family_efficiency(evaluations, genomes)
    if family_efficiency:
        sections.append("| Family | Avg Quality | Avg Time (s) | Avg Params | Quality / Second | Quality / 1k Params |")
        sections.append("|--------|-------------|--------------|------------|------------------|---------------------|")
        for row in family_efficiency:
            sections.append(
                f"| {row['family']} | {row['avg_quality']:.6f} | {row['avg_train_seconds']:.4f} | "
                f"{row['avg_parameter_count']:.1f} | {row['quality_per_second']:.6f} | {row['quality_per_kparam']:.6f} |"
            )
        sections.append("")
    else:
        sections.append("No family efficiency data available.")
        sections.append("")

    sections.append("## Operator Efficiency")
    sections.append("")
    operator_efficiency = _compute_operator_efficiency(evaluations, genomes, lineage)
    if operator_efficiency:
        sections.append("| Operator | Family | Avg Quality | Avg Time (s) | Avg Params | Quality / Second | Quality / 1k Params |")
        sections.append("|----------|--------|-------------|--------------|------------|------------------|---------------------|")
        for row in operator_efficiency:
            sections.append(
                f"| {row['operator']} | {row['family']} | {row['avg_quality']:.6f} | {row['avg_train_seconds']:.4f} | "
                f"{row['avg_parameter_count']:.1f} | {row['quality_per_second']:.6f} | {row['quality_per_kparam']:.6f} |"
            )
        sections.append("")
    else:
        sections.append("No operator efficiency data available.")
        sections.append("")

    sections.append("## Family Survival")
    sections.append("")
    family_survival = _compute_family_survival(evaluations, genomes)
    if family_survival:
        sections.append("| Generation | Families Active | Breakdown |")
        sections.append("|------------|-----------------|-----------|")
        for row in family_survival:
            sections.append(
                f"| {row['generation']} | {row['active_families']} | {row['breakdown']} |"
            )
        sections.append("")
    else:
        sections.append("No family survival data available.")
        sections.append("")

    sections.append("## Archive Turnover")
    sections.append("")
    archive_turnover = _compute_archive_turnover(archives)
    if archive_turnover:
        sections.append("| Generation | Archive Members | New Members | Retained |")
        sections.append("|------------|-----------------|-------------|----------|")
        for row in archive_turnover:
            sections.append(
                f"| {row['generation']} | {row['members']} | {row['new_members']} | {row['retained']} |"
            )
        sections.append("")
    else:
        sections.append("No archive history available.")
        sections.append("")

    sections.append("## Benchmark Specialists")
    sections.append("")
    specialist_summary = _compute_specialist_summary(archives)
    if specialist_summary:
        sections.append("| Benchmark | Families With Specialists | Top Families |")
        sections.append("|-----------|--------------------------|--------------|")
        for row in specialist_summary:
            sections.append(
                f"| {row['benchmark']} | {row['family_count']} | {row['families']} |"
            )
        sections.append("")
    else:
        sections.append("No specialist archive data available.")
        sections.append("")

    # Failure patterns
    sections.append("## Failure Patterns")
    sections.append("")
    failure_patterns = _compute_failure_patterns(evaluations)
    if failure_patterns:
        sections.append("| Failure | Count |")
        sections.append("|---------|-------|")
        for failure, count in failure_patterns.items():
            sections.append(f"| {failure} | {count} |")
        sections.append("")
    else:
        sections.append("No failed evaluations recorded.")
        sections.append("")

    sections.append("## Failure Heatmap")
    sections.append("")
    failure_heatmap = _compute_failure_heatmap(evaluations)
    if failure_heatmap:
        failure_columns = sorted({failure for row in failure_heatmap for failure in row["failures"]})
        sections.append("| Benchmark | " + " | ".join(failure_columns) + " |")
        sections.append("|" + "---|" * (len(failure_columns) + 1))
        for row in failure_heatmap:
            values = [str(row["failures"].get(column, 0)) for column in failure_columns]
            sections.append(f"| {row['benchmark']} | " + " | ".join(values) + " |")
        sections.append("")
    else:
        sections.append("No benchmark failure hotspots recorded.")
        sections.append("")

    # Per-benchmark results
    sections.append("## Per-Benchmark Results")
    sections.append("")
    if best_per_benchmark:
        sections.append("| Benchmark | Best Quality | Metric | Params | Time (s) |")
        sections.append("|-----------|-------------|--------|--------|----------|")
        for bid, best in sorted(best_per_benchmark.items()):
            quality = best.get("quality", 0.0)
            metric = best.get("metric_name", "?")
            params = best.get("parameter_count", "?")
            time_s = best.get("train_seconds")
            time_str = f"{time_s:.2f}" if time_s is not None else "?"
            sections.append(f"| {bid} | {quality:.6f} | {metric} | {params} | {time_str} |")
        sections.append("")
    else:
        sections.append("No benchmark results available.")
        sections.append("")

    text = "\n".join(sections)

    if output_path:
        output_path = Path(output_path)
        output_path.parent.mkdir(parents=True, exist_ok=True)
        output_path.write_text(text, encoding="utf-8")

    return text


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------


def _load_config(run_dir: Path) -> RunConfig:
    config_path = run_dir / "config.yaml"
    if config_path.exists():
        return load_config(config_path)
    config_json = run_dir / "config.json"
    if config_json.exists():
        data = json.loads(config_json.read_text(encoding="utf-8"))
        return RunConfig.model_validate(data)
    return RunConfig()


def _resolve_run_id(store: RunStore) -> str:
    row = store.conn.execute(
        "SELECT run_id FROM runs ORDER BY created_at DESC LIMIT 1"
    ).fetchone()
    if row:
        return row[0]
    return "default"


def _load_runtime_metadata(run_dir: Path) -> dict[str, Any]:
    summary_path = run_dir / "summary.json"
    if summary_path.exists():
        try:
            summary = json.loads(summary_path.read_text(encoding="utf-8"))
        except json.JSONDecodeError:
            summary = {}
    else:
        summary = {}
    return {
        "runtime_backend": str(summary.get("runtime_backend") or "unknown"),
        "runtime_version": str(summary.get("runtime_version") or "unknown"),
        "precision_mode": str(summary.get("precision_mode") or "fp32"),
        "wall_clock_seconds": summary.get("wall_clock_seconds", summary.get("elapsed_seconds")),
    }


def _select_best(
    genomes: list[ModelGenome],
    evaluations: list[dict],
) -> ModelGenome | None:
    if not genomes:
        return None

    genome_qualities: dict[str, list[float]] = {}
    for ev in evaluations:
        gid = ev.get("genome_id", "")
        q = ev.get("quality")
        if gid and q is not None and _is_successful_evaluation(ev):
            genome_qualities.setdefault(gid, []).append(float(q))

    if not genome_qualities:
        return genomes[0]

    best_id = max(
        genome_qualities,
        key=lambda gid: sum(genome_qualities[gid]) / len(genome_qualities[gid]),
    )
    for g in genomes:
        if g.genome_id == best_id:
            return g
    return genomes[0]


def _compute_generation_stats(
    evaluations: list[dict], latest_gen: int,
) -> dict[int, dict[str, Any]]:
    """Compute per-generation quality statistics."""
    gen_data: dict[int, list[float]] = {}
    gen_counts: dict[int, int] = {}

    for ev in evaluations:
        gen = ev.get("generation")
        q = ev.get("quality")
        if gen is not None and q is not None and _is_successful_evaluation(ev):
            gen_data.setdefault(gen, []).append(float(q))
            gen_counts[gen] = gen_counts.get(gen, 0) + 1

    stats: dict[int, dict[str, Any]] = {}
    for gen, qualities in gen_data.items():
        stats[gen] = {
            "best": max(qualities),
            "avg": sum(qualities) / len(qualities),
            "count": gen_counts.get(gen, 0),
        }
    return stats


def _compute_family_benchmark_wins(
    best_per_benchmark: dict[str, dict[str, Any]],
    genomes: list[ModelGenome],
) -> dict[str, int]:
    genome_families = {genome.genome_id: genome.family for genome in genomes}
    counts = Counter()
    for best in best_per_benchmark.values():
        family = genome_families.get(best.get("genome_id", ""))
        if family:
            counts[family] += 1
    return dict(counts.most_common())


def _compute_operator_mix(lineage: list[dict[str, Any]]) -> dict[str, int]:
    counts = Counter()
    for row in lineage:
        label = row.get("mutation_summary") or row.get("operator_kind")
        if label:
            counts[str(label)] += 1
    return dict(counts.most_common())


def _failure_label(row: dict[str, Any]) -> str | None:
    reason = row.get("failure_reason")
    if reason:
        return str(reason)
    status = row.get("status")
    if status in {None, "ok"}:
        return None
    return str(status)


def _is_successful_evaluation(row: dict[str, Any]) -> bool:
    return _failure_label(row) is None


def _compute_failure_patterns(evaluations: list[dict[str, Any]]) -> dict[str, int]:
    counts = Counter()
    for row in evaluations:
        reason = _failure_label(row)
        if reason:
            counts[reason] += 1
    return dict(counts.most_common())


def _compute_operator_success(
    evaluations: list[dict[str, Any]],
    genomes: list[ModelGenome],
    lineage: list[dict[str, Any]],
) -> list[dict[str, Any]]:
    family_by_genome = {genome.genome_id: genome.family for genome in genomes}
    operator_by_genome = {
        row["genome_id"]: str(row.get("mutation_summary") or row.get("operator_kind") or "unknown")
        for row in lineage
    }
    grouped: dict[tuple[str, str, str], list[float]] = {}
    for row in evaluations:
        if not _is_successful_evaluation(row):
            continue
        genome_id = row.get("genome_id", "")
        operator = operator_by_genome.get(genome_id)
        family = family_by_genome.get(genome_id)
        benchmark = row.get("benchmark_id")
        quality = row.get("quality")
        if operator is None or family is None or benchmark is None or quality is None:
            continue
        grouped.setdefault((operator, family, str(benchmark)), []).append(float(quality))

    rows = []
    for (operator, family, benchmark), qualities in grouped.items():
        rows.append({
            "operator": operator,
            "family": family,
            "benchmark": benchmark,
            "avg_quality": sum(qualities) / len(qualities),
            "count": len(qualities),
        })
    rows.sort(key=lambda row: (-row["avg_quality"], row["operator"], row["family"], row["benchmark"]))
    return rows[:12]


def _compute_inheritance_summary(evaluations: list[dict[str, Any]]) -> dict[str, Any] | None:
    total = len(evaluations)
    if total == 0:
        return None
    hits = [row for row in evaluations if row.get("inheritance_hit")]
    misses = [row for row in evaluations if not row.get("inheritance_hit")]
    hit_qualities = [float(row["quality"]) for row in hits if row.get("quality") is not None]
    miss_qualities = [float(row["quality"]) for row in misses if row.get("quality") is not None]
    return {
        "hits": len(hits),
        "rate": (len(hits) / total) * 100.0,
        "avg_quality_hit": sum(hit_qualities) / len(hit_qualities) if hit_qualities else float("nan"),
        "avg_quality_miss": sum(miss_qualities) / len(miss_qualities) if miss_qualities else float("nan"),
    }


def _compute_efficiency_summary(evaluations: list[dict[str, Any]]) -> dict[str, float] | None:
    valid = [row for row in evaluations if _is_successful_evaluation(row)]
    if not valid:
        return None
    avg_quality = sum(float(row.get("quality") or 0.0) for row in valid) / len(valid)
    avg_time = sum(float(row.get("train_seconds") or 0.0) for row in valid) / len(valid)
    avg_params = sum(float(row.get("parameter_count") or 0.0) for row in valid) / len(valid)
    return {
        "avg_quality": avg_quality,
        "avg_train_seconds": avg_time,
        "avg_parameter_count": avg_params,
        "quality_per_second": avg_quality / max(1e-9, avg_time),
        "quality_per_kparam": avg_quality / max(1.0, avg_params / 1000.0),
    }


def _compute_family_efficiency(
    evaluations: list[dict[str, Any]],
    genomes: list[ModelGenome],
) -> list[dict[str, Any]]:
    family_by_genome = {genome.genome_id: genome.family for genome in genomes}
    grouped: dict[str, list[dict[str, Any]]] = {}
    for row in evaluations:
        if not _is_successful_evaluation(row):
            continue
        family = family_by_genome.get(row.get("genome_id", ""))
        if family is None:
            continue
        grouped.setdefault(family, []).append(row)
    return _efficiency_rows(grouped, group_label="family")


def _compute_operator_efficiency(
    evaluations: list[dict[str, Any]],
    genomes: list[ModelGenome],
    lineage: list[dict[str, Any]],
) -> list[dict[str, Any]]:
    family_by_genome = {genome.genome_id: genome.family for genome in genomes}
    operator_by_genome = {
        row["genome_id"]: str(row.get("mutation_summary") or row.get("operator_kind") or "unknown")
        for row in lineage
    }
    grouped: dict[tuple[str, str], list[dict[str, Any]]] = {}
    for row in evaluations:
        if not _is_successful_evaluation(row):
            continue
        genome_id = row.get("genome_id", "")
        operator = operator_by_genome.get(genome_id)
        family = family_by_genome.get(genome_id)
        if operator is None or family is None:
            continue
        grouped.setdefault((operator, family), []).append(row)

    rows = []
    for (operator, family), items in grouped.items():
        payload = _aggregate_efficiency(items)
        rows.append({
            "operator": operator,
            "family": family,
            **payload,
        })
    rows.sort(key=lambda row: (-row["quality_per_second"], -row["quality_per_kparam"], row["operator"], row["family"]))
    return rows[:12]


def _compute_family_survival(
    evaluations: list[dict[str, Any]],
    genomes: list[ModelGenome],
) -> list[dict[str, Any]]:
    family_by_genome = {genome.genome_id: genome.family for genome in genomes}
    grouped: dict[int, Counter] = {}
    for row in evaluations:
        generation = row.get("generation")
        genome_id = row.get("genome_id")
        if generation is None or genome_id not in family_by_genome:
            continue
        grouped.setdefault(int(generation), Counter())[family_by_genome[genome_id]] += 1

    rows = []
    for generation in sorted(grouped):
        counter = grouped[generation]
        rows.append({
            "generation": generation,
            "active_families": len(counter),
            "breakdown": ", ".join(f"{family}({count})" for family, count in counter.most_common()),
        })
    return rows


def _compute_archive_turnover(archives: list[dict[str, Any]]) -> list[dict[str, Any]]:
    grouped: dict[int, set[str]] = {}
    for row in archives:
        generation = row.get("generation")
        genome_id = row.get("genome_id")
        if generation is None or genome_id is None:
            continue
        grouped.setdefault(int(generation), set()).add(str(genome_id))

    rows = []
    previous: set[str] = set()
    for generation in sorted(grouped):
        members = grouped[generation]
        rows.append({
            "generation": generation,
            "members": len(members),
            "new_members": len(members - previous),
            "retained": len(members & previous),
        })
        previous = members
    return rows


def _compute_failure_heatmap(evaluations: list[dict[str, Any]]) -> list[dict[str, Any]]:
    grouped: dict[str, Counter] = {}
    for row in evaluations:
        failure = _failure_label(row)
        benchmark = row.get("benchmark_id")
        if not failure or benchmark is None:
            continue
        failure_kind = str(failure).split(":", 1)[0]
        grouped.setdefault(str(benchmark), Counter())[failure_kind] += 1

    rows = []
    for benchmark, failures in sorted(grouped.items()):
        rows.append({"benchmark": benchmark, "failures": dict(failures)})
    return rows


def _compute_specialist_summary(archives: list[dict[str, Any]]) -> list[dict[str, Any]]:
    grouped: dict[str, set[str]] = {}
    for row in archives:
        archive_kind = str(row.get("archive_kind", ""))
        if not archive_kind.startswith("specialist:"):
            continue
        _, benchmark, family = archive_kind.split(":", 2)
        grouped.setdefault(benchmark, set()).add(family)
    return [
        {
            "benchmark": benchmark,
            "family_count": len(families),
            "families": ", ".join(sorted(families)),
        }
        for benchmark, families in sorted(grouped.items())
    ]


def _efficiency_rows(
    grouped: dict[str, list[dict[str, Any]]],
    *,
    group_label: str,
) -> list[dict[str, Any]]:
    rows = []
    for label, items in grouped.items():
        payload = _aggregate_efficiency(items)
        rows.append({
            group_label: label,
            **payload,
        })
    rows.sort(key=lambda row: (-row["quality_per_second"], -row["quality_per_kparam"], row[group_label]))
    return rows


def _aggregate_efficiency(items: list[dict[str, Any]]) -> dict[str, float]:
    avg_quality = sum(float(row.get("quality") or 0.0) for row in items) / len(items)
    avg_time = sum(float(row.get("train_seconds") or 0.0) for row in items) / len(items)
    avg_params = sum(float(row.get("parameter_count") or 0.0) for row in items) / len(items)
    return {
        "avg_quality": avg_quality,
        "avg_train_seconds": avg_time,
        "avg_parameter_count": avg_params,
        "quality_per_second": avg_quality / max(1e-9, avg_time),
        "quality_per_kparam": avg_quality / max(1.0, avg_params / 1000.0),
    }
