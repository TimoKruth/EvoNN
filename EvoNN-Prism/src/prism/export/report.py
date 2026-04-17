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
    store.close()

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
        if gid and q is not None and ev.get("failure_reason") is None:
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
        if gen is not None and q is not None and ev.get("failure_reason") is None:
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


def _compute_failure_patterns(evaluations: list[dict[str, Any]]) -> dict[str, int]:
    counts = Counter()
    for row in evaluations:
        reason = row.get("failure_reason")
        if reason:
            counts[str(reason)] += 1
    return dict(counts.most_common())
