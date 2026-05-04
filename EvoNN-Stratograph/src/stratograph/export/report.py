"""Report rendering for prototype runs."""
from __future__ import annotations

import json
from collections import Counter
from pathlib import Path
from typing import Any

from stratograph.genome import dict_to_genome
from stratograph.storage import RunStore


def load_runtime_metadata(budget_meta: dict[str, Any]) -> dict[str, str]:
    """Normalize persisted runtime metadata for CLI/report/export surfaces."""
    runtime_policy = budget_meta.get("runtime_execution_policy") or {}
    return {
        "requested_runtime_backend": str(budget_meta.get("runtime_backend_requested") or "auto"),
        "runtime_backend": str(budget_meta.get("runtime_backend") or "unknown"),
        "runtime_version": str(budget_meta.get("runtime_version") or "unknown"),
        "precision_mode": str(budget_meta.get("precision_mode") or "fp32"),
        "runtime_backend_limitations": str(budget_meta.get("runtime_backend_limitations") or ""),
        "runtime_policy_name": str(runtime_policy.get("runtime_policy_name") or "unknown"),
    }


def load_report_context(run_dir: str | Path) -> dict[str, Any]:
    """Load one run plus derived reporting summaries from the run DB."""
    run_dir = Path(run_dir)
    status_path = run_dir / "status.json"
    checkpoint_path = run_dir / "checkpoint.json"
    status_payload = json.loads(status_path.read_text(encoding="utf-8")) if status_path.exists() else {}

    with RunStore(run_dir / "metrics.duckdb") as store:
        runs = store.load_runs()
        if not runs:
            raise ValueError(f"No runs found in {run_dir}")

        run = runs[0]
        results = store.load_results(run["run_id"])
        genomes = store.load_genomes(run["run_id"])
        budget_meta = store.load_budget_metadata(run["run_id"])

    ok_results = [record for record in results if record["status"] == "ok"]
    non_ok_results = [record for record in results if record["status"] != "ok"]
    failed_results = [record for record in results if record["status"] == "failed"]
    skipped_results = [record for record in results if record["status"] == "skipped"]

    def quality_key(record: dict[str, Any]) -> tuple[float, str]:
        quality = record.get("quality")
        return (float(quality) if quality is not None else float("-inf"), str(record.get("benchmark_name") or ""))

    best_by_benchmark: dict[str, dict[str, Any]] = {}
    for record in ok_results:
        benchmark_name = str(record.get("benchmark_name") or "")
        current = best_by_benchmark.get(benchmark_name)
        if current is None or quality_key(record) > quality_key(current):
            best_by_benchmark[benchmark_name] = record
    best_results = sorted(best_by_benchmark.values(), key=quality_key, reverse=True)
    representative_genome = _select_representative_genome(genomes, best_results)
    hierarchy_leaders = summarize_hierarchy_leaders(genomes, representative_genome=representative_genome)
    hierarchy_evidence = summarize_hierarchy_evidence(genomes)

    return {
        "run_dir": run_dir,
        "run": run,
        "results": results,
        "genomes": genomes,
        "budget_meta": budget_meta,
        "status": status_payload,
        "status_path": status_path,
        "checkpoint_path": checkpoint_path,
        "ok_results": ok_results,
        "non_ok_results": non_ok_results,
        "failed_results": failed_results,
        "skipped_results": skipped_results,
        "best_results": best_results,
        "representative_genome": representative_genome,
        "hierarchy_leaders": hierarchy_leaders,
        "hierarchy_evidence": hierarchy_evidence,
    }


def _select_representative_genome(
    genomes: list[dict[str, Any]],
    best_results: list[dict[str, Any]],
) -> Any | None:
    """Return the strongest available hierarchical genome for report/inspect surfaces."""
    if not genomes:
        return None

    genomes_by_id = {
        str(record.get("genome_id")): record
        for record in genomes
        if record.get("genome_id") not in {None, ""}
    }
    for result in best_results:
        genome_id = result.get("genome_id")
        if genome_id in {None, ""}:
            continue
        payload_record = genomes_by_id.get(str(genome_id))
        if payload_record is None:
            continue
        decoded = _decode_genome_payload(payload_record)
        if decoded is not None:
            return decoded

    for record in genomes:
        decoded = _decode_genome_payload(record)
        if decoded is not None:
            return decoded

    return None


def _decode_genome_payload(record: dict[str, Any]) -> Any | None:
    try:
        return dict_to_genome(record["payload"])
    except Exception:
        return None


def summarize_hierarchy_leaders(
    genomes: list[dict[str, Any]],
    *,
    representative_genome: Any | None = None,
) -> dict[str, dict[str, Any]]:
    decoded: list[Any] = []
    for record in genomes:
        genome = _decode_genome_payload(record)
        if genome is not None:
            decoded.append(genome)
    if representative_genome is not None and all(genome.genome_id != representative_genome.genome_id for genome in decoded):
        decoded.append(representative_genome)
    if not decoded:
        return {}

    def _leader_payload(genome: Any, value: float | int, notes: str) -> dict[str, Any]:
        return {
            "genome_id": genome.genome_id,
            "value": float(value) if isinstance(value, float) else value,
            "notes": notes,
        }

    highest_reuse = max(decoded, key=lambda genome: (genome.reuse_ratio, genome.macro_depth, -len(genome.cell_library)))
    deepest_macro = max(decoded, key=lambda genome: (genome.macro_depth, genome.reuse_ratio, len(genome.macro_nodes)))
    largest_cell_library = max(decoded, key=lambda genome: (len(genome.cell_library), genome.reuse_ratio, genome.macro_depth))

    leaders = {
        "highest_reuse": _leader_payload(
            highest_reuse,
            highest_reuse.reuse_ratio,
            f"cell_library={len(highest_reuse.cell_library)} macro_depth={highest_reuse.macro_depth}",
        ),
        "deepest_macro": _leader_payload(
            deepest_macro,
            deepest_macro.macro_depth,
            f"reuse_ratio={deepest_macro.reuse_ratio:.4f} macro_nodes={len(deepest_macro.macro_nodes)}",
        ),
        "largest_cell_library": _leader_payload(
            largest_cell_library,
            len(largest_cell_library.cell_library),
            f"reuse_ratio={largest_cell_library.reuse_ratio:.4f} avg_cell_depth={largest_cell_library.average_cell_depth:.2f}",
        ),
    }
    if representative_genome is not None:
        leaders["representative"] = _leader_payload(
            representative_genome,
            representative_genome.reuse_ratio,
            f"macro_depth={representative_genome.macro_depth} avg_cell_depth={representative_genome.average_cell_depth:.2f}",
        )
    return leaders


def summarize_hierarchy_evidence(genomes: list[dict[str, Any]]) -> dict[str, Any]:
    """Summarize hierarchy-specific signals across stored winner genomes."""
    decoded = [genome for record in genomes if (genome := _decode_genome_payload(record)) is not None]
    if not decoded:
        return {
            "genome_count": 0,
            "mean_reuse_ratio": 0.0,
            "max_reuse_ratio": 0.0,
            "mean_macro_depth": 0.0,
            "mean_cell_library_size": 0.0,
            "unique_motif_count": 0,
            "repeated_motif_count": 0,
        }

    motif_counts: Counter[str] = Counter()
    for genome in decoded:
        used_cell_ids = {node.cell_id for node in genome.macro_nodes}
        for cell_id in used_cell_ids:
            cell = genome.cell_library[cell_id]
            motif_counts[_cell_signature(cell)] += 1

    return {
        "genome_count": len(decoded),
        "mean_reuse_ratio": sum(genome.reuse_ratio for genome in decoded) / len(decoded),
        "max_reuse_ratio": max(genome.reuse_ratio for genome in decoded),
        "mean_macro_depth": sum(genome.macro_depth for genome in decoded) / len(decoded),
        "mean_cell_library_size": sum(len(genome.cell_library) for genome in decoded) / len(decoded),
        "unique_motif_count": len(motif_counts),
        "repeated_motif_count": sum(1 for count in motif_counts.values() if count > 1),
    }


def _cell_signature(cell: Any) -> str:
    node_sig = ",".join(f"{node.kind.value}:{node.activation.value}:{node.width}" for node in cell.nodes)
    edge_sig = ",".join(
        f"{edge.source}->{edge.target}" for edge in sorted(cell.edges, key=lambda item: (item.source, item.target))
    )
    return f"{node_sig}|{edge_sig}"



def _render_metric(value: Any) -> str:
    if value is None:
        return "---"
    return f"{float(value):.6f}"



def _render_quality(value: Any) -> str:
    if value is None:
        return "---"
    return f"{float(value):.4f}"



def _render_seconds(value: Any) -> str:
    if value is None:
        return "---"
    return f"{float(value):.3f}"


def _escape_markdown_cell(value: Any) -> str:
    return str(value).replace("|", "\\|").replace("\n", "<br>")


def summarize_failure_patterns(non_ok_results: list[dict[str, Any]]) -> list[tuple[str, int]]:
    """Return compact failure-pattern counts for inspect/report parity."""
    counts = Counter(
        str(record.get("failure_reason") or record.get("status") or "unknown")
        for record in non_ok_results
    )
    return sorted(counts.items(), key=lambda item: (-item[1], item[0]))



def write_report(run_dir: str | Path) -> Path:
    """Render prototype markdown report from run DB."""
    context = load_report_context(run_dir)
    run_dir = context["run_dir"]
    run = context["run"]
    results = context["results"]
    genomes = context["genomes"]
    budget_meta = context["budget_meta"]
    status_payload = context["status"]
    runtime_meta = load_runtime_metadata(budget_meta)
    ok_results = context["ok_results"]
    non_ok_results = context["non_ok_results"]
    failed_results = context["failed_results"]
    skipped_results = context["skipped_results"]
    best_results = context["best_results"]
    representative_genome = context["representative_genome"]
    hierarchy_leaders = context["hierarchy_leaders"]
    hierarchy_evidence = context["hierarchy_evidence"]
    failure_patterns = summarize_failure_patterns(non_ok_results)

    lines = [
        "# Stratograph Prototype Report",
        "",
        f"- Run ID: `{run['run_id']}`",
        f"- Seed: `{run['seed']}`",
        f"- Created At: `{budget_meta.get('created_at') or run.get('created_at') or 'unknown'}`",
        f"- Run State: `{status_payload.get('state', 'unknown')}`",
        f"- Runtime: `{runtime_meta['runtime_backend']}`",
        f"- Requested Runtime: `{runtime_meta['requested_runtime_backend']}`",
        f"- Runtime Version: `{runtime_meta['runtime_version']}`",
        f"- Runtime Policy: `{runtime_meta['runtime_policy_name']}`",
        f"- Precision Mode: `{runtime_meta['precision_mode']}`",
        f"- Architecture Mode: `{budget_meta.get('architecture_mode', 'unknown')}`",
        f"- Benchmarks: `{len(results)}`",
        f"- Genomes Stored: `{len(genomes)}`",
        f"- Effective Training Epochs: `{budget_meta.get('effective_training_epochs', 'unknown')}`",
        f"- Wall Clock Seconds: `{_render_seconds(budget_meta.get('wall_clock_seconds'))}`",
        f"- Status Mix: `ok={len(ok_results)}, skipped={len(skipped_results)}, failed={len(failed_results)}`",
        f"- Completed Benchmarks: `{status_payload.get('completed_count', len(ok_results) + len(skipped_results) + len(failed_results))}/{status_payload.get('total_benchmarks', len(results))}`",
        f"- Remaining Benchmarks: `{status_payload.get('remaining_count', 0)}`",
        f"- Novelty Mean: `{budget_meta.get('novelty_score_mean', 0.0):.4f}`",
        f"- Occupied Niches: `{budget_meta.get('map_elites_occupied_niches', 0)}`",
        f"- Parent Selection: `{budget_meta.get('parent_selection_strategy', 'unknown')}`",
        f"- Mutation Pressure: `{budget_meta.get('mutation_pressure', 'unknown')}`",
        f"- Hierarchy Policy: `{budget_meta.get('hierarchy_selection_policy', 'unknown')}`",
        f"- Slot Integrity: `{(budget_meta.get('benchmark_slot_integrity') or {}).get('matches_evaluation_count', 'unknown')}`",
    ]
    if context["status_path"].exists():
        lines.append(f"- Status Artifact: `{context['status_path'].name}`")
    if context["checkpoint_path"].exists():
        lines.append(f"- Checkpoint Artifact: `{context['checkpoint_path'].name}`")
    if runtime_meta["runtime_backend_limitations"]:
        lines.append(f"- Runtime Limitations: `{runtime_meta['runtime_backend_limitations']}`")
    if representative_genome is not None:
        lines.extend([
            "",
            "## Hierarchy Summary",
            "",
            "| Property | Value |",
            "|---|---|",
            f"| Representative Genome | `{representative_genome.genome_id}` |",
            f"| Macro Nodes | `{len(representative_genome.macro_nodes)}` |",
            f"| Enabled Macro Edges | `{sum(1 for edge in representative_genome.macro_edges if edge.enabled)}` |",
            f"| Cell Library Size | `{len(representative_genome.cell_library)}` |",
            f"| Macro Depth | `{representative_genome.macro_depth}` |",
            f"| Avg Cell Depth | `{representative_genome.average_cell_depth:.2f}` |",
            f"| Reuse Ratio | `{representative_genome.reuse_ratio:.4f}` |",
        ])
    if hierarchy_leaders:
        lines.extend([
            "",
            "## Hierarchy Leaders",
            "",
            "| Leader Type | Genome | Value | Notes |",
            "|---|---|---:|---|",
        ])
        for leader_type, payload in hierarchy_leaders.items():
            leader_value = payload.get("value")
            rendered_value = f"{float(leader_value):.4f}" if isinstance(leader_value, float) else str(leader_value)
            lines.append(
                f"| {leader_type} | `{payload.get('genome_id', 'unknown')}` | {rendered_value} | {_escape_markdown_cell(payload.get('notes') or '')} |"
            )
    lines.extend([
        "",
        "## Hierarchy Evidence",
        "",
        "| Signal | Value |",
        "|---|---:|",
        f"| Winner Genomes | {hierarchy_evidence['genome_count']} |",
        f"| Mean Reuse Ratio | {hierarchy_evidence['mean_reuse_ratio']:.4f} |",
        f"| Max Reuse Ratio | {hierarchy_evidence['max_reuse_ratio']:.4f} |",
        f"| Mean Macro Depth | {hierarchy_evidence['mean_macro_depth']:.2f} |",
        f"| Mean Cell Library Size | {hierarchy_evidence['mean_cell_library_size']:.2f} |",
        f"| Unique Motifs | {hierarchy_evidence['unique_motif_count']} |",
        f"| Repeated Motifs | {hierarchy_evidence['repeated_motif_count']} |",
    ])
    slot_plan = budget_meta.get("benchmark_slot_plan") or status_payload.get("benchmark_slot_plan") or []
    if slot_plan:
        lines.extend([
            "",
            "## Benchmark Slot Plan",
            "",
            "| Benchmark | State | Configured | Completed | Failed | Invalid |",
            "|---|---|---:|---:|---:|---:|",
        ])
        for item in slot_plan:
            lines.append(
                f"| {_escape_markdown_cell(item.get('benchmark_name', 'unknown'))} | "
                f"{_escape_markdown_cell(item.get('state', 'unknown'))} | "
                f"{int(item.get('configured_slots', 0) or 0)} | "
                f"{int(item.get('completed_slots', 0) or 0)} | "
                f"{int(item.get('failed_slots', 0) or 0)} | "
                f"{int(item.get('invalid_slots', 0) or 0)} |"
            )
    lines.extend([
        "",
        "## Benchmarks",
        "",
        "| Benchmark | Metric | Direction | Status | Notes |",
        "|---|---|---|---|---|",
    ])
    for record in results:
        lines.append(
            f"| {record['benchmark_name']} | {record['metric_name']} | "
            f"{record['metric_direction']} | {record['status']} | "
            f"{record['failure_reason'] or record['architecture_summary'] or ''} |"
        )

    lines.extend(
        [
            "",
            "## Best Benchmarks",
            "",
            "| Benchmark | Metric | Value | Quality | Params | Train Seconds | Genome | Architecture |",
            "|---|---|---:|---:|---:|---:|---|---|",
        ]
    )
    if best_results:
        for record in best_results:
            lines.append(
                "| {benchmark} | {metric} | {value} | {quality} | {params} | {seconds} | {genome} | {architecture} |".format(
                    benchmark=record["benchmark_name"],
                    metric=record["metric_name"],
                    value=_render_metric(record.get("metric_value")),
                    quality=_render_quality(record.get("quality")),
                    params=record.get("parameter_count") if record.get("parameter_count") is not None else "---",
                    seconds=_render_seconds(record.get("train_seconds")),
                    genome=record.get("genome_id") or "—",
                    architecture=record.get("architecture_summary") or "—",
                )
            )
    else:
        lines.append("| none | — | --- | --- | --- | --- | — | — |")

    lines.extend([
        "",
        "## Failure Patterns",
        "",
        "| Reason | Count |",
        "|---|---:|",
    ])
    if failure_patterns:
        for reason, count in failure_patterns:
            lines.append(f"| {_escape_markdown_cell(reason)} | {count} |")
    else:
        lines.append("| none | 0 |")

    lines.extend([
        "",
        "## Failure Details",
        "",
        "| Benchmark | Reason |",
        "|---|---|",
    ])
    if failed_results:
        for record in failed_results:
            lines.append(
                f"| {_escape_markdown_cell(record['benchmark_name'])} | "
                f"{_escape_markdown_cell(record.get('failure_reason') or 'unknown')} |"
            )
    else:
        lines.append("| none | no failed benchmarks |")

    path = run_dir / "report.md"
    path.write_text("\n".join(lines) + "\n", encoding="utf-8")
    return path
