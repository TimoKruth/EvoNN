"""Reporting helpers for Primordia exports."""
from __future__ import annotations

import json
from collections import Counter
from pathlib import Path
from typing import Any


def _escape_markdown_cell(value: Any) -> str:
    return str(value).replace("|", "\\|").replace("\n", "<br>")


def _render_markdown_cell(value: Any, missing: str = "—") -> str:
    if value is None or value == "":
        return missing
    return _escape_markdown_cell(value)


def load_runtime_metadata(summary: dict[str, Any]) -> dict[str, Any]:
    """Return normalized runtime metadata from run summary artifacts."""

    return {
        "runtime": str(summary.get("runtime") or "unknown"),
        "runtime_backend_requested": str(
            summary.get("runtime_backend_requested") or summary.get("runtime") or "unknown"
        ),
        "runtime_version": str(summary.get("runtime_version") or "unknown"),
        "runtime_backend_limitations": str(summary.get("runtime_backend_limitations") or ""),
        "runtime_execution_policy": summary.get("runtime_execution_policy"),
        "precision_mode": str(summary.get("precision_mode") or "fp32"),
    }


def _metric_values_match(left: Any, right: Any) -> bool:
    if left is None or right is None:
        return False
    return abs(float(left) - float(right)) <= 1e-12


def enrich_best_results(
    best_results: list[dict[str, Any]],
    trial_records: list[dict[str, Any]],
) -> list[dict[str, Any]]:
    """Fill missing winner metadata from trial artifacts for backward-compatible reports."""

    trial_by_genome: dict[str, dict[str, Any]] = {}
    trial_by_benchmark_primitive: dict[tuple[str, str], list[dict[str, Any]]] = {}
    trial_by_benchmark: dict[str, list[dict[str, Any]]] = {}
    for record in trial_records:
        genome_id = record.get("genome_id")
        if genome_id:
            trial_by_genome[str(genome_id)] = record
        benchmark_name = str(record.get("benchmark_name") or "")
        primitive_name = str(record.get("primitive_name") or "")
        if benchmark_name and primitive_name:
            trial_by_benchmark_primitive.setdefault((benchmark_name, primitive_name), []).append(record)
        if benchmark_name:
            trial_by_benchmark.setdefault(benchmark_name, []).append(record)

    enriched: list[dict[str, Any]] = []
    for record in best_results:
        match = None
        genome_id = record.get("genome_id")
        if genome_id:
            match = trial_by_genome.get(str(genome_id))
        if match is None:
            benchmark_name = str(record.get("benchmark_name") or "")
            primitive_name = str(record.get("primitive_name") or "")
            benchmark_candidates = trial_by_benchmark.get(benchmark_name, [])
            if benchmark_name and primitive_name:
                primitive_candidates = trial_by_benchmark_primitive.get((benchmark_name, primitive_name), [])
                if len(primitive_candidates) == 1:
                    match = primitive_candidates[0]
            if match is None and benchmark_candidates:
                metric_name = record.get("metric_name")
                status = record.get("status")
                metric_value = record.get("metric_value")
                precise_candidates = [
                    candidate for candidate in benchmark_candidates
                    if (metric_name in {None, ""} or candidate.get("metric_name") == metric_name)
                    and (status in {None, ""} or candidate.get("status") == status)
                    and (
                        metric_value is None
                        or _metric_values_match(candidate.get("metric_value"), metric_value)
                    )
                ]
                if len(precise_candidates) == 1:
                    match = precise_candidates[0]
                elif len(benchmark_candidates) == 1:
                    match = benchmark_candidates[0]

        merged = dict(record)
        if match is not None:
            for field in (
                "primitive_family",
                "benchmark_group",
                "architecture_summary",
                "genome_id",
                "runtime",
                "runtime_version",
            ):
                if merged.get(field) in {None, ""} and match.get(field) not in {None, ""}:
                    merged[field] = match.get(field)
        enriched.append(merged)
    return enriched


def load_best_results(run_dir: str | Path, summary: dict[str, Any] | None = None) -> list[dict[str, Any]]:
    """Load benchmark winners from the summary first, then fall back to best_results.json."""

    summary_best_results = list((summary or {}).get("best_results") or [])
    if summary_best_results:
        return summary_best_results

    best_results_path = Path(run_dir) / "best_results.json"
    if not best_results_path.exists():
        return []

    try:
        loaded = json.loads(best_results_path.read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError):
        return []

    return loaded if isinstance(loaded, list) else []


def build_primitive_bank_summary(
    *,
    summary: dict[str, Any],
    best_results: list[dict[str, Any]],
    trial_records: list[dict[str, Any]],
) -> dict[str, Any]:
    best_results = enrich_best_results(best_results, trial_records)
    by_family: dict[str, list[dict[str, Any]]] = {}
    for record in trial_records:
        family = str(record.get("primitive_family") or "unknown")
        by_family.setdefault(family, []).append(record)

    best_by_family: dict[str, dict[str, Any]] = {}
    for family, records in by_family.items():
        ok_records = [record for record in records if record.get("status") == "ok"]
        source = ok_records or records
        best_by_family[family] = max(
            source,
            key=lambda record: float(record.get("search_score")) if record.get("search_score") is not None else float(record.get("quality")) if record.get("quality") is not None else float("-inf"),
        )

    wins: dict[str, list[str]] = {}
    for record in best_results:
        if record.get("status") != "ok":
            continue
        family = str(record.get("primitive_family") or "unknown")
        benchmark_name = str(record.get("benchmark_name") or "")
        if benchmark_name:
            wins.setdefault(family, []).append(benchmark_name)

    usage = summary.get("primitive_usage", {})
    families = sorted(set(usage) | set(best_by_family) | set(wins))
    primitive_families = []
    for family in families:
        representative = best_by_family.get(family, {})
        won = sorted(wins.get(family, []))
        benchmark_groups = sorted({str(record.get("benchmark_group") or "unknown") for record in by_family.get(family, []) if record.get("benchmark_group")})
        primitive_families.append(
            {
                "family": family,
                "evaluation_count": int(usage.get(family, 0)),
                "benchmark_wins": len(won),
                "benchmarks_won": won,
                "supporting_benchmarks": won,
                "benchmark_groups": benchmark_groups,
                "best_metric_name": representative.get("metric_name"),
                "best_metric_value": representative.get("metric_value"),
                "best_search_score": representative.get("search_score"),
                "best_generation": representative.get("generation"),
                "representative_genome_id": representative.get("genome_id"),
                "representative_architecture_summary": representative.get("architecture_summary"),
            }
        )

    return {
        "system": "primordia",
        "run_id": summary.get("run_id"),
        "run_name": summary.get("run_name"),
        "runtime": summary.get("runtime"),
        "runtime_version": summary.get("runtime_version"),
        "precision_mode": summary.get("precision_mode") or "fp32",
        "primitive_families": primitive_families,
    }



def summarize_failure_patterns(non_ok_results: list[dict[str, Any]]) -> list[tuple[str, int]]:
    """Aggregate non-OK trial outcomes for shared inspect/report surfaces."""

    counts = Counter(
        str(record.get("failure_reason") or record.get("status") or "unknown")
        for record in non_ok_results
        if record.get("status") != "ok"
    )
    return counts.most_common()



def write_report(run_dir: str | Path) -> Path:
    """Write or refresh the Primordia report from run artifacts when possible."""

    run_dir = Path(run_dir)
    report_path = run_dir / "report.md"

    summary_path = run_dir / "summary.json"
    if summary_path.exists():
        summary = json.loads(summary_path.read_text(encoding="utf-8"))
        runtime_meta = load_runtime_metadata(summary)
        trial_records_path = run_dir / "trial_records.json"
        primitive_bank_path = run_dir / "primitive_bank_summary.json"
        seed_candidates_path = run_dir / "seed_candidates.json"
        trial_records = json.loads(trial_records_path.read_text(encoding="utf-8")) if trial_records_path.exists() else []
        best_results = enrich_best_results(load_best_results(run_dir, summary), trial_records)
        primitive_bank = (
            json.loads(primitive_bank_path.read_text(encoding="utf-8"))
            if primitive_bank_path.exists()
            else build_primitive_bank_summary(summary=summary, best_results=best_results, trial_records=trial_records)
        )
        seed_candidates = (
            json.loads(seed_candidates_path.read_text(encoding="utf-8"))
            if seed_candidates_path.exists()
            else {"seed_candidates": [], "benchmark_seeds": []}
        )

        lines = [
            "# Primordia Run Report",
            "",
            f"- Run ID: `{summary.get('run_id', run_dir.name)}`",
            f"- Runtime: `{runtime_meta['runtime']}`",
            f"- Runtime Requested: `{runtime_meta['runtime_backend_requested']}`",
            f"- Runtime Version: `{runtime_meta['runtime_version']}`",
            f"- Precision Mode: `{runtime_meta['precision_mode']}`",
            f"- Evaluations: `{summary.get('evaluation_count', 0)}`",
            f"- Target Evaluations: `{summary.get('target_evaluation_count', 'n/a')}`",
            f"- Benchmarks: `{summary.get('benchmark_count', 0)}`",
            f"- Completed Benchmarks: `{len(summary.get('completed_benchmarks') or [])}`",
            f"- Budget Policy: `{summary.get('budget_policy_name', 'unknown')}`",
            f"- Selection Mode: `{summary.get('selection_mode', 'metric_only')}`",
            f"- Primitive Search Policy: `{summary.get('primitive_search_policy', 'unknown')}`",
            f"- Seed Selection Policy: `{summary.get('seed_selection_policy', 'unknown')}`",
            f"- Wall Clock Seconds: `{float(summary.get('wall_clock_seconds', 0.0)):.3f}`",
            "",
            "## Search Policy",
            "",
            "| Setting | Value |",
            "|---|---|",
        ]
        search_policy = summary.get("search_policy") or {}
        runtime_policy = summary.get("runtime_execution_policy")
        if isinstance(runtime_policy, dict):
            lines.append("| runtime_policy_name | {} |".format(_render_markdown_cell(runtime_policy.get("runtime_policy_name"))))
        if runtime_meta.get("runtime_backend_limitations"):
            lines.append("| runtime_backend_limitations | {} |".format(_render_markdown_cell(runtime_meta["runtime_backend_limitations"])))
        for key in [
            "population_size",
            "elite_fraction",
            "mutation_rounds_per_parent",
            "family_exploration_floor",
            "novelty_weight",
            "complexity_penalty_weight",
            "max_candidates_per_benchmark",
        ]:
            lines.append(f"| {_render_markdown_cell(key)} | {_render_markdown_cell(search_policy.get(key), missing='default')} |")
        lines.extend([
            "",
            "## Benchmark Slot Plan",
            "",
            "| Benchmark | Group | Raw Slots | Effective Slots | Families |",
            "|---|---|---:|---:|---:|",
        ])
        slot_rows = summary.get("benchmark_slot_plan") or []
        if slot_rows:
            for row in slot_rows:
                lines.append(
                    "| {benchmark} | {group} | {raw_slots} | {effective_slots} | {family_count} |".format(
                        benchmark=_render_markdown_cell(row.get("benchmark_name"), missing="unknown"),
                        group=_render_markdown_cell(row.get("benchmark_group"), missing="unknown"),
                        raw_slots=int(row.get("raw_slots", 0)),
                        effective_slots=int(row.get("effective_slots", 0)),
                        family_count=int(row.get("family_count", 0)),
                    )
                )
        else:
            lines.append("| none | none | 0 | 0 | 0 |")
        lines.extend([
            "",
            "## Primitive Usage",
            "",
            "| Primitive Family | Evaluations |",
            "|---|---:|",
        ])
        primitive_usage = summary.get("primitive_usage", {})
        if primitive_usage:
            for family, count in primitive_usage.items():
                lines.append(f"| {_render_markdown_cell(family, missing='none')} | {count} |")
        else:
            lines.append("| none | 0 |")
        lines.extend([
            "",
            "## Primitive Bank Summary",
            "",
            "| Family | Evaluations | Benchmark Wins | Won Benchmarks | Best Metric | Best Value | Representative Genome | Representative Architecture |",
            "|---|---:|---:|---|---|---:|---|---|",
        ])
        primitive_rows = primitive_bank.get("primitive_families") or []
        if primitive_rows:
            for row in primitive_rows:
                won = ", ".join(row.get("benchmarks_won") or []) or "—"
                best_value = row.get("best_metric_value")
                rendered_value = "---" if best_value is None else f"{float(best_value):.6f}"
                lines.append(
                    "| {family} | {evaluation_count} | {benchmark_wins} | {won} | {best_metric} | {best_value} | {genome} | {architecture} |".format(
                        family=_render_markdown_cell(row.get("family"), missing="unknown"),
                        evaluation_count=int(row.get("evaluation_count", 0)),
                        benchmark_wins=int(row.get("benchmark_wins", 0)),
                        won=_render_markdown_cell(won),
                        best_metric=_render_markdown_cell(row.get("best_metric_name")),
                        best_value=rendered_value,
                        genome=_render_markdown_cell(row.get("representative_genome_id")),
                        architecture=_render_markdown_cell(row.get("representative_architecture_summary")),
                    )
                )
        else:
            lines.append("| none | 0 | 0 | — | — | --- | — | — |")
        lines.extend([
            "",
            "## Benchmark Leaders",
            "",
            "| Benchmark | Group | Leader Family | Leader Genome | Generation | Search Score | Metric | Value | Status |",
            "|---|---|---|---|---:|---:|---|---:|---|",
        ])
        benchmark_leaders = summary.get("benchmark_leaders") or []
        if benchmark_leaders:
            for row in benchmark_leaders:
                metric_value = row.get("metric_value")
                rendered_value = "---" if metric_value is None else f"{float(metric_value):.6f}"
                search_score = row.get("leader_search_score")
                rendered_score = "---" if search_score is None else f"{float(search_score):.6f}"
                generation = row.get("leader_generation")
                rendered_generation = 0 if generation is None else int(generation)
                lines.append(
                    "| {benchmark} | {group} | {family} | {genome} | {generation} | {score} | {metric} | {value} | {status} |".format(
                        benchmark=_render_markdown_cell(row.get("benchmark_name"), missing="unknown"),
                        group=_render_markdown_cell(row.get("benchmark_group"), missing="unknown"),
                        family=_render_markdown_cell(row.get("leader_family"), missing="unknown"),
                        genome=_render_markdown_cell(row.get("leader_genome_id")),
                        generation=rendered_generation,
                        score=rendered_score,
                        metric=_render_markdown_cell(row.get("metric_name")),
                        value=rendered_value,
                        status=_render_markdown_cell(row.get("status"), missing="unknown"),
                    )
                )
        else:
            lines.append("| none | none | none | — | 0 | --- | — | --- | none |")
        lines.extend([
            "",
            "## Family Leaders",
            "",
            "| Family | Evaluations | Benchmark Wins | Best Gen | Best Search Score | Groups | Supporting Benchmarks | Representative Genome | Representative Architecture |",
            "|---|---:|---:|---:|---:|---|---|---|---|",
        ])
        family_leaders = summary.get("family_leaders") or []
        if family_leaders:
            for row in family_leaders:
                groups = ", ".join(row.get("benchmark_groups") or []) or "—"
                benchmarks = ", ".join(row.get("supporting_benchmarks") or []) or "—"
                best_search_score = row.get("best_search_score")
                rendered_score = "---" if best_search_score is None else f"{float(best_search_score):.6f}"
                best_generation = row.get("best_generation")
                rendered_generation = 0 if best_generation is None else int(best_generation)
                lines.append(
                    "| {family} | {evaluation_count} | {benchmark_wins} | {best_generation} | {best_search_score} | {groups} | {benchmarks} | {genome} | {architecture} |".format(
                        family=_render_markdown_cell(row.get("family"), missing="unknown"),
                        evaluation_count=int(row.get("evaluation_count", 0)),
                        benchmark_wins=int(row.get("benchmark_wins", 0)),
                        best_generation=rendered_generation,
                        best_search_score=rendered_score,
                        groups=_render_markdown_cell(groups),
                        benchmarks=_render_markdown_cell(benchmarks),
                        genome=_render_markdown_cell(row.get("representative_genome_id")),
                        architecture=_render_markdown_cell(row.get("representative_architecture_summary")),
                    )
                )
        else:
            lines.append("| none | 0 | 0 | 0 | --- | — | — | — | — |")
        lines.extend([
            "",
            "## Transfer Seed Candidates",
            "",
            "| Rank | Family | Groups | Benchmark Wins | Representative Genome | Representative Architecture |",
            "|---:|---|---|---:|---|---|",
        ])
        seed_rows = seed_candidates.get("seed_candidates") or []
        if seed_rows:
            for row in seed_rows[:8]:
                lines.append(
                    "| {rank} | {family} | {groups} | {wins} | {genome} | {architecture} |".format(
                        rank=int(row.get("seed_rank", 0)),
                        family=_render_markdown_cell(row.get("family"), missing="unknown"),
                        groups=_render_markdown_cell(", ".join(row.get("benchmark_groups") or [])),
                        wins=int(row.get("benchmark_wins", 0)),
                        genome=_render_markdown_cell(row.get("representative_genome_id")),
                        architecture=_render_markdown_cell(row.get("representative_architecture_summary")),
                    )
                )
        else:
            lines.append("| 0 | none | — | 0 | — | — |")
        lines.extend([
            "",
            "## Benchmark Group Coverage",
            "",
            "| Group | Benchmarks |",
            "|---|---:|",
        ])
        group_counts = summary.get("group_counts", {})
        if group_counts:
            for group, count in group_counts.items():
                lines.append(f"| {_render_markdown_cell(group, missing='none')} | {count} |")
        else:
            lines.append("| none | 0 |")
        failure_patterns = summarize_failure_patterns(trial_records)
        lines.extend([
            "",
            "## Failure Summary",
            "",
            f"- Failure Count: `{int(summary.get('failure_count', 0))}`",
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
            "## Best Primitive Per Benchmark",
            "",
            "| Benchmark | Primitive | Metric | Value | Status |",
            "|---|---|---|---:|---|",
        ])
        for record in best_results:
            value = "---" if record.get("metric_value") is None else f"{float(record['metric_value']):.6f}"
            lines.append(
                "| {benchmark} | {primitive} | {metric} | {value} | {status} |".format(
                    benchmark=_render_markdown_cell(record.get("benchmark_name"), missing="unknown"),
                    primitive=_render_markdown_cell(record.get("primitive_name"), missing="unknown"),
                    metric=_render_markdown_cell(record.get("metric_name")),
                    value=value,
                    status=_render_markdown_cell(record.get("status"), missing="unknown"),
                )
            )
        report_path.write_text("\n".join(lines) + "\n", encoding="utf-8")
        return report_path

    best_path = run_dir / "best_results.json"
    if not best_path.exists() and report_path.exists():
        return report_path

    records = json.loads(best_path.read_text(encoding="utf-8")) if best_path.exists() else []
    lines = [
        "# Primordia Export Report",
        "",
        "| Benchmark | Primitive | Metric | Value | Status |",
        "|---|---|---|---:|---|",
    ]
    for record in records:
        value = "---" if record.get("metric_value") is None else f"{float(record['metric_value']):.6f}"
        lines.append(
            "| {benchmark} | {primitive} | {metric} | {value} | {status} |".format(
                benchmark=_render_markdown_cell(record.get("benchmark_name"), missing="unknown"),
                primitive=_render_markdown_cell(record.get("primitive_name"), missing="unknown"),
                metric=_render_markdown_cell(record.get("metric_name")),
                value=value,
                status=_render_markdown_cell(record.get("status"), missing="unknown"),
            )
        )
    report_path.write_text("\n".join(lines) + "\n", encoding="utf-8")
    return report_path
