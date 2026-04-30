"""Performance baseline planning helpers for Compare-owned artifacts."""

from __future__ import annotations

from collections import defaultdict
from datetime import datetime, timezone
import html
import json
from pathlib import Path
import shutil
import subprocess
from typing import Any

import yaml

from evonn_compare.contracts.performance import (
    PerformanceBackendTarget,
    PerformanceBaselineArtifacts,
    PerformanceBaselineManifest,
    PerformanceMetrics,
    PerformancePackCoverage,
    PerformanceQualityGuard,
    PerformanceReviewReferences,
    PerformanceRow,
    PerformanceSystemBackendSummary,
    PerformanceSystemCoverage,
    PerformanceTrustGuard,
)
from evonn_compare.ingest.loader import SystemIngestor
from evonn_compare.contracts.parity import load_parity_pack, resolve_pack_path
from evonn_shared.manifests import summary_core_from_results


SYSTEM_BACKEND_TARGETS: dict[str, tuple[PerformanceBackendTarget, ...]] = {
    "prism": (
        PerformanceBackendTarget(backend_class="mlx_truth", backend_label="mlx", host_label="macos"),
    ),
    "topograph": (
        PerformanceBackendTarget(backend_class="mlx_truth", backend_label="mlx", host_label="macos"),
    ),
    "stratograph": (
        PerformanceBackendTarget(backend_class="mlx_truth", backend_label="mlx", host_label="macos"),
        PerformanceBackendTarget(backend_class="linux_fallback", backend_label="fallback", host_label="linux"),
    ),
    "primordia": (
        PerformanceBackendTarget(backend_class="mlx_truth", backend_label="mlx", host_label="macos"),
        PerformanceBackendTarget(backend_class="linux_fallback", backend_label="fallback", host_label="linux"),
    ),
    "contenders": (
        PerformanceBackendTarget(backend_class="linux_fallback", backend_label="sklearn", host_label="linux"),
    ),
}

REVIEW_WORKFLOW_DOC = "PERFORMANCE_OPTIMIZATION_WORKFLOW.md"
REVIEW_PR_TEMPLATE = ".github/pull_request_template.md"
REVIEW_CHILD_ISSUE_TEMPLATE = (
    "PERFORMANCE_OPTIMIZATION_WORKFLOW.md#optimization-child-issue-template"
)
REVIEW_BRANCH_OUTCOME_RECORDING = (
    "PERFORMANCE_OPTIMIZATION_WORKFLOW.md#branch-outcome-recording"
)


def initialize_performance_baseline(
    *,
    output_root: Path,
    packs: list[str],
    budgets: list[int],
    seeds: list[int],
    cache_modes: list[str],
    systems: list[str],
    matrix_workspaces: list[Path] | None = None,
) -> dict[str, str | int]:
    """Create the canonical planned performance-baseline artifact set."""

    generated_at = datetime.now(timezone.utc).isoformat()
    git_sha = _resolve_git_sha()
    baseline_root = output_root.resolve()
    raw_runs_root = baseline_root / "raw_runs"
    raw_runs_root.mkdir(parents=True, exist_ok=True)

    normalized_systems = [_normalize_system_name(value) for value in systems]
    rows: list[PerformanceRow] = []
    pack_summaries: list[PerformancePackCoverage] = []
    for pack_name in packs:
        pack_path = resolve_pack_path(pack_name)
        pack_spec = load_parity_pack(pack_path)
        pack_summaries.append(
            PerformancePackCoverage(
                pack_name=pack_spec.name,
                pack_path=str(pack_path),
                tier=pack_spec.tier,
                benchmark_count=len(pack_spec.benchmarks),
                default_budget=pack_spec.budget_policy.evaluation_count,
            )
        )
        for system_name in normalized_systems:
            system_root = raw_runs_root / system_name
            system_root.mkdir(parents=True, exist_ok=True)
            for backend in SYSTEM_BACKEND_TARGETS[system_name]:
                for budget in budgets:
                    for seed in seeds:
                        for cache_mode in cache_modes:
                            case_id = (
                                f"{system_name}__{backend.backend_label}__{pack_spec.name}"
                                f"__eval{budget}__seed{seed}__{cache_mode}"
                            )
                            rows.append(
                                PerformanceRow(
                                    record_type="planned_performance_baseline_case",
                                    status="planned",
                                    generated_at=generated_at,
                                    git_sha=git_sha,
                                    case_id=case_id,
                                    system=system_name,
                                    backend_class=backend.backend_class,
                                    backend_label=backend.backend_label,
                                    host_label=backend.host_label,
                                    pack_name=pack_spec.name,
                                    pack_path=str(pack_path),
                                    pack_tier=pack_spec.tier,
                                    benchmark_count=len(pack_spec.benchmarks),
                                    budget=budget,
                                    seed=seed,
                                    cache_mode=cache_mode,
                                    accounting_tags=("full_budget",),
                                    actual_evaluations=None,
                                    cached_evaluations=None,
                                    resumed_evaluations=None,
                                    screened_evaluations=None,
                                    deduplicated_evaluations=None,
                                    reduced_fidelity_evaluations=None,
                                    worker_count=None,
                                    precision=None,
                                    device=None,
                                    raw_run_dir=str(system_root / case_id),
                                    metrics=PerformanceMetrics(),
                                    quality_guard=PerformanceQualityGuard(
                                        status="pending",
                                        median_rank=None,
                                        median_rank_delta_vs_baseline=None,
                                        quality_delta_vs_baseline=None,
                                    ),
                                    trust_guard=PerformanceTrustGuard(
                                        required_state="same-or-better",
                                        observed_state=None,
                                        status="pending",
                                    ),
                                    notes=[
                                        "planned baseline case only; fill metrics after execution",
                                    ],
                                )
                            )

    materialization_notes: list[str] = []
    if matrix_workspaces:
        rows, materialization_notes = _materialize_rows(
            rows=rows,
            raw_runs_root=raw_runs_root,
            matrix_workspaces=matrix_workspaces,
        )

    perf_rows_path = baseline_root / "perf_rows.jsonl"
    perf_rows_path.write_text(
        "".join(json.dumps(row.model_dump(mode="json"), sort_keys=True) + "\n" for row in rows),
        encoding="utf-8",
    )

    system_counts = _build_system_counts(rows)
    manifest = PerformanceBaselineManifest(
        generated_at=generated_at,
        git_sha=git_sha,
        baseline_root=str(baseline_root),
        packs=pack_summaries,
        budgets=budgets,
        seeds=seeds,
        cache_modes=cache_modes,
        systems=normalized_systems,
        supported_backends={
            system_name: list(SYSTEM_BACKEND_TARGETS[system_name])
            for system_name in normalized_systems
        },
        planned_case_count=len(rows),
        status_counts=_build_status_counts(rows),
        system_counts=system_counts,
        artifacts=PerformanceBaselineArtifacts(
            raw_runs=str(raw_runs_root),
            perf_rows=str(perf_rows_path),
            baseline_summary=str(baseline_root / "baseline_summary.md"),
            perf_dashboard=str(baseline_root / "perf_dashboard.html"),
            perf_dashboard_json=str(baseline_root / "perf_dashboard.json"),
        ),
        review_references=PerformanceReviewReferences(
            workflow_doc=REVIEW_WORKFLOW_DOC,
            pull_request_template=REVIEW_PR_TEMPLATE,
            child_issue_template=REVIEW_CHILD_ISSUE_TEMPLATE,
            branch_outcome_recording=REVIEW_BRANCH_OUTCOME_RECORDING,
        ),
    )

    manifest_path = baseline_root / "baseline_manifest.json"
    manifest_path.write_text(json.dumps(manifest.model_dump(mode="json"), indent=2), encoding="utf-8")

    summary_path = baseline_root / "baseline_summary.md"
    summary_path.write_text(
        _render_baseline_summary(
            generated_at=generated_at,
            git_sha=git_sha,
            baseline_root=baseline_root,
            packs=pack_summaries,
            budgets=budgets,
            seeds=seeds,
            cache_modes=cache_modes,
            system_counts=system_counts,
            planned_case_count=len(rows),
            status_counts=manifest.status_counts,
            review_references=manifest.review_references,
            materialization_notes=materialization_notes,
        ),
        encoding="utf-8",
    )

    dashboard_payload = {
        "generated_at": generated_at,
        "git_sha": git_sha,
        "baseline_root": str(baseline_root),
        "planned_case_count": len(rows),
        "packs": [entry.pack_name for entry in pack_summaries],
        "budgets": budgets,
        "seeds": seeds,
        "cache_modes": cache_modes,
        "systems": [entry.model_dump(mode="json") for entry in system_counts],
        "status_counts": manifest.status_counts,
        "sample_rows": [row.model_dump(mode="json") for row in rows[:20]],
        "review_references": manifest.review_references.model_dump(mode="json"),
        "materialization_notes": materialization_notes,
    }
    dashboard_json_path = baseline_root / "perf_dashboard.json"
    dashboard_json_path.write_text(json.dumps(dashboard_payload, indent=2), encoding="utf-8")
    dashboard_path = baseline_root / "perf_dashboard.html"
    dashboard_path.write_text(_render_dashboard_html(dashboard_payload), encoding="utf-8")

    return {
        "mode": "materialized" if matrix_workspaces else "plan",
        "baseline_root": str(baseline_root),
        "baseline_manifest": str(manifest_path),
        "baseline_summary": str(summary_path),
        "raw_runs": str(raw_runs_root),
        "perf_rows": str(perf_rows_path),
        "perf_dashboard": str(dashboard_path),
        "perf_dashboard_json": str(dashboard_json_path),
        "planned_case_count": len(rows),
    }


def _normalize_system_name(value: str) -> str:
    normalized = value.strip().lower()
    if normalized not in SYSTEM_BACKEND_TARGETS:
        available = ", ".join(sorted(SYSTEM_BACKEND_TARGETS))
        raise ValueError(f"unknown system '{value}'; available: {available}")
    return normalized


def _build_system_counts(rows: list[PerformanceRow]) -> list[PerformanceSystemCoverage]:
    counts: dict[tuple[str, str], int] = {}
    for row in rows:
        counts[(row.system, row.backend_class)] = counts.get((row.system, row.backend_class), 0) + 1
    summaries: list[PerformanceSystemCoverage] = []
    for system_name in sorted({row.system for row in rows}):
        backend_rows = [
            PerformanceSystemBackendSummary(
                backend_class=backend.backend_class,
                backend_label=backend.backend_label,
                host_label=backend.host_label,
                planned_case_count=counts.get((system_name, backend.backend_class), 0),
            )
            for backend in SYSTEM_BACKEND_TARGETS[system_name]
        ]
        summaries.append(
            PerformanceSystemCoverage(
                system=system_name,
                planned_case_count=sum(item.planned_case_count for item in backend_rows),
                backends=backend_rows,
            )
        )
    return summaries


def _build_status_counts(rows: list[PerformanceRow]) -> dict[str, int]:
    counts: dict[str, int] = {}
    for row in rows:
        counts[row.status] = counts.get(row.status, 0) + 1
    return counts


def _materialize_rows(
    *,
    rows: list[PerformanceRow],
    raw_runs_root: Path,
    matrix_workspaces: list[Path],
) -> tuple[list[PerformanceRow], list[str]]:
    planned_by_key = {
        _row_lookup_key(row): row
        for row in rows
    }
    measured_rows: list[PerformanceRow] = []
    notes: list[str] = []
    unmatched_sources: list[str] = []
    filled_keys: set[tuple[str, str, str, int, int, str]] = set()

    for workspace in matrix_workspaces:
        for source in _discover_measured_sources(workspace):
            lookup_key = (
                source["system"],
                source["backend_class"],
                source["pack_name"],
                source["budget"],
                source["seed"],
                "cold",
            )
            planned_row = planned_by_key.get(lookup_key)
            if planned_row is None:
                unmatched_sources.append(
                    f"{source['system']} {source['backend_class']} {source['pack_name']} eval{source['budget']} seed{source['seed']}"
                )
                continue
            measured_rows.append(
                _build_measured_row(
                    planned_row=planned_row,
                    source=source,
                    raw_runs_root=raw_runs_root,
                )
            )
            filled_keys.add(lookup_key)

    for row in rows:
        lookup_key = _row_lookup_key(row)
        if lookup_key in filled_keys:
            continue
        measured_rows.append(row)

    if matrix_workspaces:
        notes.append(
            "Matrix workspaces imported: "
            + ", ".join(str(path.resolve()) for path in matrix_workspaces)
        )
    if unmatched_sources:
        notes.append(
            "Measured sources with no matching planned baseline row: "
            + "; ".join(sorted(unmatched_sources))
        )
    return measured_rows, notes


def _discover_measured_sources(matrix_workspace: Path) -> list[dict[str, Any]]:
    workspace_path = matrix_workspace.resolve()
    matrix_path = workspace_path / "matrix.yaml"
    if not matrix_path.exists():
        raise FileNotFoundError(f"matrix workspace missing matrix.yaml: {workspace_path}")
    payload = yaml.safe_load(matrix_path.read_text(encoding="utf-8")) or {}
    cases = payload.get("cases") or []
    sources: list[dict[str, Any]] = []
    for case in cases:
        summary_path = _resolve_case_summary_path(case)
        if summary_path is None or not summary_path.exists():
            continue
        summary_payload = json.loads(summary_path.read_text(encoding="utf-8"))
        rank_map = _build_system_rank_map(summary_payload)
        trust_states = _build_system_trust_states(summary_payload)
        for system_name, run_dir in _extract_case_run_dirs(case).items():
            run_dir_path = Path(run_dir)
            if not run_dir_path.exists():
                continue
            ingestor = SystemIngestor(run_dir_path)
            manifest = ingestor.load_manifest()
            results = ingestor.load_results()
            backend_signature = _resolve_backend_signature(manifest.system, manifest.device.framework)
            if backend_signature is None:
                continue
            backend_class, backend_label = backend_signature
            sources.append(
                {
                    "workspace": workspace_path,
                    "summary_path": summary_path.resolve(),
                    "system": str(manifest.system),
                    "backend_class": backend_class,
                    "backend_label": backend_label,
                    "host_label": _resolve_host_label(manifest.device.device_name, manifest.device.framework),
                    "pack_name": _normalize_pack_name(manifest.pack_name, manifest.budget.evaluation_count),
                    "budget": manifest.budget.evaluation_count,
                    "seed": manifest.seed,
                    "run_dir": run_dir_path.resolve(),
                    "manifest": manifest,
                    "results": results,
                    "median_rank": rank_map.get(system_name),
                    "trust_state": trust_states.get(system_name),
                }
            )
    return sources


def _resolve_case_summary_path(case: dict[str, Any]) -> Path | None:
    report_dir = case.get("report_dir")
    if report_dir:
        return Path(report_dir) / "fair_matrix_summary.json"
    summary_output_path = case.get("summary_output_path")
    if summary_output_path:
        return Path(summary_output_path).with_suffix(".json")
    return None


def _extract_case_run_dirs(case: dict[str, Any]) -> dict[str, str]:
    mapping: dict[str, str] = {}
    for system_name in SYSTEM_BACKEND_TARGETS:
        run_dir = case.get(f"{system_name}_run_dir")
        if run_dir is None and system_name == "contenders":
            run_dir = case.get("contender_run_dir")
        if run_dir:
            mapping[system_name] = str(run_dir)
    return mapping


def _build_system_rank_map(summary_payload: dict[str, Any]) -> dict[str, float]:
    benchmark_rows: dict[str, list[dict[str, Any]]] = defaultdict(list)
    for row in summary_payload.get("trend_rows") or []:
        benchmark_id = row.get("benchmark_id")
        if benchmark_id:
            benchmark_rows[str(benchmark_id)].append(row)

    per_system_ranks: dict[str, list[float]] = defaultdict(list)
    for rows in benchmark_rows.values():
        ranked = _rank_benchmark_rows(rows)
        for system_name, rank in ranked.items():
            per_system_ranks[system_name].append(rank)

    return {
        system_name: _median(values)
        for system_name, values in per_system_ranks.items()
        if values
    }


def _rank_benchmark_rows(rows: list[dict[str, Any]]) -> dict[str, float]:
    if not rows:
        return {}
    metric_direction = str(rows[0].get("metric_direction") or "max")

    def sort_key(row: dict[str, Any]) -> tuple[int, float]:
        metric_value = row.get("metric_value")
        if metric_value is None:
            return (1, 0.0)
        value = float(metric_value)
        adjusted = -value if metric_direction == "max" else value
        return (0, adjusted)

    ordered = sorted(rows, key=sort_key)
    ranks: dict[str, float] = {}
    next_rank = 1
    index = 0
    while index < len(ordered):
        current = ordered[index]
        metric_value = current.get("metric_value")
        peer_index = index + 1
        while peer_index < len(ordered) and ordered[peer_index].get("metric_value") == metric_value:
            peer_index += 1
        tied_rank = float(next_rank + ((peer_index - index - 1) / 2))
        for peer in ordered[index:peer_index]:
            system_name = peer.get("system")
            if system_name:
                ranks[str(system_name)] = tied_rank
        next_rank = peer_index + 1
        index = peer_index
    return ranks


def _build_system_trust_states(summary_payload: dict[str, Any]) -> dict[str, str]:
    states: dict[str, str] = {}
    for row in summary_payload.get("trend_rows") or []:
        system_name = row.get("system")
        if not system_name or system_name in states:
            continue
        fairness = row.get("fairness_metadata") or {}
        observed_state = fairness.get("system_operating_state") or row.get("system_operating_state")
        if observed_state:
            states[str(system_name)] = str(observed_state)
    return states


def _build_measured_row(
    *,
    planned_row: PerformanceRow,
    source: dict[str, Any],
    raw_runs_root: Path,
) -> PerformanceRow:
    manifest = source["manifest"]
    results = source["results"]
    result_payloads = [record.model_dump(mode="json") for record in results]
    summary = summary_core_from_results(
        results=result_payloads,
        parameter_counts=[record.parameter_count for record in results if record.parameter_count is not None],
    )
    accounted_evaluations = manifest.budget.accounted_evaluations()
    wall_clock_seconds = manifest.budget.wall_clock_seconds
    evals_per_second = None
    if accounted_evaluations not in (None, 0) and wall_clock_seconds not in (None, 0):
        evals_per_second = float(accounted_evaluations) / float(wall_clock_seconds)
    cache_hit_rate = None
    if accounted_evaluations not in (None, 0) and manifest.budget.cached_evaluations is not None:
        cache_hit_rate = float(manifest.budget.cached_evaluations) / float(accounted_evaluations)
    reuse_rate = None
    if accounted_evaluations not in (None, 0) and manifest.budget.resumed_evaluations is not None:
        reuse_rate = float(manifest.budget.resumed_evaluations) / float(accounted_evaluations)
    train_seconds = _sum_optional(record.train_seconds for record in results)
    peak_memory_mb = _max_optional(record.peak_memory_mb for record in results)
    failure_count = int(summary["failure_count"])

    canonical_run_dir = Path(planned_row.raw_run_dir)
    _link_raw_run_dir(canonical_run_dir, source["run_dir"])

    notes = [
        f"materialized from fair-matrix workspace {source['workspace']}",
        f"source summary {source['summary_path']}",
    ]
    if planned_row.host_label != source["host_label"]:
        notes.append(
            f"host_label observed as {source['host_label']} instead of planned {planned_row.host_label}"
        )
    if failure_count > 0:
        notes.append(f"failure_count={failure_count}")

    row_status = "failed" if failure_count > 0 or manifest.budget.partial_run else "measured"
    trust_state = source["trust_state"] or "unknown"
    return PerformanceRow(
        record_type="measured_performance_baseline_case",
        status=row_status,
        generated_at=planned_row.generated_at,
        git_sha=planned_row.git_sha,
        case_id=planned_row.case_id,
        system=planned_row.system,
        backend_class=source["backend_class"],
        backend_label=source["backend_label"],
        host_label=source["host_label"],
        pack_name=planned_row.pack_name,
        pack_path=planned_row.pack_path,
        pack_tier=planned_row.pack_tier,
        benchmark_count=planned_row.benchmark_count,
        budget=planned_row.budget,
        seed=planned_row.seed,
        cache_mode=planned_row.cache_mode,
        accounting_tags=manifest.budget.resolved_accounting_tags(),
        actual_evaluations=manifest.budget.actual_evaluations,
        cached_evaluations=manifest.budget.cached_evaluations,
        resumed_evaluations=manifest.budget.resumed_evaluations,
        screened_evaluations=manifest.budget.screened_evaluations,
        deduplicated_evaluations=manifest.budget.deduplicated_evaluations,
        reduced_fidelity_evaluations=manifest.budget.reduced_fidelity_evaluations,
        worker_count=None,
        precision=manifest.device.precision_mode,
        device=manifest.device.device_name,
        raw_run_dir=str(canonical_run_dir),
        metrics=PerformanceMetrics(
            wall_clock_seconds=wall_clock_seconds,
            evals_per_second=evals_per_second,
            train_seconds=train_seconds,
            data_load_seconds=None,
            cache_hit_rate=cache_hit_rate,
            reuse_rate=reuse_rate,
            failure_count=failure_count,
            peak_memory_mb=peak_memory_mb,
        ),
        quality_guard=PerformanceQualityGuard(
            status="recorded",
            median_rank=source["median_rank"],
            median_rank_delta_vs_baseline=None,
            quality_delta_vs_baseline=None,
        ),
        trust_guard=PerformanceTrustGuard(
            required_state="same-or-better",
            observed_state=trust_state,
            status="recorded",
        ),
        notes=notes,
    )


def _row_lookup_key(row: PerformanceRow) -> tuple[str, str, str, int, int, str]:
    return (row.system, row.backend_class, row.pack_name, row.budget, row.seed, row.cache_mode)


def _resolve_backend_signature(system_name: str, framework: str | None) -> tuple[str, str] | None:
    normalized_framework = (framework or "").strip().lower()
    if normalized_framework == "mlx":
        return ("mlx_truth", "mlx")
    if system_name == "contenders":
        return ("linux_fallback", "sklearn")
    if normalized_framework in {"numpy-fallback", "portable-sklearn"}:
        return ("linux_fallback", "fallback")
    return None


def _resolve_host_label(device_name: str, framework: str | None) -> str:
    lowered_device = device_name.strip().lower()
    lowered_framework = (framework or "").strip().lower()
    if "mac" in lowered_device or lowered_device in {"arm64", "apple-silicon"}:
        return "macos"
    if lowered_framework in {"numpy-fallback", "portable-sklearn"}:
        return "linux"
    if "linux" in lowered_device or "x86" in lowered_device or "amd64" in lowered_device:
        return "linux"
    return "macos" if lowered_framework == "mlx" else "linux"


def _normalize_pack_name(pack_name: str, budget: int) -> str:
    suffix = f"_eval{budget}"
    if pack_name.endswith(suffix):
        return pack_name[: -len(suffix)]
    return pack_name


def _median(values: list[float]) -> float:
    ordered = sorted(float(value) for value in values)
    midpoint = len(ordered) // 2
    if len(ordered) % 2:
        return ordered[midpoint]
    return float((ordered[midpoint - 1] + ordered[midpoint]) / 2)


def _sum_optional(values: Any) -> float | None:
    present = [float(value) for value in values if value is not None]
    if not present:
        return None
    return float(sum(present))


def _max_optional(values: Any) -> float | None:
    present = [float(value) for value in values if value is not None]
    if not present:
        return None
    return max(present)


def _link_raw_run_dir(target: Path, source: Path) -> None:
    target.parent.mkdir(parents=True, exist_ok=True)
    if target.exists() or target.is_symlink():
        if target.is_dir() and not target.is_symlink():
            shutil.rmtree(target)
        else:
            target.unlink()
    target.symlink_to(source, target_is_directory=True)


def _render_baseline_summary(
    *,
    generated_at: str,
    git_sha: str,
    baseline_root: Path,
    packs: list[PerformancePackCoverage],
    budgets: list[int],
    seeds: list[int],
    cache_modes: list[str],
    system_counts: list[PerformanceSystemCoverage],
    planned_case_count: int,
    status_counts: dict[str, int],
    review_references: PerformanceReviewReferences,
    materialization_notes: list[str],
) -> str:
    measured_count = status_counts.get("measured", 0) + status_counts.get("failed", 0)
    lines = [
        "# Performance Baseline",
        "",
        f"- Generated at: `{generated_at}`",
        f"- Git SHA: `{git_sha}`",
        f"- Baseline root: `{baseline_root}`",
        f"- Total cases: `{planned_case_count}`",
        f"- Status counts: `{json.dumps(status_counts, sort_keys=True)}`",
        "",
        "## Matrix",
        "",
        f"- Packs: `{', '.join(entry.pack_name for entry in packs)}`",
        f"- Budgets: `{', '.join(str(value) for value in budgets)}`",
        f"- Seeds: `{', '.join(str(value) for value in seeds)}`",
        f"- Cache modes: `{', '.join(cache_modes)}`",
        "",
        "## Pack Coverage",
        "",
        "| Pack | Tier | Benchmarks | Default Budget |",
        "| --- | --- | --- | --- |",
    ]
    for entry in packs:
        lines.append(
            f"| {entry.pack_name} | {entry.tier} | {entry.benchmark_count} | {entry.default_budget} |"
        )
    lines.extend(
        [
            "",
            "## System Coverage",
            "",
            "| System | Backends | Planned Cases |",
            "| --- | --- | --- |",
        ]
    )
    for entry in system_counts:
        backends = ", ".join(
            f"{backend.backend_label} ({backend.backend_class}, {backend.host_label})"
            for backend in entry.backends
        )
        lines.append(f"| {entry.system} | {backends} | {entry.planned_case_count} |")
    lines.extend(
        [
            "",
            "## Review Workflow",
            "",
            f"- Workflow doc: `{review_references.workflow_doc}`",
            f"- PR template: `{review_references.pull_request_template}`",
            f"- Optimization child issue template: `{review_references.child_issue_template}`",
            f"- Branch outcome recording: `{review_references.branch_outcome_recording}`",
            "- Required PR/issue evidence: baseline artifact path, after-change artifact path, exact dashboard/history slices, quality verdict, trust-state verdict, and budget-accounting verdict.",
            "- Outcome policy: accepted branches close as `done`, rejected branches stay open for revision, and scrapped branches close as `cancelled` with artifact links preserved.",
            "",
            "## Artifact Notes",
            "",
            (
                "- `perf_rows.jsonl` includes measured rows imported from fair-matrix workspaces."
                if measured_count
                else "- `perf_rows.jsonl` is the normalized planned-run dataset for later execution fills."
            ),
            (
                "- `raw_runs/` contains canonical symlink targets for imported raw run artifacts."
                if measured_count
                else "- `raw_runs/` is reserved for per-case raw artifacts once the runner is wired in."
            ),
            (
                "- `perf_dashboard.html` now reflects mixed planned/measured baseline coverage."
                if measured_count
                else "- `perf_dashboard.html` is a static planning dashboard, not yet a measured delta view."
            ),
        ]
    )
    if materialization_notes:
        lines.extend(["", "## Materialization Notes", ""])
        lines.extend(f"- {note}" for note in materialization_notes)
    return "\n".join(lines) + "\n"


def _render_dashboard_html(payload: dict[str, object]) -> str:
    systems = payload.get("systems") or []
    review_references = payload.get("review_references") or {}
    rows = []
    for entry in systems:
        backend_cells = ", ".join(
            f"{backend['backend_label']} ({backend['backend_class']}, {backend['host_label']})"
            for backend in entry["backends"]
        )
        rows.append(
            "<tr>"
            f"<td>{html.escape(str(entry['system']))}</td>"
            f"<td>{html.escape(backend_cells)}</td>"
            f"<td>{html.escape(str(entry['planned_case_count']))}</td>"
            "</tr>"
        )
    systems_table = "\n".join(rows)
    packs = ", ".join(str(value) for value in payload.get("packs") or [])
    budgets = ", ".join(str(value) for value in payload.get("budgets") or [])
    seeds = ", ".join(str(value) for value in payload.get("seeds") or [])
    cache_modes = ", ".join(str(value) for value in payload.get("cache_modes") or [])
    workflow_doc = html.escape(str(review_references.get("workflow_doc") or "unknown"))
    pr_template = html.escape(str(review_references.get("pull_request_template") or "unknown"))
    child_issue_template = html.escape(str(review_references.get("child_issue_template") or "unknown"))
    branch_outcome_recording = html.escape(
        str(review_references.get("branch_outcome_recording") or "unknown")
    )
    return f"""<!DOCTYPE html>
<html lang="en">
  <head>
    <meta charset="utf-8">
    <title>EvoNN Performance Baseline Plan</title>
    <style>
      :root {{
        color-scheme: light;
        font-family: "Iosevka", "SFMono-Regular", "Menlo", monospace;
        background: #f4efe6;
        color: #1b1b1b;
      }}
      body {{
        margin: 0;
        padding: 2rem;
        background:
          radial-gradient(circle at top left, rgba(208, 143, 74, 0.22), transparent 35%),
          linear-gradient(180deg, #f7f1e7 0%, #efe3d0 100%);
      }}
      main {{
        max-width: 1100px;
        margin: 0 auto;
        background: rgba(255, 252, 247, 0.9);
        border: 1px solid #c8b79d;
        border-radius: 20px;
        padding: 1.5rem;
        box-shadow: 0 18px 40px rgba(73, 49, 18, 0.12);
      }}
      h1, h2 {{
        margin-top: 0;
      }}
      .meta {{
        display: grid;
        grid-template-columns: repeat(auto-fit, minmax(220px, 1fr));
        gap: 0.75rem;
        margin-bottom: 1.5rem;
      }}
      .card {{
        background: #fffaf1;
        border: 1px solid #dcc9ab;
        border-radius: 14px;
        padding: 0.9rem 1rem;
      }}
      table {{
        width: 100%;
        border-collapse: collapse;
      }}
      th, td {{
        text-align: left;
        border-bottom: 1px solid #e6d8c3;
        padding: 0.7rem 0.5rem;
        vertical-align: top;
      }}
      code {{
        font-size: 0.92em;
      }}
      pre {{
        background: #221a12;
        color: #f7f1e7;
        border-radius: 14px;
        padding: 1rem;
        overflow-x: auto;
      }}
    </style>
  </head>
  <body>
    <main>
      <h1>Performance Baseline</h1>
      <div class="meta">
        <div class="card"><strong>Generated</strong><br>{html.escape(str(payload.get("generated_at")))}</div>
        <div class="card"><strong>Git SHA</strong><br><code>{html.escape(str(payload.get("git_sha")))}</code></div>
        <div class="card"><strong>Total Cases</strong><br>{html.escape(str(payload.get("planned_case_count")))}</div>
        <div class="card"><strong>Baseline Root</strong><br><code>{html.escape(str(payload.get("baseline_root")))}</code></div>
      </div>
      <h2>Matrix</h2>
      <p><strong>Packs:</strong> {html.escape(packs)}</p>
      <p><strong>Budgets:</strong> {html.escape(budgets)}</p>
      <p><strong>Seeds:</strong> {html.escape(seeds)}</p>
      <p><strong>Cache Modes:</strong> {html.escape(cache_modes)}</p>
      <p><strong>Status Counts:</strong> <code>{html.escape(json.dumps(payload.get("status_counts") or {}, sort_keys=True))}</code></p>
      <h2>Review Workflow</h2>
      <p><strong>Workflow doc:</strong> <code>{workflow_doc}</code></p>
      <p><strong>PR template:</strong> <code>{pr_template}</code></p>
      <p><strong>Child issue template:</strong> <code>{child_issue_template}</code></p>
      <p><strong>Branch outcome recording:</strong> <code>{branch_outcome_recording}</code></p>
      <p>Every optimization branch must carry the baseline artifact path, after-change artifact path, exact dashboard/history slices reviewed, and explicit quality, trust-state, and budget-accounting verdicts.</p>
      <h2>System Coverage</h2>
      <table>
        <thead>
          <tr>
            <th>System</th>
            <th>Backends</th>
            <th>Planned Cases</th>
          </tr>
        </thead>
        <tbody>
          {systems_table}
        </tbody>
      </table>
      <h2>Sample Rows</h2>
      <pre>{html.escape(json.dumps(payload.get("sample_rows") or [], indent=2))}</pre>
    </main>
  </body>
</html>
"""


def _resolve_git_sha() -> str:
    compare_root = Path(__file__).resolve().parents[3]
    try:
        result = subprocess.run(
            ["git", "rev-parse", "--short", "HEAD"],
            cwd=compare_root.parent,
            check=True,
            capture_output=True,
            text=True,
        )
    except (OSError, subprocess.CalledProcessError):
        return "unknown"
    return result.stdout.strip() or "unknown"
