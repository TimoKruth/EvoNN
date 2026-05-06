"""Output-quality inspection and performance artifact generation."""

from __future__ import annotations

from collections import Counter
from dataclasses import dataclass
import json
import platform
from pathlib import Path
from typing import Any, Iterable, Mapping

from evonn_shared.contracts import (
    ArtifactCompletenessEnvelope,
    DiagnosticsEnvelope,
    PerformanceEnvelope,
    RuntimeEnvelope,
)
from evonn_shared.manifests import write_json


QUALITY_LEVELS = ("L0", "L1", "L2", "L3", "L4")
SYSTEM_ORDER = ("prism", "topograph", "stratograph", "primordia", "contenders")
ENGINE_EVIDENCE_REQUIREMENTS = {
    "prism": ("family_distribution", "family_benchmark_wins", "operator_mix"),
    "topograph": ("topology_size", "parallel_cache_behavior", "mutation_pressure_policy", "topology_selection_policy"),
    "stratograph": ("macro_depth", "cell_library_size", "reuse_ratio", "hierarchy_evidence"),
    "primordia": ("primitive_count", "primitive_bank_size", "primitive_usage", "group_counts"),
    "contenders": ("contender_family_coverage", "optional_dependency_skips", "baseline_floor_policy_stage", "baseline_floor_evidence"),
}


@dataclass(frozen=True)
class OutputQualityRecord:
    """Normalized output-quality result for one compare-grade run directory."""

    run_dir: Path
    system: str
    run_id: str
    quality_level: str
    measurement_state: str
    manifest_path: Path
    results_path: Path
    summary_path: Path | None
    artifact_completeness: ArtifactCompletenessEnvelope
    runtime: RuntimeEnvelope
    performance: PerformanceEnvelope
    diagnostics: DiagnosticsEnvelope
    benchmark_count: int
    completed_benchmark_count: int
    failed_benchmark_count: int

    def payload(self) -> dict[str, Any]:
        return {
            "run_dir": str(self.run_dir),
            "system": self.system,
            "run_id": self.run_id,
            "quality_level": self.quality_level,
            "measurement_state": self.measurement_state,
            "manifest_path": str(self.manifest_path),
            "results_path": str(self.results_path),
            "summary_path": None if self.summary_path is None else str(self.summary_path),
            "artifact_completeness": self.artifact_completeness.model_dump(mode="json"),
            "runtime": self.runtime.model_dump(mode="json"),
            "performance": self.performance.model_dump(mode="json"),
            "diagnostics": self.diagnostics.model_dump(mode="json"),
            "benchmark_count": self.benchmark_count,
            "completed_benchmark_count": self.completed_benchmark_count,
            "failed_benchmark_count": self.failed_benchmark_count,
        }


def inspect_paths(paths: Iterable[Path], *, write_run_artifacts: bool = True) -> list[OutputQualityRecord]:
    """Inspect every compare-grade run discovered below the supplied paths."""

    seen: set[Path] = set()
    records: list[OutputQualityRecord] = []
    for path in paths:
        for run_dir in discover_run_dirs([path]):
            if run_dir in seen:
                continue
            seen.add(run_dir)
            records.append(inspect_run_dir(run_dir, write_run_artifacts=write_run_artifacts))
    return records


def discover_run_dirs(paths: Iterable[Path]) -> list[Path]:
    """Discover directories that contain compare-grade manifest/results artifacts."""

    discovered: set[Path] = set()
    for raw_path in paths:
        path = raw_path.resolve()
        if path.is_file():
            path = path.parent
        if _is_run_dir(path):
            discovered.add(path)
            continue
        for manifest_path in path.rglob("manifest.json"):
            if _is_ignored_path(manifest_path):
                continue
            candidate = manifest_path.parent
            if _is_run_dir(candidate):
                discovered.add(candidate)
    return sorted(discovered)


def inspect_run_dir(run_dir: Path, *, write_run_artifacts: bool = True) -> OutputQualityRecord:
    """Inspect one run directory and optionally write normalized output artifacts."""

    run_path = run_dir.resolve()
    manifest_path = run_path / "manifest.json"
    results_path = run_path / "results.json"
    summary_path = run_path / "summary.json"

    manifest = _read_json(manifest_path)
    result_payloads = json.loads(results_path.read_text(encoding="utf-8"))
    results = result_payloads if isinstance(result_payloads, list) else []
    summary = _read_json(summary_path) if summary_path.exists() else {}

    artifacts = _artifact_completeness(run_path, manifest=manifest, summary_exists=summary_path.exists())
    runtime = _runtime_envelope(manifest=manifest, summary=summary)
    performance = _performance_envelope(manifest=manifest, results=results, summary=summary)
    diagnostics = _diagnostics_envelope(
        manifest=manifest,
        summary=summary,
        results=results,
        artifacts=artifacts,
        runtime=runtime,
        performance=performance,
        summary_exists=summary_path.exists(),
    )
    quality_level = _quality_level(artifacts=artifacts, diagnostics=diagnostics)
    measurement_state = _measurement_state(performance)

    record = OutputQualityRecord(
        run_dir=run_path,
        system=str(manifest.get("system") or "unknown"),
        run_id=str(manifest.get("run_id") or run_path.name),
        quality_level=quality_level,
        measurement_state=measurement_state,
        manifest_path=manifest_path,
        results_path=results_path,
        summary_path=summary_path if summary_path.exists() else None,
        artifact_completeness=artifacts,
        runtime=runtime,
        performance=performance,
        diagnostics=diagnostics,
        benchmark_count=len(_manifest_benchmarks(manifest)),
        completed_benchmark_count=sum(1 for result in results if _result_status(result) == "ok"),
        failed_benchmark_count=sum(1 for result in results if _result_status(result) == "failed"),
    )

    if write_run_artifacts:
        write_json(run_path / "runtime.json", runtime.model_dump(mode="json"))
        write_json(run_path / "performance.json", performance.model_dump(mode="json"))
        write_json(run_path / "diagnostics.json", diagnostics.model_dump(mode="json"))
        write_json(run_path / "output_quality_report.json", record.payload())
        (run_path / "output_quality_report.md").write_text(render_run_markdown(record), encoding="utf-8")

    return record


def write_aggregate_report(records: list[OutputQualityRecord], output_path: Path) -> dict[str, Path]:
    """Write aggregate markdown and JSON reports for one or more inspected workspaces."""

    output_path.parent.mkdir(parents=True, exist_ok=True)
    output_path.write_text(render_aggregate_markdown(records), encoding="utf-8")
    json_path = output_path.with_suffix(".json")
    write_json(json_path, {"runs": [record.payload() for record in records]})
    return {"markdown": output_path, "json": json_path}


def render_run_markdown(record: OutputQualityRecord) -> str:
    perf = record.performance
    runtime = record.runtime
    lines = [
        f"# Output Quality: {record.system}",
        "",
        "| Field | Value |",
        "| --- | --- |",
        f"| Run ID | `{record.run_id}` |",
        f"| Quality Level | `{record.quality_level}` |",
        f"| Measurement State | `{record.measurement_state}` |",
        f"| Benchmarks OK | {record.completed_benchmark_count}/{record.benchmark_count} |",
        f"| Backend | `{runtime.runtime_backend or 'unknown'}` |",
        f"| Requested Backend | `{runtime.runtime_backend_requested or 'unknown'}` |",
        f"| Wall Clock Seconds | {_fmt_float(perf.wall_clock_seconds)} |",
        f"| Evals/Sec | {_fmt_float(perf.evals_per_second)} |",
        f"| Cache Reuse Rate | {_fmt_float(perf.cache_reuse_rate)} |",
        f"| Missing Required Artifacts | `{', '.join(record.artifact_completeness.required_missing) or 'none'}` |",
        f"| Missing L2 Fields | `{', '.join(record.diagnostics.missing_l2_fields) or 'none'}` |",
        f"| Missing L3 Fields | `{', '.join(record.diagnostics.missing_l3_fields) or 'none'}` |",
        f"| Missing L4 Fields | `{', '.join(record.diagnostics.missing_l4_fields) or 'none'}` |",
    ]
    return "\n".join(lines) + "\n"


def render_aggregate_markdown(records: list[OutputQualityRecord]) -> str:
    lines = [
        "# Output Quality Overview",
        "",
        "| Workspace | System | Level | Measurement | OK | Failed | Backend | Wall s | Eval/s | Cache Reuse | Missing L3 |",
        "| --- | --- | --- | --- | ---: | ---: | --- | ---: | ---: | ---: | --- |",
    ]
    for record in sorted(records, key=_record_sort_key):
        lines.append(
            "| {workspace} | {system} | `{level}` | `{measurement}` | {ok}/{total} | {failed} | "
            "`{backend}` | {wall} | {eps} | {cache} | `{missing}` |".format(
                workspace=_workspace_label(record.run_dir),
                system=record.system,
                level=record.quality_level,
                measurement=record.measurement_state,
                ok=record.completed_benchmark_count,
                total=record.benchmark_count,
                failed=record.failed_benchmark_count,
                backend=record.runtime.runtime_backend or "unknown",
                wall=_fmt_float(record.performance.wall_clock_seconds),
                eps=_fmt_float(record.performance.evals_per_second),
                cache=_fmt_float(record.performance.cache_reuse_rate),
                missing=", ".join(record.diagnostics.missing_l3_fields) or "none",
            )
        )

    lines.extend(_comparison_markdown(records))
    return "\n".join(lines) + "\n"


def _comparison_markdown(records: list[OutputQualityRecord]) -> list[str]:
    labels = list(dict.fromkeys(_workspace_label(record.run_dir) for record in records))
    if len(labels) < 2:
        return []
    left_label, right_label = labels[-2], labels[-1]
    by_key = {(_workspace_label(record.run_dir), record.system): record for record in records}
    lines = [
        "",
        f"## Last Two Workspace Delta: `{left_label}` -> `{right_label}`",
        "",
        "| System | Level | Measurement | Wall Delta s | Eval/s Delta | OK Delta | Missing L3 Delta |",
        "| --- | --- | --- | ---: | ---: | ---: | ---: |",
    ]
    for system in SYSTEM_ORDER:
        left = by_key.get((left_label, system))
        right = by_key.get((right_label, system))
        if left is None or right is None:
            continue
        lines.append(
            "| {system} | `{left_level}->{right_level}` | `{left_measure}->{right_measure}` | "
            "{wall_delta} | {eps_delta} | {ok_delta} | {missing_delta} |".format(
                system=system,
                left_level=left.quality_level,
                right_level=right.quality_level,
                left_measure=left.measurement_state,
                right_measure=right.measurement_state,
                wall_delta=_fmt_delta(left.performance.wall_clock_seconds, right.performance.wall_clock_seconds),
                eps_delta=_fmt_delta(left.performance.evals_per_second, right.performance.evals_per_second),
                ok_delta=right.completed_benchmark_count - left.completed_benchmark_count,
                missing_delta=len(right.diagnostics.missing_l3_fields) - len(left.diagnostics.missing_l3_fields),
            )
        )
    return lines


def _artifact_completeness(
    run_dir: Path,
    *,
    manifest: Mapping[str, Any],
    summary_exists: bool,
) -> ArtifactCompletenessEnvelope:
    artifacts = _mapping(manifest.get("artifacts"))
    config_snapshot = _first_text(artifacts.get("config_snapshot")) or "config.yaml"
    report_markdown = _first_text(artifacts.get("report_markdown")) or "report.md"
    required = {
        "manifest.json": run_dir / "manifest.json",
        "results.json": run_dir / "results.json",
        "summary.json": run_dir / "summary.json",
        "config_snapshot": _artifact_path(run_dir, config_snapshot),
        "report_markdown": _artifact_path(run_dir, report_markdown),
    }
    optional_paths = {
        "model_summary_json": artifacts.get("model_summary_json"),
        "genome_summary_json": artifacts.get("genome_summary_json"),
        "contender_summary_json": artifacts.get("contender_summary_json"),
        "dataset_manifest_json": artifacts.get("dataset_manifest_json"),
        "raw_database": artifacts.get("raw_database"),
        "performance.json": "performance.json",
        "diagnostics.json": "diagnostics.json",
    }
    missing = tuple(name for name, path in required.items() if not path.exists())
    optional_present: list[str] = []
    optional_missing: list[str] = []
    for name, rel_path in optional_paths.items():
        if rel_path is None:
            optional_missing.append(name)
        elif _artifact_path(run_dir, str(rel_path)).exists():
            optional_present.append(name)
        else:
            optional_missing.append(name)
    if not summary_exists and "summary.json" not in missing:
        missing = (*missing, "summary.json")
    return ArtifactCompletenessEnvelope(
        required_present=not missing,
        required_missing=missing,
        optional_present=tuple(optional_present),
        optional_missing=tuple(optional_missing),
    )


def _runtime_envelope(*, manifest: Mapping[str, Any], summary: Mapping[str, Any]) -> RuntimeEnvelope:
    device = _mapping(manifest.get("device"))
    budget = _mapping(manifest.get("budget"))
    runtime_policy = _mapping(summary.get("runtime_execution_policy") or budget.get("runtime_execution_policy"))
    requested = _first_text(
        summary.get("runtime_backend_requested"),
        summary.get("requested_runtime_backend"),
        device.get("framework_requested"),
    )
    resolved = _first_text(
        summary.get("runtime_backend"),
        runtime_policy.get("runtime_backend"),
        device.get("framework"),
    )
    return RuntimeEnvelope(
        runtime_backend_requested=requested,
        runtime_backend=resolved,
        runtime_backend_limitations=_first_text(
            summary.get("runtime_backend_limitations"),
            device.get("framework_limitations"),
        ),
        device_name=_first_text(device.get("device_name")),
        framework=_first_text(device.get("framework")),
        framework_version=_first_text(device.get("framework_version")),
        precision_mode=_first_text(device.get("precision_mode")),
        hardware_class=_first_text(summary.get("hardware_class"), device.get("device_name"), platform.machine()),
        worker_count=_first_int(
            budget.get("resolved_parallel_workers_max"),
            budget.get("resolved_worker_count"),
            summary.get("worker_count"),
        ),
        os=platform.system().lower(),
        python_version=platform.python_version(),
    )


def _performance_envelope(
    *,
    manifest: Mapping[str, Any],
    results: list[Mapping[str, Any]],
    summary: Mapping[str, Any],
) -> PerformanceEnvelope:
    budget = _mapping(manifest.get("budget"))
    wall_clock = _first_float(budget.get("wall_clock_seconds"), summary.get("wall_clock_seconds"))
    train_values = [
        float(result["train_seconds"])
        for result in results
        if isinstance(result, Mapping) and result.get("train_seconds") is not None
    ]
    train_total = sum(train_values) if train_values else None
    train_mean = train_total / len(train_values) if train_values else None
    actual_evals = _accounted_evaluations(budget)
    evals_per_second = (float(actual_evals) / wall_clock) if wall_clock and wall_clock > 0 else None
    median_quality = _first_float(summary.get("median_benchmark_quality"))
    quality_per_second = (median_quality / wall_clock) if wall_clock and wall_clock > 0 and median_quality is not None else None
    peak_values = [
        float(result["peak_memory_mb"])
        for result in results
        if isinstance(result, Mapping) and result.get("peak_memory_mb") is not None
    ]
    cache_reuse_count = _first_int(
        budget.get("cache_reused_count"),
        budget.get("reused_count"),
        budget.get("cached_evaluations"),
    )
    cache_reuse_rate = _first_float(budget.get("cache_reuse_rate"))
    if cache_reuse_rate is None and cache_reuse_count is not None:
        cache_reuse_rate = cache_reuse_count / max(int(actual_evals), 1)

    values = {
        "wall_clock_seconds": wall_clock,
        "benchmark_total_seconds": _first_float(budget.get("benchmark_total_seconds"), summary.get("benchmark_total_seconds")),
        "data_load_seconds": _first_float(budget.get("data_load_seconds"), summary.get("data_load_seconds")),
        "evaluation_seconds": _first_float(budget.get("evaluation_seconds"), summary.get("evaluation_seconds")),
        "export_seconds": _first_float(summary.get("export_seconds")),
        "train_seconds_total": train_total,
        "train_seconds_mean": train_mean,
        "evals_per_second": evals_per_second,
        "quality_per_second": quality_per_second,
        "cache_hits": _first_int(budget.get("data_cache_hits"), budget.get("cache_hits")),
        "cache_misses": _first_int(budget.get("data_cache_misses")),
        "cache_reuse_count": cache_reuse_count,
        "cache_reuse_rate": cache_reuse_rate,
        "requested_worker_count": _first_int(budget.get("requested_parallel_workers")),
        "resolved_worker_count": _first_int(budget.get("resolved_parallel_workers_max")),
        "worker_clamp_reason": _first_text(summary.get("worker_clamp_reason")),
        "peak_memory_mb": max(peak_values) if peak_values else None,
    }
    unavailable = tuple(key for key, value in values.items() if value is None)
    notes = tuple(f"{key} unavailable" for key in unavailable)
    return PerformanceEnvelope(**values, unavailable_fields=unavailable, notes=notes)


def _diagnostics_envelope(
    *,
    manifest: Mapping[str, Any],
    summary: Mapping[str, Any],
    results: list[Mapping[str, Any]],
    artifacts: ArtifactCompletenessEnvelope,
    runtime: RuntimeEnvelope,
    performance: PerformanceEnvelope,
    summary_exists: bool,
) -> DiagnosticsEnvelope:
    budget = _mapping(manifest.get("budget"))
    fairness = _mapping(manifest.get("fairness"))
    result_ids = {str(result.get("benchmark_id")) for result in results if isinstance(result, Mapping)}
    manifest_ids = {
        str(entry.get("benchmark_id"))
        for entry in _manifest_benchmarks(manifest)
        if isinstance(entry, Mapping)
    }
    status_counts = Counter(_result_status(result) for result in results)
    missing_l2 = []
    if not fairness:
        missing_l2.append("fairness")
    else:
        if fairness.get("data_signature") is None:
            missing_l2.append("fairness.data_signature")
        if fairness.get("code_version") is None:
            missing_l2.append("fairness.code_version")
    if budget.get("actual_evaluations") is None:
        missing_l2.append("budget.actual_evaluations")
    if not budget.get("evaluation_semantics"):
        missing_l2.append("budget.evaluation_semantics")
    if not budget.get("budget_policy_name"):
        missing_l2.append("budget.budget_policy_name")
    if result_ids != manifest_ids:
        missing_l2.append("results.coverage_matches_manifest")

    missing_l3 = []
    if not runtime.runtime_backend:
        missing_l3.append("runtime.runtime_backend")
    if not runtime.device_name:
        missing_l3.append("runtime.device_name")
    if not runtime.framework:
        missing_l3.append("runtime.framework")
    if performance.wall_clock_seconds is None:
        missing_l3.append("performance.wall_clock_seconds")
    if performance.evals_per_second is None:
        missing_l3.append("performance.evals_per_second")
    if performance.train_seconds_total is None:
        missing_l3.append("performance.train_seconds_total")

    engine_evidence = _mapping(summary.get("engine_evidence"))
    missing_l4 = []
    for field in ENGINE_EVIDENCE_REQUIREMENTS.get(str(manifest.get("system") or "unknown"), ()): 
        if field not in engine_evidence or engine_evidence.get(field) in (None, "", [], {}):
            missing_l4.append(f"engine_evidence.{field}")

    warnings = []
    if not summary_exists:
        warnings.append("summary.json missing")
    warnings.extend(f"{field} unavailable" for field in performance.unavailable_fields)

    return DiagnosticsEnvelope(
        status="ok" if not artifacts.required_missing and not missing_l2 else "needs-attention",
        benchmark_status_counts=dict(sorted(status_counts.items())),
        failure_reason_by_benchmark={
            str(result.get("benchmark_id")): str(result.get("failure_reason"))
            for result in results
            if isinstance(result, Mapping) and result.get("failure_reason")
        },
        missing_required_artifacts=artifacts.required_missing,
        missing_l2_fields=tuple(missing_l2),
        missing_l3_fields=tuple(missing_l3),
        missing_l4_fields=tuple(missing_l4),
        warnings=tuple(warnings),
    )


def _quality_level(*, artifacts: ArtifactCompletenessEnvelope, diagnostics: DiagnosticsEnvelope) -> str:
    if artifacts.required_missing:
        return "L0"
    if diagnostics.missing_l2_fields:
        return "L1"
    if diagnostics.missing_l3_fields:
        return "L2"
    if diagnostics.missing_l4_fields:
        return "L3"
    return "L4"


def _measurement_state(performance: PerformanceEnvelope) -> str:
    if performance.wall_clock_seconds is not None and performance.evals_per_second is not None:
        return "measurable"
    if performance.wall_clock_seconds is not None:
        return "runtime-only"
    return "limited"


def _is_run_dir(path: Path) -> bool:
    return (path / "manifest.json").exists() and (path / "results.json").exists()


def _is_ignored_path(path: Path) -> bool:
    parts = set(path.parts)
    return bool({".venv", "__pycache__", ".pytest_cache", ".ruff_cache"} & parts)


def _read_json(path: Path) -> dict[str, Any]:
    payload = json.loads(path.read_text(encoding="utf-8"))
    return payload if isinstance(payload, dict) else {}


def _mapping(value: Any) -> Mapping[str, Any]:
    return value if isinstance(value, Mapping) else {}


def _manifest_benchmarks(manifest: Mapping[str, Any]) -> list[Mapping[str, Any]]:
    benchmarks = manifest.get("benchmarks")
    return benchmarks if isinstance(benchmarks, list) else []


def _result_status(result: Mapping[str, Any]) -> str:
    return str(result.get("status") or "missing")


def _accounted_evaluations(budget: Mapping[str, Any]) -> int:
    actual = _first_int(budget.get("actual_evaluations"))
    cached = _first_int(budget.get("cached_evaluations")) or 0
    if actual is not None:
        return actual + cached
    return _first_int(budget.get("evaluation_count")) or 0


def _artifact_path(run_dir: Path, value: str) -> Path:
    path = Path(value)
    return path if path.is_absolute() else run_dir / path


def _first_text(*values: Any) -> str | None:
    for value in values:
        if value is None:
            continue
        text = str(value).strip()
        if text:
            return text
    return None


def _first_float(*values: Any) -> float | None:
    for value in values:
        if value is None:
            continue
        try:
            return float(value)
        except (TypeError, ValueError):
            continue
    return None


def _first_int(*values: Any) -> int | None:
    for value in values:
        if value is None:
            continue
        try:
            return int(value)
        except (TypeError, ValueError):
            continue
    return None


def _workspace_label(run_dir: Path) -> str:
    parts = run_dir.parts
    if "runs" in parts:
        index = parts.index("runs")
        if index > 0:
            return parts[index - 1]
    return run_dir.parent.name


def _record_sort_key(record: OutputQualityRecord) -> tuple[int, str, str]:
    try:
        system_index = SYSTEM_ORDER.index(record.system)
    except ValueError:
        system_index = len(SYSTEM_ORDER)
    return (system_index, _workspace_label(record.run_dir), record.run_id)


def _fmt_float(value: float | None) -> str:
    return "n/a" if value is None else f"{value:.4f}"


def _fmt_delta(left: float | None, right: float | None) -> str:
    if left is None or right is None:
        return "n/a"
    return f"{right - left:+.4f}"
