"""Shared performance report orchestration for before/after review surfaces."""

from __future__ import annotations

from collections import Counter, defaultdict
from dataclasses import dataclass
from datetime import datetime, timezone
import json
from pathlib import Path
from statistics import median
from typing import Any

from evonn_compare.contracts.performance import PerformanceBaselineManifest, PerformanceRow
from evonn_compare.reporting.performance_dashboard import render_performance_dashboard_html
from evonn_compare.reporting.performance_report_md import render_performance_report_markdown

KNOWN_OUTCOMES = {"candidate", "accepted", "rejected-for-revision", "scrapped"}
QUALITY_FAIL_STATUSES = {"fail", "regressed"}
TRUST_FAIL_STATUSES = {"fail", "regressed"}


@dataclass(frozen=True)
class PerformanceDataset:
    label: str
    source_path: Path
    outcome: str
    rows: list[PerformanceRow]
    manifest: PerformanceBaselineManifest | None = None


def build_performance_report(
    *,
    baseline_label: str,
    baseline_path: Path,
    candidate_specs: list[tuple[str, Path]],
    outcomes: dict[str, str],
    compare_label: str | None,
    output_root: Path,
) -> dict[str, str | int]:
    """Load canonical performance rows and render report artifacts."""

    baseline = load_performance_dataset(
        label=baseline_label,
        source=baseline_path,
        outcome="baseline",
    )
    candidates = [
        load_performance_dataset(
            label=label,
            source=source_path,
            outcome=outcomes.get(label, "candidate"),
        )
        for label, source_path in candidate_specs
    ]
    selected_compare = compare_label or (candidates[0].label if candidates else None)
    payload = build_performance_report_payload(
        baseline=baseline,
        candidates=candidates,
        compare_label=selected_compare,
    )

    output_dir = output_root.resolve()
    output_dir.mkdir(parents=True, exist_ok=True)
    json_path = output_dir / "performance_report.json"
    markdown_path = output_dir / "performance_report.md"
    html_path = output_dir / "performance_dashboard.html"
    json_path.write_text(json.dumps(payload, indent=2), encoding="utf-8")
    markdown_path.write_text(render_performance_report_markdown(payload), encoding="utf-8")
    html_path.write_text(render_performance_dashboard_html(payload), encoding="utf-8")

    return {
        "baseline_label": baseline.label,
        "baseline_source": str(baseline.source_path),
        "candidate_count": len(candidates),
        "compare_label": selected_compare or "",
        "report_json": str(json_path),
        "report_markdown": str(markdown_path),
        "dashboard_html": str(html_path),
        "history_count": len(payload["optimization_history"]),
    }


def build_performance_report_payload(
    *,
    baseline: PerformanceDataset,
    candidates: list[PerformanceDataset],
    compare_label: str | None,
) -> dict[str, Any]:
    """Build the canonical performance review payload from normalized rows."""

    generated_at = datetime.now(timezone.utc).isoformat()
    datasets = [baseline, *candidates]
    dataset_summaries = [_build_dataset_summary(dataset) for dataset in datasets]
    comparison_rows = {
        candidate.label: _build_delta_rows(baseline=baseline, candidate=candidate)
        for candidate in candidates
    }
    primary_label = compare_label if compare_label in comparison_rows else None
    primary_deltas = comparison_rows.get(primary_label or "", [])
    optimization_history = [
        _build_history_entry(
            baseline=baseline,
            candidate=candidate,
            deltas=comparison_rows[candidate.label],
        )
        for candidate in candidates
    ]
    review_references = (
        baseline.manifest.review_references.model_dump(mode="json")
        if baseline.manifest is not None
        else None
    )
    return {
        "generated_at": generated_at,
        "baseline_label": baseline.label,
        "compare_label": primary_label,
        "review_references": review_references,
        "datasets": dataset_summaries,
        "primary_comparison": {
            "available": primary_label is not None,
            "baseline_label": baseline.label,
            "candidate_label": primary_label,
            "summary": _aggregate_delta_rows(primary_deltas),
            "deltas": primary_deltas,
        },
        "optimization_history": optimization_history,
    }


def load_performance_dataset(*, label: str, source: Path, outcome: str) -> PerformanceDataset:
    """Resolve a dataset directory or JSONL path into validated performance rows."""

    if outcome != "baseline" and outcome not in KNOWN_OUTCOMES:
        available = ", ".join(sorted(KNOWN_OUTCOMES))
        raise ValueError(f"unknown outcome '{outcome}'; expected one of: {available}")

    source_path = source.resolve()
    rows_path, manifest = _resolve_rows_path(source_path)
    rows = [
        PerformanceRow.model_validate_json(line)
        for line in rows_path.read_text(encoding="utf-8").splitlines()
        if line.strip()
    ]
    if not rows:
        raise ValueError(f"performance dataset '{label}' at {rows_path} does not contain any rows")
    return PerformanceDataset(
        label=label,
        source_path=rows_path,
        outcome=outcome,
        rows=rows,
        manifest=manifest,
    )


def _resolve_rows_path(source: Path) -> tuple[Path, PerformanceBaselineManifest | None]:
    if source.is_file():
        if source.name == "baseline_manifest.json":
            manifest = PerformanceBaselineManifest.model_validate_json(
                source.read_text(encoding="utf-8")
            )
            rows_path = Path(manifest.artifacts.perf_rows)
            if not rows_path.is_absolute():
                rows_path = (source.parent / rows_path).resolve()
            if not rows_path.exists():
                raise ValueError(f"manifest points to missing perf_rows.jsonl: {rows_path}")
            return rows_path, manifest
        return source, _load_manifest_from_rows(source)

    if not source.exists():
        raise ValueError(f"performance dataset path does not exist: {source}")

    direct_rows = source / "perf_rows.jsonl"
    if direct_rows.exists():
        return direct_rows, _load_manifest_from_rows(direct_rows)

    manifest_path = source / "baseline_manifest.json"
    if manifest_path.exists():
        manifest = PerformanceBaselineManifest.model_validate_json(
            manifest_path.read_text(encoding="utf-8")
        )
        rows_path = Path(manifest.artifacts.perf_rows)
        if not rows_path.is_absolute():
            rows_path = (manifest_path.parent / rows_path).resolve()
        if not rows_path.exists():
            raise ValueError(f"manifest points to missing perf_rows.jsonl: {rows_path}")
        return rows_path, manifest

    raise ValueError(
        f"could not resolve perf_rows.jsonl from {source}; supply the file directly or a baseline root"
    )


def _load_manifest_from_rows(rows_path: Path) -> PerformanceBaselineManifest | None:
    manifest_path = rows_path.with_name("baseline_manifest.json")
    if not manifest_path.exists():
        return None
    return PerformanceBaselineManifest.model_validate_json(manifest_path.read_text(encoding="utf-8"))


def _build_dataset_summary(dataset: PerformanceDataset) -> dict[str, Any]:
    status_counts = Counter(row.status for row in dataset.rows)
    quality_status_counts = Counter(row.quality_guard.status for row in dataset.rows)
    trust_status_counts = Counter(row.trust_guard.status for row in dataset.rows)
    slices: list[dict[str, Any]] = []
    grouped: dict[tuple[str, ...], list[PerformanceRow]] = defaultdict(list)
    for row in dataset.rows:
        grouped[_slice_key(row)].append(row)

    for key in sorted(grouped):
        rows = grouped[key]
        slices.append(
            {
                "system": key[0],
                "backend_class": key[1],
                "backend_label": key[2],
                "host_label": key[3],
                "pack_name": key[4],
                "budget": int(key[5]),
                "cache_mode": key[6],
                "accounting_tags": list(_collect_accounting_tags(rows)),
                "actual_evaluations": _sum_optional_row_field(rows, "actual_evaluations"),
                "cached_evaluations": _sum_optional_row_field(rows, "cached_evaluations"),
                "resumed_evaluations": _sum_optional_row_field(rows, "resumed_evaluations"),
                "screened_evaluations": _sum_optional_row_field(rows, "screened_evaluations"),
                "deduplicated_evaluations": _sum_optional_row_field(rows, "deduplicated_evaluations"),
                "reduced_fidelity_evaluations": _sum_optional_row_field(rows, "reduced_fidelity_evaluations"),
                "case_count": len(rows),
                "seed_count": len({row.seed for row in rows}),
                "status_counts": dict(sorted(Counter(row.status for row in rows).items())),
                "median_wall_clock_seconds": _median_metric(rows, "wall_clock_seconds"),
                "median_evals_per_second": _median_metric(rows, "evals_per_second"),
                "median_train_seconds": _median_metric(rows, "train_seconds"),
                "median_data_load_seconds": _median_metric(rows, "data_load_seconds"),
                "median_cache_hit_rate": _median_metric(rows, "cache_hit_rate"),
                "median_reuse_rate": _median_metric(rows, "reuse_rate"),
                "total_failure_count": sum((row.metrics.failure_count or 0) for row in rows),
                "median_quality_delta_vs_baseline": _median_quality_delta(rows),
                "median_rank_delta_vs_baseline": _median_rank_delta(rows),
                "quality_regression_count": sum(1 for row in rows if _quality_regressed(row)),
                "trust_regression_count": sum(1 for row in rows if _trust_regressed(row)),
            }
        )

    return {
        "label": dataset.label,
        "outcome": dataset.outcome,
        "source_path": str(dataset.source_path),
        "manifest_path": None if dataset.manifest is None else str(dataset.source_path.with_name("baseline_manifest.json")),
        "row_count": len(dataset.rows),
        "accounting_tags": list(_collect_accounting_tags(dataset.rows)),
        "status_counts": dict(sorted(status_counts.items())),
        "quality_status_counts": dict(sorted(quality_status_counts.items())),
        "trust_status_counts": dict(sorted(trust_status_counts.items())),
        "review_references": (
            None if dataset.manifest is None else dataset.manifest.review_references.model_dump(mode="json")
        ),
        "slices": slices,
    }


def _build_delta_rows(*, baseline: PerformanceDataset, candidate: PerformanceDataset) -> list[dict[str, Any]]:
    baseline_index = {_case_key(row): row for row in baseline.rows}
    candidate_index = {_case_key(row): row for row in candidate.rows}
    slice_state: dict[tuple[str, ...], dict[str, Any]] = {}

    for case_key, candidate_row in candidate_index.items():
        slice_key = _slice_key(candidate_row)
        state = slice_state.setdefault(
            slice_key,
            {
                "pairs": [],
                "candidate_only": 0,
                "baseline_only": 0,
            },
        )
        baseline_row = baseline_index.get(case_key)
        if baseline_row is None:
            state["candidate_only"] += 1
        else:
            state["pairs"].append((baseline_row, candidate_row))

    for case_key, baseline_row in baseline_index.items():
        if case_key in candidate_index:
            continue
        slice_key = _slice_key(baseline_row)
        state = slice_state.setdefault(
            slice_key,
            {
                "pairs": [],
                "candidate_only": 0,
                "baseline_only": 0,
            },
        )
        state["baseline_only"] += 1

    delta_rows: list[dict[str, Any]] = []
    for key in sorted(slice_state):
        state = slice_state[key]
        pairs: list[tuple[PerformanceRow, PerformanceRow]] = state["pairs"]
        baseline_rows = [pair[0] for pair in pairs]
        candidate_rows = [pair[1] for pair in pairs]
        wall_clock_pct = _metric_pct_delta(baseline_rows, candidate_rows, "wall_clock_seconds")
        evals_pct = _metric_pct_delta(baseline_rows, candidate_rows, "evals_per_second")
        failure_delta = sum((row.metrics.failure_count or 0) for row in candidate_rows) - sum(
            (row.metrics.failure_count or 0) for row in baseline_rows
        )
        quality_regressions = sum(1 for row in candidate_rows if _quality_regressed(row))
        trust_regressions = sum(1 for row in candidate_rows if _trust_regressed(row))
        delta_rows.append(
            {
                "candidate_label": candidate.label,
                "candidate_outcome": candidate.outcome,
                "system": key[0],
                "backend_class": key[1],
                "backend_label": key[2],
                "host_label": key[3],
                "pack_name": key[4],
                "budget": int(key[5]),
                "cache_mode": key[6],
                "matched_case_count": len(pairs),
                "candidate_only_case_count": int(state["candidate_only"]),
                "baseline_only_case_count": int(state["baseline_only"]),
                "baseline_accounting_tags": list(_collect_accounting_tags(baseline_rows)),
                "candidate_accounting_tags": list(_collect_accounting_tags(candidate_rows)),
                "candidate_actual_evaluations": _sum_optional_row_field(candidate_rows, "actual_evaluations"),
                "candidate_cached_evaluations": _sum_optional_row_field(candidate_rows, "cached_evaluations"),
                "candidate_resumed_evaluations": _sum_optional_row_field(candidate_rows, "resumed_evaluations"),
                "candidate_screened_evaluations": _sum_optional_row_field(candidate_rows, "screened_evaluations"),
                "candidate_deduplicated_evaluations": _sum_optional_row_field(candidate_rows, "deduplicated_evaluations"),
                "candidate_reduced_fidelity_evaluations": _sum_optional_row_field(candidate_rows, "reduced_fidelity_evaluations"),
                "baseline_median_wall_clock_seconds": _median_metric(baseline_rows, "wall_clock_seconds"),
                "candidate_median_wall_clock_seconds": _median_metric(candidate_rows, "wall_clock_seconds"),
                "wall_clock_delta_pct": wall_clock_pct,
                "baseline_median_evals_per_second": _median_metric(baseline_rows, "evals_per_second"),
                "candidate_median_evals_per_second": _median_metric(candidate_rows, "evals_per_second"),
                "evals_per_second_delta_pct": evals_pct,
                "cache_hit_rate_delta": _metric_delta(baseline_rows, candidate_rows, "cache_hit_rate"),
                "reuse_rate_delta": _metric_delta(baseline_rows, candidate_rows, "reuse_rate"),
                "failure_count_delta": failure_delta,
                "median_quality_delta_vs_baseline": _median_quality_delta(candidate_rows),
                "median_rank_delta_vs_baseline": _median_rank_delta(candidate_rows),
                "quality_regression_count": quality_regressions,
                "trust_regression_count": trust_regressions,
                "verdict": _verdict_for_delta(
                    wall_clock_delta_pct=wall_clock_pct,
                    evals_per_second_delta_pct=evals_pct,
                    quality_regression_count=quality_regressions,
                    trust_regression_count=trust_regressions,
                    failure_count_delta=failure_delta,
                ),
            }
        )
    return delta_rows


def _build_history_entry(
    *,
    baseline: PerformanceDataset,
    candidate: PerformanceDataset,
    deltas: list[dict[str, Any]],
) -> dict[str, Any]:
    summary = _aggregate_delta_rows(deltas)
    return {
        "label": candidate.label,
        "outcome": candidate.outcome,
        "source_path": str(candidate.source_path),
        "baseline_label": baseline.label,
        **summary,
    }


def _aggregate_delta_rows(deltas: list[dict[str, Any]]) -> dict[str, Any]:
    wall_clock_values = [
        float(row["wall_clock_delta_pct"])
        for row in deltas
        if row.get("wall_clock_delta_pct") is not None
    ]
    evals_values = [
        float(row["evals_per_second_delta_pct"])
        for row in deltas
        if row.get("evals_per_second_delta_pct") is not None
    ]
    failure_delta = sum(int(row.get("failure_count_delta") or 0) for row in deltas)
    quality_regressions = sum(int(row.get("quality_regression_count") or 0) for row in deltas)
    trust_regressions = sum(int(row.get("trust_regression_count") or 0) for row in deltas)
    matched_cases = sum(int(row.get("matched_case_count") or 0) for row in deltas)
    candidate_only_cases = sum(int(row.get("candidate_only_case_count") or 0) for row in deltas)
    baseline_only_cases = sum(int(row.get("baseline_only_case_count") or 0) for row in deltas)
    baseline_accounting_tags = sorted(
        {
            str(tag)
            for row in deltas
            for tag in row.get("baseline_accounting_tags") or []
        }
    )
    candidate_accounting_tags = sorted(
        {
            str(tag)
            for row in deltas
            for tag in row.get("candidate_accounting_tags") or []
        }
    )
    wall_clock_pct = None if not wall_clock_values else median(wall_clock_values)
    evals_pct = None if not evals_values else median(evals_values)
    return {
        "slice_count": len(deltas),
        "matched_case_count": matched_cases,
        "candidate_only_case_count": candidate_only_cases,
        "baseline_only_case_count": baseline_only_cases,
        "baseline_accounting_tags": baseline_accounting_tags,
        "candidate_accounting_tags": candidate_accounting_tags,
        "candidate_actual_evaluations": _sum_optional_delta_field(deltas, "candidate_actual_evaluations"),
        "candidate_cached_evaluations": _sum_optional_delta_field(deltas, "candidate_cached_evaluations"),
        "candidate_resumed_evaluations": _sum_optional_delta_field(deltas, "candidate_resumed_evaluations"),
        "candidate_screened_evaluations": _sum_optional_delta_field(deltas, "candidate_screened_evaluations"),
        "candidate_deduplicated_evaluations": _sum_optional_delta_field(deltas, "candidate_deduplicated_evaluations"),
        "candidate_reduced_fidelity_evaluations": _sum_optional_delta_field(deltas, "candidate_reduced_fidelity_evaluations"),
        "median_wall_clock_delta_pct": wall_clock_pct,
        "median_evals_per_second_delta_pct": evals_pct,
        "failure_count_delta": failure_delta,
        "quality_regression_count": quality_regressions,
        "trust_regression_count": trust_regressions,
        "verdict": _verdict_for_delta(
            wall_clock_delta_pct=wall_clock_pct,
            evals_per_second_delta_pct=evals_pct,
            quality_regression_count=quality_regressions,
            trust_regression_count=trust_regressions,
            failure_count_delta=failure_delta,
        ),
    }


def _slice_key(row: PerformanceRow) -> tuple[str, ...]:
    return (
        row.system,
        row.backend_class,
        row.backend_label,
        row.host_label,
        row.pack_name,
        str(row.budget),
        row.cache_mode,
    )


def _case_key(row: PerformanceRow) -> tuple[str, ...]:
    return (
        row.system,
        row.backend_class,
        row.backend_label,
        row.host_label,
        row.pack_name,
        str(row.budget),
        str(row.seed),
        row.cache_mode,
    )


def _median_metric(rows: list[PerformanceRow], field_name: str) -> float | None:
    values = [
        float(value)
        for row in rows
        if (value := getattr(row.metrics, field_name)) is not None
    ]
    return None if not values else float(median(values))


def _metric_delta(
    baseline_rows: list[PerformanceRow],
    candidate_rows: list[PerformanceRow],
    field_name: str,
) -> float | None:
    baseline_value = _median_metric(baseline_rows, field_name)
    candidate_value = _median_metric(candidate_rows, field_name)
    if baseline_value is None or candidate_value is None:
        return None
    return float(candidate_value - baseline_value)


def _metric_pct_delta(
    baseline_rows: list[PerformanceRow],
    candidate_rows: list[PerformanceRow],
    field_name: str,
) -> float | None:
    baseline_value = _median_metric(baseline_rows, field_name)
    candidate_value = _median_metric(candidate_rows, field_name)
    if baseline_value is None or candidate_value is None or abs(baseline_value) <= 1e-12:
        return None
    return float(((candidate_value - baseline_value) / baseline_value) * 100.0)


def _collect_accounting_tags(rows: list[PerformanceRow]) -> tuple[str, ...]:
    return tuple(
        sorted(
            {
                str(tag)
                for row in rows
                for tag in row.accounting_tags
            }
        )
    )


def _sum_optional_row_field(rows: list[PerformanceRow], field_name: str) -> int | None:
    values = [
        int(value)
        for row in rows
        if (value := getattr(row, field_name)) is not None
    ]
    return None if not values else sum(values)


def _sum_optional_delta_field(rows: list[dict[str, Any]], field_name: str) -> int | None:
    values = [
        int(value)
        for row in rows
        if (value := row.get(field_name)) is not None
    ]
    return None if not values else sum(values)


def _median_quality_delta(rows: list[PerformanceRow]) -> float | None:
    values = [
        float(value)
        for row in rows
        if (value := row.quality_guard.quality_delta_vs_baseline) is not None
    ]
    return None if not values else float(median(values))


def _median_rank_delta(rows: list[PerformanceRow]) -> float | None:
    values = [
        float(value)
        for row in rows
        if (value := row.quality_guard.median_rank_delta_vs_baseline) is not None
    ]
    return None if not values else float(median(values))


def _quality_regressed(row: PerformanceRow) -> bool:
    status = row.quality_guard.status.lower()
    if status in QUALITY_FAIL_STATUSES:
        return True
    if row.quality_guard.quality_delta_vs_baseline is not None and row.quality_guard.quality_delta_vs_baseline < 0:
        return True
    if row.quality_guard.median_rank_delta_vs_baseline is not None and row.quality_guard.median_rank_delta_vs_baseline > 0:
        return True
    return False


def _trust_regressed(row: PerformanceRow) -> bool:
    status = row.trust_guard.status.lower()
    observed_state = (row.trust_guard.observed_state or "").lower()
    return status in TRUST_FAIL_STATUSES or observed_state in {"regressed", "worse", "downgraded"}


def _verdict_for_delta(
    *,
    wall_clock_delta_pct: float | None,
    evals_per_second_delta_pct: float | None,
    quality_regression_count: int,
    trust_regression_count: int,
    failure_count_delta: int,
) -> str:
    has_guardrail_regression = (
        quality_regression_count > 0 or trust_regression_count > 0 or failure_count_delta > 0
    )
    faster = (
        (wall_clock_delta_pct is not None and wall_clock_delta_pct < 0)
        or (evals_per_second_delta_pct is not None and evals_per_second_delta_pct > 0)
    )
    slower = (
        (wall_clock_delta_pct is not None and wall_clock_delta_pct > 0)
        or (evals_per_second_delta_pct is not None and evals_per_second_delta_pct < 0)
    )
    if has_guardrail_regression and faster:
        return "faster-with-guardrail-regressions"
    if has_guardrail_regression:
        return "guardrail-regressed"
    if faster:
        return "faster-no-regression"
    if slower:
        return "slower-no-regression"
    return "flat-or-incomplete"
