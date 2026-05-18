"""Performance baseline workflow for compare-grade run artifacts."""

from __future__ import annotations

from collections import defaultdict
from datetime import datetime, timezone
import json
from pathlib import Path
import re
from typing import Any, Iterable

from evonn_compare.output_quality import OutputQualityRecord, inspect_paths
from evonn_shared.manifests import write_json

DEFAULT_REQUIRED_BUDGETS = (64, 256, 1000)
DECISION_GRADE_LANE_STATES = {"trusted-core", "trusted-extended"}
CANONICAL_SYSTEMS = ("prism", "topograph", "stratograph", "primordia", "contenders")
MIN_CLAIM_WALL_CLOCK_SECONDS = 0.1


def build_performance_baseline(
    *,
    inputs: Iterable[Path],
    output_root: Path | None = None,
    label: str | None = None,
    required_budgets: tuple[int, ...] = DEFAULT_REQUIRED_BUDGETS,
    write_run_artifacts: bool = True,
) -> dict[str, Any]:
    """Build a performance-baseline bundle from compare-grade run directories."""

    records = inspect_paths(list(inputs), write_run_artifacts=write_run_artifacts)
    generated_at = datetime.now(timezone.utc).isoformat()

    run_rows = [_record_payload(record, required_budgets=required_budgets) for record in records]
    comparison_cohorts = _comparison_cohort_stats(run_rows)
    performance_series = _performance_series_stats(run_rows)
    systems = _aggregate_systems(
        run_rows,
        required_budgets=required_budgets,
        comparison_cohorts=comparison_cohorts,
        performance_series=performance_series,
    )
    code_versions = sorted({str(row["code_version"]) for row in run_rows if row.get("code_version")})
    code_version_tag = code_versions[0] if len(code_versions) == 1 else ("mixed" if code_versions else "unknown")
    bundle_root = (output_root or Path("performance_baselines")) / f"{_stamp(generated_at)}-{_slug(code_version_tag)}"
    bundle_root.mkdir(parents=True, exist_ok=True)

    overview = {
        "generated_at": generated_at,
        "label": label or "performance-baseline",
        "code_versions": code_versions,
        "code_version_tag": code_version_tag,
        "required_budgets": list(required_budgets),
        "run_count": len(run_rows),
        "accepted_run_count": sum(1 for row in run_rows if row["counts_for_baseline"]),
        "comparison_cohorts": comparison_cohorts,
        "performance_series": performance_series,
        "systems": systems,
    }

    json_path = bundle_root / "performance_baseline.json"
    md_path = bundle_root / "performance_baseline.md"
    jsonl_path = bundle_root / "run_records.jsonl"

    write_json(json_path, {**overview, "runs": run_rows})
    md_path.write_text(_render_markdown(overview, run_rows), encoding="utf-8")
    jsonl_path.write_text("".join(json.dumps(row) + "\n" for row in run_rows), encoding="utf-8")

    return {
        "bundle_root": str(bundle_root.resolve()),
        "json": str(json_path.resolve()),
        "markdown": str(md_path.resolve()),
        "jsonl": str(jsonl_path.resolve()),
        "run_count": len(run_rows),
        "accepted_run_count": overview["accepted_run_count"],
        "system_count": len(systems),
    }


def _record_payload(record: OutputQualityRecord, *, required_budgets: tuple[int, ...]) -> dict[str, Any]:
    manifest = json.loads(record.manifest_path.read_text(encoding="utf-8"))
    summary = json.loads(record.summary_path.read_text(encoding="utf-8")) if record.summary_path is not None else {}
    budget = _as_int(((manifest.get("budget") or {}).get("evaluation_count")))
    pack_name = str(manifest.get("pack_name") or manifest.get("benchmark_pack_id") or "unknown")
    fair_matrix_context = _fair_matrix_context(record=record, manifest=manifest)
    fairness = manifest.get("fairness") or {}
    code_version = fairness.get("code_version")
    seed = _as_int(fairness.get("seed") if fairness else manifest.get("seed"))
    lane_operating_state = _first_text(
        fair_matrix_context.get("lane_operating_state"),
        summary.get("lane_operating_state"),
        (summary.get("fairness") or {}).get("lane_operating_state") if isinstance(summary.get("fairness"), dict) else None,
        (summary.get("fairness_metadata") or {}).get("lane_operating_state") if isinstance(summary.get("fairness_metadata"), dict) else None,
    )
    fairness_ok = all(
        fairness.get(field) not in (None, "")
        for field in ("benchmark_pack_id", "seed", "evaluation_count", "budget_policy_name", "data_signature", "code_version")
    )
    quality_ok = record.quality_level in {"L3", "L4"}
    counts = fairness_ok and quality_ok and record.measurement_state == "measurable" and budget in required_budgets
    benchmark_throughput = _ratio(record.completed_benchmark_count, record.performance.wall_clock_seconds)
    failure_adjusted_throughput = _ratio(
        record.completed_benchmark_count / max(record.benchmark_count, 1),
        record.performance.wall_clock_seconds,
    )
    pack_family = _canonical_pack_name(str(fairness.get("benchmark_pack_id") or fair_matrix_context.get("pack_name") or pack_name))
    comparison_key = _comparison_key(
        pack_name=pack_family,
        seed=seed,
        code_version=code_version,
    )
    system_performance_key = _system_performance_key(
        pack_name=pack_family,
        seed=seed,
        system=record.system,
        backend=record.runtime.runtime_backend,
        hardware_class=record.runtime.hardware_class,
        code_version=code_version,
    )
    return {
        "system": record.system,
        "run_id": record.run_id,
        "run_dir": str(record.run_dir),
        "pack_name": pack_name,
        "pack_family": pack_family,
        "budget": budget,
        "seed": seed,
        "code_version": code_version,
        "quality_level": record.quality_level,
        "measurement_state": record.measurement_state,
        "lane_operating_state": lane_operating_state,
        "counts_for_baseline": counts,
        "runtime": {
            "backend": record.runtime.runtime_backend,
            "backend_requested": record.runtime.runtime_backend_requested,
            "device_name": record.runtime.device_name,
            "hardware_class": record.runtime.hardware_class,
            "framework": record.runtime.framework,
            "framework_version": record.runtime.framework_version,
            "worker_count": record.runtime.worker_count,
        },
        "performance": {
            "wall_clock_seconds": record.performance.wall_clock_seconds,
            "evals_per_second": record.performance.evals_per_second,
            "quality_per_second": record.performance.quality_per_second,
            "cache_reuse_rate": record.performance.cache_reuse_rate,
            "peak_memory_mb": record.performance.peak_memory_mb,
            "benchmark_throughput": benchmark_throughput,
            "failure_adjusted_throughput": failure_adjusted_throughput,
        },
        "benchmark_count": record.benchmark_count,
        "completed_benchmark_count": record.completed_benchmark_count,
        "failed_benchmark_count": record.failed_benchmark_count,
        "comparison_key": comparison_key,
        "system_performance_key": system_performance_key,
        "exclusion_reasons": [],
    }


def _aggregate_systems(
    run_rows: list[dict[str, Any]],
    *,
    required_budgets: tuple[int, ...],
    comparison_cohorts: dict[str, dict[str, Any]],
    performance_series: dict[str, dict[str, Any]],
) -> list[dict[str, Any]]:
    grouped: dict[str, list[dict[str, Any]]] = defaultdict(list)
    for row in run_rows:
        grouped[str(row["system"])].append(row)

    systems = []
    for system, rows in sorted(grouped.items()):
        for row in rows:
            row["exclusion_reasons"] = _exclusion_reasons(
                row,
                required_budgets=required_budgets,
                comparison_cohorts=comparison_cohorts,
            )
            row["counts_for_baseline"] = not row["exclusion_reasons"]
        accepted = [row for row in rows if row["counts_for_baseline"]]
        budgets_present = sorted({int(row["budget"]) for row in accepted if row.get("budget") is not None})
        missing_budgets = [budget for budget in required_budgets if budget not in budgets_present]
        claim_ready = False
        selected_comparison_cohort = None
        selected_performance_key = None
        performance_claim_warnings = []
        candidate_series = _candidate_performance_series(
            rows=accepted,
            performance_series=performance_series,
            required_budgets=required_budgets,
        )
        if accepted:
            if len(candidate_series) > 1:
                performance_claim_warnings.append("multiple-performance-series")
            selected = _select_performance_series(candidate_series)
            selected_performance_key = selected["system_performance_key"] if selected else None
            selected_rows = [row for row in accepted if row.get("system_performance_key") == selected_performance_key]
            selected_comparison_keys = sorted({str(row["comparison_key"]) for row in selected_rows})
            if len(selected_comparison_keys) > 1:
                performance_claim_warnings.append("multiple-comparison-cohorts-for-selected-series")
            selected_comparison_cohort = selected_comparison_keys[0] if len(selected_comparison_keys) == 1 else None
            stats = comparison_cohorts.get(selected_comparison_cohort or "", {})
            claim_ready = (
                selected is not None
                and selected["has_required_budgets"]
                and selected_comparison_cohort is not None
                and _comparison_ready(stats, required_budgets=required_budgets)
            )
        systems.append(
            {
                "system": system,
                "run_count": len(rows),
                "accepted_run_count": len(accepted),
                "budgets_present": budgets_present,
                "missing_budgets": missing_budgets,
                "performance_claim_ready": claim_ready,
                "selected_comparison_cohort": selected_comparison_cohort,
                "selected_system_performance_key": selected_performance_key,
                "performance_series": candidate_series,
                "performance_claim_warnings": performance_claim_warnings,
                "backend_labels": sorted({row["runtime"].get("backend") or "unknown" for row in accepted}),
                "hardware_labels": sorted({row["runtime"].get("hardware_class") or row["runtime"].get("device_name") or "unknown" for row in accepted}),
                "median_wall_clock_seconds": _median([row["performance"].get("wall_clock_seconds") for row in accepted]),
                "median_evals_per_second": _median([row["performance"].get("evals_per_second") for row in accepted]),
                "median_quality_per_second": _median([row["performance"].get("quality_per_second") for row in accepted]),
                "median_benchmark_throughput": _median([row["performance"].get("benchmark_throughput") for row in accepted]),
                "median_failure_adjusted_throughput": _median([row["performance"].get("failure_adjusted_throughput") for row in accepted]),
                "median_cache_reuse_rate": _median([row["performance"].get("cache_reuse_rate") for row in accepted]),
                "max_peak_memory_mb": _max([row["performance"].get("peak_memory_mb") for row in accepted]),
                "excluded_runs": [
                    {"run_id": row["run_id"], "budget": row["budget"], "reasons": row["exclusion_reasons"]}
                    for row in rows
                    if not row["counts_for_baseline"]
                ],
            }
        )
    return systems


def _candidate_performance_series(
    *,
    rows: list[dict[str, Any]],
    performance_series: dict[str, dict[str, Any]],
    required_budgets: tuple[int, ...],
) -> list[dict[str, Any]]:
    rows_by_series: dict[str, list[dict[str, Any]]] = defaultdict(list)
    for row in rows:
        rows_by_series[str(row["system_performance_key"])].append(row)
    candidates = []
    for series_key, series_rows in sorted(rows_by_series.items()):
        series = performance_series.get(series_key, {})
        budgets = sorted({int(row["budget"]) for row in series_rows if row.get("budget") is not None})
        missing_budgets = [budget for budget in required_budgets if budget not in budgets]
        candidates.append(
            {
                "system_performance_key": series_key,
                "backend": series.get("backend"),
                "hardware_class": series.get("hardware_class"),
                "budgets": budgets,
                "missing_budgets": missing_budgets,
                "has_required_budgets": not missing_budgets,
                "run_ids": sorted({str(row["run_id"]) for row in series_rows}),
            }
        )
    return sorted(candidates, key=lambda row: (not row["has_required_budgets"], -len(row["budgets"]), row["system_performance_key"]))


def _select_performance_series(candidate_series: list[dict[str, Any]]) -> dict[str, Any] | None:
    if not candidate_series:
        return None
    return min(candidate_series, key=lambda row: (not row["has_required_budgets"], -len(row["budgets"]), row["system_performance_key"]))


def _comparison_cohort_stats(run_rows: list[dict[str, Any]]) -> dict[str, dict[str, Any]]:
    grouped: dict[str, list[dict[str, Any]]] = defaultdict(list)
    for row in run_rows:
        grouped[str(row["comparison_key"])].append(row)
    payload = {}
    for comparison_key, rows in grouped.items():
        lane_states = sorted({str(row.get("lane_operating_state") or "unknown") for row in rows})
        backend_labels = sorted({str(row["runtime"].get("backend") or "unknown") for row in rows})
        hardware_labels = sorted(
            {str(row["runtime"].get("hardware_class") or row["runtime"].get("device_name") or "unknown") for row in rows}
        )
        systems_by_budget: dict[str, list[str]] = {}
        lane_states_by_budget: dict[str, list[str]] = {}
        for budget in sorted({int(row["budget"]) for row in rows if row.get("budget") is not None}):
            budget_rows = [row for row in rows if row.get("budget") == budget]
            systems_by_budget[str(budget)] = sorted({str(row["system"]) for row in budget_rows})
            lane_states_by_budget[str(budget)] = sorted({str(row.get("lane_operating_state") or "unknown") for row in budget_rows})
        payload[comparison_key] = {
            "comparison_key": comparison_key,
            "pack_name": rows[0].get("pack_name"),
            "pack_family": rows[0].get("pack_family"),
            "seed": rows[0].get("seed"),
            "code_version": rows[0].get("code_version"),
            "backend": backend_labels[0] if len(backend_labels) == 1 else "mixed",
            "backend_labels": backend_labels,
            "hardware_class": hardware_labels[0] if len(hardware_labels) == 1 else "mixed",
            "hardware_labels": hardware_labels,
            "systems": sorted({str(row["system"]) for row in rows}),
            "budgets": sorted({int(row["budget"]) for row in rows if row.get("budget") is not None}),
            "systems_by_budget": systems_by_budget,
            "lane_states": lane_states,
            "lane_states_by_budget": lane_states_by_budget,
            "lane_operating_state": lane_states[0] if len(lane_states) == 1 else "mixed",
        }
    return payload


def _performance_series_stats(run_rows: list[dict[str, Any]]) -> dict[str, dict[str, Any]]:
    grouped: dict[str, list[dict[str, Any]]] = defaultdict(list)
    for row in run_rows:
        grouped[str(row["system_performance_key"])].append(row)
    payload = {}
    for series_key, rows in grouped.items():
        payload[series_key] = {
            "system_performance_key": series_key,
            "comparison_keys": sorted({str(row["comparison_key"]) for row in rows}),
            "system": rows[0].get("system"),
            "pack_family": rows[0].get("pack_family"),
            "seed": rows[0].get("seed"),
            "code_version": rows[0].get("code_version"),
            "backend": rows[0]["runtime"].get("backend"),
            "hardware_class": rows[0]["runtime"].get("hardware_class"),
            "budgets": sorted({int(row["budget"]) for row in rows if row.get("budget") is not None}),
            "run_ids": sorted({str(row["run_id"]) for row in rows}),
        }
    return payload


def _render_markdown(overview: dict[str, Any], run_rows: list[dict[str, Any]]) -> str:
    lines = [
        "# Performance Baseline",
        "",
        f"- label: `{overview['label']}`",
        f"- generated_at: `{overview['generated_at']}`",
        f"- code_version_tag: `{overview['code_version_tag']}`",
        f"- code_versions: `{', '.join(overview['code_versions']) or 'unknown'}`",
        f"- required_budgets: `{', '.join(str(v) for v in overview['required_budgets'])}`",
        f"- accepted_runs: `{overview['accepted_run_count']}/{overview['run_count']}`",
        "",
        "## Comparison Cohorts",
        "",
        "| Comparison Key | Pack | Seed | Code | Lane State | Backends | Hardware | Systems | Budgets | Complete Budgets |",
        "| --- | --- | ---: | --- | --- | --- | --- | --- | --- | --- |",
    ]
    for cohort in overview["comparison_cohorts"].values():
        lines.append(
            "| {key} | `{pack}` | {seed} | `{code}` | `{lane}` | `{backends}` | `{hardware}` | `{systems}` | `{budgets}` | `{complete}` |".format(
                key=cohort["comparison_key"],
                pack=cohort.get("pack_family") or cohort["pack_name"],
                seed=cohort["seed"] if cohort["seed"] is not None else 0,
                code=cohort["code_version"] or "unknown",
                lane=cohort["lane_operating_state"],
                backends=", ".join(cohort.get("backend_labels") or [cohort.get("backend") or "unknown"]),
                hardware=", ".join(cohort.get("hardware_labels") or [cohort.get("hardware_class") or "unknown"]),
                systems=", ".join(cohort["systems"]),
                budgets=", ".join(str(v) for v in cohort["budgets"]),
                complete=", ".join(
                    budget
                    for budget, systems in sorted((cohort.get("systems_by_budget") or {}).items(), key=lambda item: int(item[0]))
                    if set(systems) == set(CANONICAL_SYSTEMS)
                ) or "none",
            )
        )
    lines.extend([
        "",
        "## System Summary",
        "",
        "| System | Accepted | Budgets | Claim Ready | Comparison | Performance Series | Warnings | Wall s | Eval/s | Quality/s | Bench/s | Failure-Adj Bench/s | Cache Reuse | Backends | Hardware |",
        "| --- | ---: | --- | --- | --- | --- | --- | ---: | ---: | ---: | ---: | ---: | ---: | --- | --- |",
    ])
    for row in overview["systems"]:
        lines.append(
            "| {system} | {accepted}/{total} | `{budgets}` | `{ready}` | `{comparison}` | `{series}` | `{warnings}` | {wall} | {eps} | {qps} | {bps} | {fbps} | {cache} | `{backends}` | `{hardware}` |".format(
                system=row["system"],
                accepted=row["accepted_run_count"],
                total=row["run_count"],
                budgets=", ".join(str(v) for v in row["budgets_present"]) or "none",
                ready="yes" if row["performance_claim_ready"] else "no",
                comparison=row["selected_comparison_cohort"] or "none",
                series=row["selected_system_performance_key"] or "none",
                warnings=", ".join(row.get("performance_claim_warnings") or []) or "none",
                wall=_fmt(row["median_wall_clock_seconds"]),
                eps=_fmt(row["median_evals_per_second"]),
                qps=_fmt(row["median_quality_per_second"]),
                bps=_fmt(row["median_benchmark_throughput"]),
                fbps=_fmt(row["median_failure_adjusted_throughput"]),
                cache=_fmt(row["median_cache_reuse_rate"]),
                backends=", ".join(row["backend_labels"]) or "unknown",
                hardware=", ".join(row["hardware_labels"]) or "unknown",
            )
        )
    lines.extend([
        "",
        "## Performance Series",
        "",
        "| Series | System | Backend | Hardware | Budgets | Comparison Keys |",
        "| --- | --- | --- | --- | --- | --- |",
    ])
    for series in overview["performance_series"].values():
        lines.append(
            "| {key} | `{system}` | `{backend}` | `{hardware}` | `{budgets}` | `{comparisons}` |".format(
                key=series["system_performance_key"],
                system=series["system"] or "unknown",
                backend=series["backend"] or "unknown",
                hardware=series["hardware_class"] or "unknown",
                budgets=", ".join(str(v) for v in series["budgets"]) or "none",
                comparisons=", ".join(series["comparison_keys"]) or "none",
            )
        )
    lines.extend(["", "## Excluded Runs", "", "| System | Run ID | Budget | Reasons |", "| --- | --- | ---: | --- |"])
    excluded_any = False
    for row in run_rows:
        if row["counts_for_baseline"]:
            continue
        excluded_any = True
        lines.append("| {system} | `{run_id}` | {budget} | `{reasons}` |".format(system=row["system"], run_id=row["run_id"], budget=row["budget"] or 0, reasons=", ".join(row["exclusion_reasons"]) or "none"))
    if not excluded_any:
        lines.append("| _none_ |  |  |  |")
    return "\n".join(lines) + "\n"


def _exclusion_reasons(
    row: dict[str, Any],
    *,
    required_budgets: tuple[int, ...],
    comparison_cohorts: dict[str, dict[str, Any]],
) -> list[str]:
    reasons = []
    if row["quality_level"] not in {"L3", "L4"}:
        reasons.append(f"quality={row['quality_level']}")
    if row["measurement_state"] != "measurable":
        reasons.append(f"measurement={row['measurement_state']}")
    if row.get("budget") not in required_budgets:
        reasons.append(f"budget-not-in-baseline-set:{row.get('budget')}")
    if row.get("seed") is None:
        reasons.append("seed-missing")
    if not row.get("code_version"):
        reasons.append("code-version-missing")
    if not row["runtime"].get("backend"):
        reasons.append("backend-missing")
    if not row["runtime"].get("hardware_class"):
        reasons.append("hardware-class-missing")
    wall_clock_seconds = row["performance"].get("wall_clock_seconds")
    if (
        wall_clock_seconds is not None
        and float(wall_clock_seconds) < MIN_CLAIM_WALL_CLOCK_SECONDS
        and (row.get("budget") or 0) > 1
    ):
        reasons.append(f"wall-clock-implausible:{wall_clock_seconds}")
    lane_state = row.get("lane_operating_state")
    if lane_state not in DECISION_GRADE_LANE_STATES:
        reasons.append(f"lane-state={lane_state or 'missing'}")
    cohort = comparison_cohorts.get(str(row.get("comparison_key")))
    if not cohort:
        reasons.append("cohort-missing")
    else:
        budget_key = str(row.get("budget"))
        systems_for_budget = set((cohort.get("systems_by_budget") or {}).get(budget_key, []))
        lane_states_for_budget = set((cohort.get("lane_states_by_budget") or {}).get(budget_key, []))
        if systems_for_budget != set(CANONICAL_SYSTEMS):
            reasons.append("incomplete-system-cohort")
        if len(lane_states_for_budget) > 1:
            reasons.append("mixed-lane-states")
    return reasons


def _comparison_ready(cohort: dict[str, Any], *, required_budgets: tuple[int, ...]) -> bool:
    systems_by_budget = cohort.get("systems_by_budget") or {}
    lane_states_by_budget = cohort.get("lane_states_by_budget") or {}
    for budget in required_budgets:
        budget_key = str(budget)
        if set(systems_by_budget.get(budget_key, [])) != set(CANONICAL_SYSTEMS):
            return False
        lane_states = set(lane_states_by_budget.get(budget_key, []))
        if not lane_states or not lane_states.issubset(DECISION_GRADE_LANE_STATES):
            return False
    return True


def _comparison_key(*, pack_name: str, seed: int | None, code_version: str | None) -> str:
    return "|".join([
        pack_name or "unknown-pack",
        str(seed if seed is not None else "unknown-seed"),
        code_version or "unknown-code-version",
    ])


def _system_performance_key(
    *,
    pack_name: str,
    seed: int | None,
    system: str,
    backend: str | None,
    hardware_class: str | None,
    code_version: str | None,
) -> str:
    return "|".join([
        pack_name or "unknown-pack",
        str(seed if seed is not None else "unknown-seed"),
        system or "unknown-system",
        backend or "unknown-backend",
        hardware_class or "unknown-hardware",
        code_version or "unknown-code-version",
    ])


def _canonical_pack_name(pack_name: str) -> str:
    return re.sub(r"_eval\d+$", "", pack_name or "unknown-pack") or "unknown-pack"


def _fair_matrix_context(*, record: OutputQualityRecord, manifest: dict[str, Any]) -> dict[str, Any]:
    workspace_root = _workspace_root_from_run_dir(record.run_dir)
    if workspace_root is None:
        return {}
    report_dir = workspace_root / "reports" / record.run_id
    summary_path = report_dir / "fair_matrix_summary.json"
    trends_json_path = report_dir / "fair_matrix_trends.json"
    trends_jsonl_path = report_dir / "fair_matrix_trends.jsonl"

    summary_payload = _read_json_if_exists(summary_path)
    summary_mapping = summary_payload if isinstance(summary_payload, dict) else {}
    trend_rows = []
    if isinstance(summary_mapping.get("trend_rows"), list):
        trend_rows = list(summary_mapping.get("trend_rows") or [])
    elif trends_json_path.exists():
        trend_rows = _read_json_if_exists(trends_json_path)
    elif trends_jsonl_path.exists():
        trend_rows = [json.loads(line) for line in trends_jsonl_path.read_text(encoding="utf-8").splitlines() if line.strip()]

    fairness = manifest.get("fairness") or {}
    budget = _as_int((manifest.get("budget") or {}).get("evaluation_count"))
    seed = _as_int(fairness.get("seed") if fairness else manifest.get("seed"))
    matches = [
        row for row in trend_rows
        if isinstance(row, dict)
        and str(row.get("system") or row.get("engine") or "") == record.system
        and str(row.get("run_id") or "") == record.run_id
        and _as_int(row.get("budget")) == budget
        and _as_int(row.get("seed")) == seed
    ]
    lane_state = None
    pack_name = None
    if matches:
        lane_states = sorted({str(row.get("lane_operating_state") or (row.get("fairness_metadata") or {}).get("lane_operating_state") or "") for row in matches if str(row.get("lane_operating_state") or (row.get("fairness_metadata") or {}).get("lane_operating_state") or "").strip()})
        lane_state = lane_states[0] if len(lane_states) == 1 else None
        pack_names = sorted({str(row.get("pack_name") or "") for row in matches if str(row.get("pack_name") or "").strip()})
        pack_name = pack_names[0] if len(pack_names) == 1 else None
    if not matches and lane_state is None and isinstance(summary_mapping.get("lane"), dict):
        lane_state = _first_text((summary_mapping.get("lane") or {}).get("operating_state"))
    return {
        "lane_operating_state": lane_state,
        "pack_name": pack_name,
    }


def _workspace_root_from_run_dir(run_dir: Path) -> Path | None:
    current = run_dir.resolve()
    for parent in current.parents:
        if parent.name == "runs":
            return parent.parent.resolve()
    return None


def _read_json_if_exists(path: Path) -> Any:
    if not path.exists():
        return {}
    return json.loads(path.read_text(encoding="utf-8"))


def _first_text(*values: Any) -> str | None:
    for value in values:
        if value is None:
            continue
        text = str(value).strip()
        if text:
            return text
    return None


def _median(values: list[Any]) -> float | None:
    nums = sorted(float(value) for value in values if value is not None)
    if not nums:
        return None
    mid = len(nums) // 2
    if len(nums) % 2:
        return nums[mid]
    return (nums[mid - 1] + nums[mid]) / 2.0


def _max(values: list[Any]) -> float | None:
    nums = [float(value) for value in values if value is not None]
    return max(nums) if nums else None


def _ratio(numerator: float | int, denominator: float | None) -> float | None:
    if denominator is None or denominator <= 0:
        return None
    return float(numerator) / float(denominator)


def _fmt(value: float | None) -> str:
    return "-" if value is None else f"{value:.3f}"


def _stamp(iso_ts: str) -> str:
    return iso_ts.replace(":", "").replace("+00:00", "Z").replace("-", "")[:15]


def _slug(value: str) -> str:
    return "".join(ch if ch.isalnum() or ch in {"-", "_"} else "-" for ch in value)[:64] or "unknown"


def _as_int(value: Any) -> int | None:
    try:
        return int(value) if value is not None else None
    except (TypeError, ValueError):
        return None
