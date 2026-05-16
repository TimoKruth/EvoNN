"""Durable promoted-evidence registry for fair-matrix runs."""

from __future__ import annotations

from collections import Counter, defaultdict
from datetime import datetime, timezone
import hashlib
import json
from pathlib import Path
import platform
import re
import shutil
import socket
import subprocess
from statistics import mean
from typing import Any

from evonn_compare.reporting.fair_matrix_dashboard import ALL_SYSTEMS, discover_fair_matrix_summaries
from evonn_compare.reporting.fair_matrix_stats import build_multi_seed_statistics

DECISION_READY_STATES = {"contract-fair", "portable-contract-fair", "trusted-core", "trusted-extended"}


def promote_evidence(
    *,
    inputs: list[Path],
    registry: Path,
    label: str | None = None,
    min_seeds: int = 2,
    copy_artifacts: bool = True,
) -> dict[str, Any]:
    """Promote fair-matrix summaries into a small durable evidence registry."""

    summary_paths = discover_fair_matrix_summaries(inputs)
    if not summary_paths:
        raise ValueError("no fair-matrix summaries found in supplied inputs")

    registry_path = registry.resolve()
    registry_path.mkdir(parents=True, exist_ok=True)
    promoted_at = datetime.now(timezone.utc).isoformat()
    existing = _load_index(registry_path)
    promoted_records: list[dict[str, Any]] = []

    for summary_path in summary_paths:
        payload = json.loads(summary_path.read_text(encoding="utf-8"))
        digest = _sha256(summary_path)
        summary_label = label or _default_label(summary_path, payload)
        record_id = f"{_slugify(summary_label)}:{digest[:16]}"
        destination_summary_path: Path | None = None
        if copy_artifacts:
            destination_summary_path = _copy_summary_artifacts(
                summary_path=summary_path,
                registry=registry_path,
                label=summary_label,
                digest=digest,
            )

        record = _record_from_summary(
            record_id=record_id,
            label=summary_label,
            promoted_at=promoted_at,
            source_summary_path=summary_path.resolve(),
            registry_summary_path=destination_summary_path,
            summary_sha256=digest,
            payload=payload,
        )
        existing[record_id] = record
        promoted_records.append(record)

    records = sorted(existing.values(), key=lambda item: (str(item["label"]), str(item["pack_name"]), int(item["budget"]), int(item["seed"]), str(item["record_id"])))
    _write_index(registry_path, records)
    report = build_evidence_report(registry=registry_path, min_seeds=min_seeds)
    return {
        "registry": str(registry_path),
        "index": str(registry_path / "index.jsonl"),
        "report_json": str(registry_path / "evidence_report.json"),
        "report_md": str(registry_path / "evidence_report.md"),
        "promoted_count": len(promoted_records),
        "registry_count": len(records),
        "groups": report["groups"],
    }


def build_evidence_report(*, registry: Path, min_seeds: int = 2) -> dict[str, Any]:
    """Build a report over promoted evidence rows."""

    registry_path = registry.resolve()
    records = list(_load_index(registry_path).values())
    summary_payloads = [_load_record_summary(record) for record in records]
    transfer_cases = _load_transfer_case_payloads(records)
    trend_rows = [row for payload in summary_payloads for row in payload.get("trend_rows") or []]
    groups = _decision_groups(records=records, trend_rows=trend_rows, min_seeds=min_seeds)
    payload = {
        "generated_at": datetime.now(timezone.utc).isoformat(),
        "registry": str(registry_path),
        "record_count": len(records),
        "summary_count": len(summary_payloads),
        "min_seeds": min_seeds,
        "groups": groups,
        "before_after_comparisons": _before_after_comparisons(groups),
        "engine_roles": _engine_roles(groups, trend_rows),
        "lm_flatline_diagnostics": _lm_flatline_diagnostics(trend_rows),
        "transfer_evidence": _transfer_evidence(summary_payloads, trend_rows, transfer_cases),
        "quality_diversity_evidence": _quality_diversity_evidence(summary_payloads, trend_rows),
        "artifact_validation": validate_registry(registry=registry_path),
    }
    registry_path.mkdir(parents=True, exist_ok=True)
    (registry_path / "evidence_report.json").write_text(json.dumps(payload, indent=2), encoding="utf-8")
    (registry_path / "evidence_report.md").write_text(_render_report_markdown(payload), encoding="utf-8")
    return payload


def validate_registry(*, registry: Path) -> dict[str, Any]:
    """Validate registry rows without requiring copied artifacts to exist."""

    return _validate_registry(registry=registry, require_artifacts=False)


def validate_registry_artifacts(*, registry: Path) -> dict[str, Any]:
    """Validate registry rows and artifact pointers."""

    return _validate_registry(registry=registry, require_artifacts=True)


def discover_registry_summaries(inputs: list[Path] | None) -> list[Path]:
    """Resolve copied or source fair-matrix summaries from evidence registries."""

    roots = inputs or [Path("evidence")]
    found: dict[Path, None] = {}
    for root in roots:
        index_path = root if root.name == "index.jsonl" else root / "index.jsonl"
        if not index_path.exists():
            continue
        for line in index_path.read_text(encoding="utf-8").splitlines():
            if not line.strip():
                continue
            record = json.loads(line)
            path_value = record.get("registry_summary_path") or record.get("source_summary_path")
            if not path_value:
                continue
            path = Path(str(path_value))
            if path.exists():
                found[path.resolve()] = None
    return sorted(found)


def _validate_registry(*, registry: Path, require_artifacts: bool) -> dict[str, Any]:
    """Validate registry rows and optionally check artifact pointers."""

    registry_path = registry.resolve()
    issues: list[str] = []
    warnings: list[str] = []
    records = list(_load_index(registry_path).values())
    seen: set[str] = set()
    for record in records:
        record_id = str(record.get("record_id") or "")
        if not record_id:
            issues.append("record without record_id")
        if record_id in seen:
            issues.append(f"duplicate record_id: {record_id}")
        seen.add(record_id)
        for field in ("label", "pack_name", "budget", "seed", "systems", "summary_sha256"):
            if record.get(field) in (None, "", []):
                issues.append(f"{record_id or '<unknown>'} missing {field}")
        artifact_paths = [
            value
            for value in (record.get("registry_summary_path"), record.get("source_summary_path"))
            if value
        ]
        existing_paths = [Path(str(value)) for value in artifact_paths if Path(str(value)).exists()]
        if require_artifacts and not existing_paths:
            issues.append(f"{record_id or '<unknown>'} has no readable summary artifact")
        elif not existing_paths:
            warnings.append(f"{record_id or '<unknown>'} has no currently readable summary artifact")
        for path in existing_paths:
            current_sha = _sha256(path)
            expected_sha = str(record.get("summary_sha256") or "")
            if path == Path(str(record.get("registry_summary_path") or "")) and expected_sha and current_sha != expected_sha:
                issues.append(f"{record_id or '<unknown>'} copied summary checksum drifted: {path}")
            elif path == Path(str(record.get("source_summary_path") or "")) and expected_sha and current_sha != expected_sha:
                warnings.append(f"{record_id or '<unknown>'} source summary checksum drifted: {path}")
    return {"ok": not issues, "record_count": len(records), "issues": issues, "warnings": warnings}


def _record_from_summary(
    *,
    record_id: str,
    label: str,
    promoted_at: str,
    source_summary_path: Path,
    registry_summary_path: Path | None,
    summary_sha256: str,
    payload: dict[str, Any],
) -> dict[str, Any]:
    lane = dict(payload.get("lane") or {})
    trend_rows = list(payload.get("trend_rows") or [])
    budget = _summary_budget(payload)
    seed = _summary_seed(payload)
    operating_state = str(lane.get("operating_state") or _first_row_value(trend_rows, "lane_operating_state") or "reference-only")
    repeatability_ready = bool(lane.get("repeatability_ready", _first_row_value(trend_rows, "lane_repeatability_ready", False)))
    budget_accounting_ok = bool(lane.get("budget_accounting_ok", _first_row_value(trend_rows, "lane_budget_accounting_ok", False)))
    row_count = len(trend_rows)
    ok_rows = sum(1 for row in trend_rows if row.get("outcome_status") == "ok")
    systems = sorted(str(value) for value in payload.get("systems") or {row.get("system") for row in trend_rows if row.get("system")})
    task_kinds = sorted({str(row.get("task_kind") or "unknown") for row in trend_rows})
    families = sorted({str(row.get("benchmark_family") or "unknown") for row in trend_rows})
    score_by_system = _score_systems(trend_rows, tuple(systems))
    source_context = _source_context(payload=payload, trend_rows=trend_rows, summary_path=source_summary_path)
    contender_floor = _contender_floor_context(source_summary_path)
    output_quality = _output_quality_context(source_summary_path)
    return {
        "record_id": record_id,
        "label": label,
        "comparison_cohort": "promoted-evidence",
        "promoted_at": promoted_at,
        "source_summary_path": str(source_summary_path),
        "registry_summary_path": str(registry_summary_path) if registry_summary_path is not None else None,
        "summary_sha256": summary_sha256,
        "pack_name": str(payload.get("pack_name") or _first_row_value(trend_rows, "pack_name") or "unknown"),
        "budget": budget,
        "seed": seed,
        "systems": systems,
        "task_kinds": task_kinds,
        "benchmark_families": families,
        "trend_row_count": row_count,
        "ok_trend_row_count": ok_rows,
        "lane_operating_state": operating_state,
        "repeatability_ready": repeatability_ready,
        "budget_accounting_ok": budget_accounting_ok,
        "decision_eligible": operating_state in DECISION_READY_STATES and budget_accounting_ok and row_count > 0,
        "score_by_system": score_by_system,
        "git": source_context["git"],
        "runtime": source_context["runtime"],
        "contender_floor": contender_floor,
        "output_quality": output_quality,
    }


def _decision_groups(*, records: list[dict[str, Any]], trend_rows: list[dict[str, Any]], min_seeds: int) -> list[dict[str, Any]]:
    stats_by_key = {
        (str(group["comparison_label"]), str(group["pack_name"]), int(group["budget"])): group
        for group in build_multi_seed_statistics(trend_rows, systems=tuple(ALL_SYSTEMS))
    }
    grouped: dict[tuple[str, str, int], list[dict[str, Any]]] = defaultdict(list)
    for record in records:
        grouped[(str(record["label"]), str(record["pack_name"]), int(record["budget"]))].append(record)

    groups: list[dict[str, Any]] = []
    for (label, pack_name, budget), group_records in sorted(grouped.items()):
        seeds = sorted({int(record["seed"]) for record in group_records if record.get("seed") is not None})
        states = sorted({str(record.get("lane_operating_state") or "unknown") for record in group_records})
        blockers = _group_blockers(group_records, min_seeds=min_seeds)
        stats = stats_by_key.get((label, pack_name, budget))
        system_rows = list(stats.get("system_rows") or []) if stats else _system_rows_from_records(group_records)
        ranked = sorted(
            system_rows,
            key=lambda row: (float(row.get("mean_score") or 0.0), str(row.get("system"))),
            reverse=True,
        )
        margin = 0.0
        if len(ranked) >= 2:
            margin = float(ranked[0].get("mean_score") or 0.0) - float(ranked[1].get("mean_score") or 0.0)
        decision_label = _decision_label(blockers=blockers, seed_count=len(seeds), margin=margin)
        groups.append(
            {
                "label": label,
                "pack_name": pack_name,
                "budget": budget,
                "seed_count": len(seeds),
                "seeds": seeds,
                "record_count": len(group_records),
                "lane_states": states,
                "decision_label": decision_label,
                "decision_margin": round(margin, 6),
                "decision_blockers": blockers,
                "leader": ranked[0]["system"] if ranked else None,
                "system_rows": system_rows,
                "pairwise": list(stats.get("pairwise") or []) if stats else [],
                "last_promoted_at": max(str(record.get("promoted_at") or "") for record in group_records),
            }
        )
    return groups


def _before_after_comparisons(groups: list[dict[str, Any]]) -> list[dict[str, Any]]:
    grouped: dict[tuple[str, int], list[dict[str, Any]]] = defaultdict(list)
    for group in groups:
        grouped[(str(group["pack_name"]), int(group["budget"]))].append(group)

    comparisons: list[dict[str, Any]] = []
    for (pack_name, budget), rows in sorted(grouped.items()):
        ordered = sorted(rows, key=lambda row: (str(row.get("last_promoted_at") or ""), str(row["label"])))
        for before, after in zip(ordered, ordered[1:], strict=False):
            before_scores = _mean_scores_by_system(before)
            after_scores = _mean_scores_by_system(after)
            common_systems = sorted(set(before_scores) & set(after_scores))
            deltas = {system: round(after_scores[system] - before_scores[system], 6) for system in common_systems}
            aggregate_delta = round(mean(deltas.values()), 6) if deltas else 0.0
            best_system_delta = max(deltas.values(), default=0.0)
            worst_system_delta = min(deltas.values(), default=0.0)
            if before.get("decision_blockers") or after.get("decision_blockers"):
                decision = "inconclusive"
            elif best_system_delta >= 0.25:
                decision = "likely_gain"
            elif aggregate_delta <= -0.1 or worst_system_delta <= -0.25:
                decision = "regression"
            else:
                decision = "no_material_change"
            comparisons.append(
                {
                    "pack_name": pack_name,
                    "budget": budget,
                    "before_label": before["label"],
                    "after_label": after["label"],
                    "common_systems": common_systems,
                    "system_score_deltas": deltas,
                    "aggregate_score_delta": aggregate_delta,
                    "best_system_score_delta": round(best_system_delta, 6),
                    "worst_system_score_delta": round(worst_system_delta, 6),
                    "decision_label": decision,
                }
            )
    return comparisons


def _mean_scores_by_system(group: dict[str, Any]) -> dict[str, float]:
    return {
        str(row["system"]): float(row.get("mean_score") or 0.0)
        for row in group.get("system_rows") or []
    }


def _group_blockers(records: list[dict[str, Any]], *, min_seeds: int) -> list[str]:
    blockers = []
    seeds = {record.get("seed") for record in records if record.get("seed") is not None}
    if len(seeds) < min_seeds:
        blockers.append(f"needs at least {min_seeds} seeds for decision-grade evidence")
    if any(not record.get("decision_eligible") for record in records):
        blockers.append("one or more records are not decision eligible")
    if any(not record.get("budget_accounting_ok") for record in records):
        blockers.append("budget accounting is incomplete")
    if any(str(record.get("lane_operating_state")) not in DECISION_READY_STATES for record in records):
        blockers.append("lane state is not decision ready")
    return blockers


def _decision_label(*, blockers: list[str], seed_count: int, margin: float) -> str:
    if any("budget" in blocker or "lane" in blocker or "eligible" in blocker for blocker in blockers):
        return "blocked"
    if blockers or seed_count <= 1:
        return "inconclusive"
    if abs(margin) <= 0.25:
        return "no_material_change"
    return "gain"


def _system_rows_from_records(records: list[dict[str, Any]]) -> list[dict[str, Any]]:
    scores: dict[str, list[float]] = defaultdict(list)
    seeds: dict[str, set[int]] = defaultdict(set)
    for record in records:
        seed = int(record["seed"]) if record.get("seed") is not None else -1
        for system, score in dict(record.get("score_by_system") or {}).items():
            scores[str(system)].append(float(score))
            seeds[str(system)].add(seed)
    rows = []
    for system in sorted(scores):
        values = scores[system]
        rows.append(
            {
                "system": system,
                "seed_count": len(seeds[system]),
                "seeds": sorted(seeds[system]),
                "mean_score": float(mean(values)) if values else 0.0,
                "score_range": (max(values) - min(values)) if values else 0.0,
            }
        )
    return rows


def _score_systems(rows: list[dict[str, Any]], systems: tuple[str, ...]) -> dict[str, float]:
    by_benchmark: dict[str, list[dict[str, Any]]] = defaultdict(list)
    for row in rows:
        if str(row.get("system") or "") in systems:
            by_benchmark[str(row.get("benchmark_id") or "unknown")].append(row)

    scores = {system: 0.0 for system in systems}
    for benchmark_rows in by_benchmark.values():
        ok_rows = [
            row
            for row in benchmark_rows
            if row.get("outcome_status") == "ok" and row.get("metric_value") is not None
        ]
        if not ok_rows:
            continue
        direction = str(ok_rows[0].get("metric_direction") or "max")
        values = [(str(row["system"]), float(row["metric_value"])) for row in ok_rows]
        best = min(value for _, value in values) if direction == "min" else max(value for _, value in values)
        winners = [system for system, value in values if abs(value - best) <= 1e-12]
        increment = 1.0 if len(winners) == 1 else 0.5
        for system in winners:
            scores[system] += increment
    return {system: round(score, 6) for system, score in scores.items()}


def _engine_roles(groups: list[dict[str, Any]], trend_rows: list[dict[str, Any]]) -> list[dict[str, Any]]:
    leaders: dict[str, int] = defaultdict(int)
    top_two: dict[str, int] = defaultdict(int)
    families: dict[str, set[str]] = defaultdict(set)
    for group in groups:
        ranked = sorted(
            group.get("system_rows") or [],
            key=lambda row: float(row.get("mean_score") or 0.0),
            reverse=True,
        )
        if ranked:
            leaders[str(ranked[0]["system"])] += 1
        for row in ranked[:2]:
            top_two[str(row["system"])] += 1
    for family, rows in _rows_by_family(trend_rows).items():
        scores = _score_systems(rows, tuple(ALL_SYSTEMS))
        if scores:
            best = max(scores.values())
            for system, score in scores.items():
                if score == best and best > 0:
                    families[system].add(family)

    rows = []
    for system in ALL_SYSTEMS:
        if leaders[system]:
            role = "leader_candidate"
        elif top_two[system]:
            role = "challenger"
        elif families[system]:
            role = "specialist"
        else:
            role = "watch"
        if system == "primordia" and families[system]:
            role = "seed_source_specialist"
        rows.append(
            {
                "system": system,
                "role": role,
                "leader_group_count": leaders[system],
                "top_two_group_count": top_two[system],
                "family_leads": sorted(families[system]),
            }
        )
    return rows


def _lm_flatline_diagnostics(trend_rows: list[dict[str, Any]]) -> list[dict[str, Any]]:
    grouped: dict[str, list[dict[str, Any]]] = defaultdict(list)
    for row in trend_rows:
        task = str(row.get("task_kind") or "")
        family = str(row.get("benchmark_family") or "")
        if task == "language_modeling" or "language" in family.lower() or str(row.get("benchmark_id") or "").endswith("_lm"):
            grouped[str(row.get("system") or "unknown")].append(row)

    diagnostics = []
    for system, rows in sorted(grouped.items()):
        metric_values = [
            round(float(row["metric_value"]), 12)
            for row in rows
            if row.get("outcome_status") == "ok" and row.get("metric_value") is not None
        ]
        unique_metric_values = sorted(set(metric_values))
        diagnostics.append(
            {
                "system": system,
                "lm_row_count": len(rows),
                "unique_metric_value_count": len(unique_metric_values),
                "flatline_suspected": len(metric_values) >= 2 and len(unique_metric_values) <= 1,
                "benchmarks": sorted({str(row.get("benchmark_id") or "unknown") for row in rows}),
                "budgets": sorted({int(row.get("budget") or 0) for row in rows}),
            }
        )
    return diagnostics


def _transfer_evidence(
    summary_payloads: list[dict[str, Any]],
    trend_rows: list[dict[str, Any]],
    transfer_cases: list[dict[str, Any]],
) -> dict[str, Any]:
    seeded_rows = [
        row
        for row in trend_rows
        if dict(row.get("fairness_metadata") or {}).get("seeding_bucket") not in (None, "", "transfer-opaque", "unseeded")
    ]
    seed_sources = sorted(
        {
            str(dict(row.get("fairness_metadata") or {}).get("seed_source_system"))
            for row in seeded_rows
            if dict(row.get("fairness_metadata") or {}).get("seed_source_system")
        }
    )
    transfer_boundaries = sorted(
        {
            str(payload.get("transfer_boundary"))
            for payload in summary_payloads
            if payload.get("transfer_boundary")
        }
    )
    proof_states = sorted(
        {
            str(payload.get("transfer_proof_state"))
            for payload in summary_payloads
            if payload.get("transfer_proof_state")
        }
    )
    case_verdicts = Counter(str(case.get("verdict") or "unknown") for case in transfer_cases)
    regime_counts = Counter(str(case.get("regime") or "unknown") for case in transfer_cases)
    return {
        "seeded_trend_row_count": len(seeded_rows),
        "seed_sources": seed_sources,
        "transfer_boundaries": transfer_boundaries,
        "proof_states": proof_states,
        "native_transfer_claim_ready": any(state == "native-transfer-evidence" for state in proof_states),
        "portable_transfer_case_count": len(transfer_cases),
        "portable_transfer_verdict_counts": dict(sorted(case_verdicts.items())),
        "portable_transfer_regime_counts": dict(sorted(regime_counts.items())),
        "portable_transfer_consensus": _portable_transfer_consensus(transfer_cases),
    }


def _portable_transfer_consensus(transfer_cases: list[dict[str, Any]]) -> dict[str, Any]:
    grouped: dict[str, list[dict[str, Any]]] = defaultdict(list)
    for case in transfer_cases:
        grouped[str(case.get("regime") or "unknown")].append(case)
    consensus = {}
    for regime, cases in sorted(grouped.items()):
        verdict_counts = Counter(str(case.get("verdict") or "unknown") for case in cases)
        if verdict_counts.get("regression", 0):
            label = "regression"
        elif verdict_counts.get("gain", 0) > verdict_counts.get("inconclusive", 0):
            label = "portable_gain_signal"
        elif verdict_counts.get("gain", 0):
            label = "inconclusive_with_gain_signal"
        else:
            label = "inconclusive"
        consensus[regime] = {
            "case_count": len(cases),
            "verdict_counts": dict(sorted(verdict_counts.items())),
            "consensus": label,
        }
    return consensus


def _quality_diversity_evidence(summary_payloads: list[dict[str, Any]], trend_rows: list[dict[str, Any]]) -> dict[str, Any]:
    archive_terms = ("archive", "descriptor", "novelty", "map_elites", "map-elites", "diversity")
    matching_rows = []
    for row in trend_rows:
        encoded = json.dumps(
            {
                "architecture_summary": row.get("architecture_summary"),
                "fairness_metadata": row.get("fairness_metadata"),
            },
            sort_keys=True,
            default=str,
        ).lower()
        if any(term in encoded for term in archive_terms):
            matching_rows.append(row)
    matching_summaries = []
    for payload in summary_payloads:
        encoded = json.dumps(payload.get("quality_diversity") or payload.get("archive_evidence") or {}, sort_keys=True, default=str).lower()
        if encoded and encoded != "{}":
            matching_summaries.append(payload)
    return {
        "descriptor_or_archive_row_count": len(matching_rows),
        "summary_archive_evidence_count": len(matching_summaries),
        "systems": sorted({str(row.get("system") or "unknown") for row in matching_rows}),
        "claim_ready": bool(matching_rows or matching_summaries),
    }


def _rows_by_family(trend_rows: list[dict[str, Any]]) -> dict[str, list[dict[str, Any]]]:
    grouped: dict[str, list[dict[str, Any]]] = defaultdict(list)
    for row in trend_rows:
        grouped[str(row.get("benchmark_family") or row.get("task_kind") or "unknown")].append(row)
    return grouped


def _load_record_summary(record: dict[str, Any]) -> dict[str, Any]:
    path_value = record.get("registry_summary_path") or record.get("source_summary_path")
    if not path_value:
        return {}
    path = Path(str(path_value))
    if not path.exists():
        return {}
    payload = json.loads(path.read_text(encoding="utf-8"))
    for row in payload.get("trend_rows") or []:
        fairness = dict(row.get("fairness_metadata") or {})
        fairness.setdefault("comparison_label", record.get("label"))
        fairness.setdefault("comparison_cohort", "promoted-evidence")
        fairness.setdefault("comparison_case_id", record.get("record_id"))
        row["fairness_metadata"] = fairness
    return payload


def _load_transfer_case_payloads(records: list[dict[str, Any]]) -> list[dict[str, Any]]:
    found: dict[Path, None] = {}
    for record in records:
        for key in ("registry_summary_path", "source_summary_path"):
            path_value = record.get(key)
            if not path_value:
                continue
            path = Path(str(path_value))
            workspace_root = _workspace_root_from_summary(path)
            if workspace_root is None:
                continue
            reports_dir = workspace_root / "reports"
            if reports_dir.exists():
                for case_path in reports_dir.rglob("*_vs_control.json"):
                    found[case_path.resolve()] = None
    cases = []
    for path in sorted(found):
        payload = json.loads(path.read_text(encoding="utf-8"))
        payload.setdefault("summary_path", str(path))
        cases.append(payload)
    return cases


def _workspace_root_from_summary(path: Path) -> Path | None:
    current = path.resolve()
    for parent in current.parents:
        if parent.name == "reports":
            return parent.parent.resolve()
    return None


def _copy_summary_artifacts(*, summary_path: Path, registry: Path, label: str, digest: str) -> Path:
    destination = registry / "runs" / _slugify(label) / f"{summary_path.parent.name}-{digest[:12]}"
    destination.mkdir(parents=True, exist_ok=True)
    destination_summary = destination / "fair_matrix_summary.json"
    shutil.copy2(summary_path, destination_summary)
    for sibling_name in ("fair_matrix_summary.md", "contender_floor_report.json", "contender_floor_report.md"):
        sibling = summary_path.with_name(sibling_name)
        if sibling.exists():
            shutil.copy2(sibling, destination / sibling_name)
    _copy_transfer_case_siblings(summary_path=summary_path, destination=destination)
    return destination_summary


def _copy_transfer_case_siblings(*, summary_path: Path, destination: Path) -> None:
    if summary_path.parent.name not in {"02-direct", "03-staged"}:
        return
    prefix = summary_path.parent.name
    source_dir = summary_path.parent.parent
    for suffix in (".json", ".md"):
        source = source_dir / f"{prefix}_vs_control{suffix}"
        if source.exists():
            shutil.copy2(source, destination / source.name)


def _source_context(*, payload: dict[str, Any], trend_rows: list[dict[str, Any]], summary_path: Path) -> dict[str, Any]:
    code_versions = sorted(
        {
            str((row.get("fairness_metadata") or {}).get("code_version") or row.get("code_version"))
            for row in trend_rows
            if ((row.get("fairness_metadata") or {}).get("code_version") or row.get("code_version"))
        }
    )
    backend_labels = sorted(
        {
            str(
                row.get("runtime_backend")
                or row.get("backend")
                or (row.get("fairness_metadata") or {}).get("runtime_backend")
                or "unknown"
            )
            for row in trend_rows
        }
    )
    hardware_labels = sorted(
        {
            str(
                row.get("hardware_class")
                or row.get("device_name")
                or (row.get("fairness_metadata") or {}).get("hardware_class")
                or "unknown"
            )
            for row in trend_rows
        }
    )
    return {
        "git": {
            "code_version": code_versions[0] if len(code_versions) == 1 else ("mixed" if code_versions else _git(["rev-parse", "HEAD"])),
            "code_versions": code_versions,
            "branch": _git(["branch", "--show-current"]),
            "dirty": _git_dirty(),
        },
        "runtime": {
            "backend_labels": backend_labels,
            "hardware_labels": hardware_labels,
            "host": socket.gethostname(),
            "platform": platform.platform(),
            "machine": platform.machine(),
            "summary_path": str(summary_path),
            "lane_preset": (payload.get("lane") or {}).get("preset"),
        },
    }


def _contender_floor_context(summary_path: Path) -> dict[str, Any]:
    report_path = summary_path.with_name("contender_floor_report.json")
    if not report_path.exists():
        return {"available": False, "path": str(report_path)}
    payload = json.loads(report_path.read_text(encoding="utf-8"))
    benchmarks = list(payload.get("benchmarks") or [])
    statuses = sorted({str(row.get("floor_status") or "unknown") for row in benchmarks})
    return {
        "available": True,
        "path": str(report_path),
        "lane_floor_status": payload.get("lane_floor_status"),
        "floor_statuses": statuses,
        "blocked_count": sum(1 for row in benchmarks if row.get("floor_status") in {"missing", "failed"}),
    }


def _output_quality_context(summary_path: Path) -> dict[str, Any]:
    report_paths = sorted(summary_path.parents[2].rglob("output_quality_report.json")) if len(summary_path.parents) >= 3 else []
    levels = []
    systems = []
    for report_path in report_paths:
        payload = json.loads(report_path.read_text(encoding="utf-8"))
        levels.append(str(payload.get("quality_level") or "L0"))
        systems.append(str(payload.get("system") or "unknown"))
    return {
        "available": bool(report_paths),
        "report_count": len(report_paths),
        "quality_levels": sorted(set(levels)),
        "systems": sorted(set(systems)),
    }


def _load_index(registry: Path) -> dict[str, dict[str, Any]]:
    index_path = registry / "index.jsonl"
    if not index_path.exists():
        return {}
    rows = {}
    for line in index_path.read_text(encoding="utf-8").splitlines():
        if not line.strip():
            continue
        payload = json.loads(line)
        rows[str(payload["record_id"])] = payload
    return rows


def _write_index(registry: Path, records: list[dict[str, Any]]) -> None:
    (registry / "index.jsonl").write_text(
        "".join(json.dumps(record, sort_keys=True) + "\n" for record in records),
        encoding="utf-8",
    )
    (registry / "registry_manifest.json").write_text(
        json.dumps(
            {
                "schema_version": "evidence-registry-v1",
                "record_count": len(records),
                "updated_at": datetime.now(timezone.utc).isoformat(),
                "index": str((registry / "index.jsonl").resolve()),
            },
            indent=2,
        ),
        encoding="utf-8",
    )
    readme = registry / "README.md"
    if not readme.exists():
        readme.write_text(_registry_readme(), encoding="utf-8")


def _registry_readme() -> str:
    return (
        "# EvoNN Evidence Registry\n\n"
        "This directory stores promoted comparison evidence, not every local run artifact.\n\n"
        "## Promotion\n\n"
        "Promote a clean fair-matrix workspace with:\n\n"
        "```bash\n"
        "uv run --package evonn-compare evonn-compare evidence promote <workspace-or-summary> --registry evidence --label <cohort>\n"
        "```\n\n"
        "## Retention\n\n"
        "- `index.jsonl` is the compact durable registry.\n"
        "- `runs/` stores copied compact summaries when promotion uses `--copy-artifacts`.\n"
        "- Large raw run directories should stay outside git unless intentionally curated.\n"
        "- Do not edit existing rows by hand; add a new promoted row or supersession metadata in a follow-up command.\n\n"
        "## Review\n\n"
        "Use `evonn-compare evidence report --registry evidence` and then inspect `evidence_report.md`.\n"
        "Use `evonn-compare evidence validate --registry evidence --require-artifacts` before citing evidence in a PR.\n"
    )


def _git(args: list[str]) -> str | None:
    try:
        return subprocess.check_output(["git", *args], cwd=Path(__file__).resolve().parents[4], text=True).strip()
    except Exception:
        return None


def _git_dirty() -> bool | None:
    status = _git(["status", "--short"])
    if status is None:
        return None
    return bool(status.strip())


def _summary_budget(payload: dict[str, Any]) -> int:
    lane = payload.get("lane") or {}
    if lane.get("expected_budget") is not None:
        return int(lane["expected_budget"])
    return int(_first_row_value(payload.get("trend_rows") or payload.get("fair_rows") or [], "budget") or 0)


def _summary_seed(payload: dict[str, Any]) -> int:
    lane = payload.get("lane") or {}
    if lane.get("expected_seed") is not None:
        return int(lane["expected_seed"])
    return int(_first_row_value(payload.get("trend_rows") or payload.get("fair_rows") or [], "seed") or 0)


def _first_row_value(rows: list[dict[str, Any]], key: str, default: Any = None) -> Any:
    if not rows:
        return default
    return rows[0].get(key, default)


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _default_label(summary_path: Path, payload: dict[str, Any]) -> str:
    pack = str(payload.get("pack_name") or "evidence")
    budget = _summary_budget(payload)
    return f"{pack}@{budget}:{summary_path.parent.name}"


def _slugify(value: str) -> str:
    return re.sub(r"[^a-z0-9]+", "-", value.lower()).strip("-") or "evidence"


def _render_report_markdown(payload: dict[str, Any]) -> str:
    group_rows = "\n".join(
        "| {label} | {pack} | {budget} | {seeds} | {decision} | {leader} | {blockers} |".format(
            label=row["label"],
            pack=row["pack_name"],
            budget=row["budget"],
            seeds=", ".join(str(seed) for seed in row["seeds"]),
            decision=row["decision_label"],
            leader=row.get("leader") or "n/a",
            blockers="; ".join(row.get("decision_blockers") or []) or "none",
        )
        for row in payload["groups"]
    )
    role_rows = "\n".join(
        f"| {row['system']} | {row['role']} | {row['leader_group_count']} | {', '.join(row['family_leads']) or 'none'} |"
        for row in payload["engine_roles"]
    )
    lm_rows = "\n".join(
        f"| {row['system']} | {row['lm_row_count']} | {row['unique_metric_value_count']} | {row['flatline_suspected']} |"
        for row in payload["lm_flatline_diagnostics"]
    )
    comparison_rows = "\n".join(
        "| {before} -> {after} | {pack} | {budget} | {delta} | {decision} |".format(
            before=row["before_label"],
            after=row["after_label"],
            pack=row["pack_name"],
            budget=row["budget"],
            delta=row["aggregate_score_delta"],
            decision=row["decision_label"],
        )
        for row in payload["before_after_comparisons"]
    )
    artifact_validation = payload.get("artifact_validation") or {}
    validation_issues = "; ".join(artifact_validation.get("issues") or []) or "none"
    validation_warnings = "; ".join(artifact_validation.get("warnings") or []) or "none"
    return (
        "# EvoNN Evidence Registry Report\n\n"
        f"- Generated At: `{payload['generated_at']}`\n"
        f"- Registry: `{payload['registry']}`\n"
        f"- Records: `{payload['record_count']}`\n"
        f"- Minimum Seeds For Decision Labels: `{payload['min_seeds']}`\n\n"
        "## Decision Groups\n\n"
        "| Label | Pack | Budget | Seeds | Decision | Leader | Blockers |\n"
        "| --- | --- | ---: | --- | --- | --- | --- |\n"
        f"{group_rows or '| n/a | n/a | 0 | n/a | inconclusive | n/a | no promoted runs |'}\n\n"
        "## Before/After Comparisons\n\n"
        "| Comparison | Pack | Budget | Aggregate Delta | Decision |\n"
        "| --- | --- | ---: | ---: | --- |\n"
        f"{comparison_rows or '| n/a | n/a | 0 | 0 | inconclusive |'}\n\n"
        "## Engine Roles\n\n"
        "| System | Role | Leader Groups | Family Leads |\n"
        "| --- | --- | ---: | --- |\n"
        f"{role_rows}\n\n"
        "## LM Flatline Diagnostics\n\n"
        "| System | LM Rows | Unique Metric Values | Flatline Suspected |\n"
        "| --- | ---: | ---: | --- |\n"
        f"{lm_rows or '| n/a | 0 | 0 | false |'}\n\n"
        "## Transfer Evidence\n\n"
        f"- Seeded Trend Rows: `{payload['transfer_evidence']['seeded_trend_row_count']}`\n"
        f"- Seed Sources: `{', '.join(payload['transfer_evidence']['seed_sources']) or 'none'}`\n"
        f"- Proof States: `{', '.join(payload['transfer_evidence']['proof_states']) or 'none'}`\n"
        f"- Native Transfer Claim Ready: `{payload['transfer_evidence']['native_transfer_claim_ready']}`\n"
        f"- Portable Transfer Cases: `{payload['transfer_evidence']['portable_transfer_case_count']}`\n"
        f"- Portable Verdict Counts: `{payload['transfer_evidence']['portable_transfer_verdict_counts']}`\n"
        f"- Portable Consensus: `{payload['transfer_evidence']['portable_transfer_consensus']}`\n\n"
        "## Quality Diversity Evidence\n\n"
        f"- Descriptor Or Archive Rows: `{payload['quality_diversity_evidence']['descriptor_or_archive_row_count']}`\n"
        f"- Summary Archive Evidence Count: `{payload['quality_diversity_evidence']['summary_archive_evidence_count']}`\n"
        f"- Claim Ready: `{payload['quality_diversity_evidence']['claim_ready']}`\n\n"
        "## Artifact Validation\n\n"
        f"- OK: `{artifact_validation.get('ok')}`\n"
        f"- Issues: `{validation_issues}`\n"
        f"- Warnings: `{validation_warnings}`\n"
    )
