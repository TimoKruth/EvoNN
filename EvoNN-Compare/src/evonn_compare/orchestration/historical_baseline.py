"""Historical baseline import helpers for compare-owned workspace refreshes."""

from __future__ import annotations

from datetime import datetime, timezone
import json
from pathlib import Path
import re
from typing import Any

from evonn_compare.reporting.fair_matrix_dashboard import discover_fair_matrix_summaries


def register_historical_baseline(
    *,
    workspace: Path,
    baseline_inputs: list[Path],
    label: str | None = None,
) -> dict[str, str | int]:
    """Import historical fair-matrix summaries into a workspace-owned baseline cohort."""

    workspace_path = workspace.resolve()
    input_paths = [path.resolve() for path in baseline_inputs]
    summary_paths = discover_fair_matrix_summaries(input_paths)
    if not summary_paths:
        raise ValueError("no fair-matrix summaries found in the supplied historical baseline inputs")

    current_contract = _summarize_workspace_contract(
        discover_fair_matrix_summaries([workspace_path / "reports"])
    )
    imported_at = datetime.now(timezone.utc).isoformat()
    baseline_label = label or _default_baseline_label(input_paths[0], summary_paths)
    baseline_slug = _slugify(baseline_label)
    baseline_root = workspace_path / "baselines" / baseline_slug
    reports_root = baseline_root / "reports"
    trends_root = baseline_root / "trends"
    reports_root.mkdir(parents=True, exist_ok=True)
    trends_root.mkdir(parents=True, exist_ok=True)

    imported_rows: list[dict[str, Any]] = []
    imported_summaries: list[dict[str, Any]] = []

    for summary_path in summary_paths:
        payload = json.loads(summary_path.read_text(encoding="utf-8"))
        integrity = _summary_integrity(summary_path, payload)
        compatibility = _summary_compatibility(payload, current_contract)
        destination_case = _slugify(summary_path.parent.name or summary_path.stem)
        destination_dir = reports_root / destination_case
        destination_dir.mkdir(parents=True, exist_ok=True)

        annotated_payload, annotated_rows = _annotate_summary_payload(
            payload=payload,
            source_summary_path=summary_path.resolve(),
            baseline_label=baseline_label,
            imported_at=imported_at,
            compatibility=compatibility,
            integrity=integrity,
        )
        imported_rows.extend(annotated_rows)

        destination_summary_path = destination_dir / "fair_matrix_summary.json"
        destination_summary_path.write_text(
            json.dumps(annotated_payload, indent=2),
            encoding="utf-8",
        )
        destination_dir.joinpath("trend_rows.json").write_text(
            json.dumps(annotated_rows, indent=2),
            encoding="utf-8",
        )
        destination_dir.joinpath("fair_matrix_summary.md").write_text(
            _render_import_stub_markdown(
                baseline_label=baseline_label,
                source_summary_path=summary_path.resolve(),
                compatibility=compatibility,
                integrity=integrity,
                imported_at=imported_at,
            ),
            encoding="utf-8",
        )

        imported_summaries.append(
            {
                "source_summary_path": str(summary_path.resolve()),
                "destination_summary_path": str(destination_summary_path.resolve()),
                "pack_name": str(annotated_payload.get("pack_name") or "unknown"),
                "budget": _summary_budget(annotated_payload),
                "seed": _summary_seed(annotated_payload),
                "systems": list(annotated_payload.get("systems") or []),
                "trend_row_count": len(annotated_rows),
                "compatibility": compatibility,
                "integrity": integrity,
            }
        )

    baseline_dataset_path = trends_root / "fair_matrix_trend_rows.jsonl"
    baseline_dataset_path.write_text(
        "".join(json.dumps(row) + "\n" for row in imported_rows),
        encoding="utf-8",
    )

    manifest_path = baseline_root / "baseline_manifest.json"
    manifest_path.write_text(
        json.dumps(
            {
                "baseline_label": baseline_label,
                "baseline_slug": baseline_slug,
                "comparison_cohort": "historical-baseline",
                "workspace": str(workspace_path),
                "source_inputs": [str(path) for path in input_paths],
                "imported_at": imported_at,
                "current_workspace_contract": current_contract,
                "summary_count": len(imported_summaries),
                "trend_row_count": len(imported_rows),
                "source_summaries": imported_summaries,
            },
            indent=2,
        ),
        encoding="utf-8",
    )

    return {
        "baseline_label": baseline_label,
        "baseline_root": str(baseline_root),
        "baseline_manifest": str(manifest_path),
        "trend_dataset": str(baseline_dataset_path),
        "summary_count": len(imported_summaries),
        "trend_row_count": len(imported_rows),
    }


def discover_workspace_trend_inputs(
    workspace: Path,
    *,
    summary_paths: list[Path] | None = None,
) -> list[Path]:
    """Resolve canonical and legacy trend sources for one workspace."""

    workspace_path = workspace.resolve()
    found: dict[Path, None] = {}
    for candidate in (
        workspace_path / "trends" / "fair_matrix_trend_rows.jsonl",
        workspace_path / "trends" / "fair_matrix_trends.jsonl",
        workspace_path / "fair_matrix_trend_rows.jsonl",
        workspace_path / "reports" / "fair_matrix_trend_rows.jsonl",
    ):
        if candidate.exists():
            found[candidate.resolve()] = None

    baselines_root = workspace_path / "baselines"
    if baselines_root.exists():
        for candidate in baselines_root.rglob("fair_matrix_trend_rows.jsonl"):
            found[candidate.resolve()] = None
        for candidate in baselines_root.rglob("fair_matrix_trends.jsonl"):
            found[candidate.resolve()] = None

    for summary_path in summary_paths or discover_fair_matrix_summaries(
        [workspace_path / "reports", baselines_root, workspace_path]
    ):
        found[summary_path.resolve()] = None
    return sorted(found)


def _annotate_summary_payload(
    *,
    payload: dict[str, Any],
    source_summary_path: Path,
    baseline_label: str,
    imported_at: str,
    compatibility: dict[str, Any],
    integrity: dict[str, Any],
) -> tuple[dict[str, Any], list[dict[str, Any]]]:
    summary_payload = dict(payload)
    pack_name = str(summary_payload.get("pack_name") or "unknown")
    budget = _summary_budget(summary_payload)
    seed = _summary_seed(summary_payload)
    case_id = f"historical-baseline:{_slugify(baseline_label)}:{source_summary_path.parent.name}:{pack_name}:{budget}:{seed}"
    annotated_rows: list[dict[str, Any]] = []
    for row in list(summary_payload.get("trend_rows") or []):
        row_payload = dict(row)
        fairness = dict(row_payload.get("fairness_metadata") or {})
        fairness.update(
            {
                "comparison_cohort": "historical-baseline",
                "comparison_label": baseline_label,
                "comparison_case_id": case_id,
                "baseline_label": baseline_label,
                "baseline_source_summary_path": str(source_summary_path),
                "baseline_imported_at": imported_at,
                "baseline_seed_relation": compatibility.get("seed_relation"),
                "baseline_pack_relation": compatibility.get("pack_relation"),
                "baseline_budget_relation": compatibility.get("budget_relation"),
                "baseline_system_overlap": list(compatibility.get("system_overlap") or []),
                "baseline_integrity_ok": bool(integrity.get("ok")),
            }
        )
        row_payload["fairness_metadata"] = fairness
        annotated_rows.append(row_payload)

    summary_payload["trend_rows"] = annotated_rows
    summary_payload["baseline_context"] = {
        "comparison_cohort": "historical-baseline",
        "baseline_label": baseline_label,
        "comparison_case_id": case_id,
        "source_summary_path": str(source_summary_path),
        "imported_at": imported_at,
        "compatibility": compatibility,
        "integrity": integrity,
    }
    return summary_payload, annotated_rows


def _summary_integrity(summary_path: Path, payload: dict[str, Any]) -> dict[str, Any]:
    rows = payload.get("trend_rows")
    systems = payload.get("systems")
    issues = []
    if not isinstance(rows, list) or not rows:
        issues.append("trend_rows missing or empty")
    if not isinstance(systems, list) or not systems:
        issues.append("systems missing or empty")
    budget = _summary_budget(payload)
    seed = _summary_seed(payload)
    if budget is None:
        issues.append("budget missing")
    if seed is None:
        issues.append("seed missing")
    return {
        "ok": not issues,
        "issues": issues,
        "summary_path": str(summary_path),
        "trend_row_count": 0 if not isinstance(rows, list) else len(rows),
    }


def _summary_compatibility(payload: dict[str, Any], current_contract: dict[str, Any]) -> dict[str, Any]:
    pack_name = str(payload.get("pack_name") or "unknown")
    budget = _summary_budget(payload)
    seed = _summary_seed(payload)
    systems = {str(value) for value in payload.get("systems") or []}
    current_packs = set(current_contract.get("packs") or [])
    current_budgets = set(current_contract.get("budgets") or [])
    current_seeds = set(current_contract.get("seeds") or [])
    current_systems = set(current_contract.get("systems") or [])

    if not current_contract.get("summary_count"):
        status = "no-current-workspace"
    elif pack_name in current_packs and budget in current_budgets and systems & current_systems:
        status = "lane-overlap"
    else:
        status = "partial-or-disjoint"

    seed_relation = "unknown"
    if current_contract.get("summary_count"):
        seed_relation = "overlap" if seed in current_seeds else "disjoint"

    assumptions = [
        f"pack relation: {'exact' if pack_name in current_packs else 'different'}",
        f"budget relation: {'overlap' if budget in current_budgets else 'disjoint'}",
        f"seed relation: {seed_relation}",
        "seed ids stay separate by comparison cohort and case id even when numeric seeds overlap",
    ]
    return {
        "status": status,
        "pack_relation": "exact" if pack_name in current_packs else "different",
        "budget_relation": "overlap" if budget in current_budgets else "disjoint",
        "seed_relation": seed_relation,
        "system_overlap": sorted(systems & current_systems),
        "current_packs": sorted(current_packs),
        "current_budgets": sorted(current_budgets),
        "current_seeds": sorted(current_seeds),
        "assumptions": assumptions,
    }


def _summarize_workspace_contract(summary_paths: list[Path]) -> dict[str, Any]:
    packs: set[str] = set()
    budgets: set[int] = set()
    seeds: set[int] = set()
    systems: set[str] = set()
    for summary_path in summary_paths:
        payload = json.loads(summary_path.read_text(encoding="utf-8"))
        packs.add(str(payload.get("pack_name") or "unknown"))
        budget = _summary_budget(payload)
        seed = _summary_seed(payload)
        if budget is not None:
            budgets.add(int(budget))
        if seed is not None:
            seeds.add(int(seed))
        systems.update(str(value) for value in payload.get("systems") or [])
    return {
        "summary_count": len(summary_paths),
        "packs": sorted(packs),
        "budgets": sorted(budgets),
        "seeds": sorted(seeds),
        "systems": sorted(systems),
    }


def _summary_budget(payload: dict[str, Any]) -> int | None:
    lane = payload.get("lane") or {}
    if lane.get("expected_budget") is not None:
        return int(lane["expected_budget"])
    fair_rows = payload.get("fair_rows") or payload.get("reference_rows") or []
    if fair_rows:
        return int(fair_rows[0]["budget"])
    trend_rows = payload.get("trend_rows") or []
    if trend_rows:
        return int(trend_rows[0]["budget"])
    return None


def _summary_seed(payload: dict[str, Any]) -> int | None:
    lane = payload.get("lane") or {}
    if lane.get("expected_seed") is not None:
        return int(lane["expected_seed"])
    fair_rows = payload.get("fair_rows") or payload.get("reference_rows") or []
    if fair_rows:
        return int(fair_rows[0]["seed"])
    trend_rows = payload.get("trend_rows") or []
    if trend_rows:
        return int(trend_rows[0]["seed"])
    return None


def _default_baseline_label(input_path: Path, summary_paths: list[Path]) -> str:
    if input_path.is_file():
        return input_path.stem
    if len(summary_paths) == 1:
        return summary_paths[0].parent.name
    return input_path.name or "historical-baseline"


def _render_import_stub_markdown(
    *,
    baseline_label: str,
    source_summary_path: Path,
    compatibility: dict[str, Any],
    integrity: dict[str, Any],
    imported_at: str,
) -> str:
    assumptions = compatibility.get("assumptions") or []
    rendered_assumptions = "\n".join(f"- {item}" for item in assumptions) or "- none recorded"
    issues = integrity.get("issues") or []
    rendered_issues = "\n".join(f"- {item}" for item in issues) or "- none"
    return (
        f"# Historical Baseline Import: {baseline_label}\n\n"
        f"- Imported At: `{imported_at}`\n"
        f"- Source Summary: `{source_summary_path}`\n"
        f"- Compatibility Status: `{compatibility.get('status', 'unknown')}`\n"
        f"- Integrity OK: `{'yes' if integrity.get('ok') else 'no'}`\n\n"
        "## Compatibility Assumptions\n\n"
        f"{rendered_assumptions}\n\n"
        "## Integrity Notes\n\n"
        f"{rendered_issues}\n"
    )


def _slugify(value: str) -> str:
    return re.sub(r"[^a-z0-9]+", "-", value.lower()).strip("-") or "historical-baseline"
