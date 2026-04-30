"""Workspace-level research campaign state for resumable compare orchestration."""

from __future__ import annotations

from datetime import datetime, timezone
import json
from pathlib import Path
from typing import Any


STATE_FILENAME = "state.json"
CASE_STATUSES = ("pending", "running", "interrupted", "failed", "completed")


class StopRequested(RuntimeError):
    """Raised when a workspace stop request should halt orchestration."""


def workspace_state_path(workspace: Path) -> Path:
    return workspace.resolve() / STATE_FILENAME


def load_workspace_state(workspace: Path) -> dict[str, Any] | None:
    path = workspace_state_path(workspace)
    if not path.exists():
        return None
    return json.loads(path.read_text(encoding="utf-8"))


def require_workspace_state(workspace: Path) -> dict[str, Any]:
    state = load_workspace_state(workspace)
    if state is None:
        raise FileNotFoundError(f"workspace state not found: {workspace_state_path(workspace)}")
    return state


def sync_workspace_state(
    *,
    workspace_kind: str,
    workspace: Path,
    manifest_path: Path,
    cases: list[Any],
    lane_preset: str | None = None,
    trend_dataset_path: Path | None = None,
) -> dict[str, Any]:
    workspace_path = workspace.resolve()
    existing = load_workspace_state(workspace_path) or {}
    existing_cases = {str(case.get("case_id")): case for case in existing.get("cases", [])}
    now = _now()
    state = {
        "schema_version": "1.0",
        "workspace_kind": workspace_kind,
        "workspace": str(workspace_path),
        "state_path": str(workspace_state_path(workspace_path)),
        "manifest_path": str(manifest_path.resolve()),
        "lane_preset": lane_preset,
        "trend_dataset_path": (
            str(trend_dataset_path.resolve())
            if trend_dataset_path is not None
            else existing.get("trend_dataset_path")
        ),
        "created_at": existing.get("created_at") or now,
        "updated_at": now,
        "stop_requested": bool(existing.get("stop_requested", False)),
        "stop_requested_at": existing.get("stop_requested_at"),
        "cases": [],
    }
    for case in cases:
        case_id = build_case_id(case)
        previous = existing_cases.get(case_id, {})
        state["cases"].append(
            {
                "case_id": case_id,
                "pack_name": str(getattr(case, "pack_name")),
                "budget": int(getattr(case, "budget")),
                "seed": int(getattr(case, "seed")),
                "lane_preset": getattr(case, "lane_preset", None),
                "systems": list(_case_systems(case)),
                "run_dirs": _case_run_dirs(case),
                "artifacts": _case_artifacts(case),
                "report_dir": _case_report_dir(case),
                "log_dir": _case_log_dir(case),
                "status": previous.get("status", "pending"),
                "attempts": int(previous.get("attempts", 0)),
                "resume_count": int(previous.get("resume_count", 0)),
                "latest_attempt_resumed": bool(previous.get("latest_attempt_resumed", False)),
                "current_stage": previous.get("current_stage"),
                "started_at": previous.get("started_at"),
                "finished_at": previous.get("finished_at"),
                "last_error": previous.get("last_error"),
                "artifact_integrity_ok": previous.get("artifact_integrity_ok"),
                "integrity_issues": list(previous.get("integrity_issues") or []),
            }
        )
    _write_workspace_state(workspace_path, state)
    return state


def clear_stop_request(workspace: Path) -> dict[str, Any]:
    state = require_workspace_state(workspace)
    state["stop_requested"] = False
    state["stop_requested_at"] = None
    state["updated_at"] = _now()
    _write_workspace_state(workspace, state)
    return state


def request_stop(workspace: Path) -> dict[str, Any]:
    state = require_workspace_state(workspace)
    state["stop_requested"] = True
    state["stop_requested_at"] = _now()
    state["updated_at"] = _now()
    _write_workspace_state(workspace, state)
    return state


def stop_requested(workspace: Path) -> bool:
    state = load_workspace_state(workspace)
    return bool(state and state.get("stop_requested"))


def mark_stale_running_cases_interrupted(workspace: Path) -> int:
    state = require_workspace_state(workspace)
    updated = 0
    for case in state.get("cases", []):
        if case.get("status") != "running":
            continue
        case["status"] = "interrupted"
        case["current_stage"] = None
        case["finished_at"] = _now()
        if not case.get("last_error"):
            case["last_error"] = "previous orchestrator exited before case completion"
        updated += 1
    if updated:
        state["updated_at"] = _now()
        _write_workspace_state(workspace, state)
    return updated


def start_case(workspace: Path, *, case_id: str, resume_requested: bool) -> dict[str, Any]:
    state = require_workspace_state(workspace)
    case = _find_case(state, case_id=case_id)
    prior_attempts = int(case.get("attempts", 0))
    prior_status = str(case.get("status") or "pending")
    case["attempts"] = prior_attempts + 1
    case["latest_attempt_resumed"] = bool(
        resume_requested and (prior_attempts > 0 or prior_status in {"running", "interrupted", "failed"})
    )
    if case["latest_attempt_resumed"]:
        case["resume_count"] = int(case.get("resume_count", 0)) + 1
    case["status"] = "running"
    case["current_stage"] = "prepare"
    case["started_at"] = _now()
    case["finished_at"] = None
    case["last_error"] = None
    case["artifact_integrity_ok"] = None
    case["integrity_issues"] = []
    state["updated_at"] = _now()
    _write_workspace_state(workspace, state)
    return case


def update_case_stage(workspace: Path, *, case_id: str, stage: str) -> dict[str, Any]:
    state = require_workspace_state(workspace)
    case = _find_case(state, case_id=case_id)
    case["current_stage"] = stage
    if case.get("status") == "pending":
        case["status"] = "running"
    state["updated_at"] = _now()
    _write_workspace_state(workspace, state)
    return case


def complete_case(workspace: Path, *, case_id: str, integrity_issues: list[str]) -> dict[str, Any]:
    state = require_workspace_state(workspace)
    case = _find_case(state, case_id=case_id)
    case["status"] = "completed"
    case["current_stage"] = None
    case["finished_at"] = _now()
    case["last_error"] = None
    case["artifact_integrity_ok"] = not integrity_issues
    case["integrity_issues"] = integrity_issues
    state["updated_at"] = _now()
    _write_workspace_state(workspace, state)
    return case


def interrupt_case(workspace: Path, *, case_id: str, reason: str) -> dict[str, Any]:
    state = require_workspace_state(workspace)
    case = _find_case(state, case_id=case_id)
    case["status"] = "interrupted"
    case["current_stage"] = None
    case["finished_at"] = _now()
    case["last_error"] = reason
    case["artifact_integrity_ok"] = False
    state["updated_at"] = _now()
    _write_workspace_state(workspace, state)
    return case


def fail_case(workspace: Path, *, case_id: str, reason: str, integrity_issues: list[str] | None = None) -> dict[str, Any]:
    state = require_workspace_state(workspace)
    case = _find_case(state, case_id=case_id)
    case["status"] = "failed"
    case["current_stage"] = None
    case["finished_at"] = _now()
    case["last_error"] = reason
    case["artifact_integrity_ok"] = False
    case["integrity_issues"] = list(integrity_issues or [])
    state["updated_at"] = _now()
    _write_workspace_state(workspace, state)
    return case


def summarize_workspace_state(state: dict[str, Any]) -> dict[str, Any]:
    counts = {status: 0 for status in CASE_STATUSES}
    resumed_case_count = 0
    integrity_failed_count = 0
    for case in state.get("cases", []):
        status = str(case.get("status") or "pending")
        counts.setdefault(status, 0)
        counts[status] += 1
        if int(case.get("resume_count", 0)) > 0 or bool(case.get("latest_attempt_resumed")):
            resumed_case_count += 1
        if case.get("artifact_integrity_ok") is False:
            integrity_failed_count += 1
    return {
        "case_count": len(state.get("cases", [])),
        "status_counts": counts,
        "resumed_case_count": resumed_case_count,
        "integrity_failed_count": integrity_failed_count,
        "stop_requested": bool(state.get("stop_requested")),
    }


def render_workspace_state_markdown(state: dict[str, Any]) -> str:
    summary = summarize_workspace_state(state)
    lines = [
        "## Campaign State",
        "",
        f"- Workspace Kind: `{state.get('workspace_kind', 'unknown')}`",
        f"- Manifest: `{state.get('manifest_path', 'unknown')}`",
        f"- Stop Requested: `{'yes' if summary['stop_requested'] else 'no'}`",
        f"- Cases: `{summary['case_count']}`",
        f"- Resumed Cases: `{summary['resumed_case_count']}`",
        f"- Integrity Failures: `{summary['integrity_failed_count']}`",
        "",
        "### Status Counts",
        "",
        "| Status | Count |",
        "|---|---:|",
    ]
    for status in CASE_STATUSES:
        lines.append(f"| {status} | {summary['status_counts'].get(status, 0)} |")
    lines.extend(
        [
            "",
            "### Case Table",
            "",
            "| Pack | Budget | Seed | Status | Stage | Attempts | Resumes | Integrity |",
            "|---|---:|---:|---|---|---:|---:|---|",
        ]
    )
    for case in _sorted_cases(state):
        integrity_ok = case.get("artifact_integrity_ok")
        if integrity_ok is None:
            integrity = "pending"
        else:
            integrity = "ok" if integrity_ok else "failed"
        lines.append(
            f"| {case['pack_name']} | {case['budget']} | {case['seed']} | {case['status']} | "
            f"{case.get('current_stage') or '---'} | {case.get('attempts', 0)} | {case.get('resume_count', 0)} | {integrity} |"
        )
        issues = list(case.get("integrity_issues") or [])
        if issues:
            lines.append(f"| {case['pack_name']} issues |  |  |  |  |  |  | {'; '.join(issues)} |")
        last_error = case.get("last_error")
        if last_error:
            lines.append(f"| {case['pack_name']} error |  |  |  |  |  |  | {last_error} |")
    return "\n".join(lines)


def build_case_id(case: Any) -> str:
    return f"{getattr(case, 'pack_name')}|budget={int(getattr(case, 'budget'))}|seed={int(getattr(case, 'seed'))}"


def case_integrity_issues(case: Any) -> list[str]:
    issues: list[str] = []
    for label, path in _case_artifacts(case).items():
        if not Path(path).exists():
            issues.append(f"missing {label}: {path}")
    for system, run_dir_value in _case_run_dirs(case).items():
        run_dir = Path(run_dir_value)
        required = [
            run_dir / "manifest.json",
            run_dir / "results.json",
            run_dir / "summary.json",
            run_dir / "report.md",
        ]
        for path in required:
            if not path.exists():
                issues.append(f"missing {system} artifact: {path}")
    return issues


def _find_case(state: dict[str, Any], *, case_id: str) -> dict[str, Any]:
    for case in state.get("cases", []):
        if case.get("case_id") == case_id:
            return case
    raise KeyError(f"unknown case id: {case_id}")


def _write_workspace_state(workspace: Path, state: dict[str, Any]) -> None:
    path = workspace_state_path(workspace)
    path.parent.mkdir(parents=True, exist_ok=True)
    temp_path = path.with_suffix(path.suffix + ".tmp")
    temp_path.write_text(json.dumps(state, indent=2), encoding="utf-8")
    temp_path.replace(path)


def _case_systems(case: Any) -> tuple[str, ...]:
    systems = getattr(case, "systems", None)
    if systems is not None:
        return tuple(str(system) for system in systems)
    return tuple(_case_run_dirs(case).keys())


def _case_run_dirs(case: Any) -> dict[str, str]:
    run_dirs: dict[str, str] = {}
    for system in ("prism", "topograph", "stratograph", "primordia", "contenders"):
        value = getattr(case, f"{system}_run_dir", None)
        if value is not None:
            run_dirs[system] = str(Path(value).resolve())
    return run_dirs


def _case_artifacts(case: Any) -> dict[str, str]:
    if hasattr(case, "summary_output_path"):
        summary_output_path = Path(getattr(case, "summary_output_path")).resolve()
        return {
            "summary_markdown": str(summary_output_path),
            "summary_json": str(summary_output_path.with_suffix(".json")),
            "lane_acceptance": str(summary_output_path.with_name("lane_acceptance.json")),
            "trend_rows": str(summary_output_path.with_name("trend_rows.json")),
            "trend_report": str(summary_output_path.with_name("trend_report.md")),
            "trend_records_json": str(summary_output_path.with_name("fair_matrix_trends.json")),
            "trend_records_jsonl": str(summary_output_path.with_name("fair_matrix_trends.jsonl")),
        }
    comparison_output_path = Path(getattr(case, "comparison_output_path")).resolve()
    return {
        "comparison_markdown": str(comparison_output_path),
        "comparison_json": str(comparison_output_path.with_suffix(".json")),
    }


def _case_report_dir(case: Any) -> str:
    report_dir = getattr(case, "report_dir", None)
    if report_dir is not None:
        return str(Path(report_dir).resolve())
    return str(Path(getattr(case, "comparison_output_path")).resolve().parent)


def _case_log_dir(case: Any) -> str:
    log_dir = getattr(case, "log_dir", None)
    if log_dir is not None:
        return str(Path(log_dir).resolve())
    if hasattr(case, "comparison_output_path"):
        return str(Path(getattr(case, "comparison_output_path")).resolve().parent / "logs")
    return ""


def _sorted_cases(state: dict[str, Any]) -> list[dict[str, Any]]:
    return sorted(
        list(state.get("cases", [])),
        key=lambda case: (int(case.get("budget", 0)), int(case.get("seed", 0)), str(case.get("pack_name", ""))),
    )


def _now() -> str:
    return datetime.now(timezone.utc).isoformat()
