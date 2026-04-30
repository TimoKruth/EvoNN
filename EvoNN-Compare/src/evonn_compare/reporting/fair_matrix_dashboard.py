"""Static dashboard renderer for accumulated fair-matrix summaries."""

from __future__ import annotations

from collections import defaultdict
from datetime import datetime, timezone
import html
import json
import os
from pathlib import Path
from typing import Any

from evonn_compare.reporting.fair_matrix_stats import (
    build_multi_seed_statistics,
    build_scope_run_summaries,
    build_scope_summary,
)
from evonn_compare.specialization import build_engine_profiles, build_family_leaderboards

ALL_SYSTEMS = ("prism", "topograph", "stratograph", "primordia", "contenders")
PROJECT_SYSTEMS = ("prism", "topograph", "stratograph", "primordia")
CONTRACT_FAIR_STATES = {"contract-fair", "portable-contract-fair", "trusted-core", "trusted-extended"}


def discover_fair_matrix_summaries(inputs: list[Path] | None) -> list[Path]:
    roots = inputs or [Path("EvoNN-Compare/manual_compare_runs")]
    found: dict[Path, None] = {}
    for root in roots:
        if root.is_file():
            if root.name == "fair_matrix_summary.json":
                found[root.resolve()] = None
            continue
        if not root.exists():
            continue
        direct = root / "fair_matrix_summary.json"
        if direct.exists():
            found[direct.resolve()] = None
            continue
        for path in root.rglob("fair_matrix_summary.json"):
            found[path.resolve()] = None
    return sorted(found)


def build_dashboard_payload(
    summary_paths: list[Path],
    *,
    output_path: Path,
    campaign_state: dict[str, Any] | None = None,
) -> dict[str, Any]:
    runs: list[dict[str, Any]] = []
    for path in summary_paths:
        payload = json.loads(path.read_text(encoding="utf-8"))
        runs.append(_build_run_entry(path=path, payload=payload, output_path=output_path))

    runs.sort(key=lambda item: (item["budget"], item["seed"], item["pack_name"], item["summary_json_path"]))
    all_rows = [row for run in runs for row in run["trend_rows"]]
    transfer = _build_transfer_dashboard_payload(summary_paths, output_path=output_path)
    campaign_state_payload = _build_campaign_state_payload(campaign_state, output_path=output_path)
    return {
        "generated_at": datetime.now(timezone.utc).isoformat(),
        "summary_count": len(runs),
        "packs": sorted({item["pack_name"] for item in runs}),
        "budgets": sorted({item["budget"] for item in runs}),
        "lane_counts": {
            "fair": sum(1 for item in runs if item["lane"]["fairness_ok"]),
            "contract_fair": sum(1 for item in runs if item["lane"]["operating_state"] in CONTRACT_FAIR_STATES),
            "trusted_core": sum(1 for item in runs if item["lane"]["operating_state"] in {"trusted-core", "trusted-extended"}),
            "trusted_extended": sum(1 for item in runs if item["lane"]["operating_state"] == "trusted-extended"),
            "repeatable": sum(1 for item in runs if item["lane"]["repeatability_ready"]),
            "artifact_complete": sum(1 for item in runs if item["lane"]["artifact_completeness_ok"]),
        },
        "runs": runs,
        "leaderboards": {
            "all_systems": _aggregate_leaderboard(runs, scope_key="all_scope", systems=ALL_SYSTEMS),
            "projects_only": _aggregate_leaderboard(runs, scope_key="project_scope", systems=PROJECT_SYSTEMS),
        },
        "specialization": {
            "family_leaderboards": {
                "all_systems": build_family_leaderboards(all_rows, systems=ALL_SYSTEMS),
                "projects_only": build_family_leaderboards(all_rows, systems=PROJECT_SYSTEMS),
            },
            "engine_profiles": {
                "all_systems": build_engine_profiles(all_rows, systems=ALL_SYSTEMS),
                "projects_only": build_engine_profiles(all_rows, systems=PROJECT_SYSTEMS),
            },
        },
        "multi_seed": {
            "all_systems": build_multi_seed_statistics(all_rows, systems=ALL_SYSTEMS),
            "projects_only": build_multi_seed_statistics(all_rows, systems=PROJECT_SYSTEMS),
        },
        "seed_scorecards": {
            "all_systems": build_scope_run_summaries(all_rows, systems=ALL_SYSTEMS),
            "projects_only": build_scope_run_summaries(all_rows, systems=PROJECT_SYSTEMS),
        },
        "transfer": transfer,
        "campaign_state": campaign_state_payload,
    }


def render_dashboard_html(payload: dict[str, Any]) -> str:
    all_systems = list(ALL_SYSTEMS)
    project_systems = list(PROJECT_SYSTEMS)
    summary_count = int(payload["summary_count"])
    budgets = ", ".join(str(value) for value in payload["budgets"]) or "none"
    packs = ", ".join(payload["packs"]) or "none"
    lane_counts = payload["lane_counts"]
    transfer = payload.get("transfer") or {"case_count": 0, "cases": [], "family_rows": [], "regimes": {}}
    campaign_state = payload.get("campaign_state") or {
        "available": False,
        "workspace_kind": None,
        "case_count": 0,
        "status_counts": {},
        "resumed_case_count": 0,
        "integrity_failed_count": 0,
        "stop_requested": False,
        "cases": [],
    }

    parts = [
        "<!doctype html>",
        "<html lang='en'>",
        "<head>",
        "<meta charset='utf-8'>",
        "<meta name='viewport' content='width=device-width, initial-scale=1'>",
        "<title>Fair Matrix Dashboard</title>",
        "<style>",
        _dashboard_css(),
        "</style>",
        "</head>",
        "<body>",
        "<div class='grain'></div>",
        "<main class='shell'>",
        "<section class='hero'>",
        "<p class='eyebrow'>EvoNN Compare</p>",
        "<h1>Fair Matrix Dashboard</h1>",
        "<p class='lede'>Static leaderboard and run inspector for fair-matrix workspaces. "
        "Aggregate evidence sits above explicit seed-by-seed snapshots so project-only rows can recalculate winners "
        "without contenders while still preserving raw variance.</p>",
        "<div class='meta-strip'>",
        f"<span><strong>Summaries</strong> {summary_count}</span>",
        f"<span><strong>Budgets</strong> {html.escape(budgets)}</span>",
        f"<span><strong>Packs</strong> {html.escape(packs)}</span>",
        f"<span><strong>Generated</strong> {html.escape(payload['generated_at'])}</span>",
        "</div>",
        "</section>",
        "<section class='cards'>",
        _stat_card("Contract-Fair", str(lane_counts["contract_fair"])),
        _stat_card("Trusted Core", str(lane_counts["trusted_core"])),
        _stat_card("Trusted Extended", str(lane_counts["trusted_extended"])),
        _stat_card("Repeatable", str(lane_counts["repeatable"])),
        _stat_card("Runs Loaded", str(summary_count)),
        "</section>",
        "<section class='panel'>",
        "<h2>Campaign State</h2>",
        "<p>Workspace state is the live orchestration surface. It stays meaningful before summary JSON exists, so interrupted or in-flight cases can be resumed without guessing from partial artifacts.</p>",
        _campaign_state_overview(campaign_state),
        _campaign_case_table(campaign_state),
        "</section>",
        "<section class='panel'>",
        "<h2>How To Read This</h2>",
        "<p><strong>Operating State</strong> is the lane-level trust label. <strong>reference-only</strong> means fairness or accounting caveats remain. <strong>contract-fair</strong> means the lane is structurally fair but not yet benchmark-complete for the core systems. <strong>portable-contract-fair</strong> means the lane is fair on a portable fallback boundary and must not be read as native MLX truth. <strong>portable-transfer-plumbing</strong> means the seeded lane proves portable seeding/export plumbing only, not native transfer behavior. <strong>trusted-core</strong> and <strong>trusted-extended</strong> add benchmark-complete coverage for the quarter-critical core and then the secondary challengers.</p>",
        "<p><strong>System States</strong> adds per-system detail inside a lane. Contenders may report <strong>benchmark-complete-optional-skips</strong> when the ratified sklearn-backed floor ran cleanly and optional boosted or torch breadth extras were unavailable.</p>",
        "<p><strong>Solo Wins</strong> means a system was uniquely best on a benchmark in the chosen scope. "
        "<strong>Shared Wins</strong> means a tie for best. <strong>Benchmark Failures</strong> and "
        "<strong>Missing Results</strong> are per-system outcome counts, not lane-level fairness failures.</p>",
        "<p><strong>Aggregate Evidence</strong> reports mean score, spread, and pairwise seed deltas across repeated seeds. "
        "<strong>Per-Seed Snapshots</strong> keep each seed's scoreboard visible so noisy wins are not mistaken for stable ones.</p>",
        "</section>",
        "<section class='panel'>",
        "<h2>Overall Leaderboard: All 5 Systems</h2>",
        _leaderboard_table(payload["leaderboards"]["all_systems"], systems=all_systems),
        "</section>",
        "<section class='panel'>",
        "<h2>Aggregate Evidence: All 5 Systems</h2>",
        _multi_seed_table(payload["multi_seed"]["all_systems"]),
        "<h3>Pairwise Seed Score Deltas: All 5 Systems</h3>",
        _pairwise_seed_table(payload["multi_seed"]["all_systems"]),
        "</section>",
        "<section class='panel'>",
        "<h2>Per-Seed Aggregate Snapshots: All 5 Systems</h2>",
        _seed_scorecard_table(payload["seed_scorecards"]["all_systems"], systems=all_systems),
        "</section>",
        "<section class='panel'>",
        "<h2>Overall Leaderboard: Projects Only</h2>",
        _leaderboard_table(payload["leaderboards"]["projects_only"], systems=project_systems),
        "</section>",
        "<section class='panel'>",
        "<h2>Aggregate Evidence: Projects Only</h2>",
        _multi_seed_table(payload["multi_seed"]["projects_only"]),
        "<h3>Pairwise Seed Score Deltas: Projects Only</h3>",
        _pairwise_seed_table(payload["multi_seed"]["projects_only"]),
        "</section>",
        "<section class='panel'>",
        "<h2>Per-Seed Aggregate Snapshots: Projects Only</h2>",
        _seed_scorecard_table(payload["seed_scorecards"]["projects_only"], systems=project_systems),
        "</section>",
        "<section class='panel'>",
        "<h2>Engine Rank By Benchmark Family: All 5 Systems</h2>",
        "<p>Family-ranked views are the specialization surface: they show whether an engine is winning where its search thesis should matter instead of only on the aggregate board.</p>",
        _family_leaderboard_table(payload["specialization"]["family_leaderboards"]["all_systems"]),
        "</section>",
        "<section class='panel'>",
        "<h2>Engine Rank By Benchmark Family: Projects Only</h2>",
        _family_leaderboard_table(payload["specialization"]["family_leaderboards"]["projects_only"]),
        "</section>",
        "<section class='panel'>",
        "<h2>Engine Profiles: All 5 Systems</h2>",
        _engine_profile_table(payload["specialization"]["engine_profiles"]["all_systems"]),
        "</section>",
        "<section class='panel'>",
        "<h2>Engine Profiles: Projects Only</h2>",
        _engine_profile_table(payload["specialization"]["engine_profiles"]["projects_only"]),
        "</section>",
        "<section class='panel'>",
        "<h2>Detailed Per-Run Table: All 5 Systems</h2>",
        _run_scope_table(payload["runs"], scope_key="all_scope", systems=all_systems),
        "</section>",
        "<section class='panel'>",
        "<h2>Detailed Per-Run Table: Projects Only</h2>",
        _run_scope_table(payload["runs"], scope_key="project_scope", systems=project_systems),
        "</section>",
    ]
    if int(transfer.get("case_count", 0)) > 0:
        parts.extend(
            [
                "<section class='panel'>",
                "<h2>Transfer Regime Overview</h2>",
                "<p>Seeded regimes stay separate from the no-seed control so the dashboard can show where transfer helped, regressed, or stayed inconclusive instead of collapsing everything into one seeded bucket.</p>",
                _transfer_overview_table(transfer),
                "</section>",
                "<section class='panel'>",
                "<h2>Transfer Delta By Benchmark Family</h2>",
                _transfer_family_table(transfer),
                "</section>",
                "<section class='panel'>",
                "<h2>Seed Source Quality vs Downstream Effect</h2>",
                "<p>Each row links back to the transfer case JSON plus the seed gate and artifact JSON so provenance can be inspected from the dashboard instead of reconstructed manually from the workspace.</p>",
                _transfer_case_table(transfer),
                "</section>",
            ]
        )
    parts.extend(
        [
        "<section class='panel'>",
        "<h2>Recent Runs</h2>",
        _recent_runs_table(payload["runs"]),
        "</section>",
        "</main>",
        "</body>",
        "</html>",
        ]
    )
    return "\n".join(parts)


def _build_run_entry(*, path: Path, payload: dict[str, Any], output_path: Path) -> dict[str, Any]:
    lane = payload.get("lane") or {}
    trend_rows = list(payload.get("trend_rows") or [])
    baseline_context = dict(payload.get("baseline_context") or {})
    output_parent = output_path.parent.resolve()
    summary_md_path = path.with_name("fair_matrix_summary.md")
    report_dir = path.parent
    operating_state = _coerce_operating_state(lane)
    comparison_label = _comparison_label(trend_rows, baseline_context=baseline_context)
    comparison_cohort = _comparison_cohort(trend_rows, baseline_context=baseline_context)
    return {
        "comparison_label": comparison_label,
        "comparison_cohort": comparison_cohort,
        "pack_name": str(payload["pack_name"]),
        "budget": int(lane.get("expected_budget") or _infer_budget(payload)),
        "seed": int(lane.get("expected_seed") or _infer_seed(payload)),
        "summary_json_path": str(path.resolve()),
        "summary_md_path": _relative_path(summary_md_path, output_parent),
        "report_dir": _relative_path(report_dir, output_parent),
        "lane": {
            "operating_state": operating_state,
            "fairness_ok": bool(lane.get("fairness_ok")),
            "repeatability_ready": bool(lane.get("repeatability_ready")),
            "artifact_completeness_ok": bool(lane.get("artifact_completeness_ok")),
            "budget_consistency_ok": bool(lane.get("budget_consistency_ok")),
            "seed_consistency_ok": bool(lane.get("seed_consistency_ok")),
            "task_coverage_ok": bool(lane.get("task_coverage_ok")),
            "budget_accounting_ok": bool(lane.get("budget_accounting_ok")),
            "core_systems_complete_ok": bool(lane.get("core_systems_complete_ok")),
            "extended_systems_complete_ok": bool(lane.get("extended_systems_complete_ok")),
            "system_operating_states": dict(lane.get("system_operating_states") or {}),
        },
        "baseline_context": baseline_context,
        "trend_rows": trend_rows,
        "system_seeding": _system_seeding_summary(trend_rows),
        "all_scope": _scope_summary(trend_rows, systems=ALL_SYSTEMS),
        "project_scope": _scope_summary(trend_rows, systems=PROJECT_SYSTEMS),
    }


def _build_campaign_state_payload(campaign_state: dict[str, Any] | None, *, output_path: Path) -> dict[str, Any]:
    if campaign_state is None:
        return {
            "available": False,
            "workspace_kind": None,
            "case_count": 0,
            "status_counts": {},
            "resumed_case_count": 0,
            "integrity_failed_count": 0,
            "stop_requested": False,
            "stop_requested_at": None,
            "cases": [],
        }
    output_parent = output_path.parent.resolve()
    cases = []
    status_counts: dict[str, int] = {}
    resumed_case_count = 0
    integrity_failed_count = 0
    for case in sorted(
        list(campaign_state.get("cases") or []),
        key=lambda item: (int(item.get("budget", 0)), int(item.get("seed", 0)), str(item.get("pack_name", ""))),
    ):
        status = str(case.get("status") or "pending")
        status_counts[status] = status_counts.get(status, 0) + 1
        resume_count = int(case.get("resume_count", 0))
        if resume_count > 0 or bool(case.get("latest_attempt_resumed")):
            resumed_case_count += 1
        integrity_ok = case.get("artifact_integrity_ok")
        if integrity_ok is False:
            integrity_failed_count += 1
        cases.append(
            {
                "case_id": str(case.get("case_id")),
                "pack_name": str(case.get("pack_name")),
                "budget": int(case.get("budget", 0)),
                "seed": int(case.get("seed", 0)),
                "status": status,
                "current_stage": case.get("current_stage"),
                "attempts": int(case.get("attempts", 0)),
                "resume_count": resume_count,
                "latest_attempt_resumed": bool(case.get("latest_attempt_resumed", False)),
                "artifact_integrity_ok": integrity_ok,
                "integrity_issues": list(case.get("integrity_issues") or []),
                "last_error": case.get("last_error"),
                "report_dir": (
                    _relative_path(Path(str(case["report_dir"])), output_parent)
                    if case.get("report_dir")
                    else None
                ),
            }
        )
    return {
        "available": True,
        "workspace_kind": campaign_state.get("workspace_kind"),
        "state_path": (
            _relative_path(Path(str(campaign_state["state_path"])), output_parent)
            if campaign_state.get("state_path")
            else None
        ),
        "manifest_path": (
            _relative_path(Path(str(campaign_state["manifest_path"])), output_parent)
            if campaign_state.get("manifest_path")
            else None
        ),
        "case_count": len(cases),
        "status_counts": status_counts,
        "resumed_case_count": resumed_case_count,
        "integrity_failed_count": integrity_failed_count,
        "stop_requested": bool(campaign_state.get("stop_requested")),
        "stop_requested_at": campaign_state.get("stop_requested_at"),
        "cases": cases,
    }


def _scope_summary(trend_rows: list[dict[str, Any]], *, systems: tuple[str, ...]) -> dict[str, Any]:
    return build_scope_summary(trend_rows, systems=systems)


def _aggregate_leaderboard(runs: list[dict[str, Any]], *, scope_key: str, systems: tuple[str, ...]) -> list[dict[str, Any]]:
    totals = {
        system: {
            "system": system,
            "runs": 0,
            "solo_wins": 0,
            "shared_wins": 0,
            "benchmark_failures": 0,
            "missing_results": 0,
        }
        for system in systems
    }
    for run in runs:
        scope = run[scope_key]
        row_map = {entry["system"]: entry for entry in scope["rows"]}
        for system in systems:
            total = totals[system]
            row = row_map[system]
            total["runs"] += 1
            total["solo_wins"] += int(row["solo_wins"])
            total["shared_wins"] += int(row["shared_wins"])
            total["benchmark_failures"] += int(row["benchmark_failures"])
            total["missing_results"] += int(row["missing_results"])
    leaderboard = []
    for system in systems:
        row = totals[system]
        row["score"] = round(float(row["solo_wins"]) + 0.5 * float(row["shared_wins"]), 2)
        leaderboard.append(row)
    return sorted(
        leaderboard,
        key=lambda item: (-item["score"], -item["solo_wins"], item["benchmark_failures"], item["system"]),
    )


def _build_transfer_dashboard_payload(summary_paths: list[Path], *, output_path: Path) -> dict[str, Any]:
    case_paths = _discover_transfer_case_summaries(summary_paths)
    cases = [
        _build_transfer_case_entry(path=path, payload=json.loads(path.read_text(encoding="utf-8")), output_path=output_path)
        for path in case_paths
    ]
    cases.sort(key=lambda item: (item["regime"], item["seed"], item["summary_json_path"]))
    regimes = {
        regime: _aggregate_transfer_cases(regime=regime, cases=[case for case in cases if case["regime"] == regime])
        for regime in ("direct", "staged")
    }
    family_rows = [
        row
        for regime in ("direct", "staged")
        for row in regimes[regime]["family_rows"]
    ]
    return {
        "case_count": len(cases),
        "cases": cases,
        "regimes": regimes,
        "family_rows": family_rows,
    }


def _discover_transfer_case_summaries(summary_paths: list[Path]) -> list[Path]:
    workspace_roots: dict[Path, None] = {}
    for path in summary_paths:
        workspace_root = _workspace_root_from_summary(path)
        if workspace_root is not None:
            workspace_roots[workspace_root] = None

    found: dict[Path, None] = {}
    for workspace_root in workspace_roots:
        reports_dir = workspace_root / "reports"
        if not reports_dir.exists():
            continue
        for path in reports_dir.rglob("*_vs_control.json"):
            found[path.resolve()] = None
    return sorted(found)


def _workspace_root_from_summary(path: Path) -> Path | None:
    current = path.resolve()
    for parent in current.parents:
        if parent.name == "reports":
            return parent.parent.resolve()
    return None


def _build_transfer_case_entry(*, path: Path, payload: dict[str, Any], output_path: Path) -> dict[str, Any]:
    output_parent = output_path.parent.resolve()
    benchmark_deltas = list(payload.get("benchmark_deltas") or [])
    gain_count = int(payload.get("gain_count") or sum(1 for row in benchmark_deltas if row.get("outcome") == "gain"))
    regression_count = int(payload.get("regression_count") or sum(1 for row in benchmark_deltas if row.get("outcome") == "regression"))
    tie_count = int(payload.get("tie_count") or sum(1 for row in benchmark_deltas if row.get("outcome") == "tie"))
    other_count = int(
        payload.get("other_count")
        or sum(1 for row in benchmark_deltas if row.get("outcome") not in {"gain", "regression", "tie"})
    )
    numeric_deltas = [
        float(row["regime_delta"])
        for row in benchmark_deltas
        if row.get("regime_delta") is not None
    ]
    seed_quality = dict(payload.get("seed_quality") or {})
    provenance = {
        "summary_json_path": _relative_path(path.resolve(), output_parent),
        "summary_md_path": _relative_path(path.with_suffix(".md"), output_parent),
        "seed_gate_json_path": _optional_relative_path(seed_quality.get("gate_path"), output_parent),
        "seed_artifact_json_path": _optional_relative_path(seed_quality.get("artifact_path") or payload.get("seed_artifact"), output_parent),
    }
    return {
        "regime": str(payload["regime"]),
        "pack_name": str(payload["pack_name"]),
        "seed": int(payload["seed"]),
        "portable_backend": payload.get("portable_backend"),
        "seed_source": str(payload.get("seed_source") or "---"),
        "seed_overlap_policy": str(payload.get("seed_overlap_policy") or "unknown"),
        "verdict": str(payload.get("verdict") or "inconclusive"),
        "regime_wins": int(payload.get("regime_wins") or 0),
        "control_wins": int(payload.get("control_wins") or 0),
        "ties": int(payload.get("ties") or 0),
        "gain_count": gain_count,
        "regression_count": regression_count,
        "tie_count": tie_count,
        "other_count": other_count,
        "mean_regime_delta": None if not numeric_deltas else sum(numeric_deltas) / len(numeric_deltas),
        "benchmark_families": sorted({str(row.get("benchmark_family") or "unknown") for row in benchmark_deltas}),
        "benchmark_deltas": benchmark_deltas,
        "seed_quality": {
            "candidate_count": seed_quality.get("candidate_count"),
            "family": seed_quality.get("family"),
            "benchmark_groups": list(seed_quality.get("benchmark_groups") or []),
            "benchmark_wins": seed_quality.get("benchmark_wins"),
            "repeat_support_count": seed_quality.get("repeat_support_count"),
            "median_quality": seed_quality.get("median_quality"),
            "overlap_policy": seed_quality.get("overlap_policy"),
            "representative_genome_id": seed_quality.get("representative_genome_id"),
            "representative_architecture_summary": seed_quality.get("representative_architecture_summary"),
        },
        "summary_json_path": provenance["summary_json_path"],
        "provenance": provenance,
    }


def _aggregate_transfer_cases(*, regime: str, cases: list[dict[str, Any]]) -> dict[str, Any]:
    verdict_counts = defaultdict(int)
    family_totals: dict[str, dict[str, Any]] = {}
    benchmark_gain_count = 0
    benchmark_regression_count = 0
    benchmark_tie_count = 0
    benchmark_other_count = 0
    total_regime_wins = 0
    total_control_wins = 0
    total_ties = 0
    for case in cases:
        verdict_counts[str(case["verdict"])] += 1
        total_regime_wins += int(case["regime_wins"])
        total_control_wins += int(case["control_wins"])
        total_ties += int(case["ties"])
        benchmark_gain_count += int(case["gain_count"])
        benchmark_regression_count += int(case["regression_count"])
        benchmark_tie_count += int(case["tie_count"])
        benchmark_other_count += int(case["other_count"])
        for row in case["benchmark_deltas"]:
            family = str(row.get("benchmark_family") or "unknown")
            total = family_totals.setdefault(
                family,
                {
                    "regime": regime,
                    "benchmark_family": family,
                    "case_count": 0,
                    "benchmark_count": 0,
                    "gain_count": 0,
                    "regression_count": 0,
                    "tie_count": 0,
                    "other_count": 0,
                    "_delta_sum": 0.0,
                    "_delta_count": 0,
                },
            )
            total["case_count"] += 1
            total["benchmark_count"] += 1
            outcome = str(row.get("outcome") or "other")
            if outcome == "gain":
                total["gain_count"] += 1
            elif outcome == "regression":
                total["regression_count"] += 1
            elif outcome == "tie":
                total["tie_count"] += 1
            else:
                total["other_count"] += 1
            if row.get("regime_delta") is not None:
                total["_delta_sum"] += float(row["regime_delta"])
                total["_delta_count"] += 1

    case_count = len(cases)
    if case_count == 0:
        consensus = "inconclusive"
    elif verdict_counts["gain"] == case_count:
        consensus = "gain"
    elif verdict_counts["regression"] == case_count:
        consensus = "regression"
    elif verdict_counts["no_gain"] == case_count:
        consensus = "no_gain"
    else:
        consensus = "inconclusive"

    family_rows = []
    for family in sorted(family_totals):
        total = family_totals[family]
        family_rows.append(
            {
                "regime": regime,
                "benchmark_family": family,
                "case_count": total["case_count"],
                "benchmark_count": total["benchmark_count"],
                "gain_count": total["gain_count"],
                "regression_count": total["regression_count"],
                "tie_count": total["tie_count"],
                "other_count": total["other_count"],
                "mean_regime_delta": None
                if total["_delta_count"] == 0
                else total["_delta_sum"] / total["_delta_count"],
            }
        )

    numeric_case_deltas = [
        float(case["mean_regime_delta"])
        for case in cases
        if case.get("mean_regime_delta") is not None
    ]
    return {
        "regime": regime,
        "case_count": case_count,
        "consensus": consensus,
        "verdict_counts": {
            "gain": verdict_counts["gain"],
            "no_gain": verdict_counts["no_gain"],
            "regression": verdict_counts["regression"],
            "inconclusive": verdict_counts["inconclusive"],
        },
        "total_regime_wins": total_regime_wins,
        "total_control_wins": total_control_wins,
        "total_ties": total_ties,
        "benchmark_gain_count": benchmark_gain_count,
        "benchmark_regression_count": benchmark_regression_count,
        "benchmark_tie_count": benchmark_tie_count,
        "benchmark_other_count": benchmark_other_count,
        "mean_regime_delta": None if not numeric_case_deltas else sum(numeric_case_deltas) / len(numeric_case_deltas),
        "family_rows": family_rows,
    }


def _campaign_state_overview(campaign_state: dict[str, Any]) -> str:
    if not campaign_state.get("available"):
        return "<p>No workspace state file was found.</p>"
    status_counts = campaign_state.get("status_counts") or {}
    rendered_counts = ", ".join(
        f"{status}={count}"
        for status, count in sorted(status_counts.items())
    ) or "none"
    lines = [
        "<ul class='meta-list'>",
        f"<li><strong>Workspace Kind</strong> {html.escape(str(campaign_state.get('workspace_kind') or 'unknown'))}</li>",
        f"<li><strong>State File</strong> {html.escape(str(campaign_state.get('state_path') or 'unknown'))}</li>",
        f"<li><strong>Manifest</strong> {html.escape(str(campaign_state.get('manifest_path') or 'unknown'))}</li>",
        f"<li><strong>Cases</strong> {int(campaign_state.get('case_count', 0))}</li>",
        f"<li><strong>Status Counts</strong> {html.escape(rendered_counts)}</li>",
        f"<li><strong>Resumed Cases</strong> {int(campaign_state.get('resumed_case_count', 0))}</li>",
        f"<li><strong>Integrity Failures</strong> {int(campaign_state.get('integrity_failed_count', 0))}</li>",
        f"<li><strong>Stop Requested</strong> {'yes' if campaign_state.get('stop_requested') else 'no'}</li>",
        "</ul>",
    ]
    return "\n".join(lines)


def _campaign_case_table(campaign_state: dict[str, Any]) -> str:
    cases = list(campaign_state.get("cases") or [])
    if not cases:
        return "<p>No campaign cases recorded.</p>"
    lines = [
        "<table>",
        "<thead><tr><th>Pack</th><th>Budget</th><th>Seed</th><th>Status</th><th>Stage</th><th>Attempts</th><th>Resumes</th><th>Integrity</th><th>Error</th><th>Report Dir</th></tr></thead>",
        "<tbody>",
    ]
    for case in cases:
        integrity_ok = case.get("artifact_integrity_ok")
        if integrity_ok is None:
            integrity = "pending"
        else:
            integrity = "ok" if integrity_ok else "failed"
        report_dir = case.get("report_dir")
        report_cell = f"<a href='{html.escape(str(report_dir))}'>open</a>" if report_dir else "---"
        error_parts = []
        if case.get("last_error"):
            error_parts.append(str(case["last_error"]))
        issues = list(case.get("integrity_issues") or [])
        if issues:
            error_parts.append("; ".join(issues))
        lines.append(
            "<tr>"
            f"<td><code>{html.escape(str(case['pack_name']))}</code></td>"
            f"<td>{int(case['budget'])}</td>"
            f"<td>{int(case['seed'])}</td>"
            f"<td><span class='tag tag-{html.escape(str(case['status']))}'>{html.escape(str(case['status']))}</span></td>"
            f"<td>{html.escape(str(case.get('current_stage') or '---'))}</td>"
            f"<td>{int(case.get('attempts', 0))}</td>"
            f"<td>{int(case.get('resume_count', 0))}</td>"
            f"<td>{html.escape(integrity)}</td>"
            f"<td>{html.escape(' | '.join(error_parts) or '---')}</td>"
            f"<td>{report_cell}</td>"
            "</tr>"
        )
    lines.extend(["</tbody>", "</table>"])
    return "\n".join(lines)


def _leaderboard_table(rows: list[dict[str, Any]], *, systems: list[str]) -> str:
    lines = [
        "<table>",
        "<thead><tr><th>Rank</th><th>System</th><th>Score</th><th>Solo Wins</th><th>Shared Wins</th>"
        "<th>Runs</th><th>Benchmark Failures</th><th>Missing Results</th></tr></thead>",
        "<tbody>",
    ]
    for index, row in enumerate(rows, start=1):
        lines.append(
            "<tr>"
            f"<td>{index}</td>"
            f"<td>{html.escape(_titleize(row['system']))}</td>"
            f"<td>{row['score']:.2f}</td>"
            f"<td>{row['solo_wins']}</td>"
            f"<td>{row['shared_wins']}</td>"
            f"<td>{row['runs']}</td>"
            f"<td>{row['benchmark_failures']}</td>"
            f"<td>{row['missing_results']}</td>"
            "</tr>"
        )
    lines.extend(["</tbody>", "</table>"])
    return "\n".join(lines)


def _family_leaderboard_table(groups: list[dict[str, Any]]) -> str:
    if not groups:
        return "<p>No runs loaded.</p>"
    lines = [
        "<table>",
        "<thead><tr><th>Comparison</th><th>Pack</th><th>Budget</th><th>Family</th><th>Benchmarks</th><th>Seeds</th><th>Rank</th><th>System</th>"
        "<th>Mean Score</th><th>Total Score</th><th>Solo Wins</th><th>Shared Wins</th><th>Failures</th><th>Missing</th></tr></thead>",
        "<tbody>",
    ]
    for group in groups:
        for row in group["system_rows"]:
            lines.append(
                "<tr>"
                f"<td>{html.escape(str(group['comparison_label']))}</td>"
                f"<td><code>{html.escape(group['pack_name'])}</code></td>"
                f"<td>{group['budget']}</td>"
                f"<td>{html.escape(str(group['benchmark_family']))}</td>"
                f"<td>{group['benchmark_count']}</td>"
                f"<td>{group['seed_count']}</td>"
                f"<td>{row['rank']}</td>"
                f"<td>{html.escape(_titleize(row['system']))}</td>"
                f"<td>{_float_cell(row['mean_score'])}</td>"
                f"<td>{_float_cell(row['score'])}</td>"
                f"<td>{row['solo_wins']}</td>"
                f"<td>{row['shared_wins']}</td>"
                f"<td>{row['benchmark_failures']}</td>"
                f"<td>{row['missing_results']}</td>"
                "</tr>"
            )
    lines.extend(["</tbody>", "</table>"])
    return "\n".join(lines)


def _engine_profile_table(profiles: list[dict[str, Any]]) -> str:
    if not profiles:
        return "<p>No runs loaded.</p>"
    lines = [
        "<table>",
        "<thead><tr><th>Comparison</th><th>System</th><th>Search Style</th><th>Expected Evidence</th><th>Branch Review Lens</th>"
        "<th>Strongest Families</th><th>Weakest Families</th><th>Failure Patterns</th><th>Architecture Signals</th><th>Status Counts</th></tr></thead>",
        "<tbody>",
    ]
    for profile in profiles:
        status_counts = profile.get("status_counts") or {}
        lines.append(
            "<tr>"
            f"<td>{html.escape(str(profile.get('comparison_label') or 'current-workspace'))}</td>"
            f"<td>{html.escape(_titleize(str(profile['system'])))}</td>"
            f"<td>{html.escape(str(profile['search_style']))}</td>"
            f"<td>{html.escape(str(profile['expected_signal']))}</td>"
            f"<td>{html.escape(str(profile['branch_review_prompt']))}</td>"
            f"<td>{html.escape(_family_profile_text(profile.get('family_strengths') or []))}</td>"
            f"<td>{html.escape(_family_profile_text(profile.get('family_weaknesses') or []))}</td>"
            f"<td>{html.escape(_failure_pattern_text(profile.get('failure_patterns') or []))}</td>"
            f"<td>{html.escape(_architecture_text(profile.get('architecture_examples') or []))}</td>"
            f"<td>{html.escape(_status_count_text(status_counts))}</td>"
            "</tr>"
        )
    lines.extend(["</tbody>", "</table>"])
    return "\n".join(lines)


def _multi_seed_table(groups: list[dict[str, Any]]) -> str:
    if not groups:
        return "<p>No runs loaded.</p>"
    lines = [
        "<table>",
        "<thead><tr><th>Comparison</th><th>Pack</th><th>Budget</th><th>Seeds</th><th>System</th><th>Mean Score</th><th>Score SD</th>"
        "<th>Seed IDs</th><th>Score Range</th><th>Best</th><th>Worst</th><th>95% CI</th><th>Mean Solo Wins</th>"
        "<th>Mean Shared Wins</th><th>Mean Failures</th><th>Mean Missing</th></tr></thead>",
        "<tbody>",
    ]
    for group in groups:
        ordered_rows = sorted(
            group["system_rows"],
            key=lambda row: (-float(row["mean_score"]), row["system"]),
        )
        seed_ids = ", ".join(str(seed) for seed in group["seeds"])
        for row in ordered_rows:
            lines.append(
                "<tr>"
                f"<td>{html.escape(str(group.get('comparison_label') or 'current-workspace'))}</td>"
                f"<td><code>{html.escape(group['pack_name'])}</code></td>"
                f"<td>{group['budget']}</td>"
                f"<td>{group['seed_count']}</td>"
                f"<td>{html.escape(_titleize(row['system']))}</td>"
                f"<td>{_float_cell(row['mean_score'])}</td>"
                f"<td>{_float_cell(row['score_stddev'])}</td>"
                f"<td>{html.escape(seed_ids)}</td>"
                f"<td>{_float_cell(row['score_range'])}</td>"
                f"<td>{_float_cell(row['best_score'])}</td>"
                f"<td>{_float_cell(row['worst_score'])}</td>"
                f"<td>{_ci_cell(row['score_ci95_low'], row['score_ci95_high'])}</td>"
                f"<td>{_float_cell(row['mean_solo_wins'])}</td>"
                f"<td>{_float_cell(row['mean_shared_wins'])}</td>"
                f"<td>{_float_cell(row['mean_benchmark_failures'])}</td>"
                f"<td>{_float_cell(row['mean_missing_results'])}</td>"
                "</tr>"
            )
    lines.extend(["</tbody>", "</table>"])
    return "\n".join(lines)


def _seed_scorecard_table(runs: list[dict[str, Any]], *, systems: list[str]) -> str:
    if not runs:
        return "<p>No runs loaded.</p>"
    header_cells = "".join(f"<th>{html.escape(_titleize(system))}</th>" for system in systems)
    lines = [
        "<table>",
        "<thead><tr><th>Comparison</th><th>Pack</th><th>Budget</th><th>Seed</th><th>Lane State</th><th>Repeatability</th><th>Accounting</th>"
        f"{header_cells}<th>Ties</th><th>Skipped</th></tr></thead>",
        "<tbody>",
    ]
    for run in runs:
        scope = run["scope"]
        row_map = {entry["system"]: entry for entry in scope["rows"]}
        system_cells = []
        for system in systems:
            row = row_map[system]
            score = float(row["solo_wins"]) + 0.5 * float(row["shared_wins"])
            cell = f"{_float_cell(score)} score"
            meta = [f"{row['solo_wins']} solo", f"{row['shared_wins']} shared"]
            if row["benchmark_failures"]:
                meta.append(f"fail {row['benchmark_failures']}")
            if row["missing_results"]:
                meta.append(f"missing {row['missing_results']}")
            system_cells.append(f"<td>{html.escape(cell)}<span class='cell-meta'>{html.escape(', '.join(meta))}</span></td>")
        lines.append(
            "<tr>"
            f"<td>{html.escape(str(run.get('comparison_label') or 'current-workspace'))}</td>"
            f"<td><code>{html.escape(run['pack_name'])}</code></td>"
            f"<td>{run['budget']}</td>"
            f"<td>{run['seed']}</td>"
            f"<td><span class='tag tag-{html.escape(run['lane_operating_state'])}'>{html.escape(run['lane_operating_state'])}</span></td>"
            f"<td>{'ready' if run['repeatability_ready'] else 'not-ready'}</td>"
            f"<td>{'ok' if run['budget_accounting_ok'] else 'incomplete'}</td>"
            + "".join(system_cells)
            + f"<td>{scope['ties']}</td>"
            + f"<td>{scope['skipped']}</td>"
            + "</tr>"
        )
    lines.extend(["</tbody>", "</table>"])
    return "\n".join(lines)


def _pairwise_seed_table(groups: list[dict[str, Any]]) -> str:
    if not groups:
        return "<p>No runs loaded.</p>"
    lines = [
        "<table>",
        "<thead><tr><th>Comparison</th><th>Pack</th><th>Budget</th><th>Pair</th><th>Seeds</th><th>Left Better</th><th>Ties</th>"
        "<th>Right Better</th><th>Mean Delta</th><th>Delta SD</th><th>95% CI</th><th>Sign Test p</th></tr></thead>",
        "<tbody>",
    ]
    for group in groups:
        for row in group["pairwise"]:
            lines.append(
                "<tr>"
                f"<td>{html.escape(str(group.get('comparison_label') or 'current-workspace'))}</td>"
                f"<td><code>{html.escape(group['pack_name'])}</code></td>"
                f"<td>{group['budget']}</td>"
                f"<td>{html.escape(_titleize(row['left_system']))} vs {html.escape(_titleize(row['right_system']))}</td>"
                f"<td>{row['seed_count']}</td>"
                f"<td>{row['left_better']}</td>"
                f"<td>{row['ties']}</td>"
                f"<td>{row['right_better']}</td>"
                f"<td>{_float_cell(row['mean_score_delta'])}</td>"
                f"<td>{_float_cell(row['score_delta_stddev'])}</td>"
                f"<td>{_ci_cell(row['score_delta_ci95_low'], row['score_delta_ci95_high'])}</td>"
                f"<td>{_float_cell(row['sign_test_p_value'])}</td>"
                "</tr>"
            )
    lines.extend(["</tbody>", "</table>"])
    return "\n".join(lines)


def _run_scope_table(runs: list[dict[str, Any]], *, scope_key: str, systems: list[str]) -> str:
    header_cells = "".join(f"<th>{html.escape(_titleize(system))}</th>" for system in systems)
    lines = [
        "<table>",
        "<thead><tr><th>Comparison</th><th>Pack</th><th>Budget</th><th>Seed</th>"
        "<th>Operating State</th><th>Repeatable</th>"
        f"{header_cells}<th>Ties</th><th>Skipped Benchmarks</th><th>Summary</th></tr></thead>",
        "<tbody>",
    ]
    for run in runs:
        scope = run[scope_key]
        row_map = {entry["system"]: entry for entry in scope["rows"]}
        system_cells = []
        for system in systems:
            row = row_map[system]
            cell = f"{row['solo_wins']} solo / {row['shared_wins']} shared"
            meta = []
            if row["benchmark_failures"]:
                meta.append(f"fail {row['benchmark_failures']}")
            if row["missing_results"]:
                meta.append(f"missing {row['missing_results']}")
            suffix = f"<span class='cell-meta'>{html.escape(', '.join(meta))}</span>" if meta else ""
            system_cells.append(f"<td>{html.escape(cell)}{suffix}</td>")
        lane_tag = run["lane"]["operating_state"]
        lines.append(
            "<tr>"
            f"<td>{html.escape(str(run.get('comparison_label') or 'current-workspace'))}</td>"
            f"<td><code>{html.escape(run['pack_name'])}</code></td>"
            f"<td>{run['budget']}</td>"
            f"<td>{run['seed']}</td>"
            f"<td><span class='tag tag-{lane_tag}'>{lane_tag}</span></td>"
            f"<td>{'yes' if run['lane']['repeatability_ready'] else 'no'}</td>"
            + "".join(system_cells)
            + f"<td>{scope['ties']}</td>"
            + f"<td>{scope['skipped']}</td>"
            + f"<td><a href='{html.escape(run['summary_md_path'])}'>summary</a></td>"
            + "</tr>"
        )
    lines.extend(["</tbody>", "</table>"])
    return "\n".join(lines)


def _recent_runs_table(runs: list[dict[str, Any]]) -> str:
    lines = [
        "<table>",
        "<thead><tr><th>Comparison</th><th>Cohort</th><th>Pack</th><th>Budget</th><th>Seed</th><th>State</th><th>Fair</th><th>Repeatable</th>"
        "<th>Accounting</th><th>Core Complete</th><th>Extended Complete</th><th>System States</th><th>Seeding</th><th>Artifact Complete</th><th>Budget OK</th><th>Seed OK</th><th>Report Dir</th></tr></thead>",
        "<tbody>",
    ]
    for run in runs:
        lane = run["lane"]
        system_states = "; ".join(
            f"{system}={state}"
            for system, state in sorted((lane.get("system_operating_states") or {}).items())
        ) or "---"
        seeding_summary = _render_system_seeding(run.get("system_seeding") or {})
        lines.append(
            "<tr>"
            f"<td>{html.escape(str(run.get('comparison_label') or 'current-workspace'))}</td>"
            f"<td>{html.escape(str(run.get('comparison_cohort') or 'current-workspace'))}</td>"
            f"<td><code>{html.escape(run['pack_name'])}</code></td>"
            f"<td>{run['budget']}</td>"
            f"<td>{run['seed']}</td>"
            f"<td><span class='tag tag-{html.escape(lane['operating_state'])}'>{html.escape(lane['operating_state'])}</span></td>"
            f"<td>{'yes' if lane['fairness_ok'] else 'no'}</td>"
            f"<td>{'yes' if lane['repeatability_ready'] else 'no'}</td>"
            f"<td>{'yes' if lane['budget_accounting_ok'] else 'no'}</td>"
            f"<td>{'yes' if lane['core_systems_complete_ok'] else 'no'}</td>"
            f"<td>{'yes' if lane['extended_systems_complete_ok'] else 'no'}</td>"
            f"<td>{html.escape(system_states)}</td>"
            f"<td>{html.escape(seeding_summary)}</td>"
            f"<td>{'yes' if lane['artifact_completeness_ok'] else 'no'}</td>"
            f"<td>{'yes' if lane['budget_consistency_ok'] else 'no'}</td>"
            f"<td>{'yes' if lane['seed_consistency_ok'] else 'no'}</td>"
            f"<td><a href='{html.escape(run['report_dir'])}'>open</a></td>"
            "</tr>"
        )
    lines.extend(["</tbody>", "</table>"])
    return "\n".join(lines)


def _transfer_overview_table(transfer: dict[str, Any]) -> str:
    lines = [
        "<table>",
        "<thead><tr><th>Regime</th><th>Cases</th><th>Consensus</th><th>Verdict Counts</th><th>Benchmark Gains</th><th>Benchmark Regressions</th><th>Benchmark Ties</th><th>Mean Delta</th></tr></thead>",
        "<tbody>",
    ]
    for regime in ("direct", "staged"):
        row = (transfer.get("regimes") or {}).get(regime) or {}
        verdict_counts = row.get("verdict_counts") or {}
        lines.append(
            "<tr>"
            f"<td>{html.escape(regime.title())}</td>"
            f"<td>{int(row.get('case_count', 0))}</td>"
            f"<td>{html.escape(str(row.get('consensus', 'inconclusive')))}</td>"
            f"<td>gain={int(verdict_counts.get('gain', 0))}, no_gain={int(verdict_counts.get('no_gain', 0))}, regression={int(verdict_counts.get('regression', 0))}, inconclusive={int(verdict_counts.get('inconclusive', 0))}</td>"
            f"<td>{int(row.get('benchmark_gain_count', 0))}</td>"
            f"<td>{int(row.get('benchmark_regression_count', 0))}</td>"
            f"<td>{int(row.get('benchmark_tie_count', 0))}</td>"
            f"<td>{_optional_float_cell(row.get('mean_regime_delta'))}</td>"
            "</tr>"
        )
    lines.extend(["</tbody>", "</table>"])
    return "\n".join(lines)


def _transfer_family_table(transfer: dict[str, Any]) -> str:
    family_rows = list(transfer.get("family_rows") or [])
    if not family_rows:
        return "<p>No transfer case summaries found.</p>"
    lines = [
        "<table>",
        "<thead><tr><th>Regime</th><th>Benchmark Family</th><th>Cases</th><th>Benchmarks</th><th>Gains</th><th>Regressions</th><th>Ties</th><th>Other</th><th>Mean Delta</th></tr></thead>",
        "<tbody>",
    ]
    for row in family_rows:
        lines.append(
            "<tr>"
            f"<td>{html.escape(_titleize(str(row['regime'])))}</td>"
            f"<td>{html.escape(str(row['benchmark_family']))}</td>"
            f"<td>{int(row['case_count'])}</td>"
            f"<td>{int(row['benchmark_count'])}</td>"
            f"<td>{int(row['gain_count'])}</td>"
            f"<td>{int(row['regression_count'])}</td>"
            f"<td>{int(row['tie_count'])}</td>"
            f"<td>{int(row['other_count'])}</td>"
            f"<td>{_optional_float_cell(row.get('mean_regime_delta'))}</td>"
            "</tr>"
        )
    lines.extend(["</tbody>", "</table>"])
    return "\n".join(lines)


def _transfer_case_table(transfer: dict[str, Any]) -> str:
    cases = list(transfer.get("cases") or [])
    if not cases:
        return "<p>No transfer case summaries found.</p>"
    lines = [
        "<table>",
        "<thead><tr><th>Regime</th><th>Seed</th><th>Verdict</th><th>Source</th><th>Overlap</th><th>Gains</th><th>Regressions</th><th>Ties</th><th>Mean Delta</th><th>Candidate Count</th><th>Benchmark Wins</th><th>Repeat Support</th><th>Median Quality</th><th>Family</th><th>JSON Drill-Down</th></tr></thead>",
        "<tbody>",
    ]
    for case in cases:
        seed_quality = case.get("seed_quality") or {}
        lines.append(
            "<tr>"
            f"<td>{html.escape(_titleize(str(case['regime'])))}</td>"
            f"<td>{int(case['seed'])}</td>"
            f"<td>{html.escape(str(case['verdict']))}</td>"
            f"<td>{html.escape(str(case['seed_source']))}</td>"
            f"<td>{html.escape(str(case['seed_overlap_policy']))}</td>"
            f"<td>{int(case['gain_count'])}</td>"
            f"<td>{int(case['regression_count'])}</td>"
            f"<td>{int(case['tie_count'])}</td>"
            f"<td>{_optional_float_cell(case.get('mean_regime_delta'))}</td>"
            f"<td>{_optional_int_cell(seed_quality.get('candidate_count'))}</td>"
            f"<td>{_optional_int_cell(seed_quality.get('benchmark_wins'))}</td>"
            f"<td>{_optional_int_cell(seed_quality.get('repeat_support_count'))}</td>"
            f"<td>{_optional_float_cell(seed_quality.get('median_quality'))}</td>"
            f"<td>{html.escape(str(seed_quality.get('family') or '---'))}</td>"
            f"<td>{_provenance_links(case.get('provenance') or {})}</td>"
            "</tr>"
        )
    lines.extend(["</tbody>", "</table>"])
    return "\n".join(lines)


def _infer_budget(payload: dict[str, Any]) -> int:
    fair_rows = payload.get("fair_rows") or payload.get("reference_rows") or []
    if fair_rows:
        return int(fair_rows[0]["budget"])
    raise ValueError("cannot infer budget from fair-matrix summary payload")


def _infer_seed(payload: dict[str, Any]) -> int:
    fair_rows = payload.get("fair_rows") or payload.get("reference_rows") or []
    if fair_rows:
        return int(fair_rows[0]["seed"])
    raise ValueError("cannot infer seed from fair-matrix summary payload")


def _coerce_operating_state(lane: dict[str, Any]) -> str:
    state = lane.get("operating_state")
    if state:
        return str(state)
    if lane.get("repeatability_ready"):
        return "trusted-core"
    if (
        lane.get("fairness_ok")
        and lane.get("artifact_completeness_ok")
        and lane.get("budget_consistency_ok")
        and lane.get("seed_consistency_ok")
        and lane.get("task_coverage_ok")
    ):
        return "contract-fair"
    return "reference-only"


def _system_seeding_summary(trend_rows: list[dict[str, Any]]) -> dict[str, dict[str, str]]:
    summary: dict[str, dict[str, set[str]]] = defaultdict(
        lambda: {"mode": set(), "source": set(), "artifact": set(), "family": set()}
    )
    for row in trend_rows:
        system = str(row.get("system") or "unknown")
        fairness = row.get("fairness_metadata") or {}
        source_system = fairness.get("seed_source_system")
        source_run_id = fairness.get("seed_source_run_id")
        source = "---"
        if source_system is not None:
            source = str(source_system)
            if source_run_id:
                source = f"{source}:{source_run_id}"
        family = fairness.get("seed_selected_family") or fairness.get("seed_target_family") or "---"
        summary[system]["mode"].add(str(fairness.get("seeding_bucket") or "transfer-opaque"))
        summary[system]["source"].add(source)
        summary[system]["artifact"].add(str(fairness.get("seed_artifact_path") or "---"))
        summary[system]["family"].add(str(family))
    return {
        system: {
            key: ", ".join(sorted(values))
            for key, values in fields.items()
        }
        for system, fields in summary.items()
    }


def _render_system_seeding(system_seeding: dict[str, dict[str, str]]) -> str:
    if not system_seeding:
        return "none"
    parts = []
    for system, fields in sorted(system_seeding.items()):
        segment = f"{_titleize(system)}: {fields.get('mode', 'transfer-opaque')}"
        source = fields.get("source") or "---"
        family = fields.get("family") or "---"
        if source != "---":
            segment += f" from {source}"
        if family != "---":
            segment += f" ({family})"
        parts.append(segment)
    return "; ".join(parts)


def _comparison_label(trend_rows: list[dict[str, Any]], *, baseline_context: dict[str, Any]) -> str:
    if baseline_context.get("baseline_label"):
        return str(baseline_context["baseline_label"])
    for row in trend_rows:
        fairness = row.get("fairness_metadata") or {}
        value = fairness.get("comparison_label") or fairness.get("baseline_label")
        if value:
            return str(value)
    return "current-workspace"


def _comparison_cohort(trend_rows: list[dict[str, Any]], *, baseline_context: dict[str, Any]) -> str:
    if baseline_context.get("comparison_cohort"):
        return str(baseline_context["comparison_cohort"])
    for row in trend_rows:
        fairness = row.get("fairness_metadata") or {}
        value = fairness.get("comparison_cohort")
        if value:
            return str(value)
    return "current-workspace"


def _family_profile_text(rows: list[dict[str, Any]]) -> str:
    if not rows:
        return "none"
    return "; ".join(
        (
            f"{row['benchmark_family']} @ {row['pack_name']}:{row['budget']}"
            f" rank {row['rank']}/{row['system_count']}"
            f" mean {_float_cell(float(row['mean_score']))}"
            f" fail {row['benchmark_failures']}"
            f" missing {row['missing_results']}"
        )
        for row in rows
    )


def _failure_pattern_text(rows: list[dict[str, Any]]) -> str:
    if not rows:
        return "none"
    return "; ".join(f"{row['reason']} ({row['count']})" for row in rows)


def _architecture_text(rows: list[dict[str, Any]]) -> str:
    if not rows:
        return "none"
    return "; ".join(f"{row['summary']} ({row['count']})" for row in rows)


def _status_count_text(status_counts: dict[str, Any]) -> str:
    return (
        f"ok={int(status_counts.get('ok', 0))}, failed={int(status_counts.get('failed', 0))}, "
        f"skipped={int(status_counts.get('skipped', 0))}, unsupported={int(status_counts.get('unsupported', 0))}, "
        f"missing={int(status_counts.get('missing', 0))}"
    )


def _relative_path(target: Path, base_dir: Path) -> str:
    return os.path.relpath(target.resolve(), start=base_dir)


def _optional_relative_path(target: str | None, base_dir: Path) -> str | None:
    if not target:
        return None
    return _relative_path(Path(target), base_dir)


def _titleize(system: str) -> str:
    return system.replace("_", " ").title()


def _float_cell(value: float) -> str:
    return f"{float(value):.3f}"


def _optional_float_cell(value: float | None) -> str:
    if value is None:
        return "---"
    return _float_cell(float(value))


def _optional_int_cell(value: Any) -> str:
    if value is None:
        return "---"
    return str(int(value))


def _ci_cell(lower: float, upper: float) -> str:
    return f"{lower:.3f} to {upper:.3f}"


def _stat_card(label: str, value: str) -> str:
    return (
        "<article class='card'>"
        f"<p class='card-label'>{html.escape(label)}</p>"
        f"<p class='card-value'>{html.escape(value)}</p>"
        "</article>"
    )


def _provenance_links(provenance: dict[str, Any]) -> str:
    links = []
    for key, label in (
        ("summary_json_path", "summary json"),
        ("summary_md_path", "summary md"),
        ("seed_gate_json_path", "seed gate"),
        ("seed_artifact_json_path", "seed artifact"),
    ):
        path = provenance.get(key)
        if path:
            links.append(f"<a href='{html.escape(str(path))}'>{html.escape(label)}</a>")
    return " &middot; ".join(links) if links else "---"


def _dashboard_css() -> str:
    return """
:root {
  --bg: #f6f0e5;
  --panel: rgba(255, 252, 246, 0.92);
  --ink: #1d1a17;
  --muted: #5f554a;
  --accent: #9b3d23;
  --accent-soft: rgba(155, 61, 35, 0.1);
  --line: rgba(29, 26, 23, 0.12);
  --good: #1f6b52;
  --warn: #8f5b00;
  --shadow: 0 18px 60px rgba(47, 33, 20, 0.09);
}
* { box-sizing: border-box; }
body {
  margin: 0;
  background:
    radial-gradient(circle at top left, rgba(155, 61, 35, 0.14), transparent 28rem),
    radial-gradient(circle at top right, rgba(31, 107, 82, 0.12), transparent 24rem),
    var(--bg);
  color: var(--ink);
  font-family: "Iowan Old Style", "Palatino Linotype", "Book Antiqua", Palatino, serif;
}
.grain {
  position: fixed;
  inset: 0;
  pointer-events: none;
  opacity: 0.08;
  background-image:
    linear-gradient(rgba(29, 26, 23, 0.03) 1px, transparent 1px),
    linear-gradient(90deg, rgba(29, 26, 23, 0.03) 1px, transparent 1px);
  background-size: 14px 14px;
}
.shell {
  width: min(1500px, calc(100vw - 3rem));
  margin: 0 auto;
  padding: 2.5rem 0 4rem;
}
.hero {
  padding: 2rem 2.2rem;
  border: 1px solid var(--line);
  background: linear-gradient(180deg, rgba(255,255,255,0.75), rgba(255,255,255,0.55));
  box-shadow: var(--shadow);
}
.eyebrow, .card-label, .meta-strip, th, td, .tag, .cell-meta {
  font-family: "SFMono-Regular", Menlo, Consolas, monospace;
}
.eyebrow {
  margin: 0 0 0.6rem;
  letter-spacing: 0.16em;
  text-transform: uppercase;
  color: var(--accent);
  font-size: 0.82rem;
}
h1, h2 {
  font-weight: 600;
  letter-spacing: -0.03em;
}
h1 {
  margin: 0;
  font-size: clamp(2.3rem, 4vw, 4.6rem);
}
.lede {
  max-width: 58rem;
  font-size: 1.05rem;
  line-height: 1.65;
  color: var(--muted);
}
.meta-strip {
  display: flex;
  flex-wrap: wrap;
  gap: 0.8rem 1.3rem;
  margin-top: 1.2rem;
  color: var(--muted);
  font-size: 0.86rem;
}
.cards {
  display: grid;
  grid-template-columns: repeat(4, minmax(0, 1fr));
  gap: 1rem;
  margin: 1rem 0 1.3rem;
}
.card, .panel {
  border: 1px solid var(--line);
  background: var(--panel);
  box-shadow: var(--shadow);
}
.card {
  padding: 1rem 1.1rem;
}
.card-label {
  margin: 0;
  font-size: 0.77rem;
  text-transform: uppercase;
  letter-spacing: 0.08em;
  color: var(--muted);
}
.card-value {
  margin: 0.45rem 0 0;
  font-size: 2rem;
}
.panel {
  padding: 1.35rem 1.4rem;
  margin-top: 1rem;
  overflow-x: auto;
}
.panel p {
  color: var(--muted);
  line-height: 1.6;
}
.meta-list {
  margin: 0 0 1rem;
  padding-left: 1.1rem;
  color: var(--muted);
  line-height: 1.7;
}
table {
  width: 100%;
  border-collapse: collapse;
  min-width: 980px;
}
th, td {
  text-align: left;
  padding: 0.8rem 0.72rem;
  border-bottom: 1px solid var(--line);
  vertical-align: top;
  font-size: 0.82rem;
}
th {
  color: var(--muted);
  text-transform: uppercase;
  letter-spacing: 0.05em;
  font-size: 0.73rem;
}
tbody tr:hover {
  background: rgba(155, 61, 35, 0.04);
}
a {
  color: var(--accent);
  text-decoration: none;
}
a:hover { text-decoration: underline; }
code {
  font-family: "SFMono-Regular", Menlo, Consolas, monospace;
  font-size: 0.8rem;
}
.tag {
  display: inline-block;
  padding: 0.18rem 0.44rem;
  border: 1px solid var(--line);
  text-transform: uppercase;
  letter-spacing: 0.06em;
  font-size: 0.7rem;
}
.tag-contract-fair {
  background: rgba(31, 107, 82, 0.12);
  color: var(--good);
}
.tag-portable-contract-fair {
  background: rgba(18, 94, 112, 0.16);
  color: #155e75;
}
.tag-trusted-core {
  background: rgba(31, 107, 82, 0.18);
  color: var(--good);
}
.tag-trusted-extended {
  background: rgba(31, 107, 82, 0.24);
  color: var(--good);
}
.tag-portable-transfer-plumbing {
  background: rgba(12, 74, 110, 0.12);
  color: #0f4c81;
}
.tag-reference-only {
  background: rgba(143, 91, 0, 0.12);
  color: var(--warn);
}
.tag-pending {
  background: rgba(95, 85, 74, 0.12);
  color: var(--muted);
}
.tag-running {
  background: rgba(12, 74, 110, 0.12);
  color: #0f4c81;
}
.tag-interrupted {
  background: rgba(143, 91, 0, 0.16);
  color: var(--warn);
}
.tag-failed {
  background: rgba(155, 61, 35, 0.16);
  color: var(--accent);
}
.tag-completed {
  background: rgba(31, 107, 82, 0.12);
  color: var(--good);
}
.cell-meta {
  display: block;
  margin-top: 0.35rem;
  color: var(--muted);
  font-size: 0.7rem;
}
@media (max-width: 980px) {
  .shell { width: min(100vw - 1rem, 100%); padding-top: 1rem; }
  .cards { grid-template-columns: repeat(2, minmax(0, 1fr)); }
  .hero, .panel, .card { padding-left: 1rem; padding-right: 1rem; }
}
"""
