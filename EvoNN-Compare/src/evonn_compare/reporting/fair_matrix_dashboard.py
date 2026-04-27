"""Static dashboard renderer for accumulated fair-matrix summaries."""

from __future__ import annotations

from collections import defaultdict
from datetime import datetime, timezone
import html
import json
import os
from pathlib import Path
from typing import Any

ALL_SYSTEMS = ("prism", "topograph", "stratograph", "primordia", "contenders")
PROJECT_SYSTEMS = ("prism", "topograph", "stratograph", "primordia")


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


def build_dashboard_payload(summary_paths: list[Path], *, output_path: Path) -> dict[str, Any]:
    runs: list[dict[str, Any]] = []
    for path in summary_paths:
        payload = json.loads(path.read_text(encoding="utf-8"))
        runs.append(_build_run_entry(path=path, payload=payload, output_path=output_path))

    runs.sort(key=lambda item: (item["budget"], item["seed"], item["pack_name"], item["summary_json_path"]))
    return {
        "generated_at": datetime.now(timezone.utc).isoformat(),
        "summary_count": len(runs),
        "packs": sorted({item["pack_name"] for item in runs}),
        "budgets": sorted({item["budget"] for item in runs}),
        "lane_counts": {
            "fair": sum(1 for item in runs if item["lane"]["fairness_ok"]),
            "contract_fair": sum(1 for item in runs if item["lane"]["operating_state"] != "reference-only"),
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
    }


def render_dashboard_html(payload: dict[str, Any]) -> str:
    all_systems = list(ALL_SYSTEMS)
    project_systems = list(PROJECT_SYSTEMS)
    summary_count = int(payload["summary_count"])
    budgets = ", ".join(str(value) for value in payload["budgets"]) or "none"
    packs = ", ".join(payload["packs"]) or "none"
    lane_counts = payload["lane_counts"]

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
        "Project-only rows recalculate winners without contenders instead of merely hiding them.</p>",
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
        "<h2>How To Read This</h2>",
        "<p><strong>Operating State</strong> is the lane-level trust label. <strong>reference-only</strong> means fairness or accounting caveats remain. <strong>contract-fair</strong> means the lane is structurally fair but not yet benchmark-complete for the core systems. <strong>trusted-core</strong> and <strong>trusted-extended</strong> add benchmark-complete coverage for the quarter-critical core and then the secondary challengers.</p>",
        "<p><strong>Solo Wins</strong> means a system was uniquely best on a benchmark in the chosen scope. "
        "<strong>Shared Wins</strong> means a tie for best. <strong>Benchmark Failures</strong> and "
        "<strong>Missing Results</strong> are per-system outcome counts, not lane-level fairness failures.</p>",
        "</section>",
        "<section class='panel'>",
        "<h2>Overall Leaderboard: All 5 Systems</h2>",
        _leaderboard_table(payload["leaderboards"]["all_systems"], systems=all_systems),
        "</section>",
        "<section class='panel'>",
        "<h2>Overall Leaderboard: Projects Only</h2>",
        _leaderboard_table(payload["leaderboards"]["projects_only"], systems=project_systems),
        "</section>",
        "<section class='panel'>",
        "<h2>Per-Run Table: All 5 Systems</h2>",
        _run_scope_table(payload["runs"], scope_key="all_scope", systems=all_systems),
        "</section>",
        "<section class='panel'>",
        "<h2>Per-Run Table: Projects Only</h2>",
        _run_scope_table(payload["runs"], scope_key="project_scope", systems=project_systems),
        "</section>",
        "<section class='panel'>",
        "<h2>Recent Runs</h2>",
        _recent_runs_table(payload["runs"]),
        "</section>",
        "</main>",
        "</body>",
        "</html>",
    ]
    return "\n".join(parts)


def _build_run_entry(*, path: Path, payload: dict[str, Any], output_path: Path) -> dict[str, Any]:
    lane = payload.get("lane") or {}
    trend_rows = list(payload.get("trend_rows") or [])
    output_parent = output_path.parent.resolve()
    summary_md_path = path.with_name("fair_matrix_summary.md")
    report_dir = path.parent
    operating_state = _coerce_operating_state(lane)
    return {
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
        "all_scope": _scope_summary(trend_rows, systems=ALL_SYSTEMS),
        "project_scope": _scope_summary(trend_rows, systems=PROJECT_SYSTEMS),
    }


def _scope_summary(trend_rows: list[dict[str, Any]], *, systems: tuple[str, ...]) -> dict[str, Any]:
    by_benchmark: dict[str, list[dict[str, Any]]] = defaultdict(list)
    for row in trend_rows:
        system = str(row["system"])
        if system in systems:
            by_benchmark[str(row["benchmark_id"])].append(row)

    rows = {system: {"system": system, "solo_wins": 0, "shared_wins": 0, "benchmark_failures": 0, "missing_results": 0} for system in systems}
    ties = 0
    skipped = 0
    for benchmark_rows in by_benchmark.values():
        ok_rows = []
        for row in benchmark_rows:
            system_row = rows[str(row["system"])]
            status = str(row.get("outcome_status") or "missing")
            if status == "failed":
                system_row["benchmark_failures"] += 1
            if status == "missing":
                system_row["missing_results"] += 1
            if status == "ok" and row.get("metric_value") is not None:
                ok_rows.append(row)
        if not ok_rows:
            skipped += 1
            continue
        direction = str(ok_rows[0]["metric_direction"])
        values = [float(row["metric_value"]) for row in ok_rows]
        best_value = max(values) if direction == "max" else min(values)
        winners = [str(row["system"]) for row in ok_rows if abs(float(row["metric_value"]) - best_value) <= 1e-12]
        if len(winners) == 1:
            rows[winners[0]]["solo_wins"] += 1
        else:
            ties += 1
            for winner in winners:
                rows[winner]["shared_wins"] += 1
    ordered_rows = [rows[system] for system in systems]
    return {"rows": ordered_rows, "ties": ties, "skipped": skipped}


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


def _run_scope_table(runs: list[dict[str, Any]], *, scope_key: str, systems: list[str]) -> str:
    header_cells = "".join(f"<th>{html.escape(_titleize(system))}</th>" for system in systems)
    lines = [
        "<table>",
        "<thead><tr><th>Pack</th><th>Budget</th><th>Seed</th>"
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
        "<thead><tr><th>Pack</th><th>Budget</th><th>Seed</th><th>State</th><th>Fair</th><th>Repeatable</th>"
        "<th>Accounting</th><th>Core Complete</th><th>Extended Complete</th><th>Artifact Complete</th><th>Budget OK</th><th>Seed OK</th><th>Report Dir</th></tr></thead>",
        "<tbody>",
    ]
    for run in runs:
        lane = run["lane"]
        lines.append(
            "<tr>"
            f"<td><code>{html.escape(run['pack_name'])}</code></td>"
            f"<td>{run['budget']}</td>"
            f"<td>{run['seed']}</td>"
            f"<td><span class='tag tag-{html.escape(lane['operating_state'])}'>{html.escape(lane['operating_state'])}</span></td>"
            f"<td>{'yes' if lane['fairness_ok'] else 'no'}</td>"
            f"<td>{'yes' if lane['repeatability_ready'] else 'no'}</td>"
            f"<td>{'yes' if lane['budget_accounting_ok'] else 'no'}</td>"
            f"<td>{'yes' if lane['core_systems_complete_ok'] else 'no'}</td>"
            f"<td>{'yes' if lane['extended_systems_complete_ok'] else 'no'}</td>"
            f"<td>{'yes' if lane['artifact_completeness_ok'] else 'no'}</td>"
            f"<td>{'yes' if lane['budget_consistency_ok'] else 'no'}</td>"
            f"<td>{'yes' if lane['seed_consistency_ok'] else 'no'}</td>"
            f"<td><a href='{html.escape(run['report_dir'])}'>open</a></td>"
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


def _relative_path(target: Path, base_dir: Path) -> str:
    return os.path.relpath(target.resolve(), start=base_dir)


def _titleize(system: str) -> str:
    return system.replace("_", " ").title()


def _stat_card(label: str, value: str) -> str:
    return (
        "<article class='card'>"
        f"<p class='card-label'>{html.escape(label)}</p>"
        f"<p class='card-value'>{html.escape(value)}</p>"
        "</article>"
    )


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
.tag-trusted-core {
  background: rgba(31, 107, 82, 0.18);
  color: var(--good);
}
.tag-trusted-extended {
  background: rgba(31, 107, 82, 0.24);
  color: var(--good);
}
.tag-reference-only {
  background: rgba(143, 91, 0, 0.12);
  color: var(--warn);
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
