#!/usr/bin/env python3
"""Extract EvoNN compare run history into stable CSVs and dashboard data.

The script scans exported compare artifacts under ``EvoNN-Compare/manual_compare_runs``
and rewrites canonical CSV outputs with stable deduplication keys, so re-running
the extractor adds new runs without duplicating already-seen rows.
"""

from __future__ import annotations

import argparse
import csv
from datetime import datetime, timezone
import json
from pathlib import Path
import sys
from typing import Any


REPO_ROOT = Path(__file__).resolve().parents[1]
DEFAULT_RUNS_ROOT = REPO_ROOT / "EvoNN-Compare" / "manual_compare_runs"
DEFAULT_DASHBOARD_ROOT = REPO_ROOT / "dashboard" / "compare-history"
SYSTEM_ORDER = ("prism", "topograph", "stratograph", "contenders", "hybrid")


def parse_args(argv: list[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--runs-root",
        type=Path,
        default=DEFAULT_RUNS_ROOT,
        help="Root containing compare run artifacts. Default: EvoNN-Compare/manual_compare_runs",
    )
    parser.add_argument(
        "--dashboard-root",
        type=Path,
        default=DEFAULT_DASHBOARD_ROOT,
        help="Output root for CSV files and dashboard assets. Default: dashboard/compare-history",
    )
    return parser.parse_args(argv)


def main(argv: list[str] | None = None) -> int:
    args = parse_args(argv)
    runs_root = args.runs_root.resolve()
    dashboard_root = args.dashboard_root.resolve()
    data_root = dashboard_root / "data"
    data_root.mkdir(parents=True, exist_ok=True)

    if not runs_root.exists():
        print(f"[extract] runs root missing: {runs_root}", file=sys.stderr)
        return 1

    run_rows, score_rows, contender_rows = collect_run_history(runs_root)
    summary_rows, parity_rows = collect_matrix_history(runs_root)

    run_rows = dedupe_rows(run_rows, "source_run_key")
    score_rows = dedupe_rows(score_rows, "score_row_key")
    contender_rows = dedupe_rows(contender_rows, "contender_row_key")
    summary_rows = dedupe_rows(summary_rows, "summary_row_key")
    parity_rows = dedupe_rows(parity_rows, "parity_row_key")

    run_rows.sort(key=lambda row: (row["created_at"], row["system"], row["run_id"], row["source_manifest_relpath"]))
    score_rows.sort(
        key=lambda row: (
            row["created_at"],
            row["system"],
            as_int(row["budget_evaluation_count"]),
            row["seed"],
            row["benchmark_id"],
        )
    )
    contender_rows.sort(
        key=lambda row: (
            row["created_at"],
            row["run_system"],
            row["run_id"],
            row["benchmark_id"],
            row["contender_name"],
        )
    )
    summary_rows.sort(key=lambda row: (as_int(row["budget"]), as_int(row["seed"]), row["summary_kind"], row["source_summary_relpath"]))
    parity_rows.sort(key=lambda row: (as_int(row["budget"]), as_int(row["seed"]), row["pair_label"], row["source_summary_relpath"]))

    write_csv(data_root / "compare_history_runs.csv", run_rows)
    write_csv(data_root / "compare_history_scores.csv", score_rows)
    write_csv(data_root / "compare_history_contenders.csv", contender_rows)
    write_csv(data_root / "compare_history_matrix.csv", summary_rows)
    write_csv(data_root / "compare_history_parity.csv", parity_rows)

    write_dashboard_assets(
        dashboard_root=dashboard_root,
        run_rows=run_rows,
        score_rows=score_rows,
        summary_rows=summary_rows,
    )

    print(f"[extract] runs={len(run_rows)} scores={len(score_rows)} contenders={len(contender_rows)}")
    print(f"[extract] matrix_rows={len(summary_rows)} parity_rows={len(parity_rows)}")
    print(f"[extract] dashboard={dashboard_root / 'index.html'}")
    return 0


def collect_run_history(runs_root: Path) -> tuple[list[dict[str, Any]], list[dict[str, Any]], list[dict[str, Any]]]:
    run_rows: list[dict[str, Any]] = []
    score_rows: list[dict[str, Any]] = []
    contender_rows: list[dict[str, Any]] = []

    for manifest_path in sorted(runs_root.rglob("manifest.json")):
        if manifest_path.parts[-2:] == ("compare-history", "manifest.json"):
            continue
        results_path = manifest_path.with_name("results.json")
        if not results_path.exists():
            continue

        manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
        results = json.loads(results_path.read_text(encoding="utf-8"))
        if not isinstance(results, list):
            continue

        system = str(manifest.get("system") or infer_system_from_path(manifest_path))
        budget = manifest.get("budget") or {}
        fairness = manifest.get("fairness") or {}
        benchmark_specs = {item["benchmark_id"]: item for item in manifest.get("benchmarks", [])}
        source_run_key = build_source_run_key(manifest_path, manifest)

        run_rows.append(
            {
                "source_run_key": source_run_key,
                "source_manifest_relpath": relpath(manifest_path),
                "source_results_relpath": relpath(results_path),
                "workspace_relpath": relpath(workspace_root_for_manifest(manifest_path)),
                "system": system,
                "version": manifest.get("version"),
                "run_id": manifest.get("run_id"),
                "run_name": manifest.get("run_name"),
                "created_at": normalize_iso(manifest.get("created_at")),
                "pack_name": manifest.get("pack_name"),
                "seed": manifest.get("seed"),
                "benchmark_count": len(manifest.get("benchmarks", [])),
                "budget_evaluation_count": budget.get("evaluation_count"),
                "budget_epochs_per_candidate": budget.get("epochs_per_candidate"),
                "budget_generations": budget.get("generations"),
                "budget_population_size": budget.get("population_size"),
                "budget_wall_clock_seconds": budget.get("wall_clock_seconds"),
                "budget_policy_name": budget.get("budget_policy_name"),
                "fairness_benchmark_pack_id": fairness.get("benchmark_pack_id"),
                "fairness_seed": fairness.get("seed"),
                "fairness_evaluation_count": fairness.get("evaluation_count"),
                "fairness_budget_policy_name": fairness.get("budget_policy_name"),
                "fairness_data_signature": fairness.get("data_signature"),
                "fairness_code_version": fairness.get("code_version"),
            }
        )

        for record in results:
            benchmark_id = str(record.get("benchmark_id"))
            spec = benchmark_specs.get(benchmark_id, {})
            score_row_key = f"{source_run_key}|{benchmark_id}"
            score_rows.append(
                {
                    "score_row_key": score_row_key,
                    "source_run_key": source_run_key,
                    "source_manifest_relpath": relpath(manifest_path),
                    "source_results_relpath": relpath(results_path),
                    "workspace_relpath": relpath(workspace_root_for_manifest(manifest_path)),
                    "system": system,
                    "run_id": manifest.get("run_id"),
                    "run_name": manifest.get("run_name"),
                    "created_at": normalize_iso(manifest.get("created_at")),
                    "pack_name": manifest.get("pack_name"),
                    "seed": manifest.get("seed"),
                    "benchmark_id": benchmark_id,
                    "task_kind": spec.get("task_kind"),
                    "metric_name": record.get("metric_name") or spec.get("metric_name"),
                    "metric_direction": record.get("metric_direction") or spec.get("metric_direction"),
                    "metric_value": record.get("metric_value"),
                    "quality": record.get("quality"),
                    "status": record.get("status"),
                    "failure_reason": record.get("failure_reason"),
                    "parameter_count": record.get("parameter_count"),
                    "train_seconds": record.get("train_seconds"),
                    "peak_memory_mb": record.get("peak_memory_mb"),
                    "architecture_summary": record.get("architecture_summary"),
                    "genome_id": record.get("genome_id"),
                    "family": record.get("family"),
                    "budget_evaluation_count": budget.get("evaluation_count"),
                    "budget_epochs_per_candidate": budget.get("epochs_per_candidate"),
                    "budget_generations": budget.get("generations"),
                    "budget_population_size": budget.get("population_size"),
                    "budget_wall_clock_seconds": budget.get("wall_clock_seconds"),
                    "budget_policy_name": budget.get("budget_policy_name"),
                    "fairness_data_signature": fairness.get("data_signature"),
                    "fairness_code_version": fairness.get("code_version"),
                }
            )

        contender_summary_path = manifest_path.with_name("contender_summary.json")
        if contender_summary_path.exists():
            contender_summary = json.loads(contender_summary_path.read_text(encoding="utf-8"))
            for index, record in enumerate(contender_summary):
                benchmark_name = str(record.get("benchmark_name"))
                contender_row_key = f"{source_run_key}|{benchmark_name}|{record.get('contender_name')}|{record.get('contender_id')}|{index}"
                contender_rows.append(
                    {
                        "contender_row_key": contender_row_key,
                        "source_run_key": source_run_key,
                        "source_contender_summary_relpath": relpath(contender_summary_path),
                        "workspace_relpath": relpath(workspace_root_for_manifest(manifest_path)),
                        "run_system": system,
                        "run_id": manifest.get("run_id"),
                        "created_at": normalize_iso(manifest.get("created_at")),
                        "pack_name": manifest.get("pack_name"),
                        "seed": manifest.get("seed"),
                        "benchmark_id": benchmark_name,
                        "contender_name": record.get("contender_name"),
                        "family": record.get("family"),
                        "metric_name": record.get("metric_name"),
                        "metric_direction": record.get("metric_direction"),
                        "metric_value": record.get("metric_value"),
                        "quality": record.get("quality"),
                        "status": record.get("status"),
                        "failure_reason": record.get("failure_reason"),
                        "parameter_count": record.get("parameter_count"),
                        "train_seconds": record.get("train_seconds"),
                        "architecture_summary": record.get("architecture_summary"),
                        "contender_id": record.get("contender_id"),
                    }
                )

    return run_rows, score_rows, contender_rows


def collect_matrix_history(runs_root: Path) -> tuple[list[dict[str, Any]], list[dict[str, Any]]]:
    summary_rows: list[dict[str, Any]] = []
    parity_rows: list[dict[str, Any]] = []

    for summary_path in sorted(runs_root.rglob("four_way_summary.json")):
        payload = json.loads(summary_path.read_text(encoding="utf-8"))
        pack_name = payload.get("pack_name")
        for section, rows in (("fair", payload.get("fair_rows", [])), ("reference", payload.get("reference_rows", []))):
            for row in rows:
                summary_row_key = f"{relpath(summary_path)}|{section}|{row.get('budget')}|{row.get('seed')}"
                evaluation_counts = row.get("evaluation_counts", {})
                wins = row.get("wins", {})
                summary_rows.append(
                    {
                        "summary_row_key": summary_row_key,
                        "source_summary_relpath": relpath(summary_path),
                        "workspace_relpath": relpath(summary_path.parents[2]),
                        "pack_name": pack_name,
                        "summary_kind": section,
                        "budget": row.get("budget"),
                        "seed": row.get("seed"),
                        "benchmark_count": row.get("benchmark_count"),
                        "prism_evaluation_count": evaluation_counts.get("prism"),
                        "topograph_evaluation_count": evaluation_counts.get("topograph"),
                        "stratograph_evaluation_count": evaluation_counts.get("stratograph"),
                        "contenders_evaluation_count": evaluation_counts.get("contenders"),
                        "prism_wins": wins.get("prism"),
                        "topograph_wins": wins.get("topograph"),
                        "stratograph_wins": wins.get("stratograph"),
                        "contenders_wins": wins.get("contenders"),
                        "ties": row.get("ties"),
                        "note": row.get("note"),
                    }
                )

        for row in payload.get("parity_rows", []):
            parity_row_key = f"{relpath(summary_path)}|{row.get('pair_label')}|{row.get('budget')}|{row.get('seed')}"
            parity_rows.append(
                {
                    "parity_row_key": parity_row_key,
                    "source_summary_relpath": relpath(summary_path),
                    "workspace_relpath": relpath(summary_path.parents[2]),
                    "pack_name": pack_name,
                    "budget": row.get("budget"),
                    "seed": row.get("seed"),
                    "pair_label": row.get("pair_label"),
                    "parity_status": row.get("parity_status"),
                    "left_eval_count": row.get("left_eval_count"),
                    "right_eval_count": row.get("right_eval_count"),
                    "left_policy": row.get("left_policy"),
                    "right_policy": row.get("right_policy"),
                    "data_signature_match": row.get("data_signature_match"),
                    "reason": row.get("reason"),
                    "comparison_report": row.get("comparison_report"),
                }
            )

    return summary_rows, parity_rows


def write_dashboard_assets(
    *,
    dashboard_root: Path,
    run_rows: list[dict[str, Any]],
    score_rows: list[dict[str, Any]],
    summary_rows: list[dict[str, Any]],
) -> None:
    dashboard_root.mkdir(parents=True, exist_ok=True)
    html_path = dashboard_root / "index.html"
    html_path.write_text(DASHBOARD_HTML, encoding="utf-8")


def write_csv(path: Path, rows: list[dict[str, Any]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    if not rows:
        path.write_text("", encoding="utf-8")
        return

    fieldnames = sorted({key for row in rows for key in row})
    with path.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=fieldnames)
        writer.writeheader()
        for row in rows:
            writer.writerow({key: serialize_csv_value(row.get(key)) for key in fieldnames})


def serialize_csv_value(value: Any) -> str:
    if value is None:
        return ""
    if isinstance(value, bool):
        return "true" if value else "false"
    if isinstance(value, (int, float)):
        return str(value)
    return str(value)


def dedupe_rows(rows: list[dict[str, Any]], key_name: str) -> list[dict[str, Any]]:
    unique: dict[str, dict[str, Any]] = {}
    for row in rows:
        unique[str(row[key_name])] = row
    return list(unique.values())


def build_source_run_key(manifest_path: Path, manifest: dict[str, Any]) -> str:
    return "|".join(
        [
            str(manifest.get("system") or infer_system_from_path(manifest_path)),
            str(manifest.get("run_id") or ""),
            normalize_iso(manifest.get("created_at") or ""),
            relpath(manifest_path),
        ]
    )


def relpath(path: Path) -> str:
    return str(path.resolve().relative_to(REPO_ROOT))


def workspace_root_for_manifest(manifest_path: Path) -> Path:
    if "runs" in manifest_path.parts:
        runs_index = manifest_path.parts.index("runs")
        return Path(*manifest_path.parts[:runs_index])
    return manifest_path.parent


def infer_system_from_path(path: Path) -> str:
    for candidate in SYSTEM_ORDER:
        if candidate in path.parts:
            return candidate
    return "unknown"


def normalize_iso(value: Any) -> str:
    if not value:
        return ""
    text = str(value)
    if text.endswith("Z"):
        return text
    try:
        return datetime.fromisoformat(text).astimezone(timezone.utc).isoformat().replace("+00:00", "Z")
    except ValueError:
        return text


def as_int(value: Any) -> int | None:
    if value in (None, ""):
        return None
    return int(value)


def as_float(value: Any) -> float | None:
    if value in (None, ""):
        return None
    return float(value)


DASHBOARD_HTML = """<!doctype html>
<html lang="en">
<head>
  <meta charset="utf-8">
  <meta name="viewport" content="width=device-width, initial-scale=1">
  <title>EvoNN Compare History</title>
  <style>
    :root {
      --bg: #f4f0e8;
      --paper: rgba(255, 252, 246, 0.92);
      --ink: #1d221d;
      --muted: #6d756c;
      --grid: rgba(29, 34, 29, 0.12);
      --accent: #b24a2e;
      --accent-soft: rgba(178, 74, 46, 0.12);
      --shadow: 0 18px 40px rgba(49, 36, 17, 0.12);
      --radius: 22px;
    }
    * { box-sizing: border-box; }
    body {
      margin: 0;
      color: var(--ink);
      background:
        radial-gradient(circle at top left, rgba(178, 74, 46, 0.12), transparent 30%),
        radial-gradient(circle at top right, rgba(33, 92, 140, 0.10), transparent 24%),
        linear-gradient(180deg, #f7f3ec 0%, #efe8dc 100%);
      font-family: "Avenir Next", "Helvetica Neue", Helvetica, sans-serif;
    }
    .shell {
      max-width: 1700px;
      margin: 0 auto;
      padding: 28px 24px 40px;
    }
    .hero {
      display: grid;
      grid-template-columns: 1.4fr 1fr;
      gap: 18px;
      align-items: end;
      margin-bottom: 18px;
    }
    .headline, .meta {
      background: var(--paper);
      border: 1px solid rgba(29, 34, 29, 0.08);
      border-radius: var(--radius);
      box-shadow: var(--shadow);
      backdrop-filter: blur(10px);
    }
    .headline {
      padding: 28px;
    }
    .eyebrow {
      margin: 0 0 10px;
      color: var(--accent);
      font-size: 12px;
      letter-spacing: 0.24em;
      text-transform: uppercase;
      font-weight: 700;
    }
    h1 {
      margin: 0;
      font-family: Baskerville, "Iowan Old Style", "Palatino Linotype", serif;
      font-size: clamp(34px, 4vw, 58px);
      line-height: 0.95;
      letter-spacing: -0.03em;
    }
    .lede {
      margin: 14px 0 0;
      max-width: 72ch;
      color: var(--muted);
      line-height: 1.55;
      font-size: 15px;
    }
    .meta {
      padding: 22px 24px;
      display: grid;
      gap: 14px;
    }
    .meta-grid {
      display: grid;
      grid-template-columns: repeat(3, minmax(0, 1fr));
      gap: 12px;
    }
    .stat {
      padding: 14px;
      border-radius: 16px;
      background: rgba(255, 255, 255, 0.65);
      border: 1px solid rgba(29, 34, 29, 0.08);
    }
    .stat-label {
      color: var(--muted);
      font-size: 11px;
      letter-spacing: 0.12em;
      text-transform: uppercase;
      margin-bottom: 6px;
    }
    .stat-value {
      font-size: 22px;
      font-weight: 700;
    }
    .controls {
      display: flex;
      flex-wrap: wrap;
      gap: 12px 14px;
      align-items: center;
      margin-bottom: 20px;
      padding: 16px 20px;
      background: rgba(255, 252, 246, 0.82);
      border: 1px solid rgba(29, 34, 29, 0.08);
      border-radius: 18px;
      box-shadow: var(--shadow);
    }
    .control-group {
      display: flex;
      flex-wrap: wrap;
      gap: 8px;
      align-items: center;
    }
    .control-label {
      color: var(--muted);
      font-size: 12px;
      text-transform: uppercase;
      letter-spacing: 0.14em;
      font-weight: 700;
    }
    button, input {
      font: inherit;
    }
    .chip {
      border: 1px solid rgba(29, 34, 29, 0.14);
      background: rgba(255, 255, 255, 0.7);
      color: var(--ink);
      padding: 7px 12px;
      border-radius: 999px;
      cursor: pointer;
      transition: 160ms ease;
    }
    .chip.active {
      background: var(--accent);
      color: white;
      border-color: var(--accent);
      box-shadow: 0 10px 18px rgba(178, 74, 46, 0.26);
    }
    .search {
      min-width: 220px;
      border-radius: 999px;
      border: 1px solid rgba(29, 34, 29, 0.14);
      background: rgba(255, 255, 255, 0.78);
      padding: 8px 14px;
    }
    .grid {
      display: grid;
      gap: 18px;
    }
    .project-card {
      background: var(--paper);
      border: 1px solid rgba(29, 34, 29, 0.08);
      border-radius: 26px;
      box-shadow: var(--shadow);
      padding: 20px 20px 18px;
    }
    .project-head {
      display: flex;
      justify-content: space-between;
      gap: 12px;
      align-items: baseline;
      margin-bottom: 14px;
    }
    .project-title {
      margin: 0;
      font-family: Baskerville, "Iowan Old Style", "Palatino Linotype", serif;
      font-size: 30px;
      letter-spacing: -0.02em;
    }
    .project-sub {
      color: var(--muted);
      font-size: 14px;
    }
    .chart-shell {
      border-radius: 20px;
      background:
        linear-gradient(180deg, rgba(255,255,255,0.84), rgba(247, 241, 230, 0.88));
      border: 1px solid rgba(29, 34, 29, 0.08);
      padding: 12px 12px 8px;
      overflow: hidden;
    }
    .chart-layout {
      display: grid;
      grid-template-columns: minmax(0, 1fr) 280px;
      gap: 16px;
      align-items: start;
    }
    svg {
      width: 100%;
      height: 360px;
      display: block;
      overflow: visible;
    }
    .legend {
      max-height: 360px;
      overflow: auto;
      padding-right: 4px;
    }
    .legend-item {
      display: flex;
      gap: 10px;
      align-items: center;
      padding: 6px 0;
      border-bottom: 1px solid rgba(29, 34, 29, 0.06);
      font-size: 13px;
    }
    .swatch {
      width: 12px;
      height: 12px;
      border-radius: 999px;
      flex: 0 0 auto;
    }
    .legend-name {
      flex: 1 1 auto;
      min-width: 0;
      overflow: hidden;
      text-overflow: ellipsis;
      white-space: nowrap;
    }
    .legend-tail {
      color: var(--muted);
      font-variant-numeric: tabular-nums;
    }
    .axis-label {
      fill: var(--muted);
      font-size: 11px;
      letter-spacing: 0.04em;
    }
    .grid-line {
      stroke: var(--grid);
      stroke-width: 1;
    }
    .series-line {
      fill: none;
      stroke-width: 2.1;
      stroke-linecap: round;
      stroke-linejoin: round;
      transition: opacity 160ms ease, stroke-width 160ms ease;
    }
    .series-point {
      stroke: rgba(255,255,255,0.92);
      stroke-width: 1.2;
      transition: opacity 160ms ease;
    }
    .empty {
      padding: 32px;
      color: var(--muted);
      text-align: center;
      border: 1px dashed rgba(29, 34, 29, 0.18);
      border-radius: 16px;
      background: rgba(255,255,255,0.55);
    }
    .tooltip {
      position: fixed;
      z-index: 10;
      pointer-events: none;
      min-width: 220px;
      max-width: 340px;
      padding: 12px 14px;
      border-radius: 14px;
      background: rgba(22, 27, 24, 0.94);
      color: white;
      box-shadow: 0 16px 36px rgba(0, 0, 0, 0.24);
      opacity: 0;
      transform: translateY(6px);
      transition: opacity 120ms ease, transform 120ms ease;
    }
    .tooltip.visible {
      opacity: 1;
      transform: translateY(0);
    }
    .tooltip-title {
      font-weight: 700;
      margin-bottom: 6px;
    }
    .tooltip-grid {
      display: grid;
      grid-template-columns: auto 1fr;
      gap: 4px 10px;
      font-size: 12px;
      color: rgba(255,255,255,0.82);
    }
    .footnote {
      margin-top: 18px;
      color: var(--muted);
      font-size: 12px;
      line-height: 1.5;
    }
    @media (max-width: 1180px) {
      .hero, .chart-layout { grid-template-columns: 1fr; }
      .legend { max-height: none; }
      svg { height: 320px; }
    }
  </style>
</head>
<body>
  <div class="shell">
    <section class="hero">
      <div class="headline">
        <p class="eyebrow">Compare History</p>
        <h1>Project score drift over time, budgets, and reruns.</h1>
        <p class="lede">
          One chart per system. One colored line per benchmark. Default view normalizes each benchmark's
          history so movement is readable even when raw scales differ hard between accuracy and perplexity.
        </p>
      </div>
      <div class="meta">
        <div class="meta-grid" id="metaGrid"></div>
        <div class="footnote">
          Point radius tracks budget size. Hover points for exact metric, budget, seed, train time, and run id.
          Raw values stay available in the tooltip even when the chart uses normalized score. Open through a tiny local server,
          for example <code>python3 -m http.server</code> inside <code>dashboard/compare-history</code>, so the browser can fetch CSV files.
        </div>
      </div>
    </section>

    <section class="controls">
      <div class="control-group">
        <span class="control-label">Y Scale</span>
        <button class="chip active" data-mode="normalized">Normalized</button>
        <button class="chip" data-mode="quality">Quality</button>
        <button class="chip" data-mode="raw">Raw Metric</button>
      </div>
      <div class="control-group">
        <span class="control-label">Budgets</span>
        <div id="budgetChips" class="control-group"></div>
      </div>
      <div class="control-group">
        <span class="control-label">Benchmark Focus</span>
        <input id="searchInput" class="search" type="search" placeholder="type benchmark id">
      </div>
    </section>

    <section class="grid" id="projectGrid"></section>
  </div>

  <div id="tooltip" class="tooltip"></div>
  <script>
    const payload = { systems: [], records: [], matrix_rows: [], generated_at: null };
    const state = {
      mode: "normalized",
      budgets: new Set(),
      search: "",
    };

    const metaGrid = document.getElementById("metaGrid");
    const budgetChips = document.getElementById("budgetChips");
    const projectGrid = document.getElementById("projectGrid");
    const tooltip = document.getElementById("tooltip");
    const searchInput = document.getElementById("searchInput");

    function formatCount(value) {
      return Intl.NumberFormat("en-US").format(value);
    }

    function formatDate(value) {
      if (!value) return "n/a";
      const date = new Date(value);
      return date.toLocaleString(undefined, {
        year: "numeric",
        month: "short",
        day: "2-digit",
        hour: "2-digit",
        minute: "2-digit",
      });
    }

    function formatMetric(value) {
      if (value == null || Number.isNaN(value)) return "n/a";
      if (Math.abs(value) >= 100) return value.toFixed(2);
      if (Math.abs(value) >= 10) return value.toFixed(3);
      return value.toFixed(4);
    }

    function palette(index, total) {
      const hue = (index * 137.508) % 360;
      const sat = 68;
      const light = 42 + ((index % 3) * 7);
      return `hsl(${hue.toFixed(1)} ${sat}% ${light}%)`;
    }

    function uniqueBenchmarks(records) {
      return [...new Set(records.map((record) => record.benchmark_id))].sort();
    }

    function uniqueBudgets(records) {
      return [...new Set(records.map((record) => record.budget))].sort((a, b) => a - b);
    }

    function parseMaybeNumber(value) {
      if (value == null || value === "") return null;
      const parsed = Number(value);
      return Number.isFinite(parsed) ? parsed : value;
    }

    function parseCsv(text) {
      const rows = [];
      let row = [];
      let cell = "";
      let inQuotes = false;
      for (let index = 0; index < text.length; index += 1) {
        const char = text[index];
        const next = text[index + 1];
        if (char === "\"") {
          if (inQuotes && next === "\"") {
            cell += "\"";
            index += 1;
          } else {
            inQuotes = !inQuotes;
          }
        } else if (char === "," && !inQuotes) {
          row.push(cell);
          cell = "";
        } else if ((char === "\n" || char === "\r") && !inQuotes) {
          if (char === "\r" && next === "\n") index += 1;
          row.push(cell);
          if (row.length > 1 || row[0] !== "") rows.push(row);
          row = [];
          cell = "";
        } else {
          cell += char;
        }
      }
      row.push(cell);
      if (row.length > 1 || row[0] !== "") rows.push(row);
      if (!rows.length) return [];
      const header = rows[0];
      return rows.slice(1).map((values) => {
        const object = {};
        header.forEach((key, index) => {
          object[key] = values[index] ?? "";
        });
        return object;
      });
    }

    async function loadCsv(path) {
      const response = await fetch(path, { cache: "no-store" });
      if (!response.ok) {
        throw new Error(`failed to load ${path}: ${response.status}`);
      }
      return parseCsv(await response.text());
    }

    function coerceScoreRow(row) {
      return {
        source_run_key: row.source_run_key,
        system: row.system,
        run_id: row.run_id,
        created_at: row.created_at,
        budget: parseMaybeNumber(row.budget_evaluation_count),
        seed: parseMaybeNumber(row.seed),
        benchmark_id: row.benchmark_id,
        task_kind: row.task_kind,
        metric_name: row.metric_name,
        metric_direction: row.metric_direction,
        metric_value: parseMaybeNumber(row.metric_value),
        quality: parseMaybeNumber(row.quality),
        train_seconds: parseMaybeNumber(row.train_seconds),
        parameter_count: parseMaybeNumber(row.parameter_count),
        pack_name: row.pack_name,
        budget_policy_name: row.budget_policy_name,
        status: row.status,
      };
    }

    function scoreAccessorFactory(records, mode) {
      if (mode === "raw") {
        return (record) => record.metric_value;
      }
      if (mode === "quality") {
        return (record) => record.quality;
      }

      const perBenchmark = new Map();
      for (const benchmarkId of uniqueBenchmarks(records)) {
        const values = records
          .filter((record) => record.benchmark_id === benchmarkId)
          .map((record) => record.quality)
          .filter((value) => value != null && Number.isFinite(value));
        if (!values.length) continue;
        perBenchmark.set(benchmarkId, {
          min: Math.min(...values),
          max: Math.max(...values),
        });
      }

      return (record) => {
        const bucket = perBenchmark.get(record.benchmark_id);
        if (!bucket || record.quality == null || !Number.isFinite(record.quality)) return null;
        const span = bucket.max - bucket.min;
        if (Math.abs(span) <= 1e-12) return 50;
        return ((record.quality - bucket.min) / span) * 100;
      };
    }

    function makeMeta() {
      const latest = payload.records
        .map((record) => record.created_at)
        .filter(Boolean)
        .sort()
        .at(-1);
      const cards = [
        ["Runs", new Set(payload.records.map((record) => record.source_run_key)).size],
        ["Systems", payload.systems.length],
        ["Benchmarks", uniqueBenchmarks(payload.records).length],
        ["Budgets", uniqueBudgets(payload.records).join(" / ")],
        ["Records", payload.records.length],
        ["Updated", formatDate(latest)],
      ];
      metaGrid.innerHTML = cards
        .map(
          ([label, value]) => `
            <div class="stat">
              <div class="stat-label">${label}</div>
              <div class="stat-value">${typeof value === "number" ? formatCount(value) : value}</div>
            </div>`
        )
        .join("");
    }

    function makeBudgetControls() {
      const budgets = uniqueBudgets(payload.records);
      if (!state.budgets.size) {
        budgets.forEach((budget) => state.budgets.add(budget));
      }
      budgetChips.innerHTML = "";
      for (const budget of budgets) {
        const button = document.createElement("button");
        button.className = "chip active";
        button.textContent = String(budget);
        button.dataset.budget = String(budget);
        button.addEventListener("click", () => {
          if (state.budgets.has(budget)) {
            state.budgets.delete(budget);
            button.classList.remove("active");
          } else {
            state.budgets.add(budget);
            button.classList.add("active");
          }
          render();
        });
        budgetChips.appendChild(button);
      }

      for (const button of document.querySelectorAll("[data-mode]")) {
        button.addEventListener("click", () => {
          state.mode = button.dataset.mode;
          document.querySelectorAll("[data-mode]").forEach((node) => node.classList.toggle("active", node === button));
          render();
        });
      }

      searchInput.addEventListener("input", (event) => {
        state.search = event.target.value.trim().toLowerCase();
        render();
      });
    }

    function seriesOpacity(benchmarkId) {
      if (!state.search) return 0.9;
      return benchmarkId.toLowerCase().includes(state.search) ? 1 : 0.08;
    }

    function showTooltip(event, html) {
      tooltip.innerHTML = html;
      tooltip.classList.add("visible");
      const x = Math.min(window.innerWidth - tooltip.offsetWidth - 16, event.clientX + 18);
      const y = Math.min(window.innerHeight - tooltip.offsetHeight - 16, event.clientY + 18);
      tooltip.style.left = `${Math.max(12, x)}px`;
      tooltip.style.top = `${Math.max(12, y)}px`;
    }

    function hideTooltip() {
      tooltip.classList.remove("visible");
    }

    function buildChart(system, records) {
      const filtered = records
        .filter((record) => state.budgets.has(record.budget))
        .sort((a, b) => new Date(a.created_at) - new Date(b.created_at) || a.budget - b.budget || a.seed - b.seed || a.benchmark_id.localeCompare(b.benchmark_id));

      const card = document.createElement("article");
      card.className = "project-card";

      const runs = [...new Map(filtered.map((record) => [record.source_run_key, record])).values()];
      const scoreAccessor = scoreAccessorFactory(filtered, state.mode);
      const budgets = uniqueBudgets(filtered);
      const benchmarkIds = uniqueBenchmarks(filtered);
      const colorByBenchmark = new Map(benchmarkIds.map((benchmarkId, index) => [benchmarkId, palette(index, benchmarkIds.length)]));

      const enriched = filtered
        .map((record) => ({ ...record, score: scoreAccessor(record) }))
        .filter((record) => record.score != null && Number.isFinite(record.score));

      const title = `<div class="project-head">
        <div>
          <h2 class="project-title">${system}</h2>
          <div class="project-sub">${runs.length} runs • budgets ${budgets.join(", ") || "n/a"} • ${benchmarkIds.length} benchmarks</div>
        </div>
      </div>`;

      if (!enriched.length || runs.length < 2) {
        card.innerHTML = `${title}<div class="empty">No chartable history for current filters.</div>`;
        return card;
      }

      const width = 1040;
      const height = 360;
      const margin = { top: 18, right: 18, bottom: 70, left: 56 };
      const innerWidth = width - margin.left - margin.right;
      const innerHeight = height - margin.top - margin.bottom;

      const runOrder = runs
        .sort((a, b) => new Date(a.created_at) - new Date(b.created_at) || a.budget - b.budget || a.seed - b.seed)
        .map((record) => record.source_run_key);
      const xIndex = new Map(runOrder.map((key, index) => [key, index]));
      const yValues = enriched.map((record) => record.score);
      const yMin = state.mode === "normalized" ? 0 : Math.min(...yValues);
      const yMax = state.mode === "normalized" ? 100 : Math.max(...yValues);
      const ySpan = Math.max(1e-9, yMax - yMin);

      const xOf = (row) => margin.left + (xIndex.get(row.source_run_key) / Math.max(1, runOrder.length - 1)) * innerWidth;
      const yOf = (row) => margin.top + (1 - ((row.score - yMin) / ySpan)) * innerHeight;

      const ticksY = 5;
      const gridLines = [];
      for (let i = 0; i <= ticksY; i += 1) {
        const ratio = i / ticksY;
        const y = margin.top + ratio * innerHeight;
        const value = yMax - ratio * ySpan;
        gridLines.push(`
          <line class="grid-line" x1="${margin.left}" x2="${width - margin.right}" y1="${y}" y2="${y}"></line>
          <text class="axis-label" x="${margin.left - 10}" y="${y + 4}" text-anchor="end">${formatMetric(value)}</text>
        `);
      }

      const tickLabels = runOrder.map((key, index) => {
        const row = runs.find((item) => item.source_run_key === key);
        const x = margin.left + (index / Math.max(1, runOrder.length - 1)) * innerWidth;
        return `
          <line class="grid-line" x1="${x}" x2="${x}" y1="${margin.top}" y2="${height - margin.bottom + 6}" style="opacity:0.55"></line>
          <text class="axis-label" x="${x}" y="${height - margin.bottom + 22}" text-anchor="middle">${row.budget}</text>
          <text class="axis-label" x="${x}" y="${height - margin.bottom + 38}" text-anchor="middle">s${row.seed}</text>
        `;
      }).join("");

      const series = benchmarkIds.map((benchmarkId) => {
        const rows = enriched.filter((record) => record.benchmark_id === benchmarkId).sort((a, b) => xIndex.get(a.source_run_key) - xIndex.get(b.source_run_key));
        if (!rows.length) return "";
        const color = colorByBenchmark.get(benchmarkId);
        const opacity = seriesOpacity(benchmarkId);
        const path = rows.map((row, index) => `${index === 0 ? "M" : "L"} ${xOf(row).toFixed(2)} ${yOf(row).toFixed(2)}`).join(" ");
        const points = rows.map((row) => {
          const radius = 3 + ((row.budget || 0) / Math.max(...budgets, 1)) * 3;
          const tooltipHtml = `
            <div class="tooltip-title">${benchmarkId}</div>
            <div class="tooltip-grid">
              <div>System</div><div>${row.system}</div>
              <div>Run</div><div>${row.run_id}</div>
              <div>Date</div><div>${formatDate(row.created_at)}</div>
              <div>Budget</div><div>${row.budget}</div>
              <div>Seed</div><div>${row.seed}</div>
              <div>Metric</div><div>${row.metric_name}: ${formatMetric(row.metric_value)}</div>
              <div>Quality</div><div>${formatMetric(row.quality)}</div>
              <div>Train Sec</div><div>${row.train_seconds == null ? "n/a" : formatMetric(row.train_seconds)}</div>
            </div>`;
          return `<circle class="series-point" cx="${xOf(row).toFixed(2)}" cy="${yOf(row).toFixed(2)}" r="${radius.toFixed(2)}" fill="${color}" fill-opacity="${Math.max(0.22, opacity)}" data-tooltip='${escapeHtml(tooltipHtml)}'></circle>`;
        }).join("");
        return `<path class="series-line" d="${path}" stroke="${color}" stroke-opacity="${opacity}" style="stroke-width:${state.search && opacity < 1 ? 1.4 : 2.2}px"></path>${points}`;
      }).join("");

      const latestByBenchmark = new Map();
      for (const benchmarkId of benchmarkIds) {
        const rows = enriched.filter((record) => record.benchmark_id === benchmarkId).sort((a, b) => new Date(a.created_at) - new Date(b.created_at));
        if (rows.length) latestByBenchmark.set(benchmarkId, rows[rows.length - 1]);
      }
      const legend = benchmarkIds
        .map((benchmarkId) => {
          const latest = latestByBenchmark.get(benchmarkId);
          return `
            <div class="legend-item" style="opacity:${seriesOpacity(benchmarkId)}">
              <span class="swatch" style="background:${colorByBenchmark.get(benchmarkId)}"></span>
              <span class="legend-name">${benchmarkId}</span>
              <span class="legend-tail">${latest ? formatMetric(state.mode === "raw" ? latest.metric_value : latest.score) : "n/a"}</span>
            </div>`;
        })
        .join("");

      card.innerHTML = `
        ${title}
        <div class="chart-shell">
          <div class="chart-layout">
            <svg viewBox="0 0 ${width} ${height}" role="img" aria-label="${system} benchmark history chart">
              ${gridLines.join("")}
              ${tickLabels}
              <text class="axis-label" x="${margin.left}" y="${height - 8}">chronological runs (budget / seed)</text>
              <text class="axis-label" x="18" y="${margin.top - 4}" transform="rotate(-90 18 ${margin.top - 4})">${state.mode === "normalized" ? "normalized quality (0-100)" : state.mode}</text>
              ${series}
            </svg>
            <div class="legend">${legend}</div>
          </div>
        </div>`;

      card.querySelectorAll("[data-tooltip]").forEach((node) => {
        node.addEventListener("mouseenter", (event) => showTooltip(event, unescapeHtml(node.dataset.tooltip)));
        node.addEventListener("mousemove", (event) => showTooltip(event, unescapeHtml(node.dataset.tooltip)));
        node.addEventListener("mouseleave", hideTooltip);
      });

      return card;
    }

    function escapeHtml(value) {
      return value.replace(/&/g, "&amp;").replace(/'/g, "&#39;").replace(/</g, "&lt;").replace(/>/g, "&gt;");
    }

    function unescapeHtml(value) {
      return value
        .replace(/&gt;/g, ">")
        .replace(/&lt;/g, "<")
        .replace(/&#39;/g, "'")
        .replace(/&amp;/g, "&");
    }

    function render() {
      projectGrid.innerHTML = "";
      for (const system of payload.systems) {
        const records = payload.records.filter((record) => record.system === system);
        projectGrid.appendChild(buildChart(system, records));
      }
    }

    async function boot() {
      try {
        const scoreRows = await loadCsv("./data/compare_history_scores.csv");
        payload.records = scoreRows
          .map(coerceScoreRow)
          .filter((record) => record.system && record.status === "ok" && record.metric_value != null);
        payload.systems = [...new Set(payload.records.map((record) => record.system))]
          .sort((left, right) => {
            const leftIndex = ["prism", "topograph", "stratograph", "contenders", "hybrid"].indexOf(left);
            const rightIndex = ["prism", "topograph", "stratograph", "contenders", "hybrid"].indexOf(right);
            return (leftIndex === -1 ? 999 : leftIndex) - (rightIndex === -1 ? 999 : rightIndex) || left.localeCompare(right);
          });
        state.budgets = new Set(uniqueBudgets(payload.records));
        makeMeta();
        makeBudgetControls();
        render();
      } catch (error) {
        projectGrid.innerHTML = `<div class="empty">Could not load CSV data. Start a local server in this folder and reload.<br><br>${String(error)}</div>`;
      }
    }

    boot();
  </script>
</body>
</html>
"""


if __name__ == "__main__":
    raise SystemExit(main())
