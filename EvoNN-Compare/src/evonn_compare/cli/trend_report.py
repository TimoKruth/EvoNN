"""CLI for merged/queryable fair-matrix trend datasets."""

from __future__ import annotations

from dataclasses import asdict
import json
from pathlib import Path

import typer

from evonn_compare.comparison.fair_matrix import MatrixTrendRow
from evonn_compare.reporting.fair_matrix_trends_md import render_fair_matrix_trend_markdown


def trend_report(
    inputs: list[str] = typer.Argument(..., help="One or more trend_rows.json, fair_matrix_summary.json, fair_matrix_trend_rows.jsonl, fair_matrix_trends.json, or fair_matrix_trends.jsonl files"),
    system: str | None = typer.Option(None, "--system", help="Filter to one system"),
    benchmark: str | None = typer.Option(None, "--benchmark", help="Filter to one benchmark_id"),
    pack: str | None = typer.Option(None, "--pack", help="Filter to one pack name"),
    output: str | None = typer.Option(None, "--output", help="Optional markdown output path"),
) -> None:
    """Merge and query accumulated fair-matrix trend datasets."""

    rows = load_trend_rows([Path(value) for value in inputs])
    rows = [
        row for row in rows
        if (system is None or row.system == system)
        and (benchmark is None or row.benchmark_id == benchmark)
        and (pack is None or row.pack_name == pack)
    ]
    markdown = render_fair_matrix_trend_markdown(rows)
    if output is not None:
        output_path = Path(output)
        output_path.write_text(markdown, encoding="utf-8")
        output_path.with_suffix(".json").write_text(
            json.dumps([asdict(row) for row in rows], indent=2, default=str),
            encoding="utf-8",
        )
    typer.echo(markdown)


def load_trend_rows(paths: list[Path]) -> list[MatrixTrendRow]:
    rows: list[MatrixTrendRow] = []
    for path in paths:
        payload = _read_payload(path)
        if isinstance(payload, list):
            candidate_rows = payload
        elif isinstance(payload, dict) and isinstance(payload.get("trend_rows"), list):
            candidate_rows = payload["trend_rows"]
        else:
            raise ValueError(f"unsupported trend payload in {path}")
        rows.extend(_coerce_trend_row(entry) for entry in candidate_rows)
    return rows


def _read_payload(path: Path):
    if path.suffix == ".jsonl":
        return [json.loads(line) for line in path.read_text(encoding="utf-8").splitlines() if line.strip()]
    return json.loads(path.read_text(encoding="utf-8"))


def _coerce_trend_row(entry: dict) -> MatrixTrendRow:
    if "pack_name" not in entry and "pack" in entry:
        return MatrixTrendRow(
            pack_name=str(entry["pack"]),
            budget=int(entry["budget"]),
            seed=int(entry["seed"]),
            system=str(entry["engine"]),
            run_id=str(entry["run_id"]),
            benchmark_id=str(entry["benchmark"]),
            metric_name=str(entry["metric_name"]),
            metric_direction=str(entry["metric_direction"]),
            metric_value=None if entry.get("metric_value") is None else float(entry["metric_value"]),
            outcome_status=str(entry["outcome_status"]),
            failure_reason=None if entry.get("failure_reason") is None else str(entry["failure_reason"]),
            evaluation_count=int((entry.get("fairness") or {}).get("evaluation_count") or entry["budget"]),
            epochs_per_candidate=int(entry.get("epochs_per_candidate") or 0),
            budget_policy_name=None if (entry.get("fairness") or {}).get("budget_policy_name") is None else str((entry.get("fairness") or {})["budget_policy_name"]),
            wall_clock_seconds=None if entry.get("wall_clock_seconds") is None else float(entry["wall_clock_seconds"]),
            matrix_scope="fair" if not entry.get("reference_only") else "reference",
            fairness_metadata=dict(entry.get("fairness") or {}),
            lane_operating_state=str((entry.get("lane") or {}).get("operating_state") or (entry.get("fairness") or {}).get("lane_operating_state") or ("fair" if not entry.get("reference_only") else "reference-only")),
            system_operating_state=str(entry.get("system_operating_state") or (entry.get("fairness") or {}).get("system_operating_state") or "unknown"),
            lane_repeatability_ready=bool((entry.get("lane") or {}).get("repeatability_ready")),
            lane_budget_accounting_ok=bool((entry.get("lane") or {}).get("budget_accounting_ok") or (entry.get("fairness") or {}).get("budget_accounting_ok")),
        )
    return MatrixTrendRow(
        pack_name=str(entry["pack_name"]),
        budget=int(entry["budget"]),
        seed=int(entry["seed"]),
        system=str(entry["system"]),
        run_id=str(entry["run_id"]),
        benchmark_id=str(entry["benchmark_id"]),
        metric_name=str(entry["metric_name"]),
        metric_direction=str(entry["metric_direction"]),
        metric_value=None if entry.get("metric_value") is None else float(entry["metric_value"]),
        outcome_status=str(entry["outcome_status"]),
        failure_reason=None if entry.get("failure_reason") is None else str(entry["failure_reason"]),
        evaluation_count=int(entry["evaluation_count"]),
        epochs_per_candidate=int(entry["epochs_per_candidate"]),
        budget_policy_name=None if entry.get("budget_policy_name") is None else str(entry["budget_policy_name"]),
        wall_clock_seconds=None if entry.get("wall_clock_seconds") is None else float(entry["wall_clock_seconds"]),
        matrix_scope=str(entry["matrix_scope"]),
        fairness_metadata=dict(entry.get("fairness_metadata") or {}),
        lane_operating_state=str(entry.get("lane_operating_state") or (entry.get("fairness_metadata") or {}).get("lane_operating_state") or "reference-only"),
        system_operating_state=str(entry.get("system_operating_state") or (entry.get("fairness_metadata") or {}).get("system_operating_state") or "unknown"),
        lane_repeatability_ready=bool(entry.get("lane_repeatability_ready")),
        lane_budget_accounting_ok=bool(entry.get("lane_budget_accounting_ok") or (entry.get("fairness_metadata") or {}).get("budget_accounting_ok")),
    )
