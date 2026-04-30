"""CLI for rendering canonical performance dashboard and delta artifacts."""

from __future__ import annotations

from pathlib import Path

import typer

from evonn_compare.orchestration.performance_report import build_performance_report


def performance_report(
    baseline: str = typer.Argument(..., help="Baseline perf_rows.jsonl file or baseline artifact root"),
    candidate: list[str] = typer.Option(
        [],
        "--candidate",
        help="Repeatable candidate spec in label=path form",
    ),
    outcome: list[str] = typer.Option(
        [],
        "--outcome",
        help="Repeatable outcome override in label=accepted|rejected-for-revision|scrapped|candidate form",
    ),
    baseline_label: str = typer.Option("baseline", "--baseline-label", help="Label used for the baseline dataset"),
    compare_label: str | None = typer.Option(
        None,
        "--compare-label",
        help="Candidate label to feature in the primary before/after delta view",
    ),
    output_root: str | None = typer.Option(
        None,
        "--output-root",
        help="Output directory; defaults to <baseline-root>/review",
    ),
) -> None:
    """Render performance report artifacts from canonical perf_rows datasets."""

    candidate_specs = [_parse_path_spec(value, option_name="--candidate") for value in candidate]
    outcome_map = {
        label: outcome_value
        for label, outcome_value in (_parse_value_spec(value, option_name="--outcome") for value in outcome)
    }
    resolved_baseline = Path(baseline)
    resolved_output_root = (
        Path(output_root)
        if output_root is not None
        else (resolved_baseline if resolved_baseline.is_dir() else resolved_baseline.parent) / "review"
    )

    result = build_performance_report(
        baseline_label=baseline_label,
        baseline_path=resolved_baseline,
        candidate_specs=[(label, Path(path)) for label, path in candidate_specs],
        outcomes=outcome_map,
        compare_label=compare_label,
        output_root=resolved_output_root,
    )
    typer.echo(f"baseline_label\t{result['baseline_label']}")
    typer.echo(f"baseline_source\t{result['baseline_source']}")
    typer.echo(f"candidate_count\t{result['candidate_count']}")
    typer.echo(f"compare_label\t{result['compare_label']}")
    typer.echo(f"report_json\t{result['report_json']}")
    typer.echo(f"report_markdown\t{result['report_markdown']}")
    typer.echo(f"dashboard_html\t{result['dashboard_html']}")
    typer.echo(f"history_count\t{result['history_count']}")


def _parse_path_spec(raw: str, *, option_name: str) -> tuple[str, str]:
    if "=" not in raw:
        raise typer.BadParameter(f"{option_name} expects label=path")
    label, path = raw.split("=", 1)
    normalized_label = label.strip()
    normalized_path = path.strip()
    if not normalized_label or not normalized_path:
        raise typer.BadParameter(f"{option_name} expects non-empty label and path")
    return normalized_label, normalized_path


def _parse_value_spec(raw: str, *, option_name: str) -> tuple[str, str]:
    if "=" not in raw:
        raise typer.BadParameter(f"{option_name} expects label=value")
    label, value = raw.split("=", 1)
    normalized_label = label.strip()
    normalized_value = value.strip()
    if not normalized_label or not normalized_value:
        raise typer.BadParameter(f"{option_name} expects non-empty label and value")
    return normalized_label, normalized_value
