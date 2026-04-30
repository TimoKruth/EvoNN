"""CLI for fair four-way matrix campaigns."""

from __future__ import annotations

import json
from pathlib import Path

import typer

from evonn_compare.cli.workspace_report import refresh_workspace_reports
from evonn_compare.contracts.parity import load_parity_pack, resolve_pack_path
from evonn_compare.orchestration.campaign_state import (
    StopRequested,
    build_case_id,
    case_integrity_issues,
    clear_stop_request,
    complete_case,
    fail_case,
    interrupt_case,
    mark_stale_running_cases_interrupted,
    require_workspace_state,
    start_case,
    stop_requested,
    update_case_stage,
    workspace_state_path,
)
from evonn_compare.orchestration.lane_presets import lane_preset_help, resolve_lane_preset
from evonn_compare.orchestration.fair_matrix import (
    prepare_fair_matrix_cases,
    reset_fair_matrix_workspace,
    run_fair_matrix_case,
)


def fair_matrix(
    pack: str | None = typer.Option(None, "--pack", help="Parity pack name or YAML path"),
    preset: str | None = typer.Option(None, "--preset", help=lane_preset_help(default_name="local")),
    seeds: str | None = typer.Option(None, "--seeds", help="Comma-separated seeds"),
    budgets: str | None = typer.Option(None, "--budgets", help="Comma-separated budgets"),
    workspace: str = typer.Option(..., "--workspace", help="Campaign workspace"),
    prism_root: str = typer.Option("EvoNN-Prism", "--prism-root"),
    topograph_root: str = typer.Option("EvoNN-Topograph", "--topograph-root"),
    stratograph_root: str = typer.Option("EvoNN-Stratograph", "--stratograph-root"),
    primordia_root: str = typer.Option("EvoNN-Primordia", "--primordia-root"),
    contenders_root: str = typer.Option("EvoNN-Contenders", "--contenders-root"),
    include_contenders: bool = typer.Option(True, "--include-contenders/--no-contenders", help="Include contender baselines in the fair-matrix run"),
    parallel: bool = typer.Option(True, "--parallel/--serial", help="Run project stages concurrently"),
    resume: bool = typer.Option(False, "--resume", help="Resume incomplete cases in an existing workspace"),
    dry_run: bool = typer.Option(False, "--dry-run", help="Only generate configs and print cases"),
    reset_workspace: bool = typer.Option(False, "--reset-workspace/--preserve-workspace", help="Remove managed workspace artifacts before generating cases"),
    open_browser: bool = typer.Option(False, "--open/--no-open", help="Open the refreshed dashboard in the default browser"),
) -> None:
    """Generate and optionally execute fair four-way compare cases."""

    preset_name = preset or (None if pack else "local")
    preset_spec = resolve_lane_preset(preset_name) if preset_name else None
    pack_name = pack or (preset_spec.pack if preset_spec else None)
    workspace_path = Path(workspace).resolve()

    if not dry_run and reset_workspace:
        reset_fair_matrix_workspace(workspace_path)

    pack_path = resolve_pack_path(pack_name)
    pack_spec = load_parity_pack(pack_path)
    paths, cases = prepare_fair_matrix_cases(
        pack_name=Path(pack_path).stem,
        base_pack_path=pack_path,
        seeds=_parse_optional_csv_ints(seeds) or (list(preset_spec.seeds) if preset_spec else [42]),
        budgets=_parse_optional_csv_ints(budgets)
        or (list(preset_spec.budgets) if preset_spec else [pack_spec.budget_policy.evaluation_count]),
        workspace=workspace_path,
        prism_root=Path(prism_root),
        topograph_root=Path(topograph_root),
        stratograph_root=Path(stratograph_root),
        primordia_root=Path(primordia_root),
        contenders_root=Path(contenders_root),
        include_contenders=include_contenders,
        lane_preset=preset_name,
    )
    state_path = workspace_state_path(workspace_path)
    state_available = state_path.exists()
    if dry_run:
        typer.echo("mode\tdry-run")
        typer.echo(f"manifest\t{paths.manifest_path}")
        typer.echo(f"state\t{state_path}")
        typer.echo(f"trend-dataset\t{paths.trends_dir / 'fair_matrix_trends.jsonl'}")
        for case in cases:
            typer.echo(json.dumps({key: str(value) if isinstance(value, Path) else value for key, value in case.__dict__.items()}))
        return

    if state_available:
        clear_stop_request(workspace_path)
    stale_running_cases = mark_stale_running_cases_interrupted(workspace_path) if resume and state_available else 0
    typer.echo("mode\texecute")
    typer.echo(f"manifest\t{paths.manifest_path}")
    typer.echo(f"state\t{state_path}")
    typer.echo(f"trend-dataset\t{paths.trends_dir / 'fair_matrix_trends.jsonl'}")
    if stale_running_cases:
        typer.echo(f"interrupted\t{stale_running_cases}")
    for index, case in enumerate(cases, start=1):
        case_id = _case_identifier(case, fallback_index=index)
        if resume and state_available and _case_already_completed(workspace_path, case_id=case_id):
            typer.echo(f"skip\t{case_id}\tcompleted")
            continue
        if state_available and stop_requested(workspace_path):
            typer.echo("stop-requested\ttrue")
            break
        if state_available:
            start_case(workspace_path, case_id=case_id, resume_requested=resume)
        try:
            summary_path = _run_case(
                case,
                prism_root=Path(prism_root),
                topograph_root=Path(topograph_root),
                stratograph_root=Path(stratograph_root),
                primordia_root=Path(primordia_root),
                contenders_root=Path(contenders_root),
                parallel=parallel,
                resume=resume,
                stage_callback=(lambda stage, workspace=workspace_path, case_ref=case_id: update_case_stage(
                    workspace,
                    case_id=case_ref,
                stage=stage,
                )) if state_available else None,
                stop_requested=(lambda workspace=workspace_path: stop_requested(workspace)) if state_available else None,
            )
            if state_available:
                integrity_issues = case_integrity_issues(case)
                complete_case(workspace_path, case_id=case_id, integrity_issues=integrity_issues)
            typer.echo(f"summary\t{summary_path}")
            for label, artifact_path in _trend_artifact_paths(summary_path).items():
                typer.echo(f"{label}\t{artifact_path}")
        except StopRequested as exc:
            if state_available:
                interrupt_case(workspace_path, case_id=case_id, reason=str(exc))
            typer.echo(f"stopped\t{case_id}")
            break
        except KeyboardInterrupt:
            if state_available:
                interrupt_case(workspace_path, case_id=case_id, reason="keyboard interrupt")
            raise
        except Exception as exc:
            if state_available:
                fail_case(workspace_path, case_id=case_id, reason=f"{type(exc).__name__}: {exc}")
            raise

    workspace_artifacts = refresh_workspace_reports(workspace=workspace_path, open_browser=open_browser)
    typer.echo(f"workspace_trend_report\t{workspace_artifacts['trend_report']}")
    typer.echo(f"workspace_trend_report_data\t{workspace_artifacts['trend_report_data']}")
    typer.echo(f"workspace_dashboard\t{workspace_artifacts['dashboard']}")
    typer.echo(f"workspace_dashboard_data\t{workspace_artifacts['dashboard_data']}")
    if open_browser:
        typer.echo(f"opened\t{Path(str(workspace_artifacts['dashboard'])).resolve().as_uri()}")


def _trend_artifact_paths(summary_path: Path) -> dict[str, Path]:
    case_dir = summary_path.parent
    workspace_dir = case_dir.parent
    return {
        "summary_json": summary_path.with_suffix(".json"),
        "lane_acceptance": case_dir / "lane_acceptance.json",
        "trend_rows": case_dir / "trend_rows.json",
        "trend_report": case_dir / "trend_report.md",
        "trend_records_json": case_dir / "fair_matrix_trends.json",
        "trend_records_jsonl": case_dir / "fair_matrix_trends.jsonl",
        "workspace_trend_rows": workspace_dir / "fair_matrix_trend_rows.jsonl",
        "workspace_trend_report": workspace_dir / "fair_matrix_trends.md",
    }


def _parse_csv_ints(raw: str) -> list[int]:
    values = [item.strip() for item in raw.split(",") if item.strip()]
    if not values:
        raise ValueError("expected at least one integer value")
    return [int(item) for item in values]


def _parse_optional_csv_ints(raw: str | None) -> list[int] | None:
    if raw is None:
        return None
    return _parse_csv_ints(raw)


def _run_case(case, **kwargs):
    try:
        return run_fair_matrix_case(case, **kwargs)
    except TypeError as exc:
        expected = ("resume", "stage_callback", "stop_requested")
        if not any(name in str(exc) for name in expected):
            raise
        fallback_kwargs = {
            key: value
            for key, value in kwargs.items()
            if key not in {"resume", "stage_callback", "stop_requested"}
        }
        return run_fair_matrix_case(case, **fallback_kwargs)


def _case_identifier(case, *, fallback_index: int) -> str:
    try:
        return build_case_id(case)
    except AttributeError:
        summary_path = getattr(case, "summary_output_path", None)
        if summary_path is not None:
            return Path(summary_path).stem
        return f"case-{fallback_index}"


def _case_already_completed(workspace: Path, *, case_id: str) -> bool:
    state = require_workspace_state(workspace)
    for case in state.get("cases", []):
        if case.get("case_id") == case_id:
            return case.get("status") == "completed"
    return False
