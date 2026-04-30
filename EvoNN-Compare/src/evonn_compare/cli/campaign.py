"""Campaign CLI for Prism vs Topograph execution."""

from __future__ import annotations

from concurrent.futures import ThreadPoolExecutor
import json
import subprocess
from pathlib import Path

import typer

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
    start_case,
    stop_requested,
    update_case_stage,
    workspace_state_path,
)
from evonn_compare.orchestration.config_gen import prepare_campaign_cases
from evonn_compare.orchestration.lane_presets import lane_preset_help, resolve_lane_preset
from evonn_compare.orchestration.runner import CampaignRunner


def campaign(
    pack: str | None = typer.Option(None, "--pack", help="Parity pack name or YAML path"),
    preset: str | None = typer.Option(None, "--preset", help=lane_preset_help(default_name="local")),
    seeds: str | None = typer.Option(None, "--seeds", help="Comma-separated seeds"),
    budgets: str | None = typer.Option(None, "--budgets", help="Comma-separated budgets"),
    workspace: str = typer.Option(..., "--workspace", help="Campaign workspace"),
    prism_root: str = typer.Option("../EvoNN-Prism", "--prism-root"),
    topograph_root: str = typer.Option("../EvoNN-Topograph", "--topograph-root"),
    parallel: bool = typer.Option(True, "--parallel/--serial", help="Run Prism and Topograph stages concurrently"),
    resume: bool = typer.Option(False, "--resume", help="Resume incomplete cases in an existing workspace"),
    dry_run: bool = typer.Option(False, "--dry-run", help="Only generate configs and print commands"),
) -> None:
    """Generate and optionally execute a Prism-vs-Topograph campaign."""

    workspace_path = Path(workspace).resolve()
    preset_name = preset or (None if pack else "local")
    preset_spec = resolve_lane_preset(preset_name) if preset_name else None
    pack_name = pack or (preset_spec.pack if preset_spec else None)

    seed_values = _parse_optional_csv_ints(seeds) or (list(preset_spec.seeds) if preset_spec else [42])
    prism_root_path = Path(prism_root).resolve()
    topograph_root_path = Path(topograph_root).resolve()
    pack_path = resolve_pack_path(pack_name)
    pack_spec = load_parity_pack(pack_path)
    budget_values = _parse_optional_csv_ints(budgets) or (
        list(preset_spec.budgets) if preset_spec else [pack_spec.budget_policy.evaluation_count]
    )
    paths, cases = prepare_campaign_cases(
        pack_name=Path(pack_path).stem,
        base_pack_path=pack_path,
        seeds=seed_values,
        budgets=budget_values,
        workspace=Path(workspace),
        topograph_root=topograph_root_path,
        lane_preset=preset_name,
    )
    runner = CampaignRunner(prism_root=prism_root_path, topograph_root=topograph_root_path)
    state_path = workspace_state_path(workspace_path)
    state_available = state_path.exists()

    if dry_run:
        typer.echo("mode\tdry-run")
        typer.echo(f"manifest\t{paths.manifest_path}")
        typer.echo(f"state\t{state_path}")
        for case in cases:
            for label, artifact_path in _campaign_artifact_paths(paths=paths, runner=runner, case=case).items():
                typer.echo(f"{label}\t{artifact_path}")
            for spec in _planned_commands(runner, case, resume=resume):
                typer.echo(json.dumps({"name": spec.name, "cwd": str(spec.cwd), "argv": spec.argv}))
        return

    if state_available:
        clear_stop_request(workspace_path)
    stale_running_cases = mark_stale_running_cases_interrupted(workspace_path) if resume and state_available else 0
    typer.echo("mode\texecute")
    typer.echo(f"manifest\t{paths.manifest_path}")
    typer.echo(f"state\t{state_path}")
    if stale_running_cases:
        typer.echo(f"interrupted\t{stale_running_cases}")
    for case in cases:
        case_id = build_case_id(case)
        if resume and state_available and _case_already_completed(workspace_path, case_id=case_id):
            typer.echo(f"skip\t{case_id}\tcompleted")
            continue
        if state_available and stop_requested(workspace_path):
            typer.echo("stop-requested\ttrue")
            break
        prism_run_dir = runner.prism_run_dir(case)
        log_dir = paths.logs_dir / f"{case.pack_name}_seed{case.seed}"
        if state_available:
            start_case(workspace_path, case_id=case_id, resume_requested=resume)
        try:
            if state_available:
                update_case_stage(workspace_path, case_id=case_id, stage="run")
            if parallel:
                for stage in _execution_stages(runner, case, resume=resume):
                    _run_stage_parallel(stage, log_dir=log_dir)
                    _raise_if_stop_requested(workspace_path, enabled=state_available, reason="stop requested after execution stage")
            else:
                for spec in _execution_commands(runner, case, resume=resume):
                    _run_command(spec, log_dir=log_dir)
                _raise_if_stop_requested(workspace_path, enabled=state_available, reason="stop requested after execution stage")
            if state_available:
                update_case_stage(workspace_path, case_id=case_id, stage="compare")
            runner.compare_exports(
                left_dir=prism_run_dir,
                right_dir=case.topograph_run_dir,
                pack_path=case.pack_path,
                output_path=case.comparison_output_path,
            )
            if state_available:
                integrity_issues = case_integrity_issues(case)
                complete_case(workspace_path, case_id=case_id, integrity_issues=integrity_issues)
            typer.echo(f"compared\t{case.comparison_output_path}")
            for label, artifact_path in _campaign_artifact_paths(paths=paths, runner=runner, case=case).items():
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


def _campaign_artifact_paths(*, paths, runner, case) -> dict[str, Path]:
    return {
        "report": case.comparison_output_path,
        "report_json": case.comparison_output_path.with_suffix(".json"),
        "prism_run_dir": runner.prism_run_dir(case),
        "topograph_run_dir": case.topograph_run_dir,
        "log_dir": paths.logs_dir / f"{case.pack_name}_seed{case.seed}",
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


def _planned_commands(runner, case, *, resume: bool):
    try:
        return runner.planned_commands(case, resume=resume)
    except TypeError as exc:
        if "resume" not in str(exc):
            raise
        return runner.planned_commands(case)


def _execution_stages(runner, case, *, resume: bool):
    try:
        return runner.execution_stages(case, resume=resume)
    except TypeError as exc:
        if "resume" not in str(exc):
            raise
        return runner.execution_stages(case)


def _execution_commands(runner, case, *, resume: bool):
    try:
        return runner.execution_commands(case, resume=resume)
    except TypeError as exc:
        if "resume" not in str(exc):
            raise
        return runner.execution_commands(case)


def _case_already_completed(workspace: Path, *, case_id: str) -> bool:
    from evonn_compare.orchestration.campaign_state import require_workspace_state

    state = require_workspace_state(workspace)
    for case in state.get("cases", []):
        if case.get("case_id") == case_id:
            return case.get("status") == "completed"
    return False


def _raise_if_stop_requested(workspace: Path, *, enabled: bool, reason: str) -> None:
    if enabled and stop_requested(workspace):
        raise StopRequested(reason)


def _run_command(spec, *, log_dir: Path) -> None:
    log_dir.mkdir(parents=True, exist_ok=True)
    log_path = log_dir / f"{spec.name}.log"
    process = subprocess.run(spec.argv, cwd=spec.cwd, text=True, capture_output=True)
    output = (process.stdout or "") + (process.stderr or "")
    log_path.write_text(output, encoding="utf-8")
    if process.returncode != 0:
        raise RuntimeError(f"{spec.name} failed; see {log_path}")


def _run_stage_parallel(specs, *, log_dir: Path) -> None:
    errors: list[Exception] = []
    with ThreadPoolExecutor(max_workers=len(specs)) as pool:
        futures = [
            pool.submit(_run_command, spec, log_dir=log_dir)
            for spec in specs
        ]
        for future in futures:
            try:
                future.result()
            except Exception as exc:
                errors.append(exc)
    if errors:
        raise errors[0]
