"""Campaign CLI for Prism vs Topograph execution."""

from __future__ import annotations

from concurrent.futures import ThreadPoolExecutor
import json
import subprocess
from pathlib import Path

import typer

from evonn_compare.contracts.parity import resolve_pack_path
from evonn_compare.orchestration.config_gen import prepare_campaign_cases
from evonn_compare.orchestration.runner import CampaignRunner


def campaign(
    pack: str = typer.Option(..., "--pack", help="Parity pack name or YAML path"),
    seeds: str = typer.Option("42", "--seeds", help="Comma-separated seeds"),
    budgets: str = typer.Option("64", "--budgets", help="Comma-separated budgets"),
    workspace: str = typer.Option(..., "--workspace", help="Campaign workspace"),
    prism_root: str = typer.Option("../EvoNN-Prism", "--prism-root"),
    topograph_root: str = typer.Option("../EvoNN-Topograph", "--topograph-root"),
    parallel: bool = typer.Option(True, "--parallel/--serial", help="Run Prism and Topograph stages concurrently"),
    dry_run: bool = typer.Option(False, "--dry-run", help="Only generate configs and print commands"),
) -> None:
    """Generate and optionally execute a Prism-vs-Topograph campaign."""

    seed_values = _parse_csv_ints(seeds)
    budget_values = _parse_csv_ints(budgets)
    prism_root_path = Path(prism_root).resolve()
    topograph_root_path = Path(topograph_root).resolve()
    pack_path = resolve_pack_path(pack)
    paths, cases = prepare_campaign_cases(
        pack_name=Path(pack_path).stem,
        base_pack_path=pack_path,
        seeds=seed_values,
        budgets=budget_values,
        workspace=Path(workspace),
        topograph_root=topograph_root_path,
    )
    runner = CampaignRunner(prism_root=prism_root_path, topograph_root=topograph_root_path)

    if dry_run:
        typer.echo("mode\tdry-run")
        for case in cases:
            for spec in runner.planned_commands(case):
                typer.echo(json.dumps({"name": spec.name, "cwd": str(spec.cwd), "argv": spec.argv}))
        return

    typer.echo("mode\texecute")
    for case in cases:
        prism_run_dir = runner.prism_run_dir(case)
        log_dir = paths.logs_dir / f"{case.pack_name}_seed{case.seed}"
        if parallel:
            for stage in runner.execution_stages(case):
                _run_stage_parallel(stage, log_dir=log_dir)
        else:
            for spec in runner.execution_commands(case):
                _run_command(spec, log_dir=log_dir)
        runner.compare_exports(
            left_dir=prism_run_dir,
            right_dir=case.topograph_run_dir,
            pack_path=case.pack_path,
            output_path=case.comparison_output_path,
        )
        typer.echo(f"compared\t{case.comparison_output_path}")
def _parse_csv_ints(raw: str) -> list[int]:
    values = [item.strip() for item in raw.split(",") if item.strip()]
    if not values:
        raise ValueError("expected at least one integer value")
    return [int(item) for item in values]


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
