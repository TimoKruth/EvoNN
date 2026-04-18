"""Contender baseline resolution helpers for compare-side orchestration."""

from __future__ import annotations

import subprocess
from dataclasses import dataclass
from pathlib import Path


@dataclass(frozen=True)
class ContenderArtifacts:
    run_dir: Path
    manifest_path: Path
    results_path: Path


def ensure_contender_run(
    *,
    contenders_root: Path,
    config_path: Path,
    run_dir: Path,
    log_dir: Path | None = None,
) -> Path:
    """Materialize contender run from cache, or fill missing baseline entries on demand."""

    run_dir = run_dir.resolve()
    contenders_root = contenders_root.resolve()
    config_path = config_path.resolve()

    materialize = [
        "uv",
        "run",
        "evonn-contenders",
        "materialize",
        "--config",
        str(config_path),
        "--run-dir",
        str(run_dir),
    ]
    outcome = _run(
        materialize,
        cwd=contenders_root,
        allow_failure=True,
        log_path=None if log_dir is None else log_dir / "contenders_materialize.log",
    )
    if outcome.returncode == 0:
        return run_dir
    if "missing from baseline cache" not in outcome.stderr and "missing from baseline cache" not in outcome.stdout:
        raise RuntimeError(
            f"contender materialize failed in {contenders_root}:\n{(outcome.stdout or '')}{(outcome.stderr or '')}"
        )

    run_cmd = [
        "uv",
        "run",
        "evonn-contenders",
        "run",
        "--config",
        str(config_path),
        "--run-dir",
        str(run_dir),
    ]
    _run(
        run_cmd,
        cwd=contenders_root,
        log_path=None if log_dir is None else log_dir / "contenders_run.log",
    )
    return run_dir


def ensure_contender_export(
    *,
    contenders_root: Path,
    config_path: Path,
    pack_path: Path,
    run_dir: Path,
    output_dir: Path | None = None,
    log_dir: Path | None = None,
) -> ContenderArtifacts:
    """Ensure contender run exists, then export compare-layer artifacts."""

    run_dir = ensure_contender_run(
        contenders_root=contenders_root,
        config_path=config_path,
        run_dir=run_dir,
        log_dir=log_dir,
    )
    resolved_output_dir = run_dir if output_dir is None else output_dir.resolve()
    resolved_output_dir.mkdir(parents=True, exist_ok=True)
    export_cmd = [
        "uv",
        "run",
        "evonn-contenders",
        "symbiosis",
        "export",
        "--run-dir",
        str(run_dir),
        "--pack-path",
        str(pack_path.resolve()),
        "--output-dir",
        str(resolved_output_dir),
    ]
    _run(
        export_cmd,
        cwd=contenders_root.resolve(),
        log_path=None if log_dir is None else log_dir / "contenders_export.log",
    )
    return ContenderArtifacts(
        run_dir=run_dir,
        manifest_path=resolved_output_dir / "manifest.json",
        results_path=resolved_output_dir / "results.json",
    )


def _run(
    argv: list[str],
    *,
    cwd: Path,
    allow_failure: bool = False,
    log_path: Path | None = None,
) -> subprocess.CompletedProcess[str]:
    if log_path is None:
        process = subprocess.run(argv, cwd=cwd, text=True, capture_output=True)
        output = (process.stdout or "") + (process.stderr or "")
    else:
        log_path.parent.mkdir(parents=True, exist_ok=True)
        with log_path.open("w", encoding="utf-8") as handle:
            handle.write(f"$ {' '.join(argv)}\n\n")
            handle.flush()
            process = subprocess.run(
                argv,
                cwd=cwd,
                text=True,
                stdout=handle,
                stderr=subprocess.STDOUT,
            )
        output = (
            log_path.read_text(encoding="utf-8")
            + (process.stdout or "")
            + (process.stderr or "")
        )
        process = subprocess.CompletedProcess(argv, process.returncode, stdout=output, stderr=process.stderr or "")
    if not allow_failure and process.returncode != 0:
        raise RuntimeError(f"command failed in {cwd}:\n{output}")
    return process
