"""Primordia resolution helpers for compare-side orchestration."""
from __future__ import annotations

import subprocess
from dataclasses import dataclass
from pathlib import Path


@dataclass(frozen=True)
class PrimordiaArtifacts:
    run_dir: Path
    manifest_path: Path
    results_path: Path


def ensure_primordia_run(
    *,
    primordia_root: Path,
    config_path: Path,
    run_dir: Path,
    log_dir: Path | None = None,
) -> Path:
    """Ensure a Primordia run exists."""

    run_dir = run_dir.resolve()
    primordia_root = primordia_root.resolve()
    config_path = config_path.resolve()
    run_cmd = [
        "uv",
        "run",
        "primordia",
        "run",
        "--config",
        str(config_path),
        "--run-dir",
        str(run_dir),
    ]
    _run(
        run_cmd,
        cwd=primordia_root,
        log_path=None if log_dir is None else log_dir / "primordia_run.log",
    )
    return run_dir


def ensure_primordia_export(
    *,
    primordia_root: Path,
    config_path: Path,
    pack_path: Path,
    run_dir: Path,
    output_dir: Path | None = None,
    log_dir: Path | None = None,
) -> PrimordiaArtifacts:
    """Ensure Primordia run exists, then export compare-layer artifacts."""

    run_dir = ensure_primordia_run(
        primordia_root=primordia_root,
        config_path=config_path,
        run_dir=run_dir,
        log_dir=log_dir,
    )
    resolved_output_dir = run_dir if output_dir is None else output_dir.resolve()
    resolved_output_dir.mkdir(parents=True, exist_ok=True)
    export_cmd = [
        "uv",
        "run",
        "primordia",
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
        cwd=primordia_root.resolve(),
        log_path=None if log_dir is None else log_dir / "primordia_export.log",
    )
    return PrimordiaArtifacts(
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
