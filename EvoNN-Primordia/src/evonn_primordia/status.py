"""Status/checkpoint helpers for Primordia runs."""
from __future__ import annotations

import json
from pathlib import Path
from typing import Any


def write_status(
    run_dir: str | Path,
    *,
    run_id: str,
    run_name: str,
    state: str,
    total_benchmarks: int,
    completed_benchmarks: list[str],
    target_evaluation_count: int,
    evaluation_count: int,
    runtime_backend: str,
) -> Path:
    run_dir = Path(run_dir)
    payload = {
        "run_id": run_id,
        "run_name": run_name,
        "state": state,
        "total_benchmarks": total_benchmarks,
        "completed_benchmarks": completed_benchmarks,
        "completed_count": len(completed_benchmarks),
        "remaining_count": max(0, total_benchmarks - len(completed_benchmarks)),
        "target_evaluation_count": target_evaluation_count,
        "evaluation_count": evaluation_count,
        "runtime_backend": runtime_backend,
    }
    path = run_dir / "status.json"
    path.write_text(json.dumps(payload, indent=2), encoding="utf-8")
    return path


def write_checkpoint(
    run_dir: str | Path,
    *,
    payload: dict[str, Any],
) -> Path:
    run_dir = Path(run_dir)
    path = run_dir / "checkpoint.json"
    path.write_text(json.dumps(payload, indent=2), encoding="utf-8")
    return path


def load_checkpoint(run_dir: str | Path) -> dict[str, Any] | None:
    run_dir = Path(run_dir)
    path = run_dir / "checkpoint.json"
    if not path.exists():
        return None
    return json.loads(path.read_text(encoding="utf-8"))
