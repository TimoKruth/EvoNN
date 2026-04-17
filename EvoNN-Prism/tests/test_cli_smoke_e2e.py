from __future__ import annotations

import json
from pathlib import Path

from typer.testing import CliRunner

from prism.cli import app


runner = CliRunner()


def test_cli_tiny_smoke_end_to_end(tmp_path: Path):
    config_path = Path(__file__).resolve().parents[1] / "configs" / "tiny_smoke" / "config.yaml"
    run_dir = tmp_path / "tiny-smoke"

    evolve = runner.invoke(app, ["evolve", "-c", str(config_path), "--run-dir", str(run_dir)])
    assert evolve.exit_code == 0, evolve.stdout

    inspect = runner.invoke(app, ["inspect", str(run_dir)])
    assert inspect.exit_code == 0, inspect.stdout

    report = runner.invoke(app, ["report", str(run_dir)])
    assert report.exit_code == 0, report.stdout

    assert (run_dir / "metrics.duckdb").exists()
    assert (run_dir / "summary.json").exists()
    assert (run_dir / "report.md").exists()

    summary = json.loads((run_dir / "summary.json").read_text(encoding="utf-8"))
    assert summary["total_evaluations"] == 1
    assert summary["best_family"] == "mlp"
