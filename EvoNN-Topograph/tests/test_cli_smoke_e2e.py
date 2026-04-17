from __future__ import annotations

import json
from pathlib import Path

from typer.testing import CliRunner

from topograph.cli import app


runner = CliRunner()


def test_cli_tiny_smoke_end_to_end(tmp_path: Path):
    root = Path(__file__).resolve().parents[1]
    config_path = root / "configs" / "tiny_smoke" / "config.yaml"
    pack_path = root / "parity_packs" / "tiny_smoke.yaml"
    run_dir = tmp_path / "tiny-smoke"

    evolve = runner.invoke(app, ["evolve", "-c", str(config_path), "--run-dir", str(run_dir)])
    assert evolve.exit_code == 0, evolve.stdout

    inspect = runner.invoke(app, ["inspect", str(run_dir)])
    assert inspect.exit_code == 0, inspect.stdout

    report = runner.invoke(app, ["report", str(run_dir)])
    assert report.exit_code == 0, report.stdout

    export = runner.invoke(app, ["export", str(run_dir)])
    assert export.exit_code == 0, export.stdout

    symbiosis = runner.invoke(
        app,
        ["symbiosis", "export", str(run_dir), "--pack", str(pack_path)],
    )
    assert symbiosis.exit_code == 0, symbiosis.stdout

    assert (run_dir / "metrics.duckdb").exists()
    assert (run_dir / "report.md").exists()
    assert (run_dir / "export.json").exists()
    assert (run_dir / "summary.json").exists()
    assert (run_dir / "manifest.json").exists()
    assert (run_dir / "results.json").exists()

    export_payload = json.loads((run_dir / "export.json").read_text(encoding="utf-8"))
    summary = json.loads((run_dir / "summary.json").read_text(encoding="utf-8"))

    assert export_payload["system"] == "topograph"
    assert export_payload["population_size"] == 1
    assert summary["system"] == "topograph"
    assert summary["generations_completed"] == 1
    assert summary["benchmarks_evaluated"] == 1
