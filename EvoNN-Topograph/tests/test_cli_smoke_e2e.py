from __future__ import annotations

import json
from pathlib import Path

import yaml
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


def test_cli_tiny_smoke_numpy_fallback_end_to_end(tmp_path: Path):
    root = Path(__file__).resolve().parents[1]
    source_config = root / "configs" / "tiny_smoke" / "config.yaml"
    pack_path = root / "parity_packs" / "tiny_smoke.yaml"
    config_payload = yaml.safe_load(source_config.read_text(encoding="utf-8"))
    config_payload["runtime"] = {"backend": "numpy-fallback", "allow_fallback": True}
    config_path = tmp_path / "config.yaml"
    config_path.write_text(yaml.safe_dump(config_payload, sort_keys=False), encoding="utf-8")
    run_dir = tmp_path / "tiny-smoke-fallback"

    evolve = runner.invoke(app, ["evolve", "-c", str(config_path), "--run-dir", str(run_dir)])
    assert evolve.exit_code == 0, evolve.stdout

    inspect = runner.invoke(app, ["inspect", str(run_dir)])
    assert inspect.exit_code == 0, inspect.stdout
    assert "numpy-fallback" in inspect.stdout

    symbiosis = runner.invoke(
        app,
        ["symbiosis", "export", str(run_dir), "--pack", str(pack_path)],
    )
    assert symbiosis.exit_code == 0, symbiosis.stdout

    summary = json.loads((run_dir / "summary.json").read_text(encoding="utf-8"))
    manifest = json.loads((run_dir / "manifest.json").read_text(encoding="utf-8"))
    assert summary["runtime_backend"] == "numpy-fallback"
    assert summary["runtime_backend_requested"] == "numpy-fallback"
    assert manifest["budget"]["benchmark_slot_integrity"]["status"] == "complete"


def test_cli_tiny_lm_smoke_end_to_end(tmp_path: Path):
    root = Path(__file__).resolve().parents[1]
    config_path = root / "configs" / "tiny_lm_smoke" / "config.yaml"
    run_dir = tmp_path / "tiny-lm-smoke"

    evolve = runner.invoke(app, ["evolve", "-c", str(config_path), "--run-dir", str(run_dir)])
    assert evolve.exit_code == 0, evolve.stdout

    inspect = runner.invoke(app, ["inspect", str(run_dir)])
    assert inspect.exit_code == 0, inspect.stdout
    assert (run_dir / "metrics.duckdb").exists()


def test_cli_tiny_pool_smoke_end_to_end(tmp_path: Path):
    root = Path(__file__).resolve().parents[1]
    config_path = root / "configs" / "tiny_pool_smoke" / "config.yaml"
    run_dir = tmp_path / "tiny-pool-smoke"

    evolve = runner.invoke(app, ["evolve", "-c", str(config_path), "--run-dir", str(run_dir)])
    assert evolve.exit_code == 0, evolve.stdout

    report = runner.invoke(app, ["report", str(run_dir)])
    assert report.exit_code == 0, report.stdout
    assert "Sampled Benchmark Order" in report.stdout
