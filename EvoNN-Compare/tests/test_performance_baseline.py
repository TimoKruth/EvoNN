import json
from pathlib import Path

from typer.testing import CliRunner

from evonn_compare.cli.main import app
from evonn_compare.orchestration.performance_baseline import build_performance_baseline


runner = CliRunner()


def _write_run(run_dir: Path, *, system: str, run_id: str, budget: int, wall_clock: float, quality: float = 0.9, seed: int = 42, pack_name: str | None = None, backend: str = "mlx", hardware_class: str = "apple_silicon", lane_operating_state: str = "trusted-core", code_version: str = "deadbeef") -> None:
    run_dir.mkdir(parents=True)
    (run_dir / "config.yaml").write_text("seed: 42\n", encoding="utf-8")
    (run_dir / "report.md").write_text("# report\n", encoding="utf-8")
    (run_dir / "summary.json").write_text(
        json.dumps(
            {
                "system": system,
                "run_id": run_id,
                "runtime_backend": backend,
                "runtime_backend_requested": "auto",
                "wall_clock_seconds": wall_clock,
                "median_benchmark_quality": quality,
                "lane_operating_state": lane_operating_state,
                "engine_evidence": _engine_evidence(system),
            }
        ),
        encoding="utf-8",
    )
    (run_dir / "manifest.json").write_text(
        json.dumps(
            {
                "schema_version": "1.0",
                "system": system,
                "run_id": run_id,
                "run_name": run_id,
                "created_at": "2026-05-06T00:00:00Z",
                "pack_name": pack_name or "tier1_core",
                "seed": seed,
                "benchmarks": [
                    {
                        "benchmark_id": "iris",
                        "task_kind": "classification",
                        "metric_name": "accuracy",
                        "metric_direction": "max",
                        "status": "ok",
                    }
                ],
                "budget": {
                    "evaluation_count": budget,
                    "epochs_per_candidate": 20,
                    "wall_clock_seconds": wall_clock,
                    "budget_policy_name": "prototype_equal_budget",
                    "actual_evaluations": budget,
                    "cached_evaluations": 0,
                    "failed_evaluations": 0,
                    "invalid_evaluations": 0,
                    "evaluation_semantics": "one candidate evaluation",
                },
                "device": {
                    "device_name": hardware_class,
                    "precision_mode": "fp32",
                    "framework": backend,
                    "framework_version": "0.31.1",
                },
                "artifacts": {
                    "config_snapshot": "config.yaml",
                    "report_markdown": "report.md",
                },
                "fairness": {
                    "benchmark_pack_id": pack_name or "tier1_core",
                    "seed": seed,
                    "evaluation_count": budget,
                    "budget_policy_name": "prototype_equal_budget",
                    "data_signature": "abc",
                    "code_version": code_version,
                },
            }
        ),
        encoding="utf-8",
    )
    (run_dir / "results.json").write_text(
        json.dumps(
            [
                {
                    "system": system,
                    "run_id": run_id,
                    "benchmark_id": "iris",
                    "metric_name": "accuracy",
                    "metric_direction": "max",
                    "metric_value": quality,
                    "quality": quality,
                    "parameter_count": 10,
                    "train_seconds": 2.0,
                    "status": "ok",
                }
            ]
        ),
        encoding="utf-8",
    )


def _engine_evidence(system: str) -> dict[str, object]:
    return {
        "prism": {"family_distribution": {"mlp": 1}, "family_benchmark_wins": {"mlp": 1}, "operator_mix": {"mutation": 1}},
        "topograph": {"topology_size": {"population_count": 1}, "parallel_cache_behavior": {"cache_reuse_rate": 0.0}, "mutation_pressure_policy": "scheduled", "topology_selection_policy": "fitness_plus_topology_diversity_elites"},
        "stratograph": {"macro_depth": 2, "cell_library_size": 4, "reuse_ratio": 0.5, "hierarchy_evidence": {"genome_count": 1}},
        "primordia": {"primitive_count": 64, "primitive_bank_size": 3, "primitive_usage": {"mlp": 3}, "group_counts": {"tabular": 1}},
        "contenders": {"contender_family_coverage": {"tabular": ["logistic"]}, "optional_dependency_skips": {}, "baseline_floor_policy_stage": "fixed_reference", "baseline_floor_evidence": {"successful_winner_count": 1}},
    }[system]


def test_build_performance_baseline_writes_bundle(tmp_path: Path) -> None:
    runs_root = tmp_path / "runs"
    for budget, wall in [(64, 10.0), (256, 20.0), (1000, 50.0)]:
        for system in ("prism", "topograph", "stratograph", "primordia", "contenders"):
            _write_run(runs_root / f"{system}{budget}", system=system, run_id=f"{system}{budget}", budget=budget, wall_clock=wall)

    result = build_performance_baseline(inputs=[runs_root], output_root=tmp_path / "baselines")

    payload = json.loads(Path(result["json"]).read_text(encoding="utf-8"))
    prism = next(row for row in payload["systems"] if row["system"] == "prism")
    contenders = next(row for row in payload["systems"] if row["system"] == "contenders")

    assert result["accepted_run_count"] == 15
    assert prism["performance_claim_ready"] is True
    assert prism["budgets_present"] == [64, 256, 1000]
    assert contenders["performance_claim_ready"] is True
    assert payload["code_version_tag"] == "deadbeef"
    assert Path(result["markdown"]).exists()
    assert Path(result["jsonl"]).exists()


def test_performance_baseline_rejects_mixed_cohorts_even_when_budget_set_exists(tmp_path: Path) -> None:
    runs_root = tmp_path / "runs"
    _write_run(runs_root / "prism64", system="prism", run_id="prism64", budget=64, wall_clock=10.0, pack_name="tier_b_core")
    _write_run(runs_root / "prism256", system="prism", run_id="prism256", budget=256, wall_clock=20.0, pack_name="tier1_core")
    _write_run(runs_root / "prism1000", system="prism", run_id="prism1000", budget=1000, wall_clock=50.0, pack_name="tier1_core")

    result = build_performance_baseline(inputs=[runs_root], output_root=tmp_path / "baselines")
    payload = json.loads(Path(result["json"]).read_text(encoding="utf-8"))
    prism = next(row for row in payload["systems"] if row["system"] == "prism")

    assert prism["performance_claim_ready"] is False
    assert any("incomplete-system-cohort" in ",".join(row["reasons"]) or "pack" in ",".join(row["reasons"]) for row in prism["excluded_runs"])


def test_performance_baseline_cli_accepts_workspace(tmp_path: Path) -> None:
    runs_root = tmp_path / "workspace"
    for budget in (96, 384):
        for system in ("prism", "topograph", "stratograph", "primordia", "contenders"):
            _write_run(runs_root / "runs" / f"{system}{budget}", system=system, run_id=f"{system}{budget}", budget=budget, wall_clock=10.0 + budget / 100.0, pack_name="tier_b_core")

    result = runner.invoke(app, ["performance-baseline", str(runs_root), "--output-root", str(tmp_path / "perf"), "--budgets", "96,384"])

    assert result.exit_code == 0
    assert "bundle_root\t" in result.stdout
    assert "accepted_runs\t10" in result.stdout
