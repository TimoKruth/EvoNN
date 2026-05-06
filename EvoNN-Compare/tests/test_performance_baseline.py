import json
from pathlib import Path

from typer.testing import CliRunner

from evonn_compare.cli.main import app
from evonn_compare.orchestration.performance_baseline import build_performance_baseline


runner = CliRunner()

SYSTEMS = ("prism", "topograph", "stratograph", "primordia", "contenders")
BACKEND_BY_SYSTEM = {
    "prism": "mlx",
    "topograph": "mlx",
    "stratograph": "mlx",
    "primordia": "mlx",
    "contenders": "scikit-learn",
}
HARDWARE_BY_SYSTEM = {
    "prism": "apple_silicon",
    "topograph": "apple_silicon",
    "stratograph": "arm64",
    "primordia": "arm64",
    "contenders": "arm64",
}


def _write_run(run_dir: Path, *, system: str, run_id: str, budget: int, wall_clock: float, quality: float = 0.9, seed: int = 42, pack_name: str | None = None, backend: str = "mlx", hardware_class: str = "apple_silicon", lane_operating_state: str | None = None, code_version: str = "deadbeef") -> None:
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
                "engine_evidence": _engine_evidence(system),
            }
        ),
        encoding="utf-8",
    )
    if lane_operating_state is not None:
        payload = json.loads((run_dir / "summary.json").read_text(encoding="utf-8"))
        payload["lane_operating_state"] = lane_operating_state
        (run_dir / "summary.json").write_text(json.dumps(payload), encoding="utf-8")
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


def _write_fair_matrix_case(workspace: Path, *, run_id: str, budget: int, seed: int, systems: tuple[str, ...], lane_operating_state: str = "trusted-core", pack_name: str | None = None) -> None:
    report_dir = workspace / "reports" / run_id
    report_dir.mkdir(parents=True, exist_ok=True)
    trend_rows = [
        {
            "pack_name": pack_name or f"tier1_core_eval{budget}",
            "budget": budget,
            "seed": seed,
            "system": system,
            "run_id": run_id,
            "benchmark_id": "iris",
            "task_kind": "classification",
            "benchmark_family": "tabular",
            "metric_name": "accuracy",
            "metric_direction": "max",
            "metric_value": 0.9,
            "architecture_summary": None,
            "outcome_status": "ok",
            "failure_reason": None,
            "evaluation_count": budget,
            "epochs_per_candidate": 20,
            "budget_policy_name": "prototype_equal_budget",
            "wall_clock_seconds": 10.0,
            "matrix_scope": "fair",
            "search_profile": system,
            "expected_specialization": system,
            "fairness_metadata": {"lane_operating_state": lane_operating_state},
            "lane_operating_state": lane_operating_state,
            "system_operating_state": "benchmark-complete",
            "lane_repeatability_ready": True,
            "lane_budget_accounting_ok": True,
        }
        for system in systems
    ]
    (report_dir / "fair_matrix_summary.json").write_text(
        json.dumps(
            {
                "pack_name": pack_name or f"tier1_core_eval{budget}",
                "systems": list(systems),
                "lane": {"operating_state": lane_operating_state, "repeatability_ready": True, "budget_accounting_ok": True},
                "trend_rows": trend_rows,
            },
            indent=2,
        ),
        encoding="utf-8",
    )


def test_build_performance_baseline_writes_bundle(tmp_path: Path) -> None:
    workspace = tmp_path / "workspace"
    runs_root = workspace / "runs"
    for budget, wall in [(64, 10.0), (256, 20.0), (1000, 50.0)]:
        run_id = f"tier1_core_eval{budget}_seed42"
        for system in SYSTEMS:
            _write_run(
                runs_root / system / run_id,
                system=system,
                run_id=run_id,
                budget=budget,
                wall_clock=wall,
                pack_name=f"tier1_core_eval{budget}",
                backend=BACKEND_BY_SYSTEM[system],
                hardware_class=HARDWARE_BY_SYSTEM[system],
            )
        _write_fair_matrix_case(workspace, run_id=run_id, budget=budget, seed=42, systems=SYSTEMS, pack_name=f"tier1_core_eval{budget}")

    result = build_performance_baseline(inputs=[workspace], output_root=tmp_path / "baselines")

    payload = json.loads(Path(result["json"]).read_text(encoding="utf-8"))
    prism = next(row for row in payload["systems"] if row["system"] == "prism")
    contenders = next(row for row in payload["systems"] if row["system"] == "contenders")

    assert result["accepted_run_count"] == 15
    assert prism["performance_claim_ready"] is True
    assert prism["budgets_present"] == [64, 256, 1000]
    assert prism["backend_labels"] == ["mlx"]
    assert contenders["backend_labels"] == ["scikit-learn"]
    assert contenders["performance_claim_ready"] is True
    assert prism["selected_comparison_cohort"] == next(iter(payload["comparison_cohorts"].keys()))
    assert "cohort_key" not in payload["runs"][0]
    assert next(iter(payload["comparison_cohorts"].values()))["backend"] == "mixed"
    assert next(iter(payload["comparison_cohorts"].values()))["hardware_class"] == "mixed"
    assert len(payload["performance_series"]) == 5
    assert payload["code_version_tag"] == "deadbeef"
    assert Path(result["markdown"]).exists()
    assert Path(result["jsonl"]).exists()


def test_performance_baseline_rejects_mixed_cohorts_even_when_budget_set_exists(tmp_path: Path) -> None:
    workspace = tmp_path / "workspace"
    runs_root = workspace / "runs"
    _write_run(runs_root / "prism" / "tier_b_core_eval64_seed42", system="prism", run_id="tier_b_core_eval64_seed42", budget=64, wall_clock=10.0, pack_name="tier_b_core_eval64")
    _write_run(runs_root / "prism" / "tier1_core_eval256_seed42", system="prism", run_id="tier1_core_eval256_seed42", budget=256, wall_clock=20.0, pack_name="tier1_core_eval256")
    _write_run(runs_root / "prism" / "tier1_core_eval1000_seed42", system="prism", run_id="tier1_core_eval1000_seed42", budget=1000, wall_clock=50.0, pack_name="tier1_core_eval1000")
    _write_fair_matrix_case(workspace, run_id="tier_b_core_eval64_seed42", budget=64, seed=42, systems=("prism",), pack_name="tier_b_core_eval64")
    _write_fair_matrix_case(workspace, run_id="tier1_core_eval256_seed42", budget=256, seed=42, systems=("prism",), pack_name="tier1_core_eval256")
    _write_fair_matrix_case(workspace, run_id="tier1_core_eval1000_seed42", budget=1000, seed=42, systems=("prism",), pack_name="tier1_core_eval1000")

    result = build_performance_baseline(inputs=[workspace], output_root=tmp_path / "baselines")
    payload = json.loads(Path(result["json"]).read_text(encoding="utf-8"))
    prism = next(row for row in payload["systems"] if row["system"] == "prism")

    assert prism["performance_claim_ready"] is False
    assert any("incomplete-system-cohort" in ",".join(row["reasons"]) or "pack" in ",".join(row["reasons"]) for row in prism["excluded_runs"])


def test_performance_baseline_flags_runtime_series_ambiguity(tmp_path: Path) -> None:
    workspace = tmp_path / "workspace"
    runs_root = workspace / "runs"
    for budget, backend, hardware_class in [(64, "mlx", "apple_silicon"), (256, "numpy", "arm64"), (1000, "mlx", "apple_silicon")]:
        run_id = f"tier1_core_eval{budget}_seed42"
        for system in SYSTEMS:
            system_backend = backend if system == "prism" else BACKEND_BY_SYSTEM[system]
            system_hardware = hardware_class if system == "prism" else HARDWARE_BY_SYSTEM[system]
            _write_run(
                runs_root / system / run_id,
                system=system,
                run_id=run_id,
                budget=budget,
                wall_clock=10.0,
                pack_name=f"tier1_core_eval{budget}",
                backend=system_backend,
                hardware_class=system_hardware,
            )
        _write_fair_matrix_case(workspace, run_id=run_id, budget=budget, seed=42, systems=SYSTEMS, pack_name=f"tier1_core_eval{budget}")

    result = build_performance_baseline(inputs=[workspace], output_root=tmp_path / "baselines")
    payload = json.loads(Path(result["json"]).read_text(encoding="utf-8"))
    prism = next(row for row in payload["systems"] if row["system"] == "prism")

    assert prism["performance_claim_ready"] is False
    assert prism["performance_claim_warnings"] == ["multiple-performance-series"]
    assert len(prism["performance_series"]) == 2
    assert all(series["has_required_budgets"] is False for series in prism["performance_series"])


def test_performance_baseline_does_not_use_summary_lane_fallback_when_matching_rows_are_ambiguous(tmp_path: Path) -> None:
    workspace = tmp_path / "workspace"
    runs_root = workspace / "runs"
    run_id = "tier1_core_eval64_seed42"
    for system in SYSTEMS:
        _write_run(
            runs_root / system / run_id,
            system=system,
            run_id=run_id,
            budget=64,
            wall_clock=10.0,
            pack_name="tier1_core_eval64",
            backend=BACKEND_BY_SYSTEM[system],
            hardware_class=HARDWARE_BY_SYSTEM[system],
        )
    _write_fair_matrix_case(workspace, run_id=run_id, budget=64, seed=42, systems=SYSTEMS, pack_name="tier1_core_eval64")

    summary_path = workspace / "reports" / run_id / "fair_matrix_summary.json"
    payload = json.loads(summary_path.read_text(encoding="utf-8"))
    prism_rows = [row for row in payload["trend_rows"] if row["system"] == "prism"]
    prism_rows[0]["lane_operating_state"] = "trusted-core"
    prism_rows[0]["fairness_metadata"]["lane_operating_state"] = "trusted-core"
    duplicate = json.loads(json.dumps(prism_rows[0]))
    duplicate["lane_operating_state"] = "trusted-extended"
    duplicate["fairness_metadata"]["lane_operating_state"] = "trusted-extended"
    payload["trend_rows"].append(duplicate)
    payload["lane"]["operating_state"] = "trusted-core"
    summary_path.write_text(json.dumps(payload, indent=2), encoding="utf-8")

    result = build_performance_baseline(inputs=[workspace], output_root=tmp_path / "baselines")
    baseline = json.loads(Path(result["json"]).read_text(encoding="utf-8"))
    prism = next(row for row in baseline["systems"] if row["system"] == "prism")

    assert prism["performance_claim_ready"] is False
    assert any("lane-state=missing" in row["reasons"] for row in prism["excluded_runs"])


def test_performance_baseline_cli_accepts_workspace(tmp_path: Path) -> None:
    runs_root = tmp_path / "workspace"
    for budget in (96, 384):
        run_id = f"tier_b_core_eval{budget}_seed42"
        for system in SYSTEMS:
            _write_run(
                runs_root / "runs" / system / run_id,
                system=system,
                run_id=run_id,
                budget=budget,
                wall_clock=10.0 + budget / 100.0,
                pack_name=f"tier_b_core_eval{budget}",
                backend=BACKEND_BY_SYSTEM[system],
                hardware_class=HARDWARE_BY_SYSTEM[system],
            )
        _write_fair_matrix_case(runs_root, run_id=run_id, budget=budget, seed=42, systems=SYSTEMS, pack_name=f"tier_b_core_eval{budget}")

    result = runner.invoke(app, ["performance-baseline", str(runs_root), "--output-root", str(tmp_path / "perf"), "--budgets", "96,384"])

    assert result.exit_code == 0
    assert "bundle_root\t" in result.stdout
    assert "accepted_runs\t10" in result.stdout
