from __future__ import annotations

import json

import yaml
from typer.testing import CliRunner

from stratograph.cli import app
from stratograph.benchmarks import get_benchmark
from stratograph.config import BenchmarkPoolConfig, load_config
from stratograph.pipeline import run_evolution
from stratograph.runtime import compile_genome
from stratograph.runtime.backends import (
    FALLBACK_LIMITATIONS,
    resolve_runtime_backend,
    resolve_runtime_backend_with_policy,
)
from stratograph.storage import RunStore


def test_resolve_runtime_backend_rejects_missing_mlx(monkeypatch) -> None:
    monkeypatch.setattr("stratograph.runtime.backends.MLX_AVAILABLE", False)
    monkeypatch.setattr("stratograph.runtime.backends.MLX_VERSION", None)
    try:
        resolve_runtime_backend("mlx")
    except RuntimeError as exc:
        assert "MLX is not available" in str(exc)
    else:  # pragma: no cover - defensive
        raise AssertionError("expected explicit mlx request to fail when MLX is unavailable")


def test_resolve_runtime_backend_rejects_auto_when_fallback_disabled(monkeypatch) -> None:
    monkeypatch.setattr("stratograph.runtime.backends.MLX_AVAILABLE", False)
    monkeypatch.setattr("stratograph.runtime.backends.MLX_VERSION", None)
    try:
        resolve_runtime_backend_with_policy("auto", allow_fallback=False)
    except RuntimeError as exc:
        assert "fallback is disabled" in str(exc)
    else:  # pragma: no cover - defensive
        raise AssertionError("expected auto backend to fail when fallback is disabled")


def test_compile_genome_can_force_numpy_fallback(repo_root) -> None:
    config = load_config(repo_root / "configs" / "working_33_plus_5_lm_smoke.yaml")
    spec = get_benchmark("moons")
    genome = config.evolution  # keep config in scope for deterministic setup
    del genome
    run_config = config.model_copy(
        update={
            "benchmark_pool": BenchmarkPoolConfig(name="compile_probe", benchmarks=["moons"]),
        }
    )
    sample_genome = run_config.model_copy  # keep linter quiet if config evolves
    del sample_genome
    from stratograph.pipeline.coordinator import _make_candidate

    candidate = _make_candidate(
        benchmark_name="moons",
        task=spec.task,
        input_dim=spec.model_input_dim,
        output_dim=spec.model_output_dim,
        seed=run_config.seed,
        candidate_index=0,
        architecture_mode=run_config.evolution.architecture_mode,
    )
    compiled = compile_genome(candidate, runtime_backend="numpy-fallback")
    encoded = compiled.encode(spec.load_data(seed=run_config.seed)[0][:4])
    assert compiled.runtime_backend == "numpy-fallback"
    assert "backend=numpy-fallback" in compiled.architecture_summary()
    assert encoded.shape[0] == 4


def test_run_evolution_records_requested_and_resolved_runtime(repo_root, tmp_path) -> None:
    base_config = load_config(repo_root / "configs" / "working_33_plus_5_lm_smoke.yaml")
    config = base_config.model_copy(
        update={
            "run_name": "runtime_backend_probe",
            "benchmark_pool": BenchmarkPoolConfig(name="runtime_backend_probe", benchmarks=["moons", "tiny_lm_synthetic"]),
            "runtime": base_config.runtime.model_copy(update={"backend": "numpy-fallback"}),
            "evolution": base_config.evolution.model_copy(update={"population_size": 2, "generations": 1}),
        }
    )
    config_path = tmp_path / "runtime_backend_probe.yaml"
    config_path.write_text(yaml.safe_dump(config.model_dump(mode="python"), sort_keys=False), encoding="utf-8")
    run_dir = tmp_path / "runtime_backend_probe"
    run_evolution(config, run_dir=run_dir, config_path=config_path)

    with RunStore(run_dir / "metrics.duckdb") as store:
        budget_meta = store.load_budget_metadata(run_dir.name)

    export_pack_path = tmp_path / "runtime_backend_probe_pack.yaml"
    export_pack_path.write_text(
        yaml.safe_dump(
            {
                "name": "runtime_backend_probe",
                "benchmarks": [
                    {
                        "benchmark_id": "moons_classification",
                        "native_ids": {"stratograph": "moons"},
                        "task_kind": "classification",
                        "metric_name": "accuracy",
                        "metric_direction": "max",
                    },
                    {
                        "benchmark_id": "tiny_lm_synthetic",
                        "native_ids": {"stratograph": "tiny_lm_synthetic"},
                        "task_kind": "language_modeling",
                        "metric_name": "perplexity",
                        "metric_direction": "min",
                    },
                ],
            },
            sort_keys=False,
        ),
        encoding="utf-8",
    )
    from stratograph.export import export_symbiosis_contract

    export_symbiosis_contract(run_dir, export_pack_path)

    summary = json.loads((run_dir / "summary.json").read_text(encoding="utf-8"))
    report = (run_dir / "report.md").read_text(encoding="utf-8")

    assert budget_meta["runtime_backend_requested"] == "numpy-fallback"
    assert budget_meta["runtime_backend"] == "numpy-fallback"
    assert budget_meta["runtime_backend_limitations"] == FALLBACK_LIMITATIONS
    assert summary["runtime_backend"] == "numpy-fallback"
    assert summary["requested_runtime_backend"] == "numpy-fallback"
    assert summary["runtime_backend_limitations"] == FALLBACK_LIMITATIONS
    assert "- Requested Runtime: `numpy-fallback`" in report
    assert "- Runtime Limitations: `" in report
    assert "- Parallelism: `serial` (requested=`1`, effective=`1`)" in report

    runner = CliRunner()
    inspect_result = runner.invoke(app, ["inspect", "--run-dir", str(run_dir)])
    assert inspect_result.exit_code == 0
    assert "Requested Runtime" in inspect_result.stdout
    assert "numpy-fallback" in inspect_result.stdout
    assert "Runtime Limitations" in inspect_result.stdout
    assert "Parallelism" in inspect_result.stdout


def test_resolve_parallelism_forces_serial_for_non_fallback_runtime() -> None:
    from stratograph.pipeline.coordinator import _resolve_parallelism

    parallelism = _resolve_parallelism(requested_workers=4, benchmark_count=3, runtime_backend="mlx")

    assert parallelism["mode"] == "serial"
    assert parallelism["effective_workers"] == 1
    assert parallelism["reason"] == "non_fallback_runtime_forced_serial"


def test_run_evolution_records_benchmark_parallelism_metadata(repo_root, tmp_path) -> None:
    base_config = load_config(repo_root / "configs" / "working_33_plus_5_lm_smoke.yaml")
    config = base_config.model_copy(
        update={
            "run_name": "runtime_parallel_probe",
            "benchmark_pool": BenchmarkPoolConfig(name="runtime_parallel_probe", benchmarks=["moons", "tiny_lm_synthetic"]),
            "runtime": base_config.runtime.model_copy(update={"backend": "numpy-fallback"}),
            "evolution": base_config.evolution.model_copy(
                update={"population_size": 1, "generations": 1, "benchmark_parallel_workers": 2}
            ),
        }
    )
    config_path = tmp_path / "runtime_parallel_probe.yaml"
    config_path.write_text(yaml.safe_dump(config.model_dump(mode="python"), sort_keys=False), encoding="utf-8")
    run_dir = tmp_path / "runtime_parallel_probe"
    run_evolution(config, run_dir=run_dir, config_path=config_path)

    with RunStore(run_dir / "metrics.duckdb") as store:
        budget_meta = store.load_budget_metadata(run_dir.name)

    export_pack_path = tmp_path / "runtime_parallel_probe_pack.yaml"
    export_pack_path.write_text(
        yaml.safe_dump(
            {
                "name": "runtime_parallel_probe",
                "benchmarks": [
                    {
                        "benchmark_id": "moons_classification",
                        "native_ids": {"stratograph": "moons"},
                        "task_kind": "classification",
                        "metric_name": "accuracy",
                        "metric_direction": "max",
                    },
                    {
                        "benchmark_id": "tiny_lm_synthetic",
                        "native_ids": {"stratograph": "tiny_lm_synthetic"},
                        "task_kind": "language_modeling",
                        "metric_name": "perplexity",
                        "metric_direction": "min",
                    },
                ],
            },
            sort_keys=False,
        ),
        encoding="utf-8",
    )
    from stratograph.export import export_symbiosis_contract

    export_symbiosis_contract(run_dir, export_pack_path)

    summary = json.loads((run_dir / "summary.json").read_text(encoding="utf-8"))
    report = (run_dir / "report.md").read_text(encoding="utf-8")

    assert budget_meta["parallelism_mode"] == "benchmark-process"
    assert budget_meta["benchmark_parallel_workers_requested"] == 2
    assert budget_meta["benchmark_parallel_workers_effective"] == 2
    assert budget_meta["parallelism_reason"] == "benchmark_process_parallelism"
    assert summary["parallelism_mode"] == "benchmark-process"
    assert summary["benchmark_parallel_workers_requested"] == 2
    assert summary["benchmark_parallel_workers_effective"] == 2
    assert "- Parallelism: `benchmark-process` (requested=`2`, effective=`2`)" in report

    runner = CliRunner()
    inspect_result = runner.invoke(app, ["inspect", "--run-dir", str(run_dir)])
    assert inspect_result.exit_code == 0
    assert "benchmark-process (requested=2, effective=2)" in inspect_result.stdout


def test_parallel_worker_failure_marks_partial_run_and_keeps_successes(repo_root, tmp_path, monkeypatch) -> None:
    from stratograph.pipeline import coordinator

    base_config = load_config(repo_root / "configs" / "working_33_plus_5_lm_smoke.yaml")
    config = base_config.model_copy(
        update={
            "run_name": "runtime_parallel_failure_probe",
            "benchmark_pool": BenchmarkPoolConfig(
                name="runtime_parallel_failure_probe",
                benchmarks=["moons", "tiny_lm_synthetic"],
            ),
            "runtime": base_config.runtime.model_copy(update={"backend": "numpy-fallback"}),
            "evolution": base_config.evolution.model_copy(
                update={"population_size": 1, "generations": 1, "benchmark_parallel_workers": 2}
            ),
        }
    )
    config_path = tmp_path / "runtime_parallel_failure_probe.yaml"
    config_path.write_text(yaml.safe_dump(config.model_dump(mode="python"), sort_keys=False), encoding="utf-8")
    run_dir = tmp_path / "runtime_parallel_failure_probe"

    class _FakeFuture:
        def __init__(self, *, value=None, error: Exception | None = None) -> None:
            self._value = value
            self._error = error

        def result(self):
            if self._error is not None:
                raise self._error
            return self._value

    class _FakeExecutor:
        def __init__(self, *args, **kwargs) -> None:
            del args, kwargs

        def __enter__(self):
            return self

        def __exit__(self, exc_type, exc, tb) -> None:
            del exc_type, exc, tb

        def submit(self, fn, config_payload, benchmark_name, architecture_mode, runtime_backend):
            if benchmark_name == "tiny_lm_synthetic":
                return _FakeFuture(error=RuntimeError("worker exploded"))
            return _FakeFuture(value=fn(config_payload, benchmark_name, architecture_mode, runtime_backend))

    monkeypatch.setattr(coordinator, "ProcessPoolExecutor", _FakeExecutor)
    monkeypatch.setattr(coordinator, "as_completed", lambda futures: list(futures))

    run_evolution(config, run_dir=run_dir, config_path=config_path)

    with RunStore(run_dir / "metrics.duckdb") as store:
        budget_meta = store.load_budget_metadata(run_dir.name)
        latest_results = store.load_latest_results(run_dir.name)

    assert budget_meta["evaluation_count"] == 1
    assert budget_meta["configured_evaluation_slots"] == 2
    assert budget_meta["failed_evaluations"] == 1
    assert budget_meta["partial_run"] is True
    assert budget_meta["completed_benchmark_count"] == 1
    assert {row["benchmark_name"]: row["status"] for row in latest_results} == {
        "moons": "ok",
        "tiny_lm_synthetic": "failed",
    }
    assert any(row["failure_reason"] == "worker exploded" for row in latest_results if row["status"] == "failed")
