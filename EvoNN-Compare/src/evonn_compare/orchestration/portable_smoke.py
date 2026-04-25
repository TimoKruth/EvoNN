"""Portable smoke-grade compare exports for systems whose native runtime is unavailable."""
from __future__ import annotations

from dataclasses import dataclass
from datetime import datetime, timezone
import json
from pathlib import Path
import platform
import shutil
import subprocess
import time
from typing import Any, Literal

import numpy as np
import yaml
from sklearn import __version__ as SKLEARN_VERSION
from sklearn.metrics import accuracy_score, mean_squared_error
from sklearn.neural_network import MLPClassifier, MLPRegressor
from sklearn.preprocessing import StandardScaler

from evonn_compare.contracts.parity import load_parity_pack
from evonn_compare.orchestration.benchmark_resolution import resolve_supported_benchmark_id
from evonn_shared.contracts import (
    ArtifactPaths,
    BenchmarkEntry,
    BudgetEnvelope,
    DeviceInfo,
    ResultRecord,
    RunManifest,
    SearchTelemetry,
)
from evonn_shared.manifests import benchmark_signature, fairness_manifest, write_json

TaskKind = Literal["classification", "regression", "language_modeling"]


@dataclass(frozen=True)
class PortableArtifacts:
    run_dir: Path
    manifest_path: Path
    results_path: Path


def ensure_prism_portable_smoke_export(
    *,
    config_path: Path,
    pack_path: Path,
    run_dir: Path,
    output_dir: Path | None = None,
    log_dir: Path | None = None,
) -> PortableArtifacts:
    payload = yaml.safe_load(config_path.read_text(encoding="utf-8"))
    allowed_families = list(payload.get("evolution", {}).get("allowed_families") or ["mlp", "sparse_mlp"])
    population_size = int(payload.get("evolution", {}).get("population_size", max(1, len(allowed_families))))
    generations = int(payload.get("evolution", {}).get("num_generations", 1))
    seed = int(payload.get("seed", 42))
    epochs = int(payload.get("training", {}).get("epochs", 1))
    return _portable_export(
        system="prism",
        config_path=config_path,
        pack_path=pack_path,
        run_dir=run_dir,
        output_dir=output_dir,
        log_dir=log_dir,
        seed=seed,
        epochs=epochs,
        candidate_specs=_prism_candidates(allowed_families, limit=max(1, population_size * generations)),
        generations=generations,
        population_size=population_size,
    )


def ensure_topograph_portable_smoke_export(
    *,
    config_path: Path,
    pack_path: Path,
    run_dir: Path,
    output_dir: Path | None = None,
    log_dir: Path | None = None,
) -> PortableArtifacts:
    payload = yaml.safe_load(config_path.read_text(encoding="utf-8"))
    population_size = int(payload.get("evolution", {}).get("population_size", 2))
    generations = int(payload.get("evolution", {}).get("num_generations", 1))
    seed = int(payload.get("seed", 42))
    epochs = int(payload.get("training", {}).get("epochs", 1))
    return _portable_export(
        system="topograph",
        config_path=config_path,
        pack_path=pack_path,
        run_dir=run_dir,
        output_dir=output_dir,
        log_dir=log_dir,
        seed=seed,
        epochs=epochs,
        candidate_specs=_topograph_candidates(limit=max(1, population_size * generations)),
        generations=generations,
        population_size=population_size,
    )


def _portable_export(
    *,
    system: str,
    config_path: Path,
    pack_path: Path,
    run_dir: Path,
    output_dir: Path | None,
    log_dir: Path | None,
    seed: int,
    epochs: int,
    candidate_specs: list[dict[str, Any]],
    generations: int,
    population_size: int,
) -> PortableArtifacts:
    run_dir = run_dir.resolve()
    resolved_output_dir = run_dir if output_dir is None else output_dir.resolve()
    run_dir.mkdir(parents=True, exist_ok=True)
    resolved_output_dir.mkdir(parents=True, exist_ok=True)
    shutil.copy2(config_path, run_dir / "config.yaml") if config_path.exists() and (run_dir / "config.yaml") != config_path else None
    if resolved_output_dir != run_dir and config_path.exists():
        shutil.copy2(config_path, resolved_output_dir / "config.yaml")

    pack = load_parity_pack(pack_path)
    started = time.perf_counter()
    results: list[dict[str, Any]] = []
    benchmark_entries: list[dict[str, Any]] = []
    best_results: list[dict[str, Any]] = []
    failures: list[str] = []

    for entry in pack.benchmarks:
        benchmark_started = time.perf_counter()
        try:
            spec = _load_shared_benchmark(entry)
            outcome = _evaluate_benchmark(spec, entry=entry, seed=seed, epochs=epochs, candidate_specs=candidate_specs)
            status = outcome["status"]
            if status == "failed":
                failures.append(f"{entry.benchmark_id}: {outcome['failure_reason']}")
            result_row = {
                "system": system,
                "run_id": run_dir.name,
                "benchmark_id": entry.benchmark_id,
                "metric_name": entry.metric_name,
                "metric_direction": entry.metric_direction,
                "metric_value": outcome.get("metric_value"),
                "quality": outcome.get("quality"),
                "parameter_count": outcome.get("parameter_count"),
                "train_seconds": round(time.perf_counter() - benchmark_started, 4),
                "peak_memory_mb": None,
                "architecture_summary": outcome.get("architecture_summary"),
                "genome_id": outcome.get("genome_id"),
                "status": status,
                "failure_reason": outcome.get("failure_reason"),
            }
        except Exception as exc:
            failures.append(f"{entry.benchmark_id}: {exc}")
            result_row = {
                "system": system,
                "run_id": run_dir.name,
                "benchmark_id": entry.benchmark_id,
                "metric_name": entry.metric_name,
                "metric_direction": entry.metric_direction,
                "metric_value": None,
                "quality": None,
                "parameter_count": None,
                "train_seconds": round(time.perf_counter() - benchmark_started, 4),
                "peak_memory_mb": None,
                "architecture_summary": None,
                "genome_id": None,
                "status": "failed",
                "failure_reason": str(exc),
            }
        results.append(result_row)
        benchmark_entries.append(
            {
                "benchmark_id": entry.benchmark_id,
                "task_kind": entry.task_kind,
                "metric_name": entry.metric_name,
                "metric_direction": entry.metric_direction,
                "status": result_row["status"],
            }
        )
        if result_row["status"] == "ok":
            best_results.append(
                {
                    "benchmark_name": entry.benchmark_id,
                    "metric_name": entry.metric_name,
                    "metric_value": result_row["metric_value"],
                    "quality": result_row["quality"],
                    "architecture_summary": result_row["architecture_summary"],
                    "genome_id": result_row["genome_id"],
                    "status": "ok",
                }
            )

    pack_name = pack.name
    evaluation_count = int(pack.budget_policy.evaluation_count)
    wall_clock = round(time.perf_counter() - started, 4)
    manifest = RunManifest(
        schema_version="1.0",
        system=system,
        run_id=run_dir.name,
        run_name=run_dir.name,
        created_at=datetime.now(timezone.utc),
        pack_name=pack_name,
        seed=seed,
        benchmarks=[BenchmarkEntry(**entry) for entry in benchmark_entries],
        budget=BudgetEnvelope(
            evaluation_count=evaluation_count,
            epochs_per_candidate=int(pack.budget_policy.epochs_per_candidate),
            effective_training_epochs=epochs,
            wall_clock_seconds=wall_clock,
            generations=generations,
            population_size=population_size,
            budget_policy_name="prototype_equal_budget",
        ),
        device=DeviceInfo(
            device_name=platform.machine(),
            precision_mode="fp32",
            framework="portable-sklearn",
            framework_version=SKLEARN_VERSION,
        ),
        artifacts=ArtifactPaths(
            config_snapshot="config.yaml",
            report_markdown="report.md",
            dataset_manifest_json="dataset_manifest.json",
        ),
        search_telemetry=SearchTelemetry(
            qd_enabled=False,
            effective_training_epochs=epochs,
        ),
        fairness=fairness_manifest(
            pack_name=pack_name,
            seed=seed,
            evaluation_count=evaluation_count,
            budget_policy_name="prototype_equal_budget",
            benchmark_entries=benchmark_entries,
            data_signature=benchmark_signature(pack_name, benchmark_entries),
            code_version=_code_version(),
        ),
    )
    dataset_manifest = [
        {
            "benchmark_id": entry.benchmark_id,
            "task_kind": entry.task_kind,
            "metric_name": entry.metric_name,
            "metric_direction": entry.metric_direction,
        }
        for entry in pack.benchmarks
    ]
    summary = {
        "system": system,
        "run_id": run_dir.name,
        "run_name": run_dir.name,
        "runtime_backend": "portable-sklearn",
        "runtime_version": SKLEARN_VERSION,
        "precision_mode": "fp32",
        "total_evaluations": evaluation_count,
        "benchmarks_evaluated": sum(1 for row in results if row["status"] == "ok"),
        "failure_count": sum(1 for row in results if row["status"] == "failed"),
        "wall_clock_seconds": wall_clock,
        "best_results": best_results,
        "failure_patterns": failures,
        "best_family": _best_family(results) if system == "prism" else None,
    }
    report_lines = [
        f"# {system.title()} Portable Smoke Report",
        "",
        f"- Run ID: `{run_dir.name}`",
        f"- Pack: `{pack_name}`",
        "- Runtime: `portable-sklearn`",
        f"- Runtime Version: `{SKLEARN_VERSION}`",
        f"- Benchmarks OK: `{summary['benchmarks_evaluated']}`",
        f"- Failures: `{summary['failure_count']}`",
        "",
        "## Results",
        "",
        "| Benchmark | Status | Metric | Value | Architecture |",
        "|---|---|---|---:|---|",
    ]
    for row in results:
        value = row["metric_value"]
        report_lines.append(
            f"| {row['benchmark_id']} | {row['status']} | {row['metric_name']} | {'---' if value is None else f'{float(value):.6f}'} | {row.get('architecture_summary') or '—'} |"
        )

    result_records = [ResultRecord(**row) for row in results]

    for target in {run_dir, resolved_output_dir}:
        (target / "manifest.json").write_text(manifest.model_dump_json(indent=2), encoding="utf-8")
        write_json(target / "results.json", [record.model_dump(mode="json") for record in result_records])
        write_json(target / "summary.json", summary)
        (target / "report.md").write_text("\n".join(report_lines) + "\n", encoding="utf-8")
        write_json(target / "dataset_manifest.json", dataset_manifest)
        if not (target / "config.yaml").exists() and config_path.exists():
            shutil.copy2(config_path, target / "config.yaml")

    if log_dir is not None:
        log_dir.mkdir(parents=True, exist_ok=True)
        write_json(log_dir / f"{system}_portable_smoke.log", {"pack": pack_name, "results": results})

    return PortableArtifacts(
        run_dir=run_dir,
        manifest_path=resolved_output_dir / "manifest.json",
        results_path=resolved_output_dir / "results.json",
    )


def _evaluate_benchmark(spec: Any, *, entry: Any, seed: int, epochs: int, candidate_specs: list[dict[str, Any]]) -> dict[str, Any]:
    if entry.task_kind == "language_modeling":
        return {
            "status": "failed",
            "failure_reason": "portable smoke fallback does not support language_modeling",
        }
    X_train, y_train, X_val, y_val = _load_benchmark_arrays(spec, benchmark_id=entry.benchmark_id, seed=seed)
    scaler = StandardScaler()
    X_train = scaler.fit_transform(np.asarray(X_train, dtype=np.float32))
    X_val = scaler.transform(np.asarray(X_val, dtype=np.float32))
    y_train = np.asarray(y_train)
    y_val = np.asarray(y_val)

    best: dict[str, Any] | None = None
    for index, candidate in enumerate(candidate_specs, start=1):
        row = _evaluate_candidate(
            X_train=X_train,
            y_train=y_train,
            X_val=X_val,
            y_val=y_val,
            task_kind=entry.task_kind,
            metric_direction=entry.metric_direction,
            epochs=epochs,
            seed=seed + index,
            candidate=candidate,
        )
        if row["status"] != "ok":
            continue
        if best is None or _is_better(row["metric_value"], best["metric_value"], entry.metric_direction):
            best = row
    if best is not None:
        best["status"] = "ok"
        best["failure_reason"] = None
        return best
    return {"status": "failed", "failure_reason": "no_valid_candidate"}


def _evaluate_candidate(
    *,
    X_train: np.ndarray,
    y_train: np.ndarray,
    X_val: np.ndarray,
    y_val: np.ndarray,
    task_kind: TaskKind,
    metric_direction: str,
    epochs: int,
    seed: int,
    candidate: dict[str, Any],
) -> dict[str, Any]:
    hidden = tuple(int(x) for x in candidate["hidden_layers"])
    try:
        if task_kind == "classification":
            model = MLPClassifier(hidden_layer_sizes=hidden, max_iter=max(1, epochs), random_state=seed)
            model.fit(X_train, y_train)
            preds = model.predict(X_val)
            metric_value = float(accuracy_score(y_val, preds))
        else:
            model = MLPRegressor(hidden_layer_sizes=hidden, max_iter=max(1, epochs), random_state=seed)
            model.fit(X_train, y_train)
            preds = model.predict(X_val)
            metric_value = float(mean_squared_error(y_val, preds))
        return {
            "metric_value": metric_value,
            "quality": metric_value if metric_direction == "max" else -metric_value,
            "parameter_count": _parameter_count(model),
            "architecture_summary": candidate["summary"],
            "genome_id": candidate["id"],
            "status": "ok",
        }
    except Exception as exc:
        return {"status": "failed", "failure_reason": str(exc)}


def _prism_candidates(allowed_families: list[str], *, limit: int) -> list[dict[str, Any]]:
    family_shapes = {
        "mlp": (32,),
        "sparse_mlp": (16,),
        "attention": (48, 24),
        "sparse_attention": (24, 12),
    }
    candidates: list[dict[str, Any]] = []
    for family in allowed_families:
        hidden = family_shapes.get(family, (32,))
        candidates.append({
            "id": f"{family}-{len(candidates)+1}",
            "hidden_layers": hidden,
            "summary": f"{family}[{'x'.join(map(str, hidden))}]",
        })
        if len(candidates) >= limit:
            return candidates
    while len(candidates) < limit:
        candidates.append({
            "id": f"mlp-{len(candidates)+1}",
            "hidden_layers": (32,),
            "summary": "mlp[32]",
        })
    return candidates


def _topograph_candidates(*, limit: int) -> list[dict[str, Any]]:
    templates = [(16,), (32,), (16, 16), (32, 16), (64,)]
    candidates: list[dict[str, Any]] = []
    while len(candidates) < limit:
        hidden = templates[len(candidates) % len(templates)]
        candidates.append({
            "id": f"topology-{len(candidates)+1}",
            "hidden_layers": hidden,
            "summary": f"layers={len(hidden)} widths={'x'.join(map(str, hidden))}",
        })
    return candidates


def _load_shared_benchmark(entry: Any) -> Any:
    from evonn_contenders.benchmarks.datasets import get_benchmark

    native_id = resolve_supported_benchmark_id(entry, "contenders")
    return get_benchmark(native_id)


def _load_benchmark_arrays(spec: Any, *, benchmark_id: str, seed: int):
    original_error: Exception | None = None
    try:
        return spec.load_data(seed=seed)
    except Exception as exc:
        original_error = exc

    from sklearn.datasets import load_diabetes, make_friedman1
    from sklearn.model_selection import train_test_split

    if benchmark_id == "diabetes_regression":
        dataset = load_diabetes()
        X_train, X_val, y_train, y_val = train_test_split(
            dataset.data,
            dataset.target,
            test_size=0.2,
            random_state=seed,
        )
        return X_train, y_train, X_val, y_val
    if benchmark_id == "friedman1_regression":
        X, y = make_friedman1(n_samples=1000, n_features=10, noise=1.0, random_state=seed)
        X_train, X_val, y_train, y_val = train_test_split(
            X,
            y,
            test_size=0.2,
            random_state=seed,
        )
        return X_train, y_train, X_val, y_val
    if original_error is not None:
        raise original_error
    raise RuntimeError(f"unable to load benchmark arrays for {benchmark_id}")


def _parameter_count(model: Any) -> int | None:
    coefs = getattr(model, "coefs_", None)
    intercepts = getattr(model, "intercepts_", None)
    if coefs is None or intercepts is None:
        return None
    return int(sum(arr.size for arr in coefs) + sum(arr.size for arr in intercepts))


def _is_better(current: float, previous: float, metric_direction: str) -> bool:
    return current > previous if metric_direction == "max" else current < previous


def _best_family(results: list[dict[str, Any]]) -> str | None:
    ok_rows = [row for row in results if row["status"] == "ok" and row.get("architecture_summary")]
    if not ok_rows:
        return None
    return str(ok_rows[0]["architecture_summary"]).split("[", 1)[0]


def _code_version() -> str | None:
    try:
        return subprocess.check_output(["git", "rev-parse", "HEAD"], cwd=Path(__file__).resolve().parents[4], text=True).strip()
    except Exception:
        return None
