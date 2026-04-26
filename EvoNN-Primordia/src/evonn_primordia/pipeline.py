"""MLX-backed primitive-first search runtime for Primordia."""
from __future__ import annotations

from datetime import datetime, timezone
import json
from pathlib import Path
import shutil
from time import perf_counter
from typing import Any

import numpy as np

from evonn_primordia.config import RunConfig
from evonn_primordia.export.report import build_primitive_bank_summary, write_report
from evonn_primordia.export.seeding import build_seed_candidates
from evonn_primordia.runtime.backends import RuntimeBindings, resolve_runtime_bindings

BUDGET_POLICY_NAME = "prototype_equal_budget"
PRECISION_MODE = "fp32"


def run_search(
    config: RunConfig,
    *,
    run_dir: str | Path,
    config_path: str | Path | None = None,
) -> Path:
    """Run Primordia search with MLX-backed family evaluation."""

    runtime = _load_runtime_bindings(config)
    runtime_backend = getattr(runtime, "runtime_backend", "unknown")
    runtime_version = getattr(runtime, "runtime_version", None)
    precision_mode = getattr(runtime, "precision_mode", PRECISION_MODE)
    run_dir = Path(run_dir)
    run_dir.mkdir(parents=True, exist_ok=True)
    if config_path is not None:
        shutil.copy2(config_path, run_dir / "config.yaml")

    run_id = run_dir.name
    run_name = config.run_name or run_id
    benchmarks = list(config.benchmark_pool.benchmarks)
    if not benchmarks:
        raise ValueError("Primordia requires at least one benchmark")

    target_evals = _target_evaluation_count(config)
    benchmark_total = len(benchmarks)
    executed_records: list[dict[str, Any]] = []
    best_results: list[dict[str, Any]] = []
    primitive_usage: dict[str, int] = {}
    group_counts = {"tabular": 0, "synthetic": 0, "image": 0, "language_modeling": 0}
    started_at = datetime.now(timezone.utc).isoformat()
    started_clock = perf_counter()

    _emit_progress(
        f"start run_id={run_id} benchmarks={benchmark_total} target_evals={target_evals} mode={config.search.mode} runtime={runtime_backend}"
    )
    for benchmark_index, benchmark_name in enumerate(benchmarks, start=1):
        spec = runtime.get_benchmark(benchmark_name)
        group = runtime.benchmark_group(spec)
        group_counts[group] += 1
        modality = _modality_for_group(group)
        allowed_families = _allowed_families(runtime, config, group, modality)
        slots = _slots_for_benchmark(
            config,
            benchmark_index=benchmark_index - 1,
            benchmark_total=benchmark_total,
            primitive_count=len(allowed_families),
        )
        if slots <= 0:
            raise ValueError(f"No evaluation slots assigned to benchmark '{benchmark_name}'")

        _emit_progress(
            f"[{benchmark_index}/{benchmark_total}] benchmark={benchmark_name} group={group} families={len(allowed_families)} slots={slots}"
        )
        try:
            x_train, y_train, x_val, y_val = spec.load_data(seed=config.seed)
            input_shape = _input_shape_for_spec(spec, group)
            x_train_np, y_train_np, x_val_np, y_val_np = _prepare_arrays(
                spec=spec,
                group=group,
                input_shape=input_shape,
                x_train=x_train,
                y_train=y_train,
                x_val=x_val,
                y_val=y_val,
            )
        except Exception as exc:
            failed = _failed_record(
                spec=spec,
                benchmark_name=benchmark_name,
                benchmark_group=group,
                reason=str(exc),
                runtime_backend=runtime_backend,
                runtime_version=runtime_version,
                precision_mode=precision_mode,
            )
            executed_records.append(failed)
            best_results.append(dict(failed))
            _emit_progress(f"[{benchmark_index}/{benchmark_total}] load-failed benchmark={benchmark_name} reason={exc}")
            continue

        benchmark_records: list[dict[str, Any]] = []
        for slot_index in range(slots):
            family = allowed_families[slot_index % len(allowed_families)]
            repeat_index = slot_index // len(allowed_families)
            primitive_label = family if repeat_index == 0 else f"{family}@r{repeat_index + 1}"
            genome = runtime.create_seed_genome(
                family,
                config.search.seed_hidden_width,
                config.search.seed_hidden_layers,
            )
            for mutation_round in range(repeat_index):
                mutated = runtime.mutate_genome(genome, mutation_round + 1, allowed_families, config)
                genome = mutated[0] if isinstance(mutated, tuple) else mutated

            try:
                compiled = runtime.compile_genome(
                    genome,
                    input_shape,
                    _resolved_output_dim_for_spec(
                        spec,
                        x_train=x_train_np,
                        y_train=y_train_np,
                        x_val=x_val_np,
                        y_val=y_val_np,
                    ),
                    modality,
                    spec.task,
                )
                result = runtime.train_and_evaluate(
                    compiled.model,
                    x_train_np,
                    y_train_np,
                    x_val_np,
                    y_val_np,
                    task=spec.task,
                    epochs=config.training.epochs_per_candidate,
                    lr=getattr(genome, "learning_rate", config.training.learning_rate),
                    batch_size=config.training.batch_size,
                    parameter_count=compiled.parameter_count,
                )
                record = {
                    "benchmark_name": benchmark_name,
                    "benchmark_group": group,
                    "primitive_name": primitive_label,
                    "primitive_family": family,
                    "metric_name": result.metric_name,
                    "metric_direction": spec.metric_direction,
                    "metric_value": result.metric_value,
                    "quality": result.quality,
                    "parameter_count": result.parameter_count,
                    "train_seconds": result.train_seconds,
                    "architecture_summary": _architecture_summary(genome),
                    "genome_id": getattr(genome, "genome_id", primitive_label),
                    "status": "ok" if result.failure_reason is None else "failed",
                    "failure_reason": result.failure_reason,
                    "seed": config.seed + benchmark_index * 1009 + slot_index,
                    "slot_index": slot_index,
                    "runtime": runtime_backend,
                    "runtime_version": runtime_version,
                    "precision_mode": precision_mode,
                }
            except Exception as exc:
                record = {
                    "benchmark_name": benchmark_name,
                    "benchmark_group": group,
                    "primitive_name": primitive_label,
                    "primitive_family": family,
                    "metric_name": spec.metric_name,
                    "metric_direction": spec.metric_direction,
                    "metric_value": None,
                    "quality": float("-inf") if spec.metric_direction == "max" else float("inf"),
                    "parameter_count": getattr(genome, "parameter_estimate", 0),
                    "train_seconds": 0.0,
                    "architecture_summary": _architecture_summary(genome),
                    "genome_id": getattr(genome, "genome_id", primitive_label),
                    "status": "failed",
                    "failure_reason": str(exc),
                    "seed": config.seed + benchmark_index * 1009 + slot_index,
                    "slot_index": slot_index,
                    "runtime": runtime_backend,
                    "runtime_version": runtime_version,
                    "precision_mode": precision_mode,
                }
            benchmark_records.append(record)
            executed_records.append(record)
            primitive_usage[family] = primitive_usage.get(family, 0) + 1
        best = _choose_best(spec.metric_direction, benchmark_records)
        best_results.append(dict(best))
        _emit_progress(
            f"[{benchmark_index}/{benchmark_total}] done benchmark={benchmark_name} best={best['primitive_name']} status={best['status']} metric={best['metric_value']}"
        )

    wall_clock_seconds = perf_counter() - started_clock
    failure_count = sum(1 for record in executed_records if record.get("status") != "ok")
    summary = {
        "system": "primordia",
        "runtime": runtime_backend,
        "runtime_version": runtime_version,
        "precision_mode": precision_mode,
        "run_id": run_id,
        "run_name": run_name,
        "status": "complete",
        "created_at": started_at,
        "seed": config.seed,
        "benchmark_count": benchmark_total,
        "evaluation_count": len(executed_records),
        "target_evaluation_count": target_evals,
        "epochs_per_candidate": config.training.epochs_per_candidate,
        "budget_policy_name": BUDGET_POLICY_NAME,
        "primitive_usage": dict(sorted(primitive_usage.items(), key=lambda item: (-item[1], item[0]))),
        "group_counts": group_counts,
        "failure_count": failure_count,
        "wall_clock_seconds": wall_clock_seconds,
        "best_results": best_results,
    }
    primitive_bank_summary = build_primitive_bank_summary(
        summary=summary,
        best_results=best_results,
        trial_records=executed_records,
    )
    seed_candidates = build_seed_candidates(
        summary=summary,
        best_results=best_results,
        trial_records=executed_records,
        primitive_bank=primitive_bank_summary,
    )
    (run_dir / "trial_records.json").write_text(json.dumps(executed_records, indent=2), encoding="utf-8")
    (run_dir / "best_results.json").write_text(json.dumps(best_results, indent=2), encoding="utf-8")
    (run_dir / "summary.json").write_text(json.dumps(summary, indent=2), encoding="utf-8")
    (run_dir / "primitive_bank_summary.json").write_text(json.dumps(primitive_bank_summary, indent=2), encoding="utf-8")
    (run_dir / "seed_candidates.json").write_text(json.dumps(seed_candidates, indent=2), encoding="utf-8")
    write_report(run_dir)
    _emit_progress(
        f"finished run_id={run_id} evaluation_count={len(executed_records)} wall_clock_seconds={wall_clock_seconds:.3f}"
    )
    return run_dir


def _load_runtime_bindings(config: RunConfig) -> RuntimeBindings:
    return resolve_runtime_bindings(config)


def _allowed_families(runtime: RuntimeBindings, config: RunConfig, group: str, modality: str) -> list[str]:
    configured = _primitive_names_for_group(config, group)
    compatible = set(runtime.compatible_families(modality))
    allowed = [family for family in configured if family in compatible]
    if not allowed:
        raise ValueError(f"No compatible Primordia families configured for group '{group}' and modality '{modality}'")
    return allowed


def _modality_for_group(group: str) -> str:
    if group == "image":
        return "image"
    if group == "language_modeling":
        return "text"
    return "tabular"


def _input_shape_for_spec(spec: Any, group: str) -> list[int]:
    if group == "image":
        return list(spec.resolved_image_shape)
    if group == "language_modeling":
        return [int(spec.model_input_dim)]
    return [int(spec.model_input_dim)]


def _output_dim_for_spec(spec: Any) -> int:
    return int(spec.model_output_dim)


def _resolved_output_dim_for_spec(
    spec: Any,
    *,
    x_train: np.ndarray,
    y_train: np.ndarray,
    x_val: np.ndarray,
    y_val: np.ndarray,
) -> int:
    output_dim = _output_dim_for_spec(spec)
    if spec.task != "language_modeling":
        return output_dim

    max_token_id = max(
        int(np.max(x_train)),
        int(np.max(y_train)),
        int(np.max(x_val)),
        int(np.max(y_val)),
    )
    return max(output_dim, max_token_id + 1)


def _prepare_arrays(
    *,
    spec: Any,
    group: str,
    input_shape: list[int],
    x_train,
    y_train,
    x_val,
    y_val,
) -> tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    x_train_np = np.asarray(x_train)
    y_train_np = np.asarray(y_train)
    x_val_np = np.asarray(x_val)
    y_val_np = np.asarray(y_val)
    if group == "image":
        x_train_np = x_train_np.reshape((x_train_np.shape[0], *input_shape)).astype(np.float32)
        x_val_np = x_val_np.reshape((x_val_np.shape[0], *input_shape)).astype(np.float32)
    elif spec.task == "language_modeling":
        x_train_np = x_train_np.astype(np.int32)
        y_train_np = y_train_np.astype(np.int32)
        x_val_np = x_val_np.astype(np.int32)
        y_val_np = y_val_np.astype(np.int32)
    else:
        x_train_np = x_train_np.astype(np.float32)
        y_train_np = y_train_np.astype(np.float32 if spec.task == "regression" else np.int32)
        x_val_np = x_val_np.astype(np.float32)
        y_val_np = y_val_np.astype(np.float32 if spec.task == "regression" else np.int32)
    return x_train_np, y_train_np, x_val_np, y_val_np


def _target_evaluation_count(config: RunConfig) -> int:
    family_count = sum(
        len(names)
        for names in (
            config.primitive_pool.tabular,
            config.primitive_pool.synthetic,
            config.primitive_pool.image,
            config.primitive_pool.language_modeling,
        )
    )
    default_budget = max(len(config.benchmark_pool.benchmarks), family_count)
    return config.search.target_evaluation_count or default_budget


def _slots_for_benchmark(
    config: RunConfig,
    *,
    benchmark_index: int,
    benchmark_total: int,
    primitive_count: int,
) -> int:
    if primitive_count <= 0:
        raise ValueError("No primitive candidates configured")
    if config.search.mode == "fixed_pool":
        return primitive_count
    target = _target_evaluation_count(config)
    if target < benchmark_total:
        raise ValueError(
            "Primordia budget_matched mode requires target_evaluation_count >= benchmark_count "
            f"({target} < {benchmark_total})"
        )
    base_slots = target // benchmark_total
    remainder = target % benchmark_total
    return base_slots + (1 if benchmark_index < remainder else 0)


def _primitive_names_for_group(config: RunConfig, group: str) -> list[str]:
    if group == "synthetic":
        return config.primitive_pool.synthetic
    if group == "image":
        return config.primitive_pool.image
    if group == "language_modeling":
        return config.primitive_pool.language_modeling
    return config.primitive_pool.tabular


def _failed_record(
    *,
    spec: Any,
    benchmark_name: str,
    benchmark_group: str,
    reason: str,
    runtime_backend: str,
    runtime_version: str | None,
    precision_mode: str,
) -> dict[str, Any]:
    return {
        "benchmark_name": benchmark_name,
        "benchmark_group": benchmark_group,
        "primitive_name": "load_failed",
        "primitive_family": "load_failed",
        "metric_name": spec.metric_name,
        "metric_direction": spec.metric_direction,
        "metric_value": None,
        "quality": None,
        "parameter_count": 0,
        "train_seconds": 0.0,
        "architecture_summary": "load_failed",
        "genome_id": f"{benchmark_name}:load_failed",
        "status": "failed",
        "failure_reason": reason,
        "seed": 0,
        "slot_index": 0,
        "runtime": runtime_backend,
        "runtime_version": runtime_version,
        "precision_mode": precision_mode,
    }


def _choose_best(metric_direction: str, records: list[dict[str, Any]]) -> dict[str, Any]:
    ok_records = [record for record in records if record["status"] == "ok" and record["metric_value"] is not None]
    if not ok_records:
        return records[0]
    reverse = metric_direction == "max"
    return sorted(ok_records, key=lambda record: record["metric_value"], reverse=reverse)[0]


def _architecture_summary(genome: Any) -> str:
    hidden_layers = getattr(genome, "hidden_layers", [])
    widths = "x".join(str(width) for width in hidden_layers) if hidden_layers else "none"
    return f"{getattr(genome, 'family', 'unknown')}[{widths}]"




def _emit_progress(message: str) -> None:
    print(f"[primordia] {message}", flush=True)
