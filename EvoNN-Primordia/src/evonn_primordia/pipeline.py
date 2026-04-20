"""Cheap primitive-first search runtime for Primordia."""
from __future__ import annotations

import importlib.util
import json
import shutil
from datetime import datetime, timezone
from pathlib import Path
from time import perf_counter
from typing import Any

from evonn_contenders.benchmarks import get_benchmark
from evonn_contenders.contenders import benchmark_group, choose_best, evaluate_contender, resolve_contenders

from evonn_primordia.config import RunConfig


BUDGET_POLICY_NAME = "prototype_equal_budget"


def run_search(
    config: RunConfig,
    *,
    run_dir: str | Path,
    config_path: str | Path | None = None,
) -> Path:
    """Run a budget-matched primitive search and persist run artifacts."""

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
        f"start run_id={run_id} benchmarks={benchmark_total} target_evals={target_evals} mode={config.search.mode}"
    )
    for benchmark_index, benchmark_name in enumerate(benchmarks, start=1):
        spec = get_benchmark(benchmark_name)
        group = benchmark_group(spec)
        group_counts[group] += 1
        primitive_names = _primitive_names_for_group(config, group)
        primitives = _resolve_primitives(group, primitive_names, allow_optional_missing=config.torch.allow_optional_missing)
        slots = _slots_for_benchmark(config, benchmark_name=benchmark_name, benchmark_total=benchmark_total, primitive_count=len(primitives))
        if slots <= 0:
            raise ValueError(f"No evaluation slots assigned to benchmark '{benchmark_name}'")

        _emit_progress(
            f"[{benchmark_index}/{benchmark_total}] benchmark={benchmark_name} group={group} primitives={len(primitives)} slots={slots}"
        )
        try:
            x_train, y_train, x_val, y_val = spec.load_data(seed=config.seed)
        except Exception as exc:
            failed = {
                "benchmark_name": benchmark_name,
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
                "failure_reason": str(exc),
                "seed": config.seed,
                "slot_index": 0,
            }
            executed_records.append(failed)
            best_results.append(dict(failed))
            _emit_progress(f"[{benchmark_index}/{benchmark_total}] load-failed benchmark={benchmark_name} reason={exc}")
            continue

        benchmark_records: list[dict[str, Any]] = []
        for slot_index in range(slots):
            primitive = primitives[slot_index % len(primitives)]
            repeat_index = slot_index // len(primitives)
            primitive_label = primitive.name if repeat_index == 0 else f"{primitive.name}@r{repeat_index + 1}"
            seed = config.seed + benchmark_index * 1009 + slot_index
            record = evaluate_contender(
                spec,
                primitive,
                seed=seed,
                contender_label=primitive_label,
                x_train=x_train,
                y_train=y_train,
                x_val=x_val,
                y_val=y_val,
                config=config,
            )
            record = {
                "benchmark_name": benchmark_name,
                "primitive_name": primitive_label,
                "primitive_family": primitive.family,
                "metric_name": record["metric_name"],
                "metric_direction": record["metric_direction"],
                "metric_value": record["metric_value"],
                "quality": record["quality"],
                "parameter_count": record["parameter_count"],
                "train_seconds": record["train_seconds"],
                "architecture_summary": record["architecture_summary"],
                "genome_id": record["contender_id"],
                "status": record["status"],
                "failure_reason": record["failure_reason"],
                "seed": seed,
                "slot_index": slot_index,
            }
            benchmark_records.append(record)
            executed_records.append(record)
            primitive_usage[primitive.name] = primitive_usage.get(primitive.name, 0) + 1
        best = choose_best(spec.metric_direction, benchmark_records)
        best_results.append(
            {
                "benchmark_name": benchmark_name,
                "primitive_name": best["primitive_name"],
                "primitive_family": best["primitive_family"],
                "metric_name": best["metric_name"],
                "metric_direction": best["metric_direction"],
                "metric_value": best["metric_value"],
                "quality": best["quality"],
                "parameter_count": best["parameter_count"],
                "train_seconds": best["train_seconds"],
                "architecture_summary": best["architecture_summary"],
                "genome_id": best["genome_id"],
                "status": best["status"],
                "failure_reason": best["failure_reason"],
                "seed": best["seed"],
                "slot_index": best["slot_index"],
            }
        )
        _emit_progress(
            f"[{benchmark_index}/{benchmark_total}] done benchmark={benchmark_name} best={best['primitive_name']} status={best['status']} metric={best['metric_value']}"
        )

    wall_clock_seconds = perf_counter() - started_clock
    summary = {
        "system": "primordia",
        "run_id": run_id,
        "run_name": run_name,
        "status": "complete",
        "created_at": started_at,
        "seed": config.seed,
        "benchmark_count": benchmark_total,
        "evaluation_count": len(executed_records),
        "target_evaluation_count": target_evals,
        "epochs_per_candidate": config.search.epochs_per_candidate,
        "budget_policy_name": BUDGET_POLICY_NAME,
        "primitive_usage": dict(sorted(primitive_usage.items(), key=lambda item: (-item[1], item[0]))),
        "group_counts": group_counts,
        "wall_clock_seconds": wall_clock_seconds,
        "best_results": best_results,
    }

    (run_dir / "trial_records.json").write_text(json.dumps(executed_records, indent=2), encoding="utf-8")
    (run_dir / "best_results.json").write_text(json.dumps(best_results, indent=2), encoding="utf-8")
    (run_dir / "summary.json").write_text(json.dumps(summary, indent=2), encoding="utf-8")
    _write_report(run_dir=run_dir, summary=summary)
    _emit_progress(
        f"finished run_id={run_id} evaluation_count={len(executed_records)} wall_clock_seconds={wall_clock_seconds:.3f}"
    )
    return run_dir


def _target_evaluation_count(config: RunConfig) -> int:
    primitive_names = sum(
        len(names)
        for names in (
            config.primitive_pool.tabular,
            config.primitive_pool.synthetic,
            config.primitive_pool.image,
            config.primitive_pool.language_modeling,
        )
    )
    default_budget = max(len(config.benchmark_pool.benchmarks), primitive_names)
    return config.search.target_evaluation_count or default_budget


def _slots_for_benchmark(
    config: RunConfig,
    *,
    benchmark_name: str,
    benchmark_total: int,
    primitive_count: int,
) -> int:
    if primitive_count <= 0:
        raise ValueError(f"No primitive candidates configured for benchmark '{benchmark_name}'")
    if config.search.mode == "fixed_pool":
        return primitive_count
    target = _target_evaluation_count(config)
    if target < benchmark_total:
        raise ValueError(
            "Primordia budget_matched mode requires target_evaluation_count >= benchmark_count "
            f"({target} < {benchmark_total})"
        )
    benchmark_index = config.benchmark_pool.benchmarks.index(benchmark_name)
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


def _resolve_primitives(group: str, names: list[str], *, allow_optional_missing: bool) -> list[Any]:
    resolved = resolve_contenders(group, names)
    available = []
    for contender in resolved:
        dependency = getattr(contender, "optional_dependency", None)
        if dependency and importlib.util.find_spec(dependency) is None and allow_optional_missing:
            continue
        available.append(contender)
    if not available:
        raise ValueError(f"No runnable primitives configured for group '{group}'")
    return available


def _write_report(*, run_dir: Path, summary: dict[str, Any]) -> None:
    lines = [
        "# Primordia Run Report",
        "",
        f"- Run ID: `{summary['run_id']}`",
        f"- Evaluations: `{summary['evaluation_count']}`",
        f"- Benchmarks: `{summary['benchmark_count']}`",
        f"- Budget Policy: `{summary['budget_policy_name']}`",
        "",
        "## Best Primitive Per Benchmark",
        "",
        "| Benchmark | Primitive | Metric | Value | Status |",
        "|---|---|---|---:|---|",
    ]
    for record in summary["best_results"]:
        value = "---" if record["metric_value"] is None else f"{float(record['metric_value']):.6f}"
        lines.append(
            f"| {record['benchmark_name']} | {record['primitive_name']} | {record['metric_name']} | {value} | {record['status']} |"
        )
    (run_dir / "report.md").write_text("\n".join(lines) + "\n", encoding="utf-8")


def _emit_progress(message: str) -> None:
    print(f"[primordia] {message}", flush=True)
