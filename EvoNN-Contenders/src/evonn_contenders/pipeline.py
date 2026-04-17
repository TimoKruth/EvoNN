"""Run contender pools against benchmark packs."""

from __future__ import annotations

from datetime import datetime, timezone
from pathlib import Path
import shutil

from evonn_contenders.benchmarks import get_benchmark
from evonn_contenders.config import RunConfig
from evonn_contenders.contender_pool import (
    benchmark_group,
    choose_best,
    contender_names_for_config,
    evaluate_contender,
    resolve_contenders,
)
from evonn_contenders.export.report import write_report
from evonn_contenders.storage import RunStore


def run_contenders(
    config: RunConfig,
    *,
    run_dir: str | Path,
    config_path: str | Path | None = None,
) -> Path:
    """Run contender pools and persist best result per benchmark."""
    run_dir = Path(run_dir)
    run_dir.mkdir(parents=True, exist_ok=True)
    if config_path is not None:
        shutil.copy2(config_path, run_dir / "config.yaml")

    run_id = run_dir.name
    created_at = datetime.now(timezone.utc).isoformat()
    store = RunStore(run_dir / "metrics.duckdb")
    store.record_run(
        run_id=run_id,
        run_name=config.run_name or run_id,
        created_at=created_at,
        seed=config.seed,
        config=config.model_dump(mode="json"),
    )

    evaluation_count = 0
    group_counts = {"tabular": 0, "synthetic": 0, "image": 0, "language_modeling": 0}

    for benchmark_name in config.benchmark_pool.benchmarks:
        spec = get_benchmark(benchmark_name)
        group = benchmark_group(spec)
        contender_names = contender_names_for_config(config, group)
        if config.selection.max_contenders_per_benchmark is not None:
            contender_names = contender_names[: config.selection.max_contenders_per_benchmark]
        contenders = resolve_contenders(group, contender_names)
        group_counts[group] += 1
        evaluation_count += len(contenders)

        try:
            x_train, y_train, x_val, y_val = spec.load_data(seed=config.seed)
        except Exception as exc:
            failed = {
                "contender_name": "load_failed",
                "family": "load_failed",
                "metric_name": spec.metric_name,
                "metric_direction": spec.metric_direction,
                "metric_value": None,
                "quality": float("-inf") if spec.metric_direction == "max" else float("inf"),
                "parameter_count": 0,
                "train_seconds": 0.0,
                "architecture_summary": "load_failed",
                "contender_id": f"{benchmark_name}:load_failed",
                "status": "failed",
                "failure_reason": str(exc),
            }
            store.record_contender(run_id=run_id, benchmark_name=benchmark_name, record=failed)
            store.record_result(run_id=run_id, benchmark_name=benchmark_name, record=failed)
            continue

        records: list[dict[str, object]] = []
        for contender in contenders:
            record = evaluate_contender(
                spec,
                contender,
                seed=config.seed,
                x_train=x_train,
                y_train=y_train,
                x_val=x_val,
                y_val=y_val,
            )
            store.record_contender(run_id=run_id, benchmark_name=benchmark_name, record=record)
            records.append(record)

        best = choose_best(spec.metric_direction, records)
        store.record_result(run_id=run_id, benchmark_name=benchmark_name, record=best)

    store.save_budget_metadata(
        run_id=run_id,
        payload={
            "evaluation_count": evaluation_count,
            "created_at": created_at,
            "benchmark_count": len(config.benchmark_pool.benchmarks),
            "tabular_benchmark_count": group_counts["tabular"],
            "synthetic_benchmark_count": group_counts["synthetic"],
            "image_benchmark_count": group_counts["image"],
            "language_modeling_benchmark_count": group_counts["language_modeling"],
            "budget_policy_name": "fixed_contender_pool",
        },
    )
    store.close()
    write_report(run_dir)
    return run_dir
