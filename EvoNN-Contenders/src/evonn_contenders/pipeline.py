"""Run contender pools against benchmark packs."""

from __future__ import annotations

from dataclasses import dataclass
from datetime import datetime, timezone
import importlib.util
from pathlib import Path
import shutil

from evonn_contenders.benchmarks import get_benchmark
from evonn_contenders.config import RunConfig, baseline_signature, resolve_baseline_id
from evonn_contenders.contender_pool import (
    benchmark_group,
    choose_best,
    contender_names_for_config,
    evaluate_contender,
    resolve_contenders,
)
from evonn_contenders.export.report import write_report
from evonn_contenders.storage import RunStore


@dataclass(frozen=True)
class ContenderTrial:
    """One concrete contender fit under a pack budget."""

    contender: object
    contender_name: str
    seed: int


def run_contenders(
    config: RunConfig,
    *,
    run_dir: str | Path,
    config_path: str | Path | None = None,
) -> Path:
    """Run contender pools, reusing baseline-cached benchmark results when available."""

    run_dir = Path(run_dir)
    run_dir.mkdir(parents=True, exist_ok=True)
    if config_path is not None:
        shutil.copy2(config_path, run_dir / "config.yaml")

    run_id = run_dir.name
    created_at = datetime.now(timezone.utc).isoformat()
    run_store = RunStore(run_dir / "metrics.duckdb")
    run_store.clear_run_records(run_id)
    run_store.record_run(
        run_id=run_id,
        run_name=config.run_name or run_id,
        created_at=created_at,
        seed=config.seed,
        config=config.model_dump(mode="json"),
    )

    baseline_id = resolve_baseline_id(config)
    baseline_sig = baseline_signature(config)
    baseline_store = RunStore(_baseline_cache_dir(config, config_path=config_path, run_dir=run_dir) / "metrics.duckdb")
    _ensure_baseline_metadata(
        baseline_store,
        baseline_id=baseline_id,
        baseline_sig=baseline_sig,
        config=config,
        created_at=created_at,
    )

    evaluation_count = 0
    executed_evaluation_count = 0
    cache_hits = 0
    group_counts = {"tabular": 0, "synthetic": 0, "image": 0, "language_modeling": 0}

    for benchmark_name in config.benchmark_pool.benchmarks:
        spec = get_benchmark(benchmark_name)
        group = benchmark_group(spec)
        trials = _resolve_trials(config, benchmark_name=benchmark_name, group=group)
        group_counts[group] += 1
        evaluation_count += len(trials)

        cached_records = baseline_store.load_contenders_for_benchmark(baseline_id, benchmark_name)
        cached_best = baseline_store.load_result_for_benchmark(baseline_id, benchmark_name)
        if cached_records and cached_best is not None and len(cached_records) == len(trials):
            run_store.replace_contenders(run_id=run_id, benchmark_name=benchmark_name, records=cached_records)
            run_store.replace_result(run_id=run_id, benchmark_name=benchmark_name, record=cached_best)
            cache_hits += 1
            continue

        try:
            x_train, y_train, x_val, y_val = spec.load_data(seed=config.seed)
        except Exception as exc:
            failed_records = [
                {
                    "contender_name": trial.contender_name,
                    "family": "load_failed",
                    "metric_name": spec.metric_name,
                    "metric_direction": spec.metric_direction,
                    "metric_value": None,
                    "quality": float("-inf") if spec.metric_direction == "max" else float("inf"),
                    "parameter_count": 0,
                    "train_seconds": 0.0,
                    "architecture_summary": "load_failed",
                    "contender_id": f"{benchmark_name}:{trial.contender_name}:seed{trial.seed}",
                    "status": "failed",
                    "failure_reason": str(exc),
                }
                for trial in trials
            ]
            run_store.replace_contenders(run_id=run_id, benchmark_name=benchmark_name, records=failed_records)
            run_store.replace_result(run_id=run_id, benchmark_name=benchmark_name, record=failed_records[0])
            baseline_store.replace_contenders(run_id=baseline_id, benchmark_name=benchmark_name, records=failed_records)
            baseline_store.replace_result(run_id=baseline_id, benchmark_name=benchmark_name, record=failed_records[0])
            continue

        records: list[dict[str, object]] = []
        for trial in trials:
            record = evaluate_contender(
                spec,
                trial.contender,
                seed=trial.seed,
                x_train=x_train,
                y_train=y_train,
                x_val=x_val,
                y_val=y_val,
                config=config,
                contender_label=trial.contender_name,
            )
            run_store.record_contender(run_id=run_id, benchmark_name=benchmark_name, record=record)
            records.append(record)
        executed_evaluation_count += len(trials)

        best = choose_best(spec.metric_direction, records)
        run_store.replace_result(run_id=run_id, benchmark_name=benchmark_name, record=best)
        baseline_store.replace_contenders(run_id=baseline_id, benchmark_name=benchmark_name, records=list(records))
        baseline_store.replace_result(run_id=baseline_id, benchmark_name=benchmark_name, record=best)

    run_store.save_budget_metadata(
        run_id=run_id,
        payload={
            "evaluation_count": evaluation_count,
            "executed_evaluation_count": executed_evaluation_count,
            "cache_hits": cache_hits,
            "created_at": created_at,
            "benchmark_count": len(config.benchmark_pool.benchmarks),
            "tabular_benchmark_count": group_counts["tabular"],
            "synthetic_benchmark_count": group_counts["synthetic"],
            "image_benchmark_count": group_counts["image"],
            "language_modeling_benchmark_count": group_counts["language_modeling"],
            "budget_policy_name": f"{config.baseline.mode}_contender_pool",
            "baseline_id": baseline_id,
            "baseline_signature": baseline_sig,
            "target_evaluation_count": config.baseline.target_evaluation_count,
        },
    )
    _update_baseline_budget_metadata(baseline_store, baseline_id=baseline_id, created_at=created_at)

    baseline_store.close()
    run_store.close()
    write_report(run_dir)
    return run_dir


def materialize_baseline_run(
    config: RunConfig,
    *,
    run_dir: str | Path,
    config_path: str | Path | None = None,
) -> Path:
    """Assemble one run directory entirely from cached baseline records."""

    run_dir = Path(run_dir)
    run_dir.mkdir(parents=True, exist_ok=True)
    if config_path is not None:
        shutil.copy2(config_path, run_dir / "config.yaml")

    run_id = run_dir.name
    created_at = datetime.now(timezone.utc).isoformat()
    baseline_id = resolve_baseline_id(config)
    baseline_sig = baseline_signature(config)
    baseline_store = RunStore(_baseline_cache_dir(config, config_path=config_path, run_dir=run_dir) / "metrics.duckdb")
    _ensure_baseline_metadata(
        baseline_store,
        baseline_id=baseline_id,
        baseline_sig=baseline_sig,
        config=config,
        created_at=created_at,
    )

    run_store = RunStore(run_dir / "metrics.duckdb")
    run_store.clear_run_records(run_id)
    run_store.record_run(
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
        group_counts[group] += 1
        trials = _resolve_trials(config, benchmark_name=benchmark_name, group=group)
        evaluation_count += len(trials)

        cached_records = baseline_store.load_contenders_for_benchmark(baseline_id, benchmark_name)
        cached_best = baseline_store.load_result_for_benchmark(baseline_id, benchmark_name)
        if not cached_records or cached_best is None or len(cached_records) != len(trials):
            baseline_store.close()
            run_store.close()
            raise ValueError(f"benchmark '{benchmark_name}' missing from baseline cache '{baseline_id}'")
        run_store.replace_contenders(run_id=run_id, benchmark_name=benchmark_name, records=cached_records)
        run_store.replace_result(run_id=run_id, benchmark_name=benchmark_name, record=cached_best)

    run_store.save_budget_metadata(
        run_id=run_id,
        payload={
            "evaluation_count": evaluation_count,
            "executed_evaluation_count": 0,
            "cache_hits": len(config.benchmark_pool.benchmarks),
            "created_at": created_at,
            "benchmark_count": len(config.benchmark_pool.benchmarks),
            "tabular_benchmark_count": group_counts["tabular"],
            "synthetic_benchmark_count": group_counts["synthetic"],
            "image_benchmark_count": group_counts["image"],
            "language_modeling_benchmark_count": group_counts["language_modeling"],
            "budget_policy_name": f"{config.baseline.mode}_contender_pool",
            "baseline_id": baseline_id,
            "baseline_signature": baseline_sig,
            "target_evaluation_count": config.baseline.target_evaluation_count,
        },
    )

    baseline_store.close()
    run_store.close()
    write_report(run_dir)
    return run_dir


def _baseline_cache_dir(
    config: RunConfig,
    *,
    config_path: str | Path | None,
    run_dir: Path,
) -> Path:
    cache_dir = Path(config.baseline.cache_dir)
    if cache_dir.is_absolute():
        root = cache_dir
    elif config_path is not None:
        root = Path(config_path).resolve().parent / cache_dir
    else:
        root = run_dir.parent / cache_dir
    baseline_dir = root / resolve_baseline_id(config)
    baseline_dir.mkdir(parents=True, exist_ok=True)
    return baseline_dir


def _ensure_baseline_metadata(
    store: RunStore,
    *,
    baseline_id: str,
    baseline_sig: str,
    config: RunConfig,
    created_at: str,
) -> None:
    runs = {run["run_id"]: run for run in store.load_runs()}
    existing = runs.get(baseline_id)
    if existing is not None:
        meta = store.load_budget_metadata(baseline_id)
        existing_sig = meta.get("baseline_signature")
        if existing_sig and existing_sig != baseline_sig:
            raise ValueError(
                f"baseline '{baseline_id}' already exists with different contender policy "
                f"({existing_sig} != {baseline_sig})"
            )
        return
    store.record_run(
        run_id=baseline_id,
        run_name=baseline_id,
        created_at=created_at,
        seed=config.seed,
        config=config.model_dump(mode="json"),
    )
    store.save_budget_metadata(
        run_id=baseline_id,
        payload={
            "evaluation_count": 0,
            "executed_evaluation_count": 0,
            "cache_hits": 0,
            "benchmark_count": 0,
            "tabular_benchmark_count": 0,
            "synthetic_benchmark_count": 0,
            "image_benchmark_count": 0,
            "language_modeling_benchmark_count": 0,
            "budget_policy_name": f"{config.baseline.mode}_contender_pool",
            "baseline_id": baseline_id,
            "baseline_signature": baseline_sig,
            "target_evaluation_count": config.baseline.target_evaluation_count,
            "created_at": created_at,
        },
    )


def _update_baseline_budget_metadata(
    store: RunStore,
    *,
    baseline_id: str,
    created_at: str,
) -> None:
    results = store.load_results(baseline_id)
    contenders = store.load_contenders(baseline_id)
    group_counts = {"tabular": 0, "synthetic": 0, "image": 0, "language_modeling": 0}
    for record in results:
        group = benchmark_group(get_benchmark(record["benchmark_name"]))
        group_counts[group] += 1
    meta = store.load_budget_metadata(baseline_id)
    store.save_budget_metadata(
        run_id=baseline_id,
        payload={
            **meta,
            "evaluation_count": len(contenders),
            "executed_evaluation_count": len(contenders),
            "cache_hits": 0,
            "benchmark_count": len(results),
            "tabular_benchmark_count": group_counts["tabular"],
            "synthetic_benchmark_count": group_counts["synthetic"],
            "image_benchmark_count": group_counts["image"],
            "language_modeling_benchmark_count": group_counts["language_modeling"],
            "created_at": created_at,
        },
    )


def _resolve_trials(config: RunConfig, *, benchmark_name: str, group: str) -> list[ContenderTrial]:
    contender_names = contender_names_for_config(config, group)
    if config.selection.max_contenders_per_benchmark is not None:
        contender_names = contender_names[: config.selection.max_contenders_per_benchmark]
    contenders = resolve_contenders(group, contender_names)
    contenders = _filter_unavailable_optional_contenders(contenders, config=config)
    if not contenders:
        raise ValueError(
            f"no runnable contenders configured for group '{group}' "
            f"on benchmark '{benchmark_name}'"
        )

    slots = len(contenders)
    if config.baseline.mode == "budget_matched":
        slots = _budget_matched_slots(config, benchmark_name=benchmark_name)

    trials: list[ContenderTrial] = []
    for slot_index in range(slots):
        contender = contenders[slot_index % len(contenders)]
        repeat_index = slot_index // len(contenders)
        contender_name = contender.name if repeat_index == 0 else f"{contender.name}@r{repeat_index + 1}"
        trial_seed = config.seed + (repeat_index * 1009) + slot_index
        trials.append(
            ContenderTrial(
                contender=contender,
                contender_name=contender_name,
                seed=trial_seed,
            )
        )
    return trials


def _filter_unavailable_optional_contenders(contenders: list[object], *, config: RunConfig) -> list[object]:
    available: list[object] = []
    for contender in contenders:
        dependency = getattr(contender, "optional_dependency", None)
        if dependency is None or importlib.util.find_spec(dependency) is not None:
            available.append(contender)
            continue

        if dependency == "torch":
            if getattr(config.torch, "allow_optional_missing", True):
                continue
        elif getattr(config.boosted_trees, "allow_optional_missing", True):
            continue

        available.append(contender)
    return available


def _budget_matched_slots(config: RunConfig, *, benchmark_name: str) -> int:
    target = config.baseline.target_evaluation_count
    if target is None:
        raise ValueError("budget_matched mode requires baseline.target_evaluation_count")
    benchmark_names = config.benchmark_pool.benchmarks
    benchmark_count = len(benchmark_names)
    if benchmark_count == 0:
        raise ValueError("budget_matched mode requires at least one benchmark")
    if target < benchmark_count:
        raise ValueError(
            "budget_matched target_evaluation_count must be >= benchmark_count "
            f"({target} < {benchmark_count})"
        )
    try:
        benchmark_index = benchmark_names.index(benchmark_name)
    except ValueError as exc:
        raise ValueError(f"benchmark '{benchmark_name}' not found in benchmark pool") from exc

    base_slots = target // benchmark_count
    remainder = target % benchmark_count
    return base_slots + (1 if benchmark_index < remainder else 0)
