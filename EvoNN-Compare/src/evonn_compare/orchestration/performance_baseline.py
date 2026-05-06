"""Performance baseline workflow for compare-grade run artifacts."""

from __future__ import annotations

from collections import defaultdict
from datetime import datetime, timezone
import json
from pathlib import Path
import subprocess
from typing import Any, Iterable

from evonn_compare.output_quality import OutputQualityRecord, inspect_paths
from evonn_shared.manifests import write_json

REQUIRED_BUDGETS = (64, 256, 1000)


def build_performance_baseline(
    *,
    inputs: Iterable[Path],
    output_root: Path | None = None,
    label: str | None = None,
    required_budgets: tuple[int, ...] = REQUIRED_BUDGETS,
    write_run_artifacts: bool = True,
) -> dict[str, Any]:
    """Build a performance-baseline bundle from compare-grade run directories."""

    records = inspect_paths(list(inputs), write_run_artifacts=write_run_artifacts)
    generated_at = datetime.now(timezone.utc).isoformat()
    git_sha = _git_sha(Path.cwd())
    bundle_root = (output_root or Path("performance_baselines")) / f"{_stamp(generated_at)}-{git_sha}"
    bundle_root.mkdir(parents=True, exist_ok=True)

    run_rows = [_record_payload(record, required_budgets=required_budgets) for record in records]
    systems = _aggregate_systems(run_rows, required_budgets=required_budgets)
    overview = {
        "generated_at": generated_at,
        "label": label or "performance-baseline",
        "git_sha": git_sha,
        "required_budgets": list(required_budgets),
        "run_count": len(run_rows),
        "accepted_run_count": sum(1 for row in run_rows if row["counts_for_baseline"]),
        "systems": systems,
    }

    json_path = bundle_root / "performance_baseline.json"
    md_path = bundle_root / "performance_baseline.md"
    jsonl_path = bundle_root / "run_records.jsonl"

    write_json(json_path, {**overview, "runs": run_rows})
    md_path.write_text(_render_markdown(overview, run_rows), encoding="utf-8")
    jsonl_path.write_text("".join(json.dumps(row) + "\n" for row in run_rows), encoding="utf-8")

    return {
        "bundle_root": str(bundle_root.resolve()),
        "json": str(json_path.resolve()),
        "markdown": str(md_path.resolve()),
        "jsonl": str(jsonl_path.resolve()),
        "run_count": len(run_rows),
        "accepted_run_count": overview["accepted_run_count"],
        "system_count": len(systems),
    }


def _record_payload(record: OutputQualityRecord, *, required_budgets: tuple[int, ...]) -> dict[str, Any]:
    manifest = json.loads(record.manifest_path.read_text(encoding="utf-8"))
    budget = _as_int(((manifest.get("budget") or {}).get("evaluation_count")))
    pack_name = str(manifest.get("pack_name") or manifest.get("benchmark_pack_id") or "unknown")
    fairness = manifest.get("fairness") or {}
    fairness_ok = all(
        fairness.get(field) not in (None, "")
        for field in ("benchmark_pack_id", "seed", "evaluation_count", "budget_policy_name", "data_signature", "code_version")
    )
    quality_ok = record.quality_level in {"L3", "L4"}
    counts = fairness_ok and quality_ok and budget in required_budgets
    benchmark_throughput = _ratio(record.completed_benchmark_count, record.performance.wall_clock_seconds)
    failure_adjusted_throughput = _ratio(
        record.completed_benchmark_count / max(record.benchmark_count, 1),
        record.performance.wall_clock_seconds,
    )
    return {
        "system": record.system,
        "run_id": record.run_id,
        "run_dir": str(record.run_dir),
        "pack_name": pack_name,
        "budget": budget,
        "quality_level": record.quality_level,
        "measurement_state": record.measurement_state,
        "counts_for_baseline": counts,
        "exclusion_reasons": _exclusion_reasons(record, fairness_ok=fairness_ok, budget=budget, required_budgets=required_budgets),
        "runtime": {
            "backend": record.runtime.runtime_backend,
            "backend_requested": record.runtime.runtime_backend_requested,
            "device_name": record.runtime.device_name,
            "hardware_class": record.runtime.hardware_class,
            "framework": record.runtime.framework,
            "framework_version": record.runtime.framework_version,
            "worker_count": record.runtime.worker_count,
        },
        "performance": {
            "wall_clock_seconds": record.performance.wall_clock_seconds,
            "evals_per_second": record.performance.evals_per_second,
            "quality_per_second": record.performance.quality_per_second,
            "cache_reuse_rate": record.performance.cache_reuse_rate,
            "peak_memory_mb": record.performance.peak_memory_mb,
            "benchmark_throughput": benchmark_throughput,
            "failure_adjusted_throughput": failure_adjusted_throughput,
        },
        "benchmark_count": record.benchmark_count,
        "completed_benchmark_count": record.completed_benchmark_count,
        "failed_benchmark_count": record.failed_benchmark_count,
    }


def _aggregate_systems(run_rows: list[dict[str, Any]], *, required_budgets: tuple[int, ...]) -> list[dict[str, Any]]:
    grouped: dict[str, list[dict[str, Any]]] = defaultdict(list)
    for row in run_rows:
        grouped[str(row["system"])].append(row)

    systems = []
    for system, rows in sorted(grouped.items()):
        accepted = [row for row in rows if row["counts_for_baseline"]]
        budgets_present = sorted({int(row["budget"]) for row in accepted if row.get("budget") is not None})
        missing_budgets = [budget for budget in required_budgets if budget not in budgets_present]
        systems.append(
            {
                "system": system,
                "run_count": len(rows),
                "accepted_run_count": len(accepted),
                "budgets_present": budgets_present,
                "missing_budgets": missing_budgets,
                "performance_claim_ready": not missing_budgets and len(accepted) >= len(required_budgets),
                "backend_labels": sorted({row["runtime"].get("backend") or "unknown" for row in accepted}),
                "hardware_labels": sorted(
                    {
                        row["runtime"].get("hardware_class")
                        or row["runtime"].get("device_name")
                        or "unknown"
                        for row in accepted
                    }
                ),
                "median_wall_clock_seconds": _median([
                    row["performance"].get("wall_clock_seconds") for row in accepted
                ]),
                "median_evals_per_second": _median([
                    row["performance"].get("evals_per_second") for row in accepted
                ]),
                "median_quality_per_second": _median([
                    row["performance"].get("quality_per_second") for row in accepted
                ]),
                "median_benchmark_throughput": _median([
                    row["performance"].get("benchmark_throughput") for row in accepted
                ]),
                "median_failure_adjusted_throughput": _median([
                    row["performance"].get("failure_adjusted_throughput") for row in accepted
                ]),
                "median_cache_reuse_rate": _median([
                    row["performance"].get("cache_reuse_rate") for row in accepted
                ]),
                "max_peak_memory_mb": _max([
                    row["performance"].get("peak_memory_mb") for row in accepted
                ]),
                "excluded_runs": [
                    {
                        "run_id": row["run_id"],
                        "budget": row["budget"],
                        "reasons": row["exclusion_reasons"],
                    }
                    for row in rows
                    if not row["counts_for_baseline"]
                ],
            }
        )
    return systems


def _render_markdown(overview: dict[str, Any], run_rows: list[dict[str, Any]]) -> str:
    lines = [
        "# Performance Baseline",
        "",
        f"- label: `{overview['label']}`",
        f"- generated_at: `{overview['generated_at']}`",
        f"- git_sha: `{overview['git_sha']}`",
        f"- required_budgets: `{', '.join(str(v) for v in overview['required_budgets'])}`",
        f"- accepted_runs: `{overview['accepted_run_count']}/{overview['run_count']}`",
        "",
        "## System Summary",
        "",
        "| System | Accepted | Budgets | Claim Ready | Wall s | Eval/s | Quality/s | Bench/s | Failure-Adj Bench/s | Cache Reuse | Backends | Hardware |",
        "| --- | ---: | --- | --- | ---: | ---: | ---: | ---: | ---: | ---: | --- | --- |",
    ]
    for row in overview["systems"]:
        lines.append(
            "| {system} | {accepted}/{total} | `{budgets}` | `{ready}` | {wall} | {eps} | {qps} | {bps} | {fbps} | {cache} | `{backends}` | `{hardware}` |".format(
                system=row["system"],
                accepted=row["accepted_run_count"],
                total=row["run_count"],
                budgets=", ".join(str(v) for v in row["budgets_present"]) or "none",
                ready="yes" if row["performance_claim_ready"] else "no",
                wall=_fmt(row["median_wall_clock_seconds"]),
                eps=_fmt(row["median_evals_per_second"]),
                qps=_fmt(row["median_quality_per_second"]),
                bps=_fmt(row["median_benchmark_throughput"]),
                fbps=_fmt(row["median_failure_adjusted_throughput"]),
                cache=_fmt(row["median_cache_reuse_rate"]),
                backends=", ".join(row["backend_labels"]) or "unknown",
                hardware=", ".join(row["hardware_labels"]) or "unknown",
            )
        )

    lines.extend(
        [
            "",
            "## Excluded Runs",
            "",
            "| System | Run ID | Budget | Reasons |",
            "| --- | --- | ---: | --- |",
        ]
    )
    excluded_any = False
    for row in run_rows:
        if row["counts_for_baseline"]:
            continue
        excluded_any = True
        lines.append(
            "| {system} | `{run_id}` | {budget} | `{reasons}` |".format(
                system=row["system"],
                run_id=row["run_id"],
                budget=row["budget"] if row["budget"] is not None else 0,
                reasons=", ".join(row["exclusion_reasons"]) or "none",
            )
        )
    if not excluded_any:
        lines.append("| _none_ |  |  |  |")
    return "\n".join(lines) + "\n"


def _exclusion_reasons(
    record: OutputQualityRecord,
    *,
    fairness_ok: bool,
    budget: int | None,
    required_budgets: tuple[int, ...],
) -> list[str]:
    reasons = []
    if record.quality_level not in {"L3", "L4"}:
        reasons.append(f"quality={record.quality_level}")
    if record.measurement_state != "measurable":
        reasons.append(f"measurement={record.measurement_state}")
    if not fairness_ok:
        reasons.append("fairness-metadata-incomplete")
    if budget not in required_budgets:
        reasons.append(f"budget-not-in-baseline-set:{budget}")
    return reasons


def _median(values: list[Any]) -> float | None:
    nums = sorted(float(value) for value in values if value is not None)
    if not nums:
        return None
    mid = len(nums) // 2
    if len(nums) % 2:
        return nums[mid]
    return (nums[mid - 1] + nums[mid]) / 2.0


def _max(values: list[Any]) -> float | None:
    nums = [float(value) for value in values if value is not None]
    return max(nums) if nums else None


def _ratio(numerator: float | int, denominator: float | None) -> float | None:
    if denominator is None or denominator <= 0:
        return None
    return float(numerator) / float(denominator)


def _fmt(value: float | None) -> str:
    return "-" if value is None else f"{value:.3f}"


def _stamp(iso_ts: str) -> str:
    return iso_ts.replace(":", "").replace("+00:00", "Z").replace("-", "")[:15]


def _git_sha(cwd: Path) -> str:
    try:
        output = subprocess.check_output(["git", "rev-parse", "--short", "HEAD"], cwd=cwd, text=True)
    except Exception:
        return "unknown"
    return output.strip() or "unknown"


def _as_int(value: Any) -> int | None:
    try:
        return int(value) if value is not None else None
    except (TypeError, ValueError):
        return None
