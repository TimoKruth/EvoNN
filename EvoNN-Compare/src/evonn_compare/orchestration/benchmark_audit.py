"""Benchmark ladder admission audit."""

from __future__ import annotations

from datetime import datetime, timezone
from pathlib import Path
from typing import Any

from evonn_compare.contracts.parity import ParityBenchmark, ParityPack, load_parity_pack
from evonn_compare.orchestration.benchmark_resolution import resolve_benchmark_support
from evonn_compare.orchestration.lane_presets import LANE_PRESETS
from evonn_shared.manifests import write_json

SYSTEMS = ("prism", "topograph", "stratograph", "primordia", "contenders")
BOUNDED_METRICS = {"accuracy", "f1", "auc"}


def audit_benchmark_pack(*, pack_name: str, output: Path | None = None) -> dict[str, Any]:
    """Audit whether a parity pack is ready for recurring benchmark lanes."""

    pack = load_parity_pack(pack_name)
    benchmark_rows = [_audit_benchmark(pack=pack, benchmark=benchmark) for benchmark in pack.benchmarks]
    budget_presets = _budget_presets_for_pack(pack)
    blocking_count = sum(1 for row in benchmark_rows if row["admission_status"] == "blocked")
    exploratory_count = sum(1 for row in benchmark_rows if row["admission_status"] == "exploratory_only")
    payload = {
        "generated_at": datetime.now(timezone.utc).isoformat(),
        "pack_name": pack.name,
        "ladder_tier": pack.ladder_tier,
        "benchmark_count": len(pack.benchmarks),
        "budget_presets": budget_presets,
        "budget_divisibility": [
            {
                "budget": budget,
                "benchmark_count": len(pack.benchmarks),
                "divisible": len(pack.benchmarks) > 0 and budget % len(pack.benchmarks) == 0,
            }
            for budget in budget_presets
        ],
        "summary": {
            "blocked_count": blocking_count,
            "exploratory_count": exploratory_count,
            "decision_grade_count": len(benchmark_rows) - blocking_count - exploratory_count,
            "audit_status": "blocked" if blocking_count else ("exploratory" if exploratory_count else "passed"),
        },
        "benchmarks": benchmark_rows,
    }
    if output is not None:
        _write_audit_outputs(payload=payload, output=output)
    return payload


def _audit_benchmark(*, pack: ParityPack, benchmark: ParityBenchmark) -> dict[str, Any]:
    support = {system: resolve_benchmark_support(benchmark, system) for system in SYSTEMS}
    required_contenders = _minimum_required_contenders(pack=pack, benchmark=benchmark)
    optional_contenders = tuple(benchmark.enhanced_optional_contenders)
    score_ceiling = _score_ceiling(pack=pack, benchmark=benchmark)
    contender_floor = _contender_floor_status(benchmark=benchmark, required_contenders=required_contenders)
    blockers = []
    notes = []
    if any(not row["supported"] for row in support.values()):
        blockers.append("unsupported-system-benchmark")
    if not benchmark.metric_name or benchmark.metric_direction not in {"max", "min"}:
        blockers.append("metric-metadata-missing")
    if benchmark.metric_name in BOUNDED_METRICS and score_ceiling is None:
        blockers.append("score-ceiling-missing")
    if contender_floor["floor_status"] in {"missing", "failed"}:
        blockers.append("required-contender-floor-missing")
    if not required_contenders:
        notes.append("missing required contender metadata keeps benchmark exploratory")

    if blockers:
        admission_status = "blocked"
    elif notes or _pack_requires_exploratory_admission(pack):
        admission_status = "exploratory_only"
    else:
        admission_status = "decision_grade"
    return {
        "benchmark_id": benchmark.benchmark_id,
        "task_kind": benchmark.task_kind,
        "benchmark_group": _benchmark_group(benchmark),
        "metric_name": benchmark.metric_name,
        "metric_direction": benchmark.metric_direction,
        "score_ceiling": score_ceiling,
        "required_contenders": list(required_contenders),
        "enhanced_optional_contenders": list(optional_contenders),
        "support": support,
        "floor_status": contender_floor["floor_status"],
        "required_contender_resolution": contender_floor["required_contender_resolution"],
        "enhanced_optional_resolution": contender_floor["enhanced_optional_resolution"],
        "admission_status": admission_status,
        "blockers": blockers,
        "notes": notes,
        "admission_notes": benchmark.admission_notes,
    }


def _contender_floor_status(*, benchmark: ParityBenchmark, required_contenders: tuple[str, ...]) -> dict[str, Any]:
    group = _benchmark_group(benchmark)
    required = _resolve_contender_names(group=group, names=required_contenders)
    optional = _resolve_contender_names(group=group, names=benchmark.enhanced_optional_contenders)
    required_ok = [row for row in required if row["status"] == "configured"]
    if not required_contenders:
        status = "missing"
    elif required_ok:
        status = "passed"
    else:
        status = "failed"
    return {
        "floor_status": status,
        "required_contender_resolution": required,
        "enhanced_optional_resolution": optional,
    }


def _minimum_required_contenders(*, pack: ParityPack, benchmark: ParityBenchmark) -> tuple[str, ...]:
    if benchmark.minimum_required_contenders:
        return tuple(benchmark.minimum_required_contenders)
    if pack.name.startswith("tier1_core"):
        group = _benchmark_group(benchmark)
        if benchmark.task_kind == "regression":
            return ("hist_gb", "extra_trees", "ridge_or_linear")
        if group == "image":
            return ("mlp_wide", "extra_trees")
        if group == "synthetic":
            return ("hist_gb", "extra_trees", "svm_nystroem_rbf")
        return ("hist_gb", "extra_trees", "linear_svc")
    return ()


def _score_ceiling(*, pack: ParityPack, benchmark: ParityBenchmark) -> float | None:
    if benchmark.score_ceiling is not None:
        return benchmark.score_ceiling
    if pack.name.startswith("tier1_core") and benchmark.metric_name in BOUNDED_METRICS:
        return 1.0
    return None


def _resolve_contender_names(*, group: str, names: tuple[str, ...]) -> list[dict[str, str]]:
    if not names:
        return []
    try:
        from evonn_contenders.contenders.registry import resolve_contenders
    except Exception as exc:
        return [{"name": name, "status": "registry-unavailable", "reason": str(exc)} for name in names]
    rows = []
    for name in names:
        try:
            contender = resolve_contenders(group, [name])[0]
            rows.append(
                {
                    "name": name,
                    "status": "configured",
                    "family": contender.family,
                    "backend": contender.backend,
                    "optional_dependency": contender.optional_dependency or "",
                }
            )
        except Exception as exc:
            rows.append({"name": name, "status": "missing", "reason": str(exc)})
    return rows


def _benchmark_group(benchmark: ParityBenchmark) -> str:
    if benchmark.benchmark_group:
        return benchmark.benchmark_group
    if benchmark.task_kind == "language_modeling":
        return "language_modeling"
    family = (benchmark.benchmark_family or "").lower()
    benchmark_id = benchmark.benchmark_id.lower()
    if "image" in family or benchmark_id.endswith("_image") or benchmark_id in {"digits_image", "fashionmnist_image", "mnist_image"}:
        return "image"
    if "synthetic" in family or benchmark_id.startswith(("moons", "circles", "blobs")):
        return "synthetic"
    return "tabular"


def _budget_presets_for_pack(pack: ParityPack) -> list[int]:
    budgets = {
        int(preset.budgets[0])
        for preset in LANE_PRESETS.values()
        if preset.pack == pack.name and preset.budgets
    }
    for usage in (pack.usage_classification or {}).values():
        if isinstance(usage, dict) and usage.get("evaluation_count") is not None:
            budgets.add(int(usage["evaluation_count"]))
    if pack.budget_policy.evaluation_count:
        budgets.add(int(pack.budget_policy.evaluation_count))
    return sorted(budgets)


def _pack_requires_exploratory_admission(pack: ParityPack) -> bool:
    if pack.ladder_tier not in {"C", "D", "E"}:
        return False
    requirements = pack.promotion_requirements or {}
    return requirements.get("promotion_status") != "decision_grade"


def _write_audit_outputs(*, payload: dict[str, Any], output: Path) -> None:
    output.parent.mkdir(parents=True, exist_ok=True)
    if output.suffix == ".json":
        json_path = output
        md_path = output.with_suffix(".md")
    else:
        md_path = output
        json_path = output.with_suffix(".json")
    write_json(json_path, payload)
    md_path.write_text(render_benchmark_audit_markdown(payload), encoding="utf-8")


def render_benchmark_audit_markdown(payload: dict[str, Any]) -> str:
    lines = [
        f"# Benchmark Audit: {payload['pack_name']}",
        "",
        f"- generated_at: `{payload['generated_at']}`",
        f"- ladder_tier: `{payload.get('ladder_tier') or 'none'}`",
        f"- benchmark_count: `{payload['benchmark_count']}`",
        f"- audit_status: `{payload['summary']['audit_status']}`",
        "",
        "## Budget Divisibility",
        "",
        "| Budget | Benchmark Count | Divisible |",
        "| ---: | ---: | --- |",
    ]
    for row in payload["budget_divisibility"]:
        lines.append(f"| {row['budget']} | {row['benchmark_count']} | {'yes' if row['divisible'] else 'no'} |")
    lines.extend([
        "",
        "## Benchmark Admission",
        "",
        "| Benchmark | Group | Metric | Required Floor | Floor Status | Admission | Blockers |",
        "| --- | --- | --- | --- | --- | --- | --- |",
    ])
    for row in payload["benchmarks"]:
        lines.append(
            "| {benchmark} | {group} | `{metric}` | `{required}` | `{floor}` | `{admission}` | `{blockers}` |".format(
                benchmark=row["benchmark_id"],
                group=row["benchmark_group"],
                metric=f"{row['metric_name']}:{row['metric_direction']}",
                required=", ".join(row["required_contenders"]) or "none",
                floor=row["floor_status"],
                admission=row["admission_status"],
                blockers=", ".join(row["blockers"]) or "none",
            )
        )
    return "\n".join(lines) + "\n"
