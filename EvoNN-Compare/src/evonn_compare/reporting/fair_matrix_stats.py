"""Seed-aware aggregate helpers for fair-matrix trend datasets."""

from __future__ import annotations

from collections import defaultdict
import itertools
from statistics import mean, pstdev
from typing import Any, Mapping
import zlib

from evonn_compare.comparison.statistics import bootstrap_confidence_interval, two_sided_sign_test


def build_scope_summary(rows: list[Any], *, systems: tuple[str, ...]) -> dict[str, Any]:
    by_benchmark: dict[str, list[Any]] = defaultdict(list)
    for row in rows:
        system = str(_row_value(row, "system"))
        if system in systems:
            by_benchmark[str(_row_value(row, "benchmark_id"))].append(row)

    summary_rows = {
        system: {
            "system": system,
            "solo_wins": 0,
            "shared_wins": 0,
            "benchmark_failures": 0,
            "missing_results": 0,
        }
        for system in systems
    }
    ties = 0
    skipped = 0
    for benchmark_rows in by_benchmark.values():
        ok_rows: list[Any] = []
        for row in benchmark_rows:
            system_row = summary_rows[str(_row_value(row, "system"))]
            status = str(_row_value(row, "outcome_status", "missing"))
            if status == "failed":
                system_row["benchmark_failures"] += 1
            if status == "missing":
                system_row["missing_results"] += 1
            if status == "ok" and _row_value(row, "metric_value") is not None:
                ok_rows.append(row)
        if not ok_rows:
            skipped += 1
            continue
        direction = str(_row_value(ok_rows[0], "metric_direction"))
        values = [float(_row_value(row, "metric_value")) for row in ok_rows]
        best_value = max(values) if direction == "max" else min(values)
        winners = [
            str(_row_value(row, "system"))
            for row in ok_rows
            if abs(float(_row_value(row, "metric_value")) - best_value) <= 1e-12
        ]
        if len(winners) == 1:
            summary_rows[winners[0]]["solo_wins"] += 1
        else:
            ties += 1
            for winner in winners:
                summary_rows[winner]["shared_wins"] += 1
    return {"rows": [summary_rows[system] for system in systems], "ties": ties, "skipped": skipped}


def build_scope_run_summaries(rows: list[Any], *, systems: tuple[str, ...]) -> list[dict[str, Any]]:
    grouped_rows: dict[tuple[str, int, int, str, str], list[Any]] = defaultdict(list)
    for row in rows:
        system = str(_row_value(row, "system"))
        if system not in systems:
            continue
        key = (
            str(_row_value(row, "pack_name")),
            int(_row_value(row, "budget")),
            int(_row_value(row, "seed")),
            _comparison_label(row),
            _comparison_case_id(row),
        )
        grouped_rows[key].append(row)

    run_summaries: list[dict[str, Any]] = []
    for (pack_name, budget, seed, comparison_label, comparison_case_id), run_rows in sorted(grouped_rows.items()):
        first_row = sorted(
            run_rows,
            key=lambda row: (
                str(_row_value(row, "system")),
                str(_row_value(row, "benchmark_id")),
                str(_row_value(row, "run_id")),
            ),
        )[0]
        run_summaries.append(
            {
                "pack_name": pack_name,
                "budget": budget,
                "seed": seed,
                "comparison_label": comparison_label,
                "comparison_cohort": _comparison_cohort(first_row),
                "comparison_case_id": comparison_case_id,
                "lane_operating_state": str(_row_value(first_row, "lane_operating_state", "reference-only")),
                "repeatability_ready": bool(_row_value(first_row, "lane_repeatability_ready", False)),
                "budget_accounting_ok": bool(_row_value(first_row, "lane_budget_accounting_ok", False)),
                "scope": build_scope_summary(run_rows, systems=systems),
            }
        )
    return run_summaries


def build_multi_seed_statistics(rows: list[Any], *, systems: tuple[str, ...]) -> list[dict[str, Any]]:
    run_summaries = build_scope_run_summaries(rows, systems=systems)
    grouped_runs: dict[tuple[str, str, int], list[dict[str, Any]]] = defaultdict(list)
    for run in run_summaries:
        grouped_runs[(str(run["comparison_label"]), str(run["pack_name"]), int(run["budget"]))].append(run)

    groups: list[dict[str, Any]] = []
    for (comparison_label, pack_name, budget), runs in sorted(grouped_runs.items()):
        ordered_runs = sorted(runs, key=lambda run: (int(run["seed"]), str(run["pack_name"])))
        system_rows = []
        for system in systems:
            per_run_rows = [_scope_row(run, system) for run in ordered_runs]
            scores = [_system_score(row) for row in per_run_rows]
            ci = _confidence_interval(scores, label=f"{pack_name}:{budget}:{system}:score")
            system_rows.append(
                {
                    "system": system,
                    "seed_count": len(scores),
                    "seeds": [int(run["seed"]) for run in ordered_runs],
                    "mean_score": _mean(scores),
                    "score_stddev": _stddev(scores),
                    "score_ci95_low": ci["low"],
                    "score_ci95_high": ci["high"],
                    "best_score": max(scores) if scores else 0.0,
                    "worst_score": min(scores) if scores else 0.0,
                    "score_range": (max(scores) - min(scores)) if scores else 0.0,
                    "mean_solo_wins": _mean([float(row["solo_wins"]) for row in per_run_rows]),
                    "mean_shared_wins": _mean([float(row["shared_wins"]) for row in per_run_rows]),
                    "mean_benchmark_failures": _mean([float(row["benchmark_failures"]) for row in per_run_rows]),
                    "mean_missing_results": _mean([float(row["missing_results"]) for row in per_run_rows]),
                }
            )

        pairwise = []
        for left_system, right_system in itertools.combinations(systems, 2):
            left_scores = [_system_score(_scope_row(run, left_system)) for run in ordered_runs]
            right_scores = [_system_score(_scope_row(run, right_system)) for run in ordered_runs]
            deltas = [left - right for left, right in zip(left_scores, right_scores, strict=True)]
            ci = _confidence_interval(deltas, label=f"{pack_name}:{budget}:{left_system}:{right_system}:delta")
            left_better = sum(1 for delta in deltas if delta > 1e-12)
            right_better = sum(1 for delta in deltas if delta < -1e-12)
            ties = sum(1 for delta in deltas if abs(delta) <= 1e-12)
            pairwise.append(
                {
                    "left_system": left_system,
                    "right_system": right_system,
                    "seed_count": len(deltas),
                    "left_better": left_better,
                    "right_better": right_better,
                    "ties": ties,
                    "mean_score_delta": _mean(deltas),
                    "score_delta_stddev": _stddev(deltas),
                    "score_delta_ci95_low": ci["low"],
                    "score_delta_ci95_high": ci["high"],
                    "sign_test_p_value": two_sided_sign_test(left_better, right_better),
                }
            )

        groups.append(
            {
                "comparison_label": comparison_label,
                "comparison_cohort": str(ordered_runs[0].get("comparison_cohort") or "current-workspace"),
                "pack_name": pack_name,
                "budget": budget,
                "seed_count": len(ordered_runs),
                "seeds": [int(run["seed"]) for run in ordered_runs],
                "system_rows": system_rows,
                "pairwise": pairwise,
            }
        )
    return groups


def _scope_row(run: dict[str, Any], system: str) -> dict[str, Any]:
    for row in run["scope"]["rows"]:
        if row["system"] == system:
            return row
    raise KeyError(f"missing scope row for {system}")


def _system_score(row: Mapping[str, Any]) -> float:
    return float(row["solo_wins"]) + 0.5 * float(row["shared_wins"])


def _confidence_interval(values: list[float], *, label: str) -> dict[str, float]:
    if not values:
        return {"low": 0.0, "high": 0.0}
    if len(values) == 1:
        return {"low": float(values[0]), "high": float(values[0])}
    ci = bootstrap_confidence_interval(values, n_resamples=2_000, seed=_stable_seed(label))
    return {"low": ci.lower, "high": ci.upper}


def _stable_seed(label: str) -> int:
    return zlib.crc32(label.encode("utf-8")) & 0xFFFFFFFF


def _mean(values: list[float]) -> float:
    if not values:
        return 0.0
    return float(mean(values))


def _stddev(values: list[float]) -> float:
    if len(values) < 2:
        return 0.0
    return float(pstdev(values))


def _row_value(row: Any, key: str, default: Any = None) -> Any:
    if isinstance(row, Mapping):
        return row.get(key, default)
    return getattr(row, key, default)


def _comparison_metadata(row: Any) -> Mapping[str, Any]:
    fairness = _row_value(row, "fairness_metadata", {})
    if isinstance(fairness, Mapping):
        return fairness
    return {}


def _comparison_label(row: Any) -> str:
    fairness = _comparison_metadata(row)
    return str(fairness.get("comparison_label") or fairness.get("baseline_label") or fairness.get("comparison_cohort") or "current-workspace")


def _comparison_cohort(row: Any) -> str:
    fairness = _comparison_metadata(row)
    return str(fairness.get("comparison_cohort") or "current-workspace")


def _comparison_case_id(row: Any) -> str:
    fairness = _comparison_metadata(row)
    value = fairness.get("comparison_case_id")
    if value:
        return str(value)
    return f"{_comparison_label(row)}:{_row_value(row, 'pack_name')}:{_row_value(row, 'budget')}:{_row_value(row, 'seed')}"
