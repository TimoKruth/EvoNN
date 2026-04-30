"""Helpers for engine-specialization reporting across fair-matrix trend rows."""

from __future__ import annotations

from collections import Counter, defaultdict
from typing import Any, Mapping


ENGINE_SPECIALIZATION_PROFILES: dict[str, dict[str, Any]] = {
    "prism": {
        "search_style": "family-policy search",
        "expected_signal": (
            "Prism changes should improve family-ranked tabular performance or broaden family coverage,"
            " not only bump one aggregate winner table."
        ),
        "branch_review_prompt": (
            "Check whether gains line up with family-policy or operator-mix changes instead of unrelated noise."
        ),
        "failure_hypotheses": (
            "family collapse",
            "operator regressions",
            "compile or runtime failures",
        ),
    },
    "topograph": {
        "search_style": "topology and skip-connection search",
        "expected_signal": (
            "Topograph changes should show up through stronger family ranks on topology-sensitive families"
            " or more coherent architecture summaries."
        ),
        "branch_review_prompt": (
            "Check whether gains appear where topology search should matter, especially image or sequence-style families."
        ),
        "failure_hypotheses": (
            "topology churn without stable wins",
            "cache or training inefficiency",
            "family transfer regressions",
        ),
    },
    "stratograph": {
        "search_style": "hierarchy and reuse search",
        "expected_signal": (
            "Stratograph changes should improve rank on families that reward hierarchy or reuse"
            " and reduce repeated benchmark failure concentration."
        ),
        "branch_review_prompt": (
            "Check whether hierarchy or motif-bias edits improve the intended benchmark families rather than only shifting aggregate ties."
        ),
        "failure_hypotheses": (
            "hierarchy overhead",
            "motif-bias mismatch",
            "reuse instability",
        ),
    },
    "primordia": {
        "search_style": "primitive and motif search",
        "expected_signal": (
            "Primordia changes should improve primitive-driven family rank or produce more credible"
            " architecture summaries for downstream seeding."
        ),
        "branch_review_prompt": (
            "Check whether primitive or motif edits improve the families they are meant to seed or stabilize."
        ),
        "failure_hypotheses": (
            "primitive mismatch",
            "weak motif transfer",
            "seed-ready artifacts without direct wins",
        ),
    },
    "contenders": {
        "search_style": "baseline floor pressure",
        "expected_signal": (
            "Contender changes should raise the floor or reveal weak EvoNN families,"
            " not create unexplained leaderboard churn."
        ),
        "branch_review_prompt": (
            "Check whether stronger baselines pressure the intended benchmark families and keep claims honest."
        ),
        "failure_hypotheses": (
            "optional baseline gaps",
            "dependency skips",
            "weak floor on target family",
        ),
    },
}

_IMAGE_HINTS = ("image", "mnist", "fashion", "cifar", "svhn")
_LANGUAGE_HINTS = ("lm", "language", "text", "stories", "sequence", "token")
_SYNTHETIC_HINTS = ("moons", "circles", "friedman")


def infer_benchmark_family(benchmark_id: str, task_kind: str | None = None, explicit_family: str | None = None) -> str:
    """Infer a stable benchmark-family label for specialization reporting."""

    if explicit_family:
        return str(explicit_family)

    normalized = benchmark_id.lower()
    task = str(task_kind or "unknown")
    if task == "language_modeling" or any(hint in normalized for hint in _LANGUAGE_HINTS):
        return "language-modeling"
    if any(hint in normalized for hint in _IMAGE_HINTS):
        if task == "regression":
            return "image-regression"
        return "image-classification"
    if task == "regression":
        if any(hint in normalized for hint in _SYNTHETIC_HINTS):
            return "synthetic-regression"
        return "tabular-regression"
    if task == "classification":
        if any(hint in normalized for hint in _SYNTHETIC_HINTS):
            return "synthetic-classification"
        return "tabular-classification"
    return task.replace("_", "-")


def infer_search_profile(system: str) -> str:
    return str(ENGINE_SPECIALIZATION_PROFILES.get(system, {}).get("search_style") or "unknown")


def infer_expected_specialization(system: str) -> str:
    return str(ENGINE_SPECIALIZATION_PROFILES.get(system, {}).get("expected_signal") or "No specialization guidance recorded.")


def build_family_leaderboards(rows: list[Any], *, systems: tuple[str, ...]) -> list[dict[str, Any]]:
    """Aggregate fair-matrix rows into per-family engine rankings."""

    run_groups: dict[tuple[str, int, int, str, str, str], list[Any]] = defaultdict(list)
    for row in rows:
        system = str(_row_value(row, "system"))
        if system not in systems:
            continue
        key = (
            _comparison_label(row),
            str(_row_value(row, "pack_name")),
            int(_row_value(row, "budget")),
            int(_row_value(row, "seed")),
            str(_benchmark_family(row)),
            _comparison_case_id(row),
        )
        run_groups[key].append(row)

    aggregated_groups: dict[tuple[str, str, int, str], dict[str, Any]] = {}
    for (comparison_label, pack_name, budget, seed, family, _group_case_id), family_rows in sorted(run_groups.items()):
        scope = _build_scope_summary(family_rows, systems=systems)
        group = aggregated_groups.setdefault(
            (comparison_label, pack_name, budget, family),
            {
                "comparison_label": comparison_label,
                "comparison_cohort": _comparison_cohort(family_rows[0]),
                "pack_name": pack_name,
                "budget": budget,
                "benchmark_family": family,
                "benchmark_ids": set(),
                "seeds": set(),
                "totals": {
                    system: {
                        "system": system,
                        "runs": 0,
                        "solo_wins": 0,
                        "shared_wins": 0,
                        "benchmark_failures": 0,
                        "missing_results": 0,
                    }
                    for system in systems
                },
            },
        )
        group["seeds"].add(seed)
        group["benchmark_ids"].update(
            str(_row_value(row, "benchmark_id"))
            for row in family_rows
        )
        row_map = {entry["system"]: entry for entry in scope["rows"]}
        for system in systems:
            total = group["totals"][system]
            entry = row_map[system]
            total["runs"] += 1
            total["solo_wins"] += int(entry["solo_wins"])
            total["shared_wins"] += int(entry["shared_wins"])
            total["benchmark_failures"] += int(entry["benchmark_failures"])
            total["missing_results"] += int(entry["missing_results"])

    family_groups: list[dict[str, Any]] = []
    for (_group_label, _pack_name, _budget, _family), group in sorted(aggregated_groups.items()):
        ranked_rows = []
        for system in systems:
            total = dict(group["totals"][system])
            total["score"] = float(total["solo_wins"]) + 0.5 * float(total["shared_wins"])
            total["mean_score"] = total["score"] / total["runs"] if total["runs"] else 0.0
            ranked_rows.append(total)
        ranked_rows = sorted(
            ranked_rows,
            key=lambda item: (-float(item["mean_score"]), -int(item["solo_wins"]), int(item["benchmark_failures"]), item["system"]),
        )
        for index, row in enumerate(ranked_rows, start=1):
            row["rank"] = index
            row["system_count"] = len(ranked_rows)
        family_groups.append(
            {
                "comparison_label": group["comparison_label"],
                "comparison_cohort": group["comparison_cohort"],
                "pack_name": group["pack_name"],
                "budget": group["budget"],
                "benchmark_family": group["benchmark_family"],
                "benchmark_count": len(group["benchmark_ids"]),
                "seed_count": len(group["seeds"]),
                "system_rows": ranked_rows,
            }
        )
    return family_groups


def build_engine_profiles(rows: list[Any], *, systems: tuple[str, ...]) -> list[dict[str, Any]]:
    """Build per-engine specialization summaries from trend rows."""

    family_groups = build_family_leaderboards(rows, systems=systems)
    per_system_rows: dict[str, list[Any]] = defaultdict(list)
    for row in rows:
        system = str(_row_value(row, "system"))
        if system in systems:
            per_system_rows[system].append(row)

    profiles: list[dict[str, Any]] = []
    labels = sorted({_comparison_label(row) for row in rows}) or ["current-workspace"]
    for comparison_label in labels:
        comparison_rows = [row for row in rows if _comparison_label(row) == comparison_label]
        comparison_cohort = _comparison_cohort(comparison_rows[0]) if comparison_rows else "current-workspace"
        for system in systems:
            system_rows = [row for row in per_system_rows.get(system, []) if _comparison_label(row) == comparison_label]
            status_counts = Counter(str(_row_value(row, "outcome_status", "missing")) for row in system_rows)
            failure_patterns = Counter(
                str(_row_value(row, "failure_reason"))
                for row in system_rows
                if _row_value(row, "outcome_status") == "failed" and _row_value(row, "failure_reason")
            )
            architecture_examples = Counter(
                str(_row_value(row, "architecture_summary"))
                for row in system_rows
                if _row_value(row, "outcome_status") == "ok" and _row_value(row, "architecture_summary")
            )
            family_rows = []
            for group in family_groups:
                if group["comparison_label"] != comparison_label:
                    continue
                for row in group["system_rows"]:
                    if row["system"] != system:
                        continue
                    family_rows.append(
                        {
                            "comparison_label": group["comparison_label"],
                            "comparison_cohort": group["comparison_cohort"],
                            "pack_name": group["pack_name"],
                            "budget": group["budget"],
                            "benchmark_family": group["benchmark_family"],
                            "benchmark_count": group["benchmark_count"],
                            "seed_count": group["seed_count"],
                            **row,
                        }
                    )
            strengths = sorted(
                family_rows,
                key=lambda row: (int(row["rank"]), -float(row["mean_score"]), int(row["benchmark_failures"]), int(row["missing_results"])),
            )[:3]
            weaknesses = sorted(
                family_rows,
                key=lambda row: (-int(row["rank"]), -int(row["benchmark_failures"]), -int(row["missing_results"]), float(row["mean_score"])),
            )[:3]
            spec = ENGINE_SPECIALIZATION_PROFILES.get(system, {})
            profiles.append(
                {
                    "comparison_label": comparison_label,
                    "comparison_cohort": comparison_cohort,
                    "system": system,
                    "search_style": str(spec.get("search_style") or "unknown"),
                    "expected_signal": str(spec.get("expected_signal") or "No specialization guidance recorded."),
                    "branch_review_prompt": str(
                        spec.get("branch_review_prompt")
                        or "Check whether changes improved the intended specialization instead of only changing the aggregate table."
                    ),
                    "failure_hypotheses": list(spec.get("failure_hypotheses") or ()),
                    "status_counts": {
                        "ok": int(status_counts.get("ok", 0)),
                        "failed": int(status_counts.get("failed", 0)),
                        "skipped": int(status_counts.get("skipped", 0)),
                        "unsupported": int(status_counts.get("unsupported", 0)),
                        "missing": int(status_counts.get("missing", 0)),
                    },
                    "family_strengths": strengths,
                    "family_weaknesses": weaknesses,
                    "failure_patterns": [
                        {"reason": reason, "count": count}
                        for reason, count in failure_patterns.most_common(3)
                    ],
                    "architecture_examples": [
                        {"summary": summary, "count": count}
                        for summary, count in architecture_examples.most_common(3)
                    ],
                }
            )
    return profiles


def _benchmark_family(row: Any) -> str:
    return infer_benchmark_family(
        str(_row_value(row, "benchmark_id")),
        task_kind=_row_value(row, "task_kind"),
        explicit_family=_row_value(row, "benchmark_family"),
    )


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


def _build_scope_summary(rows: list[Any], *, systems: tuple[str, ...]) -> dict[str, Any]:
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
