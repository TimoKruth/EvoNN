"""Four-way fairness summary helpers."""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Any

from evonn_compare.comparison.engine import ComparisonResult
from evonn_shared.contracts import ResultRecord, RunManifest
from evonn_compare.contracts.parity import ParityPack


SYSTEM_ORDER = ("prism", "topograph", "stratograph", "primordia", "contenders")
FOUR_PROJECT_SYSTEM_ORDER = ("prism", "topograph", "stratograph", "primordia")


@dataclass(frozen=True)
class MatrixBudgetRow:
    budget: int
    seed: int
    benchmark_count: int
    evaluation_counts: dict[str, int]
    wins: dict[str, int]
    ties: int
    note: str | None = None


@dataclass(frozen=True)
class PairParityRow:
    budget: int
    seed: int
    pair_label: str
    parity_status: str
    left_eval_count: int
    right_eval_count: int
    left_policy: str | None
    right_policy: str | None
    data_signature_match: bool
    reason: str | None
    comparison_report: Path


@dataclass(frozen=True)
class LaneMetadata:
    preset: str | None
    pack_name: str
    expected_budget: int
    expected_seed: int
    artifact_completeness_ok: bool
    fairness_ok: bool
    task_coverage_ok: bool
    budget_consistency_ok: bool
    seed_consistency_ok: bool
    observed_task_kinds: tuple[str, ...]
    acceptance_notes: tuple[str, ...]
    repeatability_ready: bool


@dataclass(frozen=True)
class MatrixTrendRow:
    pack_name: str
    budget: int
    seed: int
    system: str
    run_id: str
    benchmark_id: str
    metric_name: str
    metric_direction: str
    metric_value: float | None
    outcome_status: str
    failure_reason: str | None
    evaluation_count: int
    epochs_per_candidate: int
    budget_policy_name: str | None
    wall_clock_seconds: float | None
    matrix_scope: str
    fairness_metadata: dict[str, Any]


@dataclass(frozen=True)
class FairMatrixSummary:
    pack_name: str
    systems: tuple[str, ...]
    lane: LaneMetadata | None
    fair_rows: list[MatrixBudgetRow]
    reference_rows: list[MatrixBudgetRow]
    parity_rows: list[PairParityRow]
    trend_rows: list[MatrixTrendRow]


def summarize_matrix_case(
    *,
    pack: ParityPack,
    budget: int,
    seed: int,
    runs: dict[str, tuple[RunManifest, list[ResultRecord]]],
    pair_results: dict[tuple[str, str], tuple[ComparisonResult, Path]],
    systems: tuple[str, ...] = SYSTEM_ORDER,
) -> tuple[MatrixBudgetRow | None, MatrixBudgetRow | None, list[PairParityRow]]:
    parity_rows: list[PairParityRow] = []
    for (left_system, right_system), (result, report_path) in sorted(pair_results.items()):
        left_manifest = result.left_manifest
        right_manifest = result.right_manifest
        parity_rows.append(
            PairParityRow(
                budget=budget,
                seed=seed,
                pair_label=f"{left_system} vs {right_system}",
                parity_status=result.parity_status,
                left_eval_count=left_manifest.budget.evaluation_count,
                right_eval_count=right_manifest.budget.evaluation_count,
                left_policy=left_manifest.budget.budget_policy_name,
                right_policy=right_manifest.budget.budget_policy_name,
                data_signature_match=_data_signature(left_manifest) == _data_signature(right_manifest),
                reason="; ".join(result.reasons) if result.reasons else None,
                comparison_report=report_path,
            )
        )

    evaluation_counts = {
        system: manifest.budget.evaluation_count
        for system, (manifest, _results) in runs.items()
    }
    wins, ties = _winner_table(pack, runs, systems=systems)
    note = _reference_note(pair_results)
    row = MatrixBudgetRow(
        budget=budget,
        seed=seed,
        benchmark_count=len(pack.benchmarks),
        evaluation_counts=evaluation_counts,
        wins=wins,
        ties=ties,
        note=note,
    )
    if note is None:
        return row, None, parity_rows
    return None, row, parity_rows


def build_matrix_summary(
    *,
    pack_name: str,
    lane: LaneMetadata | None = None,
    fair_rows: list[MatrixBudgetRow],
    reference_rows: list[MatrixBudgetRow],
    parity_rows: list[PairParityRow],
    trend_rows: list[MatrixTrendRow] | None = None,
    systems: tuple[str, ...] = SYSTEM_ORDER,
) -> FairMatrixSummary:
    return FairMatrixSummary(
        pack_name=pack_name,
        systems=systems,
        lane=lane,
        fair_rows=sorted(fair_rows, key=lambda row: (row.budget, row.seed)),
        reference_rows=sorted(reference_rows, key=lambda row: (row.budget, row.seed)),
        parity_rows=sorted(parity_rows, key=lambda row: (row.budget, row.seed, row.pair_label)),
        trend_rows=sorted(
            trend_rows or [],
            key=lambda row: (row.budget, row.seed, row.system, row.benchmark_id),
        ),
    )


def build_matrix_trend_rows(
    *,
    pack: ParityPack,
    budget: int,
    seed: int,
    runs: dict[str, tuple[RunManifest, list[ResultRecord]]],
    pair_results: dict[tuple[str, str], tuple[ComparisonResult, Path]],
    systems: tuple[str, ...] = SYSTEM_ORDER,
) -> list[MatrixTrendRow]:
    matrix_scope = "fair" if _reference_note(pair_results) is None else "reference"
    by_system = {
        system: {record.benchmark_id: record for record in results}
        for system, (_manifest, results) in runs.items()
    }
    rows: list[MatrixTrendRow] = []
    for benchmark in pack.benchmarks:
        for system in systems:
            manifest_results = runs.get(system)
            if manifest_results is None:
                continue
            manifest, _results = manifest_results
            record = by_system.get(system, {}).get(benchmark.benchmark_id)
            benchmark_entry = next(
                (entry for entry in manifest.benchmarks if entry.benchmark_id == benchmark.benchmark_id),
                None,
            )
            status = record.status if record is not None else (benchmark_entry.status if benchmark_entry is not None else "missing")
            metric_name = record.metric_name if record is not None else benchmark.metric_name
            metric_direction = record.metric_direction if record is not None else benchmark.metric_direction
            fairness = manifest.fairness
            fairness_metadata = {
                "benchmark_pack_id": fairness.benchmark_pack_id if fairness is not None else manifest.pack_name,
                "seed": fairness.seed if fairness is not None else manifest.seed,
                "evaluation_count": fairness.evaluation_count if fairness is not None else manifest.budget.evaluation_count,
                "budget_policy_name": fairness.budget_policy_name if fairness is not None else manifest.budget.budget_policy_name,
                "data_signature": fairness.data_signature if fairness is not None else None,
                "code_version": fairness.code_version if fairness is not None else None,
                "pairwise_fairness_ok": matrix_scope == "fair",
            }
            rows.append(
                MatrixTrendRow(
                    pack_name=pack.name,
                    budget=budget,
                    seed=seed,
                    system=system,
                    run_id=manifest.run_id,
                    benchmark_id=benchmark.benchmark_id,
                    metric_name=metric_name,
                    metric_direction=metric_direction,
                    metric_value=record.metric_value if record is not None else None,
                    outcome_status=status,
                    failure_reason=record.failure_reason if record is not None else None,
                    evaluation_count=manifest.budget.evaluation_count,
                    epochs_per_candidate=manifest.budget.epochs_per_candidate,
                    budget_policy_name=manifest.budget.budget_policy_name,
                    wall_clock_seconds=manifest.budget.wall_clock_seconds,
                    matrix_scope=matrix_scope,
                    fairness_metadata=fairness_metadata,
                )
            )
    return rows


def _winner_table(
    pack: ParityPack,
    runs: dict[str, tuple[RunManifest, list[ResultRecord]]],
    *,
    systems: tuple[str, ...] = SYSTEM_ORDER,
) -> tuple[dict[str, int], int]:
    wins = {system: 0 for system in systems}
    by_system = {
        system: {record.benchmark_id: record for record in results}
        for system, (_manifest, results) in runs.items()
    }
    ties = 0
    for benchmark in pack.benchmarks:
        contenders: list[tuple[str, float]] = []
        for system in systems:
            records = by_system.get(system, {})
            record = records.get(benchmark.benchmark_id)
            if record is None or record.status != "ok" or record.metric_value is None:
                continue
            contenders.append((system, float(record.metric_value)))
        if not contenders:
            ties += 1
            continue
        values = [value for _system, value in contenders]
        best = max(values) if benchmark.metric_direction == "max" else min(values)
        winners = [system for system, value in contenders if abs(value - best) <= 1e-12]
        if len(winners) == 1:
            wins[winners[0]] += 1
        else:
            ties += 1
    return wins, ties


def _reference_note(pair_results: dict[tuple[str, str], tuple[ComparisonResult, Path]]) -> str | None:
    bad_pairs = [
        f"{left}/{right}: {result.parity_status} ({'; '.join(result.reasons) if result.reasons else 'no reason'})"
        for (left, right), (result, _path) in sorted(pair_results.items())
        if result.parity_status != "fair"
    ]
    if not bad_pairs:
        return None
    return " | ".join(bad_pairs)


def _data_signature(manifest: RunManifest) -> str | None:
    if manifest.fairness is not None:
        return manifest.fairness.data_signature
    return None
