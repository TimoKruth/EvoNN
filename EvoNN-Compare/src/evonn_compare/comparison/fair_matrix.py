"""Four-way fairness summary helpers."""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path

from evonn_compare.comparison.engine import ComparisonResult
from evonn_compare.contracts.models import ResultRecord, RunManifest
from evonn_compare.contracts.parity import ParityPack


SYSTEM_ORDER = ("prism", "topograph", "stratograph", "contenders")


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
class FairMatrixSummary:
    pack_name: str
    fair_rows: list[MatrixBudgetRow]
    reference_rows: list[MatrixBudgetRow]
    parity_rows: list[PairParityRow]


def summarize_matrix_case(
    *,
    pack: ParityPack,
    budget: int,
    seed: int,
    runs: dict[str, tuple[RunManifest, list[ResultRecord]]],
    pair_results: dict[tuple[str, str], tuple[ComparisonResult, Path]],
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
    wins, ties = _winner_table(pack, runs)
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
    fair_rows: list[MatrixBudgetRow],
    reference_rows: list[MatrixBudgetRow],
    parity_rows: list[PairParityRow],
) -> FairMatrixSummary:
    return FairMatrixSummary(
        pack_name=pack_name,
        fair_rows=sorted(fair_rows, key=lambda row: (row.budget, row.seed)),
        reference_rows=sorted(reference_rows, key=lambda row: (row.budget, row.seed)),
        parity_rows=sorted(parity_rows, key=lambda row: (row.budget, row.seed, row.pair_label)),
    )


def _winner_table(
    pack: ParityPack,
    runs: dict[str, tuple[RunManifest, list[ResultRecord]]],
) -> tuple[dict[str, int], int]:
    wins = {system: 0 for system in SYSTEM_ORDER}
    by_system = {
        system: {record.benchmark_id: record for record in results}
        for system, (_manifest, results) in runs.items()
    }
    ties = 0
    for benchmark in pack.benchmarks:
        contenders: list[tuple[str, float]] = []
        for system in SYSTEM_ORDER:
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
