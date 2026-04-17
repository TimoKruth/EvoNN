"""Head-to-head comparison engine for exported Prism/Topograph/Hybrid runs."""

from __future__ import annotations

from pydantic import BaseModel, ConfigDict

from evonn_compare.adapters.slots import canonical_slot
from evonn_compare.comparison.budget import BudgetComparator
from evonn_compare.contracts.models import ResultRecord, RunManifest
from evonn_compare.contracts.parity import ParityPack


class ComparisonMatchup(BaseModel):
    """One aligned benchmark comparison."""

    model_config = ConfigDict(frozen=True)

    benchmark_id: str
    metric_name: str
    metric_direction: str
    left_value: float | None
    right_value: float | None
    left_status: str
    right_status: str
    winner: str
    delta: float | None = None
    note: str | None = None


class ComparisonSummary(BaseModel):
    """Aggregate comparison counts."""

    model_config = ConfigDict(frozen=True)

    left_wins: int
    right_wins: int
    evonn_wins: int
    evonn2_wins: int
    ties: int
    unsupported: int
    skipped: int
    failed: int


class ComparisonResult(BaseModel):
    """Top-level comparison payload."""

    model_config = ConfigDict(frozen=True)

    pack_name: str
    parity_status: str
    reasons: list[str]
    left_manifest: RunManifest
    right_manifest: RunManifest
    matchups: list[ComparisonMatchup]
    summary: ComparisonSummary


class ComparisonEngine:
    """Compare two exported run directories under a parity pack."""

    def __init__(self, budget: BudgetComparator | None = None) -> None:
        self.budget = budget or BudgetComparator()

    def compare(
        self,
        *,
        left_manifest: RunManifest,
        left_results: list[ResultRecord],
        right_manifest: RunManifest,
        right_results: list[ResultRecord],
        pack: ParityPack,
    ) -> ComparisonResult:
        budget_result = self.budget.compare(left_manifest, right_manifest, pack)
        left_by_id = {record.benchmark_id: record for record in left_results}
        right_by_id = {record.benchmark_id: record for record in right_results}

        matchups: list[ComparisonMatchup] = []
        left_wins = 0
        right_wins = 0
        ties = 0
        unsupported = 0
        skipped = 0
        failed = 0

        for benchmark in pack.benchmarks:
            left = left_by_id.get(benchmark.benchmark_id)
            right = right_by_id.get(benchmark.benchmark_id)
            matchup = self._compare_one(
                benchmark_id=benchmark.benchmark_id,
                metric_name=benchmark.metric_name,
                metric_direction=benchmark.metric_direction,
                left=left,
                right=right,
            )
            matchups.append(matchup)

            if matchup.winner == "left":
                left_wins += 1
            elif matchup.winner == "right":
                right_wins += 1
            elif matchup.winner == "tie":
                ties += 1
            elif matchup.winner == "unsupported":
                unsupported += 1
            elif matchup.winner == "skipped":
                skipped += 1
            else:
                failed += 1

        left_slot = canonical_slot(left_manifest.system)
        right_slot = canonical_slot(right_manifest.system)
        evonn_wins = left_wins if left_slot == "evonn" else right_wins if right_slot == "evonn" else 0
        evonn2_wins = left_wins if left_slot == "evonn2" else right_wins if right_slot == "evonn2" else 0

        return ComparisonResult(
            pack_name=pack.name,
            parity_status=budget_result.status,
            reasons=budget_result.reasons,
            left_manifest=left_manifest,
            right_manifest=right_manifest,
            matchups=matchups,
            summary=ComparisonSummary(
                left_wins=left_wins,
                right_wins=right_wins,
                evonn_wins=evonn_wins,
                evonn2_wins=evonn2_wins,
                ties=ties,
                unsupported=unsupported,
                skipped=skipped,
                failed=failed,
            ),
        )

    def _compare_one(
        self,
        *,
        benchmark_id: str,
        metric_name: str,
        metric_direction: str,
        left: ResultRecord | None,
        right: ResultRecord | None,
    ) -> ComparisonMatchup:
        if left is None or right is None:
            return ComparisonMatchup(
                benchmark_id=benchmark_id,
                metric_name=metric_name,
                metric_direction=metric_direction,
                left_value=left.metric_value if left else None,
                right_value=right.metric_value if right else None,
                left_status=left.status if left else "missing",
                right_status=right.status if right else "missing",
                winner="failed",
                note="missing result record",
            )

        pair_statuses = {left.status, right.status}
        if pair_statuses & {"unsupported"}:
            return self._status_matchup(benchmark_id, metric_name, metric_direction, left, right, "unsupported")
        if pair_statuses & {"skipped", "missing"}:
            return self._status_matchup(benchmark_id, metric_name, metric_direction, left, right, "skipped")
        if pair_statuses & {"failed"}:
            return self._status_matchup(benchmark_id, metric_name, metric_direction, left, right, "failed")

        if left.metric_value is None or right.metric_value is None:
            return self._status_matchup(
                benchmark_id,
                metric_name,
                metric_direction,
                left,
                right,
                "failed",
                note="metric missing for ok record",
            )

        delta = float(left.metric_value - right.metric_value)
        if abs(delta) <= 1e-12:
            winner = "tie"
        elif metric_direction == "max":
            winner = "left" if delta > 0 else "right"
        else:
            winner = "left" if delta < 0 else "right"

        return ComparisonMatchup(
            benchmark_id=benchmark_id,
            metric_name=metric_name,
            metric_direction=metric_direction,
            left_value=left.metric_value,
            right_value=right.metric_value,
            left_status=left.status,
            right_status=right.status,
            winner=winner,
            delta=delta,
        )

    @staticmethod
    def _status_matchup(
        benchmark_id: str,
        metric_name: str,
        metric_direction: str,
        left: ResultRecord,
        right: ResultRecord,
        winner: str,
        *,
        note: str | None = None,
    ) -> ComparisonMatchup:
        return ComparisonMatchup(
            benchmark_id=benchmark_id,
            metric_name=metric_name,
            metric_direction=metric_direction,
            left_value=left.metric_value,
            right_value=right.metric_value,
            left_status=left.status,
            right_status=right.status,
            winner=winner,
            note=note,
        )
