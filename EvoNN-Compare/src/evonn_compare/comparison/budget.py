"""Budget comparison helpers for compare run pairs."""

from __future__ import annotations

from pydantic import BaseModel, ConfigDict

from evonn_compare.contracts.models import RunManifest
from evonn_compare.contracts.parity import ParityPack


class BudgetComparison(BaseModel):
    """Budget comparison result for two manifests."""

    model_config = ConfigDict(frozen=True)

    status: str
    reasons: list[str]


class BudgetComparator:
    """Classify a run pair as fair, asymmetric, or incomparable."""

    def compare(
        self,
        left: RunManifest,
        right: RunManifest,
        pack: ParityPack,
    ) -> BudgetComparison:
        tolerance = pack.budget_policy.budget_tolerance_pct / 100.0
        reasons: list[str] = []

        coverage_left = {entry.benchmark_id for entry in left.benchmarks}
        coverage_right = {entry.benchmark_id for entry in right.benchmarks}
        expected = {entry.benchmark_id for entry in pack.benchmarks}
        if coverage_left != expected or coverage_right != expected:
            reasons.append("benchmark coverage does not match the parity pack")
            return BudgetComparison(status="incomparable", reasons=reasons)

        left_pack_id = left.fairness.benchmark_pack_id if left.fairness is not None else left.pack_name
        right_pack_id = right.fairness.benchmark_pack_id if right.fairness is not None else right.pack_name
        if left_pack_id != pack.name or right_pack_id != pack.name or left_pack_id != right_pack_id:
            reasons.append("benchmark pack ID mismatch")
            return BudgetComparison(status="incomparable", reasons=reasons)

        left_data_signature = left.fairness.data_signature if left.fairness is not None else None
        right_data_signature = right.fairness.data_signature if right.fairness is not None else None
        if left_data_signature != right_data_signature:
            reasons.append("data signature mismatch")
            return BudgetComparison(status="incomparable", reasons=reasons)

        eval_diff = _relative_diff(
            left.budget.evaluation_count,
            right.budget.evaluation_count,
        )
        epoch_diff = _relative_diff(
            left.budget.epochs_per_candidate,
            right.budget.epochs_per_candidate,
        )

        if eval_diff > 0.25 or epoch_diff > 0.25:
            reasons.append("budget mismatch exceeds 25%")
            return BudgetComparison(status="incomparable", reasons=reasons)

        if pack.seed_policy.required and left.seed != right.seed:
            reasons.append("seed mismatch")

        if eval_diff > tolerance:
            reasons.append("evaluation_count mismatch exceeds pack tolerance")
        if epoch_diff > tolerance:
            reasons.append("epochs_per_candidate mismatch exceeds pack tolerance")
        left_policy = _normalize_budget_policy(
            left.fairness.budget_policy_name if left.fairness is not None else left.budget.budget_policy_name
        )
        right_policy = _normalize_budget_policy(
            right.fairness.budget_policy_name if right.fairness is not None else right.budget.budget_policy_name
        )
        if left_policy != right_policy:
            if {left_policy, right_policy} <= {None, "prototype_equal_budget"}:
                pass
            elif left_policy and right_policy:
                reasons.append(f"budget policy mismatch: {left_policy} vs {right_policy}")
            else:
                reasons.append("budget policy missing on one side")

        status = "fair" if not reasons else "asymmetric"
        return BudgetComparison(status=status, reasons=reasons)


def _relative_diff(left: int, right: int) -> float:
    if right == 0:
        return 0.0 if left == 0 else float("inf")
    return abs(left - right) / right


def _normalize_budget_policy(name: str | None) -> str | None:
    if not name:
        return None
    if name == "budget_matched_contender_pool":
        return "prototype_equal_budget"
    if name == "fixed_contender_pool":
        return "fixed_reference_contender_pool"
    return name
