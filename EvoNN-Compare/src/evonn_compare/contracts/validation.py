"""Validation helpers for manifests, result records, and parity packs."""

from __future__ import annotations

from pathlib import Path

from pydantic import BaseModel, ConfigDict

from evonn_compare.contracts.parity import ParityPack
from evonn_shared.contracts import ResultRecord, RunManifest


class ValidationIssue(BaseModel):
    """Single validation finding."""

    model_config = ConfigDict(frozen=True)

    level: str
    code: str
    message: str


class ValidationReport(BaseModel):
    """Structured validation result."""

    model_config = ConfigDict(frozen=True)

    issues: list[ValidationIssue]

    @property
    def ok(self) -> bool:
        return not any(issue.level == "error" for issue in self.issues)


def validate_contract(
    manifest: RunManifest,
    results: list[ResultRecord],
    pack: ParityPack,
    run_dir: Path | None = None,
) -> ValidationReport:
    """Validate an exported run against a parity pack."""

    issues: list[ValidationIssue] = []

    if manifest.pack_name != pack.name:
        issues.append(
            ValidationIssue(
                level="error",
                code="pack_name_mismatch",
                message=f"manifest pack_name '{manifest.pack_name}' does not match '{pack.name}'",
            )
        )

    if manifest.fairness is not None:
        if manifest.fairness.benchmark_pack_id != pack.name:
            issues.append(
                ValidationIssue(
                    level="error",
                    code="fairness_pack_mismatch",
                    message=(
                        "fairness benchmark_pack_id mismatch: "
                        f"{manifest.fairness.benchmark_pack_id} vs {pack.name}"
                    ),
                )
            )
        if manifest.fairness.seed != manifest.seed:
            issues.append(
                ValidationIssue(
                    level="error",
                    code="fairness_seed_mismatch",
                    message=f"fairness seed {manifest.fairness.seed} does not match manifest seed {manifest.seed}",
                )
            )
        if manifest.fairness.evaluation_count != manifest.budget.evaluation_count:
            issues.append(
                ValidationIssue(
                    level="error",
                    code="fairness_eval_mismatch",
                    message=(
                        "fairness evaluation_count does not match manifest budget: "
                        f"{manifest.fairness.evaluation_count} vs {manifest.budget.evaluation_count}"
                    ),
                )
            )
        manifest_policy = _normalize_budget_policy(manifest.budget.budget_policy_name)
        fairness_policy = _normalize_budget_policy(manifest.fairness.budget_policy_name)
        if manifest_policy != fairness_policy:
            issues.append(
                ValidationIssue(
                    level="error",
                    code="fairness_policy_mismatch",
                    message=(
                        "fairness budget_policy_name does not match manifest budget policy: "
                        f"{manifest.fairness.budget_policy_name} vs {manifest.budget.budget_policy_name}"
                    ),
                )
            )
        if manifest.fairness.data_signature is None:
            issues.append(
                ValidationIssue(
                    level="warning",
                    code="fairness_data_signature_missing",
                    message="fairness data_signature is missing",
                )
            )
        if manifest.fairness.code_version is None:
            issues.append(
                ValidationIssue(
                    level="warning",
                    code="fairness_code_version_missing",
                    message="fairness code_version is missing",
                )
            )

    pack_benchmarks = {entry.benchmark_id: entry for entry in pack.benchmarks}
    manifest_benchmarks = {entry.benchmark_id: entry for entry in manifest.benchmarks}

    missing_manifest = sorted(set(pack_benchmarks) - set(manifest_benchmarks))
    for benchmark_id in missing_manifest:
        issues.append(
            ValidationIssue(
                level="error",
                code="missing_manifest_benchmark",
                message=f"manifest missing benchmark '{benchmark_id}'",
            )
        )

    extra_manifest = sorted(set(manifest_benchmarks) - set(pack_benchmarks))
    for benchmark_id in extra_manifest:
        issues.append(
            ValidationIssue(
                level="warning",
                code="extra_manifest_benchmark",
                message=f"manifest includes benchmark '{benchmark_id}' not present in pack",
            )
        )

    for benchmark_id, manifest_entry in manifest_benchmarks.items():
        pack_entry = pack_benchmarks.get(benchmark_id)
        if pack_entry is None:
            continue
        if manifest_entry.metric_name != pack_entry.metric_name:
            issues.append(
                ValidationIssue(
                    level="error",
                    code="metric_name_mismatch",
                    message=f"benchmark '{benchmark_id}' metric_name mismatch",
                )
            )
        if manifest_entry.metric_direction != pack_entry.metric_direction:
            issues.append(
                ValidationIssue(
                    level="error",
                    code="metric_direction_mismatch",
                    message=f"benchmark '{benchmark_id}' metric_direction mismatch",
                )
            )
        if manifest_entry.task_kind != pack_entry.task_kind:
            issues.append(
                ValidationIssue(
                    level="error",
                    code="task_kind_mismatch",
                    message=f"benchmark '{benchmark_id}' task_kind mismatch",
                )
            )

    tolerance = pack.budget_policy.budget_tolerance_pct / 100.0
    if _relative_diff(
        manifest.budget.evaluation_count,
        pack.budget_policy.evaluation_count,
    ) > tolerance:
        issues.append(
            ValidationIssue(
                level="error",
                code="budget_eval_mismatch",
                message=(
                    "evaluation_count exceeds tolerance: "
                    f"{manifest.budget.evaluation_count} vs {pack.budget_policy.evaluation_count}"
                ),
            )
        )

    if _relative_diff(
        manifest.budget.epochs_per_candidate,
        pack.budget_policy.epochs_per_candidate,
    ) > tolerance:
        issues.append(
            ValidationIssue(
                level="error",
                code="budget_epochs_mismatch",
                message=(
                    "epochs_per_candidate exceeds tolerance: "
                    f"{manifest.budget.epochs_per_candidate} vs "
                    f"{pack.budget_policy.epochs_per_candidate}"
                ),
            )
        )

    if manifest.budget.actual_evaluations is None:
        issues.append(
            ValidationIssue(
                level="warning",
                code="budget_actual_evaluations_missing",
                message="budget actual_evaluations is missing",
            )
        )
    elif (
        not manifest.budget.partial_run
        and (covered_evaluations := manifest.budget.covered_evaluations()) is not None
        and covered_evaluations < manifest.budget.evaluation_count
    ):
        issues.append(
            ValidationIssue(
                level="warning",
                code="budget_actual_lt_declared",
                message=(
                    "budget accounted_evaluations plus explicit reduced-calculation coverage is lower than declared "
                    "evaluation_count without partial_run=true: "
                    f"{covered_evaluations} vs {manifest.budget.evaluation_count}"
                ),
            )
        )

    if not manifest.budget.evaluation_semantics:
        issues.append(
            ValidationIssue(
                level="warning",
                code="budget_semantics_missing",
                message="budget evaluation_semantics is missing",
            )
        )

    expected_keys = {
        (entry.benchmark_id, entry.metric_name)
        for entry in pack.benchmarks
    }
    seen_keys: set[tuple[str, str]] = set()
    for record in results:
        key = (record.benchmark_id, record.metric_name)
        if key in seen_keys:
            issues.append(
                ValidationIssue(
                    level="error",
                    code="duplicate_result",
                    message=(
                        f"duplicate result record for benchmark '{record.benchmark_id}' "
                        f"and metric '{record.metric_name}'"
                    ),
                )
            )
        seen_keys.add(key)
        if record.status == "failed" and not record.failure_reason:
            issues.append(
                ValidationIssue(
                    level="error",
                    code="failed_without_reason",
                    message=f"failed result for '{record.benchmark_id}' is missing failure_reason",
                )
            )

    missing_results = sorted(expected_keys - seen_keys)
    for benchmark_id, metric_name in missing_results:
        issues.append(
            ValidationIssue(
                level="error",
                code="missing_result",
                message=f"missing result for benchmark '{benchmark_id}' and metric '{metric_name}'",
            )
        )

    if run_dir is not None:
        for attr_name in ("config_snapshot", "report_markdown"):
            rel_path = getattr(manifest.artifacts, attr_name)
            if not _artifact_exists(run_dir, rel_path):
                issues.append(
                    ValidationIssue(
                        level="error",
                        code="missing_artifact",
                        message=f"artifact '{attr_name}' does not exist: {rel_path}",
                    )
                )

    return ValidationReport(issues=issues)


def _relative_diff(actual: int, expected: int) -> float:
    if expected == 0:
        return 0.0 if actual == 0 else float("inf")
    return abs(actual - expected) / expected


def _artifact_exists(run_dir: Path, artifact_path: str) -> bool:
    path = Path(artifact_path)
    if not path.is_absolute():
        path = run_dir / path
    return path.exists()


def _normalize_budget_policy(name: str | None) -> str | None:
    if not name:
        return None
    if name == "budget_matched_contender_pool":
        return "prototype_equal_budget"
    if name == "fixed_contender_pool":
        return "fixed_reference_contender_pool"
    return name
