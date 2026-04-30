"""Shared contender baseline-coverage helpers."""

from __future__ import annotations

import importlib.util
from typing import Any, Sequence

from evonn_contenders.benchmarks import get_benchmark
from evonn_contenders.contenders.registry import benchmark_group, resolve_configured_contenders
from evonn_shared.contracts import BaselineCoverageEnvelope

STEADY_STATE_POLICY_REASON = (
    "trusted recurring lanes ratify the sklearn-backed contender pool as the required floor; "
    "torch and boosted extras widen breadth when installed but are not required for benchmark completeness"
)


def build_baseline_coverage(*, config: Any, benchmark_names: Sequence[str]) -> BaselineCoverageEnvelope:
    """Describe which configured contender baselines were intentionally skipped."""

    active_groups = sorted({benchmark_group(get_benchmark(name)) for name in benchmark_names})
    optional_dependency_skips: dict[str, tuple[str, ...]] = {}
    notes: list[str] = []
    for group in active_groups:
        skipped = [
            contender.name
            for contender in resolve_configured_contenders(config, group)
            if is_optional_skip(contender=contender, config=config)
        ]
        if skipped:
            optional_dependency_skips[group] = tuple(sorted(skipped))
            notes.append(f"{group}: {', '.join(sorted(skipped))}")
    if notes:
        notes.insert(0, "optional dependency backends skipped under required-only completeness policy")
    return BaselineCoverageEnvelope(
        benchmark_complete_policy="required_only_optional_skips_allowed",
        policy_stage="steady_state",
        policy_reason=STEADY_STATE_POLICY_REASON,
        optional_dependency_skips=optional_dependency_skips,
        notes=tuple(notes),
    )


def is_optional_skip(*, contender: Any, config: Any) -> bool:
    dependency = getattr(contender, "optional_dependency", None)
    if dependency is None or importlib.util.find_spec(dependency) is not None:
        return False
    if dependency == "torch":
        return bool(getattr(config.torch, "allow_optional_missing", True))
    return bool(getattr(config.boosted_trees, "allow_optional_missing", True))
