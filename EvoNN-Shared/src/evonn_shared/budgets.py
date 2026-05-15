"""Normalized budget metadata models for compare-grade EvoNN runs."""

from __future__ import annotations

from pydantic import BaseModel, ConfigDict


class BudgetContract(BaseModel):
    """Minimal normalized budget vocabulary derived from BUDGET_CONTRACT.md.

    This model intentionally stays close to the contract vocabulary instead of
    enforcing narrower runtime semantics. Producers may know only part of the
    budget envelope, while comparison code can still consume the normalized
    fields that are present.
    """

    model_config = ConfigDict(frozen=True)

    # Benchmark-surface budget.
    pack_id: str
    benchmark_tier: str | None = None

    # Evaluation and wall-clock accounting.
    evaluation_budget: int
    actual_evaluations: int | None = None
    wall_clock_budget_seconds: float | None = None
    actual_wall_clock_seconds: float | None = None

    # Per-candidate optimization budget.
    optimization_unit: str | None = None
    optimization_budget: int | float | None = None

    # Hardware and artifact-size constraints.
    hardware_class: str | None = None
    worker_count: int | None = None
    memory_ceiling_mb: float | None = None
    parameter_cap: int | None = None
    latency_target_ms: float | None = None

    # Transfer/seeding provenance.
    seeding_ladder: str | None = None
    seed_source_system: str | None = None
    seed_source_run_id: str | None = None
