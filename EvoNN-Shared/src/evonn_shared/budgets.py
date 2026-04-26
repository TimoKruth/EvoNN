"""Normalized budget metadata models for compare-grade EvoNN runs."""

from __future__ import annotations

from pydantic import BaseModel, ConfigDict


class BudgetContract(BaseModel):
    """Minimal normalized budget vocabulary derived from BUDGET_CONTRACT.md."""

    model_config = ConfigDict(frozen=True)

    pack_id: str
    benchmark_tier: str | None = None
    evaluation_budget: int
    actual_evaluations: int | None = None
    wall_clock_budget_seconds: float | None = None
    actual_wall_clock_seconds: float | None = None
    optimization_unit: str | None = None
    optimization_budget: int | float | None = None
    hardware_class: str | None = None
    worker_count: int | None = None
    memory_ceiling_mb: float | None = None
    parameter_cap: int | None = None
    latency_target_ms: float | None = None
    seeding_ladder: str | None = None
    seed_source_system: str | None = None
    seed_source_run_id: str | None = None
