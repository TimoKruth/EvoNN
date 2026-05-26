"""Canonical benchmark mapping and parity-pack loading."""

from __future__ import annotations

from pathlib import Path

from evonn_shared.benchmarks import (
    CANONICAL_BENCHMARK_IDS as CANONICAL_BENCHMARK_IDS,
    canonical_benchmark_id,
    load_parity_pack_payload,
    native_benchmark_id,
    native_id_from_entry,
    parity_pack_search_dirs,
    resolve_parity_pack_path,
)
from pydantic import BaseModel, ConfigDict

from stratograph.benchmarks.datasets import get_benchmark
from stratograph.benchmarks.spec import BenchmarkSpec, MetricDirection, TaskKind


_PACKAGE_DIR = Path(__file__).resolve().parent
_PROJECT_ROOT = _PACKAGE_DIR.parent.parent.parent
_PACK_ENV_VAR = "STRATOGRAPH_PARITY_PACK_DIRS"
_DEFAULT_PACK_SEARCH_DIRS = [
    _PROJECT_ROOT / "parity_packs",
    _PROJECT_ROOT / "parity_packs" / "generated",
]


class BudgetPolicy(BaseModel):
    """Budget block from parity pack."""

    model_config = ConfigDict(frozen=True)

    evaluation_count: int
    epochs_per_candidate: int
    budget_tolerance_pct: float = 10.0


class SeedPolicy(BaseModel):
    """Seed policy from parity pack."""

    model_config = ConfigDict(frozen=True)

    mode: str = "shared"
    required: bool = True


class ParityBenchmark(BaseModel):
    """Single parity benchmark entry."""

    model_config = ConfigDict(frozen=True)

    benchmark_id: str
    native_ids: dict[str, str] | None = None
    task_kind: TaskKind
    metric_name: str
    metric_direction: MetricDirection


class ParityPack(BaseModel):
    """Minimal parity-pack payload."""

    model_config = ConfigDict(frozen=True)

    name: str
    benchmarks: list[ParityBenchmark]
    budget_policy: BudgetPolicy | None = None
    seed_policy: SeedPolicy | None = None


def _pack_search_dirs() -> list[Path]:
    return parity_pack_search_dirs(default_dirs=_DEFAULT_PACK_SEARCH_DIRS, env_var=_PACK_ENV_VAR)


def get_canonical_id(native_name: str) -> str:
    """Map native benchmark id to canonical compare id."""
    return canonical_benchmark_id(native_name)


def get_native_id(canonical_id: str) -> str:
    """Map canonical compare id to Stratograph native id when known."""
    return native_benchmark_id(canonical_id)


def resolve_pack_path(pack_ref: str | Path) -> Path:
    """Resolve pack path or pack name."""
    return resolve_parity_pack_path(
        pack_ref,
        search_dirs=_pack_search_dirs(),
        env_var=_PACK_ENV_VAR,
    )


def fallback_native_id(entry: ParityBenchmark, system: str = "stratograph") -> str:
    """Resolve best native id from mixed legacy/new pack fields."""
    return native_id_from_entry(entry, system=system)


def load_parity_pack(pack_path: str | Path) -> ParityPack:
    """Load simple or rich parity pack YAML."""
    resolved = resolve_pack_path(pack_path)
    payload = load_parity_pack_payload(resolved)
    entries = payload.get("benchmarks", [])

    if entries and isinstance(entries[0], str):
        benchmarks = []
        for native_name in entries:
            spec = get_benchmark(native_name)
            benchmarks.append(
                ParityBenchmark(
                    benchmark_id=get_canonical_id(native_name),
                    native_ids={"stratograph": native_name},
                    task_kind=spec.task,
                    metric_name=spec.metric_name,
                    metric_direction=spec.metric_direction,
                )
            )
        return ParityPack(
            name=payload.get("name", resolved.stem),
            benchmarks=benchmarks,
            budget_policy=BudgetPolicy(
                evaluation_count=len(entries),
                epochs_per_candidate=1,
            ),
            seed_policy=SeedPolicy(),
        )

    return ParityPack.model_validate(payload)


def load_pack_specs(pack_path: str | Path) -> list[BenchmarkSpec]:
    """Load pack into Stratograph benchmark specs."""
    pack = load_parity_pack(pack_path)
    return [get_benchmark(fallback_native_id(entry)) for entry in pack.benchmarks]
