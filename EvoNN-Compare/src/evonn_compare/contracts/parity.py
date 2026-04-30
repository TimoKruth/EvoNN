"""Parity pack models and YAML loading helpers."""

from __future__ import annotations

import os
from pathlib import Path
from typing import Literal

import yaml
from pydantic import BaseModel, ConfigDict


class BudgetPolicy(BaseModel):
    """Budget policy shared by runs in a parity pack."""

    model_config = ConfigDict(frozen=True)

    evaluation_count: int
    epochs_per_candidate: int
    budget_tolerance_pct: float = 10.0


class ExplorationPolicy(BaseModel):
    """Controls whether efficiency features are disabled for deeper exploration.

    When ``enabled`` is True, the campaign config generator will override
    multi-fidelity scheduling, promotion screening, and Lamarckian epoch
    scaling so every candidate receives equal training budget.  This is
    useful at high evaluation counts (256+) to test whether slow-starting
    topologies bloom given more compute.
    """

    model_config = ConfigDict(frozen=True)

    enabled: bool = False
    disable_multi_fidelity: bool = True
    disable_promotion_screen: bool = True
    disable_epoch_scaling: bool = True
    description: str = "Full-exploration mode: all candidates get equal training budget"


class SeedPolicy(BaseModel):
    """Seed policy for parity or campaign execution."""

    model_config = ConfigDict(frozen=True)

    mode: Literal["shared", "campaign"]
    required: bool = True


class ParityBenchmark(BaseModel):
    """Single benchmark entry in a parity pack."""

    model_config = ConfigDict(frozen=True)

    benchmark_id: str
    native_ids: dict[str, str] | None = None
    task_kind: Literal["classification", "regression", "language_modeling"]
    benchmark_family: str | None = None
    metric_name: str
    metric_direction: Literal["max", "min"]
    comparison_status: Literal["supported", "asymmetric", "unsupported"] = "supported"
    notes: str = ""


class ParityPack(BaseModel):
    """Validated parity pack definition."""

    model_config = ConfigDict(frozen=True)

    name: str
    tier: Literal[1, 2, 3]
    description: str
    benchmarks: list[ParityBenchmark]
    budget_policy: BudgetPolicy
    seed_policy: SeedPolicy
    exploration_policy: ExplorationPolicy | None = None


COMPARE_ROOT = Path(__file__).resolve().parents[3]
WORKSPACE_ROOT = COMPARE_ROOT.parent
DEFAULT_PACKS_DIR = COMPARE_ROOT / "parity_packs"
_SHARED_ROOT_ENV_VAR = "EVONN_SHARED_BENCHMARKS_DIR"


def _shared_pack_dirs() -> list[Path]:
    shared_root = os.environ.get(_SHARED_ROOT_ENV_VAR)
    root = Path(shared_root).expanduser() if shared_root else WORKSPACE_ROOT / "shared-benchmarks"
    return [
        root / "suites" / "parity",
        root / "suites",
    ]


def _pack_search_dirs(packs_dir: Path | None = None) -> list[Path]:
    if packs_dir is not None:
        return [packs_dir]
    return [DEFAULT_PACKS_DIR, *_shared_pack_dirs()]


def resolve_pack_path(pack_name: str, packs_dir: Path | None = None) -> Path:
    """Resolve a pack name or explicit path to a YAML file."""

    candidate = Path(pack_name)
    if candidate.suffix in {".yaml", ".yml"} and candidate.exists():
        return candidate.resolve()

    for directory in _pack_search_dirs(packs_dir):
        for suffix in (".yaml", ".yml"):
            direct = directory / f"{pack_name}{suffix}"
            if direct.exists():
                return direct.resolve()

    search_dirs = ", ".join(str(path) for path in _pack_search_dirs(packs_dir))
    raise FileNotFoundError(f"Parity pack '{pack_name}' not found in {search_dirs}")


def load_parity_pack(path_or_name: str | Path, packs_dir: Path | None = None) -> ParityPack:
    """Load a parity pack from YAML."""

    path = resolve_pack_path(str(path_or_name), packs_dir=packs_dir)
    data = yaml.safe_load(path.read_text())
    return ParityPack.model_validate(data)


def list_parity_packs(packs_dir: Path | None = None) -> list[Path]:
    """List available parity pack YAML files."""
    resolved: dict[str, Path] = {}
    for directory in _pack_search_dirs(packs_dir):
        if not directory.exists():
            continue
        for path in sorted(directory.iterdir()):
            if path.suffix not in {".yaml", ".yml"}:
                continue
            resolved.setdefault(path.name, path)
    return sorted(resolved.values())


def parity_summary(pack: ParityPack) -> str:
    """Render a compact markdown summary of a parity pack."""

    lines = [
        f"# Parity Pack: {pack.name}",
        "",
        f"**Tier:** {pack.tier}",
        f"**Description:** {pack.description.strip()}",
        "",
        "| Benchmark ID | Task | Family | Metric | Direction | Status |",
        "|---|---|---|---|---|---|",
    ]
    for benchmark in pack.benchmarks:
        lines.append(
            f"| {benchmark.benchmark_id} | {benchmark.task_kind} | {benchmark.benchmark_family or '---'} | "
            f"{benchmark.metric_name} | {benchmark.metric_direction} | {benchmark.comparison_status} |"
        )
    lines.extend(
        [
            "",
            f"**Budget:** evaluation_count={pack.budget_policy.evaluation_count}, "
            f"epochs_per_candidate={pack.budget_policy.epochs_per_candidate}, "
            f"tolerance={pack.budget_policy.budget_tolerance_pct:.1f}%",
            f"**Seed Policy:** mode={pack.seed_policy.mode}, required={pack.seed_policy.required}",
        ]
    )
    if pack.exploration_policy is not None:
        ep = pack.exploration_policy
        lines.append(
            f"**Exploration:** enabled={ep.enabled}, "
            f"disable_multi_fidelity={ep.disable_multi_fidelity}, "
            f"disable_promotion_screen={ep.disable_promotion_screen}, "
            f"disable_epoch_scaling={ep.disable_epoch_scaling}"
        )
    return "\n".join(lines)


def write_single_benchmark_packs(pack: ParityPack, output_dir: Path) -> list[Path]:
    """Materialize one parity-pack YAML per benchmark from a multi-benchmark pack."""

    output_dir.mkdir(parents=True, exist_ok=True)
    written: list[Path] = []
    for benchmark in pack.benchmarks:
        payload = {
            "name": f"{pack.name}__{benchmark.benchmark_id}",
            "tier": pack.tier,
            "description": f"{pack.description.strip()} Single-benchmark slice for {benchmark.benchmark_id}.",
            "benchmarks": [benchmark.model_dump()],
            "budget_policy": pack.budget_policy.model_dump(),
            "seed_policy": pack.seed_policy.model_dump(),
        }
        if pack.exploration_policy is not None:
            payload["exploration_policy"] = pack.exploration_policy.model_dump()
        path = output_dir / f"{benchmark.benchmark_id}.yaml"
        path.write_text(yaml.safe_dump(payload, sort_keys=False, allow_unicode=False), encoding="utf-8")
        written.append(path)
    return written
