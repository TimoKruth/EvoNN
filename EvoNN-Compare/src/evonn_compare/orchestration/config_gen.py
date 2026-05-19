"""Generate compare-project campaign configs for Prism and Topograph."""

from __future__ import annotations

from copy import deepcopy
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Any

import yaml

from evonn_compare.contracts.parity import load_parity_pack
from evonn_compare.orchestration.benchmark_resolution import resolve_supported_benchmark_ids
from evonn_compare.orchestration.campaign_state import sync_workspace_state


COMPARE_ROOT = Path(__file__).resolve().parents[3]
WORKSPACE_ROOT = COMPARE_ROOT.parent
DEFAULT_PRISM_ROOT = WORKSPACE_ROOT / "EvoNN-Prism"
DEFAULT_TOPOGRAPH_ROOT = WORKSPACE_ROOT / "EvoNN-Topograph"


@dataclass(frozen=True)
class CampaignPaths:
    workspace: Path
    packs_dir: Path
    prism_configs_dir: Path
    topograph_configs_dir: Path
    reports_dir: Path
    logs_dir: Path
    manifest_path: Path


@dataclass(frozen=True)
class CampaignCase:
    pack_name: str
    lane_preset: str | None
    seed: int
    budget: int
    pack_path: Path
    prism_config_path: Path
    topograph_config_path: Path
    topograph_run_dir: Path
    comparison_output_path: Path


@dataclass(frozen=True)
class PrismFamilyPlan:
    families: list[str]
    min_population_size: int
    preferred_population_cap: int


def generate_budget_pack(
    *,
    base_pack_path: Path,
    budget: int,
    output_dir: Path,
    epochs_per_candidate: int | None = None,
    base_payload: dict[str, Any] | None = None,
) -> Path:
    payload = deepcopy(base_payload) if base_payload is not None else yaml.safe_load(base_pack_path.read_text(encoding="utf-8"))
    if payload.get("include_packs"):
        expanded_pack = load_parity_pack(base_pack_path)
        payload = {
            **payload,
            "benchmarks": [benchmark.model_dump(mode="json") for benchmark in expanded_pack.benchmarks],
        }
    payload["name"] = f"{payload['name']}_eval{budget}"
    payload["budget_policy"]["evaluation_count"] = budget
    if epochs_per_candidate is not None:
        payload["budget_policy"]["epochs_per_candidate"] = epochs_per_candidate
    payload["seed_policy"]["mode"] = "campaign"
    output_dir.mkdir(parents=True, exist_ok=True)
    output_path = output_dir / f"{payload['name']}.yaml"
    output_path.write_text(yaml.safe_dump(payload, sort_keys=False), encoding="utf-8")
    return output_path


def generate_prism_config(
    *,
    output_path: Path,
    pack_path: Path,
    seed: int,
    budget: int,
) -> Path:
    pack = load_parity_pack(pack_path)
    benchmark_ids = resolve_supported_benchmark_ids(pack.benchmarks, "prism")
    family_plan = _prism_family_plan(pack, budget=budget)
    patch = _prism_budget_patch(
        budget=budget,
        benchmark_count=len(pack.benchmarks),
        required_family_count=family_plan.min_population_size,
        preferred_population_cap=family_plan.preferred_population_cap,
    )
    payload = {
        "seed": seed,
        "benchmark_pack": {
            "pack_name": str(pack_path),
            "benchmark_ids": benchmark_ids,
        },
        "training": {
            "epochs": pack.budget_policy.epochs_per_candidate,
            "batch_size": 32,
            "learning_rate": 0.001,
            "multi_fidelity": False,
            "weight_inheritance": True,
        },
        "evolution": {
            "elite_per_benchmark": max(1, patch["evolution"]["population_size"] // 4),
            "allowed_families": family_plan.families,
        },
    }
    _deep_update(payload, patch)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    output_path.write_text(yaml.safe_dump(payload, sort_keys=False), encoding="utf-8")
    return output_path


def generate_topograph_config(
    *,
    output_path: Path,
    pack_path: Path,
    seed: int,
    budget: int,
    run_dir: Path,
    primordia_seed_candidates_path: Path | None = None,
) -> Path:
    pack = load_parity_pack(pack_path)
    benchmark_ids = resolve_supported_benchmark_ids(pack.benchmarks, "topograph")
    patch = _topograph_budget_patch(budget=budget, benchmark_count=len(pack.benchmarks))
    parallel_workers = _topograph_parallel_workers(pack)
    payload = {
        "seed": seed,
        "benchmark": benchmark_ids[0],
        "benchmark_pool": {
            "benchmarks": benchmark_ids,
            "sample_k": len(benchmark_ids),
            "training_epochs_override": pack.budget_policy.epochs_per_candidate,
        },
        "training": {
            "epochs": pack.budget_policy.epochs_per_candidate,
            "learning_rate": 0.001,
            "batch_size": 32,
            "multi_fidelity": False,
            "weight_inheritance": True,
            "parallel_workers": parallel_workers,
        },
        "speciation": {"enabled": True, "threshold": 3.0},
        "run_dir": str(run_dir),
    }
    if primordia_seed_candidates_path is not None:
        payload["benchmark_pool"]["primordia_seed_candidates_path"] = str(primordia_seed_candidates_path)
    _deep_update(payload, patch)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    output_path.write_text(yaml.safe_dump(payload, sort_keys=False), encoding="utf-8")
    return output_path


def _topograph_parallel_workers(pack: Any) -> int:
    """Avoid process-pool stalls on heavier expanded-ladder benchmarks."""

    expanded_blocker_groups = {"language_modeling"}
    expanded_blocker_ids = {
        "fashionmnist_image",
        "mnist_image",
    }
    for benchmark in pack.benchmarks:
        benchmark_id = str(getattr(benchmark, "benchmark_id", ""))
        group = str(getattr(benchmark, "benchmark_group", "") or "")
        if benchmark_id.startswith("openml_") or benchmark_id in expanded_blocker_ids or group in expanded_blocker_groups:
            return 1
    return 2


def prepare_campaign_cases(
    *,
    pack_name: str,
    base_pack_path: Path,
    seeds: list[int],
    budgets: list[int],
    workspace: Path,
    topograph_root: Path,
    lane_preset: str | None = None,
) -> tuple[CampaignPaths, list[CampaignCase]]:
    workspace = workspace.resolve()
    paths = CampaignPaths(
        workspace=workspace,
        packs_dir=workspace / "packs",
        prism_configs_dir=workspace / "configs" / "prism",
        topograph_configs_dir=workspace / "configs" / "topograph",
        reports_dir=workspace / "reports",
        logs_dir=workspace / "logs",
        manifest_path=workspace / "campaign.yaml",
    )
    for directory in (
        paths.workspace,
        paths.packs_dir,
        paths.prism_configs_dir,
        paths.topograph_configs_dir,
        paths.reports_dir,
        paths.logs_dir,
    ):
        directory.mkdir(parents=True, exist_ok=True)

    base_payload = yaml.safe_load(base_pack_path.read_text(encoding="utf-8"))
    cases: list[CampaignCase] = []
    for budget in budgets:
        pack_path = generate_budget_pack(
            base_pack_path=base_pack_path,
            budget=budget,
            output_dir=paths.packs_dir,
            base_payload=base_payload,
        )
        pack = load_parity_pack(pack_path)
        for seed in seeds:
            pair_name = f"{pack.name}_seed{seed}"
            prism_config_path = paths.prism_configs_dir / f"{pair_name}.yaml"
            topograph_config_path = paths.topograph_configs_dir / f"{pair_name}.yaml"
            topograph_run_dir = topograph_root / "runs" / pair_name
            generate_prism_config(
                output_path=prism_config_path,
                pack_path=pack_path,
                seed=seed,
                budget=budget,
            )
            generate_topograph_config(
                output_path=topograph_config_path,
                pack_path=pack_path,
                seed=seed,
                budget=budget,
                run_dir=topograph_run_dir,
            )
            cases.append(
                CampaignCase(
                    pack_name=pack.name,
                    lane_preset=lane_preset,
                    seed=seed,
                    budget=budget,
                    pack_path=pack_path,
                    prism_config_path=prism_config_path,
                    topograph_config_path=topograph_config_path,
                    topograph_run_dir=topograph_run_dir,
                    comparison_output_path=paths.reports_dir / f"{pair_name}.md",
                )
            )

    payload = {
        "pack_name": pack_name,
        "lane_preset": lane_preset,
        "seeds": seeds,
        "budgets": budgets,
        "cases": [
            {
                key: str(value) if isinstance(value, Path) else value
                for key, value in asdict(case).items()
            }
            for case in cases
        ],
    }
    paths.manifest_path.write_text(yaml.safe_dump(payload, sort_keys=False), encoding="utf-8")
    sync_workspace_state(
        workspace_kind="campaign",
        workspace=paths.workspace,
        manifest_path=paths.manifest_path,
        cases=cases,
        lane_preset=lane_preset,
    )
    return paths, cases


def _prism_family_plan(pack, *, budget: int) -> PrismFamilyPlan:
    benchmark_count = len(pack.benchmarks)
    if benchmark_count == 0:
        return PrismFamilyPlan(["mlp"], min_population_size=1, preferred_population_cap=8)
    units = budget // benchmark_count
    has_lm = any(entry.task_kind == "language_modeling" for entry in pack.benchmarks)
    has_non_lm = any(entry.task_kind != "language_modeling" for entry in pack.benchmarks)
    has_image = any(_is_image_benchmark(entry) for entry in pack.benchmarks)
    has_tabular_or_synthetic = any(
        entry.task_kind != "language_modeling" and not _is_image_benchmark(entry)
        for entry in pack.benchmarks
    )
    if has_lm and has_non_lm:
        for families in (
            ["mlp", "sparse_mlp", "attention", "sparse_attention"],
            ["attention", "sparse_attention"],
            ["attention"],
        ):
            if _has_exact_factorization(units, min_population_size=len(families), preferred_population_cap=8):
                return PrismFamilyPlan(families, min_population_size=len(families), preferred_population_cap=8)
        return PrismFamilyPlan(["attention"], min_population_size=1, preferred_population_cap=8)
    if has_lm:
        if _has_exact_factorization(units, min_population_size=2, preferred_population_cap=8):
            return PrismFamilyPlan(["attention", "sparse_attention"], min_population_size=2, preferred_population_cap=8)
        return PrismFamilyPlan(["attention"], min_population_size=1, preferred_population_cap=8)

    if has_image and has_tabular_or_synthetic:
        for families in (
            ["mlp", "sparse_mlp", "moe_mlp", "conv2d", "lite_conv2d"],
            ["mlp", "sparse_mlp", "conv2d", "lite_conv2d"],
        ):
            if _has_exact_factorization(units, min_population_size=2 * len(families), preferred_population_cap=16):
                return PrismFamilyPlan(
                    families,
                    min_population_size=2 * len(families),
                    preferred_population_cap=16,
                )
            if _has_exact_factorization(units, min_population_size=len(families), preferred_population_cap=8):
                return PrismFamilyPlan(families, min_population_size=len(families), preferred_population_cap=8)
        for families in (["mlp", "sparse_mlp"], ["mlp"]):
            if _has_exact_factorization(units, min_population_size=len(families), preferred_population_cap=8):
                return PrismFamilyPlan(families, min_population_size=len(families), preferred_population_cap=8)
        return PrismFamilyPlan(["mlp"], min_population_size=1, preferred_population_cap=8)

    if has_image:
        for families in (
            ["mlp", "sparse_mlp", "conv2d", "lite_conv2d"],
            ["conv2d", "lite_conv2d"],
            ["mlp", "sparse_mlp"],
            ["mlp"],
        ):
            if _has_exact_factorization(units, min_population_size=len(families), preferred_population_cap=8):
                return PrismFamilyPlan(families, min_population_size=len(families), preferred_population_cap=8)
        return PrismFamilyPlan(["mlp"], min_population_size=1, preferred_population_cap=8)

    for families in (
        ["mlp", "sparse_mlp", "moe_mlp"],
        ["mlp", "sparse_mlp"],
        ["mlp"],
    ):
        preferred_cap = 12 if len(families) > 2 else 8
        min_population_size = 2 * len(families) if len(families) > 2 else len(families)
        if _has_exact_factorization(units, min_population_size=min_population_size, preferred_population_cap=preferred_cap):
            return PrismFamilyPlan(
                families,
                min_population_size=min_population_size,
                preferred_population_cap=preferred_cap,
            )
    return PrismFamilyPlan(["mlp"], min_population_size=1, preferred_population_cap=8)


def _is_image_benchmark(entry: Any) -> bool:
    benchmark_group = str(getattr(entry, "benchmark_group", "") or "").replace("_", "-")
    benchmark_id = str(getattr(entry, "benchmark_id", "") or "")
    modality = str(getattr(entry, "modality", "") or "")
    return benchmark_group in {"image", "image-classification"} or "image" in benchmark_id or modality == "image"


def _prism_budget_patch(
    *,
    budget: int,
    benchmark_count: int,
    required_family_count: int = 1,
    preferred_population_cap: int = 8,
) -> dict[str, Any]:
    if budget % benchmark_count != 0:
        raise ValueError(f"budget {budget} not divisible by benchmark_count {benchmark_count}")
    units = budget // benchmark_count
    if units < required_family_count:
        raise ValueError(
            f"budget {budget} too small for prism pack coverage: need at least "
            f"{benchmark_count * required_family_count} evaluations"
        )
    population_size, generations = _exact_factorization(
        units,
        preferred_population_cap=preferred_population_cap,
        min_population_size=required_family_count,
    )
    return {
        "evolution": {
            "population_size": population_size,
            "offspring_per_generation": population_size,
            "num_generations": generations,
        }
    }


def _topograph_budget_patch(*, budget: int, benchmark_count: int) -> dict[str, Any]:
    if budget % benchmark_count != 0:
        raise ValueError(f"budget {budget} not divisible by benchmark_count {benchmark_count}")
    units = budget // benchmark_count
    preferred_cap = 4 if units <= 16 else 8
    population_size, generations = _exact_factorization(units, preferred_population_cap=preferred_cap)
    return {
        "evolution": {
            "population_size": population_size,
            "num_generations": generations,
            "elite_count": max(1, population_size // 4),
        }
    }


def _has_exact_factorization(
    units: int,
    *,
    min_population_size: int,
    preferred_population_cap: int,
) -> bool:
    try:
        _exact_factorization(
            units,
            preferred_population_cap=preferred_population_cap,
            min_population_size=min_population_size,
        )
    except ValueError:
        return False
    return True


def _exact_factorization(
    units: int,
    *,
    preferred_population_cap: int,
    min_population_size: int = 1,
) -> tuple[int, int]:
    if units <= 0:
        raise ValueError(f"units must be positive, got {units}")
    for population_size in range(min(preferred_population_cap, units), min_population_size - 1, -1):
        if units % population_size == 0:
            return population_size, units // population_size
    raise ValueError(
        f"units {units} cannot be exactly factored with population >= {min_population_size} "
        f"and cap {preferred_population_cap}"
    )


def _deep_update(target: dict[str, Any], patch: dict[str, Any]) -> None:
    for key, value in patch.items():
        if isinstance(value, dict) and isinstance(target.get(key), dict):
            _deep_update(target[key], value)
        else:
            target[key] = value
