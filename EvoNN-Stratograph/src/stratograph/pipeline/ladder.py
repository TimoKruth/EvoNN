"""Execution-ladder generation and runner."""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path

import yaml

from stratograph.benchmarks.parity import fallback_native_id, load_parity_pack
from stratograph.config import load_config
from stratograph.export import export_symbiosis_contract
from stratograph.pipeline.coordinator import run_evolution

_SHARED_FULL_PACK = "shared_33plus5"
_FULL_LADDER_BUDGETS = [38, 76, 152, 304, 608]


@dataclass(frozen=True)
class LadderCase:
    name: str
    config_path: Path
    pack_path: Path
    run_dir: Path


def build_execution_ladder(workspace: str | Path) -> list[LadderCase]:
    workspace = Path(workspace)
    packs_dir = workspace / "packs"
    configs_dir = workspace / "configs"
    runs_dir = workspace / "runs"
    packs_dir.mkdir(parents=True, exist_ok=True)
    configs_dir.mkdir(parents=True, exist_ok=True)
    runs_dir.mkdir(parents=True, exist_ok=True)

    cases: list[LadderCase] = []
    cases.extend(
        [
            _single_case(
                packs_dir=packs_dir,
                configs_dir=configs_dir,
                runs_dir=runs_dir,
                case_name="single_moons_classification",
                benchmark_names=["moons"],
            ),
            _single_case(
                packs_dir=packs_dir,
                configs_dir=configs_dir,
                runs_dir=runs_dir,
                case_name="single_digits_image",
                benchmark_names=["digits"],
            ),
            _single_case(
                packs_dir=packs_dir,
                configs_dir=configs_dir,
                runs_dir=runs_dir,
                case_name="single_tiny_lm_synthetic",
                benchmark_names=["tiny_lm_synthetic"],
            ),
            _single_case(
                packs_dir=packs_dir,
                configs_dir=configs_dir,
                runs_dir=runs_dir,
                case_name="lm_only_5pack",
                benchmark_names=[
                    "tiny_lm_synthetic",
                    "tinystories_lm",
                    "tinystories_lm_smoke",
                    "wikitext2_lm",
                    "wikitext2_lm_smoke",
                ],
            ),
        ]
    )
    for budget in _FULL_LADDER_BUDGETS:
        cases.append(
            _full_pack_case(
                packs_dir=packs_dir,
                configs_dir=configs_dir,
                runs_dir=runs_dir,
                source_pack_name=_SHARED_FULL_PACK,
                budget=budget,
            )
        )
    return cases


def run_execution_ladder(workspace: str | Path) -> list[Path]:
    cases = build_execution_ladder(workspace)
    manifests: list[Path] = []
    for case in cases:
        config = load_config(case.config_path)
        run_evolution(config, run_dir=case.run_dir, config_path=case.config_path)
        manifest_path, _ = export_symbiosis_contract(case.run_dir, case.pack_path)
        manifests.append(manifest_path)
    return manifests


def _single_case(
    *,
    packs_dir: Path,
    configs_dir: Path,
    runs_dir: Path,
    case_name: str,
    benchmark_names: list[str],
) -> LadderCase:
    pack_path = packs_dir / f"{case_name}.yaml"
    config_path = configs_dir / f"{case_name}.yaml"
    run_dir = runs_dir / case_name
    _write_pack(pack_path, case_name, benchmark_names, evaluation_count=len(benchmark_names))
    _write_config(config_path, case_name, benchmark_names, evaluation_count=len(benchmark_names))
    return LadderCase(name=case_name, config_path=config_path, pack_path=pack_path, run_dir=run_dir)


def _full_pack_case(
    *,
    packs_dir: Path,
    configs_dir: Path,
    runs_dir: Path,
    source_pack_name: str,
    budget: int,
) -> LadderCase:
    pack = load_parity_pack(source_pack_name)
    benchmark_names = [fallback_native_id(entry) for entry in pack.benchmarks]
    case_name = f"{source_pack_name}_eval{budget}"
    pack_path = packs_dir / f"{case_name}.yaml"
    config_path = configs_dir / f"{case_name}_seed42.yaml"
    run_dir = runs_dir / f"{case_name}_seed42"
    _write_pack(pack_path, case_name, benchmark_names, evaluation_count=budget)
    _write_config(config_path, case_name, benchmark_names, evaluation_count=budget)
    return LadderCase(name=case_name, config_path=config_path, pack_path=pack_path, run_dir=run_dir)


def _write_pack(path: Path, name: str, benchmark_names: list[str], evaluation_count: int) -> None:
    from stratograph.benchmarks import get_benchmark
    from stratograph.benchmarks.parity import get_canonical_id

    payload = {
        "name": name,
        "tier": 3,
        "description": f"Stratograph execution ladder case {name}.",
        "benchmarks": [
            {
                "benchmark_id": get_canonical_id(benchmark_name),
                "native_ids": {"stratograph": benchmark_name},
                "task_kind": get_benchmark(benchmark_name).task,
                "metric_name": get_benchmark(benchmark_name).metric_name,
                "metric_direction": get_benchmark(benchmark_name).metric_direction,
            }
            for benchmark_name in benchmark_names
        ],
        "budget_policy": {
            "evaluation_count": evaluation_count,
            "epochs_per_candidate": 1,
            "budget_tolerance_pct": 10.0,
        },
        "seed_policy": {"mode": "shared", "required": True},
    }
    path.write_text(yaml.safe_dump(payload, sort_keys=False), encoding="utf-8")


def _write_config(path: Path, name: str, benchmark_names: list[str], evaluation_count: int) -> None:
    benchmark_count = len(benchmark_names)
    units = max(1, evaluation_count // max(1, benchmark_count))
    if units >= 8:
        population_size, generations = 4, max(1, units // 4)
    elif units >= 4:
        population_size, generations = 4, 1
    elif units >= 2:
        population_size, generations = 2, 1
    else:
        population_size, generations = 1, 1
    payload = {
        "seed": 42,
        "run_name": name,
        "benchmark_pool": {"name": name, "benchmarks": benchmark_names},
        "training": {
            "epochs": 1,
            "batch_size": 32,
            "learning_rate": 0.001,
            "multi_fidelity": False,
            "weight_inheritance": True,
        },
        "evolution": {
            "population_size": population_size,
            "generations": generations,
            "elite_per_benchmark": 1,
        },
    }
    path.write_text(yaml.safe_dump(payload, sort_keys=False), encoding="utf-8")
