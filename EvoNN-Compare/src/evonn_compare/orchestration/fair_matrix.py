"""Fair four-way campaign generation and execution."""

from __future__ import annotations

from concurrent.futures import ThreadPoolExecutor
from dataclasses import asdict, dataclass
import itertools
import json
from pathlib import Path
import subprocess
from typing import Any

import yaml

from evonn_compare.adapters.slots import fallback_native_id
from evonn_compare.comparison import build_matrix_summary, summarize_matrix_case
from evonn_compare.comparison.engine import ComparisonEngine
from evonn_compare.contracts.parity import load_parity_pack
from evonn_compare.ingest.loader import SystemIngestor
from evonn_compare.orchestration.config_gen import (
    DEFAULT_PRISM_ROOT,
    DEFAULT_TOPOGRAPH_ROOT,
    WORKSPACE_ROOT,
    generate_budget_pack,
    generate_prism_config,
    generate_topograph_config,
)
from evonn_compare.orchestration.contenders import ensure_contender_export
from evonn_compare.reporting import render_comparison_markdown, render_fair_matrix_markdown


COMPARE_ROOT = Path(__file__).resolve().parents[3]
DEFAULT_STRATOGRAPH_ROOT = WORKSPACE_ROOT / "EvoNN-Stratograph"
DEFAULT_CONTENDERS_ROOT = WORKSPACE_ROOT / "EvoNN-Contenders"
SYSTEM_ORDER = ("prism", "topograph", "stratograph", "contenders")


@dataclass(frozen=True)
class MatrixPaths:
    workspace: Path
    packs_dir: Path
    run_roots_dir: Path
    prism_configs_dir: Path
    topograph_configs_dir: Path
    stratograph_configs_dir: Path
    contender_configs_dir: Path
    reports_dir: Path
    logs_dir: Path
    manifest_path: Path


@dataclass(frozen=True)
class MatrixCase:
    pack_name: str
    seed: int
    budget: int
    pack_path: Path
    prism_config_path: Path
    topograph_config_path: Path
    stratograph_config_path: Path
    contender_config_path: Path
    prism_run_dir: Path
    topograph_run_dir: Path
    stratograph_run_dir: Path
    contender_run_dir: Path
    report_dir: Path
    summary_output_path: Path
    log_dir: Path


@dataclass(frozen=True)
class CommandSpec:
    name: str
    cwd: Path
    argv: list[str]


def prepare_fair_matrix_cases(
    *,
    pack_name: str,
    base_pack_path: Path,
    seeds: list[int],
    budgets: list[int],
    workspace: Path,
    prism_root: Path = DEFAULT_PRISM_ROOT,
    topograph_root: Path = DEFAULT_TOPOGRAPH_ROOT,
    stratograph_root: Path = DEFAULT_STRATOGRAPH_ROOT,
    contenders_root: Path = DEFAULT_CONTENDERS_ROOT,
) -> tuple[MatrixPaths, list[MatrixCase]]:
    workspace = workspace.resolve()
    paths = MatrixPaths(
        workspace=workspace,
        packs_dir=workspace / "packs",
        run_roots_dir=workspace / "runs",
        prism_configs_dir=workspace / "configs" / "prism",
        topograph_configs_dir=workspace / "configs" / "topograph",
        stratograph_configs_dir=workspace / "configs" / "stratograph",
        contender_configs_dir=workspace / "configs" / "contenders",
        reports_dir=workspace / "reports",
        logs_dir=workspace / "logs",
        manifest_path=workspace / "matrix.yaml",
    )
    for directory in (
        paths.workspace,
        paths.packs_dir,
        paths.run_roots_dir,
        paths.prism_configs_dir,
        paths.topograph_configs_dir,
        paths.stratograph_configs_dir,
        paths.contender_configs_dir,
        paths.reports_dir,
        paths.logs_dir,
    ):
        directory.mkdir(parents=True, exist_ok=True)

    base_payload = yaml.safe_load(base_pack_path.read_text(encoding="utf-8"))
    cases: list[MatrixCase] = []
    for budget in budgets:
        pack_path = generate_budget_pack(
            base_pack_path=base_pack_path,
            budget=budget,
            output_dir=paths.packs_dir,
            base_payload=base_payload,
        )
        pack = load_parity_pack(pack_path)
        for seed in seeds:
            case_name = f"{pack.name}_seed{seed}"
            report_dir = paths.reports_dir / case_name
            case = MatrixCase(
                pack_name=pack.name,
                seed=seed,
                budget=budget,
                pack_path=pack_path,
                prism_config_path=paths.prism_configs_dir / f"{case_name}.yaml",
                topograph_config_path=paths.topograph_configs_dir / f"{case_name}.yaml",
                stratograph_config_path=paths.stratograph_configs_dir / f"{case_name}.yaml",
                contender_config_path=paths.contender_configs_dir / f"{case_name}.yaml",
                prism_run_dir=paths.run_roots_dir / "prism" / case_name,
                topograph_run_dir=paths.run_roots_dir / "topograph" / case_name,
                stratograph_run_dir=paths.run_roots_dir / "stratograph" / case_name,
                contender_run_dir=paths.run_roots_dir / "contenders" / case_name,
                report_dir=report_dir,
                summary_output_path=report_dir / "four_way_summary.md",
                log_dir=paths.logs_dir / case_name,
            )
            generate_prism_config(
                output_path=case.prism_config_path,
                pack_path=pack_path,
                seed=seed,
                budget=budget,
            )
            generate_topograph_config(
                output_path=case.topograph_config_path,
                pack_path=pack_path,
                seed=seed,
                budget=budget,
                run_dir=case.topograph_run_dir,
            )
            generate_stratograph_config(
                output_path=case.stratograph_config_path,
                pack_path=pack_path,
                seed=seed,
                budget=budget,
            )
            generate_contender_config(
                output_path=case.contender_config_path,
                pack_path=pack_path,
                seed=seed,
                budget=budget,
                run_name=case_name,
            )
            cases.append(case)

    payload = {
        "pack_name": pack_name,
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
    return paths, cases


def generate_stratograph_config(
    *,
    output_path: Path,
    pack_path: Path,
    seed: int,
    budget: int,
) -> Path:
    pack = load_parity_pack(pack_path)
    if not pack.benchmarks:
        raise ValueError("pack must contain at least one benchmark")
    if budget % len(pack.benchmarks) != 0:
        raise ValueError(f"budget {budget} not divisible by benchmark_count {len(pack.benchmarks)}")
    benchmark_ids = [fallback_native_id(entry, "stratograph") for entry in pack.benchmarks]
    population_size, generations = _exact_factorization(budget // len(pack.benchmarks), preferred_population_cap=8)
    payload = {
        "seed": seed,
        "run_name": output_path.stem,
        "benchmark_pool": {
            "name": pack.name,
            "benchmarks": benchmark_ids,
        },
        "training": {
            "epochs": pack.budget_policy.epochs_per_candidate,
            "batch_size": 32,
            "learning_rate": 0.001,
            "multi_fidelity": False,
            "weight_inheritance": True,
        },
        "evolution": {
            "population_size": population_size,
            "generations": generations,
            "elite_per_benchmark": max(1, population_size // 4),
            "architecture_mode": "two_level_shared",
        },
    }
    output_path.parent.mkdir(parents=True, exist_ok=True)
    output_path.write_text(yaml.safe_dump(payload, sort_keys=False), encoding="utf-8")
    return output_path


def generate_contender_config(
    *,
    output_path: Path,
    pack_path: Path,
    seed: int,
    budget: int,
    run_name: str,
) -> Path:
    pack = load_parity_pack(pack_path)
    payload = {
        "seed": seed,
        "run_name": run_name,
        "benchmark_pool": {
            "name": pack.name,
            "benchmarks": [_contender_native_id(entry) for entry in pack.benchmarks],
        },
        "baseline": {
            "mode": "budget_matched",
            "target_evaluation_count": budget,
            "cache_dir": ".baseline-cache",
        },
    }
    output_path.parent.mkdir(parents=True, exist_ok=True)
    output_path.write_text(yaml.safe_dump(payload, sort_keys=False), encoding="utf-8")
    return output_path


def run_fair_matrix_case(
    case: MatrixCase,
    *,
    prism_root: Path = DEFAULT_PRISM_ROOT,
    topograph_root: Path = DEFAULT_TOPOGRAPH_ROOT,
    stratograph_root: Path = DEFAULT_STRATOGRAPH_ROOT,
    contenders_root: Path = DEFAULT_CONTENDERS_ROOT,
    parallel: bool = True,
) -> Path:
    case.report_dir.mkdir(parents=True, exist_ok=True)
    case.log_dir.mkdir(parents=True, exist_ok=True)

    stages = [
        [
            CommandSpec(
                name="prism_run",
                cwd=prism_root.resolve(),
                argv=["uv", "run", "prism", "evolve", "--config", str(case.prism_config_path), "--run-dir", str(case.prism_run_dir)],
            ),
            CommandSpec(
                name="topograph_run",
                cwd=topograph_root.resolve(),
                argv=["uv", "run", "topograph", "evolve", "--config", str(case.topograph_config_path), "--run-dir", str(case.topograph_run_dir)],
            ),
            CommandSpec(
                name="stratograph_run",
                cwd=stratograph_root.resolve(),
                argv=["uv", "run", "stratograph", "evolve", "--config", str(case.stratograph_config_path), "--run-dir", str(case.stratograph_run_dir)],
            ),
        ],
        [
            CommandSpec(
                name="prism_export",
                cwd=prism_root.resolve(),
                argv=[
                    "uv",
                    "run",
                    "prism",
                    "symbiosis",
                    "export",
                    str(case.prism_run_dir),
                    "--pack",
                    str(case.pack_path),
                    "--output-dir",
                    str(case.prism_run_dir),
                ],
            ),
            CommandSpec(
                name="topograph_export",
                cwd=topograph_root.resolve(),
                argv=[
                    "uv",
                    "run",
                    "topograph",
                    "symbiosis",
                    "export",
                    str(case.topograph_run_dir),
                    "--pack",
                    str(case.pack_path),
                    "--output-dir",
                    str(case.topograph_run_dir),
                ],
            ),
            CommandSpec(
                name="stratograph_export",
                cwd=stratograph_root.resolve(),
                argv=[
                    "uv",
                    "run",
                    "stratograph",
                    "symbiosis",
                    "export",
                    "--run-dir",
                    str(case.stratograph_run_dir),
                    "--pack-path",
                    str(case.pack_path),
                    "--output-dir",
                    str(case.stratograph_run_dir),
                ],
            ),
        ],
    ]

    for stage in stages:
        if parallel:
            _run_stage_parallel(stage, log_dir=case.log_dir)
        else:
            for spec in stage:
                _run_command(spec, log_dir=case.log_dir)

    ensure_contender_export(
        contenders_root=contenders_root.resolve(),
        config_path=case.contender_config_path,
        pack_path=case.pack_path,
        run_dir=case.contender_run_dir,
        output_dir=case.contender_run_dir,
        log_dir=case.log_dir,
    )

    pack = load_parity_pack(case.pack_path)
    ingestors = {
        system: SystemIngestor(run_dir)
        for system, run_dir in {
            "prism": case.prism_run_dir,
            "topograph": case.topograph_run_dir,
            "stratograph": case.stratograph_run_dir,
            "contenders": case.contender_run_dir,
        }.items()
    }
    runs = {
        system: (ingestor.load_manifest(), ingestor.load_results())
        for system, ingestor in ingestors.items()
    }

    pair_results: dict[tuple[str, str], tuple[Any, Path]] = {}
    for left_system, right_system in itertools.combinations(SYSTEM_ORDER, 2):
        left_manifest, left_results = runs[left_system]
        right_manifest, right_results = runs[right_system]
        result = ComparisonEngine().compare(
            left_manifest=left_manifest,
            left_results=left_results,
            right_manifest=right_manifest,
            right_results=right_results,
            pack=pack,
        )
        report_path = case.report_dir / f"{left_system}_vs_{right_system}.md"
        report_path.write_text(render_comparison_markdown(result), encoding="utf-8")
        report_path.with_suffix(".json").write_text(
            json.dumps(result.model_dump(mode="json"), indent=2),
            encoding="utf-8",
        )
        pair_results[(left_system, right_system)] = (result, report_path)

    fair_row, reference_row, parity_rows = summarize_matrix_case(
        pack=pack,
        budget=case.budget,
        seed=case.seed,
        runs=runs,
        pair_results=pair_results,
    )
    summary = build_matrix_summary(
        pack_name=case.pack_name,
        fair_rows=[fair_row] if fair_row is not None else [],
        reference_rows=[reference_row] if reference_row is not None else [],
        parity_rows=parity_rows,
    )
    case.summary_output_path.write_text(render_fair_matrix_markdown(summary), encoding="utf-8")
    case.summary_output_path.with_suffix(".json").write_text(
        json.dumps(asdict(summary), indent=2, default=str),
        encoding="utf-8",
    )
    return case.summary_output_path


def _run_command(spec: CommandSpec, *, log_dir: Path) -> None:
    log_dir.mkdir(parents=True, exist_ok=True)
    log_path = log_dir / f"{spec.name}.log"
    with log_path.open("w", encoding="utf-8") as handle:
        handle.write(f"$ {' '.join(spec.argv)}\n\n")
        handle.flush()
        process = subprocess.run(
            spec.argv,
            cwd=spec.cwd,
            text=True,
            stdout=handle,
            stderr=subprocess.STDOUT,
        )
    if process.returncode != 0:
        raise RuntimeError(f"{spec.name} failed; see {log_path}")


def _run_stage_parallel(specs: list[CommandSpec], *, log_dir: Path) -> None:
    errors: list[Exception] = []
    with ThreadPoolExecutor(max_workers=len(specs)) as pool:
        futures = [pool.submit(_run_command, spec, log_dir=log_dir) for spec in specs]
        for future in futures:
            try:
                future.result()
            except Exception as exc:
                errors.append(exc)
    if errors:
        raise errors[0]


def _exact_factorization(units: int, *, preferred_population_cap: int) -> tuple[int, int]:
    if units <= 0:
        raise ValueError(f"units must be positive, got {units}")
    for population_size in range(min(preferred_population_cap, units), 0, -1):
        if units % population_size == 0:
            return population_size, units // population_size
    return units, 1


def _contender_native_id(entry) -> str:
    native_ids = entry.native_ids or {}
    return (
        native_ids.get("contenders")
        or native_ids.get("stratograph")
        or native_ids.get("prism")
        or native_ids.get("topograph")
        or native_ids.get("evonn")
        or native_ids.get("evonn2")
        or entry.benchmark_id
    )
