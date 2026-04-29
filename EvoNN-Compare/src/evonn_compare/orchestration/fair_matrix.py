"""Fair four-way campaign generation and execution."""

from __future__ import annotations

from concurrent.futures import ThreadPoolExecutor
from dataclasses import asdict, dataclass
from datetime import datetime, timezone
import itertools
import json
from pathlib import Path
import shutil
import subprocess
from typing import Any

import yaml

from evonn_compare.comparison import (
    build_matrix_summary,
    build_matrix_trend_rows,
    summarize_matrix_case,
)
from evonn_compare.comparison.engine import ComparisonEngine
from evonn_compare.contracts.parity import load_parity_pack
from evonn_compare.ingest.loader import SystemIngestor
from evonn_compare.orchestration.benchmark_resolution import resolve_supported_benchmark_ids
from evonn_compare.orchestration.config_gen import (
    DEFAULT_PRISM_ROOT,
    DEFAULT_TOPOGRAPH_ROOT,
    WORKSPACE_ROOT,
    generate_budget_pack,
    generate_prism_config,
    generate_topograph_config,
)
from evonn_compare.orchestration.contenders import ensure_contender_export
from evonn_compare.orchestration.portable_smoke import (
    ensure_prism_portable_smoke_export,
    ensure_topograph_portable_smoke_export,
)
from evonn_compare.orchestration.primordia import ensure_primordia_export
from evonn_compare.reporting import (
    render_comparison_markdown,
    render_fair_matrix_markdown,
    render_fair_matrix_trend_markdown,
)
from evonn_compare.comparison.fair_matrix import (
    CORE_TRUSTED_SYSTEMS,
    EXTENDED_TRUSTED_SYSTEMS,
    LaneMetadata,
)
from evonn_shared.contracts import ArtifactPaths, BenchmarkEntry, BudgetEnvelope, DeviceInfo, ResultRecord, RunManifest
from evonn_shared.manifests import benchmark_signature, fairness_manifest, summary_core_from_results, write_json


COMPARE_ROOT = Path(__file__).resolve().parents[3]
DEFAULT_STRATOGRAPH_ROOT = WORKSPACE_ROOT / "EvoNN-Stratograph"
DEFAULT_PRIMORDIA_ROOT = WORKSPACE_ROOT / "EvoNN-Primordia"
DEFAULT_CONTENDERS_ROOT = WORKSPACE_ROOT / "EvoNN-Contenders"
SYSTEM_ORDER = ("prism", "topograph", "stratograph", "primordia", "contenders")
FOUR_PROJECT_SYSTEM_ORDER = ("prism", "topograph", "stratograph", "primordia")
MANAGED_WORKSPACE_ROOTS = ("packs", "runs", "configs", "reports", "trends", "logs")
MANAGED_WORKSPACE_FILES = (
    "matrix.yaml",
    "fair_matrix_dashboard.html",
    "fair_matrix_dashboard.json",
    "fair_matrix_trend_rows.jsonl",
    "fair_matrix_trends.md",
)


@dataclass(frozen=True)
class MatrixPaths:
    workspace: Path
    packs_dir: Path
    run_roots_dir: Path
    prism_configs_dir: Path
    topograph_configs_dir: Path
    stratograph_configs_dir: Path
    primordia_configs_dir: Path
    contender_configs_dir: Path
    reports_dir: Path
    trends_dir: Path
    logs_dir: Path
    manifest_path: Path


@dataclass(frozen=True)
class MatrixCase:
    pack_name: str
    lane_preset: str | None
    seed: int
    budget: int
    pack_path: Path
    prism_config_path: Path
    topograph_config_path: Path
    stratograph_config_path: Path
    primordia_config_path: Path
    contender_config_path: Path | None
    prism_run_dir: Path
    topograph_run_dir: Path
    stratograph_run_dir: Path
    primordia_run_dir: Path
    contender_run_dir: Path | None
    report_dir: Path
    summary_output_path: Path
    trend_dataset_path: Path
    log_dir: Path
    systems: tuple[str, ...]


@dataclass(frozen=True)
class CommandSpec:
    name: str
    cwd: Path
    argv: list[str]


def reset_fair_matrix_workspace(workspace: Path) -> None:
    workspace_path = workspace.resolve()
    for relative_path in MANAGED_WORKSPACE_ROOTS:
        _remove_generated_path(workspace_path / relative_path)
    for relative_path in MANAGED_WORKSPACE_FILES:
        _remove_generated_path(workspace_path / relative_path)


def _reset_case_outputs(case: MatrixCase) -> None:
    generated_paths = [
        case.report_dir,
        case.log_dir,
        case.prism_run_dir,
        case.topograph_run_dir,
        case.stratograph_run_dir,
        case.primordia_run_dir,
    ]
    if case.contender_run_dir is not None:
        generated_paths.append(case.contender_run_dir)
    for path in generated_paths:
        _remove_generated_path(path)


def _remove_generated_path(path: Path) -> None:
    if not path.exists() and not path.is_symlink():
        return
    if path.is_dir() and not path.is_symlink():
        shutil.rmtree(path)
        return
    path.unlink()


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
    primordia_root: Path = DEFAULT_PRIMORDIA_ROOT,
    contenders_root: Path = DEFAULT_CONTENDERS_ROOT,
    include_contenders: bool = True,
    lane_preset: str | None = None,
) -> tuple[MatrixPaths, list[MatrixCase]]:
    workspace = workspace.resolve()
    paths = MatrixPaths(
        workspace=workspace,
        packs_dir=workspace / "packs",
        run_roots_dir=workspace / "runs",
        prism_configs_dir=workspace / "configs" / "prism",
        topograph_configs_dir=workspace / "configs" / "topograph",
        stratograph_configs_dir=workspace / "configs" / "stratograph",
        primordia_configs_dir=workspace / "configs" / "primordia",
        contender_configs_dir=workspace / "configs" / "contenders",
        reports_dir=workspace / "reports",
        trends_dir=workspace / "trends",
        logs_dir=workspace / "logs",
        manifest_path=workspace / "matrix.yaml",
    )
    required_dirs = [
        paths.workspace,
        paths.packs_dir,
        paths.run_roots_dir,
        paths.prism_configs_dir,
        paths.topograph_configs_dir,
        paths.stratograph_configs_dir,
        paths.primordia_configs_dir,
        paths.reports_dir,
        paths.trends_dir,
        paths.logs_dir,
    ]
    if include_contenders:
        required_dirs.append(paths.contender_configs_dir)
    for directory in required_dirs:
        directory.mkdir(parents=True, exist_ok=True)

    base_payload = yaml.safe_load(base_pack_path.read_text(encoding="utf-8"))
    systems = SYSTEM_ORDER if include_contenders else FOUR_PROJECT_SYSTEM_ORDER
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
                lane_preset=lane_preset,
                seed=seed,
                budget=budget,
                pack_path=pack_path,
                prism_config_path=paths.prism_configs_dir / f"{case_name}.yaml",
                topograph_config_path=paths.topograph_configs_dir / f"{case_name}.yaml",
                stratograph_config_path=paths.stratograph_configs_dir / f"{case_name}.yaml",
                primordia_config_path=paths.primordia_configs_dir / f"{case_name}.yaml",
                contender_config_path=(paths.contender_configs_dir / f"{case_name}.yaml") if include_contenders else None,
                prism_run_dir=paths.run_roots_dir / "prism" / case_name,
                topograph_run_dir=paths.run_roots_dir / "topograph" / case_name,
                stratograph_run_dir=paths.run_roots_dir / "stratograph" / case_name,
                primordia_run_dir=paths.run_roots_dir / "primordia" / case_name,
                contender_run_dir=(paths.run_roots_dir / "contenders" / case_name) if include_contenders else None,
                report_dir=report_dir,
                summary_output_path=report_dir / "fair_matrix_summary.md",
                trend_dataset_path=paths.trends_dir / "fair_matrix_trends.jsonl",
                log_dir=paths.logs_dir / case_name,
                systems=systems,
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
            generate_primordia_config(
                output_path=case.primordia_config_path,
                pack_path=pack_path,
                seed=seed,
                budget=budget,
                run_name=case_name,
            )
            if include_contenders:
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
        "systems": list(systems),
        "trends_dir": str(paths.trends_dir),
        "trend_dataset": str(paths.trends_dir / "fair_matrix_trends.jsonl"),
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
    benchmark_ids = resolve_supported_benchmark_ids(pack.benchmarks, "stratograph")
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
            "benchmarks": resolve_supported_benchmark_ids(pack.benchmarks, "contenders"),
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


def generate_primordia_config(
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
            "benchmarks": resolve_supported_benchmark_ids(pack.benchmarks, "primordia"),
        },
        "search": {
            "mode": "budget_matched",
            "target_evaluation_count": budget,
        },
        "training": {
            "epochs_per_candidate": pack.budget_policy.epochs_per_candidate,
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
    primordia_root: Path = DEFAULT_PRIMORDIA_ROOT,
    contenders_root: Path = DEFAULT_CONTENDERS_ROOT,
    parallel: bool = True,
) -> Path:
    _reset_case_outputs(case)
    case.report_dir.mkdir(parents=True, exist_ok=True)
    case.log_dir.mkdir(parents=True, exist_ok=True)
    pack = load_parity_pack(case.pack_path)
    failures: dict[str, str] = {}

    use_portable_prism = not _native_runtime_available(prism_root.resolve(), "prism.pipeline.coordinator")
    use_portable_topograph = not _native_runtime_available(topograph_root.resolve(), "topograph.pipeline.coordinator")

    stage_runs: list[CommandSpec] = []
    stage_exports: list[CommandSpec] = []
    if not use_portable_prism:
        stage_runs.append(
            CommandSpec(
                name="prism_run",
                cwd=prism_root.resolve(),
                argv=["uv", "run", "prism", "evolve", "--config", str(case.prism_config_path), "--run-dir", str(case.prism_run_dir)],
            )
        )
        stage_exports.append(
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
            )
        )
    if not use_portable_topograph:
        stage_runs.append(
            CommandSpec(
                name="topograph_run",
                cwd=topograph_root.resolve(),
                argv=["uv", "run", "topograph", "evolve", "--config", str(case.topograph_config_path), "--run-dir", str(case.topograph_run_dir)],
            )
        )
        stage_exports.append(
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
            )
        )

    stage_runs.append(
        CommandSpec(
            name="stratograph_run",
            cwd=stratograph_root.resolve(),
            argv=["uv", "run", "stratograph", "evolve", "--config", str(case.stratograph_config_path), "--run-dir", str(case.stratograph_run_dir)],
        )
    )
    stage_exports.append(
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
        )
    )

    run_failures = _execute_stage_specs(specs=stage_runs, log_dir=case.log_dir, parallel=parallel)
    failures.update(run_failures)
    export_specs = [spec for spec in stage_exports if _system_name_from_stage(spec.name) not in failures]
    export_failures = _execute_stage_specs(specs=export_specs, log_dir=case.log_dir, parallel=parallel)
    failures.update(export_failures)

    if use_portable_prism:
        try:
            ensure_prism_portable_smoke_export(
                config_path=case.prism_config_path,
                pack_path=case.pack_path,
                run_dir=case.prism_run_dir,
                output_dir=case.prism_run_dir,
                log_dir=case.log_dir,
            )
        except Exception as exc:
            failures["prism"] = f"prism portable export failed: {type(exc).__name__}: {exc}"
    if use_portable_topograph:
        try:
            ensure_topograph_portable_smoke_export(
                config_path=case.topograph_config_path,
                pack_path=case.pack_path,
                run_dir=case.topograph_run_dir,
                output_dir=case.topograph_run_dir,
                log_dir=case.log_dir,
            )
        except Exception as exc:
            failures["topograph"] = f"topograph portable export failed: {type(exc).__name__}: {exc}"

    try:
        ensure_primordia_export(
            primordia_root=primordia_root.resolve(),
            config_path=case.primordia_config_path,
            pack_path=case.pack_path,
            run_dir=case.primordia_run_dir,
            output_dir=case.primordia_run_dir,
            log_dir=case.log_dir,
        )
    except Exception as exc:
        failures["primordia"] = f"primordia export failed: {type(exc).__name__}: {exc}"
    if case.contender_config_path is not None and case.contender_run_dir is not None:
        try:
            ensure_contender_export(
                contenders_root=contenders_root.resolve(),
                config_path=case.contender_config_path,
                pack_path=case.pack_path,
                run_dir=case.contender_run_dir,
                output_dir=case.contender_run_dir,
                log_dir=case.log_dir,
            )
        except Exception as exc:
            failures["contenders"] = f"contenders export failed: {type(exc).__name__}: {exc}"

    run_dirs = {
        "prism": case.prism_run_dir,
        "topograph": case.topograph_run_dir,
        "stratograph": case.stratograph_run_dir,
        "primordia": case.primordia_run_dir,
    }
    if case.contender_run_dir is not None:
        run_dirs["contenders"] = case.contender_run_dir
    for system, failure_reason in failures.items():
        if system not in run_dirs:
            continue
        _materialize_failed_run_export(
            case=case,
            pack=pack,
            system=system,
            run_dir=run_dirs[system],
            failure_reason=failure_reason,
        )
    runs = _load_runs_with_failure_fallback(case=case, pack=pack, run_dirs=run_dirs, failures=failures)

    pair_results: dict[tuple[str, str], tuple[Any, Path]] = {}
    for left_system, right_system in itertools.combinations(case.systems, 2):
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
        systems=case.systems,
    )
    lane = _build_lane_metadata(case=case, runs=runs, pair_results=pair_results)
    summary = build_matrix_summary(
        pack_name=case.pack_name,
        lane=lane,
        fair_rows=[fair_row] if fair_row is not None else [],
        reference_rows=[reference_row] if reference_row is not None else [],
        parity_rows=parity_rows,
        trend_rows=build_matrix_trend_rows(
            pack=pack,
            budget=case.budget,
            seed=case.seed,
            runs=runs,
            pair_results=pair_results,
            lane=lane,
            systems=case.systems,
        ),
        systems=case.systems,
    )
    case.summary_output_path.write_text(render_fair_matrix_markdown(summary), encoding="utf-8")
    case.summary_output_path.with_suffix(".json").write_text(
        json.dumps(asdict(summary), indent=2, default=str),
        encoding="utf-8",
    )
    case.summary_output_path.with_name("lane_acceptance.json").write_text(
        json.dumps(asdict(summary.lane) if summary.lane is not None else {}, indent=2, default=str),
        encoding="utf-8",
    )
    _write_trend_artifacts(case, summary.trend_rows)
    trend_records = _build_trend_records(case=case, pack=pack, runs=runs, lane=lane)
    case.summary_output_path.with_name("fair_matrix_trends.json").write_text(
        json.dumps(trend_records, indent=2),
        encoding="utf-8",
    )
    case.summary_output_path.with_name("fair_matrix_trends.jsonl").write_text(
        "".join(json.dumps(record) + "\n" for record in trend_records),
        encoding="utf-8",
    )
    case.trend_dataset_path.parent.mkdir(parents=True, exist_ok=True)
    with case.trend_dataset_path.open("a", encoding="utf-8") as handle:
        for record in trend_records:
            handle.write(json.dumps(record) + "\n")
    return case.summary_output_path


def _write_trend_artifacts(case: MatrixCase, trend_rows: list[Any]) -> None:
    case.report_dir.mkdir(parents=True, exist_ok=True)
    trend_rows_path = case.report_dir / "trend_rows.json"
    trend_rows_path.write_text(
        json.dumps([asdict(row) for row in trend_rows], indent=2, default=str),
        encoding="utf-8",
    )
    trend_report_path = case.report_dir / "trend_report.md"
    trend_report_path.write_text(
        render_fair_matrix_trend_markdown(trend_rows),
        encoding="utf-8",
    )
    workspace_jsonl = case.report_dir.parent / "fair_matrix_trend_rows.jsonl"
    with workspace_jsonl.open("a", encoding="utf-8") as handle:
        for row in trend_rows:
            handle.write(json.dumps(asdict(row), default=str) + "\n")
    row_type = type(trend_rows[0]) if trend_rows else None
    workspace_rows = []
    for line in workspace_jsonl.read_text(encoding="utf-8").splitlines():
        if not line.strip():
            continue
        payload = json.loads(line)
        workspace_rows.append(row_type(**payload) if row_type is not None else payload)
    workspace_report_path = case.report_dir.parent / "fair_matrix_trends.md"
    workspace_report_path.write_text(
        render_fair_matrix_trend_markdown(workspace_rows),
        encoding="utf-8",
    )


def _build_lane_metadata(
    *,
    case: MatrixCase,
    runs: dict[str, tuple[Any, list[Any]]],
    pair_results: dict[tuple[str, str], tuple[Any, Path]],
) -> LaneMetadata:
    case_systems = tuple(getattr(case, "systems", tuple(runs.keys())))
    artifact_completeness_ok = True
    observed_task_kinds: set[str] = set()
    budget_consistency_ok = True
    seed_consistency_ok = True
    budget_accounting_ok = True
    system_operating_states: dict[str, str] = {}
    system_accounting_issues: dict[str, list[str]] = {}
    system_coverage_notes: dict[str, str] = {}
    system_benchmark_complete: dict[str, bool] = {}
    for system, (manifest, _results) in runs.items():
        run_dir = _run_dir_for_system(case, system)
        required = [
            run_dir / "manifest.json",
            run_dir / "results.json",
            run_dir / "summary.json",
            run_dir / "report.md",
        ]
        dataset_manifest = getattr(manifest.artifacts, "dataset_manifest_json", None)
        if dataset_manifest:
            required.append(run_dir / str(dataset_manifest))
        system_artifacts_ok = all(path.exists() for path in required)
        if not system_artifacts_ok:
            artifact_completeness_ok = False
        if manifest.budget.evaluation_count != case.budget:
            budget_consistency_ok = False
        if manifest.seed != case.seed:
            seed_consistency_ok = False
        accounting_issues = _budget_accounting_issues(manifest)
        system_accounting_issues[system] = accounting_issues
        if accounting_issues:
            budget_accounting_ok = False
        coverage = _system_coverage_assessment(manifest=manifest, results=_results)
        system_benchmark_complete[system] = coverage["benchmark_complete"]
        if coverage["note"] is not None:
            system_coverage_notes[system] = coverage["note"]
        if not system_artifacts_ok:
            system_operating_states[system] = "artifacts-missing"
        elif manifest.budget.partial_run:
            system_operating_states[system] = "partial-run"
        elif accounting_issues:
            system_operating_states[system] = "accounting-incomplete"
        else:
            system_operating_states[system] = coverage["operating_state"]
        observed_task_kinds.update(entry.task_kind for entry in manifest.benchmarks)

    fairness_ok = all(result.parity_status == "fair" for result, _report_path in pair_results.values())
    task_coverage_ok = {"classification", "regression"}.issubset(observed_task_kinds)
    contract_fair_ok = (
        artifact_completeness_ok
        and fairness_ok
        and task_coverage_ok
        and budget_consistency_ok
        and seed_consistency_ok
        and budget_accounting_ok
    )
    core_systems_complete_ok = all(
        system in case_systems and system_benchmark_complete.get(system, False)
        for system in CORE_TRUSTED_SYSTEMS
    )
    extended_systems_complete_ok = all(
        system in case_systems and system_benchmark_complete.get(system, False)
        for system in EXTENDED_TRUSTED_SYSTEMS
    )
    if contract_fair_ok and core_systems_complete_ok and extended_systems_complete_ok:
        operating_state = "trusted-extended"
    elif contract_fair_ok and core_systems_complete_ok:
        operating_state = "trusted-core"
    elif contract_fair_ok:
        operating_state = "contract-fair"
    else:
        operating_state = "reference-only"
    acceptance_notes: list[str] = []
    if not artifact_completeness_ok:
        acceptance_notes.append("missing required artifacts")
    if not fairness_ok:
        acceptance_notes.append("pairwise fairness checks not all fair")
    if not task_coverage_ok:
        acceptance_notes.append("lane must cover both classification and regression")
    if not budget_consistency_ok:
        acceptance_notes.append("manifest evaluation counts drift from requested lane budget")
    if not seed_consistency_ok:
        acceptance_notes.append("manifest seeds drift from requested lane seed")
    if not budget_accounting_ok:
        accounting_notes = []
        for system in sorted(system_accounting_issues):
            issues = system_accounting_issues[system]
            if issues:
                accounting_notes.append(f"{system} ({'; '.join(issues)})")
        acceptance_notes.append("budget accounting incomplete: " + ", ".join(accounting_notes))
    for system in sorted(system_coverage_notes):
        acceptance_notes.append(f"{system} coverage: {system_coverage_notes[system]}")
    if contract_fair_ok and not core_systems_complete_ok:
        missing_core = [
            f"{system}={system_operating_states.get(system, 'not-participating')}"
            for system in CORE_TRUSTED_SYSTEMS
            if system not in case_systems or not system_benchmark_complete.get(system, False)
        ]
        acceptance_notes.append("trusted-core unmet: " + ", ".join(missing_core))
    if contract_fair_ok and core_systems_complete_ok and not extended_systems_complete_ok:
        missing_extended = [
            f"{system}={system_operating_states.get(system, 'not-participating')}"
            for system in EXTENDED_TRUSTED_SYSTEMS
            if system not in case_systems or not system_benchmark_complete.get(system, False)
        ]
        acceptance_notes.append("trusted-extended unmet: " + ", ".join(missing_extended))
    return LaneMetadata(
        preset=case.lane_preset,
        pack_name=case.pack_name,
        expected_budget=case.budget,
        expected_seed=case.seed,
        artifact_completeness_ok=artifact_completeness_ok,
        fairness_ok=fairness_ok,
        task_coverage_ok=task_coverage_ok,
        budget_consistency_ok=budget_consistency_ok,
        seed_consistency_ok=seed_consistency_ok,
        budget_accounting_ok=budget_accounting_ok,
        core_systems_complete_ok=core_systems_complete_ok,
        extended_systems_complete_ok=extended_systems_complete_ok,
        observed_task_kinds=tuple(sorted(observed_task_kinds)),
        system_operating_states=dict(sorted(system_operating_states.items())),
        operating_state=operating_state,
        acceptance_notes=tuple(acceptance_notes),
        repeatability_ready=operating_state in {"trusted-core", "trusted-extended"},
    )


def _run_dir_for_system(case: MatrixCase, system: str) -> Path:
    mapping = {
        "prism": case.prism_run_dir,
        "topograph": case.topograph_run_dir,
        "stratograph": case.stratograph_run_dir,
        "primordia": case.primordia_run_dir,
        "contenders": case.contender_run_dir,
    }
    run_dir = mapping[system]
    if run_dir is None:
        raise ValueError(f"no run dir for system {system}")
    return run_dir


def _budget_accounting_issues(manifest: Any) -> list[str]:
    issues: list[str] = []
    budget = manifest.budget
    if budget.actual_evaluations is None:
        issues.append("missing actual_evaluations")
    elif budget.partial_run:
        issues.append("partial_run=true")
    elif budget.actual_evaluations != budget.evaluation_count:
        issues.append(
            f"actual_evaluations {budget.actual_evaluations} != declared {budget.evaluation_count}"
        )
    if not budget.evaluation_semantics:
        issues.append("missing evaluation_semantics")
    if budget.resumed_from_run_id is not None and budget.resumed_evaluations is None:
        issues.append("missing resumed_evaluations")
    return issues


def _system_benchmark_complete(*, manifest: Any, results: list[Any]) -> bool:
    return bool(_system_coverage_assessment(manifest=manifest, results=results)["benchmark_complete"])


def _system_coverage_assessment(*, manifest: Any, results: list[Any]) -> dict[str, Any]:
    result_by_benchmark = {record.benchmark_id: record for record in results}
    blocking_benchmarks: list[str] = []
    for entry in manifest.benchmarks:
        if entry.status != "ok":
            blocking_benchmarks.append(_render_blocking_benchmark(entry.benchmark_id, entry.status, None))
            continue
        record = result_by_benchmark.get(entry.benchmark_id)
        if record is None:
            blocking_benchmarks.append(_render_blocking_benchmark(entry.benchmark_id, "missing", None))
            continue
        if record.status != "ok":
            blocking_benchmarks.append(_render_blocking_benchmark(entry.benchmark_id, record.status, record.failure_reason))
    if blocking_benchmarks:
        return {
            "benchmark_complete": False,
            "operating_state": "benchmark-incomplete",
            "note": "blocking benchmarks: " + "; ".join(blocking_benchmarks),
        }
    coverage = getattr(manifest, "baseline_coverage", None)
    optional_skips = getattr(coverage, "optional_dependency_skips", {}) if coverage is not None else {}
    if not optional_skips:
        return {
            "benchmark_complete": True,
            "operating_state": "benchmark-complete",
            "note": None,
        }
    rendered_skips = ", ".join(
        f"{group}=[{', '.join(names)}]"
        for group, names in sorted(optional_skips.items())
    )
    policy = getattr(coverage, "benchmark_complete_policy", None)
    if policy == "all_configured_contenders_required":
        return {
            "benchmark_complete": False,
            "operating_state": "benchmark-incomplete-optional-skips",
            "note": f"configured optional baselines were skipped: {rendered_skips}",
        }
    policy_stage = getattr(coverage, "policy_stage", "temporary")
    policy_reason = getattr(coverage, "policy_reason", None)
    note = f"optional baselines skipped but tolerated by policy: {rendered_skips}"
    if policy_stage == "steady_state":
        note = f"optional baselines skipped under ratified steady-state policy: {rendered_skips}"
    if policy_reason:
        note = f"{note} ({policy_reason})"
    return {
        "benchmark_complete": True,
        "operating_state": "benchmark-complete-optional-skips",
        "note": note,
    }


def _render_blocking_benchmark(benchmark_id: str, status: str, failure_reason: str | None) -> str:
    if failure_reason:
        return f"{benchmark_id}={status} ({failure_reason})"
    return f"{benchmark_id}={status}"


def _execute_stage_specs(*, specs: list[CommandSpec], log_dir: Path, parallel: bool) -> dict[str, str]:
    if not specs:
        return {}
    if parallel:
        return _run_stage_parallel(specs, log_dir=log_dir)
    failures: dict[str, str] = {}
    for spec in specs:
        try:
            _run_command(spec, log_dir=log_dir)
        except Exception as exc:
            failures[_system_name_from_stage(spec.name)] = str(exc)
    return failures


def _load_runs_with_failure_fallback(
    *,
    case: MatrixCase,
    pack: Any,
    run_dirs: dict[str, Path],
    failures: dict[str, str],
) -> dict[str, tuple[Any, list[Any]]]:
    runs: dict[str, tuple[Any, list[Any]]] = {}
    for system, run_dir in run_dirs.items():
        ingestor = SystemIngestor(run_dir)
        try:
            runs[system] = (ingestor.load_manifest(), ingestor.load_results())
        except Exception as exc:
            failure_reason = failures.get(system) or f"ingest failed: {type(exc).__name__}: {exc}"
            _materialize_failed_run_export(
                case=case,
                pack=pack,
                system=system,
                run_dir=run_dir,
                failure_reason=failure_reason,
            )
            fallback_ingestor = SystemIngestor(run_dir)
            runs[system] = (fallback_ingestor.load_manifest(), fallback_ingestor.load_results())
    return runs


def _materialize_failed_run_export(
    *,
    case: MatrixCase,
    pack: Any,
    system: str,
    run_dir: Path,
    failure_reason: str,
) -> None:
    run_dir.mkdir(parents=True, exist_ok=True)
    benchmark_entries = [
        {
            "benchmark_id": entry.benchmark_id,
            "task_kind": entry.task_kind,
            "metric_name": entry.metric_name,
            "metric_direction": entry.metric_direction,
            "status": "failed",
        }
        for entry in pack.benchmarks
    ]
    result_payloads = [
        {
            "system": system,
            "run_id": run_dir.name,
            "benchmark_id": entry.benchmark_id,
            "metric_name": entry.metric_name,
            "metric_direction": entry.metric_direction,
            "metric_value": None,
            "quality": None,
            "parameter_count": None,
            "train_seconds": None,
            "peak_memory_mb": None,
            "architecture_summary": None,
            "genome_id": None,
            "status": "failed",
            "failure_reason": failure_reason,
        }
        for entry in pack.benchmarks
    ]
    results = [ResultRecord(**payload) for payload in result_payloads]
    manifest = RunManifest(
        schema_version="1.0",
        system=system,
        run_id=run_dir.name,
        run_name=run_dir.name,
        created_at=datetime.now(timezone.utc),
        pack_name=pack.name,
        seed=case.seed,
        benchmarks=[BenchmarkEntry(**entry) for entry in benchmark_entries],
        budget=BudgetEnvelope(
            evaluation_count=case.budget,
            epochs_per_candidate=int(pack.budget_policy.epochs_per_candidate),
            effective_training_epochs=int(pack.budget_policy.epochs_per_candidate),
            wall_clock_seconds=None,
            generations=None,
            population_size=None,
            budget_policy_name=str(getattr(pack.budget_policy, "budget_policy_name", None) or "fair_matrix_stage_failure"),
            actual_evaluations=0,
            cached_evaluations=0,
            failed_evaluations=case.budget,
            invalid_evaluations=0,
            partial_run=True,
            evaluation_semantics="fair-matrix stage failure emitted synthetic failed records to preserve lane artifacts",
        ),
        device=DeviceInfo(
            device_name="unknown",
            precision_mode="unknown",
            framework="fair-matrix-orchestrator",
            framework_version=None,
        ),
        artifacts=ArtifactPaths(
            config_snapshot="config.yaml",
            report_markdown="report.md",
            dataset_manifest_json="dataset_manifest.json",
        ),
        fairness=fairness_manifest(
            pack_name=pack.name,
            seed=case.seed,
            evaluation_count=case.budget,
            budget_policy_name="fair_matrix_stage_failure",
            benchmark_entries=benchmark_entries,
            data_signature=benchmark_signature(pack.name, benchmark_entries),
            code_version=_code_version(),
        ),
    )
    dataset_manifest = [
        {
            "benchmark_id": entry.benchmark_id,
            "task_kind": entry.task_kind,
            "metric_name": entry.metric_name,
            "metric_direction": entry.metric_direction,
        }
        for entry in pack.benchmarks
    ]
    summary = {
        "system": system,
        "run_id": run_dir.name,
        "run_name": run_dir.name,
        "runtime_backend": "fair-matrix-stage-failure",
        "runtime_version": None,
        "precision_mode": "unknown",
        "total_evaluations": case.budget,
        **summary_core_from_results(results=result_payloads, parameter_counts=[]),
        "wall_clock_seconds": None,
        "best_results": [],
        "best_family": None,
        "failure_reason": failure_reason,
    }
    report_lines = [
        f"# {system.title()} Fair Matrix Failure Report",
        "",
        f"- Run ID: `{run_dir.name}`",
        f"- Pack: `{pack.name}`",
        f"- Seed: `{case.seed}`",
        f"- Budget: `{case.budget}`",
        f"- Failure Reason: `{failure_reason}`",
        "",
        "## Blocking Benchmarks",
        "",
    ]
    report_lines.extend(f"- `{entry.benchmark_id}` failed before export completed" for entry in pack.benchmarks)
    config_source = _config_path_for_system(case, system)
    if config_source.exists():
        shutil.copy2(config_source, run_dir / "config.yaml")
    (run_dir / "manifest.json").write_text(manifest.model_dump_json(indent=2), encoding="utf-8")
    write_json(run_dir / "results.json", [record.model_dump(mode="json") for record in results])
    write_json(run_dir / "summary.json", summary)
    (run_dir / "report.md").write_text("\n".join(report_lines) + "\n", encoding="utf-8")
    write_json(run_dir / "dataset_manifest.json", dataset_manifest)


def _config_path_for_system(case: MatrixCase, system: str) -> Path:
    mapping = {
        "prism": case.prism_config_path,
        "topograph": case.topograph_config_path,
        "stratograph": case.stratograph_config_path,
        "primordia": case.primordia_config_path,
        "contenders": case.contender_config_path,
    }
    config_path = mapping[system]
    if config_path is None:
        raise ValueError(f"no config path for system {system}")
    return config_path


def _build_trend_records(
    *,
    case: MatrixCase,
    pack: Any,
    runs: dict[str, tuple[Any, list[Any]]],
    lane: LaneMetadata,
) -> list[dict[str, Any]]:
    generated_at = datetime.now(timezone.utc).isoformat()
    trend_records: list[dict[str, Any]] = []

    for system in case.systems:
        manifest, results = runs[system]
        results_by_benchmark = {record.benchmark_id: record for record in results}
        fairness = manifest.fairness
        for benchmark in pack.benchmarks:
            record = results_by_benchmark.get(benchmark.benchmark_id)
            trend_records.append(
                {
                    "generated_at": generated_at,
                    "lane_preset": case.lane_preset,
                    "pack": case.pack_name,
                    "benchmark": benchmark.benchmark_id,
                    "task_kind": benchmark.task_kind,
                    "engine": system,
                    "run_id": manifest.run_id,
                    "run_name": manifest.run_name,
                    "created_at": manifest.created_at.isoformat(),
                    "seed": manifest.seed,
                    "budget": manifest.budget.evaluation_count,
                    "outcome_status": record.status if record is not None else "missing",
                    "metric_name": benchmark.metric_name,
                    "metric_direction": benchmark.metric_direction,
                    "metric_value": float(record.metric_value) if record is not None and record.metric_value is not None else None,
                    "quality": float(record.quality) if record is not None and record.quality is not None else None,
                    "failure_reason": record.failure_reason if record is not None else None,
                    "fairness": {
                        "benchmark_pack_id": fairness.benchmark_pack_id if fairness is not None else manifest.pack_name,
                        "seed": fairness.seed if fairness is not None else manifest.seed,
                        "evaluation_count": fairness.evaluation_count if fairness is not None else manifest.budget.evaluation_count,
                        "budget_policy_name": fairness.budget_policy_name if fairness is not None else manifest.budget.budget_policy_name,
                        "data_signature": fairness.data_signature if fairness is not None else None,
                        "code_version": fairness.code_version if fairness is not None else None,
                        "pairwise_fairness_ok": lane.fairness_ok,
                        "lane_operating_state": lane.operating_state,
                        "system_operating_state": lane.system_operating_states.get(system, "unknown"),
                        "budget_accounting_ok": lane.budget_accounting_ok,
                    },
                    "lane": {
                        "operating_state": lane.operating_state,
                        "repeatability_ready": lane.repeatability_ready,
                        "budget_accounting_ok": lane.budget_accounting_ok,
                    },
                    "system_operating_state": lane.system_operating_states.get(system, "unknown"),
                    "artifact_paths": {
                        "manifest": str(_run_dir_for_system(case, system) / "manifest.json"),
                        "results": str(_run_dir_for_system(case, system) / "results.json"),
                        "summary": str(_run_dir_for_system(case, system) / "summary.json"),
                        "report": str(_run_dir_for_system(case, system) / "report.md"),
                    },
                }
            )

    trend_records.sort(key=lambda item: (item["engine"], item["benchmark"]))
    return trend_records


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


def _run_stage_parallel(specs: list[CommandSpec], *, log_dir: Path) -> dict[str, str]:
    errors: dict[str, str] = {}
    with ThreadPoolExecutor(max_workers=len(specs)) as pool:
        futures = {
            pool.submit(_run_command, spec, log_dir=log_dir): spec
            for spec in specs
        }
        for future, spec in futures.items():
            try:
                future.result()
            except Exception as exc:
                errors[_system_name_from_stage(spec.name)] = str(exc)
    return errors


def _system_name_from_stage(stage_name: str) -> str:
    return stage_name.split("_", 1)[0]


def _native_runtime_available(project_root: Path, module_name: str) -> bool:
    try:
        process = subprocess.run(
            ["uv", "run", "python", "-c", f"import importlib; importlib.import_module({module_name!r})"],
            cwd=project_root,
            stdout=subprocess.DEVNULL,
            stderr=subprocess.DEVNULL,
            check=False,
        )
    except OSError:
        return False
    return process.returncode == 0


def _exact_factorization(units: int, *, preferred_population_cap: int) -> tuple[int, int]:
    if units <= 0:
        raise ValueError(f"units must be positive, got {units}")
    for population_size in range(min(preferred_population_cap, units), 0, -1):
        if units % population_size == 0:
            return population_size, units // population_size
    return units, 1


def _code_version() -> str | None:
    try:
        return subprocess.check_output(["git", "rev-parse", "HEAD"], cwd=Path(__file__).resolve().parents[4], text=True).strip()
    except Exception:
        return None


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


def _primordia_native_id(entry) -> str:
    native_ids = entry.native_ids or {}
    return (
        native_ids.get("primordia")
        or native_ids.get("contenders")
        or native_ids.get("stratograph")
        or native_ids.get("prism")
        or native_ids.get("topograph")
        or native_ids.get("evonn")
        or native_ids.get("evonn2")
        or entry.benchmark_id
    )
