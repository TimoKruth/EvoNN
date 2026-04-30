"""Canonical seeded-vs-unseeded compare workspace publishing."""

from __future__ import annotations

from dataclasses import asdict
import json
from pathlib import Path
import shutil
from typing import Any

from evonn_compare.cli.workspace_report import refresh_workspace_reports
from evonn_compare.comparison.engine import ComparisonEngine, ComparisonResult
from evonn_compare.comparison.fair_matrix import LaneMetadata, build_matrix_summary, build_matrix_trend_rows
from evonn_compare.contracts.parity import load_parity_pack, resolve_pack_path
from evonn_compare.ingest.loader import SystemIngestor
from evonn_compare.orchestration.config_gen import generate_budget_pack, generate_topograph_config
from evonn_compare.orchestration.fair_matrix import generate_primordia_config
from evonn_compare.orchestration.portable_smoke import ensure_topograph_portable_smoke_export
from evonn_compare.orchestration.primordia import ensure_primordia_export
from evonn_compare.reporting.compare_md import render_comparison_markdown
from evonn_compare.reporting.fair_matrix_md import render_fair_matrix_markdown


def publish_seeded_vs_unseeded_workspace(
    *,
    workspace: Path,
    pack_name: str,
    seed: int,
    budget: int | None,
    primordia_root: Path,
    topograph_root: Path,
) -> dict[str, str | int]:
    """Publish a portable Primordia->Topograph seeded control workspace."""

    workspace = workspace.resolve()
    _reset_workspace(workspace)

    base_pack_path = resolve_pack_path(pack_name)
    base_pack = load_parity_pack(base_pack_path)
    effective_budget = budget or int(base_pack.budget_policy.evaluation_count)
    budget_pack_path = generate_budget_pack(
        base_pack_path=base_pack_path,
        budget=effective_budget,
        output_dir=workspace / "packs",
    )
    pack = load_parity_pack(budget_pack_path)

    configs_dir = workspace / "configs"
    runs_dir = workspace / "runs"
    reports_dir = workspace / "reports"
    trends_dir = workspace / "trends"
    logs_dir = workspace / "logs"
    for directory in (configs_dir, runs_dir, reports_dir, trends_dir, logs_dir):
        directory.mkdir(parents=True, exist_ok=True)

    primordia_case_name = f"{pack.name}_seed{seed}_source"
    primordia_config_path = generate_primordia_config(
        output_path=configs_dir / "primordia" / f"{primordia_case_name}.yaml",
        pack_path=budget_pack_path,
        seed=seed,
        budget=effective_budget,
        run_name=primordia_case_name,
    )
    primordia_run_dir = runs_dir / "primordia" / primordia_case_name
    ensure_primordia_export(
        primordia_root=primordia_root.resolve(),
        config_path=primordia_config_path,
        pack_path=budget_pack_path,
        run_dir=primordia_run_dir,
        output_dir=primordia_run_dir,
        log_dir=logs_dir,
    )
    seed_artifact_path = primordia_run_dir / "seed_candidates.json"
    if not seed_artifact_path.exists():
        raise FileNotFoundError(f"Primordia seed artifact not found: {seed_artifact_path}")

    case_specs = [
        {
            "name": "01-unseeded",
            "seed_artifact_path": None,
            "report_dir": reports_dir / "01-unseeded",
        },
        {
            "name": "02-seeded",
            "seed_artifact_path": seed_artifact_path,
            "report_dir": reports_dir / "02-seeded",
        },
    ]
    workspace_trend_rows: list[dict[str, Any]] = []
    case_summary_paths: list[Path] = []
    comparison_engine = ComparisonEngine()

    for case_spec in case_specs:
        case_name = case_spec["name"]
        topograph_config_path = generate_topograph_config(
            output_path=configs_dir / "topograph" / f"{case_name}.yaml",
            pack_path=budget_pack_path,
            seed=seed,
            budget=effective_budget,
            run_dir=runs_dir / "topograph" / case_name,
            primordia_seed_candidates_path=case_spec["seed_artifact_path"],
        )
        topograph_run_dir = runs_dir / "topograph" / case_name
        ensure_topograph_portable_smoke_export(
            config_path=topograph_config_path,
            pack_path=budget_pack_path,
            run_dir=topograph_run_dir,
            output_dir=topograph_run_dir,
            log_dir=logs_dir,
        )

        runs = {
            "primordia": _load_run(primordia_run_dir),
            "topograph": _load_run(topograph_run_dir),
        }
        comparison = comparison_engine.compare(
            left_manifest=runs["primordia"][0],
            left_results=runs["primordia"][1],
            right_manifest=runs["topograph"][0],
            right_results=runs["topograph"][1],
            pack=pack,
        )
        lane = _build_lane_metadata(
            pack_name=pack.name,
            budget=effective_budget,
            seed=seed,
            runs=runs,
            comparison=comparison,
            seeded=case_spec["seed_artifact_path"] is not None,
        )
        trend_rows = build_matrix_trend_rows(
            pack=pack,
            budget=effective_budget,
            seed=seed,
            runs=runs,
            pair_results={("primordia", "topograph"): (comparison, case_spec["report_dir"] / "primordia_vs_topograph.md")},
            lane=lane,
            systems=("primordia", "topograph"),
        )
        summary = build_matrix_summary(
            pack_name=pack.name,
            lane=lane,
            fair_rows=[] if comparison.parity_status != "fair" else [
                _matrix_budget_row(pack, effective_budget, seed, runs, comparison)
            ],
            reference_rows=[] if comparison.parity_status == "fair" else [
                _matrix_budget_row(pack, effective_budget, seed, runs, comparison)
            ],
            parity_rows=[
                _pair_parity_row(
                    budget=effective_budget,
                    seed=seed,
                    comparison=comparison,
                    report_path=case_spec["report_dir"] / "primordia_vs_topograph.md",
                )
            ],
            trend_rows=trend_rows,
            systems=("primordia", "topograph"),
        )
        _write_case_artifacts(
            report_dir=case_spec["report_dir"],
            summary=summary,
            comparison=comparison,
        )
        case_summary_paths.append(case_spec["report_dir"] / "fair_matrix_summary.json")
        workspace_trend_rows.extend(asdict(row) for row in trend_rows)

    trend_dataset_path = trends_dir / "fair_matrix_trend_rows.jsonl"
    trend_dataset_path.write_text(
        "".join(json.dumps(row) + "\n" for row in workspace_trend_rows),
        encoding="utf-8",
    )

    unseeded_runs = _load_run(runs_dir / "topograph" / "01-unseeded")
    seeded_runs = _load_run(runs_dir / "topograph" / "02-seeded")
    seeded_vs_unseeded = comparison_engine.compare(
        left_manifest=unseeded_runs[0],
        left_results=unseeded_runs[1],
        right_manifest=seeded_runs[0],
        right_results=seeded_runs[1],
        pack=pack,
    )
    summary_paths = _write_seeded_vs_unseeded_summary(
        reports_dir=reports_dir,
        comparison=seeded_vs_unseeded,
    )

    workspace_artifacts = refresh_workspace_reports(workspace=workspace)
    return {
        "workspace": str(workspace),
        "pack_path": str(budget_pack_path),
        "primordia_run_dir": str(primordia_run_dir),
        "seed_artifact": str(seed_artifact_path),
        "unseeded_run_dir": str((runs_dir / "topograph" / "01-unseeded").resolve()),
        "seeded_run_dir": str((runs_dir / "topograph" / "02-seeded").resolve()),
        "seeded_vs_unseeded_report": summary_paths["markdown"],
        "seeded_vs_unseeded_data": summary_paths["json"],
        "summary_count": len(case_summary_paths),
        "trend_dataset": str(trend_dataset_path),
        "trend_report": str(workspace_artifacts["trend_report"]),
        "trend_report_data": str(workspace_artifacts["trend_report_data"]),
        "dashboard": str(workspace_artifacts["dashboard"]),
        "dashboard_data": str(workspace_artifacts["dashboard_data"]),
    }


def _load_run(run_dir: Path):
    ingestor = SystemIngestor(run_dir)
    return ingestor.load_manifest(), ingestor.load_results()


def _build_lane_metadata(
    *,
    pack_name: str,
    budget: int,
    seed: int,
    runs: dict[str, tuple[Any, list[Any]]],
    comparison: ComparisonResult,
    seeded: bool,
) -> LaneMetadata:
    manifests = [manifest for manifest, _results in runs.values()]
    observed_task_kinds = tuple(
        sorted(
            {
                entry.task_kind
                for manifest in manifests
                for entry in manifest.benchmarks
            }
        )
    )
    artifact_completeness_ok = True
    fairness_ok = comparison.parity_status == "fair"
    task_coverage_ok = {"classification", "regression"}.issubset(set(observed_task_kinds))
    budget_consistency_ok = all(manifest.budget.evaluation_count == budget for manifest in manifests)
    seed_consistency_ok = all(manifest.seed == seed for manifest in manifests)
    budget_accounting_ok = budget_consistency_ok and fairness_ok
    system_operating_states = {
        manifest.system: _system_operating_state(manifest)
        for manifest in manifests
    }
    repeatability_ready = False
    acceptance_notes = [
        "portable topograph exporter used for host-portable reproduction only",
        "portable seeded-vs-unseeded lane validates seeding contract plumbing, not native MLX transfer behavior",
        "seeded topograph case consumes the Primordia seed_candidates.json artifact directly",
        "topograph seeding mode is visible in manifests, compare markdown, trend rows, and dashboard payloads",
    ]
    if not seeded:
        acceptance_notes[2] = "unseeded control case keeps Topograph on the same pack/budget/seed without a Primordia seed artifact"
    operating_state = (
        "portable-transfer-plumbing"
        if fairness_ok and task_coverage_ok and budget_consistency_ok and seed_consistency_ok
        else "reference-only"
    )
    return LaneMetadata(
        preset="seeded-compare",
        pack_name=pack_name,
        expected_budget=budget,
        expected_seed=seed,
        artifact_completeness_ok=artifact_completeness_ok,
        fairness_ok=fairness_ok,
        task_coverage_ok=task_coverage_ok,
        budget_consistency_ok=budget_consistency_ok,
        seed_consistency_ok=seed_consistency_ok,
        budget_accounting_ok=budget_accounting_ok,
        core_systems_complete_ok=all(_all_ok(manifest) for manifest in manifests),
        extended_systems_complete_ok=True,
        observed_task_kinds=observed_task_kinds,
        system_operating_states=system_operating_states,
        operating_state=operating_state,
        acceptance_notes=tuple(acceptance_notes),
        repeatability_ready=repeatability_ready,
    )


def _system_operating_state(manifest) -> str:
    if not _all_ok(manifest):
        return "partial-run"
    if manifest.device.framework == "portable-sklearn":
        return "portable-smoke"
    return "benchmark-complete"


def _all_ok(manifest) -> bool:
    return all(entry.status == "ok" for entry in manifest.benchmarks)


def _matrix_budget_row(pack, budget: int, seed: int, runs, comparison: ComparisonResult):
    wins = {"primordia": comparison.summary.left_wins, "topograph": comparison.summary.right_wins}
    evaluation_counts = {
        system: manifest.budget.evaluation_count
        for system, (manifest, _results) in runs.items()
    }
    from evonn_compare.comparison.fair_matrix import MatrixBudgetRow

    return MatrixBudgetRow(
        budget=budget,
        seed=seed,
        benchmark_count=len(pack.benchmarks),
        evaluation_counts=evaluation_counts,
        wins=wins,
        ties=comparison.summary.ties,
        note=None if comparison.parity_status == "fair" else "; ".join(comparison.reasons) or comparison.parity_status,
    )


def _pair_parity_row(*, budget: int, seed: int, comparison: ComparisonResult, report_path: Path):
    from evonn_compare.comparison.fair_matrix import PairParityRow

    return PairParityRow(
        budget=budget,
        seed=seed,
        pair_label="primordia vs topograph",
        parity_status=comparison.parity_status,
        left_eval_count=comparison.left_manifest.budget.evaluation_count,
        right_eval_count=comparison.right_manifest.budget.evaluation_count,
        left_policy=comparison.left_manifest.budget.budget_policy_name,
        right_policy=comparison.right_manifest.budget.budget_policy_name,
        data_signature_match=_data_signature(comparison.left_manifest) == _data_signature(comparison.right_manifest),
        reason="; ".join(comparison.reasons) if comparison.reasons else None,
        comparison_report=report_path,
    )


def _data_signature(manifest) -> str | None:
    fairness = manifest.fairness
    return None if fairness is None else fairness.data_signature


def _write_case_artifacts(
    *,
    report_dir: Path,
    summary,
    comparison: ComparisonResult,
) -> None:
    report_dir.mkdir(parents=True, exist_ok=True)
    summary_md_path = report_dir / "fair_matrix_summary.md"
    summary_json_path = report_dir / "fair_matrix_summary.json"
    pair_md_path = report_dir / "primordia_vs_topograph.md"
    pair_json_path = pair_md_path.with_suffix(".json")
    summary_md_path.write_text(render_fair_matrix_markdown(summary) + "\n", encoding="utf-8")
    summary_json_path.write_text(json.dumps(asdict(summary), indent=2, default=str) + "\n", encoding="utf-8")
    pair_md_path.write_text(render_comparison_markdown(comparison) + "\n", encoding="utf-8")
    pair_json_path.write_text(json.dumps(comparison.model_dump(mode="json"), indent=2) + "\n", encoding="utf-8")
    (report_dir / "lane_acceptance.json").write_text(
        json.dumps(asdict(summary.lane) if summary.lane is not None else {}, indent=2, default=str) + "\n",
        encoding="utf-8",
    )
    trend_rows = [asdict(row) for row in summary.trend_rows]
    (report_dir / "trend_rows.json").write_text(json.dumps(trend_rows, indent=2) + "\n", encoding="utf-8")
    (report_dir / "fair_matrix_trends.json").write_text(json.dumps(trend_rows, indent=2) + "\n", encoding="utf-8")
    (report_dir / "fair_matrix_trends.jsonl").write_text(
        "".join(json.dumps(row) + "\n" for row in trend_rows),
        encoding="utf-8",
    )


def _write_seeded_vs_unseeded_summary(*, reports_dir: Path, comparison: ComparisonResult) -> dict[str, str]:
    report_path = reports_dir / "seeded_vs_unseeded_summary.md"
    json_path = report_path.with_suffix(".json")
    report_payload = _seeded_vs_unseeded_payload(comparison)
    lines = [
        "# Canonical Seeded vs Unseeded Compare",
        "",
        f"- Pack: `{comparison.pack_name}`",
        f"- Left: `{comparison.left_manifest.run_id}`",
        f"- Right: `{comparison.right_manifest.run_id}`",
        f"- Transfer Boundary: `{report_payload['transfer_boundary']}`",
        f"- Transfer Proof State: `{report_payload['transfer_proof_state']}`",
        f"- Verdict: `{report_payload['verdict']}`",
        f"- Seeded Wins: `{report_payload['seeded_wins']}`",
        f"- Unseeded Wins: `{report_payload['unseeded_wins']}`",
        f"- Ties: `{report_payload['ties']}`",
        "",
        "## Benchmark Deltas",
        "",
        "| Benchmark | Direction | Unseeded | Seeded | Seed-Adjusted Delta | Outcome |",
        "|---|---|---:|---:|---:|---|",
    ]
    for row in report_payload["benchmark_deltas"]:
        left_value = "---" if row["unseeded_metric"] is None else f"{float(row['unseeded_metric']):.6f}"
        right_value = "---" if row["seeded_metric"] is None else f"{float(row['seeded_metric']):.6f}"
        delta_value = "---" if row["seed_adjusted_delta"] is None else f"{float(row['seed_adjusted_delta']):.6f}"
        lines.append(
            f"| {row['benchmark_id']} | {row['metric_direction']} | {left_value} | {right_value} | {delta_value} | {row['outcome']} |"
        )
    lines.extend(["", render_comparison_markdown(comparison)])
    report_path.write_text("\n".join(lines) + "\n", encoding="utf-8")
    json_path.write_text(json.dumps(report_payload, indent=2) + "\n", encoding="utf-8")
    return {"markdown": str(report_path), "json": str(json_path)}


def _seeded_vs_unseeded_payload(comparison: ComparisonResult) -> dict[str, Any]:
    seeded = comparison.right_manifest
    seeding = seeded.seeding
    benchmark_deltas: list[dict[str, Any]] = []
    for matchup in comparison.matchups:
        delta = None
        outcome = matchup.winner
        if matchup.left_status == "ok" and matchup.right_status == "ok" and matchup.left_value is not None and matchup.right_value is not None:
            if matchup.metric_direction == "max":
                delta = float(matchup.right_value - matchup.left_value)
            else:
                delta = float(matchup.left_value - matchup.right_value)
            if abs(delta) <= 1e-12:
                outcome = "tie"
            elif delta > 0:
                outcome = "seeded_gain"
            else:
                outcome = "seeded_regression"
        benchmark_deltas.append(
            {
                "benchmark_id": matchup.benchmark_id,
                "metric_direction": matchup.metric_direction,
                "unseeded_metric": matchup.left_value,
                "seeded_metric": matchup.right_value,
                "seed_adjusted_delta": delta,
                "outcome": outcome,
            }
        )
    seeded_wins = comparison.summary.right_wins
    unseeded_wins = comparison.summary.left_wins
    if seeded_wins > 0 and unseeded_wins == 0:
        verdict = "gain"
    elif unseeded_wins > 0 and seeded_wins == 0:
        verdict = "regression"
    elif seeded_wins == 0 and unseeded_wins == 0:
        verdict = "no_gain"
    else:
        verdict = "inconclusive"
    return {
        "lane_name": "primordia_to_topograph_seeded_vs_unseeded",
        "pack_name": comparison.pack_name,
        "seed": seeded.seed,
        "portable_backend": seeded.device.framework,
        "transfer_boundary": "portable-topograph-seeding-contract",
        "transfer_proof_state": (
            "portable-plumbing-only"
            if seeded.device.framework == "portable-sklearn"
            else "native-runtime-transfer"
        ),
        "seed_artifact": None if seeding is None else seeding.seed_artifact_path,
        "seed_source_run_id": None if seeding is None else seeding.seed_source_run_id,
        "seed_selected_family": None if seeding is None else seeding.seed_selected_family,
        "seed_representative_architecture_summary": None if seeding is None else seeding.representative_architecture_summary,
        "unseeded_run_dir": comparison.left_manifest.run_id,
        "seeded_run_dir": comparison.right_manifest.run_id,
        "seeded_wins": seeded_wins,
        "unseeded_wins": unseeded_wins,
        "ties": comparison.summary.ties,
        "verdict": verdict,
        "benchmark_deltas": benchmark_deltas,
    }


def _reset_workspace(workspace: Path) -> None:
    managed_dirs = ["configs", "logs", "packs", "reports", "runs", "trends"]
    managed_files = ["fair_matrix_dashboard.html", "fair_matrix_dashboard.json"]
    for directory in managed_dirs:
        path = workspace / directory
        if path.exists():
            shutil.rmtree(path)
    for filename in managed_files:
        path = workspace / filename
        if path.exists():
            path.unlink()
