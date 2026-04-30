"""Compare-facing transfer-regime workspace publishing."""

from __future__ import annotations

from collections import Counter
from dataclasses import asdict
import json
from pathlib import Path
import shutil
from statistics import median
from typing import Any, Literal

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


TransferRegime = Literal["none", "direct", "staged"]
_REGIME_ORDER: tuple[TransferRegime, ...] = ("none", "direct", "staged")
_REGIME_CASE_PREFIX = {"none": "01", "direct": "02", "staged": "03"}


def publish_transfer_regime_workspace(
    *,
    workspace: Path,
    pack_name: str,
    seeds: list[int],
    budget: int | None,
    primordia_root: Path,
    topograph_root: Path,
) -> dict[str, str | int]:
    """Publish a compare-facing workspace for no-seed, direct, and staged transfer regimes."""

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
    seed_artifacts_dir = workspace / "seed_artifacts"
    for directory in (configs_dir, runs_dir, reports_dir, trends_dir, logs_dir, seed_artifacts_dir):
        directory.mkdir(parents=True, exist_ok=True)

    comparison_engine = ComparisonEngine()
    workspace_trend_rows: list[dict[str, Any]] = []
    case_summary_paths: list[Path] = []
    aggregate_payloads: list[dict[str, Any]] = []
    first_primordia_run_dir: Path | None = None
    last_staged_seed_artifact: Path | None = None

    for seed in seeds:
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
            log_dir=logs_dir / f"seed{seed}",
        )
        if first_primordia_run_dir is None:
            first_primordia_run_dir = primordia_run_dir

        direct_seed_artifact = primordia_run_dir / "seed_candidates.json"
        direct_gate = _write_seed_artifact_gate(
            artifact_path=direct_seed_artifact,
            output_path=seed_artifacts_dir / f"seed{seed}_direct_quality.json",
            expected_systems={"primordia"},
            required_overlap_policy="family-overlapping",
        )

        seed_report_dir = reports_dir / f"seed{seed}"
        seed_report_dir.mkdir(parents=True, exist_ok=True)
        topograph_runs: dict[TransferRegime, dict[str, Any]] = {}
        staged_gate: dict[str, Any] | None = None
        staged_seed_artifact: Path | None = None

        for regime in _REGIME_ORDER:
            case_name = f"{_REGIME_CASE_PREFIX[regime]}-{_regime_slug(regime)}"
            case_report_dir = seed_report_dir / case_name
            topograph_seed_artifact: Path | None = None
            seeding_ladder: Literal["direct", "staged"] | None = None
            seed_gate: dict[str, Any] | None = None

            if regime == "direct":
                topograph_seed_artifact = direct_seed_artifact
                seeding_ladder = "direct"
                seed_gate = direct_gate
            elif regime == "staged":
                direct_run = topograph_runs["direct"]
                staged_seed_artifact = _build_staged_seed_artifact(
                    source_manifest=direct_run["manifest"],
                    source_results=direct_run["results"],
                    output_path=seed_artifacts_dir / f"seed{seed}_staged_seed_candidates.json",
                )
                last_staged_seed_artifact = staged_seed_artifact
                staged_gate = _write_seed_artifact_gate(
                    artifact_path=staged_seed_artifact,
                    output_path=seed_artifacts_dir / f"seed{seed}_staged_quality.json",
                    expected_systems={"topograph"},
                    required_overlap_policy="benchmark-overlapping",
                )
                topograph_seed_artifact = staged_seed_artifact
                seeding_ladder = "staged"
                seed_gate = staged_gate

            topograph_config_path = generate_topograph_config(
                output_path=configs_dir / "topograph" / f"{case_name}_seed{seed}.yaml",
                pack_path=budget_pack_path,
                seed=seed,
                budget=effective_budget,
                run_dir=runs_dir / "topograph" / f"{case_name}_seed{seed}",
                primordia_seed_candidates_path=topograph_seed_artifact,
            )
            topograph_run_dir = runs_dir / "topograph" / f"{case_name}_seed{seed}"
            ensure_topograph_portable_smoke_export(
                config_path=topograph_config_path,
                pack_path=budget_pack_path,
                run_dir=topograph_run_dir,
                output_dir=topograph_run_dir,
                log_dir=logs_dir / f"seed{seed}",
                seeding_ladder=seeding_ladder,
            )

            runs = {
                "primordia": _load_run(primordia_run_dir),
                "topograph": _load_run(topograph_run_dir),
            }
            topograph_manifest, topograph_results = runs["topograph"]
            topograph_runs[regime] = {
                "manifest": topograph_manifest,
                "results": topograph_results,
                "run_dir": topograph_run_dir,
                "seed_gate": seed_gate,
            }
            comparison = comparison_engine.compare(
                left_manifest=runs["primordia"][0],
                left_results=runs["primordia"][1],
                right_manifest=topograph_manifest,
                right_results=topograph_results,
                pack=pack,
            )
            lane = _build_lane_metadata(
                pack_name=pack.name,
                budget=effective_budget,
                seed=seed,
                runs=runs,
                comparison=comparison,
                regime=regime,
                seed_gate=seed_gate,
            )
            trend_rows = build_matrix_trend_rows(
                pack=pack,
                budget=effective_budget,
                seed=seed,
                runs=runs,
                pair_results={("primordia", "topograph"): (comparison, case_report_dir / "primordia_vs_topograph.md")},
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
                        report_path=case_report_dir / "primordia_vs_topograph.md",
                    )
                ],
                trend_rows=trend_rows,
                systems=("primordia", "topograph"),
            )
            _write_case_artifacts(
                report_dir=case_report_dir,
                summary=summary,
                comparison=comparison,
            )
            case_summary_paths.append(case_report_dir / "fair_matrix_summary.json")
            workspace_trend_rows.extend(asdict(row) for row in trend_rows)

        control_run = topograph_runs["none"]
        for regime in ("direct", "staged"):
            regime_run = topograph_runs[regime]
            regime_comparison = comparison_engine.compare(
                left_manifest=control_run["manifest"],
                left_results=control_run["results"],
                right_manifest=regime_run["manifest"],
                right_results=regime_run["results"],
                pack=pack,
            )
            payload = _transfer_regime_payload(
                regime=regime,
                comparison=regime_comparison,
                pack=pack,
                seed_gate=regime_run["seed_gate"],
            )
            _write_transfer_regime_summary(
                output_prefix=seed_report_dir / f"{_REGIME_CASE_PREFIX[regime]}-{_regime_slug(regime)}_vs_control",
                regime=regime,
                comparison=regime_comparison,
                payload=payload,
            )
            aggregate_payloads.append(payload)

    trend_dataset_path = trends_dir / "fair_matrix_trend_rows.jsonl"
    trend_dataset_path.write_text(
        "".join(json.dumps(row) + "\n" for row in workspace_trend_rows),
        encoding="utf-8",
    )

    transfer_summary_paths = _write_workspace_transfer_summary(
        reports_dir=reports_dir,
        payloads=aggregate_payloads,
    )
    workspace_artifacts = refresh_workspace_reports(workspace=workspace)
    return {
        "workspace": str(workspace),
        "pack_path": str(budget_pack_path),
        "primordia_run_dir": "" if first_primordia_run_dir is None else str(first_primordia_run_dir),
        "direct_seed_artifact": str(direct_seed_artifact),
        "staged_seed_artifact": "" if last_staged_seed_artifact is None else str(last_staged_seed_artifact),
        "transfer_summary_report": transfer_summary_paths["markdown"],
        "transfer_summary_data": transfer_summary_paths["json"],
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
    regime: TransferRegime,
    seed_gate: dict[str, Any] | None,
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
    fairness_ok = comparison.parity_status == "fair"
    budget_consistency_ok = all(manifest.budget.evaluation_count == budget for manifest in manifests)
    seed_consistency_ok = all(manifest.seed == seed for manifest in manifests)
    artifact_completeness_ok = seed_gate["passed"] if seed_gate is not None else True
    budget_accounting_ok = budget_consistency_ok and fairness_ok
    system_operating_states = {
        manifest.system: _system_operating_state(manifest)
        for manifest in manifests
    }
    acceptance_notes = [
        "regime buckets stay separated by workspace layout and manifest seeding metadata",
        "transfer outcomes are classified against the no-seed control, not collapsed into one seeded bucket",
    ]
    if regime == "none":
        acceptance_notes.append("control run keeps the same pack, budget, and seed with seeding disabled")
    elif regime == "direct":
        acceptance_notes.append("direct run consumes the upstream Primordia seed artifact only after quality gates pass")
    else:
        acceptance_notes.append("staged run consumes a compare-owned staged seed artifact derived from a prior seeded Topograph run")
    if seed_gate is not None:
        overlap_policy = seed_gate.get("top_candidate", {}).get("seed_overlap_policy") or "unknown"
        acceptance_notes.append(f"transfer provenance gate passed with overlap policy `{overlap_policy}`")

    operating_state = (
        "portable-transfer-plumbing"
        if fairness_ok and artifact_completeness_ok and budget_consistency_ok and seed_consistency_ok
        else "reference-only"
    )
    return LaneMetadata(
        preset="transfer-regimes",
        pack_name=pack_name,
        expected_budget=budget,
        expected_seed=seed,
        artifact_completeness_ok=artifact_completeness_ok,
        fairness_ok=fairness_ok,
        task_coverage_ok=bool(observed_task_kinds),
        budget_consistency_ok=budget_consistency_ok,
        seed_consistency_ok=seed_consistency_ok,
        budget_accounting_ok=budget_accounting_ok,
        core_systems_complete_ok=all(_all_ok(manifest) for manifest in manifests),
        extended_systems_complete_ok=True,
        observed_task_kinds=observed_task_kinds,
        system_operating_states=system_operating_states,
        operating_state=operating_state,
        acceptance_notes=tuple(acceptance_notes),
        repeatability_ready=False,
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


def _write_transfer_regime_summary(
    *,
    output_prefix: Path,
    regime: TransferRegime,
    comparison: ComparisonResult,
    payload: dict[str, Any],
) -> dict[str, str]:
    report_path = output_prefix.with_suffix(".md")
    json_path = output_prefix.with_suffix(".json")
    regime_label = _regime_slug(regime)
    lines = [
        f"# {regime_label.replace('-', ' ').title()} vs No-Seed Control",
        "",
        f"- Pack: `{comparison.pack_name}`",
        f"- Seed: `{comparison.right_manifest.seed}`",
        f"- Regime: `{payload['regime']}`",
        f"- Seed Source: `{payload['seed_source']}`",
        f"- Seed Artifact: `{payload['seed_artifact']}`",
        f"- Overlap Policy: `{payload['seed_overlap_policy']}`",
        f"- Verdict: `{payload['verdict']}`",
        f"- Regime Wins: `{payload['regime_wins']}`",
        f"- Control Wins: `{payload['control_wins']}`",
        f"- Ties: `{payload['ties']}`",
        "",
        "## Benchmark Deltas",
        "",
        "| Benchmark | Direction | Control | Regime | Delta | Outcome |",
        "|---|---|---:|---:|---:|---|",
    ]
    for row in payload["benchmark_deltas"]:
        left_value = "---" if row["control_metric"] is None else f"{float(row['control_metric']):.6f}"
        right_value = "---" if row["regime_metric"] is None else f"{float(row['regime_metric']):.6f}"
        delta_value = "---" if row["regime_delta"] is None else f"{float(row['regime_delta']):.6f}"
        lines.append(
            f"| {row['benchmark_id']} | {row['metric_direction']} | {left_value} | {right_value} | {delta_value} | {row['outcome']} |"
        )
    lines.extend(["", render_comparison_markdown(comparison)])
    report_path.write_text("\n".join(lines) + "\n", encoding="utf-8")
    json_path.write_text(json.dumps(payload, indent=2) + "\n", encoding="utf-8")
    return {"markdown": str(report_path), "json": str(json_path)}


def _transfer_regime_payload(
    *,
    regime: TransferRegime,
    comparison: ComparisonResult,
    pack,
    seed_gate: dict[str, Any] | None,
) -> dict[str, Any]:
    transfer_manifest = comparison.right_manifest
    seeding = transfer_manifest.seeding
    benchmark_families = {}
    for benchmark in pack.benchmarks:
        benchmark_families[benchmark.benchmark_id] = benchmark.task_kind
    benchmark_deltas: list[dict[str, Any]] = []
    gain_count = 0
    regression_count = 0
    tie_count = 0
    other_count = 0
    numeric_deltas: list[float] = []
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
                outcome = "gain"
            else:
                outcome = "regression"
            numeric_deltas.append(delta)
        if outcome == "gain":
            gain_count += 1
        elif outcome == "regression":
            regression_count += 1
        elif outcome == "tie":
            tie_count += 1
        else:
            other_count += 1
        benchmark_deltas.append(
            {
                "benchmark_id": matchup.benchmark_id,
                "benchmark_family": benchmark_families.get(matchup.benchmark_id, "unknown"),
                "metric_direction": matchup.metric_direction,
                "control_metric": matchup.left_value,
                "regime_metric": matchup.right_value,
                "regime_delta": delta,
                "outcome": outcome,
                "control_status": matchup.left_status,
                "regime_status": matchup.right_status,
                "note": matchup.note,
            }
        )

    regime_wins = comparison.summary.right_wins
    control_wins = comparison.summary.left_wins
    if regime_wins > 0 and control_wins == 0:
        verdict = "gain"
    elif control_wins > 0 and regime_wins == 0:
        verdict = "regression"
    elif regime_wins == 0 and control_wins == 0:
        verdict = "no_gain"
    else:
        verdict = "inconclusive"

    seed_quality = None
    if seed_gate is not None:
        top_candidate = dict(seed_gate.get("top_candidate") or {})
        seed_quality = {
            "gate_path": seed_gate.get("gate_path"),
            "artifact_path": seed_gate.get("artifact_path") or (None if seeding is None else seeding.seed_artifact_path),
            "candidate_count": seed_gate.get("candidate_count"),
            "family": top_candidate.get("family"),
            "benchmark_groups": top_candidate.get("benchmark_groups") or [],
            "benchmark_wins": top_candidate.get("benchmark_wins"),
            "repeat_support_count": top_candidate.get("repeat_support_count"),
            "median_quality": top_candidate.get("median_quality"),
            "overlap_policy": top_candidate.get("seed_overlap_policy"),
            "representative_genome_id": top_candidate.get("representative_genome_id"),
            "representative_architecture_summary": top_candidate.get("representative_architecture_summary"),
        }

    return {
        "regime": regime,
        "pack_name": comparison.pack_name,
        "seed": transfer_manifest.seed,
        "portable_backend": transfer_manifest.device.framework,
        "control_run_id": comparison.left_manifest.run_id,
        "regime_run_id": transfer_manifest.run_id,
        "seed_source": _seed_source_label(seeding),
        "seed_artifact": None if seeding is None else seeding.seed_artifact_path,
        "seed_overlap_policy": "unknown" if seeding is None or seeding.seed_overlap_policy is None else seeding.seed_overlap_policy,
        "regime_wins": regime_wins,
        "control_wins": control_wins,
        "ties": comparison.summary.ties,
        "verdict": verdict,
        "gain_count": gain_count,
        "regression_count": regression_count,
        "tie_count": tie_count,
        "other_count": other_count,
        "mean_regime_delta": None if not numeric_deltas else sum(numeric_deltas) / len(numeric_deltas),
        "benchmark_deltas": benchmark_deltas,
        "seed_quality": seed_quality,
    }


def _write_workspace_transfer_summary(*, reports_dir: Path, payloads: list[dict[str, Any]]) -> dict[str, str]:
    report_path = reports_dir / "transfer_regime_summary.md"
    json_path = report_path.with_suffix(".json")
    by_regime: dict[str, list[dict[str, Any]]] = {regime: [] for regime in ("direct", "staged")}
    for payload in payloads:
        by_regime[str(payload["regime"])].append(payload)

    summary = {
        "regimes": {
            regime: _aggregate_regime_payloads(regime=regime, payloads=rows)
            for regime, rows in by_regime.items()
        }
    }
    lines = [
        "# Transfer Regime Summary",
        "",
        "This workspace keeps `none`, `direct`, and `staged` as separate transfer buckets and aggregates transfer evidence against the no-seed control.",
        "",
    ]
    for regime in ("direct", "staged"):
        aggregate = summary["regimes"][regime]
        lines.extend(
            [
                f"## {regime.title()}",
                "",
                f"- Cases: `{aggregate['case_count']}`",
                f"- Consensus: `{aggregate['consensus']}`",
                f"- Verdict Counts: `gain={aggregate['verdict_counts']['gain']}`, `no_gain={aggregate['verdict_counts']['no_gain']}`, `regression={aggregate['verdict_counts']['regression']}`, `inconclusive={aggregate['verdict_counts']['inconclusive']}`",
                f"- Total Regime Wins: `{aggregate['total_regime_wins']}`",
                f"- Total Control Wins: `{aggregate['total_control_wins']}`",
                f"- Total Ties: `{aggregate['total_ties']}`",
                "",
                "| Seed | Verdict | Regime Wins | Control Wins | Ties | Seed Source | Overlap Policy |",
                "|---:|---|---:|---:|---:|---|---|",
            ]
        )
        for row in aggregate["cases"]:
            lines.append(
                f"| {row['seed']} | {row['verdict']} | {row['regime_wins']} | {row['control_wins']} | {row['ties']} | {row['seed_source']} | {row['seed_overlap_policy']} |"
            )
        lines.append("")

    report_path.write_text("\n".join(lines) + "\n", encoding="utf-8")
    json_path.write_text(json.dumps(summary, indent=2) + "\n", encoding="utf-8")
    return {"markdown": str(report_path), "json": str(json_path)}


def _aggregate_regime_payloads(*, regime: str, payloads: list[dict[str, Any]]) -> dict[str, Any]:
    verdict_counts = Counter(str(payload["verdict"]) for payload in payloads)
    case_count = len(payloads)
    if case_count == 0:
        consensus = "inconclusive"
    elif verdict_counts["gain"] == case_count:
        consensus = "gain"
    elif verdict_counts["regression"] == case_count:
        consensus = "regression"
    elif verdict_counts["no_gain"] == case_count:
        consensus = "no_gain"
    else:
        consensus = "inconclusive"
    return {
        "regime": regime,
        "case_count": case_count,
        "consensus": consensus,
        "verdict_counts": {
            "gain": verdict_counts["gain"],
            "no_gain": verdict_counts["no_gain"],
            "regression": verdict_counts["regression"],
            "inconclusive": verdict_counts["inconclusive"],
        },
        "total_regime_wins": sum(int(payload["regime_wins"]) for payload in payloads),
        "total_control_wins": sum(int(payload["control_wins"]) for payload in payloads),
        "total_ties": sum(int(payload["ties"]) for payload in payloads),
        "cases": [
            {
                "seed": payload["seed"],
                "verdict": payload["verdict"],
                "regime_wins": payload["regime_wins"],
                "control_wins": payload["control_wins"],
                "ties": payload["ties"],
                "seed_source": payload["seed_source"],
                "seed_overlap_policy": payload["seed_overlap_policy"],
            }
            for payload in sorted(payloads, key=lambda row: int(row["seed"]))
        ],
    }


def _seed_source_label(seeding) -> str:
    if seeding is None or seeding.seed_source_system is None:
        return "---"
    if seeding.seed_source_run_id:
        return f"{seeding.seed_source_system}:{seeding.seed_source_run_id}"
    return seeding.seed_source_system


def _build_staged_seed_artifact(
    *,
    source_manifest,
    source_results,
    output_path: Path,
) -> Path:
    ok_results = [
        record for record in source_results
        if record.status == "ok"
    ]
    if not ok_results:
        raise ValueError("cannot build staged seed artifact without at least one successful direct run result")

    qualities = [
        float(record.quality)
        for record in ok_results
        if record.quality is not None
    ]
    benchmark_groups = sorted(
        {
            entry.task_kind
            for entry in source_manifest.benchmarks
            if entry.status == "ok"
        }
    )
    seeding = source_manifest.seeding
    selected_family = None if seeding is None else seeding.seed_selected_family
    representative = next((record for record in ok_results if record.architecture_summary), ok_results[0])
    family = selected_family or _family_from_architecture_summary(representative.architecture_summary) or "topograph"
    payload = {
        "system": "topograph",
        "run_id": source_manifest.run_id,
        "run_name": source_manifest.run_name,
        "runtime": source_manifest.device.framework,
        "runtime_version": source_manifest.device.framework_version,
        "seed_candidates": [
            {
                "seed_rank": 1,
                "family": family,
                "benchmark_groups": benchmark_groups or ["mixed"],
                "evaluation_count": source_manifest.budget.evaluation_count,
                "benchmark_wins": len(ok_results),
                "benchmarks_won": [record.benchmark_id for record in ok_results],
                "supporting_benchmarks": [record.benchmark_id for record in ok_results],
                "repeat_support_count": len(ok_results),
                "median_quality": None if not qualities else median(qualities),
                "median_quality_by_group": {},
                "representative_genome_id": representative.genome_id,
                "representative_architecture_summary": representative.architecture_summary,
                "best_metric_name": representative.metric_name,
                "best_metric_value": representative.metric_value,
                "seed_overlap_policy": "benchmark-overlapping",
            }
        ],
        "benchmark_seeds": [
            {
                "benchmark_name": record.benchmark_id,
                "benchmark_group": next(
                    (
                        entry.task_kind
                        for entry in source_manifest.benchmarks
                        if entry.benchmark_id == record.benchmark_id
                    ),
                    "unknown",
                ),
                "family": family,
                "genome_id": record.genome_id,
                "architecture_summary": record.architecture_summary,
                "metric_name": record.metric_name,
                "metric_value": record.metric_value,
                "runtime": source_manifest.device.framework,
                "runtime_version": source_manifest.device.framework_version,
            }
            for record in ok_results
        ],
        "lineage": [
            {
                "system": source_manifest.system,
                "run_id": source_manifest.run_id,
                "seeding_ladder": None if seeding is None else seeding.seeding_ladder,
                "seed_source_system": None if seeding is None else seeding.seed_source_system,
                "seed_source_run_id": None if seeding is None else seeding.seed_source_run_id,
            }
        ],
    }
    output_path.parent.mkdir(parents=True, exist_ok=True)
    output_path.write_text(json.dumps(payload, indent=2) + "\n", encoding="utf-8")
    return output_path


def _family_from_architecture_summary(summary: str | None) -> str | None:
    if not summary:
        return None
    if "[" in summary:
        return summary.split("[", 1)[0].strip() or None
    if "=" in summary:
        return summary.split("=", 1)[0].strip() or None
    return None


def _write_seed_artifact_gate(
    *,
    artifact_path: Path,
    output_path: Path,
    expected_systems: set[str],
    required_overlap_policy: str,
) -> dict[str, Any]:
    if not artifact_path.exists():
        raise FileNotFoundError(f"seed artifact not found: {artifact_path}")
    payload = json.loads(artifact_path.read_text(encoding="utf-8"))
    gate = _seed_artifact_gate_payload(
        artifact_path=artifact_path,
        payload=payload,
        expected_systems=expected_systems,
        required_overlap_policy=required_overlap_policy,
    )
    gate["gate_path"] = str(output_path)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    output_path.write_text(json.dumps(gate, indent=2) + "\n", encoding="utf-8")
    if not gate["passed"]:
        reasons = "; ".join(gate["errors"]) or "unknown seed-artifact validation error"
        raise ValueError(f"seed artifact quality gates failed for {artifact_path}: {reasons}")
    return gate


def _seed_artifact_gate_payload(
    *,
    artifact_path: Path,
    payload: dict[str, Any],
    expected_systems: set[str],
    required_overlap_policy: str,
) -> dict[str, Any]:
    errors: list[str] = []
    warnings: list[str] = []
    system = str(payload.get("system") or "").strip()
    run_id = str(payload.get("run_id") or "").strip()
    candidates = list(payload.get("seed_candidates") or [])
    top_candidate = dict(candidates[0]) if candidates else {}
    overlap_policy = str(top_candidate.get("seed_overlap_policy") or required_overlap_policy)

    if not system:
        errors.append("missing top-level system")
    elif system not in expected_systems:
        errors.append(f"unexpected seed source system '{system}'")
    if not run_id:
        errors.append("missing top-level run_id")
    if not candidates:
        errors.append("seed_candidates is empty")
    else:
        rank = int(top_candidate.get("seed_rank", 0) or 0)
        if rank < 1:
            errors.append("top candidate seed_rank must be >= 1")
        if not str(top_candidate.get("family") or "").strip():
            errors.append("top candidate family is missing")
        benchmark_groups = [str(group).strip() for group in top_candidate.get("benchmark_groups") or [] if str(group).strip()]
        if not benchmark_groups:
            errors.append("top candidate benchmark_groups is empty")
        if int(top_candidate.get("benchmark_wins", 0) or 0) <= 0 and int(top_candidate.get("repeat_support_count", 0) or 0) <= 0:
            errors.append("top candidate has no benchmark wins or repeat support evidence")
        if not str(top_candidate.get("representative_architecture_summary") or "").strip():
            warnings.append("top candidate representative_architecture_summary is missing")
        if not str(top_candidate.get("representative_genome_id") or "").strip():
            warnings.append("top candidate representative_genome_id is missing")
        if overlap_policy != required_overlap_policy:
            errors.append(
                f"top candidate seed_overlap_policy '{overlap_policy}' does not match required '{required_overlap_policy}'"
            )

    gate = {
        "artifact_path": str(artifact_path),
        "passed": not errors,
        "system": system or None,
        "run_id": run_id or None,
        "candidate_count": len(candidates),
        "required_overlap_policy": required_overlap_policy,
        "errors": errors,
        "warnings": warnings,
        "top_candidate": {
            "family": top_candidate.get("family"),
            "seed_rank": top_candidate.get("seed_rank"),
            "benchmark_groups": top_candidate.get("benchmark_groups") or [],
            "benchmark_wins": top_candidate.get("benchmark_wins"),
            "repeat_support_count": top_candidate.get("repeat_support_count"),
            "median_quality": top_candidate.get("median_quality"),
            "seed_overlap_policy": overlap_policy,
            "representative_genome_id": top_candidate.get("representative_genome_id"),
            "representative_architecture_summary": top_candidate.get("representative_architecture_summary"),
        },
    }
    return gate


def _regime_slug(regime: TransferRegime) -> str:
    return "no-seed" if regime == "none" else regime


def _reset_workspace(workspace: Path) -> None:
    managed_dirs = ["configs", "logs", "packs", "reports", "runs", "trends", "seed_artifacts"]
    managed_files = ["fair_matrix_dashboard.html", "fair_matrix_dashboard.json"]
    for directory in managed_dirs:
        path = workspace / directory
        if path.exists():
            shutil.rmtree(path)
    for filename in managed_files:
        path = workspace / filename
        if path.exists():
            path.unlink()
