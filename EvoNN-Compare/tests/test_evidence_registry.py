import json
from pathlib import Path

from typer.testing import CliRunner

from evonn_compare.cli.main import app
from evonn_compare.orchestration.evidence_registry import promote_evidence, validate_registry, validate_registry_artifacts


runner = CliRunner()


def test_promote_evidence_writes_registry_report_and_decision_groups(tmp_path: Path) -> None:
    workspace = tmp_path / "workspace"
    for seed, prism_score, topograph_score in ((42, 0.90, 0.82), (43, 0.88, 0.81)):
        _write_summary(workspace, seed=seed, prism_score=prism_score, topograph_score=topograph_score)

    result = promote_evidence(
        inputs=[workspace],
        registry=tmp_path / "evidence",
        label="topograph-transfer-slice",
        min_seeds=2,
    )

    assert result["promoted_count"] == 2
    assert result["registry_count"] == 2
    payload = json.loads((tmp_path / "evidence" / "evidence_report.json").read_text(encoding="utf-8"))
    assert payload["record_count"] == 2
    assert payload["groups"][0]["decision_label"] == "gain"
    assert payload["groups"][0]["leader"] == "prism"
    assert payload["artifact_validation"]["ok"] is True
    assert payload["transfer_evidence"]["seeded_trend_row_count"] == 2
    assert payload["lm_flatline_diagnostics"][0]["flatline_suspected"] is True
    assert payload["quality_diversity_evidence"]["claim_ready"] is True
    assert "EvoNN Evidence Registry Report" in (tmp_path / "evidence" / "evidence_report.md").read_text(encoding="utf-8")
    assert validate_registry(registry=tmp_path / "evidence")["ok"] is True
    assert validate_registry_artifacts(registry=tmp_path / "evidence")["ok"] is True
    assert (tmp_path / "evidence" / "README.md").exists()


def test_evidence_cli_promote_report_and_validate(tmp_path: Path) -> None:
    workspace = tmp_path / "workspace"
    _write_summary(workspace, seed=42, prism_score=0.8, topograph_score=0.8)
    registry = tmp_path / "evidence"

    promoted = runner.invoke(
        app,
        [
            "evidence",
            "promote",
            str(workspace),
            "--registry",
            str(registry),
            "--label",
            "single-seed-smoke",
            "--min-seeds",
            "2",
        ],
    )

    assert promoted.exit_code == 0
    assert "promoted\t1" in promoted.stdout
    report = json.loads((registry / "evidence_report.json").read_text(encoding="utf-8"))
    assert report["groups"][0]["decision_label"] == "inconclusive"

    refreshed = runner.invoke(app, ["evidence", "report", "--registry", str(registry), "--min-seeds", "1"])
    assert refreshed.exit_code == 0
    assert "records\t1" in refreshed.stdout

    validated = runner.invoke(app, ["evidence", "validate", "--registry", str(registry)])
    assert validated.exit_code == 0
    assert "ok\tTrue" in validated.stdout

    dashboard = runner.invoke(app, ["dashboard", str(registry), "--output", str(tmp_path / "dashboard.html"), "--no-open"])
    assert dashboard.exit_code == 0
    assert "summaries\t1" in dashboard.stdout


def test_evidence_report_adds_before_after_comparison_labels(tmp_path: Path) -> None:
    before = tmp_path / "before"
    after = tmp_path / "after"
    _write_summary(before, seed=42, prism_score=0.5, topograph_score=0.4)
    _write_summary(before, seed=43, prism_score=0.5, topograph_score=0.4)
    _write_summary(after, seed=42, prism_score=0.9, topograph_score=0.8)
    _write_summary(after, seed=43, prism_score=0.9, topograph_score=0.8)
    registry = tmp_path / "evidence"

    promote_evidence(inputs=[before], registry=registry, label="before-change", min_seeds=2)
    promote_evidence(inputs=[after], registry=registry, label="after-change", min_seeds=2)

    payload = json.loads((registry / "evidence_report.json").read_text(encoding="utf-8"))
    comparisons = payload["before_after_comparisons"]
    assert comparisons
    assert comparisons[0]["decision_label"] == "likely_gain"


def test_evidence_report_loads_transfer_case_verdicts(tmp_path: Path) -> None:
    workspace = tmp_path / "transfer"
    _write_summary(workspace, seed=42, prism_score=0.8, topograph_score=0.8, report_parts=("seed42", "02-direct"))
    case_path = workspace / "reports" / "seed42" / "02-direct_vs_control.json"
    case_path.write_text(
        json.dumps({"regime": "direct", "seed": 42, "verdict": "gain", "gain_count": 3, "regression_count": 0}),
        encoding="utf-8",
    )

    promote_evidence(inputs=[workspace], registry=tmp_path / "evidence", label="portable-transfer", min_seeds=1)

    payload = json.loads((tmp_path / "evidence" / "evidence_report.json").read_text(encoding="utf-8"))
    assert payload["transfer_evidence"]["portable_transfer_case_count"] == 1
    assert payload["transfer_evidence"]["portable_transfer_verdict_counts"] == {"gain": 1}
    assert payload["transfer_evidence"]["portable_transfer_consensus"]["direct"]["consensus"] == "portable_gain_signal"


def _write_summary(
    workspace: Path,
    *,
    seed: int,
    prism_score: float,
    topograph_score: float,
    report_parts: tuple[str, ...] | None = None,
) -> None:
    summary_dir = workspace / "reports" / Path(*report_parts) if report_parts else workspace / "reports" / f"tier_b_core_v2_eval96_seed{seed}"
    summary_dir.mkdir(parents=True, exist_ok=True)
    rows = [
        _row("prism", "openml_gas_sensor", prism_score, seed=seed),
        _row("topograph", "openml_gas_sensor", topograph_score, seed=seed),
        _row("stratograph", "openml_gas_sensor", 0.70, seed=seed),
        _row("primordia", "openml_gas_sensor", 0.72, seed=seed, seeded=True),
        _row("contenders", "openml_gas_sensor", 0.75, seed=seed),
        _row("prism", "wikitext2_lm", 10.0, seed=seed, task_kind="language_modeling", family="language-modeling", direction="min"),
        _row("topograph", "wikitext2_lm", 11.0, seed=seed, task_kind="language_modeling", family="language-modeling", direction="min"),
        _row("stratograph", "wikitext2_lm", 12.0, seed=seed, task_kind="language_modeling", family="language-modeling", direction="min"),
        _row("primordia", "wikitext2_lm", 10.8, seed=seed, task_kind="language_modeling", family="language-modeling", direction="min"),
        _row("contenders", "wikitext2_lm", 10.7, seed=seed, task_kind="language_modeling", family="language-modeling", direction="min"),
    ]
    (summary_dir / "fair_matrix_summary.json").write_text(
        json.dumps(
            {
                "pack_name": "tier_b_core_v2",
                "systems": ["prism", "topograph", "stratograph", "primordia", "contenders"],
                "lane": {
                    "expected_budget": 96,
                    "expected_seed": seed,
                    "operating_state": "trusted-extended",
                    "repeatability_ready": True,
                    "budget_accounting_ok": True,
                },
                "fair_rows": [],
                "reference_rows": [],
                "trend_rows": rows,
            },
            indent=2,
        ),
        encoding="utf-8",
    )


def _row(
    system: str,
    benchmark_id: str,
    value: float,
    *,
    seed: int,
    task_kind: str = "classification",
    family: str = "tabular",
    direction: str = "max",
    seeded: bool = False,
) -> dict[str, object]:
    fairness_metadata: dict[str, object] = {
        "lane_operating_state": "trusted-extended",
        "comparison_label": "topograph-transfer-slice",
        "comparison_cohort": "current-workspace",
        "comparison_case_id": f"tier_b_core_v2:96:{seed}",
    }
    if seeded:
        fairness_metadata.update(
            {
                "seeding_bucket": "direct",
                "seed_source_system": "primordia",
                "seed_source_run_id": f"primordia-seed{seed}",
            }
        )
    return {
        "pack_name": "tier_b_core_v2",
        "budget": 96,
        "seed": seed,
        "system": system,
        "run_id": f"{system}-seed{seed}",
        "benchmark_id": benchmark_id,
        "task_kind": task_kind,
        "benchmark_family": family,
        "metric_name": "perplexity" if task_kind == "language_modeling" else "accuracy",
        "metric_direction": direction,
        "metric_value": value,
        "outcome_status": "ok",
        "lane_operating_state": "trusted-extended",
        "lane_repeatability_ready": True,
        "lane_budget_accounting_ok": True,
        "architecture_summary": "novelty_descriptor=[0.2,0.8]" if system == "topograph" else None,
        "fairness_metadata": fairness_metadata,
    }
