import json
from pathlib import Path

from typer.testing import CliRunner

from evonn_compare.cli import fair_matrix as fair_matrix_cli
from evonn_compare.cli.main import app
from evonn_compare.comparison.fair_matrix import build_matrix_trend_rows
from evonn_compare.comparison.engine import ComparisonEngine
from evonn_compare.contracts.parity import load_parity_pack
from evonn_compare.ingest.loader import SystemIngestor
from evonn_compare.orchestration.lane_presets import resolve_lane_preset
from test_compare import PACK_PATH, _write_run

runner = CliRunner()


def test_root_help() -> None:
    result = runner.invoke(app, ["--help"])
    assert result.exit_code == 0
    assert "Prism" in result.stdout
    assert "Topograph" in result.stdout


def test_validate_help() -> None:
    result = runner.invoke(app, ["validate", "--help"])
    assert result.exit_code == 0
    assert "--pack" in result.stdout


def test_campaign_help() -> None:
    result = runner.invoke(app, ["campaign", "--help"])
    assert result.exit_code == 0
    assert "--workspace" in result.stdout
    assert "smoke" in result.stdout
    assert "local" in result.stdout


def test_fair_matrix_help() -> None:
    result = runner.invoke(app, ["fair-matrix", "--help"])
    assert result.exit_code == 0
    assert "--primordia-root" in result.stdout
    assert "--no-contenders" in result.stdout
    assert "--preset" in result.stdout
    assert "smoke" in result.stdout
    assert "local" in result.stdout


def test_trend_report_help() -> None:
    result = runner.invoke(app, ["trend-report", "--help"])
    assert result.exit_code == 0
    assert "--system" in result.stdout
    assert "--benchmark" in result.stdout
    assert "--output" in result.stdout


def test_hybrid_help() -> None:
    result = runner.invoke(app, ["hybrid", "run", "--help"])
    assert result.exit_code == 0
    assert "--population" in result.stdout


def test_campaign_preset_smoke_dry_run(tmp_path) -> None:
    result = runner.invoke(app, ["campaign", "--preset", "smoke", "--workspace", str(tmp_path), "--dry-run"])
    assert result.exit_code == 0
    assert "tier1_core_smoke_eval16" in result.stdout


def test_campaign_defaults_to_smoke_dry_run(tmp_path) -> None:
    result = runner.invoke(app, ["campaign", "--workspace", str(tmp_path), "--dry-run"])
    assert result.exit_code == 0
    assert "tier1_core_smoke_eval16" in result.stdout


def test_fair_matrix_preset_smoke_dry_run(tmp_path) -> None:
    result = runner.invoke(app, ["fair-matrix", "--preset", "smoke", "--workspace", str(tmp_path), "--dry-run"])
    assert result.exit_code == 0
    assert "tier1_core_smoke_eval16" in result.stdout
    assert "trend-dataset\t" in result.stdout
    assert "fair_matrix_trends.jsonl" in result.stdout


def test_fair_matrix_defaults_to_smoke_dry_run(tmp_path) -> None:
    result = runner.invoke(app, ["fair-matrix", "--workspace", str(tmp_path), "--dry-run"])
    assert result.exit_code == 0
    assert "tier1_core_smoke_eval16" in result.stdout
    assert "trend-dataset\t" in result.stdout
    assert "fair_matrix_trends.jsonl" in result.stdout


def test_fair_matrix_execute_surfaces_manifest_and_trend_dataset(tmp_path, monkeypatch) -> None:
    def fake_run_fair_matrix_case(case, **_kwargs):
        case.summary_output_path.parent.mkdir(parents=True, exist_ok=True)
        case.summary_output_path.write_text("# summary\n", encoding="utf-8")
        return case.summary_output_path

    monkeypatch.setattr("evonn_compare.cli.fair_matrix.run_fair_matrix_case", fake_run_fair_matrix_case)

    result = runner.invoke(app, ["fair-matrix", "--preset", "smoke", "--workspace", str(tmp_path), "--serial"])
    assert result.exit_code == 0
    assert "mode\texecute" in result.stdout
    assert "manifest\t" in result.stdout
    assert "trend-dataset\t" in result.stdout
    assert "summary\t" in result.stdout


def test_fair_matrix_prints_trend_artifact_paths(monkeypatch, tmp_path: Path) -> None:
    summary_path = tmp_path / "reports" / "case" / "fair_matrix_summary.md"

    def fake_prepare_fair_matrix_cases(**_kwargs):
        case = type("Case", (), {"__dict__": {"summary_output_path": summary_path}})()
        paths = type(
            "Paths",
            (),
            {
                "manifest_path": tmp_path / "matrix.yaml",
                "trends_dir": tmp_path / "trends",
            },
        )()
        return paths, [case]

    def fake_run_fair_matrix_case(*_args, **_kwargs):
        return summary_path

    monkeypatch.setattr(fair_matrix_cli, "prepare_fair_matrix_cases", fake_prepare_fair_matrix_cases)
    monkeypatch.setattr(fair_matrix_cli, "run_fair_matrix_case", fake_run_fair_matrix_case)

    result = runner.invoke(app, ["fair-matrix", "--pack", "tier1_core", "--workspace", str(tmp_path)])

    assert result.exit_code == 0
    assert f"summary\t{summary_path}" in result.stdout
    assert f"trend_rows\t{summary_path.parent / 'trend_rows.json'}" in result.stdout
    assert f"trend_report\t{summary_path.parent / 'trend_report.md'}" in result.stdout
    assert f"workspace_trend_rows\t{summary_path.parent.parent / 'fair_matrix_trend_rows.jsonl'}" in result.stdout
    assert f"workspace_trend_report\t{summary_path.parent.parent / 'fair_matrix_trends.md'}" in result.stdout


def test_trend_report_filters_rows_and_writes_outputs(tmp_path: Path) -> None:
    pack = load_parity_pack(PACK_PATH)
    prism_dir = tmp_path / "prism"
    topograph_dir = tmp_path / "topograph"
    _write_run(prism_dir, system="prism", score_shift=0.02)
    _write_run(topograph_dir, system="topograph")

    prism = SystemIngestor(prism_dir)
    topograph = SystemIngestor(topograph_dir)
    runs = {
        "prism": (prism.load_manifest(), prism.load_results()),
        "topograph": (topograph.load_manifest(), topograph.load_results()),
    }
    result = ComparisonEngine().compare(
        left_manifest=runs["prism"][0],
        left_results=runs["prism"][1],
        right_manifest=runs["topograph"][0],
        right_results=runs["topograph"][1],
        pack=pack,
    )
    trend_rows = build_matrix_trend_rows(
        pack=pack,
        budget=64,
        seed=42,
        runs=runs,
        pair_results={("prism", "topograph"): (result, Path("prism_vs_topograph.md"))},
        systems=("prism", "topograph"),
    )
    trend_path = tmp_path / "trend_rows.jsonl"
    trend_path.write_text(
        "".join(json.dumps(row.__dict__, default=str) + "\n" for row in trend_rows),
        encoding="utf-8",
    )
    output_path = tmp_path / "trend_report.md"

    cli_result = runner.invoke(
        app,
        [
            "trend-report",
            str(trend_path),
            "--system",
            "prism",
            "--benchmark",
            "iris_classification",
            "--output",
            str(output_path),
        ],
    )

    assert cli_result.exit_code == 0
    assert output_path.exists()
    assert output_path.with_suffix(".json").exists()
    markdown = output_path.read_text(encoding="utf-8")
    assert "# Fair Matrix Trends: tier1_core" in markdown
    assert "- Systems: `prism`" in markdown
    assert "| prism | iris_classification | 1 | 0.820000 |" in markdown


def test_trend_report_accepts_structured_fair_matrix_trends_jsonl(tmp_path: Path) -> None:
    trend_path = tmp_path / "fair_matrix_trends.jsonl"
    trend_path.write_text(
        "".join(
            json.dumps(record) + "\n"
            for record in [
                {
                    "pack": "tier1_core_smoke",
                    "benchmark": "iris_classification",
                    "task_kind": "classification",
                    "engine": "prism",
                    "run_id": "prism-run",
                    "run_name": "prism-run",
                    "created_at": "2026-04-01T00:00:00+00:00",
                    "seed": 42,
                    "budget": 16,
                    "outcome_status": "ok",
                    "metric_name": "accuracy",
                    "metric_direction": "max",
                    "metric_value": 0.81,
                    "quality": 0.81,
                    "failure_reason": None,
                    "fairness": {
                        "benchmark_pack_id": "tier1_core_smoke",
                        "seed": 42,
                        "evaluation_count": 16,
                        "budget_policy_name": "evolutionary_search",
                        "data_signature": "shared-signature",
                        "code_version": "deadbeef",
                    },
                    "artifact_paths": {
                        "manifest": "prism/manifest.json",
                        "results": "prism/results.json",
                        "summary": "prism/summary.json",
                        "report": "prism/report.md",
                    },
                }
            ]
        ),
        encoding="utf-8",
    )

    result = runner.invoke(app, ["trend-report", str(trend_path), "--system", "prism"])

    assert result.exit_code == 0
    assert "# Fair Matrix Trends: tier1_core_smoke" in result.stdout
    assert "- Systems: `prism`" in result.stdout
    assert "| prism | iris_classification | 1 | 0.810000 | 0.810000 | 0.000000 | ok | 16 | 42 | fair |" in result.stdout


def test_resolve_lane_preset_rejects_unknown_name() -> None:
    try:
        resolve_lane_preset("unknown")
    except ValueError as exc:
        assert "available" in str(exc)
    else:
        raise AssertionError("expected ValueError")
