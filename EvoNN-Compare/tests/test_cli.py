import json
import re
from pathlib import Path

from typer.testing import CliRunner

from evonn_compare.cli import fair_matrix as fair_matrix_cli
from evonn_compare.cli import seeded_compare as seeded_compare_cli
from evonn_compare.cli.main import app
from evonn_compare.comparison.fair_matrix import build_matrix_trend_rows
from evonn_compare.comparison.engine import ComparisonEngine
from evonn_compare.contracts.parity import load_parity_pack
from evonn_compare.ingest.loader import SystemIngestor
from evonn_compare.orchestration.lane_presets import resolve_lane_preset
from test_compare import PACK_PATH, _write_run

runner = CliRunner()


_ANSI_RE = re.compile(r"\x1b\[[0-9;]*[A-Za-z]")


def _normalized_cli_output(text: str) -> str:
    text = _ANSI_RE.sub("", text)
    return re.sub(r"\s+", " ", text)


def _invoke_help(*args: str):
    return runner.invoke(app, [*args, "--help"], color=False)


def test_root_help() -> None:
    result = runner.invoke(app, ["--help"])
    assert result.exit_code == 0
    text = _normalized_cli_output(result.stdout)
    assert "Prism" in text
    assert "Topograph" in text


def test_validate_help() -> None:
    result = _invoke_help("validate")
    assert result.exit_code == 0
    assert "--pack" in _normalized_cli_output(result.stdout)


def test_campaign_help() -> None:
    result = _invoke_help("campaign")
    assert result.exit_code == 0
    text = _normalized_cli_output(result.stdout)
    assert "--workspace" in text
    assert "smoke" in text
    assert "local" in text
    assert "overnight" in text
    assert "weekend" in text


def test_fair_matrix_help() -> None:
    result = _invoke_help("fair-matrix")
    assert result.exit_code == 0
    text = _normalized_cli_output(result.stdout)
    assert "--primordia-root" in text
    assert "--no-contenders" in text
    assert "--preset" in text
    assert "--no-open" in text
    assert "smoke" in text
    assert "local" in text
    assert "overnight" in text
    assert "weekend" in text


def test_trend_report_help() -> None:
    result = _invoke_help("trend-report")
    assert result.exit_code == 0
    text = _normalized_cli_output(result.stdout)
    assert "--system" in text
    assert "--benchmark" in text
    assert "--output" in text


def test_dashboard_help() -> None:
    result = _invoke_help("dashboard")
    assert result.exit_code == 0
    text = _normalized_cli_output(result.stdout)
    assert "--output" in text
    assert "manual_compare_runs" in text
    assert "--no-open" in text


def test_workspace_report_help() -> None:
    result = _invoke_help("workspace-report")
    assert result.exit_code == 0
    text = _normalized_cli_output(result.stdout)
    assert "Fair-matrix workspace root" in text
    assert "--dashboard-output" in text
    assert "--trend-output" in text


def test_seeded_compare_help() -> None:
    result = _invoke_help("seeded-compare")
    assert result.exit_code == 0
    text = _normalized_cli_output(result.stdout)
    assert "--workspace" in text
    assert "--pack" in text
    assert "--primordia-root" in text
    assert "--topograph-root" in text
    assert "--no-open" in text


def test_hybrid_help() -> None:
    result = _invoke_help("hybrid", "run")
    assert result.exit_code == 0
    assert "--population" in _normalized_cli_output(result.stdout)


def test_campaign_preset_smoke_dry_run(tmp_path) -> None:
    result = runner.invoke(app, ["campaign", "--preset", "smoke", "--workspace", str(tmp_path), "--dry-run"])
    assert result.exit_code == 0
    assert "manifest\t" in result.stdout
    assert "report\t" in result.stdout
    assert "report_json\t" in result.stdout
    assert "prism_run_dir\t" in result.stdout
    assert "topograph_run_dir\t" in result.stdout
    assert "log_dir\t" in result.stdout
    assert "tier1_core_smoke_eval16" in result.stdout


def test_campaign_defaults_to_smoke_dry_run(tmp_path) -> None:
    result = runner.invoke(app, ["campaign", "--workspace", str(tmp_path), "--dry-run"])
    assert result.exit_code == 0
    assert "manifest\t" in result.stdout
    assert "report\t" in result.stdout
    assert "report_json\t" in result.stdout
    assert "prism_run_dir\t" in result.stdout
    assert "topograph_run_dir\t" in result.stdout
    assert "log_dir\t" in result.stdout
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


def test_campaign_preset_overnight_dry_run(tmp_path) -> None:
    result = runner.invoke(app, ["campaign", "--preset", "overnight", "--workspace", str(tmp_path), "--dry-run"])
    assert result.exit_code == 0
    assert "tier1_core_eval256" in result.stdout


def test_fair_matrix_preset_weekend_dry_run(tmp_path) -> None:
    result = runner.invoke(app, ["fair-matrix", "--preset", "weekend", "--workspace", str(tmp_path), "--dry-run"])
    assert result.exit_code == 0
    assert "tier1_core_eval1000" in result.stdout
    assert "trend-dataset\t" in result.stdout


def test_campaign_execute_surfaces_manifest(tmp_path, monkeypatch) -> None:
    class FakeRunner:
        def __init__(self, **_kwargs):
            pass

        def prism_run_dir(self, case):
            return Path("/tmp") / case.prism_config_path.stem

        def execution_commands(self, _case):
            return []

        def execution_stages(self, _case):
            return []

        def compare_exports(self, **_kwargs):
            return None

    monkeypatch.setattr("evonn_compare.cli.campaign.CampaignRunner", FakeRunner)

    result = runner.invoke(app, ["campaign", "--preset", "smoke", "--workspace", str(tmp_path), "--serial"])
    assert result.exit_code == 0
    assert "mode\texecute" in result.stdout
    assert "manifest\t" in result.stdout
    assert "compared\t" in result.stdout
    assert "report\t" in result.stdout
    assert "report_json\t" in result.stdout
    assert "prism_run_dir\t" in result.stdout
    assert "topograph_run_dir\t" in result.stdout
    assert "log_dir\t" in result.stdout


def test_fair_matrix_execute_surfaces_manifest_and_trend_dataset(tmp_path, monkeypatch) -> None:
    def fake_run_fair_matrix_case(case, **_kwargs):
        case.summary_output_path.parent.mkdir(parents=True, exist_ok=True)
        case.summary_output_path.write_text("# summary\n", encoding="utf-8")
        return case.summary_output_path

    def fake_refresh_workspace_reports(*, workspace, open_browser=False):
        return {
            "workspace": str(workspace),
            "summary_count": 1,
            "trend_dataset": str(Path(workspace) / "trends" / "fair_matrix_trend_rows.jsonl"),
            "trend_report": str(Path(workspace) / "trends" / "fair_matrix_trends.md"),
            "trend_report_data": str(Path(workspace) / "trends" / "fair_matrix_trends.json"),
            "dashboard": str(Path(workspace) / "fair_matrix_dashboard.html"),
            "dashboard_data": str(Path(workspace) / "fair_matrix_dashboard.json"),
        }

    monkeypatch.setattr("evonn_compare.cli.fair_matrix.run_fair_matrix_case", fake_run_fair_matrix_case)
    monkeypatch.setattr("evonn_compare.cli.fair_matrix.refresh_workspace_reports", fake_refresh_workspace_reports)

    result = runner.invoke(app, ["fair-matrix", "--preset", "smoke", "--workspace", str(tmp_path), "--serial"])
    assert result.exit_code == 0
    assert "mode\texecute" in result.stdout
    assert "manifest\t" in result.stdout
    assert "trend-dataset\t" in result.stdout
    assert "summary\t" in result.stdout
    assert "workspace_trend_report\t" in result.stdout
    assert "workspace_trend_report_data\t" in result.stdout
    assert "workspace_dashboard\t" in result.stdout
    assert "workspace_dashboard_data\t" in result.stdout


def test_fair_matrix_open_flag_surfaces_opened_dashboard_uri(tmp_path, monkeypatch) -> None:
    captured: dict[str, object] = {}

    def fake_run_fair_matrix_case(case, **_kwargs):
        case.summary_output_path.parent.mkdir(parents=True, exist_ok=True)
        case.summary_output_path.write_text("# summary\n", encoding="utf-8")
        return case.summary_output_path

    def fake_refresh_workspace_reports(*, workspace, open_browser=False):
        captured["open_browser"] = open_browser
        return {
            "workspace": str(workspace),
            "summary_count": 1,
            "trend_dataset": str(Path(workspace) / "trends" / "fair_matrix_trend_rows.jsonl"),
            "trend_report": str(Path(workspace) / "trends" / "fair_matrix_trends.md"),
            "trend_report_data": str(Path(workspace) / "trends" / "fair_matrix_trends.json"),
            "dashboard": str(Path(workspace) / "fair_matrix_dashboard.html"),
            "dashboard_data": str(Path(workspace) / "fair_matrix_dashboard.json"),
        }

    monkeypatch.setattr("evonn_compare.cli.fair_matrix.run_fair_matrix_case", fake_run_fair_matrix_case)
    monkeypatch.setattr("evonn_compare.cli.fair_matrix.refresh_workspace_reports", fake_refresh_workspace_reports)

    result = runner.invoke(app, ["fair-matrix", "--preset", "smoke", "--workspace", str(tmp_path), "--serial", "--open"])

    assert result.exit_code == 0
    assert captured["open_browser"] is True
    assert f"opened\t{(tmp_path / 'fair_matrix_dashboard.html').resolve().as_uri()}" in result.stdout


def test_seeded_compare_surfaces_workspace_artifacts(tmp_path, monkeypatch) -> None:
    def fake_publish_seeded_vs_unseeded_workspace(**kwargs):
        workspace = Path(kwargs["workspace"])
        return {
            "workspace": str(workspace),
            "pack_path": str(workspace / "packs" / "tier1_core_smoke_eval16.yaml"),
            "primordia_run_dir": str(workspace / "runs" / "primordia" / "source"),
            "seed_artifact": str(workspace / "runs" / "primordia" / "source" / "seed_candidates.json"),
            "unseeded_run_dir": str(workspace / "runs" / "topograph" / "01-unseeded"),
            "seeded_run_dir": str(workspace / "runs" / "topograph" / "02-seeded"),
            "seeded_vs_unseeded_report": str(workspace / "reports" / "seeded_vs_unseeded_summary.md"),
            "seeded_vs_unseeded_data": str(workspace / "reports" / "seeded_vs_unseeded_summary.json"),
            "trend_dataset": str(workspace / "trends" / "fair_matrix_trend_rows.jsonl"),
            "trend_report": str(workspace / "trends" / "fair_matrix_trends.md"),
            "trend_report_data": str(workspace / "trends" / "fair_matrix_trends.json"),
            "dashboard": str(workspace / "fair_matrix_dashboard.html"),
            "dashboard_data": str(workspace / "fair_matrix_dashboard.json"),
        }

    monkeypatch.setattr(
        "evonn_compare.cli.seeded_compare.publish_seeded_vs_unseeded_workspace",
        fake_publish_seeded_vs_unseeded_workspace,
    )

    result = runner.invoke(app, ["seeded-compare", "--workspace", str(tmp_path)])

    assert result.exit_code == 0
    assert f"workspace\t{tmp_path}" in result.stdout
    assert f"seed_artifact\t{tmp_path / 'runs' / 'primordia' / 'source' / 'seed_candidates.json'}" in result.stdout
    assert f"dashboard\t{tmp_path / 'fair_matrix_dashboard.html'}" in result.stdout


def test_seeded_compare_open_flag_opens_dashboard(tmp_path, monkeypatch) -> None:
    opened: list[str] = []

    def fake_publish_seeded_vs_unseeded_workspace(**kwargs):
        workspace = Path(kwargs["workspace"])
        return {
            "workspace": str(workspace),
            "pack_path": str(workspace / "packs" / "tier1_core_smoke_eval16.yaml"),
            "primordia_run_dir": str(workspace / "runs" / "primordia" / "source"),
            "seed_artifact": str(workspace / "runs" / "primordia" / "source" / "seed_candidates.json"),
            "unseeded_run_dir": str(workspace / "runs" / "topograph" / "01-unseeded"),
            "seeded_run_dir": str(workspace / "runs" / "topograph" / "02-seeded"),
            "seeded_vs_unseeded_report": str(workspace / "reports" / "seeded_vs_unseeded_summary.md"),
            "seeded_vs_unseeded_data": str(workspace / "reports" / "seeded_vs_unseeded_summary.json"),
            "trend_dataset": str(workspace / "trends" / "fair_matrix_trend_rows.jsonl"),
            "trend_report": str(workspace / "trends" / "fair_matrix_trends.md"),
            "trend_report_data": str(workspace / "trends" / "fair_matrix_trends.json"),
            "dashboard": str(workspace / "fair_matrix_dashboard.html"),
            "dashboard_data": str(workspace / "fair_matrix_dashboard.json"),
        }

    monkeypatch.setattr(
        "evonn_compare.cli.seeded_compare.publish_seeded_vs_unseeded_workspace",
        fake_publish_seeded_vs_unseeded_workspace,
    )
    monkeypatch.setattr(seeded_compare_cli.webbrowser, "open", lambda url: opened.append(url) or True)

    result = runner.invoke(app, ["seeded-compare", "--workspace", str(tmp_path), "--open"])

    assert result.exit_code == 0
    expected_url = (tmp_path / "fair_matrix_dashboard.html").resolve().as_uri()
    assert opened == [expected_url]
    assert f"opened\t{expected_url}" in result.stdout


def test_fair_matrix_execute_resets_managed_workspace_before_prepare(tmp_path, monkeypatch) -> None:
    workspace = tmp_path / "workspace"
    stale_report = workspace / "reports" / "stale" / "fair_matrix_summary.json"
    stale_dataset = workspace / "trends" / "fair_matrix_trend_rows.jsonl"
    stale_dashboard = workspace / "fair_matrix_dashboard.html"
    for artifact in (stale_report, stale_dataset, stale_dashboard):
        artifact.parent.mkdir(parents=True, exist_ok=True)
        artifact.write_text("stale\n", encoding="utf-8")

    def fake_prepare_fair_matrix_cases(**kwargs):
        assert not stale_report.exists()
        assert not stale_dataset.exists()
        assert not stale_dashboard.exists()
        workspace_path = Path(kwargs["workspace"])
        case_dir = workspace_path / "reports" / "tier1_core_eval64_seed42"
        case = type(
            "Case",
            (),
            {
                "summary_output_path": case_dir / "fair_matrix_summary.md",
            },
        )()
        paths = type(
            "Paths",
            (),
            {
                "manifest_path": workspace_path / "matrix.yaml",
                "trends_dir": workspace_path / "trends",
            },
        )()
        paths.manifest_path.parent.mkdir(parents=True, exist_ok=True)
        paths.manifest_path.write_text("pack_name: tier1_core\n", encoding="utf-8")
        return paths, [case]

    def fake_run_fair_matrix_case(case, **_kwargs):
        case.summary_output_path.parent.mkdir(parents=True, exist_ok=True)
        case.summary_output_path.write_text("# summary\n", encoding="utf-8")
        return case.summary_output_path

    def fake_refresh_workspace_reports(*, workspace, open_browser=False):
        return {
            "workspace": str(workspace),
            "summary_count": 1,
            "trend_dataset": str(Path(workspace) / "trends" / "fair_matrix_trend_rows.jsonl"),
            "trend_report": str(Path(workspace) / "trends" / "fair_matrix_trends.md"),
            "trend_report_data": str(Path(workspace) / "trends" / "fair_matrix_trends.json"),
            "dashboard": str(Path(workspace) / "fair_matrix_dashboard.html"),
            "dashboard_data": str(Path(workspace) / "fair_matrix_dashboard.json"),
        }

    monkeypatch.setattr(fair_matrix_cli, "prepare_fair_matrix_cases", fake_prepare_fair_matrix_cases)
    monkeypatch.setattr(fair_matrix_cli, "run_fair_matrix_case", fake_run_fair_matrix_case)
    monkeypatch.setattr(fair_matrix_cli, "refresh_workspace_reports", fake_refresh_workspace_reports)

    result = runner.invoke(app, ["fair-matrix", "--preset", "smoke", "--workspace", str(workspace), "--serial"])

    assert result.exit_code == 0
    assert "mode\texecute" in result.stdout
    assert not stale_report.exists()
    assert not stale_dataset.exists()
    assert not stale_dashboard.exists()


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

    def fake_refresh_workspace_reports(*, workspace, open_browser=False):
        return {
            "workspace": str(workspace),
            "summary_count": 1,
            "trend_dataset": str(Path(workspace) / "trends" / "fair_matrix_trend_rows.jsonl"),
            "trend_report": str(Path(workspace) / "trends" / "fair_matrix_trends.md"),
            "trend_report_data": str(Path(workspace) / "trends" / "fair_matrix_trends.json"),
            "dashboard": str(Path(workspace) / "fair_matrix_dashboard.html"),
            "dashboard_data": str(Path(workspace) / "fair_matrix_dashboard.json"),
        }

    monkeypatch.setattr(fair_matrix_cli, "prepare_fair_matrix_cases", fake_prepare_fair_matrix_cases)
    monkeypatch.setattr(fair_matrix_cli, "run_fair_matrix_case", fake_run_fair_matrix_case)
    monkeypatch.setattr(fair_matrix_cli, "refresh_workspace_reports", fake_refresh_workspace_reports)

    result = runner.invoke(app, ["fair-matrix", "--pack", "tier1_core", "--workspace", str(tmp_path)])

    assert result.exit_code == 0
    assert f"summary\t{summary_path}" in result.stdout
    assert f"summary_json\t{summary_path.with_suffix('.json')}" in result.stdout
    assert f"lane_acceptance\t{summary_path.parent / 'lane_acceptance.json'}" in result.stdout
    assert f"trend_rows\t{summary_path.parent / 'trend_rows.json'}" in result.stdout
    assert f"trend_report\t{summary_path.parent / 'trend_report.md'}" in result.stdout
    assert f"trend_records_json\t{summary_path.parent / 'fair_matrix_trends.json'}" in result.stdout
    assert f"trend_records_jsonl\t{summary_path.parent / 'fair_matrix_trends.jsonl'}" in result.stdout
    assert f"workspace_trend_rows\t{summary_path.parent.parent / 'fair_matrix_trend_rows.jsonl'}" in result.stdout
    assert f"workspace_trend_report\t{summary_path.parent.parent / 'fair_matrix_trends.md'}" in result.stdout
    assert f"workspace_trend_report_data\t{tmp_path / 'trends' / 'fair_matrix_trends.json'}" in result.stdout
    assert f"workspace_dashboard\t{tmp_path / 'fair_matrix_dashboard.html'}" in result.stdout
    assert f"workspace_dashboard_data\t{tmp_path / 'fair_matrix_dashboard.json'}" in result.stdout


def test_fair_matrix_default_smoke_persists_lane_preset(monkeypatch, tmp_path: Path) -> None:
    captured: dict[str, object] = {}

    def fake_prepare_fair_matrix_cases(**kwargs):
        captured.update(kwargs)
        return type(
            "Paths",
            (),
            {
                "manifest_path": tmp_path / "matrix.yaml",
                "trends_dir": tmp_path / "trends",
            },
        )(), []

    monkeypatch.setattr(fair_matrix_cli, "prepare_fair_matrix_cases", fake_prepare_fair_matrix_cases)

    result = runner.invoke(app, ["fair-matrix", "--workspace", str(tmp_path), "--dry-run"])

    assert result.exit_code == 0
    assert captured["lane_preset"] == "smoke"


def test_fair_matrix_pack_without_preset_defaults_to_pack_budget(monkeypatch, tmp_path: Path) -> None:
    captured: dict[str, object] = {}

    def fake_prepare_fair_matrix_cases(**kwargs):
        captured.update(kwargs)
        return type(
            "Paths",
            (),
            {
                "manifest_path": tmp_path / "matrix.yaml",
                "trends_dir": tmp_path / "trends",
            },
        )(), []

    monkeypatch.setattr(fair_matrix_cli, "prepare_fair_matrix_cases", fake_prepare_fair_matrix_cases)

    result = runner.invoke(app, ["fair-matrix", "--pack", "tier1_core_smoke", "--workspace", str(tmp_path), "--dry-run"])

    assert result.exit_code == 0
    assert captured["lane_preset"] is None
    assert captured["budgets"] == [16]


def test_campaign_default_smoke_persists_lane_preset(monkeypatch, tmp_path: Path) -> None:
    captured: dict[str, object] = {}

    def fake_prepare_campaign_cases(**kwargs):
        captured.update(kwargs)
        return type(
            "Paths",
            (),
            {
                "manifest_path": tmp_path / "campaign.yaml",
                "logs_dir": tmp_path / "logs",
            },
        )(), []

    monkeypatch.setattr("evonn_compare.cli.campaign.prepare_campaign_cases", fake_prepare_campaign_cases)

    result = runner.invoke(app, ["campaign", "--workspace", str(tmp_path), "--dry-run"])

    assert result.exit_code == 0
    assert captured["lane_preset"] == "smoke"


def test_campaign_pack_without_preset_defaults_to_pack_budget(monkeypatch, tmp_path: Path) -> None:
    captured: dict[str, object] = {}

    def fake_prepare_campaign_cases(**kwargs):
        captured.update(kwargs)
        return type(
            "Paths",
            (),
            {
                "manifest_path": tmp_path / "campaign.yaml",
                "logs_dir": tmp_path / "logs",
            },
        )(), []

    monkeypatch.setattr("evonn_compare.cli.campaign.prepare_campaign_cases", fake_prepare_campaign_cases)

    result = runner.invoke(app, ["campaign", "--pack", "tier1_core_smoke", "--workspace", str(tmp_path), "--dry-run"])

    assert result.exit_code == 0
    assert captured["lane_preset"] is None
    assert captured["budgets"] == [16]


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
    assert f"report\t{output_path}" in cli_result.stdout
    assert f"report_json\t{output_path.with_suffix('.json')}" in cli_result.stdout
    markdown = output_path.read_text(encoding="utf-8")
    assert "# Fair Matrix Trends: tier1_core" in markdown
    assert "- Systems: `prism`" in markdown
    assert "- Budget Accounting: `incomplete`" in markdown
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
    assert "- Budget Accounting: `incomplete`" in result.stdout
    assert "| prism | iris_classification | 1 | 0.810000 | 0.810000 | 0.000000 | ok | 16 | 42 | fair | transfer-opaque | --- | fair | incomplete | not-ready | unknown |" in result.stdout


def test_compare_output_surfaces_report_paths(tmp_path: Path) -> None:
    prism_dir = tmp_path / "prism"
    topograph_dir = tmp_path / "topograph"
    _write_run(prism_dir, system="prism", score_shift=0.02)
    _write_run(topograph_dir, system="topograph")
    output_path = tmp_path / "compare_report.md"

    result = runner.invoke(
        app,
        [
            "compare",
            str(prism_dir),
            str(topograph_dir),
            "--pack",
            str(PACK_PATH),
            "--output",
            str(output_path),
        ],
    )

    assert result.exit_code == 0
    assert output_path.exists()
    assert output_path.with_suffix(".json").exists()
    assert f"report\t{output_path}" in result.stdout
    assert f"report_json\t{output_path.with_suffix('.json')}" in result.stdout


def test_workspace_report_refreshes_trend_and_dashboard_outputs(tmp_path: Path) -> None:
    workspace = tmp_path / "workspace"
    trends_dir = workspace / "trends"
    reports_dir = workspace / "reports" / "tier1_core_eval64_seed42"
    trends_dir.mkdir(parents=True)
    reports_dir.mkdir(parents=True)
    (trends_dir / "fair_matrix_trend_rows.jsonl").write_text(
        json.dumps(_dashboard_row("prism", "iris_classification", 0.8)) + "\n",
        encoding="utf-8",
    )
    (reports_dir / "fair_matrix_summary.json").write_text(
        json.dumps(
            {
                "pack_name": "tier1_core_eval64",
                "systems": ["prism"],
                "lane": {
                    "preset": None,
                    "pack_name": "tier1_core_eval64",
                    "expected_budget": 64,
                    "expected_seed": 42,
                    "operating_state": "contract-fair",
                    "artifact_completeness_ok": True,
                    "fairness_ok": True,
                    "task_coverage_ok": True,
                    "budget_consistency_ok": True,
                    "seed_consistency_ok": True,
                    "budget_accounting_ok": True,
                    "core_systems_complete_ok": False,
                    "extended_systems_complete_ok": False,
                    "observed_task_kinds": ["classification"],
                    "system_operating_states": {"prism": "benchmark-complete"},
                    "acceptance_notes": [],
                    "repeatability_ready": False,
                },
                "fair_rows": [],
                "reference_rows": [],
                "parity_rows": [],
                "trend_rows": [_dashboard_row("prism", "iris_classification", 0.8)],
            },
            indent=2,
        ),
        encoding="utf-8",
    )

    result = runner.invoke(app, ["workspace-report", str(workspace)])

    assert result.exit_code == 0
    assert f"trend-report\t{workspace / 'trends' / 'fair_matrix_trends.md'}" in result.stdout
    assert f"trend-report-data\t{workspace / 'trends' / 'fair_matrix_trends.json'}" in result.stdout
    assert f"dashboard\t{workspace / 'fair_matrix_dashboard.html'}" in result.stdout
    assert (workspace / "trends" / "fair_matrix_trends.md").exists()
    assert (workspace / "trends" / "fair_matrix_trends.json").exists()
    assert (workspace / "fair_matrix_dashboard.html").exists()
    assert "Lane States" in (workspace / "trends" / "fair_matrix_trends.md").read_text(encoding="utf-8")


def test_workspace_report_normalizes_legacy_root_dataset_and_dedupes_rows(tmp_path: Path) -> None:
    workspace = tmp_path / "workspace"
    reports_dir = workspace / "reports" / "tier1_core_eval64_seed42"
    reports_dir.mkdir(parents=True)
    row = _dashboard_row("prism", "iris_classification", 0.8)
    (workspace / "fair_matrix_trend_rows.jsonl").write_text(
        json.dumps(row) + "\n" + json.dumps(row) + "\n",
        encoding="utf-8",
    )
    (reports_dir / "fair_matrix_summary.json").write_text(
        json.dumps(
            {
                "pack_name": "tier1_core_eval64",
                "systems": ["prism"],
                "lane": {
                    "preset": None,
                    "pack_name": "tier1_core_eval64",
                    "expected_budget": 64,
                    "expected_seed": 42,
                    "operating_state": "contract-fair",
                    "artifact_completeness_ok": True,
                    "fairness_ok": True,
                    "task_coverage_ok": True,
                    "budget_consistency_ok": True,
                    "seed_consistency_ok": True,
                    "budget_accounting_ok": True,
                    "core_systems_complete_ok": False,
                    "extended_systems_complete_ok": False,
                    "observed_task_kinds": ["classification"],
                    "system_operating_states": {"prism": "benchmark-complete"},
                    "acceptance_notes": [],
                    "repeatability_ready": False,
                },
                "fair_rows": [],
                "reference_rows": [],
                "parity_rows": [],
                "trend_rows": [row],
            },
            indent=2,
        ),
        encoding="utf-8",
    )

    result = runner.invoke(app, ["workspace-report", str(workspace)])

    assert result.exit_code == 0
    canonical_dataset = workspace / "trends" / "fair_matrix_trend_rows.jsonl"
    assert canonical_dataset.exists()
    lines = [line for line in canonical_dataset.read_text(encoding="utf-8").splitlines() if line.strip()]
    assert len(lines) == 1
    assert f"trend-dataset\t{canonical_dataset}" in result.stdout


def test_dashboard_recomputes_project_only_winners(tmp_path: Path) -> None:
    summary_dir = tmp_path / "workspace" / "reports" / "tier1_core_eval64_seed42"
    summary_dir.mkdir(parents=True)
    summary_path = summary_dir / "fair_matrix_summary.json"
    summary_path.write_text(
        json.dumps(
            {
                "pack_name": "tier1_core_eval64",
                "systems": ["prism", "topograph", "stratograph", "primordia", "contenders"],
                "lane": {
                    "preset": None,
                    "pack_name": "tier1_core_eval64",
                    "expected_budget": 64,
                    "expected_seed": 42,
                    "operating_state": "trusted-extended",
                    "artifact_completeness_ok": True,
                    "fairness_ok": True,
                    "task_coverage_ok": True,
                    "budget_consistency_ok": True,
                    "seed_consistency_ok": True,
                    "budget_accounting_ok": True,
                    "core_systems_complete_ok": True,
                    "extended_systems_complete_ok": True,
                    "observed_task_kinds": ["classification"],
                    "system_operating_states": {
                        "prism": "benchmark-complete",
                        "topograph": "benchmark-complete",
                        "stratograph": "benchmark-complete",
                        "primordia": "benchmark-complete",
                        "contenders": "benchmark-complete",
                    },
                    "acceptance_notes": [],
                    "repeatability_ready": True,
                },
                "fair_rows": [],
                "reference_rows": [],
                "parity_rows": [],
                "trend_rows": [
                    _dashboard_row("prism", "iris_classification", 0.80),
                    _dashboard_row("topograph", "iris_classification", 0.90),
                    _dashboard_row("stratograph", "iris_classification", 0.70),
                    _dashboard_row("primordia", "iris_classification", 0.60),
                    _dashboard_row("contenders", "iris_classification", 0.95),
                    _dashboard_row("prism", "wine_classification", 0.85),
                    _dashboard_row("topograph", "wine_classification", 0.83),
                    _dashboard_row("stratograph", "wine_classification", None, outcome_status="missing"),
                    _dashboard_row("primordia", "wine_classification", None, outcome_status="failed", failure_reason="boom"),
                    _dashboard_row("contenders", "wine_classification", 0.84),
                ],
            },
            indent=2,
        ),
        encoding="utf-8",
    )
    output_path = tmp_path / "dashboard.html"

    result = runner.invoke(app, ["dashboard", str(tmp_path / "workspace"), "--output", str(output_path)])

    assert result.exit_code == 0
    assert output_path.exists()
    payload = json.loads(output_path.with_suffix(".json").read_text(encoding="utf-8"))
    run = payload["runs"][0]
    all_scope = {entry["system"]: entry for entry in run["all_scope"]["rows"]}
    project_scope = {entry["system"]: entry for entry in run["project_scope"]["rows"]}
    assert all_scope["contenders"]["solo_wins"] == 1
    assert all_scope["topograph"]["solo_wins"] == 0
    assert project_scope["topograph"]["solo_wins"] == 1
    assert project_scope["prism"]["solo_wins"] == 1
    assert run["lane"]["operating_state"] == "trusted-extended"
    assert run["project_scope"]["skipped"] == 0
    html = output_path.read_text(encoding="utf-8")
    assert "Overall Leaderboard: Projects Only" in html
    assert "Trusted Extended" in html


def test_dashboard_payload_surfaces_run_level_seeding_metadata(tmp_path: Path) -> None:
    summary_dir = tmp_path / "workspace" / "reports" / "tier1_core_eval64_seed42"
    summary_dir.mkdir(parents=True)
    (summary_dir / "fair_matrix_summary.json").write_text(
        json.dumps(
            {
                "pack_name": "tier1_core_eval64",
                "systems": ["primordia", "topograph"],
                "lane": {
                    "preset": "seeded-compare",
                    "pack_name": "tier1_core_eval64",
                    "expected_budget": 64,
                    "expected_seed": 42,
                    "operating_state": "contract-fair",
                    "artifact_completeness_ok": True,
                    "fairness_ok": True,
                    "task_coverage_ok": True,
                    "budget_consistency_ok": True,
                    "seed_consistency_ok": True,
                    "budget_accounting_ok": True,
                    "core_systems_complete_ok": True,
                    "extended_systems_complete_ok": True,
                    "observed_task_kinds": ["classification", "regression"],
                    "system_operating_states": {
                        "primordia": "benchmark-complete",
                        "topograph": "portable-smoke",
                    },
                    "acceptance_notes": [],
                    "repeatability_ready": True,
                },
                "fair_rows": [{"budget": 64, "seed": 42, "benchmark_count": 1, "evaluation_counts": {"primordia": 64, "topograph": 64}, "wins": {"primordia": 0, "topograph": 1}, "ties": 0, "note": None}],
                "reference_rows": [],
                "parity_rows": [],
                "trend_rows": [
                    _dashboard_row("primordia", "iris_classification", 0.75),
                    _dashboard_row(
                        "topograph",
                        "iris_classification",
                        0.80,
                        seeding_bucket="direct",
                        seed_source_system="primordia",
                        seed_source_run_id="prim-run-7",
                        seed_artifact_path="/tmp/seed_candidates.json",
                        seed_selected_family="mlp",
                    ),
                ],
            },
            indent=2,
        ),
        encoding="utf-8",
    )
    output_path = tmp_path / "dashboard.html"

    result = runner.invoke(app, ["dashboard", str(tmp_path / "workspace"), "--output", str(output_path)])

    assert result.exit_code == 0
    payload = json.loads(output_path.with_suffix(".json").read_text(encoding="utf-8"))
    run = payload["runs"][0]
    assert run["system_seeding"]["topograph"]["mode"] == "direct"
    assert run["system_seeding"]["topograph"]["source"] == "primordia:prim-run-7"
    assert run["system_seeding"]["topograph"]["artifact"] == "/tmp/seed_candidates.json"
    html = output_path.read_text(encoding="utf-8")
    assert "Topograph: direct from primordia:prim-run-7 (mlp)" in html


def test_dashboard_opens_browser_by_default(tmp_path: Path, monkeypatch) -> None:
    summary_dir = tmp_path / "workspace" / "reports" / "tier1_core_eval64_seed42"
    summary_dir.mkdir(parents=True)
    (summary_dir / "fair_matrix_summary.json").write_text(
        json.dumps(
            {
                "pack_name": "tier1_core_eval64",
                "systems": ["prism"],
                "lane": {
                    "preset": None,
                    "pack_name": "tier1_core_eval64",
                    "expected_budget": 64,
                    "expected_seed": 42,
                    "operating_state": "contract-fair",
                    "artifact_completeness_ok": True,
                    "fairness_ok": True,
                    "task_coverage_ok": True,
                    "budget_consistency_ok": True,
                    "seed_consistency_ok": True,
                    "budget_accounting_ok": True,
                    "core_systems_complete_ok": False,
                    "extended_systems_complete_ok": False,
                    "observed_task_kinds": ["classification"],
                    "system_operating_states": {"prism": "benchmark-complete"},
                    "acceptance_notes": [],
                    "repeatability_ready": False,
                },
                "fair_rows": [],
                "reference_rows": [],
                "parity_rows": [],
                "trend_rows": [_dashboard_row("prism", "iris_classification", 0.8)],
            },
            indent=2,
        ),
        encoding="utf-8",
    )
    output_path = tmp_path / "dashboard.html"
    opened: list[str] = []
    monkeypatch.setattr("evonn_compare.cli.dashboard.webbrowser.open", lambda url: opened.append(url) or True)

    result = runner.invoke(app, ["dashboard", str(tmp_path / "workspace"), "--output", str(output_path)])

    assert result.exit_code == 0
    assert opened == [output_path.resolve().as_uri()]
    assert f"opened\t{output_path.resolve().as_uri()}" in result.stdout


def test_resolve_lane_preset_rejects_unknown_name() -> None:
    try:
        resolve_lane_preset("unknown")
    except ValueError as exc:
        assert "available" in str(exc)
    else:
        raise AssertionError("expected ValueError")


def test_resolve_lane_preset_exposes_tier1_core_budget_ladder() -> None:
    assert resolve_lane_preset("local").budgets == (64,)
    assert resolve_lane_preset("overnight").budgets == (256,)
    assert resolve_lane_preset("weekend").budgets == (1000,)


def _dashboard_row(
    system: str,
    benchmark_id: str,
    metric_value: float | None,
    *,
    outcome_status: str = "ok",
    failure_reason: str | None = None,
    seeding_bucket: str = "transfer-opaque",
    seed_source_system: str | None = None,
    seed_source_run_id: str | None = None,
    seed_artifact_path: str | None = None,
    seed_selected_family: str | None = None,
) -> dict[str, object]:
    return {
        "pack_name": "tier1_core_eval64",
        "budget": 64,
        "seed": 42,
        "system": system,
        "run_id": "tier1_core_eval64_seed42",
        "benchmark_id": benchmark_id,
        "metric_name": "accuracy",
        "metric_direction": "max",
        "metric_value": metric_value,
        "outcome_status": outcome_status,
        "failure_reason": failure_reason,
        "evaluation_count": 64,
        "epochs_per_candidate": 20,
        "budget_policy_name": "prototype_equal_budget",
        "wall_clock_seconds": 1.0,
        "matrix_scope": "fair",
        "fairness_metadata": {
            "benchmark_pack_id": "tier1_core_eval64",
            "seed": 42,
            "evaluation_count": 64,
            "budget_policy_name": "prototype_equal_budget",
            "data_signature": "shared",
            "code_version": "deadbeef",
            "seeding_bucket": seeding_bucket,
            "seed_source_system": seed_source_system,
            "seed_source_run_id": seed_source_run_id,
            "seed_artifact_path": seed_artifact_path,
            "seed_selected_family": seed_selected_family,
            "pairwise_fairness_ok": True,
        },
    }
