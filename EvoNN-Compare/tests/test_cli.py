from typer.testing import CliRunner

from evonn_compare.cli.main import app
from evonn_compare.orchestration.lane_presets import resolve_lane_preset

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


def test_fair_matrix_help() -> None:
    result = runner.invoke(app, ["fair-matrix", "--help"])
    assert result.exit_code == 0
    assert "--primordia-root" in result.stdout
    assert "--no-contenders" in result.stdout
    assert "--preset" in result.stdout


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


def test_resolve_lane_preset_rejects_unknown_name() -> None:
    try:
        resolve_lane_preset("unknown")
    except ValueError as exc:
        assert "available" in str(exc)
    else:
        raise AssertionError("expected ValueError")
