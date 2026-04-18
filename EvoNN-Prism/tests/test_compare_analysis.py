from __future__ import annotations

from pathlib import Path

from typer.testing import CliRunner

from prism.analysis.compare import aggregate_compare_rows, parse_four_way_summary, render_compare_analysis
from prism.cli import app


runner = CliRunner()


def _write_summary(path: Path, budget: int, seed: int, prism: int, topograph: int, stratograph: int, contenders: int, ties: int) -> None:
    path.write_text(
        "\n".join(
            [
                "# Fair Matrix",
                "",
                "## Fair Search-Budget Results",
                "",
                "| Budget | Seed | Benchmarks | Prism Evals | Prism Wins | Topograph Evals | Topograph Wins | Stratograph Evals | Stratograph Wins | Contenders Evals | Contenders Wins | Ties |",
                "| ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: |",
                f"| {budget} | {seed} | 38 | {budget} | {prism} | {budget} | {topograph} | {budget} | {stratograph} | {budget} | {contenders} | {ties} |",
                "",
            ]
        ),
        encoding="utf-8",
    )


def test_compare_analysis_parses_and_aggregates(tmp_path: Path):
    first = tmp_path / "seed42.md"
    second = tmp_path / "seed43.md"
    _write_summary(first, 304, 42, 2, 1, 2, 29, 4)
    _write_summary(second, 304, 43, 1, 3, 2, 28, 4)

    row = parse_four_way_summary(first)
    summary = aggregate_compare_rows([first, second])
    text = render_compare_analysis([first, second])

    assert row.prism_wins == 2
    assert summary[304]["prism_wins_avg"] == 1.5
    assert "Prism Compare Analysis" in text
    assert "Extra eval budget gives weak Prism return" in text


def test_cli_analyze_compare_writes_output(tmp_path: Path):
    first = tmp_path / "seed42.md"
    second = tmp_path / "seed43.md"
    output = tmp_path / "analysis.md"
    _write_summary(first, 304, 42, 2, 1, 2, 29, 4)
    _write_summary(second, 912, 43, 4, 1, 3, 26, 5)

    result = runner.invoke(app, ["analyze-compare", str(first), str(second), "--output", str(output)])

    assert result.exit_code == 0
    assert output.exists()
    text = output.read_text(encoding="utf-8")
    assert "Aggregated Results" in text
    assert "Prism Actions" in text
