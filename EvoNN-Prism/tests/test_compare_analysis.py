from __future__ import annotations

from pathlib import Path

from typer.testing import CliRunner

from prism.analysis.compare import aggregate_compare_rows, parse_four_way_summary, render_compare_analysis
from prism.analysis.matrix import render_matrix_analysis
from prism.cli import app
from prism.genome import ModelGenome
from prism.storage import RunStore


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


def test_render_matrix_analysis_aggregates_runs(tmp_path: Path):
    matrix_root = tmp_path / "matrix"
    report_dir = matrix_root / "reports" / "pack_seed42"
    run_dir = matrix_root / "runs" / "prism" / "pack_seed42"
    report_dir.mkdir(parents=True)
    run_dir.mkdir(parents=True)

    _write_summary(report_dir / "four_way_summary.md", 304, 42, 2, 1, 2, 29, 4)

    genome = ModelGenome(family="mlp", hidden_layers=[16, 8], activation="relu", dropout=0.1)
    with RunStore(run_dir / "metrics.duckdb") as store:
        store.save_run(run_dir.name, {"seed": 42})
        store.save_genome(run_dir.name, genome)
        store.save_evaluation(run_dir.name, genome.genome_id, 0, "moons", "accuracy", 0.91, 0.91, 100, 0.2, inherited_from="parent")
        store.save_lineage(run_dir.name, genome.genome_id, "parent", 0, "mutation:width")
        store.save_archive(run_dir.name, 0, "specialist:moons:mlp", "moons", genome.genome_id, 0.91)
        store.save_archive(run_dir.name, 1, "pareto", None, genome.genome_id, 0.92)

    text = render_matrix_analysis(matrix_root)

    assert "Run Diagnostics" in text
    assert "Inheritance Hit Rate" in text
    assert "Avg Specialists" in text
    assert "Avg New Archive Members" in text
    assert "mlp(1)" in text
    assert "Specialist archives active" in text


def test_cli_analyze_matrix_writes_output(tmp_path: Path):
    matrix_root = tmp_path / "matrix"
    report_dir = matrix_root / "reports" / "pack_seed42"
    run_dir = matrix_root / "runs" / "prism" / "pack_seed42"
    output = tmp_path / "matrix_analysis.md"
    report_dir.mkdir(parents=True)
    run_dir.mkdir(parents=True)
    _write_summary(report_dir / "four_way_summary.md", 304, 42, 2, 1, 2, 29, 4)

    genome = ModelGenome(family="mlp", hidden_layers=[16, 8], activation="relu", dropout=0.1)
    with RunStore(run_dir / "metrics.duckdb") as store:
        store.save_run(run_dir.name, {"seed": 42})
        store.save_genome(run_dir.name, genome)
        store.save_evaluation(run_dir.name, genome.genome_id, 0, "moons", "accuracy", 0.91, 0.91, 100, 0.2)

    result = runner.invoke(app, ["analyze-matrix", str(matrix_root), "--output", str(output)])

    assert result.exit_code == 0
    assert output.exists()
    text = output.read_text(encoding="utf-8")
    assert "Run Diagnostics" in text
