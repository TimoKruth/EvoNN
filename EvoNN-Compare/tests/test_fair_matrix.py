from pathlib import Path
import subprocess

import pytest

from evonn_compare.comparison.fair_matrix import (
    build_matrix_summary,
    build_matrix_trend_rows,
    summarize_matrix_case,
)
from evonn_compare.comparison.engine import ComparisonEngine
from evonn_compare.contracts.parity import load_parity_pack
from evonn_compare.ingest.loader import SystemIngestor
from evonn_compare.orchestration.fair_matrix import (
    MatrixCase,
    _native_runtime_available,
    _write_trend_artifacts,
)
from evonn_compare.reporting.fair_matrix_md import render_fair_matrix_markdown
from test_compare import PACK_PATH, _write_run


def test_fair_matrix_markdown_splits_fair_and_reference_rows(tmp_path: Path) -> None:
    pack = load_parity_pack(PACK_PATH)
    systems = {
        "prism": tmp_path / "prism",
        "topograph": tmp_path / "topograph",
        "stratograph": tmp_path / "stratograph",
        "primordia": tmp_path / "primordia",
        "contenders": tmp_path / "contenders",
    }
    for system, run_dir in systems.items():
        _write_run(run_dir, system=system)

    ingestors = {system: SystemIngestor(path) for system, path in systems.items()}
    runs = {
        system: (ingestor.load_manifest(), ingestor.load_results())
        for system, ingestor in ingestors.items()
    }
    pair_results = {}
    for left, right in (("prism", "topograph"), ("prism", "stratograph"), ("prism", "primordia"), ("prism", "contenders")):
        result = ComparisonEngine().compare(
            left_manifest=runs[left][0],
            left_results=runs[left][1],
            right_manifest=runs[right][0],
            right_results=runs[right][1],
            pack=pack,
        )
        pair_results[(left, right)] = (result, Path(f"{left}_vs_{right}.md"))

    fair_row, reference_row, parity_rows = summarize_matrix_case(
        pack=pack,
        budget=64,
        seed=42,
        runs=runs,
        pair_results=pair_results,
    )
    summary = build_matrix_summary(
        pack_name=pack.name,
        fair_rows=[fair_row] if fair_row is not None else [],
        reference_rows=[reference_row] if reference_row is not None else [],
        parity_rows=parity_rows,
    )
    markdown = render_fair_matrix_markdown(summary)

    assert "## Fair Search-Budget Results" in markdown
    assert "## Reference Baseline Results" in markdown
    assert "## Parity/Validity Check" in markdown
    assert "| 64 | 42 |" in markdown


def test_fair_matrix_markdown_includes_lane_metadata() -> None:
    from evonn_compare.comparison.fair_matrix import LaneMetadata

    summary = build_matrix_summary(
        pack_name="tier1_core_smoke_eval16",
        lane=LaneMetadata(
            preset="smoke",
            pack_name="tier1_core_smoke_eval16",
            expected_budget=16,
            expected_seed=42,
            artifact_completeness_ok=True,
            fairness_ok=True,
            repeatability_ready=True,
        ),
        fair_rows=[],
        reference_rows=[],
        parity_rows=[],
    )

    markdown = render_fair_matrix_markdown(summary)

    assert "## Lane Metadata" in markdown
    assert "- Preset: `smoke`" in markdown
    assert "- Repeatability Ready: `yes`" in markdown


def test_fair_matrix_reference_row_for_nonfair_pair(tmp_path: Path) -> None:
    pack = load_parity_pack(PACK_PATH)
    systems = {
        "prism": tmp_path / "prism",
        "topograph": tmp_path / "topograph",
        "stratograph": tmp_path / "stratograph",
        "primordia": tmp_path / "primordia",
        "contenders": tmp_path / "contenders",
    }
    for system, run_dir in systems.items():
        _write_run(run_dir, system=system)
    _write_run(systems["contenders"], system="contenders", budget_policy_name="fixed_contender_pool")

    ingestors = {system: SystemIngestor(path) for system, path in systems.items()}
    runs = {
        system: (ingestor.load_manifest(), ingestor.load_results())
        for system, ingestor in ingestors.items()
    }
    result = ComparisonEngine().compare(
        left_manifest=runs["prism"][0],
        left_results=runs["prism"][1],
        right_manifest=runs["contenders"][0],
        right_results=runs["contenders"][1],
        pack=pack,
    )
    fair_row, reference_row, _parity_rows = summarize_matrix_case(
        pack=pack,
        budget=64,
        seed=42,
        runs=runs,
        pair_results={("prism", "contenders"): (result, Path("prism_vs_contenders.md"))},
    )

    assert fair_row is None
    assert reference_row is not None
    assert "prism/contenders" in reference_row.note


def test_fair_matrix_markdown_omits_contenders_when_not_requested(tmp_path: Path) -> None:
    pack = load_parity_pack(PACK_PATH)
    systems = {
        "prism": tmp_path / "prism",
        "topograph": tmp_path / "topograph",
        "stratograph": tmp_path / "stratograph",
        "primordia": tmp_path / "primordia",
    }
    for system, run_dir in systems.items():
        _write_run(run_dir, system=system)

    ingestors = {system: SystemIngestor(path) for system, path in systems.items()}
    runs = {
        system: (ingestor.load_manifest(), ingestor.load_results())
        for system, ingestor in ingestors.items()
    }
    pair_results = {}
    for left, right in (("prism", "topograph"), ("prism", "stratograph"), ("prism", "primordia")):
        result = ComparisonEngine().compare(
            left_manifest=runs[left][0],
            left_results=runs[left][1],
            right_manifest=runs[right][0],
            right_results=runs[right][1],
            pack=pack,
        )
        pair_results[(left, right)] = (result, Path(f"{left}_vs_{right}.md"))

    fair_row, reference_row, parity_rows = summarize_matrix_case(
        pack=pack,
        budget=64,
        seed=42,
        runs=runs,
        pair_results=pair_results,
        systems=("prism", "topograph", "stratograph", "primordia"),
    )
    summary = build_matrix_summary(
        pack_name=pack.name,
        fair_rows=[fair_row] if fair_row is not None else [],
        reference_rows=[reference_row] if reference_row is not None else [],
        parity_rows=parity_rows,
        systems=("prism", "topograph", "stratograph", "primordia"),
    )
    markdown = render_fair_matrix_markdown(summary)

    assert "Contenders Evals" not in markdown
    assert "Contenders Wins" not in markdown


def test_build_matrix_trend_rows_capture_minimum_longitudinal_dimensions(tmp_path: Path) -> None:
    pack = load_parity_pack(PACK_PATH)
    systems = {
        "prism": tmp_path / "prism",
        "topograph": tmp_path / "topograph",
    }
    for system, run_dir in systems.items():
        _write_run(run_dir, system=system)

    ingestors = {system: SystemIngestor(path) for system, path in systems.items()}
    runs = {
        system: (ingestor.load_manifest(), ingestor.load_results())
        for system, ingestor in ingestors.items()
    }
    pair_results = {}
    result = ComparisonEngine().compare(
        left_manifest=runs["prism"][0],
        left_results=runs["prism"][1],
        right_manifest=runs["topograph"][0],
        right_results=runs["topograph"][1],
        pack=pack,
    )
    pair_results[("prism", "topograph")] = (result, Path("prism_vs_topograph.md"))

    trend_rows = build_matrix_trend_rows(
        pack=pack,
        budget=64,
        seed=42,
        runs=runs,
        pair_results=pair_results,
        systems=("prism", "topograph"),
    )

    assert len(trend_rows) == len(pack.benchmarks) * 2
    first = trend_rows[0]
    assert first.pack_name == pack.name
    assert first.system in {"prism", "topograph"}
    assert first.run_id == f"{first.system}-run"
    assert first.benchmark_id == pack.benchmarks[0].benchmark_id
    assert first.metric_direction == pack.benchmarks[0].metric_direction
    assert first.outcome_status == "ok"
    assert first.matrix_scope == "fair"
    assert first.fairness_metadata["benchmark_pack_id"] == pack.name
    assert first.fairness_metadata["seed"] == 42
    assert first.fairness_metadata["evaluation_count"] == 64


def test_write_trend_artifacts_persists_case_and_workspace_reports(tmp_path: Path) -> None:
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
    case = MatrixCase(
        pack_name=pack.name,
        lane_preset="smoke",
        seed=42,
        budget=64,
        pack_path=PACK_PATH,
        prism_config_path=tmp_path / "prism.yaml",
        topograph_config_path=tmp_path / "topograph.yaml",
        stratograph_config_path=tmp_path / "stratograph.yaml",
        primordia_config_path=tmp_path / "primordia.yaml",
        contender_config_path=None,
        prism_run_dir=prism_dir,
        topograph_run_dir=topograph_dir,
        stratograph_run_dir=tmp_path / "stratograph-run",
        primordia_run_dir=tmp_path / "primordia-run",
        contender_run_dir=None,
        report_dir=tmp_path / "reports" / "case",
        summary_output_path=tmp_path / "reports" / "case" / "fair_matrix_summary.md",
        log_dir=tmp_path / "logs" / "case",
        systems=("prism", "topograph"),
    )

    _write_trend_artifacts(case, trend_rows)

    case_report = case.report_dir / "trend_report.md"
    workspace_jsonl = case.report_dir.parent / "fair_matrix_trend_rows.jsonl"
    workspace_report = case.report_dir.parent / "fair_matrix_trends.md"
    assert case_report.exists()
    assert workspace_jsonl.exists()
    assert workspace_report.exists()
    assert "# Fair Matrix Trends: tier1_core" in case_report.read_text(encoding="utf-8")
    assert "- Systems: `prism, topograph`" in workspace_report.read_text(encoding="utf-8")


def test_native_runtime_available_checks_target_project_environment(monkeypatch: pytest.MonkeyPatch, tmp_path: Path) -> None:
    calls: list[dict[str, object]] = []

    def fake_run(argv, *, cwd, stdout, stderr, check):
        calls.append({"argv": argv, "cwd": cwd, "check": check})
        return subprocess.CompletedProcess(argv, 0)

    monkeypatch.setattr(subprocess, "run", fake_run)

    assert _native_runtime_available(tmp_path, "prism.pipeline.coordinator") is True
    assert calls == [
        {
            "argv": [
                "uv",
                "run",
                "python",
                "-c",
                "import importlib; importlib.import_module('prism.pipeline.coordinator')",
            ],
            "cwd": tmp_path,
            "check": False,
        }
    ]


def test_native_runtime_available_returns_false_when_probe_cannot_launch(monkeypatch: pytest.MonkeyPatch, tmp_path: Path) -> None:
    def fake_run(argv, *, cwd, stdout, stderr, check):
        raise FileNotFoundError("uv missing")

    monkeypatch.setattr(subprocess, "run", fake_run)

    assert _native_runtime_available(tmp_path, "prism.pipeline.coordinator") is False
