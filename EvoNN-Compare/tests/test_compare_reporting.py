from datetime import datetime, timezone
from pathlib import Path

from evonn_compare.comparison.engine import ComparisonEngine
from evonn_compare.comparison.fair_matrix import build_matrix_trend_rows
from evonn_shared.contracts import (
    ArtifactPaths,
    BenchmarkEntry,
    BudgetEnvelope,
    DeviceInfo,
    RunManifest,
    SearchTelemetry,
    SeedingEnvelope,
)
from evonn_compare.contracts.parity import load_parity_pack
from evonn_compare.ingest.loader import SystemIngestor
from evonn_compare.reporting.compare_md import render_comparison_markdown
from evonn_compare.reporting.compare_md import _telemetry_row
from evonn_compare.reporting.fair_matrix_trends_md import render_fair_matrix_trend_markdown
from test_compare import PACK_PATH, _write_run


def test_telemetry_row_preserves_zero_effective_epochs() -> None:
    manifest = RunManifest(
        schema_version="1.0",
        system="topograph",
        run_id="topograph-run",
        run_name="topograph-run",
        created_at=datetime(2026, 4, 1, tzinfo=timezone.utc),
        pack_name="pack",
        seed=42,
        benchmarks=[
            BenchmarkEntry(
                benchmark_id="iris_classification",
                task_kind="classification",
                metric_name="accuracy",
                metric_direction="max",
                status="ok",
            )
        ],
        budget=BudgetEnvelope(
            evaluation_count=64,
            epochs_per_candidate=20,
            effective_training_epochs=7,
        ),
        device=DeviceInfo(device_name="apple_silicon", precision_mode="fp32"),
        artifacts=ArtifactPaths(
            config_snapshot="config_snapshot.json",
            report_markdown="report.md",
        ),
        search_telemetry=SearchTelemetry(
            qd_enabled=True,
            effective_training_epochs=0,
        ),
    )

    row = _telemetry_row(manifest)

    assert "| transfer-opaque | --- | --- | --- | 0 | yes |" in row


def test_telemetry_row_surfaces_seeding_provenance() -> None:
    manifest = RunManifest(
        schema_version="1.0",
        system="topograph",
        run_id="topograph-run",
        run_name="topograph-run",
        created_at=datetime(2026, 4, 1, tzinfo=timezone.utc),
        pack_name="pack",
        seed=42,
        benchmarks=[
            BenchmarkEntry(
                benchmark_id="iris_classification",
                task_kind="classification",
                metric_name="accuracy",
                metric_direction="max",
                status="ok",
            )
        ],
        budget=BudgetEnvelope(evaluation_count=64, epochs_per_candidate=20),
        device=DeviceInfo(device_name="apple_silicon", precision_mode="fp32"),
        artifacts=ArtifactPaths(
            config_snapshot="config_snapshot.json",
            report_markdown="report.md",
        ),
        seeding=SeedingEnvelope(
            seeding_enabled=True,
            seeding_ladder="direct",
            seed_source_system="primordia",
            seed_source_run_id="prim-run-7",
            seed_artifact_path="seed_candidates.json",
            seed_target_family="tabular",
            seed_selected_family="sparse_mlp",
            seed_rank=1,
            seed_overlap_policy="family-overlapping",
        ),
    )

    row = _telemetry_row(manifest)

    assert "| direct | primordia:prim-run-7 | seed_candidates.json | sparse_mlp->tabular |" in row


def test_render_fair_matrix_trend_markdown_summarizes_longitudinal_rows(tmp_path: Path) -> None:
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

    markdown = render_fair_matrix_trend_markdown(trend_rows)

    assert "# Fair Matrix Trends: tier1_core" in markdown
    assert "## Trend Dataset Summary" in markdown
    assert "- Systems: `prism, topograph`" in markdown
    assert "- Fairness Scope: `fair`" in markdown
    assert "- Seeding Buckets: `transfer-opaque`" in markdown
    assert "- Budget Accounting: `incomplete`" in markdown
    assert "- Repeatability: `not-ready`" in markdown
    assert "- Unique Lane Runs: `1`" in markdown
    assert "## Lane Health By Budget" in markdown
    assert "| 64 | 2 | reference-only | incomplete | not-ready |" in markdown
    assert "## Outcome Status by System" in markdown
    assert "| prism | 8 | 0 | 0 | 0 | 0 |" in markdown
    assert "## Benchmark Trend View" in markdown
    assert "| prism | iris_classification | 1 | 0.820000 | 0.820000 | 0.000000 | ok | 64 | 42 | fair | transfer-opaque | --- | reference-only | incomplete | not-ready | unknown |" in markdown


def test_render_comparison_markdown_disambiguates_same_system_seed_modes(tmp_path: Path) -> None:
    pack = load_parity_pack(PACK_PATH)
    unseeded_dir = tmp_path / "unseeded"
    seeded_dir = tmp_path / "seeded"
    _write_run(unseeded_dir, system="topograph")
    _write_run(
        seeded_dir,
        system="topograph",
        score_shift=0.02,
        seeding=SeedingEnvelope(
            seeding_enabled=True,
            seeding_ladder="direct",
            seed_source_system="primordia",
            seed_source_run_id="prim-run-7",
            seed_artifact_path="seed_candidates.json",
            seed_selected_family="mlp",
            seed_overlap_policy="family-overlapping",
        ),
    )

    unseeded = SystemIngestor(unseeded_dir)
    seeded = SystemIngestor(seeded_dir)
    result = ComparisonEngine().compare(
        left_manifest=unseeded.load_manifest(),
        left_results=unseeded.load_results(),
        right_manifest=seeded.load_manifest(),
        right_results=seeded.load_results(),
        pack=pack,
    )

    markdown = render_comparison_markdown(result)

    assert "Topograph (unseeded) wins" in markdown
    assert "Topograph (direct) wins" in markdown
    assert "| Topograph (unseeded) | topograph-run |" in markdown
    assert "| Topograph (direct) | topograph-run |" in markdown
