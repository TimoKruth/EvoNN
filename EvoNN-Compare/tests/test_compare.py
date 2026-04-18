import json
from datetime import datetime, timezone
from pathlib import Path

from evonn_compare.comparison.engine import ComparisonEngine
from evonn_compare.contracts.models import (
    ArtifactPaths,
    BenchmarkEntry,
    BudgetEnvelope,
    DeviceInfo,
    ResultRecord,
    RunManifest,
)
from evonn_compare.contracts.parity import load_parity_pack
from evonn_compare.ingest.loader import SystemIngestor
from evonn_compare.reporting.compare_md import render_comparison_markdown

PACK_PATH = Path(__file__).resolve().parents[1] / "parity_packs" / "tier1_core.yaml"


def _write_run(
    run_dir: Path,
    *,
    system: str,
    evaluation_count: int = 64,
    score_shift: float = 0.0,
    budget_policy_name: str | None = None,
) -> None:
    run_dir.mkdir(parents=True, exist_ok=True)
    (run_dir / "config_snapshot.json").write_text("{}", encoding="utf-8")
    (run_dir / "report.md").write_text("# Report\n", encoding="utf-8")
    pack = load_parity_pack(PACK_PATH)

    manifest = RunManifest(
        schema_version="1.0",
        system=system,
        run_id=f"{system}-run",
        run_name=f"{system}-run",
        created_at=datetime(2026, 4, 1, tzinfo=timezone.utc),
        pack_name=pack.name,
        seed=42,
        benchmarks=[
            BenchmarkEntry(
                benchmark_id=entry.benchmark_id,
                task_kind=entry.task_kind,
                metric_name=entry.metric_name,
                metric_direction=entry.metric_direction,
                status="ok",
            )
            for entry in pack.benchmarks
        ],
        budget=BudgetEnvelope(
            evaluation_count=evaluation_count,
            epochs_per_candidate=20,
            budget_policy_name=budget_policy_name,
        ),
        device=DeviceInfo(device_name="apple_silicon", precision_mode="fp32"),
        artifacts=ArtifactPaths(
            config_snapshot="config_snapshot.json",
            report_markdown="report.md",
        ),
    )
    results = []
    for index, entry in enumerate(pack.benchmarks):
        metric_value = 0.80 + score_shift + (0.01 * index) if entry.metric_direction == "max" else 0.20 - score_shift + (0.01 * index)
        results.append(
            ResultRecord(
                system=system,
                run_id=f"{system}-run",
                benchmark_id=entry.benchmark_id,
                metric_name=entry.metric_name,
                metric_direction=entry.metric_direction,
                metric_value=metric_value,
                status="ok",
            )
        )

    (run_dir / "manifest.json").write_text(manifest.model_dump_json(indent=2), encoding="utf-8")
    (run_dir / "results.json").write_text(
        json.dumps([result.model_dump(mode="json") for result in results], indent=2),
        encoding="utf-8",
    )


def test_compare_counts_wins(tmp_path: Path) -> None:
    left_dir = tmp_path / "prism"
    right_dir = tmp_path / "topograph"
    _write_run(left_dir, system="prism", score_shift=0.02)
    _write_run(right_dir, system="topograph")

    pack = load_parity_pack(PACK_PATH)
    left = SystemIngestor(left_dir)
    right = SystemIngestor(right_dir)
    result = ComparisonEngine().compare(
        left_manifest=left.load_manifest(),
        left_results=left.load_results(),
        right_manifest=right.load_manifest(),
        right_results=right.load_results(),
        pack=pack,
    )

    assert result.parity_status == "fair"
    assert result.summary.left_wins == 8
    assert result.summary.right_wins == 0
    assert result.summary.evonn_wins == 8
    assert result.summary.evonn2_wins == 0


def test_compare_markdown_uses_runtime_labels(tmp_path: Path) -> None:
    left_dir = tmp_path / "prism"
    right_dir = tmp_path / "topograph"
    _write_run(left_dir, system="prism", score_shift=0.02)
    _write_run(right_dir, system="topograph")

    pack = load_parity_pack(PACK_PATH)
    left = SystemIngestor(left_dir)
    right = SystemIngestor(right_dir)
    result = ComparisonEngine().compare(
        left_manifest=left.load_manifest(),
        left_results=left.load_results(),
        right_manifest=right.load_manifest(),
        right_results=right.load_results(),
        pack=pack,
    )
    markdown = render_comparison_markdown(result)
    assert "Prism wins" in markdown
    assert "Topograph wins" in markdown


def test_compare_marks_budget_policy_mismatch_as_asymmetric(tmp_path: Path) -> None:
    left_dir = tmp_path / "prism"
    right_dir = tmp_path / "contenders"
    _write_run(left_dir, system="prism", budget_policy_name="evolutionary_search")
    _write_run(right_dir, system="contenders", budget_policy_name="fixed_contender_pool")

    pack = load_parity_pack(PACK_PATH)
    left = SystemIngestor(left_dir)
    right = SystemIngestor(right_dir)
    result = ComparisonEngine().compare(
        left_manifest=left.load_manifest(),
        left_results=left.load_results(),
        right_manifest=right.load_manifest(),
        right_results=right.load_results(),
        pack=pack,
    )

    assert result.parity_status == "asymmetric"
    assert "budget policy mismatch" in result.reasons[0]


def test_compare_treats_budget_matched_missing_policy_as_fair(tmp_path: Path) -> None:
    left_dir = tmp_path / "prism"
    right_dir = tmp_path / "contenders"
    _write_run(left_dir, system="prism")
    _write_run(right_dir, system="contenders", budget_policy_name="prototype_equal_budget")

    pack = load_parity_pack(PACK_PATH)
    left = SystemIngestor(left_dir)
    right = SystemIngestor(right_dir)
    result = ComparisonEngine().compare(
        left_manifest=left.load_manifest(),
        left_results=left.load_results(),
        right_manifest=right.load_manifest(),
        right_results=right.load_results(),
        pack=pack,
    )

    assert result.parity_status == "fair"
