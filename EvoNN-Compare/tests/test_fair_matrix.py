import json
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
    _build_lane_metadata,
    _build_trend_records,
    _native_runtime_available,
    _write_trend_artifacts,
    run_fair_matrix_case,
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
            task_coverage_ok=True,
            budget_consistency_ok=True,
            seed_consistency_ok=True,
            budget_accounting_ok=True,
            core_systems_complete_ok=True,
            extended_systems_complete_ok=False,
            observed_task_kinds=("classification", "regression"),
            system_operating_states={
                "prism": "benchmark-complete",
                "topograph": "benchmark-complete",
                "contenders": "benchmark-complete",
            },
            operating_state="trusted-core",
            acceptance_notes=(),
            repeatability_ready=True,
        ),
        fair_rows=[],
        reference_rows=[],
        parity_rows=[],
    )

    markdown = render_fair_matrix_markdown(summary)

    assert "## Lane Metadata" in markdown
    assert "- Preset: `smoke`" in markdown
    assert "- Operating State: `trusted-core`" in markdown
    assert "- Budget Accounting: `ok`" in markdown
    assert "- Task Coverage: `ok` (classification, regression)" in markdown
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
    assert first.lane_operating_state == "reference-only"
    assert first.system_operating_state == "unknown"
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
        trend_dataset_path=tmp_path / "trends" / "fair_matrix_trends.jsonl",
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


def test_build_trend_records_preserves_longitudinal_fields(tmp_path: Path) -> None:
    pack = load_parity_pack(PACK_PATH)
    run_dirs = {
        "prism": tmp_path / "prism",
        "topograph": tmp_path / "topograph",
    }
    _write_run(run_dirs["prism"], system="prism", score_shift=0.02, budget_policy_name="evolutionary_search")
    _write_run(run_dirs["topograph"], system="topograph", budget_policy_name="evolutionary_search")

    ingestors = {system: SystemIngestor(path) for system, path in run_dirs.items()}
    runs = {
        system: (ingestor.load_manifest(), ingestor.load_results())
        for system, ingestor in ingestors.items()
    }

    class CaseStub:
        lane_preset = "smoke"
        pack_name = pack.name
        budget = 64
        seed = 42
        systems = ("prism", "topograph")
        prism_run_dir = run_dirs["prism"]
        topograph_run_dir = run_dirs["topograph"]
        stratograph_run_dir = tmp_path / "stratograph"
        primordia_run_dir = tmp_path / "primordia"
        contender_run_dir = None

    pair_result = ComparisonEngine().compare(
        left_manifest=runs["prism"][0],
        left_results=runs["prism"][1],
        right_manifest=runs["topograph"][0],
        right_results=runs["topograph"][1],
        pack=pack,
    )
    lane = _build_lane_metadata(
        case=CaseStub(),
        runs=runs,
        pair_results={("prism", "topograph"): (pair_result, Path("prism_vs_topograph.md"))},
    )

    trend_records = _build_trend_records(case=CaseStub(), pack=pack, runs=runs, lane=lane)

    assert len(trend_records) == len(pack.benchmarks) * 2
    first = trend_records[0]
    assert first["lane_preset"] == "smoke"
    assert first["pack"] == pack.name
    assert first["engine"] in {"prism", "topograph"}
    assert first["benchmark"]
    assert first["task_kind"] in {"classification", "regression", "language_modeling"}
    assert first["seed"] == 42
    assert first["budget"] == 64
    assert first["outcome_status"] == "ok"
    assert first["metric_name"]
    assert first["metric_direction"] in {"max", "min"}
    assert first["fairness"]["benchmark_pack_id"] == pack.name
    assert first["fairness"]["evaluation_count"] == 64
    assert first["fairness"]["lane_operating_state"] == "contract-fair"
    assert first["system_operating_state"] == "benchmark-complete"
    assert first["lane"]["operating_state"] == "contract-fair"
    assert first["artifact_paths"]["manifest"].endswith("manifest.json")
    assert first["artifact_paths"]["summary"].endswith("summary.json")


def test_build_lane_metadata_requires_summary_artifact_for_repeatability(tmp_path: Path) -> None:
    pack = load_parity_pack(PACK_PATH)
    run_dirs = {
        "prism": tmp_path / "prism",
        "topograph": tmp_path / "topograph",
    }
    _write_run(run_dirs["prism"], system="prism", score_shift=0.02, budget_policy_name="evolutionary_search")
    _write_run(run_dirs["topograph"], system="topograph", budget_policy_name="evolutionary_search")
    (run_dirs["topograph"] / "summary.json").unlink()

    ingestors = {system: SystemIngestor(path) for system, path in run_dirs.items()}
    runs = {
        system: (ingestor.load_manifest(), ingestor.load_results())
        for system, ingestor in ingestors.items()
    }
    pair_result = ComparisonEngine().compare(
        left_manifest=runs["prism"][0],
        left_results=runs["prism"][1],
        right_manifest=runs["topograph"][0],
        right_results=runs["topograph"][1],
        pack=pack,
    )

    class CaseStub:
        lane_preset = "smoke"
        pack_name = pack.name
        budget = 64
        seed = 42
        prism_run_dir = run_dirs["prism"]
        topograph_run_dir = run_dirs["topograph"]
        stratograph_run_dir = tmp_path / "stratograph"
        primordia_run_dir = tmp_path / "primordia"
        contender_run_dir = None

    lane = _build_lane_metadata(
        case=CaseStub(),
        runs=runs,
        pair_results={("prism", "topograph"): (pair_result, Path("prism_vs_topograph.md"))},
    )

    assert lane.fairness_ok is True
    assert lane.task_coverage_ok is True
    assert lane.artifact_completeness_ok is False
    assert lane.repeatability_ready is False
    assert "missing required artifacts" in lane.acceptance_notes


def test_run_fair_matrix_case_emits_trend_artifacts(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    case = MatrixCase(
        pack_name="tier1_core",
        lane_preset="smoke",
        seed=42,
        budget=64,
        pack_path=PACK_PATH,
        prism_config_path=tmp_path / "configs" / "prism.yaml",
        topograph_config_path=tmp_path / "configs" / "topograph.yaml",
        stratograph_config_path=tmp_path / "configs" / "stratograph.yaml",
        primordia_config_path=tmp_path / "configs" / "primordia.yaml",
        contender_config_path=None,
        prism_run_dir=tmp_path / "runs" / "prism",
        topograph_run_dir=tmp_path / "runs" / "topograph",
        stratograph_run_dir=tmp_path / "runs" / "stratograph",
        primordia_run_dir=tmp_path / "runs" / "primordia",
        contender_run_dir=None,
        report_dir=tmp_path / "reports",
        summary_output_path=tmp_path / "reports" / "fair_matrix_summary.md",
        trend_dataset_path=tmp_path / "trends" / "fair_matrix_trends.jsonl",
        log_dir=tmp_path / "logs",
        systems=("prism", "topograph", "stratograph", "primordia"),
    )
    for path in [
        case.prism_config_path,
        case.topograph_config_path,
        case.stratograph_config_path,
        case.primordia_config_path,
    ]:
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_text("seed: 42\n", encoding="utf-8")

    monkeypatch.setattr("evonn_compare.orchestration.fair_matrix._native_runtime_available", lambda *_args, **_kwargs: False)

    def fake_run_command(spec, *, log_dir):
        if spec.name == "stratograph_export":
            _write_run(case.stratograph_run_dir, system="stratograph", budget_policy_name="evolutionary_search")

    def fake_prism_export(**kwargs):
        _write_run(kwargs["run_dir"], system="prism", score_shift=0.02, budget_policy_name="evolutionary_search")

    def fake_topograph_export(**kwargs):
        _write_run(kwargs["run_dir"], system="topograph", budget_policy_name="evolutionary_search")

    def fake_primordia_export(**kwargs):
        _write_run(kwargs["run_dir"], system="primordia", budget_policy_name="evolutionary_search")

    monkeypatch.setattr("evonn_compare.orchestration.fair_matrix._run_command", fake_run_command)
    monkeypatch.setattr("evonn_compare.orchestration.fair_matrix.ensure_prism_portable_smoke_export", fake_prism_export)
    monkeypatch.setattr("evonn_compare.orchestration.fair_matrix.ensure_topograph_portable_smoke_export", fake_topograph_export)
    monkeypatch.setattr("evonn_compare.orchestration.fair_matrix.ensure_primordia_export", fake_primordia_export)

    summary_path = run_fair_matrix_case(case, parallel=False)

    trend_json = case.report_dir / "fair_matrix_trends.json"
    trend_jsonl = case.report_dir / "fair_matrix_trends.jsonl"
    lane_acceptance = case.report_dir / "lane_acceptance.json"
    assert summary_path == case.summary_output_path
    assert summary_path.exists()
    assert trend_json.exists()
    assert trend_jsonl.exists()
    assert lane_acceptance.exists()
    assert case.trend_dataset_path.exists()

    trend_records = json.loads(trend_json.read_text(encoding="utf-8"))
    assert len(trend_records) == len(load_parity_pack(PACK_PATH).benchmarks) * 4
    assert all(record["artifact_paths"]["summary"].endswith("summary.json") for record in trend_records)
    assert all(record["lane"]["operating_state"] == "contract-fair" for record in trend_records)
    assert all(record["lane"]["budget_accounting_ok"] is True for record in trend_records)
    assert sum(1 for _line in trend_jsonl.read_text(encoding="utf-8").splitlines() if _line.strip()) == len(trend_records)
    assert sum(1 for _line in case.trend_dataset_path.read_text(encoding="utf-8").splitlines() if _line.strip()) == len(trend_records)

    lane_payload = json.loads(lane_acceptance.read_text(encoding="utf-8"))
    assert lane_payload["operating_state"] == "contract-fair"
    assert lane_payload["budget_accounting_ok"] is True


def test_build_lane_metadata_rejects_task_or_budget_seed_drift(tmp_path: Path) -> None:
    pack = load_parity_pack(PACK_PATH)
    run_dirs = {
        "prism": tmp_path / "prism",
        "topograph": tmp_path / "topograph",
    }
    _write_run(run_dirs["prism"], system="prism", evaluation_count=64)
    _write_run(run_dirs["topograph"], system="topograph", evaluation_count=32)

    manifest_payload = json.loads((run_dirs["topograph"] / "manifest.json").read_text(encoding="utf-8"))
    manifest_payload["seed"] = 7
    (run_dirs["topograph"] / "manifest.json").write_text(json.dumps(manifest_payload, indent=2), encoding="utf-8")

    ingestors = {system: SystemIngestor(path) for system, path in run_dirs.items()}
    runs = {
        system: (ingestor.load_manifest(), ingestor.load_results())
        for system, ingestor in ingestors.items()
    }
    pair_result = ComparisonEngine().compare(
        left_manifest=runs["prism"][0],
        left_results=runs["prism"][1],
        right_manifest=runs["topograph"][0],
        right_results=runs["topograph"][1],
        pack=pack,
    )

    class CaseStub:
        lane_preset = "smoke"
        pack_name = pack.name
        budget = 64
        seed = 42
        prism_run_dir = run_dirs["prism"]
        topograph_run_dir = run_dirs["topograph"]
        stratograph_run_dir = tmp_path / "stratograph"
        primordia_run_dir = tmp_path / "primordia"
        contender_run_dir = None

    lane = _build_lane_metadata(
        case=CaseStub(),
        runs=runs,
        pair_results={("prism", "topograph"): (pair_result, Path("prism_vs_topograph.md"))},
    )

    assert lane.budget_consistency_ok is False
    assert lane.seed_consistency_ok is False
    assert lane.repeatability_ready is False
    assert "manifest evaluation counts drift from requested lane budget" in lane.acceptance_notes
    assert "manifest seeds drift from requested lane seed" in lane.acceptance_notes


def test_build_lane_metadata_distinguishes_contract_fair_from_trusted_core(tmp_path: Path) -> None:
    pack = load_parity_pack(PACK_PATH)
    run_dirs = {
        "prism": tmp_path / "prism",
        "topograph": tmp_path / "topograph",
        "contenders": tmp_path / "contenders",
        "stratograph": tmp_path / "stratograph",
    }
    _write_run(run_dirs["prism"], system="prism", score_shift=0.02, budget_policy_name="evolutionary_search")
    _write_run(run_dirs["topograph"], system="topograph", budget_policy_name="evolutionary_search")
    _write_run(run_dirs["contenders"], system="contenders", budget_policy_name="evolutionary_search")
    contenders_manifest_path = run_dirs["contenders"] / "manifest.json"
    contenders_manifest = json.loads(contenders_manifest_path.read_text(encoding="utf-8"))
    contenders_manifest["baseline_coverage"] = {
        "benchmark_complete_policy": "required_only_optional_skips_allowed",
        "optional_dependency_skips": {"tabular": ["xgb_small", "lgbm_small"]},
        "notes": ["optional dependency backends skipped under required-only completeness policy"],
    }
    contenders_manifest_path.write_text(json.dumps(contenders_manifest, indent=2), encoding="utf-8")

    ingestors = {system: SystemIngestor(path) for system, path in run_dirs.items() if path.exists()}
    runs = {
        system: (ingestor.load_manifest(), ingestor.load_results())
        for system, ingestor in ingestors.items()
    }
    pair_results = {}
    for left, right in (("prism", "topograph"), ("prism", "contenders")):
        result = ComparisonEngine().compare(
            left_manifest=runs[left][0],
            left_results=runs[left][1],
            right_manifest=runs[right][0],
            right_results=runs[right][1],
            pack=pack,
        )
        pair_results[(left, right)] = (result, Path(f"{left}_vs_{right}.md"))

    class CaseStub:
        lane_preset = "local"
        pack_name = pack.name
        budget = 64
        seed = 42
        prism_run_dir = run_dirs["prism"]
        topograph_run_dir = run_dirs["topograph"]
        stratograph_run_dir = run_dirs["stratograph"]
        primordia_run_dir = tmp_path / "primordia"
        contender_run_dir = run_dirs["contenders"]
        systems = ("prism", "topograph", "contenders")

    lane = _build_lane_metadata(
        case=CaseStub(),
        runs=runs,
        pair_results=pair_results,
    )

    assert lane.budget_accounting_ok is True
    assert lane.operating_state == "trusted-core"
    assert lane.core_systems_complete_ok is True
    assert lane.extended_systems_complete_ok is False
    assert lane.repeatability_ready is True
    assert lane.system_operating_states["contenders"] == "benchmark-complete-optional-skips"
    assert any("contenders coverage policy" in note for note in lane.acceptance_notes)
    assert any("trusted-extended unmet" in note for note in lane.acceptance_notes)


def test_build_lane_metadata_can_block_on_strict_optional_skip_policy(tmp_path: Path) -> None:
    pack = load_parity_pack(PACK_PATH)
    run_dirs = {
        "prism": tmp_path / "prism",
        "topograph": tmp_path / "topograph",
        "contenders": tmp_path / "contenders",
    }
    _write_run(run_dirs["prism"], system="prism", score_shift=0.02, budget_policy_name="evolutionary_search")
    _write_run(run_dirs["topograph"], system="topograph", budget_policy_name="evolutionary_search")
    _write_run(run_dirs["contenders"], system="contenders", budget_policy_name="evolutionary_search")
    contenders_manifest_path = run_dirs["contenders"] / "manifest.json"
    contenders_manifest = json.loads(contenders_manifest_path.read_text(encoding="utf-8"))
    contenders_manifest["baseline_coverage"] = {
        "benchmark_complete_policy": "all_configured_contenders_required",
        "optional_dependency_skips": {"tabular": ["xgb_small"]},
        "notes": ["configured optional baselines are required on this lane"],
    }
    contenders_manifest_path.write_text(json.dumps(contenders_manifest, indent=2), encoding="utf-8")

    ingestors = {system: SystemIngestor(path) for system, path in run_dirs.items()}
    runs = {
        system: (ingestor.load_manifest(), ingestor.load_results())
        for system, ingestor in ingestors.items()
    }
    pair_results = {}
    for left, right in (("prism", "topograph"), ("prism", "contenders")):
        result = ComparisonEngine().compare(
            left_manifest=runs[left][0],
            left_results=runs[left][1],
            right_manifest=runs[right][0],
            right_results=runs[right][1],
            pack=pack,
        )
        pair_results[(left, right)] = (result, Path(f"{left}_vs_{right}.md"))

    class CaseStub:
        lane_preset = "local"
        pack_name = pack.name
        budget = 64
        seed = 42
        prism_run_dir = run_dirs["prism"]
        topograph_run_dir = run_dirs["topograph"]
        stratograph_run_dir = tmp_path / "stratograph"
        primordia_run_dir = tmp_path / "primordia"
        contender_run_dir = run_dirs["contenders"]
        systems = ("prism", "topograph", "contenders")

    lane = _build_lane_metadata(case=CaseStub(), runs=runs, pair_results=pair_results)

    assert lane.operating_state == "contract-fair"
    assert lane.core_systems_complete_ok is False
    assert lane.repeatability_ready is False
    assert lane.system_operating_states["contenders"] == "benchmark-incomplete-optional-skips"
