import json
from datetime import datetime, timezone
from pathlib import Path

from evonn_compare.comparison.engine import ComparisonEngine
from evonn_compare.contracts.parity import load_parity_pack
from evonn_compare.ingest.loader import SystemIngestor
from evonn_compare.orchestration.transfer_regimes import (
    _seed_artifact_gate_payload,
    _transfer_regime_payload,
    publish_transfer_regime_workspace,
)
from evonn_shared.contracts import (
    ArtifactPaths,
    BenchmarkEntry,
    BudgetEnvelope,
    DeviceInfo,
    ResultRecord,
    RunManifest,
    SeedingEnvelope,
)


REPO_ROOT = Path(__file__).resolve().parents[2]
TIER_B_PACK_PATH = REPO_ROOT / "shared-benchmarks" / "suites" / "parity" / "tier_b_core.yaml"


def test_seed_artifact_gate_requires_expected_overlap_policy(tmp_path: Path) -> None:
    artifact_path = tmp_path / "seed_candidates.json"
    payload = {
        "system": "primordia",
        "run_id": "prim-tier-b-seed42",
        "seed_candidates": [
            {
                "seed_rank": 1,
                "family": "sparse_mlp",
                "benchmark_groups": ["classification", "regression"],
                "benchmark_wins": 2,
                "repeat_support_count": 4,
                "seed_overlap_policy": "family-overlapping",
                "representative_genome_id": "prim-g7",
                "representative_architecture_summary": "sparse_mlp[64x64]",
            }
        ],
    }
    artifact_path.write_text(json.dumps(payload, indent=2), encoding="utf-8")

    gate = _seed_artifact_gate_payload(
        artifact_path=artifact_path,
        payload=payload,
        expected_systems={"primordia"},
        required_overlap_policy="family-overlapping",
    )

    assert gate["passed"] is True
    assert gate["top_candidate"]["seed_overlap_policy"] == "family-overlapping"
    assert gate["errors"] == []


def test_transfer_regime_payload_classifies_gain_and_records_provenance(tmp_path: Path) -> None:
    control_dir = tmp_path / "control"
    direct_dir = tmp_path / "direct"
    _write_named_run(
        control_dir,
        pack_path=TIER_B_PACK_PATH,
        system="topograph",
        run_id="topograph-none-seed42",
        score_shift=0.0,
        seeding=SeedingEnvelope(seeding_enabled=False, seeding_ladder="none"),
    )
    _write_named_run(
        direct_dir,
        pack_path=TIER_B_PACK_PATH,
        system="topograph",
        run_id="topograph-direct-seed42",
        score_shift=0.03,
        seeding=SeedingEnvelope(
            seeding_enabled=True,
            seeding_ladder="direct",
            seed_source_system="primordia",
            seed_source_run_id="prim-tier-b-seed42",
            seed_artifact_path="seed_candidates.json",
            seed_selected_family="sparse_mlp",
            seed_overlap_policy="family-overlapping",
        ),
    )
    pack = load_parity_pack(TIER_B_PACK_PATH)
    control = SystemIngestor(control_dir)
    direct = SystemIngestor(direct_dir)
    comparison = ComparisonEngine().compare(
        left_manifest=control.load_manifest(),
        left_results=control.load_results(),
        right_manifest=direct.load_manifest(),
        right_results=direct.load_results(),
        pack=pack,
    )

    payload = _transfer_regime_payload(regime="direct", comparison=comparison, pack=pack, seed_gate=None)

    assert payload["verdict"] == "gain"
    assert payload["seed_source"] == "primordia:prim-tier-b-seed42"
    assert payload["seed_overlap_policy"] == "family-overlapping"
    assert payload["gain_count"] == 4
    assert payload["regression_count"] == 0
    assert payload["benchmark_deltas"][0]["benchmark_family"] in {"classification", "regression", "language_modeling"}
    assert all(row["outcome"] in {"gain", "tie"} for row in payload["benchmark_deltas"])


def test_publish_transfer_regime_workspace_aggregates_multi_seed_evidence(tmp_path: Path, monkeypatch) -> None:
    def fake_ensure_primordia_export(**kwargs):
        run_dir = Path(kwargs["run_dir"])
        seed = _extract_seed(run_dir.name)
        _write_named_run(
            run_dir,
            pack_path=TIER_B_PACK_PATH,
            system="primordia",
            run_id=f"primordia-seed{seed}-source",
            score_shift=0.0,
            seeding=SeedingEnvelope(seeding_enabled=False, seeding_ladder="none"),
        )
        (run_dir / "seed_candidates.json").write_text(
            json.dumps(
                {
                    "system": "primordia",
                    "run_id": f"primordia-seed{seed}-source",
                    "seed_candidates": [
                        {
                            "seed_rank": 1,
                            "family": "sparse_mlp",
                            "benchmark_groups": ["classification", "regression"],
                            "benchmark_wins": 2,
                            "repeat_support_count": 4,
                            "seed_overlap_policy": "family-overlapping",
                            "representative_genome_id": "prim-g7",
                            "representative_architecture_summary": "sparse_mlp[64x64]",
                        }
                    ],
                },
                indent=2,
            ),
            encoding="utf-8",
        )

    def fake_ensure_topograph_portable_smoke_export(**kwargs):
        run_dir = Path(kwargs["run_dir"])
        seed = _extract_seed(run_dir.name)
        seeding_ladder = kwargs.get("seeding_ladder")
        if seeding_ladder == "direct":
            score_shift = 0.03
            seeding = SeedingEnvelope(
                seeding_enabled=True,
                seeding_ladder="direct",
                seed_source_system="primordia",
                seed_source_run_id=f"primordia-seed{seed}-source",
                seed_artifact_path=str(Path(kwargs["config_path"]).resolve().parent.parent.parent / "seed_artifacts" / f"seed{seed}_direct_quality.json"),
                seed_selected_family="sparse_mlp",
                seed_overlap_policy="family-overlapping",
            )
        elif seeding_ladder == "staged":
            score_shift = 0.0 if seed == 41 else -0.02
            seeding = SeedingEnvelope(
                seeding_enabled=True,
                seeding_ladder="staged",
                seed_source_system="topograph",
                seed_source_run_id=f"topograph-direct-seed{seed}",
                seed_artifact_path=str(run_dir.parents[1] / "seed_artifacts" / f"seed{seed}_staged_seed_candidates.json"),
                seed_selected_family="sparse_mlp",
                seed_overlap_policy="benchmark-overlapping",
            )
        else:
            score_shift = 0.0
            seeding = SeedingEnvelope(seeding_enabled=False, seeding_ladder="none")
        regime_name = "none" if seeding_ladder is None else str(seeding_ladder)
        _write_named_run(
            run_dir,
            pack_path=TIER_B_PACK_PATH,
            system="topograph",
            run_id=f"topograph-{regime_name}-seed{seed}",
            score_shift=score_shift,
            seeding=seeding,
        )

    def fake_refresh_workspace_reports(*, workspace):
        workspace = Path(workspace)
        return {
            "workspace": str(workspace),
            "summary_count": 6,
            "trend_dataset": str(workspace / "trends" / "fair_matrix_trend_rows.jsonl"),
            "trend_report": str(workspace / "trends" / "fair_matrix_trends.md"),
            "trend_report_data": str(workspace / "trends" / "fair_matrix_trends.json"),
            "dashboard": str(workspace / "fair_matrix_dashboard.html"),
            "dashboard_data": str(workspace / "fair_matrix_dashboard.json"),
        }

    monkeypatch.setattr(
        "evonn_compare.orchestration.transfer_regimes.ensure_primordia_export",
        fake_ensure_primordia_export,
    )
    monkeypatch.setattr(
        "evonn_compare.orchestration.transfer_regimes.ensure_topograph_portable_smoke_export",
        fake_ensure_topograph_portable_smoke_export,
    )
    monkeypatch.setattr(
        "evonn_compare.orchestration.transfer_regimes.refresh_workspace_reports",
        fake_refresh_workspace_reports,
    )

    artifacts = publish_transfer_regime_workspace(
        workspace=tmp_path / "workspace",
        pack_name="tier_b_core",
        seeds=[41, 42],
        budget=64,
        primordia_root=REPO_ROOT / "EvoNN-Primordia",
        topograph_root=REPO_ROOT / "EvoNN-Topograph",
    )

    summary = json.loads(Path(str(artifacts["transfer_summary_data"])).read_text(encoding="utf-8"))
    direct = summary["regimes"]["direct"]
    staged = summary["regimes"]["staged"]
    staged_seed1 = json.loads((tmp_path / "workspace" / "reports" / "seed41" / "03-staged_vs_control.json").read_text(encoding="utf-8"))

    assert direct["case_count"] == 2
    assert direct["consensus"] == "gain"
    assert staged["case_count"] == 2
    assert staged["consensus"] == "inconclusive"
    assert staged_seed1["seed_source"] == "topograph:topograph-direct-seed41"
    assert staged_seed1["seed_overlap_policy"] == "benchmark-overlapping"
    assert staged_seed1["benchmark_deltas"][0]["benchmark_family"] in {"classification", "regression", "language_modeling"}
    assert staged_seed1["seed_quality"]["gate_path"].endswith("seed41_staged_quality.json")
    assert staged_seed1["seed_quality"]["artifact_path"].endswith("seed41_staged_seed_candidates.json")
    assert staged_seed1["seed_quality"]["benchmark_wins"] == 4
    assert (tmp_path / "workspace" / "seed_artifacts" / "seed41_direct_quality.json").exists()
    assert (tmp_path / "workspace" / "seed_artifacts" / "seed41_staged_quality.json").exists()


def _write_named_run(
    run_dir: Path,
    *,
    pack_path: Path,
    system: str,
    run_id: str,
    score_shift: float,
    seeding: SeedingEnvelope,
) -> None:
    pack = load_parity_pack(pack_path)
    run_dir.mkdir(parents=True, exist_ok=True)
    (run_dir / "config.yaml").write_text("seed: 42\n", encoding="utf-8")
    (run_dir / "report.md").write_text("# Report\n", encoding="utf-8")
    manifest = RunManifest(
        schema_version="1.0",
        system=system,
        run_id=run_id,
        run_name=run_id,
        created_at=datetime(2026, 4, 1, tzinfo=timezone.utc),
        pack_name=pack.name,
        seed=_extract_seed(run_id),
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
            evaluation_count=64,
            epochs_per_candidate=pack.budget_policy.epochs_per_candidate,
            budget_policy_name="prototype_equal_budget",
            actual_evaluations=64,
            evaluation_semantics="one candidate evaluation counted at the compare surface",
        ),
        device=DeviceInfo(
            device_name="portable-test",
            precision_mode="fp32",
            framework="portable-sklearn",
            framework_version="1.7-test",
        ),
        artifacts=ArtifactPaths(
            config_snapshot="config.yaml",
            report_markdown="report.md",
        ),
        seeding=seeding,
        fairness={
            "benchmark_pack_id": pack.name,
            "seed": _extract_seed(run_id),
            "evaluation_count": 64,
            "budget_policy_name": "prototype_equal_budget",
            "data_signature": "shared-tier-b-signature",
            "code_version": "deadbeefcafebabe",
        },
    )
    results = []
    for index, entry in enumerate(pack.benchmarks):
        metric_value = 0.80 + score_shift + (0.01 * index) if entry.metric_direction == "max" else 0.20 - score_shift + (0.01 * index)
        results.append(
            ResultRecord(
                system=system,
                run_id=run_id,
                benchmark_id=entry.benchmark_id,
                metric_name=entry.metric_name,
                metric_direction=entry.metric_direction,
                metric_value=metric_value,
                quality=metric_value if entry.metric_direction == "max" else -metric_value,
                architecture_summary="sparse_mlp[64x64]",
                genome_id=f"{system}-{index+1}",
                status="ok",
            )
        )
    (run_dir / "manifest.json").write_text(manifest.model_dump_json(indent=2), encoding="utf-8")
    (run_dir / "results.json").write_text(
        json.dumps([result.model_dump(mode="json") for result in results], indent=2),
        encoding="utf-8",
    )
    (run_dir / "summary.json").write_text(
        json.dumps(
            {
                "run_id": run_id,
                "system": system,
                "pack_name": pack.name,
                "best_results": [result.model_dump(mode="json") for result in results],
            },
            indent=2,
        ),
        encoding="utf-8",
    )


def _extract_seed(value: str) -> int:
    marker = "seed"
    if marker not in value:
        return 42
    suffix = value.split(marker, 1)[1]
    digits = "".join(ch for ch in suffix if ch.isdigit())
    return int(digits) if digits else 42
