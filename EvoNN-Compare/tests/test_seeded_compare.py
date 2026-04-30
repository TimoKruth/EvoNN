from pathlib import Path

from evonn_compare.comparison.engine import ComparisonEngine
from evonn_compare.contracts.parity import load_parity_pack
from evonn_compare.ingest.loader import SystemIngestor
from evonn_compare.orchestration.seeded_compare import _build_lane_metadata, _seeded_vs_unseeded_payload
from evonn_shared.contracts import SeedingEnvelope
from test_compare import PACK_PATH, _write_run


def test_seeded_compare_marks_portable_boundary_as_plumbing_only(tmp_path: Path) -> None:
    pack = load_parity_pack(PACK_PATH)
    primordia_dir = tmp_path / "primordia"
    unseeded_dir = tmp_path / "topograph-unseeded"
    seeded_dir = tmp_path / "topograph-seeded"

    _write_run(primordia_dir, system="primordia")
    _write_run(unseeded_dir, system="topograph", framework="portable-sklearn", framework_version="1.7-portable")
    _write_run(
        seeded_dir,
        system="topograph",
        score_shift=0.02,
        framework="portable-sklearn",
        framework_version="1.7-portable",
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

    primordia = SystemIngestor(primordia_dir)
    unseeded = SystemIngestor(unseeded_dir)
    seeded = SystemIngestor(seeded_dir)
    comparison = ComparisonEngine().compare(
        left_manifest=unseeded.load_manifest(),
        left_results=unseeded.load_results(),
        right_manifest=seeded.load_manifest(),
        right_results=seeded.load_results(),
        pack=pack,
    )
    lane = _build_lane_metadata(
        pack_name=pack.name,
        budget=64,
        seed=42,
        runs={
            "primordia": (primordia.load_manifest(), primordia.load_results()),
            "topograph": (seeded.load_manifest(), seeded.load_results()),
        },
        comparison=comparison,
        seeded=True,
    )
    payload = _seeded_vs_unseeded_payload(comparison)

    assert lane.operating_state == "portable-transfer-plumbing"
    assert lane.repeatability_ready is False
    assert any("not native MLX transfer behavior" in note for note in lane.acceptance_notes)
    assert lane.system_operating_states["topograph"] == "portable-smoke"
    assert payload["portable_backend"] == "portable-sklearn"
    assert payload["transfer_boundary"] == "portable-topograph-seeding-contract"
    assert payload["transfer_proof_state"] == "portable-plumbing-only"
