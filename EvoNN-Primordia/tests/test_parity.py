from __future__ import annotations

from pathlib import Path

from evonn_primordia.benchmarks.parity import fallback_native_id, load_parity_pack, resolve_pack_path


REPO_ROOT = Path(__file__).resolve().parents[2]


def test_shared_parity_pack_resolves_from_shared_benchmarks() -> None:
    resolved = resolve_pack_path("shared_33plus5")
    assert resolved == REPO_ROOT / "shared-benchmarks" / "suites" / "parity" / "shared_33plus5.yaml"


def test_simple_pack_uses_primordia_native_ids(tmp_path) -> None:
    pack_path = tmp_path / "mini_pack.yaml"
    pack_path.write_text(
        """
name: mini_pack
benchmarks:
  - iris
  - tiny_lm_synthetic
""".strip()
        + "\n",
        encoding="utf-8",
    )

    pack = load_parity_pack(pack_path)

    assert pack.name == "mini_pack"
    assert [entry.native_ids for entry in pack.benchmarks] == [
        {"primordia": "iris"},
        {"primordia": "tiny_lm_synthetic"},
    ]


def test_tier1_compare_pack_preserves_regression_native_ids_for_primordia() -> None:
    pack_path = REPO_ROOT / "EvoNN-Compare" / "parity_packs" / "tier1_core_smoke.yaml"

    pack = load_parity_pack(pack_path)
    native_by_id = {entry.benchmark_id: fallback_native_id(entry, "primordia") for entry in pack.benchmarks}

    assert native_by_id["diabetes_regression"] == "diabetes"
    assert native_by_id["friedman1_regression"] == "friedman1"
