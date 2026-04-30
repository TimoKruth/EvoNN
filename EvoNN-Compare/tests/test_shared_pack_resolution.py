from pathlib import Path

from evonn_compare.contracts.parity import load_parity_pack, resolve_pack_path


REPO_ROOT = Path(__file__).resolve().parents[2]


def test_tier_b_core_resolves_from_shared_benchmarks() -> None:
    resolved = resolve_pack_path("tier_b_core")

    assert resolved == REPO_ROOT / "shared-benchmarks" / "suites" / "parity" / "tier_b_core.yaml"


def test_tier_b_core_loads_expected_benchmark_families() -> None:
    pack = load_parity_pack("tier_b_core")

    assert pack.name == "tier_b_core"
    assert [entry.benchmark_id for entry in pack.benchmarks] == [
        "openml_gas_sensor",
        "openml_cpu_activity",
        "fashionmnist_image",
        "tinystories_lm_smoke",
    ]
