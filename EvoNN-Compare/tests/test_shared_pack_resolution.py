from pathlib import Path

from evonn_compare.contracts.parity import load_parity_pack, resolve_pack_path
from evonn_compare.orchestration.benchmark_audit import audit_benchmark_pack


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
        "tinystories_lm",
    ]


def test_expanded_ladder_pack_metadata_loads() -> None:
    pack = load_parity_pack("tier_b_core_v2")
    cumulative = load_parity_pack("tier_b_core_v2_cumulative")

    assert pack.name == "tier_b_core_v2"
    assert pack.ladder_tier == "B"
    assert len(pack.benchmarks) == 6
    assert pack.benchmarks[0].minimum_required_contenders
    assert pack.benchmarks[0].score_ceiling == 1.0

    assert cumulative.name == "tier_b_core_v2_cumulative"
    assert cumulative.include_packs == ("tier_a_contract", "tier_b_core_v2")
    assert len(cumulative.benchmarks) == 14
    assert len({entry.benchmark_id for entry in cumulative.benchmarks}) == 14


def test_tier_d_broad_pack_is_promoted_after_repeated_clean_proofs() -> None:
    pack = load_parity_pack("tier_d_broad_shared")
    cumulative = load_parity_pack("tier_d_broad_shared_cumulative")

    assert pack.ladder_tier == "D"
    assert len(pack.benchmarks) == 5
    assert pack.promotion_requirements["promotion_status"] == "decision_grade"

    audit = audit_benchmark_pack(pack_name="tier_d_broad_shared")
    assert audit["summary"]["blocked_count"] == 0
    assert audit["summary"]["exploratory_count"] == 0
    assert audit["summary"]["decision_grade_count"] == 5
    assert audit["summary"]["audit_status"] == "passed"

    assert cumulative.ladder_tier == "D"
    assert cumulative.include_packs == ("tier_c_architecture_sensitive_cumulative", "tier_d_broad_shared")
    assert len(cumulative.benchmarks) == 27
    assert len({entry.benchmark_id for entry in cumulative.benchmarks}) == 27
