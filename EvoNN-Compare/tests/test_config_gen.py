from pathlib import Path

import yaml

from evonn_compare.orchestration.config_gen import (
    generate_budget_pack,
    generate_prism_config,
    generate_topograph_config,
)
from evonn_compare.orchestration.fair_matrix import (
    generate_contender_config,
    generate_stratograph_config,
    prepare_fair_matrix_cases,
)


def test_generate_budget_pack_sets_campaign_budget(tmp_path: Path) -> None:
    base_pack = Path(__file__).resolve().parents[1] / "parity_packs" / "tier1_core.yaml"
    output = generate_budget_pack(base_pack_path=base_pack, budget=128, output_dir=tmp_path)
    payload = yaml.safe_load(output.read_text(encoding="utf-8"))
    assert payload["name"] == "tier1_core_eval128"
    assert payload["budget_policy"]["evaluation_count"] == 128


def test_generate_prism_and_topograph_configs_use_legacy_slots(tmp_path: Path) -> None:
    base_pack = Path(__file__).resolve().parents[1] / "parity_packs" / "tier1_core.yaml"
    pack_path = generate_budget_pack(base_pack_path=base_pack, budget=64, output_dir=tmp_path / "packs")

    prism_path = generate_prism_config(
        output_path=tmp_path / "configs" / "prism.yaml",
        pack_path=pack_path,
        seed=42,
        budget=64,
    )
    topograph_path = generate_topograph_config(
        output_path=tmp_path / "configs" / "topograph.yaml",
        pack_path=pack_path,
        seed=42,
        budget=64,
        run_dir=tmp_path / "runs" / "topograph",
    )

    prism_payload = yaml.safe_load(prism_path.read_text(encoding="utf-8"))
    topograph_payload = yaml.safe_load(topograph_path.read_text(encoding="utf-8"))

    assert prism_payload["benchmark_pack"]["benchmark_ids"][0] == "iris_classification"
    assert topograph_payload["benchmark_pool"]["benchmarks"][0] == "iris"
    assert prism_payload["evolution"]["num_generations"] == 1
    assert topograph_payload["training"]["parallel_workers"] == 2
def test_generate_prism_config_uses_mlp_and_attention_for_mixed_lm_pack(tmp_path: Path) -> None:
    pack_path = tmp_path / "mixed.yaml"
    pack_path.write_text(
        yaml.safe_dump(
            {
                "name": "mixed_pack",
                "tier": 1,
                "description": "mixed",
                "benchmarks": [
                    {
                        "benchmark_id": "iris_classification",
                        "native_ids": {"prism": "iris"},
                        "task_kind": "classification",
                        "metric_name": "accuracy",
                        "metric_direction": "max",
                    },
                    {
                        "benchmark_id": "tiny_lm_synthetic",
                        "native_ids": {"prism": "tiny_lm_synthetic"},
                        "task_kind": "language_modeling",
                        "metric_name": "perplexity",
                        "metric_direction": "min",
                    },
                ],
                "budget_policy": {
                    "evaluation_count": 4,
                    "epochs_per_candidate": 1,
                },
                "seed_policy": {
                    "mode": "campaign",
                    "required": True,
                },
            },
            sort_keys=False,
        ),
        encoding="utf-8",
    )
    prism_path = generate_prism_config(
        output_path=tmp_path / "configs" / "prism.yaml",
        pack_path=pack_path,
        seed=42,
        budget=4,
    )
    prism_payload = yaml.safe_load(prism_path.read_text(encoding="utf-8"))
    assert prism_payload["evolution"]["allowed_families"] == ["mlp", "attention"]


def test_generate_prism_config_uses_broad_families_when_mixed_lm_budget_allows_it(tmp_path: Path) -> None:
    pack_path = tmp_path / "mixed.yaml"
    pack_path.write_text(
        yaml.safe_dump(
            {
                "name": "mixed_pack",
                "tier": 1,
                "description": "mixed",
                "benchmarks": [
                    {
                        "benchmark_id": "iris_classification",
                        "native_ids": {"prism": "iris"},
                        "task_kind": "classification",
                        "metric_name": "accuracy",
                        "metric_direction": "max",
                    },
                    {
                        "benchmark_id": "tiny_lm_synthetic",
                        "native_ids": {"prism": "tiny_lm_synthetic"},
                        "task_kind": "language_modeling",
                        "metric_name": "perplexity",
                        "metric_direction": "min",
                    },
                ],
                "budget_policy": {
                    "evaluation_count": 8,
                    "epochs_per_candidate": 1,
                },
                "seed_policy": {
                    "mode": "campaign",
                    "required": True,
                },
            },
            sort_keys=False,
        ),
        encoding="utf-8",
    )
    prism_path = generate_prism_config(
        output_path=tmp_path / "configs" / "prism.yaml",
        pack_path=pack_path,
        seed=42,
        budget=8,
    )
    prism_payload = yaml.safe_load(prism_path.read_text(encoding="utf-8"))
    assert prism_payload["evolution"]["allowed_families"] == [
        "mlp",
        "sparse_mlp",
        "attention",
        "sparse_attention",
    ]


def test_generate_prism_config_rejects_impossible_mixed_lm_budget(tmp_path: Path) -> None:
    pack_path = tmp_path / "mixed.yaml"
    pack_path.write_text(
        yaml.safe_dump(
            {
                "name": "mixed_pack",
                "tier": 1,
                "description": "mixed",
                "benchmarks": [
                    {
                        "benchmark_id": "iris_classification",
                        "native_ids": {"prism": "iris"},
                        "task_kind": "classification",
                        "metric_name": "accuracy",
                        "metric_direction": "max",
                    },
                    {
                        "benchmark_id": "tiny_lm_synthetic",
                        "native_ids": {"prism": "tiny_lm_synthetic"},
                        "task_kind": "language_modeling",
                        "metric_name": "perplexity",
                        "metric_direction": "min",
                    },
                ],
                "budget_policy": {
                    "evaluation_count": 2,
                    "epochs_per_candidate": 1,
                },
                "seed_policy": {
                    "mode": "campaign",
                    "required": True,
                },
            },
            sort_keys=False,
        ),
        encoding="utf-8",
    )
    try:
        generate_prism_config(
            output_path=tmp_path / "configs" / "prism.yaml",
            pack_path=pack_path,
            seed=42,
            budget=2,
        )
    except ValueError as exc:
        assert "too small for prism pack coverage" in str(exc)
    else:
        raise AssertionError("expected ValueError")


def test_generate_stratograph_and_contender_configs_match_budget(tmp_path: Path) -> None:
    base_pack = Path(__file__).resolve().parents[1] / "parity_packs" / "tier1_core.yaml"
    pack_path = generate_budget_pack(base_pack_path=base_pack, budget=64, output_dir=tmp_path / "packs")

    stratograph_path = generate_stratograph_config(
        output_path=tmp_path / "configs" / "stratograph.yaml",
        pack_path=pack_path,
        seed=42,
        budget=64,
    )
    contender_path = generate_contender_config(
        output_path=tmp_path / "configs" / "contenders.yaml",
        pack_path=pack_path,
        seed=42,
        budget=64,
        run_name="demo",
    )

    stratograph_payload = yaml.safe_load(stratograph_path.read_text(encoding="utf-8"))
    contender_payload = yaml.safe_load(contender_path.read_text(encoding="utf-8"))

    assert stratograph_payload["benchmark_pool"]["benchmarks"][0] == "iris"
    assert (
        stratograph_payload["evolution"]["population_size"]
        * stratograph_payload["evolution"]["generations"]
        * len(stratograph_payload["benchmark_pool"]["benchmarks"])
        == 64
    )
    assert contender_payload["baseline"]["mode"] == "budget_matched"
    assert contender_payload["baseline"]["target_evaluation_count"] == 64


def test_prepare_fair_matrix_cases_writes_all_system_configs(tmp_path: Path) -> None:
    base_pack = Path(__file__).resolve().parents[1] / "parity_packs" / "tier1_core.yaml"
    paths, cases = prepare_fair_matrix_cases(
        pack_name="tier1_core",
        base_pack_path=base_pack,
        seeds=[42],
        budgets=[64],
        workspace=tmp_path / "matrix",
        prism_root=tmp_path / "Prism",
        topograph_root=tmp_path / "Topograph",
        stratograph_root=tmp_path / "Stratograph",
        contenders_root=tmp_path / "Contenders",
    )

    assert paths.manifest_path.exists()
    assert len(cases) == 1
    case = cases[0]
    assert case.prism_config_path.exists()
    assert case.topograph_config_path.exists()
    assert case.stratograph_config_path.exists()
    assert case.contender_config_path.exists()
