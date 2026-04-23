from pathlib import Path
import subprocess

import pytest
import yaml

from evonn_compare.orchestration import benchmark_resolution
from evonn_compare.orchestration.benchmark_resolution import resolve_supported_benchmark_id
from evonn_compare.orchestration.config_gen import (
    generate_budget_pack,
    generate_prism_config,
    generate_topograph_config,
)
from evonn_compare.orchestration.fair_matrix import (
    generate_contender_config,
    generate_primordia_config,
    generate_stratograph_config,
    prepare_fair_matrix_cases,
)


def test_generate_budget_pack_sets_campaign_budget(tmp_path: Path) -> None:
    base_pack = Path(__file__).resolve().parents[1] / "parity_packs" / "tier1_core.yaml"
    output = generate_budget_pack(base_pack_path=base_pack, budget=128, output_dir=tmp_path)
    payload = yaml.safe_load(output.read_text(encoding="utf-8"))
    assert payload["name"] == "tier1_core_eval128"
    assert payload["budget_policy"]["evaluation_count"] == 128


def test_resolve_supported_benchmark_id_probes_target_project_environment_when_local_loader_unavailable(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    entry = type(
        "Entry",
        (),
        {
            "benchmark_id": "credit_g_classification",
            "native_ids": {"prism": "credit_g", "evonn": "credit_g_classification"},
        },
    )()

    benchmark_resolution._get_benchmark_loader.cache_clear()
    monkeypatch.setattr(benchmark_resolution, "_get_benchmark_loader", lambda system: None)

    calls: list[dict[str, object]] = []

    def fake_run(argv, *, cwd, stdout, stderr, check):
        calls.append({"argv": argv, "cwd": cwd, "check": check})
        code = 0 if "loader('credit_g')" in argv[-1] else 1
        return subprocess.CompletedProcess(argv, code)

    monkeypatch.setattr(benchmark_resolution.subprocess, "run", fake_run)

    assert resolve_supported_benchmark_id(entry, "prism") == "credit_g"
    assert calls[0]["cwd"] == benchmark_resolution._PROJECT_ROOTS["prism"]


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
    assert topograph_payload["benchmark_pool"]["benchmarks"][0] in {"iris", "iris_classification"}
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

    assert stratograph_payload["benchmark_pool"]["benchmarks"][0] in {"iris", "iris_classification"}
    assert (
        stratograph_payload["evolution"]["population_size"]
        * stratograph_payload["evolution"]["generations"]
        * len(stratograph_payload["benchmark_pool"]["benchmarks"])
        == 64
    )
    assert contender_payload["baseline"]["mode"] == "budget_matched"
    assert contender_payload["baseline"]["target_evaluation_count"] == 64


def test_generate_smoke_configs_resolve_supported_benchmark_ids_across_systems(tmp_path: Path) -> None:
    base_pack = Path(__file__).resolve().parents[1] / "parity_packs" / "tier1_core_smoke.yaml"
    pack_path = generate_budget_pack(base_pack_path=base_pack, budget=16, output_dir=tmp_path / "packs")

    prism_path = generate_prism_config(
        output_path=tmp_path / "configs" / "prism.yaml",
        pack_path=pack_path,
        seed=42,
        budget=16,
    )
    stratograph_path = generate_stratograph_config(
        output_path=tmp_path / "configs" / "stratograph.yaml",
        pack_path=pack_path,
        seed=42,
        budget=16,
    )
    primordia_path = generate_primordia_config(
        output_path=tmp_path / "configs" / "primordia.yaml",
        pack_path=pack_path,
        seed=42,
        budget=16,
        run_name="demo",
    )
    contender_path = generate_contender_config(
        output_path=tmp_path / "configs" / "contenders.yaml",
        pack_path=pack_path,
        seed=42,
        budget=16,
        run_name="demo",
    )

    prism_payload = yaml.safe_load(prism_path.read_text(encoding="utf-8"))
    stratograph_payload = yaml.safe_load(stratograph_path.read_text(encoding="utf-8"))
    primordia_payload = yaml.safe_load(primordia_path.read_text(encoding="utf-8"))
    contender_payload = yaml.safe_load(contender_path.read_text(encoding="utf-8"))

    assert "friedman1" in prism_payload["benchmark_pack"]["benchmark_ids"]
    assert "credit_g" in prism_payload["benchmark_pack"]["benchmark_ids"]
    assert "diabetes" in stratograph_payload["benchmark_pool"]["benchmarks"]
    assert "friedman1" in stratograph_payload["benchmark_pool"]["benchmarks"]
    assert "diabetes" in primordia_payload["benchmark_pool"]["benchmarks"]
    assert "credit_g" in contender_payload["benchmark_pool"]["benchmarks"]


def test_generate_primordia_config_sets_training_epochs_from_pack_budget(tmp_path: Path) -> None:
    base_pack = Path(__file__).resolve().parents[1] / "parity_packs" / "tier1_core.yaml"
    pack_path = generate_budget_pack(base_pack_path=base_pack, budget=64, output_dir=tmp_path / "packs")
    pack_payload = yaml.safe_load(pack_path.read_text(encoding="utf-8"))

    primordia_path = generate_primordia_config(
        output_path=tmp_path / "configs" / "primordia.yaml",
        pack_path=pack_path,
        seed=42,
        budget=64,
        run_name="demo",
    )

    primordia_payload = yaml.safe_load(primordia_path.read_text(encoding="utf-8"))

    assert primordia_payload["search"]["mode"] == "budget_matched"
    assert primordia_payload["search"]["target_evaluation_count"] == 64
    assert primordia_payload["training"]["epochs_per_candidate"] == pack_payload["budget_policy"]["epochs_per_candidate"]


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


def test_prepare_fair_matrix_cases_can_skip_contenders(tmp_path: Path) -> None:
    base_pack = Path(__file__).resolve().parents[1] / "parity_packs" / "tier1_core.yaml"
    _paths, cases = prepare_fair_matrix_cases(
        pack_name="tier1_core",
        base_pack_path=base_pack,
        seeds=[42],
        budgets=[64],
        workspace=tmp_path / "matrix",
        prism_root=tmp_path / "Prism",
        topograph_root=tmp_path / "Topograph",
        stratograph_root=tmp_path / "Stratograph",
        contenders_root=tmp_path / "Contenders",
        include_contenders=False,
    )

    case = cases[0]
    assert case.prism_config_path.exists()
    assert case.topograph_config_path.exists()
    assert case.stratograph_config_path.exists()
    assert case.contender_config_path is None
    assert case.contender_run_dir is None
