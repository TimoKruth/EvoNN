import json

from stratograph.analysis import analyze_run_motifs, run_ablation_matrix, run_ablation_suite
from stratograph.analysis.ablation import AblationPack
from stratograph.config import BenchmarkPoolConfig, load_config
from stratograph.genome import dict_to_genome
from stratograph.pipeline import run_evolution


def test_resume_checkpoint_and_best_genomes(repo_root, tmp_path) -> None:
    config_path = repo_root / "configs" / "working_33_plus_5_lm_smoke.yaml"
    config = load_config(config_path).model_copy(
        update={
            "benchmark_pool": BenchmarkPoolConfig(name="mini", benchmarks=["moons", "tiny_lm_synthetic"]),
            "run_name": "resume_test",
        }
    )
    run_dir = tmp_path / "resume_test"
    run_evolution(config, run_dir=run_dir, config_path=config_path, resume=False)
    checkpoint = json.loads((run_dir / "checkpoint.json").read_text(encoding="utf-8"))
    assert checkpoint["completed_count"] == 2
    assert (run_dir / "best_genomes" / "moons.json").exists()
    run_evolution(config, run_dir=run_dir, config_path=config_path, resume=True)
    status = json.loads((run_dir / "status.json").read_text(encoding="utf-8"))
    assert status["state"] == "completed"
    assert status["completed_count"] == 2


def test_two_level_shared_run_keeps_reuse(repo_root, tmp_path) -> None:
    config_path = repo_root / "configs" / "working_33_plus_5_lm_smoke.yaml"
    config = load_config(config_path).model_copy(
        update={
            "benchmark_pool": BenchmarkPoolConfig(name="mini", benchmarks=["moons"]),
            "run_name": "shared_reuse_test",
            "evolution": load_config(config_path).evolution.model_copy(update={"architecture_mode": "two_level_shared"}),
        }
    )
    run_dir = tmp_path / "shared_reuse_test"
    run_evolution(config, run_dir=run_dir, config_path=config_path)
    payload = json.loads((run_dir / "best_genomes" / "moons.json").read_text(encoding="utf-8"))
    genome = dict_to_genome(payload["genome"])
    assert genome.reuse_ratio > 0.0


def test_ablation_suite_outputs_report(repo_root, tmp_path) -> None:
    config_path = repo_root / "configs" / "working_33_plus_5_lm_smoke.yaml"
    config = load_config(config_path).model_copy(
        update={
            "benchmark_pool": BenchmarkPoolConfig(name="mini", benchmarks=["moons"]),
            "run_name": "ablate_test",
        }
    )
    report_path = run_ablation_suite(config, workspace=tmp_path / "ablation", config_path=config_path)
    payload = json.loads((report_path.with_suffix(".json")).read_text(encoding="utf-8"))
    assert report_path.exists()
    assert payload["variants"] == [
        "flat_macro",
        "two_level_unshared",
        "two_level_shared",
        "two_level_shared_no_clone",
        "two_level_shared_no_motif_bias",
    ]


def test_ablation_matrix_outputs_report(repo_root, tmp_path) -> None:
    config_path = repo_root / "configs" / "working_33_plus_5_lm_smoke.yaml"
    config = load_config(config_path).model_copy(update={"run_name": "matrix_test"})
    report_path = run_ablation_matrix(
        config,
        workspace=tmp_path / "matrix",
        config_path=config_path,
        variants=["flat_macro", "two_level_shared"],
        packs={
            "mini_pack": AblationPack(
                name="mini_pack",
                benchmarks=["moons"],
                population_size=2,
                generations=1,
                description="mini",
            )
        },
        include_mixed_from_config=False,
    )
    payload = json.loads((report_path.with_suffix(".json")).read_text(encoding="utf-8"))
    assert report_path.exists()
    assert "mini_pack" in payload["packs"]


def test_motif_analysis_outputs_report(repo_root, tmp_path) -> None:
    config_path = repo_root / "configs" / "working_33_plus_5_lm_smoke.yaml"
    config = load_config(config_path).model_copy(
        update={
            "benchmark_pool": BenchmarkPoolConfig(name="mini", benchmarks=["moons"]),
            "run_name": "motif_test",
        }
    )
    run_dir = tmp_path / "motif_test"
    run_evolution(config, run_dir=run_dir, config_path=config_path)
    report_path = analyze_run_motifs(run_dir)
    payload = json.loads((run_dir / "motifs_report.json").read_text(encoding="utf-8"))
    assert report_path.exists()
    assert payload["total_unique_motifs"] >= 1
