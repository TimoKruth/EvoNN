from stratograph.benchmarks import get_benchmark
from stratograph.benchmarks.lm import available_lm_caches, resolve_lm_cache_path, warm_lm_cache
from stratograph.genome import HierarchicalGenome
from stratograph.pipeline.evaluator import evaluate_candidate_with_state


def test_shared_lm_cache_resolves() -> None:
    path = resolve_lm_cache_path("tinystories_lm")
    assert path.name == "tinystories_lm.npz"
    assert path.exists()


def test_warm_lm_cache_copies_to_target(tmp_path, monkeypatch) -> None:
    monkeypatch.setenv("STRATOGRAPH_LM_CACHE_DIR", str(tmp_path / "missing"))
    copied = warm_lm_cache(["tinystories_lm"], target_dir=tmp_path / "repo_cache")
    assert copied[0].exists()
    assert copied[0].name == "tinystories_lm.npz"


def test_available_lm_caches_lists_canonical() -> None:
    names = available_lm_caches()
    assert "tinystories_lm" in names
    assert "wikitext2_lm" in names


def test_evaluator_returns_inheritable_training_state() -> None:
    spec = get_benchmark("moons")
    genome = HierarchicalGenome.create_seed(
        benchmark_name=spec.name,
        task=spec.task,
        input_dim=spec.model_input_dim,
        output_dim=spec.model_output_dim,
        seed=42,
    )
    data = spec.load_data(seed=42)
    first = evaluate_candidate_with_state(genome, spec, data=data)
    second = evaluate_candidate_with_state(genome, spec, data=data, inherited_state=first.training_artifact)
    assert first.record.status == "ok"
    assert second.record.status == "ok"
    assert first.training_artifact is not None
    assert first.training_artifact.task == "classification"
    assert first.training_artifact.model_name == "neural_classifier"


def test_lm_evaluator_returns_neural_head() -> None:
    spec = get_benchmark("tiny_lm_synthetic")
    genome = HierarchicalGenome.create_seed(
        benchmark_name=spec.name,
        task=spec.task,
        input_dim=spec.model_input_dim,
        output_dim=spec.model_output_dim,
        seed=42,
    )
    data = spec.load_data(seed=42)
    result = evaluate_candidate_with_state(genome, spec, data=data)
    assert result.record.status == "ok"
    assert result.training_artifact is not None
    assert result.training_artifact.model_name == "neural_lm_head"
