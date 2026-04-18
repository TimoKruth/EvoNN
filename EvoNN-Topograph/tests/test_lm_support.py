from __future__ import annotations

import importlib.util
import random
import subprocess
from pathlib import Path

import mlx.core as mx
import numpy as np

from topograph.benchmarks.parity import get_benchmark
from topograph.benchmarks.registry import DatasetRegistry
from topograph.benchmarks.spec import BenchmarkSpec
from topograph.config import load_config
from topograph.export.symbiosis import _benchmark_metric_direction, _benchmark_metric_name
from topograph.genome.genome import Genome, InnovationCounter
from topograph.nn.compiler import compile_genome
from topograph.nn.train import _compute_metric
from topograph.pipeline.coordinator import _create_seed_population
from topograph.pipeline.evaluate import BenchmarkDataCache, EvaluationMemo, GenerationState, _resolve_model_output_dim, evaluate_pool
from topograph.cache import WeightCache


def _write_lm_cache(root: Path, dataset: str) -> None:
    cache_dir = root / "datasets"
    cache_dir.mkdir(parents=True, exist_ok=True)
    np.savez(
        cache_dir / f"{dataset}.npz",
        x_train=np.arange(600 * 256, dtype=np.int32).reshape(600, 256),
        y_train=np.arange(600 * 256, dtype=np.int64).reshape(600, 256),
        x_val=np.arange(200 * 256, dtype=np.int32).reshape(200, 256),
        y_val=np.arange(200 * 256, dtype=np.int64).reshape(200, 256),
    )


def test_catalog_lists_language_modeling_benchmarks():
    registry = DatasetRegistry()
    names = {meta.name for meta in registry.list(task="language_modeling")}
    assert {"tiny_lm_synthetic", "tinystories_lm", "wikitext2_lm"} <= names


def test_tiny_lm_synthetic_loads_sequence_targets():
    spec = get_benchmark("tiny_lm_synthetic")
    x_train, y_train, x_val, y_val = spec.load_data(seed=7, validation_split=0.2)

    assert spec.task == "language_modeling"
    assert x_train.ndim == 2
    assert y_train.ndim == 2
    assert x_train.shape[1] == 128
    assert y_train.shape[1] == 128
    assert x_train.dtype == np.int32
    assert y_train.dtype == np.int64
    assert x_train.shape[0] > x_val.shape[0]


def test_lm_cache_spec_accepts_explicit_npz_path(tmp_path: Path):
    cache_path = tmp_path / "toy_lm.npz"
    np.savez(
        cache_path,
        x_train=np.arange(24, dtype=np.int32).reshape(3, 8),
        y_train=np.arange(24, dtype=np.int64).reshape(3, 8),
        x_val=np.arange(16, dtype=np.int32).reshape(2, 8),
        y_val=np.arange(16, dtype=np.int64).reshape(2, 8),
    )

    spec = BenchmarkSpec(
        name="toy_lm",
        task="language_modeling",
        source="lm_cache",
        dataset=str(cache_path),
        input_dim=8,
        num_classes=32,
    )

    x_train, y_train, x_val, y_val = spec.load_data()
    assert x_train.shape == (3, 8)
    assert y_train.shape == (3, 8)
    assert x_val.shape == (2, 8)
    assert y_val.shape == (2, 8)


def test_tinystories_smoke_catalog_caps_shared_cache(tmp_path: Path, monkeypatch):
    _write_lm_cache(tmp_path, "tinystories_lm")
    monkeypatch.setenv("TOPOGRAPH_LM_CACHE_DIR", str(tmp_path))
    spec = get_benchmark("tinystories_lm_smoke")
    x_train, y_train, x_val, y_val = spec.load_data(seed=42)

    assert spec.task == "language_modeling"
    assert x_train.shape == (512, 256)
    assert y_train.shape == (512, 256)
    assert x_val.shape == (128, 256)
    assert y_val.shape == (128, 256)


def test_wikitext2_smoke_catalog_caps_shared_cache(tmp_path: Path, monkeypatch):
    _write_lm_cache(tmp_path, "wikitext2_lm")
    monkeypatch.setenv("TOPOGRAPH_LM_CACHE_DIR", str(tmp_path))
    spec = get_benchmark("wikitext2_lm_smoke")
    x_train, y_train, x_val, y_val = spec.load_data(seed=42)

    assert spec.task == "language_modeling"
    assert x_train.shape == (512, 256)
    assert y_train.shape == (512, 256)
    assert x_val.shape == (128, 256)
    assert y_val.shape == (128, 256)


def test_lm_cache_resolves_repo_shared_cache(tmp_path: Path, monkeypatch):
    _write_lm_cache(tmp_path, "tinystories_lm")
    monkeypatch.delenv("TOPOGRAPH_LM_CACHE_DIR", raising=False)
    monkeypatch.setattr(
        "topograph.benchmarks.lm.DEFAULT_SHARED_CACHE_DIR",
        tmp_path / "missing-default",
    )
    monkeypatch.setattr(
        "topograph.benchmarks.lm.DEFAULT_REPO_SHARED_CACHE_DIR",
        tmp_path / "datasets",
    )
    monkeypatch.setattr(
        "topograph.benchmarks.lm.DEFAULT_LOCAL_CACHE_DIR",
        tmp_path / "missing-local",
    )

    spec = get_benchmark("tinystories_lm")
    x_train, y_train, x_val, y_val = spec.load_data(seed=42)

    assert x_train.shape == (600, 256)
    assert y_train.shape == (600, 256)
    assert x_val.shape == (200, 256)
    assert y_val.shape == (200, 256)


def test_lm_output_dim_expands_to_observed_token_range():
    spec = get_benchmark("tinystories_lm")
    x_train = np.array([[0, 1, 4095]], dtype=np.int32)
    y_train = np.array([[1, 2, 4096]], dtype=np.int64)

    assert spec.num_classes == 4096
    assert _resolve_model_output_dim(benchmark_spec=spec, X_train=x_train, y_train=y_train) == 4097


def test_evaluate_pool_uses_expanded_lm_output_dim():
    cfg = load_config(Path(__file__).resolve().parents[1] / "configs" / "tiny_smoke" / "config.yaml")
    cfg = cfg.model_validate(
        cfg.model_dump(mode="json")
        | {"benchmark_pool": {"benchmarks": ["tinystories_lm_smoke"], "sample_k": 1}}
    )
    rng = random.Random(42)
    state = GenerationState(
        generation=0,
        population=_create_seed_population(cfg, InnovationCounter(), rng),
    )

    out = evaluate_pool(
        state,
        cfg,
        [get_benchmark("tinystories_lm_smoke")],
        cache=WeightCache(),
        data_cache=BenchmarkDataCache(),
        evaluation_memo=EvaluationMemo(),
    )

    assert out.benchmark_results[0]["status"] == "ok"
    assert out.benchmark_results[0]["metric_value"] is not None


def test_language_modeling_metrics_use_perplexity():
    probs = np.array(
        [
            [[0.7, 0.2, 0.1], [0.1, 0.8, 0.1]],
            [[0.2, 0.3, 0.5], [0.6, 0.2, 0.2]],
        ],
        dtype=np.float32,
    )
    targets = np.array([[0, 1], [2, 0]], dtype=np.int64)

    metric_name, metric_direction, metric_value, quality = _compute_metric(
        "language_modeling", targets, probs,
    )

    assert metric_name == "perplexity"
    assert metric_direction == "min"
    assert metric_value >= 1.0
    assert quality <= 0.0
    assert _benchmark_metric_name("language_modeling") == "perplexity"
    assert _benchmark_metric_direction("language_modeling") == "min"


def test_compile_genome_language_modeling_returns_sequence_vocab_probs():
    rng = random.Random(42)
    genome = Genome.create_seed(InnovationCounter(), rng, num_layers=3)
    model = compile_genome(
        genome,
        input_dim=8,
        num_classes=32,
        task="language_modeling",
    )

    x = mx.array(np.random.randint(0, 32, size=(2, 8), dtype=np.int32))
    probs = model(x)
    probs_np = np.array(probs)

    assert probs_np.shape == (2, 8, 32)
    np.testing.assert_allclose(probs_np.sum(axis=-1), 1.0, atol=1e-4)


def test_tiny_lm_smoke_preset_and_script(monkeypatch):
    root = Path(__file__).resolve().parents[1]
    config_path = root / "configs" / "tiny_lm_synthetic_smoke.yaml"
    cfg = load_config(config_path)

    assert cfg.benchmark == "tiny_lm_synthetic"
    assert cfg.training.epochs == 1
    assert cfg.evolution.population_size == 3

    script_path = root / "scripts" / "smoke_tiny_lm.py"
    spec = importlib.util.spec_from_file_location("topograph_smoke_tiny_lm", script_path)
    module = importlib.util.module_from_spec(spec)
    assert spec.loader is not None
    spec.loader.exec_module(module)

    calls: dict[str, object] = {}

    def fake_run(cmd, cwd):
        calls["cmd"] = cmd
        calls["cwd"] = cwd
        return subprocess.CompletedProcess(cmd, 0)

    monkeypatch.setattr(subprocess, "run", fake_run)
    monkeypatch.setattr(
        module.sys,
        "argv",
        ["smoke_tiny_lm.py", "--config", str(config_path), "--run-dir", str(root / "runs" / "lm-test")],
    )

    exit_code = module.main()
    assert exit_code == 0
    assert calls["cmd"][:4] == ["uv", "run", "topograph", "evolve"]


def test_tinystories_smoke_preset_and_script(monkeypatch):
    root = Path(__file__).resolve().parents[1]
    config_path = root / "configs" / "tinystories_lm_smoke.yaml"
    cfg = load_config(config_path)

    assert cfg.benchmark == "tinystories_lm_smoke"
    assert cfg.training.batch_size == 8
    assert cfg.evolution.population_size == 2

    script_path = root / "scripts" / "smoke_tinystories_lm.py"
    spec = importlib.util.spec_from_file_location("topograph_smoke_tinystories_lm", script_path)
    module = importlib.util.module_from_spec(spec)
    assert spec.loader is not None
    spec.loader.exec_module(module)

    calls: dict[str, object] = {}

    def fake_run(cmd, cwd):
        calls["cmd"] = cmd
        calls["cwd"] = cwd
        return subprocess.CompletedProcess(cmd, 0)

    monkeypatch.setattr(subprocess, "run", fake_run)
    monkeypatch.setattr(
        module.sys,
        "argv",
        ["smoke_tinystories_lm.py", "--config", str(config_path), "--run-dir", str(root / "runs" / "ts-test")],
    )

    exit_code = module.main()
    assert exit_code == 0
    assert calls["cmd"][:4] == ["uv", "run", "topograph", "evolve"]


def test_wikitext2_smoke_preset_and_script(monkeypatch):
    root = Path(__file__).resolve().parents[1]
    config_path = root / "configs" / "wikitext2_lm_smoke.yaml"
    cfg = load_config(config_path)

    assert cfg.benchmark == "wikitext2_lm_smoke"
    assert cfg.training.batch_size == 8
    assert cfg.evolution.population_size == 2

    script_path = root / "scripts" / "smoke_wikitext2_lm.py"
    spec = importlib.util.spec_from_file_location("topograph_smoke_wikitext2_lm", script_path)
    module = importlib.util.module_from_spec(spec)
    assert spec.loader is not None
    spec.loader.exec_module(module)

    calls: dict[str, object] = {}

    def fake_run(cmd, cwd):
        calls["cmd"] = cmd
        calls["cwd"] = cwd
        return subprocess.CompletedProcess(cmd, 0)

    monkeypatch.setattr(subprocess, "run", fake_run)
    monkeypatch.setattr(
        module.sys,
        "argv",
        ["smoke_wikitext2_lm.py", "--config", str(config_path), "--run-dir", str(root / "runs" / "wt2-test")],
    )

    exit_code = module.main()
    assert exit_code == 0
    assert calls["cmd"][:4] == ["uv", "run", "topograph", "evolve"]


def test_33plus5_smoke_script_has_exact_benchmark_set(monkeypatch):
    root = Path(__file__).resolve().parents[1]
    script_path = root / "scripts" / "smoke_33plus5_bench.py"
    spec = importlib.util.spec_from_file_location("topograph_smoke_33plus5", script_path)
    module = importlib.util.module_from_spec(spec)
    assert spec.loader is not None
    spec.loader.exec_module(module)

    assert len(module.SHARED_33_BENCHMARKS) == 33
    assert len(module.LM_5_BENCHMARKS) == 5
    assert len(module.BENCHMARKS) == 38

    monkeypatch.setattr(module.sys, "argv", ["smoke_33plus5_bench.py", "--dry-run", "--limit", "2"])
    exit_code = module.main()
    assert exit_code == 0
