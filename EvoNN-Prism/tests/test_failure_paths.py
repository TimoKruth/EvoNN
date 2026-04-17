from __future__ import annotations

import builtins
import sys
from random import Random
from types import ModuleType, SimpleNamespace

import numpy as np
import pytest

from prism.benchmarks.datasets import get_benchmark, load_openml
from prism.benchmarks.lm import resolve_lm_cache_path
from prism.benchmarks.parity import resolve_pack_path
from prism.benchmarks.preprocess import Preprocessor
from prism.benchmarks.spec import BenchmarkSpec
from prism.config import RunConfig
from prism.genome import ModelGenome
from prism.pipeline import evaluate as evaluate_mod
from prism.pipeline import reproduce as reproduce_mod
from prism.pipeline.evaluate import GenerationState, _evaluate_single
from prism.runtime.training import EvaluationResult, train_and_evaluate


def _sample_genome(family: str = "mlp") -> ModelGenome:
    return ModelGenome(family=family, hidden_layers=[16, 8], activation="relu", dropout=0.1)


def _sample_spec(task: str = "classification"):
    return SimpleNamespace(
        id="moons",
        modality="tabular",
        task=task,
        input_shape=[2],
        num_classes=2,
        output_dim=2,
        load_data=lambda seed=42: (
            np.zeros((4, 2), dtype=np.float32),
            np.zeros(4, dtype=np.int64),
            np.zeros((2, 2), dtype=np.float32),
            np.zeros(2, dtype=np.int64),
        ),
    )


def test_evaluate_single_reports_compile_error(monkeypatch):
    monkeypatch.setattr(
        "prism.families.compiler.compile_genome",
        lambda *args, **kwargs: (_ for _ in ()).throw(ValueError("bad family")),
    )

    result = _evaluate_single(
        genome=_sample_genome(),
        spec=_sample_spec(),
        training=RunConfig().training,
        epoch_scale=1.0,
        cache=None,
    )

    assert result.failure_reason == "compile_error:ValueError"
    assert result.parameter_count == 0
    assert result.quality == float("-inf")


def test_evaluate_skips_existing_results(monkeypatch):
    genome = _sample_genome()
    existing = EvaluationResult(
        metric_name="accuracy",
        metric_value=0.9,
        quality=0.9,
        parameter_count=10,
        train_seconds=0.1,
    )
    state = GenerationState(
        generation=0,
        population=[genome],
        results={genome.genome_id: {"moons": existing}},
        total_evaluations=0,
        parent_ids={genome.genome_id: []},
    )

    def _unexpected(*args, **kwargs):
        raise AssertionError("evaluation should have been skipped")

    monkeypatch.setattr(evaluate_mod, "_evaluate_single", _unexpected)

    updated = evaluate_mod.evaluate(state, RunConfig(), [SimpleNamespace(id="moons")])

    assert updated.total_evaluations == 0
    assert updated.results[genome.genome_id]["moons"] is existing


def test_evaluate_single_uses_parent_inheritance_and_stores_success(monkeypatch):
    genome = _sample_genome()
    model = object()
    transfer_calls: list[str] = []
    stored_ids: list[str] = []

    class FakeCache:
        def transfer_weights(self, parent_id, child_model):
            assert child_model is model
            transfer_calls.append(parent_id)
            return parent_id == "parent-b"

        def store(self, genome_id, child_model):
            assert child_model is model
            stored_ids.append(genome_id)

    monkeypatch.setattr(
        "prism.families.compiler.compile_genome",
        lambda *args, **kwargs: SimpleNamespace(model=model, parameter_count=7),
    )
    monkeypatch.setattr(
        evaluate_mod,
        "train_and_evaluate",
        lambda *args, **kwargs: EvaluationResult(
            metric_name="accuracy",
            metric_value=0.95,
            quality=0.95,
            parameter_count=7,
            train_seconds=0.2,
        ),
    )

    result = _evaluate_single(
        genome=genome,
        spec=_sample_spec(),
        training=RunConfig().training,
        epoch_scale=1.0,
        cache=FakeCache(),
        parent_ids=["parent-a", "parent-b", "parent-c"],
    )

    assert transfer_calls == ["parent-a", "parent-b"]
    assert stored_ids == [genome.genome_id]
    assert result.failure_reason is None
    assert result.metric_value == 0.95


def test_reproduce_records_crossover_lineage(monkeypatch):
    parent_a = _sample_genome("mlp")
    parent_b = _sample_genome("sparse_mlp")
    child = _sample_genome("mlp").model_copy(update={"hidden_layers": [32, 16]})

    state = SimpleNamespace(
        population=[parent_a, parent_b],
        archives={},
        results={
            parent_a.genome_id: {
                "moons": EvaluationResult("accuracy", 0.8, 0.8, 10, 0.1),
            },
            parent_b.genome_id: {
                "moons": EvaluationResult("accuracy", 0.9, 0.9, 12, 0.1),
            },
        },
    )
    config = RunConfig.model_validate(
        {
            "evolution": {
                "offspring_per_generation": 1,
                "crossover_rate": 1.0,
                "tournament_size": 2,
                "allowed_families": ["mlp", "sparse_mlp"],
            }
        }
    )

    picks = iter([parent_a, parent_b])
    monkeypatch.setattr(reproduce_mod, "tournament_select", lambda *args, **kwargs: next(picks))
    monkeypatch.setattr(reproduce_mod, "crossover", lambda left, right, rng: child)

    offspring, lineage = reproduce_mod.reproduce(state, config, Random(0))

    assert offspring == [child]
    assert lineage == [
        {
            "genome_id": child.genome_id,
            "parent_ids": [parent_a.genome_id, parent_b.genome_id],
            "operator": "crossover",
        }
    ]


def test_reproduce_retries_until_child_is_novel(monkeypatch):
    parent = _sample_genome("mlp")
    novel_child = parent.model_copy(update={"hidden_layers": [32, 16]})

    state = SimpleNamespace(
        population=[parent],
        archives={},
        results={
            parent.genome_id: {
                "moons": EvaluationResult("accuracy", 0.8, 0.8, 10, 0.1),
            },
        },
    )
    config = RunConfig.model_validate(
        {
            "evolution": {
                "offspring_per_generation": 1,
                "crossover_rate": 0.0,
                "tournament_size": 1,
                "allowed_families": ["mlp"],
            }
        }
    )

    attempts = iter([
        (parent, "width"),
        (novel_child, "width"),
    ])

    monkeypatch.setattr(reproduce_mod, "apply_random_mutation", lambda *args, **kwargs: next(attempts))

    offspring, lineage = reproduce_mod.reproduce(state, config, Random(0))

    assert offspring == [novel_child]
    assert lineage[0]["genome_id"] == novel_child.genome_id


def test_undercovered_parent_bias_rewards_rare_successes():
    common = _sample_genome("mlp")
    rare = _sample_genome("sparse_mlp")
    third = _sample_genome("attention")
    state = SimpleNamespace(
        population=[common, rare, third],
        results={
            common.genome_id: {"shared": EvaluationResult("accuracy", 0.8, 0.8, 10, 0.1)},
            rare.genome_id: {"rare": EvaluationResult("accuracy", 0.7, 0.7, 10, 0.1)},
            third.genome_id: {"shared": EvaluationResult("accuracy", 0.6, 0.6, 10, 0.1)},
        },
    )

    base = {
        common.genome_id: 0.8,
        rare.genome_id: 0.7,
        third.genome_id: 0.6,
    }
    boosted = reproduce_mod._apply_undercovered_parent_bias(state, base, bias=1.0)

    assert boosted[rare.genome_id] > boosted[common.genome_id]


def test_get_benchmark_missing_catalog_gives_explicit_error(tmp_path):
    with pytest.raises(FileNotFoundError, match="PRISM_CATALOG_DIR") as exc:
        get_benchmark("ghost", catalog_dir=tmp_path / "missing")

    assert "Benchmark catalog not found" in str(exc.value)


def test_load_openml_requires_data_extra(monkeypatch):
    real_import = builtins.__import__

    def fake_import(name, *args, **kwargs):
        if name in {"openml", "pandas"}:
            raise ImportError("missing optional dependency")
        return real_import(name, *args, **kwargs)

    monkeypatch.setattr(builtins, "__import__", fake_import)

    spec = BenchmarkSpec(id="adult", task="classification", source="openml", source_id=1590)
    with pytest.raises(ImportError, match="uv sync --extra data"):
        load_openml(spec)


def test_resolve_lm_cache_path_uses_shared_root(tmp_path, monkeypatch):
    shared_root = tmp_path / "shared-benchmarks"
    cache_dir = shared_root / "lm_cache"
    cache_dir.mkdir(parents=True)
    fixture = cache_dir / "demo_lm.npz"
    fixture.write_bytes(b"fixture")

    monkeypatch.delenv("PRISM_LM_CACHE_DIR", raising=False)
    monkeypatch.setenv("EVONN_SHARED_BENCHMARKS_DIR", str(shared_root))

    assert resolve_lm_cache_path("demo_lm") == fixture


def test_resolve_pack_path_uses_shared_root(tmp_path, monkeypatch):
    shared_root = tmp_path / "shared-benchmarks"
    suites_dir = shared_root / "suites" / "parity"
    suites_dir.mkdir(parents=True)
    fixture = suites_dir / "demo_pack.yaml"
    fixture.write_text("name: demo\nbenchmarks: []\n", encoding="utf-8")

    monkeypatch.delenv("PRISM_PARITY_PACK_DIRS", raising=False)
    monkeypatch.setenv("EVONN_SHARED_BENCHMARKS_DIR", str(shared_root))

    assert resolve_pack_path("demo_pack") == fixture


def test_preprocessor_handles_nans_constant_columns_and_reset():
    X = np.array(
        [
            [1.0, np.nan, 5.0],
            [1.0, 3.0, 5.0],
            [1.0, 7.0, 5.0],
        ],
        dtype=np.float32,
    )
    prep = Preprocessor()

    transformed = prep.fit_transform(X)

    assert np.isfinite(transformed).all()
    assert np.allclose(transformed[:, 0], 0.0)
    assert np.allclose(transformed[:, 2], 0.0)

    restored = prep.inverse_transform(transformed)
    assert restored.shape == X.shape
    assert np.allclose(restored[:, 1], np.array([5.0, 3.0, 7.0], dtype=np.float32))

    prep.reset()
    assert prep.fitted is False
    with pytest.raises(RuntimeError, match="not fitted"):
        prep.transform(X)


class _FakeLoss:
    def __init__(self, value: float):
        self._value = value

    def item(self) -> float:
        return self._value


class _FakeOptimizer:
    def __init__(self, learning_rate: float, weight_decay: float):
        self.learning_rate = learning_rate
        self.weight_decay = weight_decay
        self.state = {}

    def update(self, model, grads) -> None:
        del model, grads


class _DummyModel:
    def __init__(self, val_preds):
        self._val_preds = np.asarray(val_preds)

    def train(self) -> None:
        return None

    def eval(self) -> None:
        return None

    def parameters(self):
        return {}

    def __call__(self, x):
        rows = len(x)
        return self._val_preds[:rows]


def _install_fake_mlx(monkeypatch, *, loss_value=None, loss_exc: Exception | None = None) -> None:
    mx_core = ModuleType("mlx.core")
    mx_core.array = lambda value: np.asarray(value)
    mx_core.eval = lambda *args: None

    nn_module = ModuleType("mlx.nn")

    def value_and_grad(model, fn):
        del model, fn

        def runner(active_model, x, y):
            del active_model, x, y
            if loss_exc is not None:
                raise loss_exc
            return _FakeLoss(loss_value), {}

        return runner

    nn_module.value_and_grad = value_and_grad
    nn_module.losses = SimpleNamespace()

    optim_module = ModuleType("mlx.optimizers")
    optim_module.AdamW = _FakeOptimizer

    mlx_pkg = ModuleType("mlx")
    mlx_pkg.core = mx_core
    mlx_pkg.nn = nn_module
    mlx_pkg.optimizers = optim_module

    monkeypatch.setitem(sys.modules, "mlx", mlx_pkg)
    monkeypatch.setitem(sys.modules, "mlx.core", mx_core)
    monkeypatch.setitem(sys.modules, "mlx.nn", nn_module)
    monkeypatch.setitem(sys.modules, "mlx.optimizers", optim_module)


def test_train_and_evaluate_reports_runtime_error(monkeypatch):
    _install_fake_mlx(monkeypatch, loss_exc=RuntimeError("boom"))

    result = train_and_evaluate(
        _DummyModel(np.zeros((2, 2), dtype=np.float32)),
        np.zeros((4, 2), dtype=np.float32),
        np.zeros(4, dtype=np.int64),
        np.zeros((2, 2), dtype=np.float32),
        np.zeros(2, dtype=np.int64),
        task="classification",
        epochs=1,
        lr=1e-3,
        batch_size=2,
        lr_schedule="constant",
        grad_clip_norm=0.0,
    )

    assert result.failure_reason == "runtime_error:RuntimeError:boom"
    assert np.isnan(result.metric_value)


def test_train_and_evaluate_returns_nan_loss_failure(monkeypatch):
    _install_fake_mlx(monkeypatch, loss_value=float("nan"))

    result = train_and_evaluate(
        _DummyModel(np.zeros((2, 2), dtype=np.float32)),
        np.zeros((4, 2), dtype=np.float32),
        np.zeros(4, dtype=np.int64),
        np.zeros((2, 2), dtype=np.float32),
        np.zeros(2, dtype=np.int64),
        task="classification",
        epochs=1,
        lr=1e-3,
        batch_size=2,
        lr_schedule="constant",
        grad_clip_norm=0.0,
    )

    assert result.failure_reason == "nan_loss"
    assert result.quality == float("-inf")
