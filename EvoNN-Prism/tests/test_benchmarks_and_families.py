from __future__ import annotations

import sys
from types import ModuleType, SimpleNamespace

import numpy as np
import pytest
from typer.testing import CliRunner

from prism.benchmarks.datasets import load_image, load_openml, load_sklearn
from prism.benchmarks.spec import BenchmarkSpec
from prism.cli import app
from prism.families.compiler import compile_genome
from prism.genome import ModelGenome, _sanitize_for_family


runner = CliRunner()


def _genome(family: str, widths: list[int], **updates) -> ModelGenome:
    base = {
        "family": family,
        "hidden_layers": widths,
        "activation": "relu",
        "dropout": 0.0,
    }
    base.update(updates)
    return ModelGenome(**base)


def test_load_sklearn_covers_builtin_generated_and_unknown():
    builtin = BenchmarkSpec(id="iris", task="classification", source="sklearn", dataset="load_iris")
    X_builtin, y_builtin = load_sklearn(builtin)
    assert X_builtin.shape[1] == 4
    assert y_builtin.ndim == 1

    generated = BenchmarkSpec(
        id="circles",
        task="classification",
        source="sklearn",
        dataset="make_circles",
        input_dim=2,
        num_classes=2,
        n_samples=64,
        noise=0.1,
        factor=0.4,
    )
    X_generated, y_generated = load_sklearn(generated, seed=7)
    assert X_generated.shape == (64, 2)
    assert y_generated.shape == (64,)

    with pytest.raises(ValueError, match="Unknown sklearn dataset"):
        load_sklearn(BenchmarkSpec(id="bad", task="classification", source="sklearn", dataset="ghost"))


def test_load_openml_requires_source_id_and_encodes_categories(monkeypatch):
    with pytest.raises(ValueError, match="missing source_id"):
        load_openml(BenchmarkSpec(id="adult", task="classification", source="openml"))

    class _Series:
        def __init__(self, values, dtype):
            self._values = values
            self.dtype = dtype

        def to_numpy(self):
            return np.array(self._values, dtype=object)

    class _Column:
        def __init__(self, values, dtype):
            self.values = list(values)
            self.dtype = dtype

        def __iter__(self):
            return iter(self.values)

    class _Frame:
        def __init__(self):
            self.columns = ["cat", "num"]
            self._columns = {
                "cat": _Column(["red", "blue", "red"], np.dtype("O")),
                "num": _Column([1.0, 2.0, np.nan], np.dtype("float32")),
            }

        def __getitem__(self, key):
            return self._columns[key]

        def __setitem__(self, key, value):
            values = list(value) if hasattr(value, "__iter__") else [value]
            self._columns[key] = _Column(values, np.dtype("float32"))

        def to_numpy(self, dtype=None, na_value=np.nan):
            del na_value
            arr = np.column_stack([self._columns["cat"].values, self._columns["num"].values])
            return arr.astype(dtype or np.float32)

    class _Dataset:
        default_target_attribute = "target"

        def get_data(self, target):
            assert target == "target"
            return _Frame(), _Series(["yes", "no", "yes"], object), [True, False], None

    openml_module = ModuleType("openml")
    openml_module.datasets = SimpleNamespace(get_dataset=lambda source_id, download_data=True: _Dataset())

    pandas_module = ModuleType("pandas")
    pandas_module.Categorical = lambda values: SimpleNamespace(codes=np.array([1, 0, 1], dtype=np.float32))

    monkeypatch.setitem(sys.modules, "openml", openml_module)
    monkeypatch.setitem(sys.modules, "pandas", pandas_module)

    X, y = load_openml(BenchmarkSpec(id="adult", task="classification", source="openml", source_id=123))

    assert X.shape == (3, 2)
    assert X.dtype == np.float32
    assert np.isfinite(X).all()
    assert y.tolist() == [1, 0, 1]


def test_load_image_digits_and_unknown_name():
    X, y = load_image(BenchmarkSpec(id="digits", task="classification", source="image", dataset="digits"))
    assert X.ndim == 2
    assert y.ndim == 1

    with pytest.raises(ValueError, match="Unknown image dataset"):
        load_image(BenchmarkSpec(id="ghost", task="classification", source="image", dataset="ghost"))


@pytest.mark.parametrize(
    ("genome", "input_shape", "output_dim", "modality", "task"),
    [
        (_genome("conv2d", [8, 8]), [8, 8, 1], 10, "image", "classification"),
        (_genome("lite_conv2d", [8, 8]), [8, 8, 1], 10, "image", "classification"),
        (_genome("conv1d", [8, 8]), [16, 4], 3, "sequence", "classification"),
        (_genome("lite_conv1d", [8, 8]), [16, 4], 3, "sequence", "classification"),
        (_genome("gru", [8, 8]), [16, 4], 3, "sequence", "classification"),
        (_genome("moe_mlp", [16, 16], num_experts=4, moe_top_k=2), [12], 3, "tabular", "classification"),
        (_genome("embedding", [16, 16], embedding_dim=16), [12], 32, "text", "language_modeling"),
        (_genome("sparse_attention", [16, 16], embedding_dim=16, num_heads=4), [12], 32, "text", "language_modeling"),
    ],
)
def test_compile_genome_covers_more_families(genome, input_shape, output_dim, modality, task):
    compiled = compile_genome(genome, input_shape, output_dim, modality, task=task)
    assert compiled.family == genome.family
    assert compiled.parameter_count > 0


def test_cli_suite_list_and_info(monkeypatch):
    monkeypatch.setattr("prism.cli.list_benchmarks", lambda: ["moons", "adult"])
    monkeypatch.setattr(
        "prism.cli.get_benchmark",
        lambda name: BenchmarkSpec(
            id=name,
            task="classification",
            source="sklearn" if name == "moons" else "openml",
            dataset="make_moons" if name == "moons" else None,
            input_dim=2 if name == "moons" else 14,
            num_classes=2,
        ),
    )
    monkeypatch.setattr("prism.cli.get_canonical_id", lambda name: f"canon::{name}")

    listed = runner.invoke(app, ["suite", "list", "--task", "classification", "--source", "sklearn"])
    info = runner.invoke(app, ["suite", "info", "moons"])

    assert listed.exit_code == 0
    assert "moons" in listed.stdout
    assert "adult" not in listed.stdout
    assert info.exit_code == 0
    assert "canon::moons" in info.stdout


def test_cli_suite_info_missing(monkeypatch):
    monkeypatch.setattr("prism.cli.get_benchmark", lambda name: (_ for _ in ()).throw(FileNotFoundError(name)))
    result = runner.invoke(app, ["suite", "info", "ghost"])
    assert result.exit_code == 1
    assert "not found" in result.stdout.lower()


def test_sanitize_for_family_resets_irrelevant_fields():
    payload = {
        "family": "mlp",
        "hidden_layers": [16, 8],
        "activation": "relu",
        "dropout": 0.0,
        "residual": False,
        "activation_sparsity": 0.5,
        "learning_rate": 1e-3,
        "kernel_size": 5,
        "embedding_dim": 256,
        "num_heads": 8,
        "norm_type": "none",
        "weight_decay": 0.0,
        "num_experts": 4,
        "moe_top_k": 2,
    }
    sanitized = _sanitize_for_family(payload)
    assert sanitized["embedding_dim"] == 64
    assert sanitized["num_heads"] == 4
    assert sanitized["kernel_size"] == 3
    assert sanitized["activation_sparsity"] == 0.0
    assert sanitized["num_experts"] == 0
