from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
import json

import pytest

from evonn_primordia.config import load_config
from evonn_primordia.export.symbiosis import export_symbiosis_contract
from evonn_primordia.pipeline import run_search


@dataclass(frozen=True)
class FakeGenome:
    family: str
    hidden_layers: list[int]
    learning_rate: float = 1e-3
    weight_decay: float = 0.0
    norm_type: str = "none"

    @property
    def genome_id(self) -> str:
        return f"{self.family}-{'x'.join(str(width) for width in self.hidden_layers)}"

    @property
    def parameter_estimate(self) -> int:
        return sum(self.hidden_layers)


@dataclass(frozen=True)
class FakeCompiledModel:
    model: object
    family: str
    parameter_count: int


@dataclass(frozen=True)
class FakeEvalResult:
    metric_name: str
    metric_value: float
    quality: float
    parameter_count: int
    train_seconds: float
    failure_reason: str | None = None


class FakeBenchmarkSpec:
    def __init__(self, name: str, task: str, metric_name: str, metric_direction: str, model_input_dim: int, model_output_dim: int) -> None:
        self.name = name
        self.task = task
        self.metric_name = metric_name
        self.metric_direction = metric_direction
        self.model_input_dim = model_input_dim
        self.model_output_dim = model_output_dim
        self.input_dim = model_input_dim
        self.num_classes = model_output_dim
        self.source = "fake"

    @property
    def resolved_image_shape(self) -> tuple[int, int, int]:
        return (8, 8, 1)

    def load_data(self, seed: int = 42):
        del seed
        if self.task == "language_modeling":
            return (
                [[1, 2, 3], [2, 3, 4]],
                [[2, 3, 4], [3, 4, 5]],
                [[1, 2, 3]],
                [[2, 3, 4]],
            )
        return (
            [[0.0] * self.model_input_dim, [1.0] * self.model_input_dim],
            [0, 1],
            [[0.5] * self.model_input_dim],
            [1],
        )


class FakeRuntimeBindings:
    def __init__(self) -> None:
        self.benchmarks = {
            "iris": FakeBenchmarkSpec("iris", "classification", "accuracy", "max", 4, 3),
            "tiny_lm_synthetic": FakeBenchmarkSpec("tiny_lm_synthetic", "language_modeling", "perplexity", "min", 3, 8),
            "circles": FakeBenchmarkSpec("circles", "classification", "accuracy", "max", 4, 2),
        }
        self.family_scores = {
            "mlp": 0.78,
            "sparse_mlp": 0.81,
            "moe_mlp": 0.79,
            "embedding": 4.6,
            "attention": 3.2,
            "sparse_attention": 3.7,
        }

    def get_benchmark(self, name: str) -> FakeBenchmarkSpec:
        return self.benchmarks[name]

    def benchmark_group(self, spec: FakeBenchmarkSpec) -> str:
        return "language_modeling" if spec.task == "language_modeling" else "tabular"

    def compatible_families(self, modality: str) -> list[str]:
        if modality == "text":
            return ["embedding", "attention", "sparse_attention"]
        return ["mlp", "sparse_mlp", "moe_mlp"]

    def create_seed_genome(self, family: str, width: int, depth: int) -> FakeGenome:
        return FakeGenome(family=family, hidden_layers=[width] * depth)

    def mutate_genome(self, genome: FakeGenome, slot_index: int, allowed_families: list[str], config) -> FakeGenome:
        del allowed_families, config
        width = genome.hidden_layers[0] + (slot_index * 8)
        return FakeGenome(family=genome.family, hidden_layers=[width] * len(genome.hidden_layers))

    def compile_genome(self, genome: FakeGenome, input_shape: list[int], output_dim: int, modality: str, task: str) -> FakeCompiledModel:
        del input_shape, output_dim, modality, task
        return FakeCompiledModel(model={"genome": genome.genome_id}, family=genome.family, parameter_count=genome.parameter_estimate)

    def train_and_evaluate(
        self,
        model: object,
        x_train,
        y_train,
        x_val,
        y_val,
        *,
        task: str,
        epochs: int,
        lr: float,
        batch_size: int,
        parameter_count: int,
    ) -> FakeEvalResult:
        del x_train, y_train, x_val, y_val, task, epochs, lr, batch_size
        family = model["genome"].split("-")[0]
        metric_value = self.family_scores[family]
        quality = metric_value if family in {"mlp", "sparse_mlp", "moe_mlp"} else -metric_value
        metric_name = "accuracy" if family in {"mlp", "sparse_mlp", "moe_mlp"} else "perplexity"
        return FakeEvalResult(
            metric_name=metric_name,
            metric_value=metric_value,
            quality=quality,
            parameter_count=parameter_count,
            train_seconds=0.01,
        )


@pytest.fixture
def fake_runtime(monkeypatch):
    runtime = FakeRuntimeBindings()
    monkeypatch.setattr("evonn_primordia.pipeline._load_runtime_bindings", lambda: runtime)
    return runtime


def test_smoke_run_and_export_uses_mlx_style_runtime(tmp_path: Path, fake_runtime) -> None:
    config_path = tmp_path / "config.yaml"
    config_path.write_text(
        """
seed: 42
run_name: smoke_primordia
benchmark_pool:
  name: smoke_pack
  benchmarks:
  - iris
  - tiny_lm_synthetic
search:
  mode: budget_matched
  target_evaluation_count: 4
training:
  epochs_per_candidate: 2
primitive_pool:
  tabular: [mlp, sparse_mlp]
  synthetic: [mlp]
  image: [mlp]
  language_modeling: [embedding, attention]
""".strip()
        + "\n",
        encoding="utf-8",
    )
    config = load_config(config_path)
    run_dir = tmp_path / "run"
    run_search(config, run_dir=run_dir, config_path=config_path)

    summary = json.loads((run_dir / "summary.json").read_text(encoding="utf-8"))
    assert summary["system"] == "primordia"
    assert summary["budget_policy_name"] == "prototype_equal_budget"
    assert summary["runtime"] == "mlx"
    assert {row["benchmark_name"] for row in summary["best_results"]} == {"iris", "tiny_lm_synthetic"}

    pack_path = tmp_path / "pack.yaml"
    pack_path.write_text(
        """
name: smoke_pack_eval4
benchmarks:
  - benchmark_id: iris_classification
    native_ids:
      primordia: iris
    task_kind: classification
    metric_name: accuracy
    metric_direction: max
  - benchmark_id: tiny_lm_synthetic
    native_ids:
      primordia: tiny_lm_synthetic
    task_kind: language_modeling
    metric_name: perplexity
    metric_direction: min
budget_policy:
  evaluation_count: 4
  epochs_per_candidate: 2
seed_policy:
  mode: campaign
  required: true
""".strip()
        + "\n",
        encoding="utf-8",
    )
    manifest_path, results_path = export_symbiosis_contract(run_dir, pack_path, run_dir)
    manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
    results = json.loads(results_path.read_text(encoding="utf-8"))

    assert manifest["system"] == "primordia"
    assert manifest["device"]["framework"] == "mlx"
    assert manifest["fairness"]["benchmark_pack_id"] == manifest["pack_name"]
    assert {record["benchmark_id"] for record in results} == {"iris_classification", "tiny_lm_synthetic"}


def test_budget_matched_mode_hits_exact_target_evaluation_count(tmp_path: Path, fake_runtime) -> None:
    config_path = tmp_path / "config.yaml"
    config_path.write_text(
        """
seed: 42
run_name: budgeted_primordia
benchmark_pool:
  name: smoke_pack
  benchmarks:
  - iris
  - circles
search:
  mode: budget_matched
  target_evaluation_count: 6
training:
  epochs_per_candidate: 1
primitive_pool:
  tabular: [mlp, sparse_mlp]
  synthetic: [mlp]
  image: [mlp]
  language_modeling: [embedding]
""".strip()
        + "\n",
        encoding="utf-8",
    )
    config = load_config(config_path)
    run_dir = tmp_path / "run"
    run_search(config, run_dir=run_dir, config_path=config_path)

    summary = json.loads((run_dir / "summary.json").read_text(encoding="utf-8"))
    trials = json.loads((run_dir / "trial_records.json").read_text(encoding="utf-8"))
    assert summary["evaluation_count"] == 6
    assert len(trials) == 6
    assert {record["benchmark_name"] for record in trials} == {"iris", "circles"}
    assert any(record["primitive_name"].endswith("@r2") for record in trials)


def test_run_search_requires_runtime_bindings(monkeypatch, tmp_path: Path) -> None:
    monkeypatch.setattr("evonn_primordia.pipeline._load_runtime_bindings", lambda: (_ for _ in ()).throw(RuntimeError("mlx unavailable")))
    config_path = tmp_path / "config.yaml"
    config_path.write_text(
        """
seed: 42
run_name: broken_primordia
benchmark_pool:
  name: smoke_pack
  benchmarks: [iris]
primitive_pool:
  tabular: [mlp]
  synthetic: [mlp]
  image: [mlp]
  language_modeling: [embedding]
""".strip()
        + "\n",
        encoding="utf-8",
    )
    config = load_config(config_path)
    with pytest.raises(RuntimeError, match="mlx unavailable"):
        run_search(config, run_dir=tmp_path / "run", config_path=config_path)
