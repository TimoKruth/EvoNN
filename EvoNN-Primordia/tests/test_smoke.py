from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
import json

import pytest

from evonn_primordia.benchmarks import get_benchmark
from evonn_primordia.config import load_config
from evonn_primordia.export.report import write_report
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


class EdgeLMBenchmarkSpec(FakeBenchmarkSpec):
    def load_data(self, seed: int = 42):
        del seed
        return (
            [[6, 7, 7], [7, 7, 7]],
            [[7, 8, 8], [8, 8, 8]],
            [[7, 7, 7]],
            [[8, 8, 8]],
        )


class FakeRuntimeBindings:
    def __init__(self, *, runtime_backend: str = "mlx", runtime_version: str | None = "fake-mlx") -> None:
        self.runtime_backend = runtime_backend
        self.runtime_version = runtime_version
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
    assert summary["runtime"] == fake_runtime.runtime_backend
    assert summary["runtime_version"] == fake_runtime.runtime_version
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
    primitive_bank = json.loads((run_dir / "primitive_bank_summary.json").read_text(encoding="utf-8"))
    assert manifest["system"] == "primordia"
    assert manifest["device"]["framework"] == "mlx"
    assert manifest["device"]["framework_version"] == summary["runtime_version"]
    assert manifest["fairness"]["benchmark_pack_id"] == manifest["pack_name"]
    assert manifest["artifacts"]["primitive_bank_summary_json"] == "primitive_bank_summary.json"
    assert manifest["search_telemetry"]["primitive_usage"] == summary["primitive_usage"]
    assert manifest["search_telemetry"]["group_counts"] == summary["group_counts"]
    assert manifest["search_telemetry"]["failure_count"] == 0
    assert primitive_bank["system"] == "primordia"
    assert primitive_bank["run_id"] == summary["run_id"]
    assert primitive_bank["runtime"] == summary["runtime"]
    assert {entry["family"] for entry in primitive_bank["primitive_families"]} == {"attention", "embedding", "mlp", "sparse_mlp"}
    sparse = next(entry for entry in primitive_bank["primitive_families"] if entry["family"] == "sparse_mlp")
    assert sparse["evaluation_count"] == 1
    assert sparse["benchmark_wins"] == 1
    assert sparse["benchmarks_won"] == ["iris"]
    assert sparse["best_metric_name"] == "accuracy"
    assert sparse["best_metric_value"] == 0.81
    assert {record["benchmark_id"] for record in results} == {"iris_classification", "tiny_lm_synthetic"}



def test_export_uses_recorded_runtime_metadata(tmp_path: Path, fake_runtime) -> None:
    config_path = tmp_path / "config.yaml"
    config_path.write_text(
        """
seed: 42
run_name: runtime_metadata
benchmark_pool:
  name: smoke_pack
  benchmarks: [iris]
search:
  mode: budget_matched
  target_evaluation_count: 1
training:
  epochs_per_candidate: 1
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
    run_dir = tmp_path / "runtime_run"
    run_search(config, run_dir=run_dir, config_path=config_path)

    summary_path = run_dir / "summary.json"
    summary = json.loads(summary_path.read_text(encoding="utf-8"))
    summary["runtime"] = "numpy-fallback"
    summary["runtime_version"] = "fallback-1.2.3"
    summary_path.write_text(json.dumps(summary, indent=2), encoding="utf-8")

    pack_path = tmp_path / "pack.yaml"
    pack_path.write_text(
        """
name: runtime_pack
benchmarks:
  - benchmark_id: iris_classification
    native_ids:
      primordia: iris
    task_kind: classification
    metric_name: accuracy
    metric_direction: max
budget_policy:
  evaluation_count: 1
  epochs_per_candidate: 1
seed_policy:
  mode: campaign
  required: true
""".strip()
        + "\n",
        encoding="utf-8",
    )

    manifest_path, _results_path = export_symbiosis_contract(run_dir, pack_path, run_dir)
    manifest = json.loads(manifest_path.read_text(encoding="utf-8"))

    assert manifest["device"]["framework"] == "numpy-fallback"
    assert manifest["device"]["framework_version"] == "fallback-1.2.3"


def test_primitive_bank_summary_ignores_failed_records_for_wins_and_representatives(tmp_path: Path) -> None:
    run_dir = tmp_path / "failed_bank"
    run_dir.mkdir()
    (run_dir / "config.yaml").write_text(
        """
seed: 42
run_name: failed_bank
benchmark_pool:
  name: smoke_pack
  benchmarks: [tiny_lm_synthetic]
training:
  epochs_per_candidate: 1
primitive_pool:
  tabular: [mlp]
  synthetic: [mlp]
  image: [mlp]
  language_modeling: [embedding, attention]
""".strip()
        + "\n",
        encoding="utf-8",
    )
    (run_dir / "summary.json").write_text(
        json.dumps(
            {
                "system": "primordia",
                "runtime": "mlx",
                "runtime_version": "fake-mlx",
                "run_id": "failed_bank",
                "run_name": "failed_bank",
                "evaluation_count": 2,
                "benchmark_count": 1,
                "budget_policy_name": "prototype_equal_budget",
                "primitive_usage": {"embedding": 1, "attention": 1},
                "group_counts": {"tabular": 0, "synthetic": 0, "image": 0, "language_modeling": 1},
                "failure_count": 1,
                "wall_clock_seconds": 0.1,
                "best_results": [
                    {
                        "benchmark_name": "tiny_lm_synthetic",
                        "primitive_name": "embedding",
                        "primitive_family": "embedding",
                        "metric_name": "perplexity",
                        "metric_direction": "min",
                        "metric_value": None,
                        "quality": float("inf"),
                        "parameter_count": 16,
                        "train_seconds": 0.0,
                        "architecture_summary": "embedding[16]",
                        "genome_id": "embedding-16",
                        "status": "failed",
                        "failure_reason": "boom",
                    }
                ],
            },
            indent=2,
        ),
        encoding="utf-8",
    )
    (run_dir / "best_results.json").write_text(
        json.dumps(
            [
                {
                    "benchmark_name": "tiny_lm_synthetic",
                    "primitive_name": "embedding",
                    "primitive_family": "embedding",
                    "metric_name": "perplexity",
                    "metric_direction": "min",
                    "metric_value": None,
                    "quality": float("inf"),
                    "parameter_count": 16,
                    "train_seconds": 0.0,
                    "architecture_summary": "embedding[16]",
                    "genome_id": "embedding-16",
                    "status": "failed",
                    "failure_reason": "boom",
                }
            ],
            indent=2,
        ),
        encoding="utf-8",
    )
    (run_dir / "trial_records.json").write_text(
        json.dumps(
            [
                {
                    "benchmark_name": "tiny_lm_synthetic",
                    "primitive_name": "embedding",
                    "primitive_family": "embedding",
                    "metric_name": "perplexity",
                    "metric_direction": "min",
                    "metric_value": None,
                    "quality": float("inf"),
                    "parameter_count": 16,
                    "train_seconds": 0.0,
                    "architecture_summary": "embedding[16]",
                    "genome_id": "embedding-16",
                    "status": "failed",
                    "failure_reason": "boom",
                },
                {
                    "benchmark_name": "tiny_lm_synthetic",
                    "primitive_name": "attention",
                    "primitive_family": "attention",
                    "metric_name": "perplexity",
                    "metric_direction": "min",
                    "metric_value": 3.2,
                    "quality": -3.2,
                    "parameter_count": 32,
                    "train_seconds": 0.01,
                    "architecture_summary": "attention[16]",
                    "genome_id": "attention-16",
                    "status": "ok",
                    "failure_reason": None,
                }
            ],
            indent=2,
        ),
        encoding="utf-8",
    )
    pack_path = tmp_path / "failed_pack.yaml"
    pack_path.write_text(
        """
name: failed_pack
benchmarks:
  - benchmark_id: tiny_lm_synthetic
    native_ids:
      primordia: tiny_lm_synthetic
    task_kind: language_modeling
    metric_name: perplexity
    metric_direction: min
budget_policy:
  evaluation_count: 2
  epochs_per_candidate: 1
seed_policy:
  mode: campaign
  required: true
""".strip()
        + "\n",
        encoding="utf-8",
    )

    export_symbiosis_contract(run_dir, pack_path, run_dir)
    primitive_bank = json.loads((run_dir / "primitive_bank_summary.json").read_text(encoding="utf-8"))

    embedding = next(entry for entry in primitive_bank["primitive_families"] if entry["family"] == "embedding")
    attention = next(entry for entry in primitive_bank["primitive_families"] if entry["family"] == "attention")
    assert embedding["benchmark_wins"] == 0
    assert embedding["benchmarks_won"] == []
    assert embedding["best_metric_value"] is None
    assert attention["benchmark_wins"] == 0
    assert attention["best_metric_value"] == 3.2
    assert attention["representative_genome_id"] == "attention-16"


def test_runtime_metadata_propagates_to_trials_summary_and_report(tmp_path: Path, monkeypatch) -> None:
    runtime = FakeRuntimeBindings(runtime_backend="numpy-fallback", runtime_version="fallback-0.9")
    monkeypatch.setattr("evonn_primordia.pipeline._load_runtime_bindings", lambda: runtime)
    config_path = tmp_path / "config.yaml"
    config_path.write_text(
        """
seed: 42
run_name: fallback_runtime
benchmark_pool:
  name: smoke_pack
  benchmarks: [iris]
search:
  mode: budget_matched
  target_evaluation_count: 1
training:
  epochs_per_candidate: 1
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
    run_dir = tmp_path / "fallback_run"
    run_search(config, run_dir=run_dir, config_path=config_path)

    summary = json.loads((run_dir / "summary.json").read_text(encoding="utf-8"))
    trials = json.loads((run_dir / "trial_records.json").read_text(encoding="utf-8"))
    report = (run_dir / "report.md").read_text(encoding="utf-8")

    assert summary["runtime"] == "numpy-fallback"
    assert summary["runtime_version"] == "fallback-0.9"
    assert all(record["runtime"] == "numpy-fallback" for record in trials)
    assert all(record["runtime_version"] == "fallback-0.9" for record in trials)
    assert "- Runtime: `numpy-fallback`" in report
    assert "- Runtime Version: `fallback-0.9`" in report
    assert "- Wall Clock Seconds: `" in report
    assert "## Primitive Usage" in report
    assert "| mlp | 1 |" in report
    assert "## Benchmark Group Coverage" in report
    assert "| tabular | 1 |" in report
    assert "## Failure Summary" in report
    assert "- Failure Count: `0`" in report


def test_report_refresh_overwrites_existing_report_with_current_summary_data(tmp_path: Path) -> None:
    run_dir = tmp_path / "refresh_report"
    run_dir.mkdir()
    (run_dir / "summary.json").write_text(
        json.dumps(
            {
                "run_id": "refresh_report",
                "runtime": "numpy-fallback",
                "runtime_version": "fallback-1.0",
                "evaluation_count": 5,
                "target_evaluation_count": 6,
                "benchmark_count": 2,
                "budget_policy_name": "prototype_equal_budget",
                "failure_count": 0,
                "primitive_usage": {"mlp": 5},
                "group_counts": {"tabular": 2},
                "wall_clock_seconds": 7.25,
                "best_results": [
                    {
                        "benchmark_name": "iris",
                        "primitive_name": "mlp-deep",
                        "metric_name": "accuracy",
                        "metric_value": 0.97,
                        "status": "ok",
                    }
                ],
            },
            indent=2,
        ),
        encoding="utf-8",
    )
    stale_report = run_dir / "report.md"
    stale_report.write_text("# stale report\n", encoding="utf-8")

    refreshed_path = write_report(run_dir)
    refreshed = refreshed_path.read_text(encoding="utf-8")

    assert refreshed_path == stale_report
    assert refreshed != "# stale report\n"
    assert "- Runtime: `numpy-fallback`" in refreshed
    assert "| mlp | 5 |" in refreshed
    assert "| iris | mlp-deep | accuracy | 0.970000 | ok |" in refreshed


def test_report_regeneration_reuses_best_results_artifact_when_summary_omits_best_rows(tmp_path: Path) -> None:
    run_dir = tmp_path / "artifact_only_report"
    run_dir.mkdir()
    (run_dir / "summary.json").write_text(
        json.dumps(
            {
                "run_id": "artifact_only_report",
                "runtime": "numpy-fallback",
                "runtime_version": "fallback-0.9",
                "evaluation_count": 2,
                "target_evaluation_count": 2,
                "benchmark_count": 1,
                "budget_policy_name": "prototype_equal_budget",
                "failure_count": 0,
                "primitive_usage": {"mlp": 2},
                "group_counts": {"tabular": 1},
                "wall_clock_seconds": 3.5,
            },
            indent=2,
        ),
        encoding="utf-8",
    )
    (run_dir / "best_results.json").write_text(
        json.dumps(
            [
                {
                    "benchmark_name": "moons",
                    "primitive_name": "mlp-wide",
                    "primitive_family": "mlp",
                    "metric_name": "accuracy",
                    "metric_value": 0.94,
                    "status": "ok",
                }
            ],
            indent=2,
        ),
        encoding="utf-8",
    )

    regenerated_path = write_report(run_dir)
    regenerated = regenerated_path.read_text(encoding="utf-8")

    assert regenerated_path == run_dir / "report.md"
    assert "## Best Primitive Per Benchmark" in regenerated
    assert "| moons | mlp-wide | accuracy | 0.940000 | ok |" in regenerated


def test_report_regeneration_uses_summary_telemetry(tmp_path: Path, fake_runtime) -> None:
    config_path = tmp_path / "config.yaml"
    config_path.write_text(
        """
seed: 42
run_name: regenerated_report
benchmark_pool:
  name: smoke_pack
  benchmarks: [iris]
search:
  mode: budget_matched
  target_evaluation_count: 1
training:
  epochs_per_candidate: 1
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
    run_dir = tmp_path / "regenerated_run"
    run_search(config, run_dir=run_dir, config_path=config_path)

    primitive_bank_summary = json.loads((run_dir / "primitive_bank_summary.json").read_text(encoding="utf-8"))
    report_path = run_dir / "report.md"
    report_path.unlink()
    regenerated_path = write_report(run_dir)
    regenerated = regenerated_path.read_text(encoding="utf-8")

    assert primitive_bank_summary["runtime"] == "mlx"
    assert primitive_bank_summary["primitive_families"] == [
        {
            "family": "mlp",
            "evaluation_count": 1,
            "benchmark_wins": 1,
            "benchmarks_won": ["iris"],
            "best_metric_name": "accuracy",
            "best_metric_value": 0.78,
            "representative_genome_id": "mlp-64x64",
            "representative_architecture_summary": "mlp[64x64]",
        }
    ]
    assert regenerated_path == report_path
    assert "- Runtime: `mlx`" in regenerated
    assert "- Wall Clock Seconds: `" in regenerated
    assert "## Primitive Usage" in regenerated
    assert "| mlp | 1 |" in regenerated
    assert "## Primitive Bank Summary" in regenerated
    assert "| Family | Evaluations | Benchmark Wins | Won Benchmarks | Best Metric | Best Value | Representative Genome | Representative Architecture |" in regenerated
    assert "| mlp | 1 | 1 | iris | accuracy | 0.780000 | mlp-64x64 | mlp[64x64] |" in regenerated
    assert "## Benchmark Group Coverage" in regenerated
    assert "| tabular | 1 |" in regenerated
    assert "## Failure Summary" in regenerated
    assert "- Failure Count: `0`" in regenerated


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


def test_real_benchmark_specs_keep_runtime_compatibility_aliases() -> None:
    iris = get_benchmark("iris")
    digits = get_benchmark("digits")
    tiny_lm = get_benchmark("tiny_lm_synthetic")

    assert iris.model_input_dim == 4
    assert iris.model_output_dim == 3
    assert digits.model_input_dim == 64
    assert digits.resolved_image_shape == (8, 8, 1)
    assert tiny_lm.model_input_dim == 128
    assert tiny_lm.model_output_dim == 256


def test_language_modeling_output_dim_expands_to_fit_max_token_id(tmp_path: Path, monkeypatch) -> None:
    runtime = FakeRuntimeBindings()
    runtime.benchmarks["edge_lm_vocab"] = EdgeLMBenchmarkSpec(
        "edge_lm_vocab",
        "language_modeling",
        "perplexity",
        "min",
        3,
        8,
    )
    captured_output_dims: list[int] = []
    base_compile = runtime.compile_genome

    def compile_and_capture(genome, input_shape, output_dim, modality, task):
        captured_output_dims.append(output_dim)
        return base_compile(genome, input_shape, output_dim, modality, task)

    runtime.compile_genome = compile_and_capture
    monkeypatch.setattr("evonn_primordia.pipeline._load_runtime_bindings", lambda: runtime)

    config_path = tmp_path / "config.yaml"
    config_path.write_text(
        """
seed: 42
run_name: edge_lm_vocab
benchmark_pool:
  name: smoke_pack
  benchmarks: [edge_lm_vocab]
search:
  mode: budget_matched
  target_evaluation_count: 1
training:
  epochs_per_candidate: 1
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
    run_dir = tmp_path / "run"
    run_search(config, run_dir=run_dir, config_path=config_path)

    results = json.loads((run_dir / "trial_records.json").read_text(encoding="utf-8"))
    assert captured_output_dims == [9]
    assert results[0]["status"] == "ok"


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
