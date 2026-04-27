from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
import json

import pytest

from evonn_compare.contracts.validation import validate_contract
from evonn_primordia.benchmarks import get_benchmark
from evonn_primordia.benchmarks.parity import load_parity_pack
from evonn_primordia.config import load_config
from evonn_primordia.export.report import enrich_best_results, write_report
from evonn_primordia.export.seeding import write_seed_candidates
from evonn_primordia.export.symbiosis import export_symbiosis_contract
from evonn_primordia.pipeline import run_search
from evonn_shared.contracts import ResultRecord, RunManifest


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
    monkeypatch.setattr("evonn_primordia.pipeline._load_runtime_bindings", lambda _config: runtime)
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
    assert summary["precision_mode"] == "fp32"
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
    compare_summary = json.loads((run_dir / "compare_summary.json").read_text(encoding="utf-8"))
    export_summary = json.loads((run_dir / "summary.json").read_text(encoding="utf-8"))
    primitive_bank = json.loads((run_dir / "primitive_bank_summary.json").read_text(encoding="utf-8"))
    seed_candidates = json.loads((run_dir / "seed_candidates.json").read_text(encoding="utf-8"))
    assert manifest["system"] == "primordia"
    assert manifest["device"]["framework"] == "mlx"
    assert manifest["device"]["framework_version"] == summary["runtime_version"]
    assert manifest["device"]["precision_mode"] == summary["precision_mode"]
    assert manifest["fairness"]["benchmark_pack_id"] == manifest["pack_name"]
    assert manifest["budget"]["evaluation_count"] == summary["target_evaluation_count"]
    assert manifest["budget"]["actual_evaluations"] == summary["evaluation_count"]
    assert manifest["budget"]["failed_evaluations"] == summary["failed_evaluations"]
    assert manifest["budget"]["cached_evaluations"] == 0
    assert manifest["budget"]["invalid_evaluations"] == summary["skipped_evaluations"]
    assert manifest["budget"]["partial_run"] is False
    assert manifest["budget"]["evaluation_semantics"]
    assert manifest["budget"]["wall_clock_seconds"] == summary["wall_clock_seconds"]
    assert manifest["artifacts"]["model_summary_json"] == "compare_summary.json"
    assert manifest["artifacts"]["primitive_bank_summary_json"] == "primitive_bank_summary.json"
    assert manifest["artifacts"]["seed_candidates_json"] == "seed_candidates.json"
    assert manifest["search_telemetry"]["primitive_usage"] == summary["primitive_usage"]
    assert manifest["search_telemetry"]["group_counts"] == summary["group_counts"]
    assert manifest["search_telemetry"]["failure_count"] == 0
    assert compare_summary["system"] == "primordia"
    assert compare_summary["run_id"] == summary["run_id"]
    assert compare_summary["runtime_backend"] == summary["runtime"]
    assert compare_summary["precision_mode"] == summary["precision_mode"]
    assert export_summary == compare_summary
    assert compare_summary["benchmarks_evaluated"] == 2
    assert compare_summary["failure_patterns"] == {}
    assert compare_summary["wall_clock_seconds"] == summary["wall_clock_seconds"]
    assert compare_summary["primitive_usage"] == summary["primitive_usage"]
    assert primitive_bank["system"] == "primordia"
    assert primitive_bank["run_id"] == summary["run_id"]
    assert primitive_bank["runtime"] == summary["runtime"]
    assert primitive_bank["precision_mode"] == summary["precision_mode"]
    assert seed_candidates["system"] == "primordia"
    assert seed_candidates["run_id"] == summary["run_id"]
    assert {entry["family"] for entry in seed_candidates["seed_candidates"]} == {"attention", "sparse_mlp"}
    top_seed = seed_candidates["seed_candidates"][0]
    assert "supporting_benchmarks" in top_seed
    assert "repeat_support_count" in top_seed
    assert "median_quality_by_group" in top_seed
    assert seed_candidates["benchmark_seeds"][0]["benchmark_group"] == "tabular"
    assert {entry["family"] for entry in primitive_bank["primitive_families"]} == {"attention", "embedding", "mlp", "sparse_mlp"}
    sparse = next(entry for entry in primitive_bank["primitive_families"] if entry["family"] == "sparse_mlp")
    assert sparse["evaluation_count"] == 1
    assert sparse["benchmark_wins"] == 1
    assert sparse["benchmarks_won"] == ["iris"]
    assert sparse["best_metric_name"] == "accuracy"
    assert sparse["best_metric_value"] == 0.81
    assert {record["benchmark_id"] for record in results} == {"iris_classification", "tiny_lm_synthetic"}

    validation_report = validate_contract(
        RunManifest.model_validate(manifest),
        [ResultRecord.model_validate(row) for row in results],
        load_parity_pack(pack_path),
        run_dir,
    )
    validation_codes = {issue.code for issue in validation_report.issues}
    assert "budget_actual_evaluations_missing" not in validation_codes
    assert "budget_semantics_missing" not in validation_codes
    assert validation_report.ok


def test_named_lane_config_can_complete_regression_and_classification_surface(tmp_path: Path, monkeypatch) -> None:
    class Phase2Runtime(FakeRuntimeBindings):
        def __init__(self) -> None:
            super().__init__(runtime_backend="numpy-fallback", runtime_version="phase2-fake")
            self.benchmarks.update(
                {
                    "diabetes": FakeBenchmarkSpec("diabetes", "regression", "mse", "min", 10, 1),
                    "friedman1": FakeBenchmarkSpec("friedman1", "regression", "mse", "min", 10, 1),
                }
            )
            self.family_scores.update({"mlp": 0.91, "sparse_mlp": 0.89, "moe_mlp": 0.87})
            self.regression_scores = {"mlp": 12.0, "sparse_mlp": 10.5, "moe_mlp": 9.25}

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
            if task == "regression":
                del x_train, y_train, x_val, y_val, epochs, lr, batch_size
                family = model["genome"].split("-")[0]
                metric_value = self.regression_scores[family]
                return FakeEvalResult(
                    metric_name="mse",
                    metric_value=metric_value,
                    quality=-metric_value,
                    parameter_count=parameter_count,
                    train_seconds=0.01,
                )
            return super().train_and_evaluate(
                model,
                x_train,
                y_train,
                x_val,
                y_val,
                task=task,
                epochs=epochs,
                lr=lr,
                batch_size=batch_size,
                parameter_count=parameter_count,
            )

    runtime = Phase2Runtime()
    monkeypatch.setattr("evonn_primordia.pipeline._load_runtime_bindings", lambda _config: runtime)
    config_path = tmp_path / "phase2.yaml"
    config_path.write_text(
        """
seed: 42
run_name: phase2_lane
runtime:
  backend: numpy-fallback
benchmark_pool:
  name: phase2_lane
  benchmarks: [iris, diabetes, friedman1]
search:
  mode: budget_matched
  target_evaluation_count: 6
training:
  epochs_per_candidate: 1
primitive_pool:
  tabular: [mlp, sparse_mlp, moe_mlp]
  synthetic: [mlp]
  image: [mlp]
  language_modeling: [embedding]
""".strip()
        + "\n",
        encoding="utf-8",
    )
    config = load_config(config_path)
    run_dir = tmp_path / "phase2_run"
    run_search(config, run_dir=run_dir, config_path=config_path)

    summary = json.loads((run_dir / "summary.json").read_text(encoding="utf-8"))

    assert summary["failure_count"] == 0
    assert summary["completed_benchmarks"] == ["iris", "diabetes", "friedman1"]
    assert {row["benchmark_name"] for row in summary["best_results"]} == {"iris", "diabetes", "friedman1"}
    assert {row["metric_name"] for row in summary["best_results"]} == {"accuracy", "mse"}


def test_search_policy_is_surfaced_and_max_candidates_cap_is_enforced(tmp_path: Path, monkeypatch) -> None:
    runtime = FakeRuntimeBindings(runtime_backend="numpy-fallback", runtime_version="phase3-fake")
    monkeypatch.setattr("evonn_primordia.pipeline._load_runtime_bindings", lambda _config: runtime)
    config_path = tmp_path / "phase3.yaml"
    config_path.write_text(
        """
seed: 42
run_name: phase3_cap
runtime:
  backend: numpy-fallback
benchmark_pool:
  name: phase3_cap
  benchmarks: [iris]
search:
  mode: budget_matched
  target_evaluation_count: 5
  population_size: 4
  elite_fraction: 0.5
  mutation_rounds_per_parent: 2
  family_exploration_floor: 1
  max_candidates_per_benchmark: 2
training:
  epochs_per_candidate: 1
primitive_pool:
  tabular: [mlp, sparse_mlp, moe_mlp]
  synthetic: [mlp]
  image: [mlp]
  language_modeling: [embedding]
""".strip()
        + "\n",
        encoding="utf-8",
    )
    config = load_config(config_path)
    run_dir = tmp_path / "phase3_run"
    run_search(config, run_dir=run_dir, config_path=config_path)
    report = (run_dir / "report.md").read_text(encoding="utf-8")
    summary = json.loads((run_dir / "summary.json").read_text(encoding="utf-8"))

    assert summary["evaluation_count"] == 2
    assert summary["search_policy"]["max_candidates_per_benchmark"] == 2
    assert summary["search_policy"]["mutation_rounds_per_parent"] == 2
    assert summary["benchmark_slot_plan"][0]["raw_slots"] == 5
    assert summary["benchmark_slot_plan"][0]["effective_slots"] == 2
    assert "## Search Policy" in report
    assert "## Benchmark Slot Plan" in report


def test_search_leader_surfaces_capture_benchmark_and_family_bests(tmp_path: Path, fake_runtime) -> None:
    config_path = tmp_path / "leaders.yaml"
    config_path.write_text(
        """
seed: 42
run_name: phase3_leaders
benchmark_pool:
  name: phase3_leaders
  benchmarks: [iris, tiny_lm_synthetic]
search:
  mode: budget_matched
  target_evaluation_count: 4
training:
  epochs_per_candidate: 1
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
    run_dir = tmp_path / "leaders_run"
    run_search(config, run_dir=run_dir, config_path=config_path)

    summary = json.loads((run_dir / "summary.json").read_text(encoding="utf-8"))
    report = (run_dir / "report.md").read_text(encoding="utf-8")
    leaders = json.loads((run_dir / "search_leaders.json").read_text(encoding="utf-8"))
    primitive_bank = json.loads((run_dir / "primitive_bank_summary.json").read_text(encoding="utf-8"))

    assert {row["benchmark_name"] for row in summary["benchmark_leaders"]} == {"iris", "tiny_lm_synthetic"}
    assert any(row["leader_family"] == "sparse_mlp" and row["benchmark_name"] == "iris" for row in summary["benchmark_leaders"])
    assert any(row["family"] == "attention" and row["benchmark_wins"] == 1 for row in summary["family_leaders"])
    assert leaders == {
        "benchmark_leaders": summary["benchmark_leaders"],
        "family_leaders": summary["family_leaders"],
    }
    sparse_family = next(row for row in primitive_bank["primitive_families"] if row["family"] == "sparse_mlp")
    assert sparse_family["best_generation"] == 0
    assert sparse_family["best_search_score"] is not None
    assert sparse_family["supporting_benchmarks"] == ["iris"]
    assert "## Benchmark Leaders" in report
    assert "## Family Leaders" in report


def test_offspring_mutation_reuses_parent_genome_payload(tmp_path: Path, monkeypatch) -> None:
    class InheritanceRuntime(FakeRuntimeBindings):
        def mutate_genome(self, genome, slot_index: int, allowed_families: list[str], config):
            del slot_index, allowed_families, config
            width = genome.hidden_layers[0] + 1
            return FakeGenome(
                family=genome.family,
                hidden_layers=[width] * len(genome.hidden_layers),
                learning_rate=getattr(genome, "learning_rate", 1e-3),
                weight_decay=getattr(genome, "weight_decay", 0.0),
                norm_type=getattr(genome, "norm_type", "none"),
            ), "width+1"

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
            width = int(model["genome"].split("-")[1].split("x")[0])
            metric_value = width / 100.0
            return FakeEvalResult(
                metric_name="accuracy",
                metric_value=metric_value,
                quality=metric_value,
                parameter_count=parameter_count,
                train_seconds=0.01,
            )

    runtime = InheritanceRuntime(runtime_backend="numpy-fallback", runtime_version="phase3-inheritance")
    monkeypatch.setattr("evonn_primordia.pipeline._load_runtime_bindings", lambda _config: runtime)
    config_path = tmp_path / "inheritance.yaml"
    config_path.write_text(
        """
seed: 42
run_name: phase3_inheritance
runtime:
  backend: numpy-fallback
benchmark_pool:
  name: phase3_inheritance
  benchmarks: [iris]
search:
  mode: budget_matched
  target_evaluation_count: 3
  population_size: 1
  mutation_rounds_per_parent: 1
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
    run_dir = tmp_path / "inheritance_run"
    run_search(config, run_dir=run_dir, config_path=config_path)
    trials = json.loads((run_dir / "trial_records.json").read_text(encoding="utf-8"))

    widths = [row["genome_payload"]["hidden_layers"][0] for row in trials]
    generations = [row["generation"] for row in trials]

    assert generations == [0, 1, 2]
    assert widths == [64, 65, 66]
    assert trials[1]["parent_genome_id"] == trials[0]["genome_id"]
    assert trials[2]["parent_genome_id"] == trials[1]["genome_id"]
    assert trials[1]["mutation_operator"] == "width+1"
    assert trials[2]["mutation_operator"] == "width+1"


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
    summary["precision_mode"] = "bf16"
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
    assert manifest["device"]["precision_mode"] == "bf16"


def test_enrich_best_results_does_not_guess_ambiguous_benchmark_only_match() -> None:
    best_results = [
        {
            "benchmark_name": "iris",
            "metric_name": "accuracy",
            "metric_value": 0.9,
            "status": "ok",
        }
    ]
    trial_records = [
        {
            "benchmark_name": "iris",
            "benchmark_group": "tabular",
            "primitive_name": "mlp-a",
            "primitive_family": "mlp",
            "metric_name": "accuracy",
            "metric_value": 0.88,
            "quality": 0.88,
            "genome_id": "g-a",
            "architecture_summary": "mlp[32]",
            "status": "ok",
        },
        {
            "benchmark_name": "iris",
            "benchmark_group": "tabular",
            "primitive_name": "mlp-b",
            "primitive_family": "sparse_mlp",
            "metric_name": "accuracy",
            "metric_value": 0.91,
            "quality": 0.91,
            "genome_id": "g-b",
            "architecture_summary": "sparse_mlp[64]",
            "status": "ok",
        },
    ]

    enriched = enrich_best_results(best_results, trial_records)

    assert enriched[0]["benchmark_name"] == "iris"
    assert "primitive_family" not in enriched[0]
    assert "genome_id" not in enriched[0]
    assert "architecture_summary" not in enriched[0]


def test_symbiosis_export_with_external_output_dir_writes_self_contained_artifacts(tmp_path: Path) -> None:
    run_dir = tmp_path / "external_export_run"
    run_dir.mkdir()
    (run_dir / "config.yaml").write_text(
        """
seed: 42
run_name: external_export_run
benchmark_pool:
  name: smoke_pack
  benchmarks: [iris]
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
    (run_dir / "summary.json").write_text(
        json.dumps(
            {
                "system": "primordia",
                "runtime": "numpy-fallback",
                "runtime_version": "fallback-1.0",
                "run_id": "external_export_run",
                "run_name": "external_export_run",
                "evaluation_count": 1,
                "benchmark_count": 1,
                "budget_policy_name": "prototype_equal_budget",
                "primitive_usage": {"mlp": 1},
                "group_counts": {"tabular": 1},
                "failure_count": 0,
                "wall_clock_seconds": 0.1,
                "best_results": [
                    {
                        "benchmark_name": "iris",
                        "benchmark_group": "tabular",
                        "primitive_name": "mlp",
                        "primitive_family": "mlp",
                        "metric_name": "accuracy",
                        "metric_direction": "max",
                        "metric_value": 0.9,
                        "quality": 0.9,
                        "parameter_count": 10,
                        "train_seconds": 0.01,
                        "architecture_summary": "mlp[32]",
                        "genome_id": "mlp-32",
                        "status": "ok",
                        "failure_reason": None,
                    }
                ],
            },
            indent=2,
        ),
        encoding="utf-8",
    )
    (run_dir / "best_results.json").write_text(
        json.dumps(json.loads((run_dir / "summary.json").read_text(encoding="utf-8"))["best_results"], indent=2),
        encoding="utf-8",
    )
    (run_dir / "trial_records.json").write_text(
        json.dumps(
            [
                {
                    "benchmark_name": "iris",
                    "benchmark_group": "tabular",
                    "primitive_name": "mlp",
                    "primitive_family": "mlp",
                    "metric_name": "accuracy",
                    "metric_direction": "max",
                    "metric_value": 0.9,
                    "quality": 0.9,
                    "parameter_count": 10,
                    "train_seconds": 0.01,
                    "architecture_summary": "mlp[32]",
                    "genome_id": "mlp-32",
                    "status": "ok",
                    "failure_reason": None,
                }
            ],
            indent=2,
        ),
        encoding="utf-8",
    )
    pack_path = tmp_path / "external_pack.yaml"
    pack_path.write_text(
        """
name: external_pack
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
    output_dir = tmp_path / "exported"

    manifest_path, _results_path = export_symbiosis_contract(run_dir, pack_path, output_dir)
    manifest = json.loads(manifest_path.read_text(encoding="utf-8"))

    assert manifest["artifacts"]["config_snapshot"] == "config.yaml"
    assert manifest["artifacts"]["report_markdown"] == "report.md"
    assert manifest["artifacts"]["model_summary_json"] == "compare_summary.json"
    assert (output_dir / "config.yaml").exists()
    assert (output_dir / "report.md").exists()
    assert (output_dir / "compare_summary.json").exists()
    assert json.loads((output_dir / "summary.json").read_text(encoding="utf-8")) == json.loads(
        (output_dir / "compare_summary.json").read_text(encoding="utf-8")
    )
    assert (output_dir / "primitive_summary.json").exists()
    assert (output_dir / "primitive_trials.json").exists()


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
    compare_summary = json.loads((run_dir / "compare_summary.json").read_text(encoding="utf-8"))
    primitive_bank = json.loads((run_dir / "primitive_bank_summary.json").read_text(encoding="utf-8"))

    embedding = next(entry for entry in primitive_bank["primitive_families"] if entry["family"] == "embedding")
    attention = next(entry for entry in primitive_bank["primitive_families"] if entry["family"] == "attention")
    assert compare_summary["failure_count"] == 1
    assert compare_summary["failure_patterns"] == {"boom": 1}
    assert embedding["benchmark_wins"] == 0
    assert embedding["benchmarks_won"] == []
    assert embedding["best_metric_value"] is None
    assert attention["benchmark_wins"] == 0
    assert attention["best_metric_value"] == 3.2
    assert attention["representative_genome_id"] == "attention-16"


def test_runtime_metadata_propagates_to_trials_summary_and_report(tmp_path: Path, monkeypatch) -> None:
    runtime = FakeRuntimeBindings(runtime_backend="numpy-fallback", runtime_version="fallback-0.9")
    monkeypatch.setattr("evonn_primordia.pipeline._load_runtime_bindings", lambda _config: runtime)
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
    assert summary["precision_mode"] == "fp32"
    assert all(record["runtime"] == "numpy-fallback" for record in trials)
    assert all(record["runtime_version"] == "fallback-0.9" for record in trials)
    assert all(record["precision_mode"] == "fp32" for record in trials)
    assert "- Runtime: `numpy-fallback`" in report
    assert "- Runtime Version: `fallback-0.9`" in report
    assert "- Precision Mode: `fp32`" in report
    assert "- Wall Clock Seconds: `" in report
    assert "## Primitive Usage" in report
    assert "| mlp | 1 |" in report
    assert "## Benchmark Group Coverage" in report
    assert "| tabular | 1 |" in report
    assert "## Failure Summary" in report
    assert "- Failure Count: `0`" in report
    assert "## Failure Patterns" in report
    assert "| none | 0 |" in report


def test_search_loop_persists_lineage_and_scoring_fields(tmp_path: Path, monkeypatch) -> None:
    runtime = FakeRuntimeBindings(runtime_backend="numpy-fallback", runtime_version="fallback-1.0")
    monkeypatch.setattr("evonn_primordia.pipeline._load_runtime_bindings", lambda _config: runtime)
    config_path = tmp_path / "config.yaml"
    config_path.write_text(
        """
seed: 42
run_name: lineage_fields
benchmark_pool:
  name: lineage_fields
  benchmarks: [iris]
search:
  mode: budget_matched
  target_evaluation_count: 5
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
    run_dir = tmp_path / "lineage_run"
    run_search(config, run_dir=run_dir, config_path=config_path)

    trials = json.loads((run_dir / "trial_records.json").read_text(encoding="utf-8"))

    assert len(trials) == 5
    assert any(record["generation"] > 0 for record in trials)
    assert any(record["parent_genome_id"] for record in trials if record["generation"] > 0)
    assert all("search_score" in record for record in trials)
    assert all("novelty_score" in record for record in trials)
    assert all("complexity_penalty" in record for record in trials)


def test_report_includes_grouped_failure_patterns_with_status_fallback(tmp_path: Path) -> None:
    run_dir = tmp_path / "failure_pattern_report"
    run_dir.mkdir()
    (run_dir / "summary.json").write_text(
        json.dumps(
            {
                "run_id": "failure_pattern_report",
                "runtime": "numpy-fallback",
                "runtime_version": "fallback-1.0",
                "evaluation_count": 3,
                "target_evaluation_count": 3,
                "benchmark_count": 3,
                "budget_policy_name": "prototype_equal_budget",
                "failure_count": 2,
                "primitive_usage": {"mlp": 3},
                "group_counts": {"tabular": 2, "language_modeling": 1},
                "wall_clock_seconds": 4.5,
            },
            indent=2,
        ),
        encoding="utf-8",
    )
    (run_dir / "trial_records.json").write_text(
        json.dumps(
            [
                {
                    "benchmark_name": "iris",
                    "primitive_name": "mlp",
                    "metric_name": "accuracy",
                    "metric_value": 0.9,
                    "status": "ok",
                },
                {
                    "benchmark_name": "tiny_lm_synthetic",
                    "primitive_name": "embedding",
                    "status": "skipped",
                },
                {
                    "benchmark_name": "cifar10_mini",
                    "primitive_name": "conv",
                    "status": "failed",
                    "failure_reason": "oom",
                },
            ],
            indent=2,
        ),
        encoding="utf-8",
    )

    report = write_report(run_dir).read_text(encoding="utf-8")

    assert "## Failure Patterns" in report
    assert "| oom | 1 |" in report
    assert "| skipped | 1 |" in report



def test_report_escapes_failure_pattern_markdown_cells(tmp_path: Path) -> None:
    run_dir = tmp_path / "escaped_failure_pattern_report"
    run_dir.mkdir()
    (run_dir / "summary.json").write_text(
        json.dumps(
            {
                "run_id": "escaped_failure_pattern_report",
                "runtime": "numpy-fallback",
                "runtime_version": "fallback-1.0",
                "evaluation_count": 1,
                "target_evaluation_count": 1,
                "benchmark_count": 1,
                "budget_policy_name": "prototype_equal_budget",
                "failure_count": 1,
                "primitive_usage": {"mlp": 1},
                "group_counts": {"tabular": 1},
                "wall_clock_seconds": 1.0,
            },
            indent=2,
        ),
        encoding="utf-8",
    )
    (run_dir / "trial_records.json").write_text(
        json.dumps(
            [
                {
                    "benchmark_name": "iris",
                    "primitive_name": "mlp",
                    "status": "failed",
                    "failure_reason": "shape|mismatch\nretry",
                }
            ],
            indent=2,
        ),
        encoding="utf-8",
    )

    report = write_report(run_dir).read_text(encoding="utf-8")

    assert "| shape\\|mismatch<br>retry | 1 |" in report



def test_report_escapes_runtime_markdown_cells_across_tables(tmp_path: Path) -> None:
    run_dir = tmp_path / "escaped_runtime_cells_report"
    run_dir.mkdir()
    (run_dir / "summary.json").write_text(
        json.dumps(
            {
                "run_id": "escaped_runtime_cells_report",
                "runtime": "numpy-fallback",
                "runtime_version": "fallback-1.0",
                "evaluation_count": 1,
                "target_evaluation_count": 1,
                "benchmark_count": 1,
                "budget_policy_name": "prototype_equal_budget",
                "failure_count": 0,
                "primitive_usage": {"mlp|family\nv2": 1},
                "group_counts": {"tab|ular\nset": 1},
                "wall_clock_seconds": 1.0,
                "best_results": [
                    {
                        "benchmark_name": "iris|variant\nv2",
                        "primitive_name": "mlp|wide",
                        "metric_name": "acc\nrate",
                        "metric_value": 0.78,
                        "status": "ok|seed",
                    }
                ],
            },
            indent=2,
        ),
        encoding="utf-8",
    )
    (run_dir / "trial_records.json").write_text("[]", encoding="utf-8")
    (run_dir / "primitive_bank_summary.json").write_text(
        json.dumps(
            {
                "system": "primordia",
                "run_id": "escaped_runtime_cells_report",
                "run_name": "escaped_runtime_cells_report",
                "runtime": "numpy-fallback",
                "runtime_version": "fallback-1.0",
                "primitive_families": [
                    {
                        "family": "mlp|family\nv2",
                        "evaluation_count": 1,
                        "benchmark_wins": 1,
                        "benchmarks_won": ["iris|variant\nv2"],
                        "best_metric_name": "acc\nrate",
                        "best_metric_value": 0.78,
                        "representative_genome_id": "gen|1",
                        "representative_architecture_summary": "mlp|deep\n64x64",
                    }
                ],
            },
            indent=2,
        ),
        encoding="utf-8",
    )
    (run_dir / "seed_candidates.json").write_text(
        json.dumps(
            {
                "seed_candidates": [
                    {
                        "seed_rank": 1,
                        "family": "attention|seed",
                        "benchmark_groups": ["group|1", "group\n2"],
                        "benchmark_wins": 2,
                        "representative_genome_id": "seed|gen",
                        "representative_architecture_summary": "seed\narch|v1",
                    }
                ],
                "benchmark_seeds": [],
            },
            indent=2,
        ),
        encoding="utf-8",
    )

    report = write_report(run_dir).read_text(encoding="utf-8")

    assert "| mlp\\|family<br>v2 | 1 |" in report
    assert "| mlp\\|family<br>v2 | 1 | 1 | iris\\|variant<br>v2 | acc<br>rate | 0.780000 | gen\\|1 | mlp\\|deep<br>64x64 |" in report
    assert "| 1 | attention\\|seed | group\\|1, group<br>2 | 2 | seed\\|gen | seed<br>arch\\|v1 |" in report
    assert "| tab\\|ular<br>set | 1 |" in report
    assert "| iris\\|variant<br>v2 | mlp\\|wide | acc<br>rate | 0.780000 | ok\\|seed |" in report


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
            "supporting_benchmarks": ["iris"],
            "benchmark_groups": ["tabular"],
            "best_metric_name": "accuracy",
            "best_metric_value": 0.78,
            "best_search_score": primitive_bank_summary["primitive_families"][0]["best_search_score"],
            "best_generation": 0,
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
    assert "## Benchmark Leaders" in regenerated
    assert "## Family Leaders" in regenerated
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
    monkeypatch.setattr("evonn_primordia.pipeline._load_runtime_bindings", lambda _config: runtime)

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
    monkeypatch.setattr("evonn_primordia.pipeline._load_runtime_bindings", lambda _config: (_ for _ in ()).throw(RuntimeError("mlx unavailable")))
    config_path = tmp_path / "config.yaml"
    config_path.write_text(
        """
seed: 42
run_name: missing_runtime
benchmark_pool:
  name: smoke_pack
  benchmarks: [iris]
primitive_pool:
  tabular: [mlp]
training:
  epochs_per_candidate: 1
""".strip()
        + "\n",
        encoding="utf-8",
    )
    config = load_config(config_path)
    with pytest.raises(RuntimeError, match="mlx unavailable"):
        run_search(config, run_dir=tmp_path / "run", config_path=config_path)


def test_write_seed_candidates_and_report_include_transfer_section(tmp_path: Path) -> None:
    run_dir = tmp_path / "seed_artifact"
    run_dir.mkdir()
    (run_dir / "summary.json").write_text(
        json.dumps(
            {
                "system": "primordia",
                "runtime": "numpy-fallback",
                "runtime_version": "fallback-1.0",
                "run_id": "seed_artifact",
                "run_name": "seed_artifact",
                "evaluation_count": 2,
                "target_evaluation_count": 2,
                "benchmark_count": 2,
                "budget_policy_name": "prototype_equal_budget",
                "primitive_usage": {"attention": 1, "sparse_mlp": 1},
                "group_counts": {"tabular": 1, "synthetic": 0, "image": 0, "language_modeling": 1},
                "failure_count": 0,
                "wall_clock_seconds": 0.2,
                "best_results": [
                    {
                        "benchmark_name": "iris",
                        "benchmark_group": "tabular",
                        "primitive_name": "sparse_mlp",
                        "primitive_family": "sparse_mlp",
                        "metric_name": "accuracy",
                        "metric_value": 0.82,
                        "status": "ok",
                        "genome_id": "sparse-1",
                        "architecture_summary": "sparse_mlp[64x64]",
                        "runtime": "numpy-fallback",
                        "runtime_version": "fallback-1.0",
                    },
                    {
                        "benchmark_name": "tiny_lm_synthetic",
                        "benchmark_group": "language_modeling",
                        "primitive_name": "attention",
                        "primitive_family": "attention",
                        "metric_name": "perplexity",
                        "metric_value": 3.1,
                        "status": "ok",
                        "genome_id": "attention-1",
                        "architecture_summary": "attention[64x64]",
                        "runtime": "numpy-fallback",
                        "runtime_version": "fallback-1.0",
                    },
                ],
            },
            indent=2,
        ),
        encoding="utf-8",
    )
    (run_dir / "trial_records.json").write_text(
        json.dumps(
            [
                {
                    "benchmark_name": "iris",
                    "benchmark_group": "tabular",
                    "primitive_name": "sparse_mlp",
                    "primitive_family": "sparse_mlp",
                    "metric_name": "accuracy",
                    "metric_value": 0.82,
                    "quality": 0.82,
                    "genome_id": "sparse-1",
                    "architecture_summary": "sparse_mlp[64x64]",
                    "status": "ok",
                },
                {
                    "benchmark_name": "tiny_lm_synthetic",
                    "benchmark_group": "language_modeling",
                    "primitive_name": "attention",
                    "primitive_family": "attention",
                    "metric_name": "perplexity",
                    "metric_value": 3.1,
                    "quality": -3.1,
                    "genome_id": "attention-1",
                    "architecture_summary": "attention[64x64]",
                    "status": "ok",
                },
                {
                    "benchmark_name": "iris",
                    "benchmark_group": "tabular",
                    "primitive_name": "attention",
                    "primitive_family": "attention",
                    "metric_name": "accuracy",
                    "metric_value": 0.71,
                    "quality": 0.71,
                    "genome_id": "attention-tabular-1",
                    "architecture_summary": "attention[32x32]",
                    "status": "ok",
                },
            ],
            indent=2,
        ),
        encoding="utf-8",
    )
    (run_dir / "best_results.json").write_text(
        json.dumps(json.loads((run_dir / "summary.json").read_text(encoding="utf-8"))["best_results"], indent=2),
        encoding="utf-8",
    )
    (run_dir / "primitive_bank_summary.json").write_text(
        json.dumps(
            {
                "system": "primordia",
                "run_id": "seed_artifact",
                "run_name": "seed_artifact",
                "runtime": "numpy-fallback",
                "runtime_version": "fallback-1.0",
                "primitive_families": [
                    {
                        "family": "attention",
                        "evaluation_count": 1,
                        "benchmark_wins": 1,
                        "benchmarks_won": ["tiny_lm_synthetic"],
                        "best_metric_name": "perplexity",
                        "best_metric_value": 3.1,
                        "representative_genome_id": "attention-1",
                        "representative_architecture_summary": "attention[64x64]",
                    }
                ],
            },
            indent=2,
        ),
        encoding="utf-8",
    )

    seed_path = write_seed_candidates(run_dir)
    report_path = write_report(run_dir)

    seed_payload = json.loads(seed_path.read_text(encoding="utf-8"))
    report = report_path.read_text(encoding="utf-8")
    assert seed_payload["seed_candidates"][0]["family"] == "attention"
    assert seed_payload["seed_candidates"][0]["benchmark_groups"] == ["language_modeling"]
    assert "## Transfer Seed Candidates" in report
    assert "attention[64x64]" in report


def test_primordia_compiler_module_imports_without_touching_mlx_runtime():
    import importlib

    module = importlib.import_module("evonn_primordia.families.compiler")
    assert hasattr(module, "compile_genome")
