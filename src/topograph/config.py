"""All configuration models in one file."""

from __future__ import annotations

from pathlib import Path
from typing import Literal

import yaml
from pydantic import BaseModel, field_validator


# --- Mutation rates ---


class MutationRates(BaseModel):
    width: float = 0.4
    activation: float = 0.2
    add_layer: float = 0.1
    remove_layer: float = 0.05
    add_connection: float = 0.15
    remove_connection: float = 0.1
    add_residual: float = 0.1
    weight_bits: float = 0.1
    activation_bits: float = 0.1
    sparsity: float = 0.1
    operator_type: float = 0.1


# --- Quantization ---


class QuantizationPhase(BaseModel):
    generations: tuple[int, int | None]
    allowed_weight_bits: list[int] = [2, 4, 8, 16]
    allowed_activation_bits: list[int] = [4, 8, 16]


# --- Evolution ---


class EvolutionConfig(BaseModel):
    population_size: int = 50
    num_generations: int = 100
    elite_count: int = 3
    crossover_ratio: float = 0.7
    mutation_rates: MutationRates = MutationRates()


# --- Training ---


class TrainingConfig(BaseModel):
    epochs: int = 30
    learning_rate: float = 0.001
    batch_size: int = 32
    validation_split: float = 0.2
    lr_schedule: Literal["cosine", "fixed"] = "cosine"
    weight_decay: float = 0.01
    grad_clip_norm: float | None = 1.0
    layer_norm: bool = True
    weight_inheritance: bool = True
    finetune_epoch_ratio: float = 0.3
    partial_epoch_ratio: float = 0.6
    multi_fidelity: bool = True
    multi_fidelity_schedule: list[float] | None = None
    parallel_workers: int = 0  # 0 = auto, 1 = sequential


# --- Early stopping ---


class EarlyStoppingConfig(BaseModel):
    window: int = 10
    threshold: float = 0.001
    enabled: bool = True


# --- Speciation ---


class SpeciationConfig(BaseModel):
    enabled: bool = False
    threshold: float = 3.0
    c1: float = 1.0
    c2: float = 1.0
    c3: float = 0.4
    stagnation_limit: int = 15


# --- Benchmark pool ---


class BenchmarkPoolConfig(BaseModel):
    benchmarks: list[str]
    sample_k: int = 3
    aggregation: Literal["percentile"] = "percentile"
    rotation_interval: int | None = None
    undercovered_benchmark_bias: float = 0.55

    @field_validator("rotation_interval")
    @classmethod
    def _validate_rotation_interval(cls, v: int | None) -> int | None:
        if v is not None and v < 1:
            raise ValueError("rotation_interval must be >= 1 or None")
        return v


# --- Device target ---


class DeviceTarget(BaseModel):
    max_model_bytes: int | None = None
    max_latency_ms: float | None = None
    mode: str = "objective"


# --- Experts ---


class ExpertsConfig(BaseModel):
    enabled: bool = False
    max_experts: int = 6
    mutation_rates_add_expert: float = 0.05
    mutation_rates_remove_expert: float = 0.03
    mutation_rates_expert_width: float = 0.15
    mutation_rates_expert_activation: float = 0.10
    mutation_rates_top_k: float = 0.05
    mutation_rates_gate_dim: float = 0.05


# --- Master run config ---


class RunConfig(BaseModel):
    seed: int = 42
    benchmark: str = "moons"
    objectives: list[str] | None = None

    evolution: EvolutionConfig = EvolutionConfig()
    training: TrainingConfig = TrainingConfig()
    early_stopping: EarlyStoppingConfig | None = None
    speciation: SpeciationConfig = SpeciationConfig()
    quantization_schedule: list[QuantizationPhase] | None = None
    benchmark_pool: BenchmarkPoolConfig | None = None
    target_device: DeviceTarget | None = None
    experts: ExpertsConfig = ExpertsConfig()

    complexity_penalty: float = 0.0
    novelty_weight: float = 0.0
    novelty_k: int = 15
    novelty_archive_size: int = 5000
    map_elites: bool = False
    benchmark_elite_archive: bool = True
    regression_parent_bias: float = 0.3


# --- Loader ---


def load_config(path: Path | str) -> RunConfig:
    with open(path) as f:
        data = yaml.safe_load(f)
    return RunConfig.model_validate(data or {})
