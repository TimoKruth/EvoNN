"""Configuration models for Prism NAS."""

from __future__ import annotations

from pathlib import Path

import yaml
from pydantic import BaseModel, Field


class TrainingConfig(BaseModel):
    """Training hyperparameters applied to each genome evaluation."""

    epochs: int = 20
    batch_size: int = 32
    learning_rate: float = 1e-3
    lr_schedule: str = "cosine"  # cosine | constant
    grad_clip_norm: float = 1.0
    early_stopping_patience: int = 3
    weight_decay: float = 0.0
    validation_split: float = 0.2
    weight_inheritance: bool = True
    multi_fidelity: bool = True
    multi_fidelity_schedule: list[float] = Field(default_factory=lambda: [0.35, 0.65, 1.0])
    benchmark_epoch_min_scale: float = 0.75
    benchmark_epoch_max_scale: float = 1.25
    efficiency_epoch_min_scale: float = 0.85
    efficiency_epoch_max_scale: float = 1.15
    operator_adaptation_rate: float = 0.35


class EvolutionConfig(BaseModel):
    """Evolutionary search parameters."""

    population_size: int = 8
    offspring_per_generation: int = 8
    num_generations: int = 10
    elite_per_benchmark: int = 3
    tournament_size: int = 3
    crossover_rate: float = 0.25
    seed_hidden_width: int = 128
    seed_hidden_layers: int = 3
    max_hidden_width: int = 512
    max_hidden_layers: int = 8
    allowed_families: list[str] | None = None
    activation_choices: list[str] = Field(default_factory=lambda: ["relu", "gelu", "tanh"])
    dropout_choices: list[float] = Field(default_factory=lambda: [0.0, 0.1, 0.2, 0.3])
    undercovered_parent_bias: float = 0.55
    family_diversity_bias: float = 0.3
    family_stale_penalty: float = 0.15
    novelty_parent_bias: float = 0.1
    family_offspring_floor: int = 1
    benchmark_specialist_offspring: int = 2
    family_prior_bias: float = 0.2
    undercovered_focus_top_k: int = 3
    efficiency_bias_start: float = 0.15
    efficiency_bias_end: float = 0.40
    efficiency_warmup_generations: int = 2
    time_penalty_weight: float = 0.6
    param_penalty_weight: float = 0.4


class BenchmarkPoolConfig(BaseModel):
    """Which benchmarks to evolve against."""

    pack_name: str = "synthetic_core"
    benchmark_ids: list[str] | None = None


class RunConfig(BaseModel):
    """Top-level configuration for a single evolutionary run."""

    seed: int = 42
    benchmark_pack: BenchmarkPoolConfig = Field(default_factory=BenchmarkPoolConfig)
    training: TrainingConfig = Field(default_factory=TrainingConfig)
    evolution: EvolutionConfig = Field(default_factory=EvolutionConfig)
    prior_run_dirs: list[str] = Field(default_factory=list)


def load_config(path: str | Path) -> RunConfig:
    """Load a RunConfig from a YAML file."""
    with open(path, encoding="utf-8") as f:
        data = yaml.safe_load(f) or {}
    return RunConfig.model_validate(data)
