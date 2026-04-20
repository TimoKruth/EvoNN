"""Run configuration models and YAML loading for Primordia."""
from __future__ import annotations

from pathlib import Path
from typing import Literal

import yaml
from pydantic import BaseModel, ConfigDict, Field


class BenchmarkPoolConfig(BaseModel):
    """Benchmark selection for one Primordia run."""

    model_config = ConfigDict(frozen=True)

    name: str = "custom"
    benchmarks: list[str] = Field(default_factory=list)


class PrimitivePoolConfig(BaseModel):
    """Primitive candidate families grouped by benchmark style."""

    model_config = ConfigDict(frozen=True)

    tabular: list[str] = Field(default_factory=lambda: ["mlp", "sparse_mlp", "moe_mlp"])
    synthetic: list[str] = Field(default_factory=lambda: ["mlp", "sparse_mlp"])
    image: list[str] = Field(default_factory=lambda: ["mlp", "conv2d", "lite_conv2d"])
    language_modeling: list[str] = Field(
        default_factory=lambda: ["embedding", "attention", "sparse_attention"]
    )


class SearchConfig(BaseModel):
    """Budget and primitive-search policy."""

    model_config = ConfigDict(frozen=True)

    mode: Literal["budget_matched", "fixed_pool"] = "budget_matched"
    target_evaluation_count: int | None = None
    seed_hidden_width: int = 64
    seed_hidden_layers: int = 2
    max_hidden_width: int = 256
    max_hidden_layers: int = 6


class TrainingConfig(BaseModel):
    """MLX training parameters for each primitive evaluation."""

    model_config = ConfigDict(frozen=True)

    epochs_per_candidate: int = 2
    batch_size: int = 32
    learning_rate: float = 1e-3
    lr_schedule: str = "cosine"
    grad_clip_norm: float = 1.0
    early_stopping_patience: int = 2
    weight_decay: float = 0.0


class RunConfig(BaseModel):
    """Top-level Primordia run config."""

    model_config = ConfigDict(frozen=True)

    seed: int = 42
    run_name: str | None = None
    benchmark_pool: BenchmarkPoolConfig
    primitive_pool: PrimitivePoolConfig = Field(default_factory=PrimitivePoolConfig)
    search: SearchConfig = Field(default_factory=SearchConfig)
    training: TrainingConfig = Field(default_factory=TrainingConfig)


def load_config(path: str | Path) -> RunConfig:
    """Load config YAML and support benchmark_pack alias payloads."""

    config_path = Path(path)
    payload = yaml.safe_load(config_path.read_text(encoding="utf-8"))
    if "benchmark_pool" not in payload and "benchmark_pack" in payload:
        pack = payload["benchmark_pack"] or {}
        payload["benchmark_pool"] = {
            "name": pack.get("pack_name", "benchmark_pack"),
            "benchmarks": pack.get("benchmark_ids", []),
        }
    return RunConfig.model_validate(payload)
