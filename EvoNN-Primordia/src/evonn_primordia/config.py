"""Run configuration models and YAML loading for Primordia."""
from __future__ import annotations

import yaml
from pathlib import Path
from typing import Literal

from pydantic import BaseModel, ConfigDict, Field


class BenchmarkPoolConfig(BaseModel):
    """Benchmark selection for one Primordia run."""

    model_config = ConfigDict(frozen=True)

    name: str = "custom"
    benchmarks: list[str] = Field(default_factory=list)


class PrimitivePoolConfig(BaseModel):
    """Primitive candidate families grouped by benchmark style."""

    model_config = ConfigDict(frozen=True)

    tabular: list[str] = Field(default_factory=lambda: ["logistic", "mlp_small", "mlp", "linear_svc"])
    synthetic: list[str] = Field(default_factory=lambda: ["linear_svc", "mlp", "hist_gb", "rbf_svc_small"])
    image: list[str] = Field(default_factory=lambda: ["logistic", "mlp_small", "mlp", "mlp_deep"])
    language_modeling: list[str] = Field(
        default_factory=lambda: ["unigram_lm", "unigram_lm_a01", "bigram_lm", "bigram_lm_a01"]
    )


class SearchConfig(BaseModel):
    """Budget and search policy for primitive evaluation."""

    model_config = ConfigDict(frozen=True)

    mode: Literal["budget_matched", "fixed_pool"] = "budget_matched"
    target_evaluation_count: int | None = None
    epochs_per_candidate: int = 1


class TorchConfig(BaseModel):
    """Torch fallback options delegated to contender runtimes when used."""

    model_config = ConfigDict(frozen=True)

    allow_optional_missing: bool = True
    device: str = "cpu"
    batch_size: int = 32
    learning_rate: float = 1e-3
    classifier_epochs: int = 2
    max_batches_per_epoch: int = 16
    lm_steps: int = 24
    max_train_samples: int = 512
    max_val_samples: int = 128
    context_length_override: int | None = 64


class RunConfig(BaseModel):
    """Top-level Primordia run config."""

    model_config = ConfigDict(frozen=True)

    seed: int = 42
    run_name: str | None = None
    benchmark_pool: BenchmarkPoolConfig
    primitive_pool: PrimitivePoolConfig = Field(default_factory=PrimitivePoolConfig)
    search: SearchConfig = Field(default_factory=SearchConfig)
    torch: TorchConfig = Field(default_factory=TorchConfig)


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
