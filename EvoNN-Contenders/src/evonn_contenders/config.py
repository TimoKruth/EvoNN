"""Run configuration models and YAML loading."""

from __future__ import annotations

from pathlib import Path

import yaml
from pydantic import BaseModel, ConfigDict, Field


class BenchmarkPoolConfig(BaseModel):
    """Benchmark selection for one run."""

    model_config = ConfigDict(frozen=True)

    name: str = "custom"
    benchmarks: list[str] = Field(default_factory=list)


class ContenderPoolConfig(BaseModel):
    """Named contender lists by benchmark style."""

    model_config = ConfigDict(frozen=True)

    tabular: list[str] = Field(
        default_factory=lambda: ["hist_gb", "extra_trees", "random_forest", "mlp", "logistic"]
    )
    synthetic: list[str] = Field(default_factory=lambda: ["hist_gb", "extra_trees", "mlp"])
    image: list[str] = Field(default_factory=lambda: ["mlp", "logistic", "random_forest"])
    language_modeling: list[str] = Field(default_factory=lambda: ["bigram_lm", "unigram_lm"])


class SelectionConfig(BaseModel):
    """Selection knobs for contender evaluation."""

    model_config = ConfigDict(frozen=True)

    max_contenders_per_benchmark: int | None = None


class RunConfig(BaseModel):
    """Top-level contender run config."""

    model_config = ConfigDict(frozen=True)

    seed: int = 42
    run_name: str | None = None
    benchmark_pool: BenchmarkPoolConfig
    contender_pool: ContenderPoolConfig = Field(default_factory=ContenderPoolConfig)
    selection: SelectionConfig = Field(default_factory=SelectionConfig)


def load_config(path: str | Path) -> RunConfig:
    """Load config YAML."""
    config_path = Path(path)
    payload = yaml.safe_load(config_path.read_text(encoding="utf-8"))
    if "benchmark_pool" not in payload and "benchmark_pack" in payload:
        pack = payload["benchmark_pack"] or {}
        payload["benchmark_pool"] = {
            "name": pack.get("pack_name", "benchmark_pack"),
            "benchmarks": pack.get("benchmark_ids", []),
        }
    return RunConfig.model_validate(payload)
