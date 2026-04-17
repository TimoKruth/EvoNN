"""Run configuration models and YAML loading."""

from __future__ import annotations

from pathlib import Path

import yaml
from pydantic import BaseModel, ConfigDict, Field


class BenchmarkPoolConfig(BaseModel):
    """Benchmark selection for a run."""

    model_config = ConfigDict(frozen=True)

    name: str = "custom"
    benchmarks: list[str] = Field(default_factory=list)


class TrainingConfig(BaseModel):
    """Training budget knobs."""

    model_config = ConfigDict(frozen=True)

    epochs: int = 1
    batch_size: int = 32
    learning_rate: float = 0.001
    multi_fidelity: bool = False
    weight_inheritance: bool = True


class EvolutionConfig(BaseModel):
    """Search knobs."""

    model_config = ConfigDict(frozen=True)

    population_size: int = 4
    generations: int = 1
    elite_per_benchmark: int = 1
    architecture_mode: str = "two_level_shared"


class RunConfig(BaseModel):
    """Top-level run config."""

    model_config = ConfigDict(frozen=True)

    seed: int = 42
    run_name: str | None = None
    benchmark_pool: BenchmarkPoolConfig
    training: TrainingConfig = Field(default_factory=TrainingConfig)
    evolution: EvolutionConfig = Field(default_factory=EvolutionConfig)


def load_config(path: str | Path) -> RunConfig:
    """Load a run config from YAML."""
    config_path = Path(path)
    payload = yaml.safe_load(config_path.read_text(encoding="utf-8"))
    if "benchmark_pool" not in payload and "benchmark_pack" in payload:
        pack = payload["benchmark_pack"] or {}
        payload["benchmark_pool"] = {
            "name": pack.get("pack_name", "benchmark_pack"),
            "benchmarks": pack.get("benchmark_ids", []),
        }
    if "evolution" in payload and "num_generations" in payload["evolution"]:
        payload["evolution"]["generations"] = payload["evolution"].pop("num_generations")
    return RunConfig.model_validate(payload)
