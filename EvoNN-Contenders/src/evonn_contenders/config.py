"""Run configuration models and YAML loading."""

from __future__ import annotations

import hashlib
import json
from pathlib import Path
from typing import Literal

import yaml
from pydantic import BaseModel, ConfigDict, Field

from evonn_contenders.benchmarks import get_benchmark
from evonn_contenders.benchmarks.parity import load_parity_pack, native_id_candidates


class BenchmarkPoolConfig(BaseModel):
    """Benchmark selection for one run."""

    model_config = ConfigDict(frozen=True)

    name: str = "custom"
    benchmarks: list[str] = Field(default_factory=list)


class ContenderPoolConfig(BaseModel):
    """Named contender lists by benchmark style."""

    model_config = ConfigDict(frozen=True)

    tabular: list[str] = Field(
        default_factory=lambda: [
            "hist_gb",
            "extra_trees",
            "mlp_wide",
            "logistic_c10",
            "xgb_small",
            "lgbm_small",
            "catboost_small",
            "linear_svc",
        ]
    )
    synthetic: list[str] = Field(
        default_factory=lambda: ["hist_gb", "extra_trees", "mlp_wide", "linear_svc", "xgb_small"]
    )
    image: list[str] = Field(default_factory=lambda: ["mlp_wide", "extra_trees", "mlp", "cnn_small"])
    language_modeling: list[str] = Field(
        default_factory=lambda: ["bigram_lm_a01", "transformer_lm_tiny", "bigram_lm_a20"]
    )


class SelectionConfig(BaseModel):
    """Selection knobs for contender evaluation."""

    model_config = ConfigDict(frozen=True)

    max_contenders_per_benchmark: int | None = None


class BaselineConfig(BaseModel):
    """Baseline cache controls for contender reuse."""

    model_config = ConfigDict(frozen=True)

    baseline_id: str | None = None
    mode: Literal["fixed_reference", "budget_matched"] = "fixed_reference"
    target_evaluation_count: int | None = None
    cache_dir: str = ".baseline-cache"


class SvmConfig(BaseModel):
    """Safety caps for expensive SVM contenders."""

    model_config = ConfigDict(frozen=True)

    kernel_svm_max_train_samples: int = 4000
    kernel_svm_max_input_dim: int = 256


class BoostedTreesConfig(BaseModel):
    """Optional boosted-library behavior."""

    model_config = ConfigDict(frozen=True)

    allow_optional_missing: bool = True


class TorchConfig(BaseModel):
    """Torch backend defaults for CNN and transformer contenders."""

    model_config = ConfigDict(frozen=True)

    allow_optional_missing: bool = True
    device: str = "cpu"
    batch_size: int = 64
    learning_rate: float = 1e-3
    classifier_epochs: int = 3
    max_batches_per_epoch: int = 32
    lm_steps: int = 40
    max_train_samples: int = 1024
    max_val_samples: int = 256
    context_length_override: int | None = 64


class RunConfig(BaseModel):
    """Top-level contender run config."""

    model_config = ConfigDict(frozen=True)

    seed: int = 42
    run_name: str | None = None
    benchmark_pool: BenchmarkPoolConfig
    contender_pool: ContenderPoolConfig = Field(default_factory=ContenderPoolConfig)
    selection: SelectionConfig = Field(default_factory=SelectionConfig)
    baseline: BaselineConfig = Field(default_factory=BaselineConfig)
    svm: SvmConfig = Field(default_factory=SvmConfig)
    boosted_trees: BoostedTreesConfig = Field(default_factory=BoostedTreesConfig)
    torch: TorchConfig = Field(default_factory=TorchConfig)


def baseline_signature(config: RunConfig) -> str:
    """Stable signature for contender policy, independent of current benchmark subset."""

    payload = {
        "seed": config.seed,
        "mode": config.baseline.mode,
        "target_evaluation_count": config.baseline.target_evaluation_count,
        "contender_pool": config.contender_pool.model_dump(mode="json"),
        "selection": config.selection.model_dump(mode="json"),
        "svm": config.svm.model_dump(mode="json"),
        "boosted_trees": config.boosted_trees.model_dump(mode="json"),
        "torch": config.torch.model_dump(mode="json"),
    }
    if config.baseline.mode == "budget_matched":
        payload["benchmark_pool"] = config.benchmark_pool.model_dump(mode="json")
    raw = json.dumps(payload, sort_keys=True, separators=(",", ":"))
    return hashlib.sha256(raw.encode("utf-8")).hexdigest()[:16]


def resolve_baseline_id(config: RunConfig) -> str:
    """Resolve baseline ID from explicit config or deterministic policy hash."""

    if config.baseline.baseline_id:
        return config.baseline.baseline_id
    return f"contenders-{config.baseline.mode}-{baseline_signature(config)}"


def load_config(path: str | Path) -> RunConfig:
    """Load config YAML."""
    config_path = Path(path)
    payload = yaml.safe_load(config_path.read_text(encoding="utf-8"))
    if "benchmark_pool" not in payload and "benchmark_pack" in payload:
        pack_ref = payload["benchmark_pack"] or {}
        benchmark_ids = list(pack_ref.get("benchmark_ids") or [])
        if not benchmark_ids and pack_ref.get("pack_name"):
            parity_pack = load_parity_pack(pack_ref["pack_name"])
            benchmark_ids = [_resolve_native_benchmark_id(entry) for entry in parity_pack.benchmarks]
        payload["benchmark_pool"] = {
            "name": pack_ref.get("pack_name", "benchmark_pack"),
            "benchmarks": benchmark_ids,
        }
    return RunConfig.model_validate(payload)


def _resolve_native_benchmark_id(entry: object) -> str:
    benchmark_id = str(getattr(entry, "benchmark_id"))
    for candidate in native_id_candidates(entry, system="contenders") or [benchmark_id]:
        try:
            get_benchmark(candidate)
            return candidate
        except Exception:
            continue
    return benchmark_id
