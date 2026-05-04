"""Runtime backend selection for Topograph."""

from __future__ import annotations

from dataclasses import dataclass
import importlib.metadata
from typing import Literal

try:  # pragma: no cover - depends on host runtime
    import mlx

    MLX_AVAILABLE = True
    MLX_VERSION = importlib.metadata.version("mlx")
except Exception:  # pragma: no cover - covered on non-MLX hosts
    mlx = None
    MLX_AVAILABLE = False
    MLX_VERSION = None


RuntimeBackendName = Literal["auto", "mlx", "numpy-fallback"]

FALLBACK_LIMITATIONS = (
    "numpy-fallback is a correctness-first Topograph path for CI, Linux, and "
    "contract validation; it is not expected to match MLX training quality."
)
RUNTIME_POLICY_NAME = "topology_runtime_policy_v1"


@dataclass(frozen=True)
class RuntimeSelection:
    requested_backend: RuntimeBackendName
    resolved_backend: Literal["mlx", "numpy-fallback"]
    runtime_version: str | None
    backend_limitations: str | None = None


@dataclass(frozen=True)
class RuntimeExecutionPolicy:
    name: str = RUNTIME_POLICY_NAME
    fallback_classifier: str = "standardized_nearest_centroid"
    fallback_regressor: str = "ridge_least_squares"
    fallback_language_model: str = "smoothed_unigram_perplexity"
    topology_selection_policy: str = "fitness_plus_topology_diversity_elites"
    mutation_pressure_policy: str = "scheduled_topology_mutation_when_rates_do_not_fire"

    def as_metadata(self, *, resolved_backend: str) -> dict[str, str]:
        return {
            "runtime_policy_name": self.name,
            "runtime_backend": resolved_backend,
            "fallback_classifier": self.fallback_classifier,
            "fallback_regressor": self.fallback_regressor,
            "fallback_language_model": self.fallback_language_model,
            "topology_selection_policy": self.topology_selection_policy,
            "mutation_pressure_policy": self.mutation_pressure_policy,
        }


def resolve_runtime_backend(requested_backend: RuntimeBackendName = "auto") -> RuntimeSelection:
    return _resolve_runtime_backend(requested_backend, allow_fallback=True)


def resolve_runtime_backend_with_policy(
    requested_backend: RuntimeBackendName = "auto",
    *,
    allow_fallback: bool = True,
) -> RuntimeSelection:
    return _resolve_runtime_backend(requested_backend, allow_fallback=allow_fallback)


def runtime_execution_policy() -> RuntimeExecutionPolicy:
    return RuntimeExecutionPolicy()


def _resolve_runtime_backend(
    requested_backend: RuntimeBackendName,
    *,
    allow_fallback: bool,
) -> RuntimeSelection:
    if requested_backend == "auto":
        if MLX_AVAILABLE:
            return RuntimeSelection("auto", "mlx", MLX_VERSION)
        if not allow_fallback:
            raise RuntimeError("Requested runtime backend 'auto' but MLX is unavailable and fallback is disabled.")
        return RuntimeSelection("auto", "numpy-fallback", None, FALLBACK_LIMITATIONS)
    if requested_backend == "mlx":
        if not MLX_AVAILABLE:
            raise RuntimeError("Requested runtime backend 'mlx' but MLX is unavailable.")
        return RuntimeSelection("mlx", "mlx", MLX_VERSION)
    if requested_backend == "numpy-fallback":
        return RuntimeSelection("numpy-fallback", "numpy-fallback", None, FALLBACK_LIMITATIONS)
    raise ValueError(f"Unsupported runtime backend: {requested_backend}")
