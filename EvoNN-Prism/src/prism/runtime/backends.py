"""Runtime backend selection and Prism runtime policy metadata."""

from __future__ import annotations

from dataclasses import dataclass
import importlib.metadata
import importlib.util
from typing import Literal

RuntimeBackendName = Literal["auto", "mlx", "numpy-fallback"]
ResolvedRuntimeBackend = Literal["mlx", "numpy-fallback"]

FALLBACK_LIMITATIONS = (
    "numpy-fallback is a correctness-first Prism path for CI, Linux, smoke, "
    "and contract validation; it is not expected to match MLX run quality."
)
RUNTIME_POLICY_NAME = "prism_runtime_policy_v1"


def _mlx_available() -> bool:
    return importlib.util.find_spec("mlx") is not None


@dataclass(frozen=True)
class RuntimeSelection:
    requested_backend: RuntimeBackendName
    resolved_backend: ResolvedRuntimeBackend
    runtime_version: str | None
    backend_limitations: str | None = None


@dataclass(frozen=True)
class RuntimeExecutionPolicy:
    name: str = RUNTIME_POLICY_NAME
    fallback_classifier: str = "standardized_nearest_centroid"
    fallback_regressor: str = "ridge_least_squares"
    fallback_language_model: str = "smoothed_unigram_perplexity"
    candidate_selection_policy: str = "archive_elite_family_balanced_efficiency_selection"
    operator_adaptation_policy: str = "family_and_operator_efficiency_feedback"

    def as_metadata(self, *, resolved_backend: str) -> dict[str, str]:
        return {
            "runtime_policy_name": self.name,
            "runtime_backend": resolved_backend,
            "fallback_classifier": self.fallback_classifier,
            "fallback_regressor": self.fallback_regressor,
            "fallback_language_model": self.fallback_language_model,
            "candidate_selection_policy": self.candidate_selection_policy,
            "operator_adaptation_policy": self.operator_adaptation_policy,
        }


def runtime_execution_policy() -> RuntimeExecutionPolicy:
    return RuntimeExecutionPolicy()


def resolve_runtime_backend(
    requested_backend: RuntimeBackendName = "auto",
) -> RuntimeSelection:
    return resolve_runtime_backend_with_policy(requested_backend, allow_fallback=True)


def resolve_runtime_backend_with_policy(
    requested_backend: RuntimeBackendName = "auto",
    *,
    allow_fallback: bool = True,
) -> RuntimeSelection:
    mlx_available = _mlx_available()
    if requested_backend in {"auto", "mlx"} and mlx_available:
        try:
            version = importlib.metadata.version("mlx")
        except importlib.metadata.PackageNotFoundError:
            version = None
        return RuntimeSelection(requested_backend, "mlx", version)

    if requested_backend == "mlx" and not allow_fallback:
        raise RuntimeError("Prism MLX backend was requested, but MLX is unavailable.")

    if requested_backend in {"auto", "numpy-fallback"} or allow_fallback:
        return RuntimeSelection(
            requested_backend,
            "numpy-fallback",
            "numpy-fallback-1",
            FALLBACK_LIMITATIONS,
        )

    raise RuntimeError(f"Unsupported Prism runtime backend: {requested_backend!r}")
