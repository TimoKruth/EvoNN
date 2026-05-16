"""Runtime backend selection and Prism runtime policy metadata."""

from __future__ import annotations

from dataclasses import dataclass
import importlib.metadata
import importlib.util
from typing import Final, Literal

RuntimeBackendName = Literal["auto", "mlx", "numpy-fallback"]
ResolvedRuntimeBackend = Literal["mlx", "numpy-fallback"]

AUTO_BACKEND: Final = "auto"
MLX_BACKEND: Final = "mlx"
NUMPY_FALLBACK_BACKEND: Final = "numpy-fallback"
NUMPY_FALLBACK_VERSION: Final = "numpy-fallback-1"

FALLBACK_LIMITATIONS = (
    "numpy-fallback is a correctness-first Prism path for CI, Linux, smoke, "
    "and contract validation; it is not expected to match MLX run quality."
)
RUNTIME_POLICY_NAME = "prism_runtime_policy_v1"


def _mlx_available() -> bool:
    """Return whether the optional MLX runtime can be imported on this host."""
    return importlib.util.find_spec(MLX_BACKEND) is not None


def _mlx_version() -> str | None:
    """Return the installed MLX package version when package metadata exists."""
    try:
        return importlib.metadata.version(MLX_BACKEND)
    except importlib.metadata.PackageNotFoundError:
        return None


@dataclass(frozen=True)
class RuntimeSelection:
    """Concrete runtime backend choice plus metadata persisted in run artifacts."""

    requested_backend: RuntimeBackendName
    resolved_backend: ResolvedRuntimeBackend
    runtime_version: str | None
    backend_limitations: str | None = None


@dataclass(frozen=True)
class RuntimeExecutionPolicy:
    """Stable names for runtime policies that affect result interpretation."""

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
    """Return the active Prism runtime execution policy metadata."""
    return RuntimeExecutionPolicy()


def resolve_runtime_backend(
    requested_backend: RuntimeBackendName = "auto",
) -> RuntimeSelection:
    """Resolve a runtime backend using Prism's default fallback policy."""
    return resolve_runtime_backend_with_policy(requested_backend, allow_fallback=True)


def resolve_runtime_backend_with_policy(
    requested_backend: RuntimeBackendName = "auto",
    *,
    allow_fallback: bool = True,
) -> RuntimeSelection:
    """Resolve a runtime backend while honoring the configured fallback policy.

    When fallback is allowed, unavailable or unknown backend requests resolve to
    the deterministic numpy fallback path. This preserves the historical CLI and
    config behavior that favors runnable smoke/CI executions over strict backend
    validation.
    """
    mlx_available = _mlx_available()
    if requested_backend in {AUTO_BACKEND, MLX_BACKEND} and mlx_available:
        return RuntimeSelection(requested_backend, MLX_BACKEND, _mlx_version())

    if requested_backend == MLX_BACKEND and not allow_fallback:
        raise RuntimeError("Prism MLX backend was requested, but MLX is unavailable.")

    if requested_backend in {AUTO_BACKEND, NUMPY_FALLBACK_BACKEND} or allow_fallback:
        return RuntimeSelection(
            requested_backend,
            NUMPY_FALLBACK_BACKEND,
            NUMPY_FALLBACK_VERSION,
            FALLBACK_LIMITATIONS,
        )

    raise RuntimeError(f"Unsupported Prism runtime backend: {requested_backend!r}")
