"""Runtime backend resolution for Stratograph."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Literal

try:  # pragma: no cover - depends on host runtime
    import mlx

    MLX_AVAILABLE = True
    MLX_VERSION = getattr(mlx, "__version__", None)
except ImportError:  # pragma: no cover - covered on non-MLX hosts
    mlx = None
    MLX_AVAILABLE = False
    MLX_VERSION = None


RuntimeBackendName = Literal["auto", "mlx", "numpy-fallback"]

FALLBACK_LIMITATIONS = (
    "numpy-fallback prioritizes correctness, CI portability, and compare-contract "
    "compatibility over MLX-quality parity or equivalent performance."
)


@dataclass(frozen=True)
class RuntimeSelection:
    requested_backend: RuntimeBackendName
    resolved_backend: Literal["mlx", "numpy-fallback"]
    runtime_version: str | None
    backend_limitations: str | None = None


def resolve_runtime_backend(requested_backend: RuntimeBackendName = "auto") -> RuntimeSelection:
    """Resolve requested backend to an executable backend on this host."""
    return _resolve_runtime_backend(requested_backend, allow_fallback=True)


def _resolve_runtime_backend(
    requested_backend: RuntimeBackendName,
    *,
    allow_fallback: bool,
) -> RuntimeSelection:
    if requested_backend == "auto":
        if MLX_AVAILABLE:
            return RuntimeSelection(
                requested_backend="auto",
                resolved_backend="mlx",
                runtime_version=MLX_VERSION,
            )
        if not allow_fallback:
            raise RuntimeError("Requested runtime backend 'auto' but MLX is not available and fallback is disabled.")
        return RuntimeSelection(
            requested_backend="auto",
            resolved_backend="numpy-fallback",
            runtime_version=None,
            backend_limitations=FALLBACK_LIMITATIONS,
        )
    if requested_backend == "mlx":
        if not MLX_AVAILABLE:
            raise RuntimeError("Requested runtime backend 'mlx' but MLX is not available on this host.")
        return RuntimeSelection(
            requested_backend="mlx",
            resolved_backend="mlx",
            runtime_version=MLX_VERSION,
        )
    if requested_backend == "numpy-fallback":
        return RuntimeSelection(
            requested_backend="numpy-fallback",
            resolved_backend="numpy-fallback",
            runtime_version=None,
            backend_limitations=FALLBACK_LIMITATIONS,
        )
    raise ValueError(f"Unsupported runtime backend: {requested_backend}")


def resolve_runtime_backend_with_policy(
    requested_backend: RuntimeBackendName = "auto",
    *,
    allow_fallback: bool = True,
) -> RuntimeSelection:
    """Resolve requested backend with explicit fallback policy."""
    return _resolve_runtime_backend(requested_backend, allow_fallback=allow_fallback)
