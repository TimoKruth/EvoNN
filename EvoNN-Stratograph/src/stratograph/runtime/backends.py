"""Runtime backend resolution for Stratograph."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Literal

try:  # pragma: no cover - depends on host runtime
    import mlx
    import mlx.core

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

RUNTIME_POLICY_NAME = "bounded_hierarchy_runtime_v2"


@dataclass(frozen=True)
class RuntimeSelection:
    requested_backend: RuntimeBackendName
    resolved_backend: Literal["mlx", "numpy-fallback"]
    runtime_version: str | None
    backend_limitations: str | None = None


@dataclass(frozen=True)
class RuntimeExecutionPolicy:
    """Portable runtime limits that should be visible in run artifacts."""

    name: str = RUNTIME_POLICY_NAME
    classification_train_cap: int = 1024
    classification_val_cap: int = 512
    openml_train_cap: int = 2048
    openml_val_cap: int = 1024
    image_train_cap: int = 768
    image_val_cap: int = 384
    lm_default_train_cap: int = 2048
    lm_default_val_cap: int = 512
    lm_train_token_cap: int = 24_576
    lm_val_token_cap: int = 8_192
    classifier_step_floor: int = 10
    image_classifier_step_floor: int = 6
    lm_step_floor: int = 8

    def as_metadata(self, *, resolved_backend: str) -> dict[str, int | str]:
        return {
            "runtime_policy_name": self.name,
            "runtime_backend": resolved_backend,
            "classification_train_cap": self.classification_train_cap,
            "classification_val_cap": self.classification_val_cap,
            "openml_train_cap": self.openml_train_cap,
            "openml_val_cap": self.openml_val_cap,
            "image_train_cap": self.image_train_cap,
            "image_val_cap": self.image_val_cap,
            "lm_default_train_cap": self.lm_default_train_cap,
            "lm_default_val_cap": self.lm_default_val_cap,
            "lm_train_token_cap": self.lm_train_token_cap,
            "lm_val_token_cap": self.lm_val_token_cap,
            "classifier_step_floor": self.classifier_step_floor,
            "image_classifier_step_floor": self.image_classifier_step_floor,
            "lm_step_floor": self.lm_step_floor,
        }


def resolve_runtime_backend(requested_backend: RuntimeBackendName = "auto") -> RuntimeSelection:
    """Resolve requested backend to an executable backend on this host."""
    return _resolve_runtime_backend(requested_backend, allow_fallback=True)


def runtime_execution_policy() -> RuntimeExecutionPolicy:
    """Return the portable evaluator policy used by both MLX and fallback paths."""
    return RuntimeExecutionPolicy()


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
