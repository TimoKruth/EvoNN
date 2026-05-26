"""Canonical benchmark ID mapping and parity pack loading for symbiosis."""

from __future__ import annotations

from pathlib import Path

from evonn_shared.benchmarks import (
    CANONICAL_BENCHMARK_IDS as CANONICAL_BENCHMARK_IDS,
    canonical_benchmark_id,
    load_parity_pack_payload,
    native_benchmark_id,
    native_id_from_entry,
    parity_pack_search_dirs,
    resolve_parity_pack_path,
)

from prism.benchmarks.spec import BenchmarkSpec


_PACKAGE_DIR = Path(__file__).resolve().parent
_PROJECT_ROOT = _PACKAGE_DIR.parent.parent.parent
_PACK_ENV_VAR = "PRISM_PARITY_PACK_DIRS"
_DEFAULT_PACK_SEARCH_DIRS = [
    _PROJECT_ROOT / "parity_packs",
    _PROJECT_ROOT / "parity_packs" / "generated",
]


def _pack_search_dirs() -> list[Path]:
    return parity_pack_search_dirs(default_dirs=_DEFAULT_PACK_SEARCH_DIRS, env_var=_PACK_ENV_VAR)


def get_canonical_id(native_name: str) -> str:
    """Map a native Prism benchmark name to its canonical symbiosis ID.

    Returns the native name unchanged if no mapping exists.
    """
    return canonical_benchmark_id(native_name)


def get_native_id(canonical_id: str) -> str:
    """Map canonical compare id to Prism native id when known."""
    return native_benchmark_id(canonical_id)


def resolve_pack_path(pack_ref: str | Path) -> Path:
    """Resolve a parity pack from a path or bare pack name."""
    return resolve_parity_pack_path(pack_ref, search_dirs=_pack_search_dirs(), env_var=_PACK_ENV_VAR)


def load_parity_pack(pack_path: str | Path) -> list[BenchmarkSpec]:
    """Load a YAML parity pack file and resolve each benchmark to a BenchmarkSpec.

    Supports two formats:

    Simple (Topograph-style)::

        name: tier1_core
        benchmarks:
          - iris
          - moons

    Rich (Symbiosis-style)::

        name: tier1_core
        benchmarks:
          - benchmark_id: iris_classification
            native_ids:
              prism: iris
              evonn2: iris
    """
    from prism.benchmarks.datasets import get_benchmark

    resolved_pack = resolve_pack_path(pack_path)

    data = load_parity_pack_payload(resolved_pack)

    entries = data.get("benchmarks", [])
    specs: list[BenchmarkSpec] = []

    for entry in entries:
        if not isinstance(entry, str | dict):
            continue
        name = native_id_from_entry(entry, system="prism")
        specs.append(get_benchmark(name))

    return specs
