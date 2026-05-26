"""Canonical benchmark ID mapping and parity pack loading for symbiosis."""

from __future__ import annotations

import os
from pathlib import Path

import yaml
from evonn_shared.benchmarks import (
    CANONICAL_BENCHMARK_IDS as CANONICAL_BENCHMARK_IDS,
    canonical_benchmark_id,
    load_parity_pack_payload,
    native_benchmark_id,
    native_id_from_entry,
    shared_benchmarks_root,
)

from topograph.benchmarks.spec import BenchmarkSpec


_PACKAGE_DIR = Path(__file__).resolve().parent
_PROJECT_ROOT = _PACKAGE_DIR.parent.parent.parent
_LOCAL_CATALOG_DIR = _PROJECT_ROOT / "benchmarks" / "catalog"
_LOCAL_SUITES_DIR = _PROJECT_ROOT / "benchmarks" / "suites"
_CATALOG_ENV_VAR = "TOPOGRAPH_CATALOG_DIR"
_SUITES_ENV_VAR = "TOPOGRAPH_SUITES_DIR"
_SHARED_ROOT_ENV_VAR = "EVONN_SHARED_BENCHMARKS_DIR"


def _shared_root_dir() -> Path:
    return shared_benchmarks_root(env_var=_SHARED_ROOT_ENV_VAR)


def _has_local_catalog() -> bool:
    return any(_LOCAL_CATALOG_DIR.glob("*.yaml"))


def _has_local_suites() -> bool:
    return any(_LOCAL_SUITES_DIR.rglob("*.yaml"))


def _resolve_catalog_dir() -> Path:
    explicit = os.environ.get(_CATALOG_ENV_VAR)
    if explicit:
        path = Path(explicit).expanduser()
        if not path.exists():
            raise FileNotFoundError(
                f"Topograph catalog override not found: {path}. "
                f"Set {_CATALOG_ENV_VAR} to a valid catalog directory."
            )
        return path
    shared = _shared_root_dir() / "catalog"
    if shared.exists():
        return shared
    if _has_local_catalog():
        return _LOCAL_CATALOG_DIR
    raise FileNotFoundError(
        f"Shared benchmark catalog not found at {shared}. "
        f"Set {_SHARED_ROOT_ENV_VAR} to the shared-benchmarks root or {_CATALOG_ENV_VAR} "
        f"to a catalog directory."
    )


def _resolve_suites_dir() -> Path:
    explicit = os.environ.get(_SUITES_ENV_VAR)
    if explicit:
        path = Path(explicit).expanduser()
        if not path.exists():
            raise FileNotFoundError(
                f"Topograph suite override not found: {path}. "
                f"Set {_SUITES_ENV_VAR} to a valid suites directory."
            )
        return path
    shared = _shared_root_dir() / "suites"
    if shared.exists():
        return shared
    if _has_local_suites():
        return _LOCAL_SUITES_DIR
    raise FileNotFoundError(
        f"Shared benchmark suites not found at {shared}. "
        f"Set {_SHARED_ROOT_ENV_VAR} to the shared-benchmarks root or {_SUITES_ENV_VAR} "
        f"to a suites directory."
    )


def get_canonical_id(native_name: str) -> str:
    """Map a native Topograph benchmark name to its canonical symbiosis ID.

    Returns the native name unchanged if no mapping exists.
    """
    return canonical_benchmark_id(native_name)


def load_parity_pack(pack_path: str | Path) -> list[BenchmarkSpec]:
    """Load a YAML parity pack file and resolve each benchmark to a BenchmarkSpec.

    Supports two formats:

    Simple::

        name: tier1_core
        benchmarks:
          - iris
          - moons

    Rich::

        name: tier1_core
        benchmarks:
          - benchmark_id: iris_classification
            native_ids:
              topograph: iris
              prism: iris
    """
    data = load_parity_pack_payload(pack_path)
    entries = data.get("benchmarks", [])
    specs: list[BenchmarkSpec] = []

    for entry in entries:
        name = native_id_from_entry(entry, system="topograph")
        if not name:
            continue
        specs.append(get_benchmark(name))

    return specs


def load_benchmark_suite_names(suite: str | Path) -> list[str]:
    """Load benchmark names from a suite YAML.

    Suite lookup order:
    1. explicit path
    2. `../shared-benchmarks/suites/topograph/<suite>.yaml`
    3. `../shared-benchmarks/suites/parity/<suite>.yaml`
    4. local `benchmarks/suites/...` fallback if shared root absent
    """
    path = _resolve_suite_path(suite)
    data = load_parity_pack_payload(path)

    names: list[str] = []
    for entry in data.get("benchmarks", []):
        name = native_id_from_entry(entry, system="topograph")
        if name:
            names.append(name)
    return names


def resolve_benchmark_pool_names(pool_cfg) -> list[str]:
    """Resolve a benchmark pool config into a deduplicated ordered benchmark list."""
    names: list[str] = []
    seen: set[str] = set()

    suite = getattr(pool_cfg, "suite", None)
    if suite:
        for name in load_benchmark_suite_names(suite):
            if name not in seen:
                seen.add(name)
                names.append(name)

    for name in getattr(pool_cfg, "benchmarks", []) or []:
        if name not in seen:
            seen.add(name)
            names.append(name)
    return names


def _resolve_suite_path(suite: str | Path) -> Path:
    suites_dir = _resolve_suites_dir()
    candidate = Path(suite).expanduser()
    if candidate.is_absolute() or candidate.suffix == ".yaml" or "/" in str(suite):
        if candidate.exists():
            return candidate
        if not candidate.is_absolute():
            relative = suites_dir / candidate
            if relative.exists():
                return relative

    simple_name = str(suite)
    search_paths = [
        suites_dir / "topograph" / f"{simple_name}.yaml",
        suites_dir / "parity" / f"{simple_name}.yaml",
        suites_dir / "common" / f"{simple_name}.yaml",
    ]
    for path in search_paths:
        if path.exists():
            return path
    raise FileNotFoundError(f"Benchmark suite not found: {suite}")


def get_benchmark(name: str) -> BenchmarkSpec:
    """Get a BenchmarkSpec by name: tries catalog YAML first, then raises."""
    catalog_dir = _resolve_catalog_dir()
    catalog_path = catalog_dir / f"{name}.yaml"
    if catalog_path.exists():
        return BenchmarkSpec.from_yaml(catalog_path)

    # Try reverse canonical lookup
    native = native_benchmark_id(name)
    if native:
        alt_path = catalog_dir / f"{native}.yaml"
        if alt_path.exists():
            return BenchmarkSpec.from_yaml(alt_path)

    raise FileNotFoundError(
        f"Benchmark '{name}' not found in catalog at {catalog_dir}"
    )


def list_benchmarks() -> list[str]:
    """List all benchmark names available in the catalog."""
    catalog_dir = _resolve_catalog_dir()
    names: list[str] = []
    for p in sorted(catalog_dir.glob("*.yaml")):
        with open(p) as f:
            data = yaml.safe_load(f)
        if data and "name" in data:
            names.append(p.stem)
    return names
