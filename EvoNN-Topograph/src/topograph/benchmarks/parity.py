"""Canonical benchmark ID mapping and parity pack loading for symbiosis."""

from __future__ import annotations

import os
from pathlib import Path

import yaml

from topograph.benchmarks.spec import BenchmarkSpec


# Maps native Topograph benchmark names to canonical symbiosis IDs.
# Kept in sync with EvoNN-2's CANONICAL_BENCHMARK_IDS.
CANONICAL_BENCHMARK_IDS: dict[str, str] = {
    # Tier 1
    "iris": "iris_classification",
    "wine": "wine_classification",
    "moons": "moons_classification",
    "digits": "digits_image",
    "diabetes": "diabetes_regression",
    "friedman1": "friedman1_regression",
    "credit_g": "credit_g_classification",
    # Tier 2
    "mnist": "mnist_image",
    "fashion_mnist": "fashionmnist_image",
    "segment": "segment_classification",
    "vehicle": "vehicle_classification",
    "qsar_biodeg": "qsar_biodeg_classification",
    "steel_plates_fault": "steel_plates_fault_classification",
    # Tier 3
    "circles": "circles_classification",
    "blobs_f2_c2": "blobs_classification",
    "phoneme": "phoneme_classification",
    "circles_n02_f3": "xor_tabular",
    # OpenML / shared tabular
    "adult": "openml_adult",
    "bank_marketing": "openml_bank_marketing",
    "blood_transfusion": "openml_blood_transfusion",
    "electricity": "openml_electricity",
    "ilpd": "openml_ilpd",
    "jungle_chess": "openml_jungle_chess",
    "kc1": "openml_kc1",
    "letter": "openml_letter",
    "mfeat_factors": "openml_mfeat_factors",
    "nomao": "openml_nomao",
    "ozone_level": "openml_ozone_level",
    "speed_dating": "openml_speed_dating",
    "wall_robot": "openml_wall_robot",
    "wilt": "openml_wilt",
    # Bridge benchmarks
    "abalone": "openml_abalone",
    "airfoil": "openml_airfoil",
    "concrete": "openml_concrete",
    "cpu_performance": "openml_cpu_activity",
    "energy_efficiency": "openml_energy_efficiency",
    "gas_sensor": "openml_gas_sensor",
    "gesture_phase": "openml_gesture_phase",
    "heart_disease": "openml_heart_disease",
    "wine_quality": "openml_wine_quality",
    # Language modeling
    "tiny_lm_synthetic": "tiny_lm_synthetic",
    "tinystories_lm": "tinystories_lm",
    "wikitext2_lm": "wikitext2_lm",
}

# Reverse lookup: canonical -> native
_REVERSE_IDS = {v: k for k, v in CANONICAL_BENCHMARK_IDS.items()}

_PACKAGE_DIR = Path(__file__).resolve().parent
_PROJECT_ROOT = _PACKAGE_DIR.parent.parent.parent
_SUPERPROJECT_ROOT = _PROJECT_ROOT.parent
_LOCAL_CATALOG_DIR = _PROJECT_ROOT / "benchmarks" / "catalog"
_LOCAL_SUITES_DIR = _PROJECT_ROOT / "benchmarks" / "suites"
_DEFAULT_SHARED_ROOT = _SUPERPROJECT_ROOT / "shared-benchmarks"
_CATALOG_ENV_VAR = "TOPOGRAPH_CATALOG_DIR"
_SUITES_ENV_VAR = "TOPOGRAPH_SUITES_DIR"
_SHARED_ROOT_ENV_VAR = "EVONN_SHARED_BENCHMARKS_DIR"


def _shared_root_dir() -> Path:
    shared_root = os.environ.get(_SHARED_ROOT_ENV_VAR)
    if shared_root:
        return Path(shared_root).expanduser()
    return _DEFAULT_SHARED_ROOT


def _resolve_catalog_dir() -> Path:
    explicit = os.environ.get(_CATALOG_ENV_VAR)
    if explicit:
        return Path(explicit).expanduser()
    shared = _shared_root_dir() / "catalog"
    if shared.exists():
        return shared
    return _LOCAL_CATALOG_DIR


def _resolve_suites_dir() -> Path:
    explicit = os.environ.get(_SUITES_ENV_VAR)
    if explicit:
        return Path(explicit).expanduser()
    shared = _shared_root_dir() / "suites"
    if shared.exists():
        return shared
    return _LOCAL_SUITES_DIR


def get_canonical_id(native_name: str) -> str:
    """Map a native Topograph benchmark name to its canonical symbiosis ID.

    Returns the native name unchanged if no mapping exists.
    """
    return CANONICAL_BENCHMARK_IDS.get(native_name, native_name)


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
    with open(pack_path) as f:
        data = yaml.safe_load(f)

    entries = data.get("benchmarks", [])
    specs: list[BenchmarkSpec] = []

    for entry in entries:
        if isinstance(entry, str):
            name = entry
        elif isinstance(entry, dict):
            native_ids = entry.get("native_ids", {})
            name = (
                native_ids.get("topograph")
                or native_ids.get("evonn2")
                or native_ids.get("hybrid")
                or entry.get("benchmark_id", "")
            )
            if name in _REVERSE_IDS:
                name = _REVERSE_IDS[name]
        else:
            continue
        specs.append(get_benchmark(name))

    return specs


def load_benchmark_suite_names(suite: str | Path) -> list[str]:
    """Load benchmark names from a suite YAML.

    Suite lookup order:
    1. explicit path
    2. `benchmarks/suites/topograph/<suite>.yaml`
    3. `benchmarks/suites/parity/<suite>.yaml`
    """
    path = _resolve_suite_path(suite)
    with open(path) as f:
        data = yaml.safe_load(f) or {}

    names: list[str] = []
    for entry in data.get("benchmarks", []):
        if isinstance(entry, str):
            names.append(entry)
            continue
        if not isinstance(entry, dict):
            continue
        native_ids = entry.get("native_ids", {})
        name = (
            native_ids.get("topograph")
            or native_ids.get("evonn2")
            or native_ids.get("hybrid")
            or entry.get("benchmark_id", "")
        )
        if name in _REVERSE_IDS:
            name = _REVERSE_IDS[name]
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
    native = _REVERSE_IDS.get(name)
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
    if not catalog_dir.exists():
        return []
    names: list[str] = []
    for p in sorted(catalog_dir.glob("*.yaml")):
        with open(p) as f:
            data = yaml.safe_load(f)
        if data and "name" in data:
            names.append(p.stem)
    return names
