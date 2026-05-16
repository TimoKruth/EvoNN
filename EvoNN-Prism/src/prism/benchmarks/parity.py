"""Canonical benchmark ID mapping and parity pack loading for symbiosis."""

from __future__ import annotations

import os
from collections.abc import Mapping
from pathlib import Path
from typing import Any, cast

import yaml

from prism.benchmarks.spec import BenchmarkSpec


# Maps native Prism benchmark names to canonical symbiosis IDs.
# Kept in sync with EvoNN, EvoNN-2, and Topograph mappings.
CANONICAL_BENCHMARK_IDS: dict[str, str] = {
    # Tier 1 — core symmetric benchmarks
    "iris": "iris_classification",
    "wine": "wine_classification",
    "moons": "moons_classification",
    "digits": "digits_image",
    "diabetes": "diabetes_regression",
    "friedman1": "friedman1_regression",
    "credit_g": "credit_g_classification",
    "breast_cancer": "breast_cancer",
    # Tier 2 — image/tabular (EvoNN-leaning)
    "mnist": "mnist_image",
    "fashion_mnist": "fashionmnist_image",
    "vehicle": "vehicle_classification",
    # Tier 3 — topology benchmarks (EvoNN-2-leaning)
    "circles": "circles_classification",
    "blobs_f2_c2": "blobs_classification",
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
    "phoneme": "openml_phoneme",
    "ozone_level": "openml_ozone_level",
    "qsar_biodeg": "openml_qsar_biodeg",
    "segment": "openml_segment",
    "speed_dating": "openml_speed_dating",
    "steel_plates_fault": "openml_steel_plates_fault",
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
}

# Reverse lookup: canonical -> native
_REVERSE_IDS: dict[str, str] = {v: k for k, v in CANONICAL_BENCHMARK_IDS.items()}
_PACKAGE_DIR = Path(__file__).resolve().parent
_PROJECT_ROOT = _PACKAGE_DIR.parent.parent.parent
_SUPERPROJECT_ROOT = _PROJECT_ROOT.parent
_PACK_ENV_VAR = "PRISM_PARITY_PACK_DIRS"
_SHARED_ROOT_ENV_VAR = "EVONN_SHARED_BENCHMARKS_DIR"
_PACK_FILE_SUFFIX = ".yaml"
_DEFAULT_PACK_SEARCH_DIRS = (
    _PROJECT_ROOT / "parity_packs",
    _PROJECT_ROOT / "parity_packs" / "generated",
)


def _shared_pack_dirs() -> list[Path]:
    shared_root = os.environ.get(_SHARED_ROOT_ENV_VAR)
    if shared_root:
        root = Path(shared_root).expanduser()
    else:
        root = _SUPERPROJECT_ROOT / "shared-benchmarks"
    return [
        root / "suites" / "parity",
        root / "suites",
    ]


def _pack_search_dirs() -> list[Path]:
    search_dirs = list(_DEFAULT_PACK_SEARCH_DIRS) + _shared_pack_dirs()
    env_value = os.environ.get(_PACK_ENV_VAR, "")
    if env_value:
        for raw_path in env_value.split(os.pathsep):
            if raw_path:
                search_dirs.append(Path(raw_path).expanduser())
    return search_dirs


def _pack_path_candidates(path: Path) -> tuple[Path, ...]:
    if path.suffix == _PACK_FILE_SUFFIX:
        return (path,)
    return (path, Path(f"{path}{_PACK_FILE_SUFFIX}"))


def _native_name_from_entry(entry: str | Mapping[str, Any]) -> str:
    if isinstance(entry, str):
        return entry

    native_ids = cast(Mapping[str, str], entry.get("native_ids", {}))
    name = (
        native_ids.get("prism")
        or native_ids.get("evonn2")
        or native_ids.get("hybrid")
        or cast(str, entry.get("benchmark_id", ""))
    )
    return _REVERSE_IDS.get(name, name)


def get_canonical_id(native_name: str) -> str:
    """Map a native Prism benchmark name to its canonical symbiosis ID.

    Returns the native name unchanged if no mapping exists.
    """
    return CANONICAL_BENCHMARK_IDS.get(native_name, native_name)


def resolve_pack_path(pack_ref: str | Path) -> Path:
    """Resolve a parity pack from a path or bare pack name."""
    path = Path(pack_ref)
    if path.exists():
        return path

    search_dirs = _pack_search_dirs()
    for directory in search_dirs:
        for candidate in _pack_path_candidates(path):
            resolved = directory / candidate
            if resolved.exists():
                return resolved

    searched = ", ".join(str(directory) for directory in search_dirs)
    raise FileNotFoundError(
        f"Parity pack not found: {pack_ref}. "
        f"Checked local repo pack dirs: {searched}. "
        f"Set {_PACK_ENV_VAR} to add external pack directories."
    )


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

    with open(resolved_pack, encoding="utf-8") as f:
        data = yaml.safe_load(f)

    entries = data.get("benchmarks", [])
    specs: list[BenchmarkSpec] = []

    for entry in entries:
        if not isinstance(entry, str | Mapping):
            continue

        specs.append(get_benchmark(_native_name_from_entry(entry)))

    return specs
