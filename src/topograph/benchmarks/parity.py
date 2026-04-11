"""Canonical benchmark ID mapping and parity pack loading for symbiosis."""

from __future__ import annotations

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
}

# Reverse lookup: canonical -> native
_REVERSE_IDS = {v: k for k, v in CANONICAL_BENCHMARK_IDS.items()}

CATALOG_DIR = Path(__file__).resolve().parent.parent.parent.parent / "benchmarks" / "catalog"


def get_canonical_id(native_name: str) -> str:
    """Map a native Topograph benchmark name to its canonical symbiosis ID.

    Returns the native name unchanged if no mapping exists.
    """
    return CANONICAL_BENCHMARK_IDS.get(native_name, native_name)


def load_parity_pack(pack_path: str | Path) -> list[BenchmarkSpec]:
    """Load a YAML parity pack file and resolve each benchmark to a BenchmarkSpec.

    Expected YAML structure::

        name: tier1_core
        benchmarks:
          - iris
          - moons
          - ...
    """
    with open(pack_path) as f:
        data = yaml.safe_load(f)

    benchmark_names: list[str] = data.get("benchmarks", [])
    specs: list[BenchmarkSpec] = []

    for name in benchmark_names:
        specs.append(get_benchmark(name))

    return specs


def get_benchmark(name: str) -> BenchmarkSpec:
    """Get a BenchmarkSpec by name: tries catalog YAML first, then raises."""
    catalog_path = CATALOG_DIR / f"{name}.yaml"
    if catalog_path.exists():
        return BenchmarkSpec.from_yaml(catalog_path)

    # Try reverse canonical lookup
    native = _REVERSE_IDS.get(name)
    if native:
        alt_path = CATALOG_DIR / f"{native}.yaml"
        if alt_path.exists():
            return BenchmarkSpec.from_yaml(alt_path)

    raise FileNotFoundError(
        f"Benchmark '{name}' not found in catalog at {CATALOG_DIR}"
    )


def list_benchmarks() -> list[str]:
    """List all benchmark names available in the catalog."""
    if not CATALOG_DIR.exists():
        return []
    names: list[str] = []
    for p in sorted(CATALOG_DIR.glob("*.yaml")):
        with open(p) as f:
            data = yaml.safe_load(f)
        if data and "name" in data:
            names.append(p.stem)
    return names
