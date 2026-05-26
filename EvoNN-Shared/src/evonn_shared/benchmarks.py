"""Benchmark identity and resolution helpers shared across EvoNN packages."""

from __future__ import annotations

import os
from pathlib import Path
from typing import Literal

import yaml
from pydantic import BaseModel, ConfigDict


TaskKind = Literal["classification", "regression", "language_modeling"]
MetricDirection = Literal["max", "min"]


class BenchmarkDescriptor(BaseModel):
    """Minimal shared benchmark descriptor for compare-grade integration."""

    model_config = ConfigDict(frozen=True)

    benchmark_id: str
    task_kind: TaskKind
    metric_name: str
    metric_direction: MetricDirection
    benchmark_group: Literal["tabular", "synthetic", "image", "language_modeling"] | None = None
    domain: str | None = None
    difficulty: Literal["smoke", "core", "hard", "stress"] | None = None
    runtime_class: Literal["ci", "local", "overnight", "weekend", "special"] | None = None
    minimum_required_contenders: tuple[str, ...] = ()
    enhanced_optional_contenders: tuple[str, ...] = ()
    score_ceiling: float | None = None
    tie_tolerance_abs: float = 1e-12
    tie_tolerance_rel: float = 1e-12
    admission_notes: str = ""
    source: str | None = None
    native_name: str | None = None


def load_benchmark_descriptors(path: str | Path) -> list[BenchmarkDescriptor]:
    """Load benchmark descriptors from a YAML list or mapping."""

    payload = yaml.safe_load(Path(path).read_text(encoding="utf-8"))
    if isinstance(payload, dict):
        benchmarks = payload.get("benchmarks", [])
    else:
        benchmarks = payload
    return [BenchmarkDescriptor.model_validate(item) for item in benchmarks]


CANONICAL_BENCHMARK_IDS: dict[str, str] = {
    "abalone": "openml_abalone",
    "adult": "openml_adult",
    "airfoil": "openml_airfoil",
    "bank_marketing": "openml_bank_marketing",
    "blobs_f2_c2": "blobs_classification",
    "blood_transfusion": "openml_blood_transfusion",
    "breast_cancer": "breast_cancer",
    "circles": "circles_classification",
    "circles_n02_f3": "xor_tabular",
    "concrete": "openml_concrete",
    "cpu_performance": "openml_cpu_activity",
    "credit_g": "credit_g_classification",
    "diabetes": "diabetes_regression",
    "digits": "digits_image",
    "electricity": "openml_electricity",
    "energy_efficiency": "openml_energy_efficiency",
    "fashion_mnist": "fashionmnist_image",
    "friedman1": "friedman1_regression",
    "friedman_regression": "friedman1_regression",
    "gas_sensor": "openml_gas_sensor",
    "gesture_phase": "openml_gesture_phase",
    "heart_disease": "openml_heart_disease",
    "ilpd": "openml_ilpd",
    "iris": "iris_classification",
    "jungle_chess": "openml_jungle_chess",
    "kc1": "openml_kc1",
    "letter": "openml_letter",
    "mfeat_factors": "openml_mfeat_factors",
    "mnist": "mnist_image",
    "moons": "moons_classification",
    "nomao": "openml_nomao",
    "ozone_level": "openml_ozone_level",
    "phoneme": "openml_phoneme",
    "qsar_biodeg": "openml_qsar_biodeg",
    "segment": "openml_segment",
    "speed_dating": "openml_speed_dating",
    "steel_plates_fault": "openml_steel_plates_fault",
    "tiny_lm_synthetic": "tiny_lm_synthetic",
    "tinystories_lm": "tinystories_lm",
    "tinystories_lm_smoke": "tinystories_lm_smoke",
    "vehicle": "vehicle_classification",
    "wall_robot": "openml_wall_robot",
    "wikitext2_lm": "wikitext2_lm",
    "wikitext2_lm_smoke": "wikitext2_lm_smoke",
    "wilt": "openml_wilt",
    "wine": "wine_classification",
    "wine_quality": "openml_wine_quality",
}


def default_shared_benchmarks_root() -> Path:
    """Return the monorepo shared-benchmarks root."""

    return Path(__file__).resolve().parents[3] / "shared-benchmarks"


def shared_benchmarks_root(*, env_var: str = "EVONN_SHARED_BENCHMARKS_DIR") -> Path:
    """Resolve the shared-benchmarks root from env or monorepo layout."""

    override = os.environ.get(env_var)
    return Path(override).expanduser() if override else default_shared_benchmarks_root()


def canonical_benchmark_id(native_name: str, *, mapping: dict[str, str] | None = None) -> str:
    """Map a native benchmark id to a canonical compare id."""

    return (mapping or CANONICAL_BENCHMARK_IDS).get(native_name, native_name)


def native_benchmark_id(
    canonical_id: str,
    *,
    mapping: dict[str, str] | None = None,
    preferred: dict[str, str] | None = None,
) -> str:
    """Map a canonical compare id back to a native id when known."""

    if preferred and canonical_id in preferred:
        return preferred[canonical_id]
    reverse = {canonical: native for native, canonical in (mapping or CANONICAL_BENCHMARK_IDS).items()}
    return reverse.get(canonical_id, canonical_id)


def unique_paths(paths: list[Path]) -> list[Path]:
    """Deduplicate paths while preserving order."""

    unique: list[Path] = []
    seen: set[Path] = set()
    for path in paths:
        if path in seen:
            continue
        seen.add(path)
        unique.append(path)
    return unique


def shared_parity_dirs(shared_root: Path | None = None) -> list[Path]:
    """Return canonical shared parity-suite search directories."""

    root = shared_root or shared_benchmarks_root()
    return [root / "suites" / "parity", root / "suites"]


def parity_pack_search_dirs(
    *,
    default_dirs: list[Path],
    env_var: str,
    shared_root_env_var: str = "EVONN_SHARED_BENCHMARKS_DIR",
) -> list[Path]:
    """Build parity-pack search dirs from local defaults, shared dirs, and env overrides."""

    root = shared_benchmarks_root(env_var=shared_root_env_var)
    search_dirs = list(default_dirs) + shared_parity_dirs(root)
    env_value = os.environ.get(env_var, "")
    if env_value:
        search_dirs.extend(Path(raw).expanduser() for raw in env_value.split(os.pathsep) if raw)
    return unique_paths(search_dirs)


def resolve_parity_pack_path(
    pack_ref: str | Path,
    *,
    search_dirs: list[Path],
    env_var: str,
) -> Path:
    """Resolve a parity pack from a path or bare pack name."""

    path = Path(pack_ref)
    if path.exists():
        return path

    candidates = [path]
    if path.suffix not in {".yaml", ".yml"}:
        candidates.extend([Path(f"{path}.yaml"), Path(f"{path}.yml")])

    for root in search_dirs:
        for candidate in candidates:
            resolved = root / candidate
            if resolved.exists():
                return resolved

    searched = ", ".join(str(directory) for directory in search_dirs)
    raise FileNotFoundError(
        f"Parity pack not found: {pack_ref}. Checked: {searched}. "
        f"Set {env_var} to add external pack directories."
    )


def load_parity_pack_payload(path: str | Path) -> dict:
    """Load shared/simple parity-pack YAML into a normalized mapping."""

    resolved = Path(path)
    payload = yaml.safe_load(resolved.read_text(encoding="utf-8")) or {}
    if "benchmarks" not in payload and "benchmark_pack" in payload:
        pack = payload["benchmark_pack"] or {}
        return {"name": pack.get("pack_name", resolved.stem), "benchmarks": pack.get("benchmark_ids", [])}
    return payload


def native_id_from_entry(
    entry: object,
    *,
    system: str,
    fallback_systems: tuple[str, ...] = ("prism", "topograph", "stratograph", "primordia", "evonn2", "hybrid", "evonn"),
    mapping: dict[str, str] | None = None,
    preferred: dict[str, str] | None = None,
) -> str:
    """Resolve a native id from a rich parity entry without engine-specific imports."""

    if isinstance(entry, str):
        return entry
    if isinstance(entry, dict):
        native_ids = entry.get("native_ids") or {}
        benchmark_id = str(entry.get("benchmark_id") or "")
    else:
        native_ids = getattr(entry, "native_ids", None) or {}
        benchmark_id = str(getattr(entry, "benchmark_id", ""))

    direct = native_ids.get(system)
    if direct:
        return str(direct)
    for fallback in fallback_systems:
        candidate = native_ids.get(fallback)
        if candidate:
            return native_benchmark_id(
                canonical_benchmark_id(str(candidate), mapping=mapping),
                mapping=mapping,
                preferred=preferred,
            )
    return native_benchmark_id(benchmark_id, mapping=mapping, preferred=preferred)
