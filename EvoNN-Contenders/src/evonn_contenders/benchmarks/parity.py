"""Canonical benchmark mapping and parity-pack loading."""

from __future__ import annotations

from pathlib import Path

import yaml
from pydantic import BaseModel, ConfigDict

from evonn_contenders.benchmarks.datasets import get_benchmark
from evonn_contenders.benchmarks.spec import BenchmarkSpec, MetricDirection, TaskKind


CANONICAL_BENCHMARK_IDS: dict[str, str] = {
    "blobs_f2_c2": "blobs_classification",
    "breast_cancer": "breast_cancer",
    "circles": "circles_classification",
    "credit_g": "credit_g_classification",
    "digits": "digits_image",
    "fashion_mnist": "fashionmnist_image",
    "iris": "iris_classification",
    "mnist": "mnist_image",
    "moons": "moons_classification",
    "adult": "openml_adult",
    "bank_marketing": "openml_bank_marketing",
    "blood_transfusion": "openml_blood_transfusion",
    "electricity": "openml_electricity",
    "gas_sensor": "openml_gas_sensor",
    "gesture_phase": "openml_gesture_phase",
    "heart_disease": "openml_heart_disease",
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
    "phoneme": "phoneme_classification",
    "qsar_biodeg": "qsar_biodeg_classification",
    "segment": "segment_classification",
    "steel_plates_fault": "steel_plates_fault_classification",
    "tiny_lm_synthetic": "tiny_lm_synthetic",
    "tinystories_lm": "tinystories_lm",
    "tinystories_lm_smoke": "tinystories_lm_smoke",
    "vehicle": "vehicle_classification",
    "wikitext2_lm": "wikitext2_lm",
    "wikitext2_lm_smoke": "wikitext2_lm_smoke",
    "wine": "wine_classification",
    "circles_n02_f3": "xor_tabular",
}

_REVERSE_IDS = {canonical: native for native, canonical in CANONICAL_BENCHMARK_IDS.items()}
_PROJECT_ROOT = Path(__file__).resolve().parents[3]
_PACK_SEARCH_DIRS = [
    _PROJECT_ROOT / "parity_packs",
    _PROJECT_ROOT / "parity_packs" / "generated",
    _PROJECT_ROOT.parent / "EvoNN-Compare" / "parity_packs",
    _PROJECT_ROOT.parent / "EvoNN-Compare" / "parity_packs" / "generated",
    _PROJECT_ROOT.parent / "EvoNN-Compare" / "manual_compare_runs" / "20260417_budget608_seed42_broad_w2_retry" / "packs",
    _PROJECT_ROOT.parent / "EvoNN-Compare" / "manual_compare_runs" / "20260416_budget38_seed42_smoke_valid" / "packs",
]


class BudgetPolicy(BaseModel):
    model_config = ConfigDict(frozen=True)
    evaluation_count: int
    epochs_per_candidate: int
    budget_tolerance_pct: float = 10.0


class SeedPolicy(BaseModel):
    model_config = ConfigDict(frozen=True)
    mode: str = "shared"
    required: bool = True


class ParityBenchmark(BaseModel):
    model_config = ConfigDict(frozen=True)
    benchmark_id: str
    native_ids: dict[str, str] | None = None
    task_kind: TaskKind
    metric_name: str
    metric_direction: MetricDirection


class ParityPack(BaseModel):
    model_config = ConfigDict(frozen=True)
    name: str
    benchmarks: list[ParityBenchmark]
    budget_policy: BudgetPolicy | None = None
    seed_policy: SeedPolicy | None = None


def get_canonical_id(native_name: str) -> str:
    return CANONICAL_BENCHMARK_IDS.get(native_name, native_name)


def get_native_id(canonical_id: str) -> str:
    return _REVERSE_IDS.get(canonical_id, canonical_id)


def native_id_candidates(entry: ParityBenchmark, system: str = "contenders") -> list[str]:
    native_ids = entry.native_ids or {}
    candidates = [
        native_ids.get(system),
        get_native_id(entry.benchmark_id),
        native_ids.get("evonn2"),
        native_ids.get("hybrid"),
        native_ids.get("stratograph"),
        native_ids.get("prism"),
        native_ids.get("topograph"),
        native_ids.get("evonn"),
        *native_ids.values(),
        entry.benchmark_id,
    ]
    ordered: list[str] = []
    seen: set[str] = set()
    for candidate in candidates:
        if not candidate or candidate in seen:
            continue
        seen.add(candidate)
        ordered.append(candidate)
    return ordered


def resolve_pack_path(pack_ref: str | Path) -> Path:
    path = Path(pack_ref)
    if path.exists():
        return path
    candidates = [path]
    if path.is_absolute():
        candidates.append(Path(path.name))
        if len(path.parts) >= 2:
            candidates.append(Path(*path.parts[-2:]))
    if path.suffix not in {".yaml", ".yml"}:
        candidates.extend([Path(f"{path}.yaml"), Path(f"{path}.yml")])
        if path.is_absolute():
            candidates.extend([Path(f"{path.name}.yaml"), Path(f"{path.name}.yml")])
    deduped_candidates: list[Path] = []
    seen: set[Path] = set()
    for candidate in candidates:
        if candidate not in seen:
            seen.add(candidate)
            deduped_candidates.append(candidate)
    for root in _PACK_SEARCH_DIRS:
        for candidate in deduped_candidates:
            resolved = root / candidate
            if resolved.exists():
                return resolved
    raise FileNotFoundError(f"Parity pack not found: {pack_ref}")


def fallback_native_id(entry: ParityBenchmark, system: str = "contenders") -> str:
    candidates = native_id_candidates(entry, system=system)
    for candidate in candidates:
        try:
            get_benchmark(candidate)
            return candidate
        except Exception:
            continue
    return candidates[0]


def load_parity_pack(pack_path: str | Path) -> ParityPack:
    resolved = resolve_pack_path(pack_path)
    payload = yaml.safe_load(resolved.read_text(encoding="utf-8"))
    if "benchmarks" not in payload and "benchmark_pack" in payload:
        pack = payload["benchmark_pack"] or {}
        payload = {"name": pack.get("pack_name", resolved.stem), "benchmarks": pack.get("benchmark_ids", [])}
    entries = payload.get("benchmarks", [])
    if entries and isinstance(entries[0], str):
        benchmarks = []
        for native_name in entries:
            spec = get_benchmark(native_name)
            benchmarks.append(
                ParityBenchmark(
                    benchmark_id=get_canonical_id(native_name),
                    native_ids={"contenders": native_name},
                    task_kind=spec.task,
                    metric_name=spec.metric_name,
                    metric_direction=spec.metric_direction,
                )
            )
        return ParityPack(
            name=payload["name"],
            benchmarks=benchmarks,
            budget_policy=BudgetPolicy(evaluation_count=len(entries), epochs_per_candidate=1),
            seed_policy=SeedPolicy(),
        )
    return ParityPack.model_validate(payload)


def load_pack_specs(pack_path: str | Path) -> list[BenchmarkSpec]:
    pack = load_parity_pack(pack_path)
    return [get_benchmark(fallback_native_id(entry)) for entry in pack.benchmarks]
