"""Built-in benchmark registry for contender repo."""

from __future__ import annotations

import os
from pathlib import Path

from evonn_contenders.benchmarks.spec import BenchmarkSpec


_BUILTIN_BENCHMARKS: dict[str, BenchmarkSpec] = {
    "blobs_f2_c2": BenchmarkSpec(name="blobs_f2_c2", task="classification", source="sklearn", dataset="make_blobs", input_dim=2, num_classes=2, n_samples=1200, centers=2, cluster_std=1.2),
    "breast_cancer": BenchmarkSpec(name="breast_cancer", task="classification", source="sklearn", dataset="load_breast_cancer", input_dim=30, num_classes=2),
    "circles": BenchmarkSpec(name="circles", task="classification", source="sklearn", dataset="make_circles", input_dim=2, num_classes=2, n_samples=1200, noise=0.08, factor=0.5),
    "credit_g": BenchmarkSpec(name="credit_g", task="classification", source="openml", dataset="credit_g", input_dim=20, num_classes=2),
    "digits": BenchmarkSpec(name="digits", task="classification", source="sklearn", dataset="load_digits", input_dim=64, num_classes=10),
    "fashion_mnist": BenchmarkSpec(name="fashion_mnist", task="classification", source="image", dataset="fashion_mnist", input_dim=784, num_classes=10),
    "iris": BenchmarkSpec(name="iris", task="classification", source="sklearn", dataset="load_iris", input_dim=4, num_classes=3),
    "mnist": BenchmarkSpec(name="mnist", task="classification", source="image", dataset="mnist", input_dim=784, num_classes=10),
    "moons": BenchmarkSpec(name="moons", task="classification", source="sklearn", dataset="make_moons", input_dim=2, num_classes=2, n_samples=1200, noise=0.18),
    "adult": BenchmarkSpec(name="adult", task="classification", source="openml", dataset="adult", input_dim=14, num_classes=2),
    "bank_marketing": BenchmarkSpec(name="bank_marketing", task="classification", source="openml", dataset="bank_marketing", input_dim=16, num_classes=2),
    "blood_transfusion": BenchmarkSpec(name="blood_transfusion", task="classification", source="openml", dataset="blood_transfusion", input_dim=4, num_classes=2),
    "electricity": BenchmarkSpec(name="electricity", task="classification", source="openml", dataset="electricity", input_dim=8, num_classes=2),
    "gas_sensor": BenchmarkSpec(name="gas_sensor", task="classification", source="openml", dataset="gas_sensor", input_dim=128, num_classes=6),
    "gesture_phase": BenchmarkSpec(name="gesture_phase", task="classification", source="openml", dataset="gesture_phase", input_dim=32, num_classes=5),
    "heart_disease": BenchmarkSpec(name="heart_disease", task="classification", source="openml", dataset="heart_disease", input_dim=13, num_classes=2),
    "ilpd": BenchmarkSpec(name="ilpd", task="classification", source="openml", dataset="ilpd", input_dim=10, num_classes=2),
    "jungle_chess": BenchmarkSpec(name="jungle_chess", task="classification", source="openml", dataset="jungle_chess", input_dim=6, num_classes=3),
    "kc1": BenchmarkSpec(name="kc1", task="classification", source="openml", dataset="kc1", input_dim=21, num_classes=2),
    "letter": BenchmarkSpec(name="letter", task="classification", source="openml", dataset="letter", input_dim=16, num_classes=26),
    "mfeat_factors": BenchmarkSpec(name="mfeat_factors", task="classification", source="openml", dataset="mfeat_factors", input_dim=216, num_classes=10),
    "nomao": BenchmarkSpec(name="nomao", task="classification", source="openml", dataset="nomao", input_dim=118, num_classes=2),
    "ozone_level": BenchmarkSpec(name="ozone_level", task="classification", source="openml", dataset="ozone_level", input_dim=72, num_classes=2),
    "speed_dating": BenchmarkSpec(name="speed_dating", task="classification", source="openml", dataset="speed_dating", input_dim=120, num_classes=2),
    "wall_robot": BenchmarkSpec(name="wall_robot", task="classification", source="openml", dataset="wall_robot", input_dim=24, num_classes=4),
    "wilt": BenchmarkSpec(name="wilt", task="classification", source="openml", dataset="wilt", input_dim=5, num_classes=2),
    "phoneme": BenchmarkSpec(name="phoneme", task="classification", source="openml", dataset="phoneme", input_dim=5, num_classes=2),
    "qsar_biodeg": BenchmarkSpec(name="qsar_biodeg", task="classification", source="openml", dataset="qsar_biodeg", input_dim=41, num_classes=2),
    "segment": BenchmarkSpec(name="segment", task="classification", source="openml", dataset="segment", input_dim=19, num_classes=7),
    "steel_plates_fault": BenchmarkSpec(name="steel_plates_fault", task="classification", source="openml", dataset="steel_plates_fault", input_dim=27, num_classes=7),
    "vehicle": BenchmarkSpec(name="vehicle", task="classification", source="openml", dataset="vehicle", input_dim=18, num_classes=4),
    "wine": BenchmarkSpec(name="wine", task="classification", source="sklearn", dataset="load_wine", input_dim=13, num_classes=3),
    "circles_n02_f3": BenchmarkSpec(name="circles_n02_f3", task="classification", source="sklearn", dataset="make_circles", input_dim=2, num_classes=2, n_samples=1200, noise=0.2, factor=0.3),
    "tiny_lm_synthetic": BenchmarkSpec(name="tiny_lm_synthetic", task="language_modeling", source="lm_synthetic", input_dim=128, num_classes=256, n_samples=2048),
    "tinystories_lm": BenchmarkSpec(name="tinystories_lm", task="language_modeling", source="lm_cache", dataset="tinystories_lm", input_dim=256, num_classes=8192, max_train_samples=4096, max_val_samples=512),
    "tinystories_lm_smoke": BenchmarkSpec(name="tinystories_lm_smoke", task="language_modeling", source="lm_cache", dataset="tinystories_lm_smoke", input_dim=256, num_classes=8192, max_train_samples=1024, max_val_samples=256),
    "wikitext2_lm": BenchmarkSpec(name="wikitext2_lm", task="language_modeling", source="lm_cache", dataset="wikitext2_lm", input_dim=256, num_classes=32768, max_train_samples=4096, max_val_samples=512),
    "wikitext2_lm_smoke": BenchmarkSpec(name="wikitext2_lm_smoke", task="language_modeling", source="lm_cache", dataset="wikitext2_lm_smoke", input_dim=256, num_classes=32768, max_train_samples=4096, max_val_samples=512),
}

_PACKAGE_DIR = Path(__file__).resolve().parent
_PROJECT_ROOT = _PACKAGE_DIR.parent.parent.parent
_SUPERPROJECT_ROOT = _PROJECT_ROOT.parent
_LOCAL_CATALOG_DIR = _PROJECT_ROOT / "benchmarks" / "catalog"
_DEFAULT_SHARED_ROOT = _SUPERPROJECT_ROOT / "shared-benchmarks"
_CATALOG_ENV_VAR = "CONTENDERS_CATALOG_DIR"
_SHARED_ROOT_ENV_VAR = "EVONN_SHARED_BENCHMARKS_DIR"


def _catalog_search_dirs() -> list[Path]:
    dirs: list[Path] = []
    explicit = os.environ.get(_CATALOG_ENV_VAR)
    if explicit:
        dirs.append(Path(explicit).expanduser())

    shared_root = os.environ.get(_SHARED_ROOT_ENV_VAR)
    if shared_root:
        root = Path(shared_root).expanduser()
    else:
        root = _DEFAULT_SHARED_ROOT
    dirs.append(root if root.name == "catalog" else root / "catalog")
    dirs.append(_LOCAL_CATALOG_DIR)

    unique: list[Path] = []
    seen: set[Path] = set()
    for path in dirs:
        if path in seen:
            continue
        seen.add(path)
        unique.append(path)
    return unique


def _catalog_specs() -> dict[str, BenchmarkSpec]:
    specs: dict[str, BenchmarkSpec] = {}
    for root in _catalog_search_dirs():
        if not root.exists():
            continue
        for path in sorted(root.glob("*.yaml")):
            try:
                spec = BenchmarkSpec.from_yaml(path)
            except Exception:
                continue
            specs.setdefault(spec.name, spec)
    return specs


def list_benchmarks() -> list[BenchmarkSpec]:
    """Return all known benchmarks sorted by native name."""
    catalog_specs = _catalog_specs()
    if catalog_specs:
        return [catalog_specs[name] for name in sorted(catalog_specs)]
    return [_BUILTIN_BENCHMARKS[name] for name in sorted(_BUILTIN_BENCHMARKS)]


def get_benchmark(name: str) -> BenchmarkSpec:
    """Resolve native benchmark id to spec."""
    catalog_specs = _catalog_specs()
    if name in catalog_specs:
        return catalog_specs[name]

    try:
        from evonn_contenders.benchmarks.parity import _REVERSE_IDS

        native = _REVERSE_IDS.get(name)
        if native and native in catalog_specs:
            return catalog_specs[native]
    except Exception:
        pass

    try:
        return _BUILTIN_BENCHMARKS[name]
    except KeyError as exc:
        raise KeyError(f"Unknown contender benchmark: {name}") from exc
