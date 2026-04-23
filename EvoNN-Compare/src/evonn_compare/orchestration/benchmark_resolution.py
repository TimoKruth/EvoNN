"""Resolve compare-pack benchmark ids to names each package can actually load."""
from __future__ import annotations

from functools import lru_cache
import importlib
from pathlib import Path
import subprocess
from typing import Callable

from evonn_compare.adapters.slots import fallback_native_id
from evonn_compare.contracts.parity import ParityBenchmark

_COMPARE_ROOT = Path(__file__).resolve().parents[3]
_WORKSPACE_ROOT = _COMPARE_ROOT.parent

_GET_BENCHMARK_IMPORTS: dict[str, tuple[str, str]] = {
    "prism": ("prism.benchmarks.datasets", "get_benchmark"),
    "topograph": ("topograph.benchmarks.parity", "get_benchmark"),
    "stratograph": ("stratograph.benchmarks", "get_benchmark"),
    "primordia": ("evonn_primordia.benchmarks", "get_benchmark"),
    "contenders": ("evonn_contenders.benchmarks.datasets", "get_benchmark"),
}

_PROJECT_ROOTS: dict[str, Path] = {
    "prism": _WORKSPACE_ROOT / "EvoNN-Prism",
    "topograph": _WORKSPACE_ROOT / "EvoNN-Topograph",
    "stratograph": _WORKSPACE_ROOT / "EvoNN-Stratograph",
    "primordia": _WORKSPACE_ROOT / "EvoNN-Primordia",
    "contenders": _WORKSPACE_ROOT / "EvoNN-Contenders",
}


@lru_cache(maxsize=None)
def _get_benchmark_loader(system: str) -> Callable[[str], object] | None:
    target = _GET_BENCHMARK_IMPORTS.get(system)
    if target is None:
        return None
    module_name, attr = target
    try:
        module = importlib.import_module(module_name)
    except Exception:
        return None
    return getattr(module, attr, None)


def resolve_supported_benchmark_id(benchmark: ParityBenchmark, system: str) -> str:
    candidates = _candidate_ids(benchmark, system)
    for candidate in candidates:
        if _benchmark_supported(system, candidate):
            return candidate
    return candidates[0]


def resolve_supported_benchmark_ids(benchmarks: list[ParityBenchmark], system: str) -> list[str]:
    return [resolve_supported_benchmark_id(entry, system) for entry in benchmarks]


def _benchmark_supported(system: str, candidate: str) -> bool:
    loader = _get_benchmark_loader(system)
    if loader is not None:
        try:
            loader(candidate)
            return True
        except Exception:
            pass
    return _probe_benchmark_in_project_env(system, candidate)


def _probe_benchmark_in_project_env(system: str, candidate: str) -> bool:
    target = _GET_BENCHMARK_IMPORTS.get(system)
    project_root = _PROJECT_ROOTS.get(system)
    if target is None or project_root is None:
        return False
    module_name, attr = target
    try:
        process = subprocess.run(
            [
                "uv",
                "run",
                "python",
                "-c",
                (
                    "import importlib; "
                    f"loader = getattr(importlib.import_module({module_name!r}), {attr!r}); "
                    f"loader({candidate!r})"
                ),
            ],
            cwd=project_root,
            stdout=subprocess.DEVNULL,
            stderr=subprocess.DEVNULL,
            check=False,
        )
    except OSError:
        return False
    return process.returncode == 0


def _candidate_ids(benchmark: ParityBenchmark, system: str) -> list[str]:
    native_ids = benchmark.native_ids or {}
    ordered = [native_ids.get(system)]
    ordered.extend(native_ids.values())
    ordered.extend([
        native_ids.get("prism"),
        native_ids.get("topograph"),
        native_ids.get("stratograph"),
        native_ids.get("primordia"),
        native_ids.get("contenders"),
        native_ids.get("evonn"),
        native_ids.get("evonn2"),
        benchmark.benchmark_id,
        fallback_native_id(benchmark, system),
    ])
    deduped: list[str] = []
    seen: set[str] = set()
    for candidate in ordered:
        if not candidate or candidate in seen:
            continue
        seen.add(candidate)
        deduped.append(candidate)
    return deduped
