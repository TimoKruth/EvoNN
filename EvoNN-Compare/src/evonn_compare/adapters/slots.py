"""Bridge Prism/Topograph to legacy symbiosis slot semantics."""

from __future__ import annotations

from evonn_compare.contracts.parity import ParityBenchmark

_PREFERRED_NATIVE_IDS: dict[str, dict[str, str]] = {
    "primordia": {
        "iris_classification": "iris",
        "wine_classification": "wine",
        "moons_classification": "moons",
        "digits_image": "digits",
        "credit_g_classification": "credit_g",
        "diabetes_regression": "diabetes",
        "friedman1_regression": "friedman1",
    },
    "contenders": {
        "iris_classification": "iris",
        "wine_classification": "wine",
        "moons_classification": "moons",
        "digits_image": "digits",
        "credit_g_classification": "credit_g",
        "diabetes_regression": "diabetes",
        "friedman1_regression": "friedman1",
    },
}


def canonical_slot(system: str) -> str:
    """Map runtime system names to legacy parity-pack slots."""

    return {
        "prism": "evonn",
        "topograph": "evonn2",
        "stratograph": "stratograph",
        "primordia": "primordia",
        "hybrid": "hybrid",
        "contenders": "contenders",
        "evonn": "evonn",
        "evonn2": "evonn2",
    }.get(system, system)


def system_display_name(system: str) -> str:
    """Human label for reports."""

    return {
        "prism": "Prism",
        "topograph": "Topograph",
        "stratograph": "Stratograph",
        "primordia": "Primordia",
        "hybrid": "Hybrid",
        "contenders": "Contenders",
        "evonn": "EvoNN",
        "evonn2": "EvoNN-2",
    }.get(system, system)


def fallback_native_id(benchmark: ParityBenchmark, system: str) -> str:
    """Resolve best native ID for a system from mixed old/new parity packs."""

    native_ids = benchmark.native_ids or {}
    if system == "prism":
        return (
            native_ids.get("prism")
            or native_ids.get("evonn")
            or native_ids.get("hybrid")
            or benchmark.benchmark_id
        )
    if system == "topograph":
        return (
            native_ids.get("topograph")
            or native_ids.get("evonn2")
            or native_ids.get("hybrid")
            or benchmark.benchmark_id
        )
    if system == "stratograph":
        return (
            native_ids.get("stratograph")
            or native_ids.get("prism")
            or native_ids.get("topograph")
            or native_ids.get("hybrid")
            or benchmark.benchmark_id
        )
    if system == "primordia":
        preferred = _PREFERRED_NATIVE_IDS["primordia"].get(benchmark.benchmark_id)
        return (
            native_ids.get("primordia")
            or preferred
            or native_ids.get("contenders")
            or native_ids.get("stratograph")
            or native_ids.get("prism")
            or native_ids.get("topograph")
            or native_ids.get("hybrid")
            or benchmark.benchmark_id
        )
    if system == "contenders":
        preferred = _PREFERRED_NATIVE_IDS["contenders"].get(benchmark.benchmark_id)
        return native_ids.get("contenders") or preferred or benchmark.benchmark_id
    return native_ids.get(system) or benchmark.benchmark_id
