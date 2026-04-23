"""Benchmark helpers for Primordia."""
from evonn_primordia.benchmarks.datasets import get_benchmark, list_benchmarks
from evonn_primordia.benchmarks.parity import fallback_native_id, load_parity_pack, resolve_pack_path


def benchmark_group(spec):
    if spec.task == "language_modeling":
        return "language_modeling"
    if spec.id in {"blobs_f2_c2", "circles", "moons", "circles_n02_f3"}:
        return "synthetic"
    if spec.modality == "image" or spec.id in {"digits", "fashion_mnist", "mnist"}:
        return "image"
    return "tabular"


__all__ = [
    "benchmark_group",
    "fallback_native_id",
    "get_benchmark",
    "list_benchmarks",
    "load_parity_pack",
    "resolve_pack_path",
]
