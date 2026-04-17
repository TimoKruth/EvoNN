"""Benchmark loaders for EvoNN-Compare hybrid runs."""

from __future__ import annotations

from pathlib import Path

import numpy as np

from evonn_compare.adapters.slots import fallback_native_id
from evonn_compare.contracts.parity import load_parity_pack
from evonn_compare.shared_benchmarks import get_benchmark

EVONN_ROOT = Path(__file__).resolve().parents[4] / "EvoNN"
EVONN_CACHE = EVONN_ROOT / ".cache" / "evonn"


def load_parity_pack_benchmarks(
    pack_name_or_path: str | Path,
    *,
    seed: int = 42,
) -> dict[str, tuple]:
    """Load benchmarks for a parity pack, including LM bridge datasets."""

    pack = load_parity_pack(pack_name_or_path)
    benchmark_map: dict[str, tuple] = {}
    lm_benchmarks: dict[str, tuple] | None = None

    for benchmark in pack.benchmarks:
        if benchmark.task_kind == "language_modeling":
            if lm_benchmarks is None:
                lm_benchmarks = load_lm_benchmarks(seed=seed)
            native_name = fallback_native_id(benchmark, "hybrid")
            if benchmark.benchmark_id in lm_benchmarks:
                benchmark_map[benchmark.benchmark_id] = lm_benchmarks[benchmark.benchmark_id]
                continue
            if native_name in lm_benchmarks:
                benchmark_map[benchmark.benchmark_id] = lm_benchmarks[native_name]
                continue
            raise FileNotFoundError(
                f"LM benchmark not available for hybrid runner: {benchmark.benchmark_id} ({native_name})"
            )

        native_name = fallback_native_id(benchmark, "topograph")
        spec = get_benchmark(native_name)
        X_train, y_train, X_val, y_val = spec.load_data(seed=seed)
        num_classes = getattr(spec, "num_classes", None)
        if benchmark.task_kind == "classification" and (num_classes is None or num_classes <= 1):
            classes = set(y_train.tolist()) | set(y_val.tolist())
            num_classes = len(classes)
        elif benchmark.task_kind == "regression":
            num_classes = 1
        benchmark_map[benchmark.benchmark_id] = (
            X_train,
            y_train,
            X_val,
            y_val,
            benchmark.task_kind,
            int(num_classes),
        )

    return benchmark_map


def load_lm_benchmarks(seed: int = 42) -> dict[str, tuple]:
    """Load hybrid LM bridge benchmarks from synthetic gen or EvoNN cache."""

    benchmarks: dict[str, tuple] = {}
    benchmarks["tiny_lm_synthetic"] = _make_tiny_lm_synthetic(seed=seed)

    for benchmark_id, vocab_size in (("tinystories_lm", 4096), ("wikitext2_lm", 4096)):
        cached = _load_cached_lm_benchmark(benchmark_id, vocab_size=vocab_size)
        if cached is not None:
            benchmarks[benchmark_id] = cached

    return benchmarks


def _make_tiny_lm_synthetic(seed: int = 42) -> tuple:
    rng = np.random.default_rng(seed)
    vocab_size = 256
    seq_len = 128
    n_train, n_val = 4000, 1000

    def _make_sequences(n: int) -> np.ndarray:
        sequences = np.zeros((n, seq_len + 1), dtype=np.int32)
        for idx in range(n):
            pattern_type = rng.integers(0, 4)
            if pattern_type == 0:
                subseq_len = rng.integers(3, 8)
                subseq = rng.integers(0, vocab_size, size=subseq_len)
                full = np.tile(subseq, (seq_len + 1) // subseq_len + 1)[: seq_len + 1]
                sequences[idx] = full
            elif pattern_type == 1:
                start = rng.integers(0, vocab_size)
                step = rng.integers(1, 4)
                sequences[idx] = np.array(
                    [(start + i * step) % vocab_size for i in range(seq_len + 1)],
                    dtype=np.int32,
                )
            elif pattern_type == 2:
                a, b = rng.integers(0, vocab_size, size=2)
                sequences[idx] = np.array(
                    [a if i % 2 == 0 else b for i in range(seq_len + 1)],
                    dtype=np.int32,
                )
            else:
                subseq_len = rng.integers(4, 12)
                subseq = rng.integers(0, vocab_size, size=subseq_len)
                full = np.tile(subseq, (seq_len + 1) // subseq_len + 1)[: seq_len + 1]
                noise_mask = rng.random(seq_len + 1) < 0.05
                noise_tokens = rng.integers(0, vocab_size, size=seq_len + 1)
                sequences[idx] = np.where(noise_mask, noise_tokens, full)
        return sequences

    train_sequences = _make_sequences(n_train)
    val_sequences = _make_sequences(n_val)
    return (
        train_sequences[:, :seq_len].astype(np.float32),
        train_sequences[:, -1].astype(np.int64),
        val_sequences[:, :seq_len].astype(np.float32),
        val_sequences[:, 1:].astype(np.int64),
        "language_modeling",
        vocab_size,
    )


def _load_cached_lm_benchmark(benchmark_id: str, *, vocab_size: int) -> tuple | None:
    cache_path = EVONN_CACHE / "datasets" / f"{benchmark_id}.npz"
    if not cache_path.exists():
        return None

    try:
        data = np.load(cache_path)
        x_train = data["x_train"]
        y_train = data["y_train"]
        x_val = data["x_val"]
        y_val = data["y_val"]

        max_train = 10000
        max_val = 2000
        if x_train.shape[0] > max_train:
            x_train = x_train[:max_train]
            y_train = y_train[:max_train]
        if x_val.shape[0] > max_val:
            x_val = x_val[:max_val]
            y_val = y_val[:max_val]

        return (
            x_train.astype(np.float32),
            y_train[:, -1].astype(np.int64),
            x_val.astype(np.float32),
            y_val.astype(np.int64),
            "language_modeling",
            vocab_size,
        )
    except (EOFError, OSError, KeyError, ValueError):
        return None
