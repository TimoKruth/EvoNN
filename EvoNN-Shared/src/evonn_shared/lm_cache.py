"""Shared language-modeling cache generation and validation helpers."""

from __future__ import annotations

from dataclasses import dataclass
import io
import json
from pathlib import Path
import urllib.request
import zipfile

import numpy as np


DEFAULT_CONTEXT_LENGTH = 256
DEFAULT_TRAIN_WINDOWS = 4096
DEFAULT_VAL_WINDOWS = 512
DEFAULT_STRIDE = 128
DEFAULT_VOCAB_SIZE = 256
LINEAR_SUCCESSOR_BLOCK_THRESHOLD = 0.20
MIN_UNIQUE_TOKENS = 32

DEFAULT_TEXT_SOURCES = {
    "tinystories_lm": {
        "url": "https://huggingface.co/datasets/roneneldan/TinyStories/resolve/main/TinyStoriesV2-GPT4-valid.txt",
        "kind": "plain",
        "max_bytes": 4_000_000,
    },
    "wikitext2_lm": {
        "url": "https://raw.githubusercontent.com/pytorch/examples/main/word_language_model/data/wikitext-2/valid.txt",
        "kind": "plain",
        "max_bytes": 2_000_000,
    },
}


@dataclass(frozen=True)
class LMCacheSpec:
    """Generation spec for a byte-level next-token LM cache."""

    dataset: str
    source: str
    source_kind: str = "plain"
    zip_member: str | None = None
    context_length: int = DEFAULT_CONTEXT_LENGTH
    train_windows: int = DEFAULT_TRAIN_WINDOWS
    val_windows: int = DEFAULT_VAL_WINDOWS
    stride: int = DEFAULT_STRIDE
    vocab_size: int = DEFAULT_VOCAB_SIZE
    max_source_bytes: int | None = None


def default_shared_root() -> Path:
    """Return the monorepo shared-benchmarks root."""

    return Path(__file__).resolve().parents[3] / "shared-benchmarks"


def default_lm_cache_dir() -> Path:
    """Return the canonical shared LM cache directory."""

    return default_shared_root() / "lm_cache"


def default_lm_cache_spec(dataset: str) -> LMCacheSpec:
    """Build a generation spec for a known LM dataset."""

    try:
        source = DEFAULT_TEXT_SOURCES[dataset]
    except KeyError as exc:
        known = ", ".join(sorted(DEFAULT_TEXT_SOURCES))
        raise ValueError(f"unknown default LM dataset {dataset!r}; known: {known}") from exc
    return LMCacheSpec(
        dataset=dataset,
        source=str(source["url"]),
        source_kind=str(source.get("kind", "plain")),
        zip_member=source.get("zip_member"),
        max_source_bytes=int(source["max_bytes"]) if source.get("max_bytes") is not None else None,
    )


def generate_lm_cache(spec: LMCacheSpec, *, output_dir: Path | None = None) -> dict[str, object]:
    """Generate a byte-level next-token NPZ cache from a real text source."""

    target_dir = output_dir or default_lm_cache_dir()
    target_dir.mkdir(parents=True, exist_ok=True)
    text = read_text_source(spec.source, kind=spec.source_kind, zip_member=spec.zip_member, max_bytes=spec.max_source_bytes)
    x_train, y_train, x_val, y_val = make_byte_lm_windows(
        text,
        context_length=spec.context_length,
        train_windows=spec.train_windows,
        val_windows=spec.val_windows,
        stride=spec.stride,
    )
    metadata = {
        "format": "evonn_byte_lm_cache_v1",
        "dataset": spec.dataset,
        "source": spec.source,
        "source_kind": spec.source_kind,
        "zip_member": spec.zip_member,
        "context_length": spec.context_length,
        "train_windows": spec.train_windows,
        "val_windows": spec.val_windows,
        "stride": spec.stride,
        "vocab_size": spec.vocab_size,
        "tokenizer": "utf8_byte",
    }
    output_path = target_dir / f"{spec.dataset}.npz"
    np.savez_compressed(
        output_path,
        x_train=x_train,
        y_train=y_train,
        x_val=x_val,
        y_val=y_val,
        metadata=json.dumps(metadata, sort_keys=True),
    )
    report = validate_lm_cache(output_path)
    report["path"] = str(output_path)
    report["metadata"] = metadata
    return report


def read_text_source(
    source: str,
    *,
    kind: str = "plain",
    zip_member: str | None = None,
    max_bytes: int | None = None,
) -> str:
    """Read text from a URL or local path."""

    payload = _read_bytes(source, max_bytes=max_bytes)
    if kind == "zip":
        with zipfile.ZipFile(io.BytesIO(payload)) as archive:
            member = zip_member or _first_text_member(archive)
            payload = archive.read(member)
    elif kind != "plain":
        raise ValueError(f"unsupported LM text source kind: {kind}")
    text = payload.decode("utf-8", errors="replace")
    if len(text.strip()) < 10_000:
        raise ValueError(f"LM text source is too small after decoding: {source}")
    return text


def make_byte_lm_windows(
    text: str,
    *,
    context_length: int = DEFAULT_CONTEXT_LENGTH,
    train_windows: int = DEFAULT_TRAIN_WINDOWS,
    val_windows: int = DEFAULT_VAL_WINDOWS,
    stride: int = DEFAULT_STRIDE,
) -> tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    """Tokenize text as UTF-8 bytes and create next-token windows."""

    if context_length <= 0:
        raise ValueError("context_length must be positive")
    if stride <= 0:
        raise ValueError("stride must be positive")
    token_bytes = np.frombuffer(text.encode("utf-8", errors="replace"), dtype=np.uint8).astype(np.int32)
    required = (train_windows + val_windows) * stride + context_length + 2
    if token_bytes.size < required:
        raise ValueError(
            "not enough text for requested LM windows: "
            f"tokens={token_bytes.size}, required>={required}"
        )

    train_x, train_y = _window_slice(token_bytes, start=0, count=train_windows, context_length=context_length, stride=stride)
    val_start = train_windows * stride + context_length
    val_x, val_y = _window_slice(
        token_bytes,
        start=val_start,
        count=val_windows,
        context_length=context_length,
        stride=stride,
    )
    return train_x, train_y, val_x, val_y


def validate_lm_cache(path: Path | str) -> dict[str, object]:
    """Validate that an LM cache looks like real byte-text next-token data."""

    cache_path = Path(path)
    payload = np.load(cache_path, allow_pickle=False)
    required = ("x_train", "y_train", "x_val", "y_val")
    missing = [name for name in required if name not in payload.files]
    blockers: list[str] = []
    warnings: list[str] = []
    if missing:
        return {
            "path": str(cache_path),
            "ok": False,
            "blockers": [f"missing arrays: {', '.join(missing)}"],
            "warnings": [],
        }

    arrays = {name: payload[name] for name in required}
    metadata = _decode_metadata(payload)
    for split in ("train", "val"):
        x = arrays[f"x_{split}"]
        y = arrays[f"y_{split}"]
        if x.shape != y.shape:
            blockers.append(f"{split} x/y shape mismatch: {x.shape} != {y.shape}")
        if x.ndim != 2:
            blockers.append(f"{split} x must be 2D, got shape {x.shape}")
        if x.size == 0 or y.size == 0:
            blockers.append(f"{split} arrays are empty")
        if np.min(x) < 0 or np.min(y) < 0:
            blockers.append(f"{split} contains negative token ids")
        if np.max(x) >= DEFAULT_VOCAB_SIZE or np.max(y) >= DEFAULT_VOCAB_SIZE:
            blockers.append(f"{split} contains token ids outside byte vocabulary")

    combined_x = np.concatenate([arrays["x_train"].reshape(-1), arrays["x_val"].reshape(-1)])
    combined_y = np.concatenate([arrays["y_train"].reshape(-1), arrays["y_val"].reshape(-1)])
    unique_tokens = int(np.unique(np.concatenate([combined_x, combined_y])).size)
    linear_successor_fraction = _linear_successor_fraction(arrays)
    next_token_alignment_fraction = _next_token_alignment_fraction(arrays)
    if unique_tokens < MIN_UNIQUE_TOKENS:
        blockers.append(f"too few unique byte tokens: {unique_tokens}")
    if linear_successor_fraction > LINEAR_SUCCESSOR_BLOCK_THRESHOLD:
        blockers.append(
            "linear successor pattern detected: "
            f"y == x + 1 for {linear_successor_fraction:.3f} of positions"
        )
    if next_token_alignment_fraction < 0.95:
        blockers.append(
            "next-token window alignment looks invalid: "
            f"y[t] == x[t+1] for {next_token_alignment_fraction:.3f} of inner positions"
        )
    if metadata.get("format") != "evonn_byte_lm_cache_v1":
        warnings.append("cache metadata missing or not generated by evonn_byte_lm_cache_v1")

    return {
        "path": str(cache_path),
        "ok": not blockers,
        "blockers": blockers,
        "warnings": warnings,
        "metadata": metadata,
        "stats": {
            "x_train_shape": list(arrays["x_train"].shape),
            "x_val_shape": list(arrays["x_val"].shape),
            "unique_tokens": unique_tokens,
            "linear_successor_fraction": linear_successor_fraction,
            "next_token_alignment_fraction": next_token_alignment_fraction,
            "train_target_entropy_bits": _entropy_bits(arrays["y_train"]),
            "val_target_entropy_bits": _entropy_bits(arrays["y_val"]),
        },
    }


def validate_default_lm_cache(dataset: str, *, cache_dir: Path | None = None) -> dict[str, object]:
    """Validate a dataset in the canonical shared LM cache directory."""

    root = cache_dir or default_lm_cache_dir()
    return validate_lm_cache(root / f"{_canonical_dataset_name(dataset)}.npz")


def _canonical_dataset_name(dataset: str) -> str:
    return dataset.removesuffix("_smoke")


def _read_bytes(source: str, *, max_bytes: int | None) -> bytes:
    if source.startswith(("http://", "https://")):
        with urllib.request.urlopen(source, timeout=60) as response:
            if max_bytes is None:
                return response.read()
            return response.read(max_bytes)
    path = Path(source).expanduser()
    data = path.read_bytes()
    return data[:max_bytes] if max_bytes is not None else data


def _first_text_member(archive: zipfile.ZipFile) -> str:
    for name in archive.namelist():
        if name.endswith(".txt") and not name.endswith("/"):
            return name
    raise ValueError("zip source does not contain a text member")


def _window_slice(
    tokens: np.ndarray,
    *,
    start: int,
    count: int,
    context_length: int,
    stride: int,
) -> tuple[np.ndarray, np.ndarray]:
    x = np.empty((count, context_length), dtype=np.int32)
    y = np.empty((count, context_length), dtype=np.int64)
    for index in range(count):
        offset = start + index * stride
        x[index] = tokens[offset : offset + context_length]
        y[index] = tokens[offset + 1 : offset + context_length + 1]
    return x, y


def _decode_metadata(payload: np.lib.npyio.NpzFile) -> dict[str, object]:
    if "metadata" not in payload.files:
        return {}
    raw = payload["metadata"]
    try:
        if raw.shape == ():
            return json.loads(str(raw.item()))
        return json.loads(str(raw.reshape(-1)[0]))
    except Exception:
        return {}


def _linear_successor_fraction(arrays: dict[str, np.ndarray]) -> float:
    matches = []
    for split in ("train", "val"):
        x = arrays[f"x_{split}"]
        y = arrays[f"y_{split}"]
        matches.append((y == (x + 1)).reshape(-1))
    return float(np.mean(np.concatenate(matches)))


def _next_token_alignment_fraction(arrays: dict[str, np.ndarray]) -> float:
    matches = []
    for split in ("train", "val"):
        x = arrays[f"x_{split}"]
        y = arrays[f"y_{split}"]
        matches.append((y[:, :-1] == x[:, 1:]).reshape(-1))
    return float(np.mean(np.concatenate(matches)))


def _entropy_bits(values: np.ndarray) -> float:
    flat = values.reshape(-1)
    counts = np.bincount(flat.astype(np.int64), minlength=DEFAULT_VOCAB_SIZE)
    probs = counts[counts > 0] / max(1, int(counts.sum()))
    return float(-np.sum(probs * np.log2(probs))) if probs.size else 0.0
