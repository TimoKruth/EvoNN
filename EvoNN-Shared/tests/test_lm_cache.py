from __future__ import annotations

from pathlib import Path

import numpy as np

from evonn_shared.lm_cache import LMCacheSpec, generate_lm_cache, make_byte_lm_windows, validate_lm_cache


def test_byte_lm_windows_are_next_token_aligned() -> None:
    text = "A small real text sample. " * 500

    x_train, y_train, x_val, y_val = make_byte_lm_windows(
        text,
        context_length=16,
        train_windows=4,
        val_windows=2,
        stride=8,
    )

    assert x_train.shape == (4, 16)
    assert y_train.shape == (4, 16)
    assert x_val.shape == (2, 16)
    assert y_val.shape == (2, 16)
    assert np.all(y_train[:, :-1] == x_train[:, 1:])
    assert np.all(y_val[:, :-1] == x_val[:, 1:])


def test_validate_lm_cache_rejects_linear_successor_fixture(tmp_path: Path) -> None:
    x = np.tile(np.arange(32, dtype=np.int32), (4, 1))
    y = x + 1
    path = tmp_path / "linear.npz"
    np.savez_compressed(path, x_train=x, y_train=y, x_val=x, y_val=y)

    report = validate_lm_cache(path)

    assert report["ok"] is False
    assert any("linear successor pattern" in blocker for blocker in report["blockers"])


def test_validate_lm_cache_reports_empty_arrays(tmp_path: Path) -> None:
    empty = np.empty((0, 32), dtype=np.int32)
    path = tmp_path / "empty.npz"
    np.savez_compressed(path, x_train=empty, y_train=empty, x_val=empty, y_val=empty)

    report = validate_lm_cache(path)

    assert report["ok"] is False
    assert any("train arrays are empty" in blocker for blocker in report["blockers"])
    assert any("val arrays are empty" in blocker for blocker in report["blockers"])


def test_generate_lm_cache_from_local_text_passes_validation(tmp_path: Path) -> None:
    text_path = tmp_path / "corpus.txt"
    text_path.write_text(
        (
            "Once upon a time, a small robot read a weathered manual. "
            "The manual contained jokes, lists, names, punctuation, and many ordinary words. "
            "0123456789 !?;: @#$% &() [] {} <>=+-*/ "
        )
        * 2000,
        encoding="utf-8",
    )

    report = generate_lm_cache(
        LMCacheSpec(
            dataset="demo_lm",
            source=str(text_path),
            context_length=32,
            train_windows=16,
            val_windows=4,
            stride=16,
        ),
        output_dir=tmp_path,
    )

    assert report["ok"] is True
    assert Path(str(report["path"])).exists()
