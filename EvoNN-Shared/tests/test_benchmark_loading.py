from __future__ import annotations

from pathlib import Path

import pytest

from evonn_shared.benchmarks import (
    canonical_benchmark_id,
    load_parity_pack_payload,
    native_benchmark_id,
    native_id_from_entry,
    parity_pack_search_dirs,
    resolve_parity_pack_path,
)


def test_canonical_and_native_benchmark_ids_round_trip() -> None:
    assert canonical_benchmark_id("digits") == "digits_image"
    assert native_benchmark_id("digits_image") == "digits"
    assert native_benchmark_id("diabetes_regression", preferred={"diabetes_regression": "diabetes"}) == "diabetes"
    assert canonical_benchmark_id("custom_engine_only") == "custom_engine_only"


def test_parity_pack_resolution_supports_shared_root_override(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    shared_root = tmp_path / "shared-benchmarks"
    pack_dir = shared_root / "suites" / "parity"
    pack_dir.mkdir(parents=True)
    pack_path = pack_dir / "demo.yaml"
    pack_path.write_text("name: demo\nbenchmarks:\n  - iris\n", encoding="utf-8")
    monkeypatch.setenv("EVONN_SHARED_BENCHMARKS_DIR", str(shared_root))

    search_dirs = parity_pack_search_dirs(default_dirs=[], env_var="DEMO_PARITY_PACK_DIRS")

    assert resolve_parity_pack_path("demo", search_dirs=search_dirs, env_var="DEMO_PARITY_PACK_DIRS") == pack_path
    assert load_parity_pack_payload(pack_path)["benchmarks"] == ["iris"]


def test_native_id_from_entry_resolves_rich_pack_fallbacks() -> None:
    entry = {
        "benchmark_id": "diabetes_regression",
        "native_ids": {
            "stratograph": "diabetes_regression",
            "topograph": "diabetes",
        },
    }

    assert native_id_from_entry(entry, system="topograph") == "diabetes"
    assert (
        native_id_from_entry(
            entry,
            system="primordia",
            fallback_systems=("topograph", "stratograph"),
            preferred={"diabetes_regression": "diabetes"},
        )
        == "diabetes"
    )
