from __future__ import annotations

import importlib.util
from pathlib import Path

import yaml


def _load_smoke_module():
    root = Path(__file__).resolve().parents[1]
    path = root / "scripts" / "smoke_33plus5_bench.py"
    spec = importlib.util.spec_from_file_location("topograph_smoke_33plus5_module", path)
    module = importlib.util.module_from_spec(spec)
    assert spec.loader is not None
    spec.loader.exec_module(module)
    return module


def _shared_benchmarks_root() -> Path:
    return Path(__file__).resolve().parents[2] / "shared-benchmarks"


def test_smoke_33plus5_has_exact_benchmark_set(monkeypatch):
    smoke = _load_smoke_module()
    assert len(smoke.SHARED_33_BENCHMARKS) == 33
    assert len(smoke.LM_5_BENCHMARKS) == 5
    assert len(smoke.BENCHMARKS) == 38

    monkeypatch.setattr(smoke.sys, "argv", ["smoke_33plus5_bench.py", "--dry-run", "--limit", "2"])
    exit_code = smoke.main()
    assert exit_code == 0


def test_shared_33plus5_parity_pack_matches_smoke_set():
    pack_path = _shared_benchmarks_root() / "suites" / "parity" / "shared_33plus5.yaml"
    payload = yaml.safe_load(pack_path.read_text(encoding="utf-8"))

    smoke = _load_smoke_module()
    expected = [native for _, native in smoke.BENCHMARKS]

    assert payload["name"] == "shared_33plus5"
    assert payload["benchmarks"] == expected
    assert len(payload["benchmarks"]) == 38


def test_render_markdown_report_contains_summary_and_lm_section():
    smoke = _load_smoke_module()
    payload = {
        "benchmark_count": 2,
        "total_elapsed_seconds": 1.23,
        "results": [
            {
                "benchmark_id": "moons_classification",
                "native_id": "moons",
                "status": "ok",
                "task": "classification",
                "metric_name": "accuracy",
                "metric_value": 0.8,
                "native_fitness": 0.2,
                "elapsed_seconds": 0.4,
            },
            {
                "benchmark_id": "tiny_lm_synthetic",
                "native_id": "tiny_lm_synthetic",
                "status": "ok",
                "task": "language_modeling",
                "metric_name": "perplexity",
                "metric_value": 123.4,
                "native_fitness": 4.8,
                "elapsed_seconds": 0.7,
            },
        ],
    }

    text = smoke.render_markdown_report(payload)
    assert "# Topograph Smoke 33+5 Report" in text
    assert "## Summary" in text
    assert "## Language Modeling" in text
    assert "tiny_lm_synthetic" in text


def test_resolve_output_dim_expands_lm_vocab_when_tokens_exceed_catalog_cap():
    smoke = _load_smoke_module()

    x_train = [[1, 2], [3, 4]]
    y_train = [[2, 3], [4, 5]]

    assert smoke._resolve_output_dim(
        task="language_modeling",
        declared=4,
        x_train=smoke.np.array(x_train),
        y_train=smoke.np.array(y_train),
    ) == 6
