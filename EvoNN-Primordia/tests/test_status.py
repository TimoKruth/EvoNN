from __future__ import annotations

import json
from pathlib import Path

from evonn_primordia.config import load_config
from evonn_primordia.pipeline import run_search
from evonn_primordia.status import load_checkpoint, write_status


def test_write_status_persists_progress_payload(tmp_path: Path) -> None:
    path = write_status(
        tmp_path,
        run_id="run-1",
        run_name="demo",
        state="running",
        total_benchmarks=4,
        completed_benchmarks=["iris", "wine"],
        target_evaluation_count=16,
        evaluation_count=8,
        runtime_backend="numpy-fallback",
    )
    payload = json.loads(path.read_text(encoding="utf-8"))

    assert payload["state"] == "running"
    assert payload["completed_count"] == 2
    assert payload["remaining_count"] == 2
    assert payload["runtime_backend"] == "numpy-fallback"


def test_run_search_emits_status_and_checkpoint_artifacts(tmp_path: Path) -> None:
    config_path = tmp_path / "config.yaml"
    config_path.write_text(
        """
seed: 42
run_name: status_run
runtime:
  backend: numpy-fallback
benchmark_pool:
  name: status_run
  benchmarks: [iris]
search:
  mode: budget_matched
  target_evaluation_count: 2
training:
  epochs_per_candidate: 1
primitive_pool:
  tabular: [mlp, sparse_mlp]
  synthetic: [mlp]
  image: [mlp]
  language_modeling: [embedding]
""".strip()
        + "\n",
        encoding="utf-8",
    )
    config = load_config(config_path)
    run_dir = tmp_path / "run"
    run_search(config, run_dir=run_dir, config_path=config_path)

    status_payload = json.loads((run_dir / "status.json").read_text(encoding="utf-8"))
    checkpoint_payload = load_checkpoint(run_dir)

    assert status_payload["state"] == "complete"
    assert status_payload["completed_count"] == 1
    assert checkpoint_payload is not None
    assert checkpoint_payload["completed_benchmark_names"] == ["iris"]
