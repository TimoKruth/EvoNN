from __future__ import annotations

import subprocess
from pathlib import Path

from evonn_compare.orchestration.contenders import ensure_contender_export


def test_contender_export_materialize_then_export(monkeypatch, tmp_path: Path) -> None:
    calls: list[list[str]] = []

    def fake_run(argv, cwd, text, capture_output):
        calls.append(argv)
        return subprocess.CompletedProcess(argv, 0, stdout="", stderr="")

    monkeypatch.setattr("subprocess.run", fake_run)
    contenders_root = tmp_path / "EvoNN-Contenders"
    config_path = contenders_root / "configs" / "cached.yaml"
    pack_path = tmp_path / "pack.yaml"
    run_dir = tmp_path / "workspace" / "contenders"
    artifacts = ensure_contender_export(
        contenders_root=contenders_root,
        config_path=config_path,
        pack_path=pack_path,
        run_dir=run_dir,
    )

    assert calls[0][3] == "materialize"
    assert calls[1][3:6] == ["symbiosis", "export", "--run-dir"]
    assert artifacts.manifest_path.name == "manifest.json"
    assert artifacts.results_path.name == "results.json"


def test_contender_export_falls_back_to_run_on_cache_miss(monkeypatch, tmp_path: Path) -> None:
    calls: list[list[str]] = []

    def fake_run(argv, cwd, text, capture_output):
        calls.append(argv)
        if len(calls) == 1:
            return subprocess.CompletedProcess(argv, 1, stdout="", stderr="benchmark 'iris' missing from baseline cache")
        return subprocess.CompletedProcess(argv, 0, stdout="", stderr="")

    monkeypatch.setattr("subprocess.run", fake_run)
    contenders_root = tmp_path / "EvoNN-Contenders"
    config_path = contenders_root / "configs" / "cached.yaml"
    pack_path = tmp_path / "pack.yaml"
    run_dir = tmp_path / "workspace" / "contenders"
    ensure_contender_export(
        contenders_root=contenders_root,
        config_path=config_path,
        pack_path=pack_path,
        run_dir=run_dir,
    )

    assert calls[0][3] == "materialize"
    assert calls[1][3] == "run"
    assert calls[2][3:6] == ["symbiosis", "export", "--run-dir"]
