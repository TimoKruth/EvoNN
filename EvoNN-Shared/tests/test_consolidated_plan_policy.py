from __future__ import annotations

import os
from pathlib import Path


ROOT = Path(__file__).resolve().parents[2]
CONSOLIDATED_PLAN = ROOT / "EVONN_CONSOLIDATED_PLAN.md"
HARD_REMAINDER_PLAN = ROOT / "EVONN_HARD_REMAINDER_PLAN.md"
THIS_TEST = Path(__file__).resolve()

OBSOLETE_ROOT_PLANS = {
    "ROADMAP.md",
    "EVONN_90_DAY_PLAN.md",
    "SEARCH_ENGINE_OUTPUT_PARITY_PLAN.md",
    "EXPANDED_BENCHMARK_COMPARISON_PLAN.md",
    "SHARED_SUBSTRATE_FOUNDATION_PLAN.md",
    "BENCHMARK_EXTRACTION_PLAN.md",
    "CONTENDER_EXPANSION_PLAN.md",
    "SEEDING_LADDERS_IMPLEMENTATION_PLAN.md",
}
OBSOLETE_PACKAGE_PLAN = "IMPLEMENTATION_PLAN.md"
OBSOLETE_PLAN_REF_MARKERS = OBSOLETE_ROOT_PLANS | {
    ".hermes/plans",
    ".hermes\\plans",
    OBSOLETE_PACKAGE_PLAN,
}
OBSOLETE_PLAN_REF_MARKER_BYTES = {
    marker.encode("ascii") for marker in OBSOLETE_PLAN_REF_MARKERS
}
IGNORED_DIRS = {
    ".git",
    ".mypy_cache",
    ".pytest_cache",
    ".ruff_cache",
    ".tox",
    ".venv",
    "__pycache__",
    "deprecated",
    "htmlcov",
    "node_modules",
}
TEXT_SUFFIXES = {".md", ".py", ".sh", ".toml", ".yaml", ".yml"}


def _repo_files() -> list[Path]:
    paths: list[Path] = []
    for dirpath, dirnames, filenames in os.walk(ROOT):
        dirnames[:] = [name for name in dirnames if name not in IGNORED_DIRS]
        base = Path(dirpath)
        for filename in filenames:
            paths.append(base / filename)
    return paths


def _text_files() -> list[Path]:
    return [path for path in _repo_files() if path.suffix in TEXT_SUFFIXES]


def test_consolidated_plan_is_the_only_active_execution_plan() -> None:
    assert CONSOLIDATED_PLAN.exists()
    assert HARD_REMAINDER_PLAN.exists()

    disallowed_paths: list[str] = []
    for path in _repo_files():
        relative = path.relative_to(ROOT)
        if (
            len(relative.parts) == 1
            and path.name.endswith("PLAN.md")
            and path not in {CONSOLIDATED_PLAN, HARD_REMAINDER_PLAN}
        ):
            disallowed_paths.append(str(relative))
        if len(relative.parts) == 1 and path.name in OBSOLETE_ROOT_PLANS:
            disallowed_paths.append(str(relative))
        if path.name == OBSOLETE_PACKAGE_PLAN:
            disallowed_paths.append(str(relative))
        if ".hermes" in relative.parts and "plans" in relative.parts:
            disallowed_paths.append(str(relative))

    assert disallowed_paths == []


def test_obsolete_plan_references_stay_inside_consolidated_plan_history() -> None:
    offenders: list[str] = []
    for path in _text_files():
        if path in {CONSOLIDATED_PLAN, THIS_TEST}:
            continue
        content = path.read_bytes()
        for marker in OBSOLETE_PLAN_REF_MARKER_BYTES:
            if marker in content:
                offenders.append(
                    f"{path.relative_to(ROOT)}: {marker.decode('ascii')}"
                )

    assert offenders == []


def test_entrypoint_docs_point_to_the_consolidated_plan() -> None:
    for relative in ("README.md", "MONOREPO.md", "VISION.md"):
        text = (ROOT / relative).read_text(encoding="utf-8")
        assert "EVONN_CONSOLIDATED_PLAN.md" in text

    plan_text = CONSOLIDATED_PLAN.read_text(encoding="utf-8")
    assert "Latest Execution Record" in plan_text
    assert "against the actual CLI" in plan_text
    assert "lane presets, pack definitions, and benchmark-audit" in plan_text
    assert "this file is the active execution" in plan_text
    assert "`EVONN_HARD_REMAINDER_PLAN.md` is the only companion backlog" in plan_text


def test_hard_remainder_plan_is_non_executing_backlog() -> None:
    text = HARD_REMAINDER_PLAN.read_text(encoding="utf-8")
    assert "`EVONN_CONSOLIDATED_PLAN.md` remains the active operating plan." in text
    assert "hard-remainder backlog" in text
