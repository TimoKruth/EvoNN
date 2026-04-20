from pathlib import Path

from evonn_compare.comparison.fair_matrix import build_matrix_summary, summarize_matrix_case
from evonn_compare.comparison.engine import ComparisonEngine
from evonn_compare.contracts.parity import load_parity_pack
from evonn_compare.ingest.loader import SystemIngestor
from evonn_compare.reporting.fair_matrix_md import render_fair_matrix_markdown
from test_compare import PACK_PATH, _write_run


def test_fair_matrix_markdown_splits_fair_and_reference_rows(tmp_path: Path) -> None:
    pack = load_parity_pack(PACK_PATH)
    systems = {
        "prism": tmp_path / "prism",
        "topograph": tmp_path / "topograph",
        "stratograph": tmp_path / "stratograph",
        "primordia": tmp_path / "primordia",
        "contenders": tmp_path / "contenders",
    }
    for system, run_dir in systems.items():
        _write_run(run_dir, system=system)

    ingestors = {system: SystemIngestor(path) for system, path in systems.items()}
    runs = {
        system: (ingestor.load_manifest(), ingestor.load_results())
        for system, ingestor in ingestors.items()
    }
    pair_results = {}
    for left, right in (("prism", "topograph"), ("prism", "stratograph"), ("prism", "primordia"), ("prism", "contenders")):
        result = ComparisonEngine().compare(
            left_manifest=runs[left][0],
            left_results=runs[left][1],
            right_manifest=runs[right][0],
            right_results=runs[right][1],
            pack=pack,
        )
        pair_results[(left, right)] = (result, Path(f"{left}_vs_{right}.md"))

    fair_row, reference_row, parity_rows = summarize_matrix_case(
        pack=pack,
        budget=64,
        seed=42,
        runs=runs,
        pair_results=pair_results,
    )
    summary = build_matrix_summary(
        pack_name=pack.name,
        fair_rows=[fair_row] if fair_row is not None else [],
        reference_rows=[reference_row] if reference_row is not None else [],
        parity_rows=parity_rows,
    )
    markdown = render_fair_matrix_markdown(summary)

    assert "## Fair Search-Budget Results" in markdown
    assert "## Reference Baseline Results" in markdown
    assert "## Parity/Validity Check" in markdown
    assert "| 64 | 42 |" in markdown


def test_fair_matrix_reference_row_for_nonfair_pair(tmp_path: Path) -> None:
    pack = load_parity_pack(PACK_PATH)
    systems = {
        "prism": tmp_path / "prism",
        "topograph": tmp_path / "topograph",
        "stratograph": tmp_path / "stratograph",
        "primordia": tmp_path / "primordia",
        "contenders": tmp_path / "contenders",
    }
    for system, run_dir in systems.items():
        _write_run(run_dir, system=system)
    _write_run(systems["contenders"], system="contenders", budget_policy_name="fixed_contender_pool")

    ingestors = {system: SystemIngestor(path) for system, path in systems.items()}
    runs = {
        system: (ingestor.load_manifest(), ingestor.load_results())
        for system, ingestor in ingestors.items()
    }
    result = ComparisonEngine().compare(
        left_manifest=runs["prism"][0],
        left_results=runs["prism"][1],
        right_manifest=runs["contenders"][0],
        right_results=runs["contenders"][1],
        pack=pack,
    )
    fair_row, reference_row, _parity_rows = summarize_matrix_case(
        pack=pack,
        budget=64,
        seed=42,
        runs=runs,
        pair_results={("prism", "contenders"): (result, Path("prism_vs_contenders.md"))},
    )

    assert fair_row is None
    assert reference_row is not None
    assert "prism/contenders" in reference_row.note
