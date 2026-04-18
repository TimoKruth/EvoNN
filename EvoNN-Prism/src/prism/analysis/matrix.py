"""Aggregate actual matrix runs and compare summaries into one Prism diagnosis."""

from __future__ import annotations

from collections import Counter
from pathlib import Path

from prism.analysis.compare import render_compare_analysis
from prism.export.report import (
    _compute_archive_turnover,
    _compute_failure_patterns,
    _compute_family_benchmark_wins,
    _compute_inheritance_summary,
    _compute_operator_mix,
)
from prism.genome import ModelGenome
from prism.storage import RunStore


def discover_matrix_artifacts(matrix_root: str | Path) -> tuple[list[Path], list[Path]]:
    matrix_root = Path(matrix_root)
    summary_paths = sorted(matrix_root.glob("reports/*/four_way_summary.md"))
    run_dirs = sorted(path for path in (matrix_root / "runs" / "prism").glob("*") if path.is_dir())
    return summary_paths, run_dirs


def render_matrix_analysis(matrix_root: str | Path) -> str:
    matrix_root = Path(matrix_root)
    summary_paths, run_dirs = discover_matrix_artifacts(matrix_root)
    compare_text = render_compare_analysis(summary_paths) if summary_paths else "# Prism Compare Analysis\n\nNo compare summaries found.\n"
    run_metrics = _aggregate_prism_runs(run_dirs)

    lines = [compare_text.rstrip(), "", "## Run Diagnostics", ""]
    if not run_metrics:
        lines.append("No Prism runs found.")
        lines.append("")
        return "\n".join(lines)

    lines.extend([
        "| Budget | Runs | Top Families | Top Operators | Inheritance Hit Rate | Avg Specialists | Avg New Archive Members | Top Failure |",
        "|---:|---:|---|---|---:|---:|---:|---|",
    ])
    for budget, payload in sorted(run_metrics.items()):
        top_families = ", ".join(f"{family}({count})" for family, count in payload["family_wins"].most_common(3)) or "---"
        top_operators = ", ".join(f"{operator}({count})" for operator, count in payload["operator_mix"].most_common(3)) or "---"
        top_failure = payload["failures"].most_common(1)[0][0] if payload["failures"] else "---"
        lines.append(
            f"| {budget} | {payload['runs']} | {top_families} | {top_operators} | "
            f"{payload['inheritance_rate']:.1f}% | {payload['specialist_count']:.1f} | "
            f"{payload['archive_turnover']:.1f} | {top_failure} |"
        )

    lines.extend([
        "",
        "## Runtime Actions",
        "",
    ])

    highest_budget = max(run_metrics)
    highest = run_metrics[highest_budget]
    dominant_families = [family for family, _ in highest["family_wins"].most_common(2)]
    if dominant_families:
        lines.append(f"- Highest-budget Prism wins mainly come from `{', '.join(dominant_families)}`. Strengthen weak families or narrow search priors.")
    if highest["inheritance_rate"] < 25.0:
        lines.append("- Inheritance hit rate low. Improve parent compatibility and warm-start reuse.")
    if highest["operator_mix"]:
        lines.append("- Feed operator success into stronger adaptive weighting; current operator mix still broad.")
    if highest["specialist_count"]:
        lines.append("- Specialist archives active. Next step: retain and transfer them across runs, not only within one run.")
    if str(highest["failures"].most_common(1)[0][0] if highest["failures"] else "").startswith("compile_error:"):
        lines.append("- Compile errors still top waste source. Tighten family parameter sanitization and compile guards.")
    lines.append("")
    return "\n".join(lines) + "\n"


def _aggregate_prism_runs(run_dirs: list[Path]) -> dict[int, dict]:
    by_budget: dict[int, dict] = {}
    for run_dir in run_dirs:
        db_path = run_dir / "metrics.duckdb"
        if not db_path.exists():
            continue
        store = RunStore(db_path)
        try:
            run_id = _resolve_run_id(store)
            evaluations = store.load_evaluations(run_id)
            genomes = _load_genomes(store, run_id)
            lineage = store.load_lineage(run_id)
            archives = store.load_archives(run_id)
            best_per_benchmark = store.load_best_per_benchmark(run_id)
        finally:
            store.close()

        budget = len(evaluations)
        bucket = by_budget.setdefault(
            budget,
            {
                "runs": 0,
                "family_wins": Counter(),
                "operator_mix": Counter(),
                "failures": Counter(),
                "inheritance_rates": [],
                "specialist_counts": [],
                "archive_turnovers": [],
            },
        )
        bucket["runs"] += 1
        bucket["family_wins"].update(_compute_family_benchmark_wins(best_per_benchmark, genomes))
        bucket["operator_mix"].update(_compute_operator_mix(lineage))
        bucket["failures"].update(_compute_failure_patterns(evaluations))
        inheritance = _compute_inheritance_summary(evaluations)
        if inheritance:
            bucket["inheritance_rates"].append(float(inheritance["rate"]))
        specialist_summary = _specialist_count(archives)
        if specialist_summary:
            bucket["specialist_counts"].append(specialist_summary)
        turnover = _archive_turnover_count(archives)
        if turnover is not None:
            bucket["archive_turnovers"].append(turnover)

    for payload in by_budget.values():
        rates = payload.pop("inheritance_rates")
        payload["inheritance_rate"] = sum(rates) / len(rates) if rates else 0.0
        specialists = payload["specialist_counts"]
        payload["specialist_count"] = sum(specialists) / len(specialists) if specialists else 0.0
        turnovers = payload.pop("archive_turnovers")
        payload["archive_turnover"] = sum(turnovers) / len(turnovers) if turnovers else 0.0
    return by_budget


def _load_genomes(store: RunStore, run_id: str) -> list[ModelGenome]:
    genomes: list[ModelGenome] = []
    for row in store.load_genomes(run_id):
        try:
            genomes.append(ModelGenome.model_validate(row))
        except Exception:
            continue
    return genomes


def _specialist_count(archives: list[dict]) -> int:
    return sum(1 for row in archives if str(row.get("archive_kind", "")).startswith("specialist:"))


def _archive_turnover_count(archives: list[dict]) -> float | None:
    turnover_rows = _compute_archive_turnover(archives)
    if not turnover_rows:
        return None
    return sum(float(row["new_members"]) for row in turnover_rows) / len(turnover_rows)


def _resolve_run_id(store: RunStore) -> str:
    row = store.conn.execute(
        "SELECT run_id FROM runs ORDER BY created_at DESC LIMIT 1"
    ).fetchone()
    if row:
        return row[0]
    return "default"
