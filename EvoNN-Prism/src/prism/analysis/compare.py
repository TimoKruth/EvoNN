"""Analyze multi-run compare summary markdown files for Prism trends."""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path

SUMMARY_COLUMN_COUNT = 12
BUDGET_COLUMN = 0
SEED_COLUMN = 1
BENCHMARKS_COLUMN = 2
PRISM_WINS_COLUMN = 4
TOPOGRAPH_WINS_COLUMN = 6
STRATOGRAPH_WINS_COLUMN = 8
CONTENDERS_WINS_COLUMN = 10
TIES_COLUMN = 11


@dataclass(frozen=True)
class CompareRow:
    budget: int
    seed: int
    benchmarks: int
    prism_wins: int
    topograph_wins: int
    stratograph_wins: int
    contenders_wins: int
    ties: int


def _parse_markdown_table_row(line: str) -> list[str] | None:
    """Return stripped markdown table cells for a data-width row."""
    if not line.startswith("| "):
        return None

    cells = [cell.strip() for cell in line.strip().strip("|").split("|")]
    if len(cells) < SUMMARY_COLUMN_COUNT:
        return None
    if cells[BUDGET_COLUMN] == "Budget" or cells[BUDGET_COLUMN].startswith("---"):
        return None
    return cells


def parse_four_way_summary(path: str | Path) -> CompareRow:
    path = Path(path)
    for line in path.read_text(encoding="utf-8").splitlines():
        cells = _parse_markdown_table_row(line)
        if cells is None:
            continue
        return CompareRow(
            budget=int(cells[BUDGET_COLUMN]),
            seed=int(cells[SEED_COLUMN]),
            benchmarks=int(cells[BENCHMARKS_COLUMN]),
            prism_wins=int(cells[PRISM_WINS_COLUMN]),
            topograph_wins=int(cells[TOPOGRAPH_WINS_COLUMN]),
            stratograph_wins=int(cells[STRATOGRAPH_WINS_COLUMN]),
            contenders_wins=int(cells[CONTENDERS_WINS_COLUMN]),
            ties=int(cells[TIES_COLUMN]),
        )
    raise ValueError(f"No fair-search-budget row found in {path}")


def aggregate_compare_rows(paths: list[str | Path]) -> dict[int, dict[str, float]]:
    rows = [parse_four_way_summary(path) for path in paths]
    by_budget: dict[int, list[CompareRow]] = {}
    for row in rows:
        by_budget.setdefault(row.budget, []).append(row)

    summary: dict[int, dict[str, float]] = {}
    for budget, budget_rows in sorted(by_budget.items()):
        count = len(budget_rows)
        prism_wins_total = sum(row.prism_wins for row in budget_rows)
        summary[budget] = {
            "runs": count,
            "benchmarks": budget_rows[0].benchmarks if budget_rows else 0,
            "prism_wins_total": prism_wins_total,
            "prism_wins_avg": prism_wins_total / count,
            "topograph_wins_avg": sum(row.topograph_wins for row in budget_rows) / count,
            "stratograph_wins_avg": sum(row.stratograph_wins for row in budget_rows) / count,
            "contenders_wins_avg": sum(row.contenders_wins for row in budget_rows) / count,
            "ties_avg": sum(row.ties for row in budget_rows) / count,
        }
    return summary


def render_compare_analysis(paths: list[str | Path]) -> str:
    summary = aggregate_compare_rows(paths)
    budgets = sorted(summary)
    lines = [
        "# Prism Compare Analysis",
        "",
        "## Aggregated Results",
        "",
        "| Budget | Runs | Prism Avg Wins | Topograph Avg Wins | Stratograph Avg Wins | Contenders Avg Wins | Avg Ties |",
        "|---:|---:|---:|---:|---:|---:|---:|",
    ]
    for budget in budgets:
        row = summary[budget]
        lines.append(
            f"| {budget} | {int(row['runs'])} | {row['prism_wins_avg']:.2f} | "
            f"{row['topograph_wins_avg']:.2f} | {row['stratograph_wins_avg']:.2f} | "
            f"{row['contenders_wins_avg']:.2f} | {row['ties_avg']:.2f} |"
        )

    lines.extend([
        "",
        "## Findings",
        "",
        f"- Prism avg wins move from {summary[budgets[0]]['prism_wins_avg']:.2f} at lowest budget to "
        f"{summary[budgets[-1]]['prism_wins_avg']:.2f} at highest budget.",
        f"- Contenders remain dominant across all budgets with avg wins between "
        f"{min(summary[budget]['contenders_wins_avg'] for budget in budgets):.2f} and "
        f"{max(summary[budget]['contenders_wins_avg'] for budget in budgets):.2f}.",
        "- Prism tracks near Topograph/Stratograph band, not contender band.",
        "",
        "## Prism Actions",
        "",
    ])

    low = summary[budgets[0]]["prism_wins_avg"]
    high = summary[budgets[-1]]["prism_wins_avg"]
    if high <= low + 0.75:
        lines.append("- Extra eval budget gives weak Prism return. Prioritize budget allocation and specialist reuse.")
    else:
        lines.append("- Prism responds to higher budget. Next step: improve conversion of extra evals into benchmark wins.")

    lines.append("- Inspect Prism run reports for operator success, inheritance payoff, archive turnover, and family win concentration.")
    lines.append("- Push LM-aware and benchmark-specialist operators where wins still collapse.")
    lines.append("")
    return "\n".join(lines) + "\n"
