"""Transfer/seeding artifact helpers for Primordia runs."""
from __future__ import annotations

import json
from pathlib import Path
from typing import Any

from evonn_primordia.export.report import build_primitive_bank_summary, load_best_results


def build_seed_candidates(
    *,
    summary: dict[str, Any],
    best_results: list[dict[str, Any]],
    trial_records: list[dict[str, Any]],
    primitive_bank: dict[str, Any] | None = None,
) -> dict[str, Any]:
    """Build a benchmark-conditioned seed manifest for later EvoNN systems."""
    primitive_bank = primitive_bank or build_primitive_bank_summary(
        summary=summary,
        best_results=best_results,
        trial_records=trial_records,
    )

    family_groups: dict[str, set[str]] = {}
    for record in trial_records:
        family = str(record.get("primitive_family") or "unknown")
        group = str(record.get("benchmark_group") or "unknown")
        family_groups.setdefault(family, set()).add(group)

    best_rows: list[dict[str, Any]] = []
    for record in best_results:
        if record.get("status") != "ok":
            continue
        family = str(record.get("primitive_family") or "unknown")
        best_rows.append(
            {
                "benchmark_name": str(record.get("benchmark_name") or "unknown"),
                "benchmark_group": str(record.get("benchmark_group") or "unknown"),
                "family": family,
                "genome_id": record.get("genome_id"),
                "architecture_summary": record.get("architecture_summary"),
                "metric_name": record.get("metric_name"),
                "metric_value": record.get("metric_value"),
                "runtime": record.get("runtime") or summary.get("runtime"),
                "runtime_version": record.get("runtime_version") or summary.get("runtime_version"),
            }
        )

    ranked_families = []
    for index, row in enumerate(
        sorted(
            primitive_bank.get("primitive_families") or [],
            key=lambda item: (
                -int(item.get("benchmark_wins", 0)),
                -int(item.get("evaluation_count", 0)),
                str(item.get("family", "unknown")),
            ),
        ),
        start=1,
    ):
        family = str(row.get("family") or "unknown")
        ranked_families.append(
            {
                "seed_rank": index,
                "family": family,
                "benchmark_groups": sorted(family_groups.get(family, {"unknown"})),
                "evaluation_count": int(row.get("evaluation_count", 0)),
                "benchmark_wins": int(row.get("benchmark_wins", 0)),
                "benchmarks_won": list(row.get("benchmarks_won") or []),
                "representative_genome_id": row.get("representative_genome_id"),
                "representative_architecture_summary": row.get("representative_architecture_summary"),
                "best_metric_name": row.get("best_metric_name"),
                "best_metric_value": row.get("best_metric_value"),
            }
        )

    return {
        "system": "primordia",
        "run_id": summary.get("run_id"),
        "run_name": summary.get("run_name"),
        "runtime": summary.get("runtime"),
        "runtime_version": summary.get("runtime_version"),
        "seed_candidates": ranked_families,
        "benchmark_seeds": best_rows,
    }


def write_seed_candidates(run_dir: str | Path) -> Path:
    """Write benchmark-conditioned seed candidates from a Primordia run."""
    run_dir = Path(run_dir)
    summary = json.loads((run_dir / "summary.json").read_text(encoding="utf-8"))
    trial_records = json.loads((run_dir / "trial_records.json").read_text(encoding="utf-8"))
    best_results = load_best_results(run_dir, summary)
    primitive_bank_path = run_dir / "primitive_bank_summary.json"
    primitive_bank = (
        json.loads(primitive_bank_path.read_text(encoding="utf-8"))
        if primitive_bank_path.exists()
        else None
    )
    payload = build_seed_candidates(
        summary=summary,
        best_results=best_results,
        trial_records=trial_records,
        primitive_bank=primitive_bank,
    )
    output_path = run_dir / "seed_candidates.json"
    output_path.write_text(json.dumps(payload, indent=2), encoding="utf-8")
    return output_path
