"""Report rendering for prototype runs."""

from __future__ import annotations

from pathlib import Path

from stratograph.storage import RunStore


def write_report(run_dir: str | Path) -> Path:
    """Render prototype markdown report from run DB."""
    run_dir = Path(run_dir)
    store = RunStore(run_dir / "metrics.duckdb")
    runs = store.load_runs()
    if not runs:
        store.close()
        raise ValueError(f"No runs found in {run_dir}")
    run = runs[0]
    results = store.load_results(run["run_id"])
    genomes = store.load_genomes(run["run_id"])
    budget_meta = store.load_budget_metadata(run["run_id"])
    store.close()

    skipped = sum(1 for record in results if record["status"] == "skipped")
    ok = sum(1 for record in results if record["status"] == "ok")
    failed = sum(1 for record in results if record["status"] == "failed")
    lines = [
        "# Stratograph Prototype Report",
        "",
        f"- Run ID: `{run['run_id']}`",
        f"- Seed: `{run['seed']}`",
        f"- Runtime: `{budget_meta.get('runtime_backend', 'unknown')}`",
        f"- Runtime Version: `{budget_meta.get('runtime_version') or 'unknown'}`",
        f"- Architecture Mode: `{budget_meta.get('architecture_mode', 'unknown')}`",
        f"- Benchmarks: `{len(results)}`",
        f"- Genomes Stored: `{len(genomes)}`",
        f"- Effective Training Epochs: `{budget_meta.get('effective_training_epochs', 'unknown')}`",
        f"- Status Mix: ok={ok}, skipped={skipped}, failed={failed}",
        f"- Novelty Mean: `{budget_meta.get('novelty_score_mean', 0.0):.4f}`",
        f"- Occupied Niches: `{budget_meta.get('map_elites_occupied_niches', 0)}`",
        "",
        "## Benchmarks",
        "",
        "| Benchmark | Metric | Direction | Status | Notes |",
        "|---|---|---|---|---|",
    ]
    for record in results:
        lines.append(
            f"| {record['benchmark_name']} | {record['metric_name']} | "
            f"{record['metric_direction']} | {record['status']} | "
            f"{record['failure_reason'] or record['architecture_summary'] or ''} |"
        )
    path = run_dir / "report.md"
    path.write_text("\n".join(lines) + "\n", encoding="utf-8")
    return path
