"""Ablation runner to test whether two-level hierarchy buys something."""

from __future__ import annotations

from dataclasses import dataclass
import json
from pathlib import Path
from statistics import mean

from stratograph.config import RunConfig
from stratograph.genome import dict_to_genome
from stratograph.pipeline.coordinator import run_evolution
from stratograph.storage import RunStore


ABLATION_VARIANTS = ["flat_macro", "two_level_unshared", "two_level_shared"]


@dataclass(frozen=True)
class AblationRun:
    variant: str
    run_dir: Path
    results: list[dict]
    genomes: dict[str, dict]


def run_ablation_suite(
    config: RunConfig,
    *,
    workspace: str | Path,
    config_path: str | Path | None = None,
    variants: list[str] | None = None,
) -> Path:
    """Run ablation variants and write summary report."""
    workspace = Path(workspace)
    runs_dir = workspace / "runs"
    runs_dir.mkdir(parents=True, exist_ok=True)
    variants = variants or ABLATION_VARIANTS
    suite_name = config.run_name or "ablation"
    completed: list[AblationRun] = []

    for variant in variants:
        variant_config = config.model_copy(
            update={
                "run_name": f"{suite_name}__{variant}",
                "evolution": config.evolution.model_copy(update={"architecture_mode": variant}),
            }
        )
        run_dir = runs_dir / variant_config.run_name
        run_evolution(variant_config, run_dir=run_dir, config_path=config_path, resume=False)
        store = RunStore(run_dir / "metrics.duckdb")
        runs = store.load_runs()
        results = store.load_results(runs[0]["run_id"])
        genomes = {row["genome_id"]: row for row in store.load_genomes(runs[0]["run_id"])}
        store.close()
        completed.append(AblationRun(variant=variant, run_dir=run_dir, results=results, genomes=genomes))

    report = _ablation_summary(completed)
    (workspace / "ablation_report.json").write_text(json.dumps(report, indent=2), encoding="utf-8")
    (workspace / "ablation_report.md").write_text(_render_ablation_markdown(report), encoding="utf-8")
    return workspace / "ablation_report.md"


def _ablation_summary(runs: list[AblationRun]) -> dict:
    by_variant = {run.variant: {record["benchmark_name"]: record for record in run.results} for run in runs}
    genome_rows = {run.variant: run.genomes for run in runs}
    benchmark_names = sorted({name for variant in by_variant.values() for name in variant})
    comparisons = []
    for benchmark_name in benchmark_names:
        entries = []
        for variant, results in by_variant.items():
            record = results.get(benchmark_name)
            if record is None:
                continue
            genome_row = genome_rows.get(variant, {}).get(record["genome_id"])
            genome = None if genome_row is None else dict_to_genome(genome_row["payload"])
            entries.append({
                "variant": variant,
                "metric_name": record["metric_name"],
                "metric_direction": record["metric_direction"],
                "metric_value": record["metric_value"],
                "parameter_count": record["parameter_count"],
                "status": record["status"],
                "reuse_ratio": None if genome is None else genome.reuse_ratio,
                "macro_depth": None if genome is None else genome.macro_depth,
                "avg_cell_depth": None if genome is None else genome.average_cell_depth,
            })
        winner = _winner(entries)
        comparisons.append({"benchmark_name": benchmark_name, "winner": winner, "entries": entries})

    pairwise = []
    baseline = by_variant.get("two_level_shared", {})
    for other in ("flat_macro", "two_level_unshared"):
        wins = 0
        losses = 0
        deltas = []
        param_deltas = []
        efficiency_wins = 0
        for benchmark_name, shared_record in baseline.items():
            other_record = by_variant.get(other, {}).get(benchmark_name)
            if other_record is None or shared_record["status"] != "ok" or other_record["status"] != "ok":
                continue
            delta = _delta(shared_record, other_record)
            deltas.append(delta)
            param_deltas.append(float(other_record["parameter_count"] - shared_record["parameter_count"]))
            if delta > 0:
                wins += 1
                if shared_record["parameter_count"] <= other_record["parameter_count"]:
                    efficiency_wins += 1
            elif delta < 0:
                losses += 1
        pairwise.append(
            {
                "left": "two_level_shared",
                "right": other,
                "wins": wins,
                "losses": losses,
                "ties": max(0, len(deltas) - wins - losses),
                "mean_delta": mean(deltas) if deltas else 0.0,
                "mean_parameter_saving": mean(param_deltas) if param_deltas else 0.0,
                "efficiency_wins": efficiency_wins,
            }
        )
    pairwise_map = {row["right"]: row for row in pairwise}

    per_variant = []
    for variant, results in by_variant.items():
        ok_results = [record for record in results.values() if record["status"] == "ok"]
        genomes = []
        for record in ok_results:
            genome_row = genome_rows.get(variant, {}).get(record["genome_id"])
            if genome_row is None:
                continue
            genomes.append(dict_to_genome(genome_row["payload"]))
        per_variant.append(
            {
                "variant": variant,
                "ok_benchmarks": len(ok_results),
                "mean_parameter_count": mean([record["parameter_count"] for record in ok_results]) if ok_results else 0.0,
                "mean_reuse_ratio": mean([genome.reuse_ratio for genome in genomes]) if genomes else 0.0,
                "mean_macro_depth": mean([genome.macro_depth for genome in genomes]) if genomes else 0.0,
                "mean_cell_depth": mean([genome.average_cell_depth for genome in genomes]) if genomes else 0.0,
            }
        )

    return {
        "variants": [run.variant for run in runs],
        "runs": [{"variant": run.variant, "run_dir": str(run.run_dir)} for run in runs],
        "per_variant": per_variant,
        "pairwise": pairwise,
        "comparisons": comparisons,
        "two_level_buys_something": all(row["wins"] >= row["losses"] for row in pairwise),
        "sharing_beats_unshared": pairwise_map.get("two_level_unshared", {}).get("wins", 0)
        >= pairwise_map.get("two_level_unshared", {}).get("losses", 0),
        "shared_beats_flat": pairwise_map.get("flat_macro", {}).get("wins", 0)
        >= pairwise_map.get("flat_macro", {}).get("losses", 0),
    }


def _winner(entries: list[dict]) -> str | None:
    ok_entries = [entry for entry in entries if entry["status"] == "ok" and entry["metric_value"] is not None]
    if not ok_entries:
        return None
    direction = ok_entries[0]["metric_direction"]
    ranked = sorted(ok_entries, key=lambda item: item["metric_value"], reverse=(direction == "max"))
    return ranked[0]["variant"]


def _delta(left: dict, right: dict) -> float:
    direction = left["metric_direction"]
    if direction == "max":
        return float(left["metric_value"] - right["metric_value"])
    return float(right["metric_value"] - left["metric_value"])


def _render_ablation_markdown(report: dict) -> str:
    lines = [
        "# Stratograph Ablation Report",
        "",
        f"- Two-Level Shared Wins Overall: `{report['two_level_buys_something']}`",
        f"- Shared Better Than Unshared: `{report['sharing_beats_unshared']}`",
        f"- Shared Better Than Flat: `{report['shared_beats_flat']}`",
        "",
        "## Variant Summary",
        "",
        "| Variant | OK Benchmarks | Mean Params | Mean Reuse | Mean Macro Depth | Mean Cell Depth |",
        "|---|---:|---:|---:|---:|---:|",
    ]
    for row in report["per_variant"]:
        lines.append(
            f"| {row['variant']} | {row['ok_benchmarks']} | {row['mean_parameter_count']:.1f} | "
            f"{row['mean_reuse_ratio']:.4f} | {row['mean_macro_depth']:.2f} | {row['mean_cell_depth']:.2f} |"
        )
    lines.extend(
        [
            "",
            "## Pairwise",
            "",
            "| Left | Right | Wins | Losses | Ties | Mean Delta | Mean Param Saving | Efficiency Wins |",
            "|---|---|---:|---:|---:|---:|---:|---:|",
        ]
    )
    for row in report["pairwise"]:
        lines.append(
            f"| {row['left']} | {row['right']} | {row['wins']} | {row['losses']} | "
            f"{row['ties']} | {row['mean_delta']:.6f} | {row['mean_parameter_saving']:.1f} | "
            f"{row['efficiency_wins']} |"
        )
    lines.extend(
        [
            "",
            "## Benchmark Winners",
            "",
            "| Benchmark | Winner | Variants |",
            "|---|---|---|",
        ]
    )
    for row in report["comparisons"]:
        variant_text = ", ".join(
            (
                f"{entry['variant']}={entry['metric_value']}"
                f" params={entry['parameter_count']}"
                f" reuse={entry['reuse_ratio']:.2f}"
            )
            if entry["metric_value"] is not None and entry["reuse_ratio"] is not None
            else f"{entry['variant']}={entry['status']}"
            for entry in row["entries"]
        )
        lines.append(f"| {row['benchmark_name']} | {row['winner'] or 'none'} | {variant_text} |")
    lines.append("")
    return "\n".join(lines)
