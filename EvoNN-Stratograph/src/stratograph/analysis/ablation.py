"""Ablation runners to test whether Stratograph hierarchy buys something."""

from __future__ import annotations

from dataclasses import dataclass
import json
from pathlib import Path
from statistics import mean

from stratograph.config import BenchmarkPoolConfig, RunConfig
from stratograph.genome import dict_to_genome
from stratograph.pipeline.coordinator import run_evolution
from stratograph.storage import RunStore


ABLATION_VARIANTS = [
    "flat_macro",
    "two_level_unshared",
    "two_level_shared",
    "two_level_shared_no_clone",
    "two_level_shared_no_motif_bias",
]


@dataclass(frozen=True)
class AblationRun:
    variant: str
    run_dir: Path
    results: list[dict]
    genomes: dict[str, dict]


@dataclass(frozen=True)
class AblationPack:
    name: str
    benchmarks: list[str]
    population_size: int
    generations: int
    description: str


ABALATION_PACKS = {
    "tabular_local": AblationPack(
        name="tabular_local",
        benchmarks=["blobs_f2_c2", "circles", "moons", "breast_cancer", "iris", "wine"],
        population_size=6,
        generations=2,
        description="Cheap local tabular/classification pack",
    ),
    "image_smoke": AblationPack(
        name="image_smoke",
        benchmarks=["digits", "mnist", "fashion_mnist"],
        population_size=4,
        generations=2,
        description="Image smoke pack",
    ),
    "lm_smoke": AblationPack(
        name="lm_smoke",
        benchmarks=["tiny_lm_synthetic", "tinystories_lm_smoke", "wikitext2_lm_smoke"],
        population_size=4,
        generations=2,
        description="Language-modeling smoke pack",
    ),
}


def run_ablation_suite(
    config: RunConfig,
    *,
    workspace: str | Path,
    config_path: str | Path | None = None,
    variants: list[str] | None = None,
) -> Path:
    """Run one pack across variants and write summary report."""
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
        completed.append(_load_ablation_run(run_dir))

    report = _ablation_summary(completed)
    (workspace / "ablation_report.json").write_text(json.dumps(report, indent=2), encoding="utf-8")
    (workspace / "ablation_report.md").write_text(_render_ablation_markdown(report), encoding="utf-8")
    return workspace / "ablation_report.md"


def run_ablation_matrix(
    config: RunConfig,
    *,
    workspace: str | Path,
    config_path: str | Path | None = None,
    variants: list[str] | None = None,
    packs: dict[str, AblationPack] | None = None,
    include_mixed_from_config: bool = True,
) -> Path:
    """Run multiple pack ablations and aggregate one matrix report."""
    workspace = Path(workspace)
    workspace.mkdir(parents=True, exist_ok=True)
    variants = variants or ABLATION_VARIANTS
    packs = dict(packs or ABALATION_PACKS)
    if include_mixed_from_config:
        packs["mixed_38_smoke"] = AblationPack(
            name="mixed_38_smoke",
            benchmarks=config.benchmark_pool.benchmarks,
            population_size=config.evolution.population_size,
            generations=max(2, config.evolution.generations),
            description="Mixed full smoke pack from config",
        )

    pack_reports: dict[str, dict] = {}
    for pack_name, pack in packs.items():
        pack_workspace = workspace / pack_name
        pack_config = config.model_copy(
            update={
                "run_name": f"{config.run_name or 'ablation_matrix'}__{pack_name}",
                "benchmark_pool": BenchmarkPoolConfig(name=pack.name, benchmarks=pack.benchmarks),
                "evolution": config.evolution.model_copy(
                    update={
                        "population_size": pack.population_size,
                        "generations": pack.generations,
                    }
                ),
            }
        )
        report_path = run_ablation_suite(pack_config, workspace=pack_workspace, config_path=config_path, variants=variants)
        pack_reports[pack_name] = json.loads(report_path.with_suffix(".json").read_text(encoding="utf-8"))
        pack_reports[pack_name]["description"] = pack.description
        pack_reports[pack_name]["benchmark_count"] = len(pack.benchmarks)

    matrix_report = _matrix_summary(pack_reports)
    (workspace / "matrix_report.json").write_text(json.dumps(matrix_report, indent=2), encoding="utf-8")
    (workspace / "matrix_report.md").write_text(_render_matrix_markdown(matrix_report), encoding="utf-8")
    return workspace / "matrix_report.md"


def _load_ablation_run(run_dir: Path) -> AblationRun:
    store = RunStore(run_dir / "metrics.duckdb")
    runs = store.load_runs()
    results = store.load_latest_results(runs[0]["run_id"])
    genomes = {row["genome_id"]: row for row in store.load_genomes(runs[0]["run_id"])}
    store.close()
    variant = run_dir.name.split("__")[-1]
    return AblationRun(variant=variant, run_dir=run_dir, results=results, genomes=genomes)


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
            entries.append(
                {
                    "variant": variant,
                    "metric_name": record["metric_name"],
                    "metric_direction": record["metric_direction"],
                    "metric_value": record["metric_value"],
                    "parameter_count": record["parameter_count"],
                    "status": record["status"],
                    "reuse_ratio": None if genome is None else genome.reuse_ratio,
                    "macro_depth": None if genome is None else genome.macro_depth,
                    "avg_cell_depth": None if genome is None else genome.average_cell_depth,
                }
            )
        comparisons.append({"benchmark_name": benchmark_name, "winner": _winner(entries), "entries": entries})

    pairwise = []
    baseline = by_variant.get("two_level_shared", {})
    for other in [variant for variant in by_variant if variant != "two_level_shared"]:
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
        "shared_beats_no_clone": pairwise_map.get("two_level_shared_no_clone", {}).get("wins", 0)
        >= pairwise_map.get("two_level_shared_no_clone", {}).get("losses", 0),
        "shared_beats_no_motif_bias": pairwise_map.get("two_level_shared_no_motif_bias", {}).get("wins", 0)
        >= pairwise_map.get("two_level_shared_no_motif_bias", {}).get("losses", 0),
    }


def _matrix_summary(pack_reports: dict[str, dict]) -> dict:
    variants = next(iter(pack_reports.values()))["variants"] if pack_reports else []
    global_pairwise: dict[str, dict[str, float]] = {}
    for pack_name, report in pack_reports.items():
        for row in report["pairwise"]:
            agg = global_pairwise.setdefault(
                row["right"],
                {"left": "two_level_shared", "right": row["right"], "wins": 0, "losses": 0, "ties": 0},
            )
            agg["wins"] += row["wins"]
            agg["losses"] += row["losses"]
            agg["ties"] += row["ties"]

    return {
        "variants": variants,
        "packs": pack_reports,
        "global_pairwise": list(global_pairwise.values()),
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
        f"- Shared Better Than No-Clone: `{report['shared_beats_no_clone']}`",
        f"- Shared Better Than No-Motif-Bias: `{report['shared_beats_no_motif_bias']}`",
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


def _render_matrix_markdown(report: dict) -> str:
    lines = [
        "# Stratograph Ablation Matrix",
        "",
        "## Global Pairwise",
        "",
        "| Left | Right | Wins | Losses | Ties |",
        "|---|---|---:|---:|---:|",
    ]
    for row in report["global_pairwise"]:
        lines.append(f"| {row['left']} | {row['right']} | {row['wins']} | {row['losses']} | {row['ties']} |")
    lines.extend(
        [
            "",
            "## Packs",
            "",
            "| Pack | Benchmarks | Shared>Unshared | Shared>Flat | Shared>NoClone | Shared>NoMotifBias |",
            "|---|---:|---|---|---|---|",
        ]
    )
    for pack_name, pack in report["packs"].items():
        lines.append(
            f"| {pack_name} | {pack['benchmark_count']} | {pack['sharing_beats_unshared']} | "
            f"{pack['shared_beats_flat']} | {pack['shared_beats_no_clone']} | "
            f"{pack['shared_beats_no_motif_bias']} |"
        )
    lines.append("")
    return "\n".join(lines)
