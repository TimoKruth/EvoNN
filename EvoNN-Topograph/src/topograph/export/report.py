"""Markdown report generation and topology analysis for Topograph runs."""

from __future__ import annotations

from collections import Counter, defaultdict, deque
from pathlib import Path
from typing import Any

from topograph.genome.codec import dict_to_genome
from topograph.genome.genome import INPUT_INNOVATION, OUTPUT_INNOVATION, Genome
from topograph.pipeline.archive import BenchmarkEliteArchive
from topograph.storage import RunStore


# ===========================================================================
# Public report generator
# ===========================================================================


def load_report_context(run_dir: str | Path) -> dict[str, Any]:
    """Load the shared Topograph run context used by inspect/report flows."""
    run_dir = Path(run_dir)
    with RunStore(run_dir / "metrics.duckdb") as store:
        run_id = _resolve_run_id(store)
        latest_gen = store.load_latest_generation(run_id)
        if latest_gen is None:
            return {
                "run_dir": run_dir,
                "run_id": run_id,
                "latest_generation": None,
                "run": {},
                "population": [],
                "best_genome": None,
                "budget_meta": {},
                "run_state": {},
                "benchmark_results": [],
                "ok_results": [],
                "failed_results": [],
                "skipped_results": [],
                "best_results": [],
                "benchmark_timings": [],
                "checkpoint_path": run_dir / "checkpoint.json",
            }

        genome_dicts = store.load_genomes(run_id, latest_gen)
        population = [dict_to_genome(d) for d in genome_dicts]
        best_genome = min(
            population,
            key=lambda g: g.fitness if g.fitness is not None else float("inf"),
        ) if population else None

        try:
            run = store.load_run(run_id)
        except ValueError:
            run = {}

        benchmark_results = store.load_benchmark_results(run_id)
        ok_results = [row for row in benchmark_results if row.get("status") == "ok"]
        failed_results = [row for row in benchmark_results if row.get("status") == "failed"]
        skipped_results = [
            row for row in benchmark_results if row.get("status") not in {None, "ok", "failed"}
        ]

        return {
            "run_dir": run_dir,
            "run_id": run_id,
            "latest_generation": latest_gen,
            "run": run,
            "population": population,
            "best_genome": best_genome,
            "budget_meta": store.load_budget_metadata(run_id) or {},
            "run_state": store.load_run_state(run_id) or {},
            "benchmark_results": benchmark_results,
            "ok_results": ok_results,
            "failed_results": failed_results,
            "skipped_results": skipped_results,
            "best_results": store.load_best_benchmark_results(run_id),
            "benchmark_timings": store.load_benchmark_timings(run_id),
            "checkpoint_path": run_dir / "checkpoint.json",
        }


def primordia_seeding_rows(seeding: dict[str, Any] | None) -> list[tuple[str, str]]:
    """Return Topograph seeding metadata rows for inspect/report surfaces."""
    if not seeding:
        return []

    rows = [
        (
            "Primordia Seeding",
            f"{seeding.get('selected_family', 'unknown')} -> {seeding.get('target_family', 'unknown')} "
            f"(rank {seeding.get('selected_rank', 'n/a')})",
        )
    ]
    if seeding.get("seed_path"):
        rows.append(("Primordia Seed Artifact", str(seeding["seed_path"])))
    if seeding.get("representative_genome_id"):
        rows.append(("Primordia Seed Genome", str(seeding["representative_genome_id"])))
    if seeding.get("representative_architecture_summary"):
        rows.append(
            ("Primordia Seed Summary", str(seeding["representative_architecture_summary"]))
        )
    return rows


def generate_report(run_dir: str | Path, output_path: str | Path | None = None) -> str:
    """Generate a markdown report for a completed run. Returns the markdown string."""
    run_dir = Path(run_dir)
    store = RunStore(run_dir / "metrics.duckdb")

    run_id = _resolve_run_id(store)
    latest_gen = store.load_latest_generation(run_id)
    if latest_gen is None:
        store.close()
        return "# Report\n\nNo generations found.\n"

    # Load final population
    genome_dicts = store.load_genomes(run_id, latest_gen)
    population = [dict_to_genome(d) for d in genome_dicts]
    best = min(population, key=lambda g: g.fitness if g.fitness is not None else float("inf"))

    # Load config from DB
    try:
        config_dict = store.load_run(run_id)
    except ValueError:
        config_dict = {}

    budget = store.load_budget_metadata(run_id) or {}
    run_state = store.load_run_state(run_id) or {}
    benchmark_timings = store.load_benchmark_timings(run_id)
    benchmark_results = store.load_benchmark_results(run_id)
    timing_summary = _summarize_benchmark_timings(benchmark_timings)
    benchmark_extremes = _benchmark_quality_extremes(store.load_best_benchmark_results(run_id))
    sampled_orders = _sampled_benchmark_orders(benchmark_timings)
    worst_trend = _worst_benchmark_trend(benchmark_results)
    benchmark_elite_archive = BenchmarkEliteArchive.from_dict(run_state.get("benchmark_elite_archive"))
    atlas_rows = _benchmark_atlas_rows(benchmark_elite_archive)
    atlas_clusters = _atlas_motif_clusters(benchmark_elite_archive)

    lines: list[str] = []

    # --- Run Summary ---
    lines.append("# Topograph Evolution Report\n")
    lines.append("## Run Summary\n")
    lines.append(f"- **Run ID:** {run_dir.name}")
    lines.append(f"- **Seed:** {config_dict.get('seed', 'N/A')}")
    lines.append(f"- **Benchmark:** {config_dict.get('benchmark', 'N/A')}")
    lines.append(f"- **Generations:** {latest_gen + 1}")
    lines.append(f"- **Population Size:** {len(population)}")
    lines.append(f"- **Runtime:** {budget.get('runtime_backend', 'unknown')}")
    lines.append(f"- **Runtime Version:** {budget.get('runtime_version') or 'unknown'}")
    lines.append(f"- **Precision Mode:** {budget.get('precision_mode') or 'unknown'}")
    if budget.get("wall_clock_seconds"):
        lines.append(f"- **Wall Clock:** {budget['wall_clock_seconds']:.1f}s")
    if budget.get("evaluation_count"):
        lines.append(f"- **Evaluations:** {budget['evaluation_count']}")
    if budget.get("evals_per_second") is not None:
        lines.append(f"- **Evaluations / Sec:** {budget['evals_per_second']:.4f}")
    if budget.get("seconds_per_eval") is not None:
        lines.append(f"- **Seconds / Eval:** {budget['seconds_per_eval']:.4f}")
    if budget.get("cache_reuse_rate") is not None:
        lines.append(
            f"- **Cache Reuse Rate:** {float(budget['cache_reuse_rate']):.1%} "
            f"({budget.get('cache_reused_count', 0)} reused / "
            f"{(budget.get('cache_reused_count', 0) + budget.get('cache_trained_count', 0))} total)"
        )
    if budget.get("data_cache_hits") is not None:
        lines.append(
            f"- **Data Cache:** {budget.get('data_cache_hits', 0)} hits / "
            f"{budget.get('data_cache_misses', 0)} misses"
        )
    if budget.get("resolved_parallel_workers_max") is not None:
        lines.append(
            f"- **Parallel Workers:** requested {budget.get('requested_parallel_workers', 1)}, "
            f"max resolved {budget['resolved_parallel_workers_max']}"
        )
    if budget.get("worker_clamp_reason_counts"):
        clamp_counts = ", ".join(
            f"{name}={count}" for name, count in budget["worker_clamp_reason_counts"].items()
        )
        lines.append(f"- **Worker Clamp Reasons:** {clamp_counts}")
    if budget.get("benchmark_elite_families"):
        family_counts = ", ".join(
            f"{family}={count}" for family, count in budget["benchmark_elite_families"].items()
        )
        lines.append(f"- **Atlas Families:** {family_counts}")
    for label, value in primordia_seeding_rows(budget.get("primordia_seeding")):
        lines.append(f"- **{label}:** {value}")
    lines.append("")

    # --- Best Genome ---
    lines.append("## Best Genome\n")
    lines.append(f"- **Fitness:** {best.fitness:.6f}" if best.fitness is not None else "- **Fitness:** N/A")
    lines.append(f"- **Layers:** {len(best.enabled_layers)}")
    lines.append(f"- **Connections:** {len(best.enabled_connections)}")
    lines.append(f"- **Parameters:** {best.param_count}")
    lines.append(f"- **Model Bytes:** {best.model_bytes}")

    summary = dag_summary(best)
    lines.append(f"- **DAG Depth:** {summary['depth']}")
    lines.append(f"- **Skip Connections:** {summary['skip_connections']}")
    lines.append(f"- **Bottleneck Layers:** {summary['bottleneck_count']}")
    lines.append(f"- **Connectivity Ratio:** {summary['connectivity_ratio']:.4f}")
    lines.append("")

    # Layer details table
    lines.append("### Layer Details\n")
    lines.append("| Innovation | Width | Activation | W-Bits | A-Bits | Operator | Order |")
    lines.append("|-----------|-------|-----------|--------|--------|----------|-------|")
    for lg in sorted(best.enabled_layers, key=lambda l: l.order):
        lines.append(
            f"| {lg.innovation} | {lg.width} | {lg.activation.value} "
            f"| {lg.weight_bits.value} | {lg.activation_bits.value} "
            f"| {lg.operator.value} | {lg.order:.2f} |"
        )
    lines.append("")

    # --- Evolution Progress ---
    lines.append("## Evolution Progress\n")
    lines.append("```")
    for g in range(latest_gen + 1):
        gen_dicts = store.load_genomes(run_id, g)
        fitnesses = [d.get("fitness") for d in gen_dicts if d.get("fitness") is not None]
        if fitnesses:
            best_f = min(fitnesses)
            avg_f = sum(fitnesses) / len(fitnesses)
            worst_f = max(fitnesses)
            bar_len = max(1, min(50, int(50 * (1 - best_f / (worst_f + 1e-10)))))
            lines.append(
                f"Gen {g:3d} |{'#' * bar_len:<50}| "
                f"best={best_f:.4f} avg={avg_f:.4f} worst={worst_f:.4f}"
            )
    lines.append("```\n")

    # --- Population Diversity ---
    lines.append("## Population Diversity\n")
    div = population_diversity(population)
    lines.append(f"- **Unique Topologies:** {div['unique_topologies']}")
    lines.append(f"- **Mean Compatibility Distance:** {div['mean_distance']:.4f}")
    lines.append(f"- **Min Distance:** {div['min_distance']:.4f}")
    lines.append(f"- **Max Distance:** {div['max_distance']:.4f}")

    op_dist = div.get("operator_distribution", {})
    if op_dist:
        lines.append("\n### Operator Distribution\n")
        lines.append("| Operator | Count | % |")
        lines.append("|----------|-------|---|")
        total_ops = sum(op_dist.values())
        for op_name, count in sorted(op_dist.items(), key=lambda x: -x[1]):
            pct = count / total_ops * 100 if total_ops else 0
            lines.append(f"| {op_name} | {count} | {pct:.0f}% |")

    depth_dist = div.get("depth_distribution", {})
    if depth_dist:
        lines.append("\n### Depth Distribution\n")
        lines.append("| Depth | Count |")
        lines.append("|-------|-------|")
        for depth, count in sorted(depth_dist.items()):
            lines.append(f"| {depth} | {count} |")
    lines.append("")

    # --- Topology Analysis ---
    lines.append("## Topology Analysis\n")
    wp = dag_width_profile(best)
    lines.append(f"- **Width Profile:** {wp}")
    prec = precision_distribution(best)
    lines.append(f"- **Weight Bits Distribution:** {prec['weight_bits']}")
    lines.append(f"- **Activation Bits Distribution:** {prec['activation_bits']}")
    lines.append("")

    # --- Benchmark Results (if multi-benchmark) ---
    bench_results = store.load_best_benchmark_results(run_id)
    if bench_results:
        lines.append("## Benchmark Results\n")
        lines.append("| Benchmark | Metric | Direction | Value | Status |")
        lines.append("|-----------|--------|-----------|-------|--------|")
        for r in bench_results:
            val = f"{r['metric_value']:.6f}" if r["metric_value"] is not None else "N/A"
            lines.append(
                f"| {r['benchmark_name']} | {r['metric_name']} "
                f"| {r['metric_direction']} | {val} | {r['status']} |"
            )
        lines.append("")

    if timing_summary["rows"]:
        lines.append("## Benchmark Timing\n")
        lines.append(
            "| Benchmark | Total | Load | Eval | Reused | Trained | Failures | "
            "Data Cache | Workers | Clamp |"
        )
        lines.append(
            "|-----------|-------|------|------|--------|---------|----------|"
            "------------|---------|-------|"
        )
        for row in timing_summary["slowest"][:5]:
            lines.append(
                f"| {row['benchmark_name']} | {row['total_seconds']:.2f}s | "
                f"{row['data_load_seconds']:.2f}s | {row['evaluation_seconds']:.2f}s | "
                f"{row['reused_count']} | {row['trained_count']} | {row['failed_count']} | "
                f"{row['data_cache_hits']}/{row['data_cache_misses']} | "
                f"{row['resolved_worker_count']} | {row['worker_clamp_reason']} |"
            )
        lines.append("")
        lines.append("### Fastest Benchmarks\n")
        lines.append("| Benchmark | Total |")
        lines.append("|-----------|-------|")
        for row in timing_summary["fastest"][:5]:
            lines.append(f"| {row['benchmark_name']} | {row['total_seconds']:.2f}s |")
        lines.append("")

    if benchmark_extremes["best"] or benchmark_extremes["worst"]:
        lines.append("## Benchmark Fitness Extremes\n")
        if benchmark_extremes["best"]:
            lines.append("### Best Benchmarks By Quality\n")
            lines.append("| Benchmark | Quality | Metric | Value |")
            lines.append("|-----------|---------|--------|-------|")
            for row in benchmark_extremes["best"]:
                value = "N/A" if row["metric_value"] is None else f"{row['metric_value']:.6f}"
                lines.append(
                    f"| {row['benchmark_name']} | {row['quality']:.6f} | "
                    f"{row['metric_name']} | {value} |"
                )
            lines.append("")
        if benchmark_extremes["worst"]:
            lines.append("### Worst Benchmarks By Quality\n")
            lines.append("| Benchmark | Quality | Metric | Value |")
            lines.append("|-----------|---------|--------|-------|")
            for row in benchmark_extremes["worst"]:
                value = "N/A" if row["metric_value"] is None else f"{row['metric_value']:.6f}"
                lines.append(
                    f"| {row['benchmark_name']} | {row['quality']:.6f} | "
                    f"{row['metric_name']} | {value} |"
                )
            lines.append("")

    if sampled_orders:
        lines.append("## Sampled Benchmark Order\n")
        lines.append("| Generation | Benchmarks |")
        lines.append("|------------|------------|")
        for row in sampled_orders:
            lines.append(f"| {row['generation']} | {', '.join(row['benchmarks'])} |")
        lines.append("")
    family_stage_history = budget.get("family_stage_history") or []
    if family_stage_history:
        lines.append("## Family Stages\n")
        lines.append("| Generation | Active Family | Benchmarks |")
        lines.append("|------------|---------------|------------|")
        for row in family_stage_history:
            lines.append(
                f"| {row['generation']} | {row['active_family']} | "
                f"{', '.join(row.get('sampled_benchmarks', []))} |"
            )
        lines.append("")

    if worst_trend:
        lines.append("## Worst Benchmark Trend\n")
        lines.append("| Generation | Benchmark | Quality | Metric | Value |")
        lines.append("|------------|-----------|---------|--------|-------|")
        for row in worst_trend:
            metric_value = row["metric_value"]
            value = "N/A" if metric_value is None else f"{float(metric_value):.6f}"
            lines.append(
                f"| {row['generation']} | {row['benchmark_name']} | "
                f"{row['quality']:.6f} | {row['metric_name']} | {value} |"
            )
            lines.append("")

    if atlas_rows:
        lines.append("## Topology Atlas\n")
        lines.append("| Benchmark | Family | Generation | Fitness | Params | Bytes | Summary | Motifs |")
        lines.append("|-----------|--------|------------|---------|--------|-------|---------|--------|")
        for row in atlas_rows:
            lines.append(
                f"| {row['benchmark_name']} | {row['benchmark_family']} | {row['generation']} | "
                f"{row['fitness']:.6f} | {row['param_count'] or 'N/A'} | "
                f"{row['model_bytes'] or 'N/A'} | {row['architecture_summary'] or 'N/A'} | "
                f"{', '.join(row['motifs'])} |"
            )
        lines.append("")
    if atlas_clusters:
        lines.append("## Atlas Motif Clusters\n")
        lines.append("| Motif | Count | Benchmarks |")
        lines.append("|-------|-------|------------|")
        for motif, payload in atlas_clusters.items():
            lines.append(
                f"| {motif} | {payload['count']} | {', '.join(payload['benchmarks'])} |"
            )
        lines.append("")

    store.close()

    report = "\n".join(lines)

    if output_path:
        Path(output_path).write_text(report, encoding="utf-8")

    return report


# ===========================================================================
# Topology analysis functions (public, full-featured)
# ===========================================================================


def dag_depth(genome: Genome) -> int:
    """Longest path from INPUT to OUTPUT in the genome DAG."""
    return _longest_path_depths(genome).get(OUTPUT_INNOVATION, 0)


def dag_width_profile(genome: Genome) -> list[int]:
    """Number of nodes at each depth level."""
    depths = _longest_path_depths(genome)
    max_d = max(depths.values(), default=-1)
    if max_d < 0:
        return []
    profile = [0] * (max_d + 1)
    for d in depths.values():
        profile[d] += 1
    return profile


def skip_connection_count(genome: Genome) -> int:
    """Count connections that skip at least one layer."""
    depths = _longest_path_depths(genome)
    count = 0
    for c in genome.enabled_connections:
        sd = depths.get(c.source)
        td = depths.get(c.target)
        if sd is not None and td is not None and td - sd > 1:
            count += 1
    return count


def bottleneck_count(genome: Genome) -> int:
    """Count layers narrower than both neighbors (width-based)."""
    layers = sorted(genome.enabled_layers, key=lambda lg: lg.order)
    if len(layers) < 3:
        return 0
    count = 0
    for i in range(1, len(layers) - 1):
        if layers[i].width < layers[i - 1].width and layers[i].width < layers[i + 1].width:
            count += 1
    return count


def precision_distribution(genome: Genome) -> dict[str, dict[str, int]]:
    """Histogram of weight_bits and activation_bits across layers."""
    wb: dict[str, int] = defaultdict(int)
    ab: dict[str, int] = defaultdict(int)
    sparsity_buckets: dict[str, int] = {
        "0.0-0.1": 0, "0.1-0.3": 0, "0.3-0.5": 0, "0.5+": 0,
    }
    for lg in genome.enabled_layers:
        wb[str(lg.weight_bits.value)] += 1
        ab[str(lg.activation_bits.value)] += 1
        s = lg.sparsity
        if s < 0.1:
            sparsity_buckets["0.0-0.1"] += 1
        elif s < 0.3:
            sparsity_buckets["0.1-0.3"] += 1
        elif s < 0.5:
            sparsity_buckets["0.3-0.5"] += 1
        else:
            sparsity_buckets["0.5+"] += 1
    return {
        "weight_bits": dict(wb),
        "activation_bits": dict(ab),
        "sparsity_buckets": sparsity_buckets,
    }


def dag_summary(genome: Genome) -> dict[str, Any]:
    """Full topology summary combining all analysis functions."""
    depth = dag_depth(genome)
    wp = dag_width_profile(genome)
    skips = skip_connection_count(genome)
    bns = bottleneck_count(genome)
    prec = precision_distribution(genome)

    n_layers = len(genome.enabled_layers)
    n_conns = len(genome.enabled_connections)
    max_possible = n_layers * (n_layers + 1) / 2 if n_layers > 0 else 1
    connectivity = n_conns / max_possible

    return {
        "depth": depth,
        "width_profile": wp,
        "skip_connections": skips,
        "bottleneck_count": bns,
        "precision": prec,
        "total_layers": n_layers,
        "total_connections": n_conns,
        "connectivity_ratio": round(connectivity, 4),
        "has_experts": len(genome.experts) > 0,
    }


def population_diversity(population: list[Genome]) -> dict[str, Any]:
    """Population-level diversity metrics."""
    if not population:
        return {
            "unique_topologies": 0,
            "mean_distance": 0.0,
            "min_distance": 0.0,
            "max_distance": 0.0,
            "operator_distribution": {},
            "depth_distribution": {},
        }

    # Unique topology signatures: (sorted layer innovations, sorted connection (src,tgt))
    signatures: set[tuple] = set()
    for g in population:
        layer_sig = tuple(sorted(lg.innovation for lg in g.enabled_layers))
        conn_sig = tuple(sorted((c.source, c.target) for c in g.enabled_connections))
        signatures.add((layer_sig, conn_sig))

    # Pairwise compatibility distances (lightweight: layer count + connection count diff)
    distances: list[float] = []
    for i in range(len(population)):
        for j in range(i + 1, len(population)):
            d = _compatibility_distance(population[i], population[j])
            distances.append(d)

    mean_dist = sum(distances) / len(distances) if distances else 0.0
    min_dist = min(distances) if distances else 0.0
    max_dist = max(distances) if distances else 0.0

    # Operator distribution
    op_counts: Counter[str] = Counter()
    for g in population:
        for lg in g.enabled_layers:
            op_counts[lg.operator.value] += 1

    # Depth distribution
    depth_counts: dict[int, int] = defaultdict(int)
    for g in population:
        d = dag_depth(g)
        depth_counts[d] += 1

    return {
        "unique_topologies": len(signatures),
        "mean_distance": round(mean_dist, 4),
        "min_distance": round(min_dist, 4),
        "max_distance": round(max_dist, 4),
        "operator_distribution": dict(op_counts),
        "depth_distribution": dict(depth_counts),
    }


def speciation_diagnostics(
    population: list[Genome], species_manager: Any = None,
) -> dict[str, Any]:
    """Speciation health metrics if speciation is enabled."""
    if not population:
        return {
            "num_species": 0,
            "species_sizes": [],
            "inter_species_distance": 0.0,
            "intra_species_distance": 0.0,
            "stagnation_risk": [],
        }

    # Simple clustering by topology signature
    clusters: dict[tuple, list[Genome]] = defaultdict(list)
    for g in population:
        sig = tuple(sorted(lg.innovation for lg in g.enabled_layers))
        clusters[sig].append(g)

    species_sizes = [len(members) for members in clusters.values()]

    # Inter/intra distances
    intra_distances: list[float] = []
    inter_distances: list[float] = []

    cluster_reps = [members[0] for members in clusters.values()]
    for i, rep_i in enumerate(cluster_reps):
        for j in range(i + 1, len(cluster_reps)):
            inter_distances.append(_compatibility_distance(rep_i, cluster_reps[j]))

    for members in clusters.values():
        for i in range(len(members)):
            for j in range(i + 1, len(members)):
                intra_distances.append(_compatibility_distance(members[i], members[j]))

    # Stagnation detection: species where all members have similar fitness
    stagnant: list[int] = []
    for idx, members in enumerate(clusters.values()):
        fitnesses = [g.fitness for g in members if g.fitness is not None]
        if len(fitnesses) >= 2:
            spread = max(fitnesses) - min(fitnesses)
            if spread < 1e-6:
                stagnant.append(idx)

    return {
        "num_species": len(clusters),
        "species_sizes": species_sizes,
        "inter_species_distance": round(
            sum(inter_distances) / len(inter_distances), 4
        ) if inter_distances else 0.0,
        "intra_species_distance": round(
            sum(intra_distances) / len(intra_distances), 4
        ) if intra_distances else 0.0,
        "stagnation_risk": stagnant,
    }


# ===========================================================================
# Private helpers
# ===========================================================================


def _longest_path_depths(genome: Genome) -> dict[int, int]:
    """Longest-path depths from INPUT for all reachable nodes."""
    adj: dict[int, list[int]] = defaultdict(list)
    for c in genome.enabled_connections:
        adj[c.source].append(c.target)
    adj = dict(adj)

    nodes: set[int] = set()
    for src, targets in adj.items():
        nodes.add(src)
        nodes.update(targets)

    if INPUT_INNOVATION not in nodes:
        return {}

    in_degree: dict[int, int] = defaultdict(int)
    for node in nodes:
        in_degree.setdefault(node, 0)
    for targets in adj.values():
        for tgt in targets:
            in_degree[tgt] += 1

    dist: dict[int, int] = {node: -1 for node in nodes}
    dist[INPUT_INNOVATION] = 0

    queue: deque[int] = deque(n for n in nodes if in_degree[n] == 0)
    while queue:
        node = queue.popleft()
        for neighbor in adj.get(node, []):
            if dist[node] >= 0:
                dist[neighbor] = max(dist[neighbor], dist[node] + 1)
            in_degree[neighbor] -= 1
            if in_degree[neighbor] == 0:
                queue.append(neighbor)

    return {n: d for n, d in dist.items() if d >= 0}


def _compatibility_distance(g1: Genome, g2: Genome) -> float:
    """Simplified NEAT compatibility distance between two genomes.

    Counts disjoint + excess layer/connection genes, weighted by c1=1.0, c2=1.0.
    """
    layer_inns_1 = {lg.innovation for lg in g1.layers}
    layer_inns_2 = {lg.innovation for lg in g2.layers}
    conn_inns_1 = {c.innovation for c in g1.connections}
    conn_inns_2 = {c.innovation for c in g2.connections}

    layer_disjoint = len(layer_inns_1.symmetric_difference(layer_inns_2))
    conn_disjoint = len(conn_inns_1.symmetric_difference(conn_inns_2))

    max_genes = max(
        len(layer_inns_1) + len(conn_inns_1),
        len(layer_inns_2) + len(conn_inns_2),
        1,
    )
    return (layer_disjoint + conn_disjoint) / max_genes


def _resolve_run_id(store: RunStore) -> str:
    """Resolve run_id from the store."""
    try:
        store.load_run("current")
        return "current"
    except ValueError:
        pass
    row = store.conn.execute(
        "SELECT run_id FROM runs ORDER BY created_at DESC LIMIT 1"
    ).fetchone()
    if row:
        return row[0]
    return "current"


def _summarize_benchmark_timings(rows: list[dict[str, Any]]) -> dict[str, Any]:
    if not rows:
        return {"rows": [], "fastest": [], "slowest": []}

    by_name: dict[str, dict[str, Any]] = {}
    for row in rows:
        current = by_name.get(row["benchmark_name"])
        if current is None:
            current = {
                "benchmark_name": row["benchmark_name"],
                "task": row["task"],
                "data_load_seconds": 0.0,
                "evaluation_seconds": 0.0,
                "total_seconds": 0.0,
                "trained_count": 0,
                "reused_count": 0,
                "failed_count": 0,
                "data_cache_hits": 0,
                "data_cache_misses": 0,
                "resolved_worker_count": 0,
                "worker_clamp_reasons": Counter(),
            }
            by_name[row["benchmark_name"]] = current
        current["data_load_seconds"] += float(row["data_load_seconds"])
        current["evaluation_seconds"] += float(row["evaluation_seconds"])
        current["total_seconds"] += float(row["total_seconds"])
        current["trained_count"] += int(row["trained_count"])
        current["reused_count"] += int(row["reused_count"])
        current["failed_count"] += int(row["failed_count"])
        current["data_cache_hits"] += int(row.get("data_cache_hits", 0))
        current["data_cache_misses"] += int(row.get("data_cache_misses", 0))
        current["resolved_worker_count"] = max(
            int(current["resolved_worker_count"]),
            int(row["resolved_worker_count"]),
        )
        current["worker_clamp_reasons"][str(row.get("worker_clamp_reason", "sequential"))] += 1

    for current in by_name.values():
        clamp_reason = current["worker_clamp_reasons"].most_common(1)
        current["worker_clamp_reason"] = clamp_reason[0][0] if clamp_reason else "sequential"
        del current["worker_clamp_reasons"]

    summarized = sorted(by_name.values(), key=lambda item: item["total_seconds"])
    return {
        "rows": summarized,
        "fastest": summarized[:5],
        "slowest": list(reversed(summarized[-5:])),
    }


def _benchmark_quality_extremes(rows: list[dict[str, Any]]) -> dict[str, list[dict[str, Any]]]:
    ok_rows = [
        row for row in rows
        if row["status"] == "ok" and row.get("quality") is not None
    ]
    ordered = sorted(ok_rows, key=lambda row: float(row["quality"]), reverse=True)
    return {
        "best": ordered[:5],
        "worst": list(reversed(ordered[-5:])),
    }


def _sampled_benchmark_orders(rows: list[dict[str, Any]]) -> list[dict[str, Any]]:
    by_generation: dict[int, list[tuple[int, str]]] = defaultdict(list)
    for row in rows:
        by_generation[int(row["generation"])].append(
            (int(row["benchmark_order"]), str(row["benchmark_name"]))
        )
    return [
        {
            "generation": generation,
            "benchmarks": [name for _, name in sorted(entries)],
        }
        for generation, entries in sorted(by_generation.items())
    ]


def _worst_benchmark_trend(rows: list[dict[str, Any]]) -> list[dict[str, Any]]:
    by_generation: dict[int, list[dict[str, Any]]] = defaultdict(list)
    for row in rows:
        by_generation[int(row["generation"])].append(row)

    trend: list[dict[str, Any]] = []
    for generation, generation_rows in sorted(by_generation.items()):
        ok_rows = [
            row for row in generation_rows
            if row["status"] == "ok" and row.get("quality") is not None
        ]
        if not ok_rows:
            continue
        worst = min(ok_rows, key=lambda row: float(row["quality"]))
        trend.append(worst)
    return trend


def _benchmark_atlas_rows(
    archive: BenchmarkEliteArchive,
) -> list[dict[str, Any]]:
    rows: list[dict[str, Any]] = []
    for elite in archive.elites.values():
        motifs = ["unknown"]
        if elite.genome is not None:
            motifs = _motif_tags(dict_to_genome(elite.genome))
        rows.append(
            {
                "benchmark_name": elite.benchmark_name,
                "benchmark_family": elite.benchmark_family,
                "generation": elite.generation,
                "fitness": elite.fitness,
                "param_count": elite.param_count,
                "model_bytes": elite.model_bytes,
                "architecture_summary": elite.architecture_summary,
                "motifs": motifs,
            }
        )
    return sorted(rows, key=lambda row: (row["benchmark_family"], row["benchmark_name"]))


def _atlas_motif_clusters(
    archive: BenchmarkEliteArchive,
) -> dict[str, dict[str, Any]]:
    clusters: dict[str, dict[str, Any]] = {}
    for elite in archive.elites.values():
        if elite.genome is None:
            continue
        for motif in _motif_tags(dict_to_genome(elite.genome)):
            current = clusters.setdefault(motif, {"count": 0, "benchmarks": []})
            current["count"] += 1
            current["benchmarks"].append(elite.benchmark_name)
    for payload in clusters.values():
        payload["benchmarks"] = sorted(payload["benchmarks"])
    return dict(sorted(clusters.items()))


def _motif_tags(genome: Genome) -> list[str]:
    summary = dag_summary(genome)
    tags: list[str] = []
    if summary["depth"] >= 4:
        tags.append("deep")
    if summary["skip_connections"] >= 2:
        tags.append("skip_heavy")
    if summary["bottleneck_count"] >= 1:
        tags.append("bottlenecked")
    if summary["connectivity_ratio"] < 0.4:
        tags.append("sparse_connectivity")
    operators = {layer.operator.value for layer in genome.enabled_layers}
    if {"attention_lite", "transformer_lite"} & operators:
        tags.append("attention")
    if len(genome.experts) > 0:
        tags.append("expert_routed")
    if not tags:
        tags.append("dense_baseline")
    return tags
