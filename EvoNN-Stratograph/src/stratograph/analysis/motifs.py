"""Analyze repeated winning cell motifs from completed runs."""

from __future__ import annotations

from collections import defaultdict
import json
from pathlib import Path

from stratograph.genome import dict_to_genome
from stratograph.storage import RunStore


def analyze_run_motifs(run_dir: str | Path) -> Path:
    """Mine repeated sub-cell motifs from best genomes in a run."""
    run_dir = Path(run_dir)
    store = RunStore(run_dir / "metrics.duckdb")
    runs = store.load_runs()
    if not runs:
        store.close()
        raise ValueError(f"No runs found in {run_dir}")
    run_id = runs[0]["run_id"]
    results = [record for record in store.load_results(run_id) if record["status"] == "ok"]
    genomes = {record["genome_id"]: record for record in store.load_genomes(run_id) if record["genome_id"]}
    store.close()

    motif_counts: dict[str, int] = defaultdict(int)
    motif_benchmarks: dict[str, set[str]] = defaultdict(set)
    motif_examples: dict[str, dict] = {}
    repeated_within_genome: list[dict] = []

    for result in results:
        genome_row = genomes.get(result["genome_id"])
        if genome_row is None:
            continue
        genome = dict_to_genome(genome_row["payload"])
        local_counts: dict[str, int] = defaultdict(int)
        used_cell_ids = {node.cell_id for node in genome.macro_nodes}
        for cell_id in used_cell_ids:
            cell = genome.cell_library[cell_id]
            sig = cell_signature(cell)
            motif_counts[sig] += 1
            local_counts[sig] += 1
            motif_benchmarks[sig].add(result["benchmark_name"])
            motif_examples.setdefault(
                sig,
                {
                    "cell_id": cell.cell_id,
                    "nodes": [node.kind.value for node in cell.nodes],
                    "activations": [node.activation.value for node in cell.nodes],
                    "edge_count": len(cell.edges),
                    "width": cell.output_width,
                },
            )
        for sig, count in local_counts.items():
            if count > 1:
                repeated_within_genome.append(
                    {
                        "benchmark_name": result["benchmark_name"],
                        "signature": sig,
                        "count": count,
                    }
                )

    payload = {
        "total_unique_motifs": len(motif_counts),
        "top_motifs": [
            {
                "signature": sig,
                "count": count,
                "benchmarks": sorted(motif_benchmarks[sig]),
                "example": motif_examples[sig],
            }
            for sig, count in sorted(motif_counts.items(), key=lambda item: (-item[1], item[0]))[:20]
        ],
        "repeated_within_genome": repeated_within_genome,
    }
    json_path = run_dir / "motifs_report.json"
    md_path = run_dir / "motifs_report.md"
    json_path.write_text(json.dumps(payload, indent=2), encoding="utf-8")
    md_path.write_text(_render_motif_markdown(payload), encoding="utf-8")
    return md_path


def cell_signature(cell) -> str:
    """Canonical signature for a cell motif."""
    node_sig = ",".join(f"{node.kind.value}:{node.activation.value}:{node.width}" for node in cell.nodes)
    edge_sig = ",".join(
        f"{edge.source}->{edge.target}" for edge in sorted(cell.edges, key=lambda item: (item.source, item.target))
    )
    return f"{node_sig}|{edge_sig}"


def _render_motif_markdown(payload: dict) -> str:
    lines = [
        "# Stratograph Motif Report",
        "",
        f"- Unique Motifs: `{payload['total_unique_motifs']}`",
        "",
        "## Top Motifs",
        "",
        "| Count | Benchmarks | Nodes | Signature |",
        "|---:|---|---|---|",
    ]
    for row in payload["top_motifs"]:
        lines.append(
            f"| {row['count']} | {', '.join(row['benchmarks'])} | "
            f"{', '.join(row['example']['nodes'])} | `{row['signature']}` |"
        )
    if payload["repeated_within_genome"]:
        lines.extend(["", "## Repeated Within Winner", "", "| Benchmark | Count | Signature |", "|---|---:|---|"])
        for row in payload["repeated_within_genome"]:
            lines.append(f"| {row['benchmark_name']} | {row['count']} | `{row['signature']}` |")
    lines.append("")
    return "\n".join(lines)
