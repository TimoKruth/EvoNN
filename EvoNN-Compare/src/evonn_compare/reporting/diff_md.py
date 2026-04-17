"""Side-by-side architecture diff renderer for one benchmark."""
from __future__ import annotations


def _fmt(value: object) -> str:
    if value is None:
        return "n/a"
    if isinstance(value, float):
        return f"{value:.4f}" if abs(value) < 100 else f"{value:.1f}"
    return str(value)


def render_diff_markdown(
    benchmark_id: str,
    evonn_info: dict,
    evonn2_info: dict,
    *,
    evonn_deep: dict | None = None,
    evonn2_deep: dict | None = None,
) -> str:
    lines: list[str] = []
    lines.append(f"# Diff: {benchmark_id}")
    lines.append("")
    lines.append("| Attribute | EvoNN | EvoNN-2 |")
    lines.append("|-----------|-------|---------|")
    rows = [
        ("Metric Value", evonn_info.get("metric_value"), evonn2_info.get("metric_value")),
        ("Parameters", evonn_info.get("parameter_count"), evonn2_info.get("parameter_count")),
        ("Train Time (s)", evonn_info.get("train_seconds"), evonn2_info.get("train_seconds")),
        ("Architecture", evonn_info.get("architecture_summary"), evonn2_info.get("architecture_summary")),
        ("Genome ID", evonn_info.get("genome_id"), evonn2_info.get("genome_id")),
    ]
    for label, left, right in rows:
        lines.append(f"| {label} | {_fmt(left)} | {_fmt(right)} |")

    lv = evonn_info.get("metric_value")
    rv = evonn2_info.get("metric_value")
    if lv is not None and rv is not None:
        delta = lv - rv
        word = "wins" if delta > 0 else "loses" if delta < 0 else "ties"
        lines.append("")
        lines.append(f"**Delta:** {delta:+.4f} (EvoNN {word})")

    if evonn_deep or evonn2_deep:
        lines.append("")
        lines.append("## Deep Context")
        if evonn_deep and evonn_deep.get("lineage"):
            lines.append("")
            lines.append("### EvoNN Lineage")
            for entry in evonn_deep["lineage"]:
                lines.append(f"- gen {entry.get('generation', '?')}: {entry.get('mutation_summary', 'unknown')} from {entry.get('parent_id', '?')}")
        if evonn2_deep and evonn2_deep.get("layer_genes"):
            lines.append("")
            lines.append("### EvoNN-2 Topology")
            for gene in evonn2_deep["layer_genes"]:
                lines.append(f"- layer {gene.get('innovation_number', '?')}: width={gene.get('width', '?')}, activation={gene.get('activation', '?')}")

    lines.append("")
    return "\n".join(lines)
