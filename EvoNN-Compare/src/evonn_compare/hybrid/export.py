"""Export hybrid results in compare-layer contract format."""

from __future__ import annotations

import json
import platform
import subprocess
import uuid
from datetime import datetime, timezone
from pathlib import Path

from evonn_shared.contracts import ArtifactPaths, BenchmarkEntry, BudgetEnvelope, DeviceInfo, ResultRecord, RunManifest
from evonn_shared.manifests import benchmark_signature, fairness_manifest


def export_hybrid_results(engine, output_dir: Path, pack_name: str) -> Path:
    output_dir.mkdir(parents=True, exist_ok=True)
    run_id = uuid.uuid4().hex[:12]

    report_path = output_dir / "report.md"
    report_path.write_text(_render_report(engine), encoding="utf-8")

    records = [engine.best_records[key] for key in sorted(engine.best_records)]
    manifest = RunManifest(
        schema_version="1.0",
        system="hybrid",
        run_id=run_id,
        run_name=f"hybrid_{pack_name}_s{engine.config.seed}",
        created_at=datetime.now(timezone.utc),
        pack_name=pack_name,
        seed=engine.config.seed,
        benchmarks=[
            BenchmarkEntry(
                benchmark_id=record.benchmark_id,
                task_kind=record.task,
                metric_name=record.metric_name,
                metric_direction=record.metric_direction,
                status=record.status,
            )
            for record in records
        ],
        budget=BudgetEnvelope(
            evaluation_count=engine.total_evaluations,
            epochs_per_candidate=engine.config.epochs,
            wall_clock_seconds=engine.wall_clock_seconds,
            generations=engine.config.generations,
            population_size=engine.config.population_size,
            budget_policy_name="prototype_equal_budget",
        ),
        device=DeviceInfo(
            device_name=platform.processor() or platform.machine(),
            precision_mode="float32",
            framework="mlx",
        ),
        artifacts=ArtifactPaths(
            config_snapshot="config_snapshot.json",
            report_markdown=report_path.name,
        ),
        fairness=fairness_manifest(
            pack_name=pack_name,
            seed=engine.config.seed,
            evaluation_count=engine.total_evaluations,
            budget_policy_name="prototype_equal_budget",
            benchmark_entries=[
                {
                    "benchmark_id": record.benchmark_id,
                    "task_kind": record.task,
                    "metric_name": record.metric_name,
                    "metric_direction": record.metric_direction,
                }
                for record in records
            ],
            data_signature=benchmark_signature(
                pack_name,
                [
                    {
                        "benchmark_id": record.benchmark_id,
                        "task_kind": record.task,
                        "metric_name": record.metric_name,
                        "metric_direction": record.metric_direction,
                    }
                    for record in records
                ],
            ),
            code_version=_code_version(),
        ),
    )

    results = [
        ResultRecord(
            system="hybrid",
            run_id=run_id,
            benchmark_id=record.benchmark_id,
            metric_name=record.metric_name,
            metric_direction=record.metric_direction,
            metric_value=record.metric_value,
            quality=record.metric_value,
            parameter_count=record.parameter_count,
            train_seconds=record.train_seconds,
            architecture_summary=record.architecture_summary,
            genome_id=record.genome_id,
            status=record.status,
            failure_reason=record.failure_reason,
        )
        for record in records
    ]

    (output_dir / "manifest.json").write_text(manifest.model_dump_json(indent=2), encoding="utf-8")
    (output_dir / "results.json").write_text(
        json.dumps([record.model_dump(mode="json") for record in results], indent=2),
        encoding="utf-8",
    )
    return output_dir


def _render_report(engine) -> str:
    lines = [
        "# EvoNN-Compare Hybrid Report",
        "",
        f"- Seed: `{engine.config.seed}`",
        f"- Population: `{engine.config.population_size}`",
        f"- Generations: `{engine.config.generations}`",
        f"- Epochs per candidate: `{engine.config.epochs}`",
        f"- Evaluation count: `{engine.total_evaluations}`",
        "",
        "## Best Results",
        "",
        "| Benchmark | Metric | Value | Loss | Genome |",
        "|---|---|---:|---:|---|",
    ]
    for record in [engine.best_records[key] for key in sorted(engine.best_records)]:
        metric_value = "---" if record.metric_value is None else f"{record.metric_value:.6f}"
        loss = "---" if record.loss == float("inf") else f"{record.loss:.6f}"
        lines.append(
            f"| {record.benchmark_id} | {record.metric_name} | {metric_value} | {loss} | {record.genome_id} |"
        )
    return "\n".join(lines) + "\n"

def _code_version() -> str | None:
    try:
        return subprocess.check_output(
            ["git", "rev-parse", "HEAD"],
            cwd=Path(__file__).resolve().parents[3],
            text=True,
        ).strip()
    except Exception:
        return None
