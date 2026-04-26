#!/usr/bin/env python3
"""Smoke test for 33 shared classification/image benchmarks + 5 LM benchmarks."""

from __future__ import annotations

import argparse
import json
import random
import sys
import time
import traceback
from pathlib import Path
from statistics import mean

import numpy as np

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT / "src") not in sys.path:
    sys.path.insert(0, str(ROOT / "src"))


SHARED_33_BENCHMARKS: list[tuple[str, str]] = [
    ("blobs_classification", "blobs_f2_c2"),
    ("breast_cancer", "breast_cancer"),
    ("circles_classification", "circles"),
    ("credit_g_classification", "credit_g"),
    ("digits_image", "digits"),
    ("fashionmnist_image", "fashion_mnist"),
    ("iris_classification", "iris"),
    ("mnist_image", "mnist"),
    ("moons_classification", "moons"),
    ("openml_adult", "adult"),
    ("openml_bank_marketing", "bank_marketing"),
    ("openml_blood_transfusion", "blood_transfusion"),
    ("openml_electricity", "electricity"),
    ("openml_gas_sensor", "gas_sensor"),
    ("openml_gesture_phase", "gesture_phase"),
    ("openml_heart_disease", "heart_disease"),
    ("openml_ilpd", "ilpd"),
    ("openml_jungle_chess", "jungle_chess"),
    ("openml_kc1", "kc1"),
    ("openml_letter", "letter"),
    ("openml_mfeat_factors", "mfeat_factors"),
    ("openml_nomao", "nomao"),
    ("openml_ozone_level", "ozone_level"),
    ("openml_speed_dating", "speed_dating"),
    ("openml_wall_robot", "wall_robot"),
    ("openml_wilt", "wilt"),
    ("phoneme_classification", "phoneme"),
    ("qsar_biodeg_classification", "qsar_biodeg"),
    ("segment_classification", "segment"),
    ("steel_plates_fault_classification", "steel_plates_fault"),
    ("vehicle_classification", "vehicle"),
    ("wine_classification", "wine"),
    ("xor_tabular", "circles_n02_f3"),
]

LM_5_BENCHMARKS: list[tuple[str, str]] = [
    ("tiny_lm_synthetic", "tiny_lm_synthetic"),
    ("tinystories_lm", "tinystories_lm"),
    ("wikitext2_lm", "wikitext2_lm"),
    ("tinystories_lm_smoke", "tinystories_lm_smoke"),
    ("wikitext2_lm_smoke", "wikitext2_lm_smoke"),
]

BENCHMARKS: list[tuple[str, str]] = SHARED_33_BENCHMARKS + LM_5_BENCHMARKS

SEED = 42
EPOCHS = 1
TABULAR_BATCH_SIZE = 16
LM_BATCH_SIZE = 8


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description="Run capped smoke eval on 33 shared + 5 LM benchmarks.",
    )
    parser.add_argument(
        "--output",
        type=Path,
        default=ROOT / "runs" / "smoke_33plus5" / "results.json",
        help="Path to JSON results file.",
    )
    parser.add_argument(
        "--report",
        type=Path,
        default=None,
        help="Optional markdown report path. Defaults next to --output.",
    )
    parser.add_argument(
        "--limit",
        type=int,
        default=None,
        help="Optional benchmark count limit for quick local checks.",
    )
    parser.add_argument(
        "--dry-run",
        action="store_true",
        help="Print selected benchmarks and exit.",
    )
    return parser


def _subset_data(
    x_train: np.ndarray,
    y_train: np.ndarray,
    x_val: np.ndarray,
    y_val: np.ndarray,
    *,
    task: str,
) -> tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    if task == "language_modeling":
        train_cap = min(128, x_train.shape[0])
        val_cap = min(32, x_val.shape[0])
        return x_train[:train_cap], y_train[:train_cap], x_val[:val_cap], y_val[:val_cap]

    train_cap = min(256, x_train.shape[0])
    val_cap = min(64, x_val.shape[0])
    return x_train[:train_cap], y_train[:train_cap], x_val[:val_cap], y_val[:val_cap]


def _prepare_data(
    benchmark_name: str,
    *,
    seed: int,
) -> tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray, str, int]:
    from topograph.benchmarks.parity import get_benchmark
    from topograph.benchmarks.preprocess import Preprocessor

    spec = get_benchmark(benchmark_name)
    x_train, y_train, x_val, y_val = spec.load_data(seed=seed, validation_split=0.2)
    x_train, y_train, x_val, y_val = _subset_data(
        x_train, y_train, x_val, y_val, task=spec.task,
    )

    if spec.task != "language_modeling":
        pp = Preprocessor()
        x_train = pp.fit_transform(x_train)
        x_val = pp.transform(x_val)

    num_classes = _resolve_output_dim(
        task=spec.task,
        declared=spec.num_classes,
        x_train=x_train,
        y_train=y_train,
    )

    return x_train, y_train, x_val, y_val, spec.task, num_classes


def _resolve_output_dim(
    *,
    task: str,
    declared: int | None,
    x_train: np.ndarray,
    y_train: np.ndarray,
) -> int:
    if task == "regression":
        return 1
    if task != "language_modeling":
        return declared or int(np.max(y_train)) + 1

    observed_max = max(
        int(np.max(x_train)) if x_train.size else 0,
        int(np.max(y_train)) if y_train.size else 0,
    )
    return max(declared or 1, observed_max + 1)


def smoke_eval(benchmark_name: str, *, seed: int = SEED) -> dict:
    from topograph.genome import Genome, InnovationCounter
    from topograph.nn.compiler import compile_genome, estimate_model_bytes
    from topograph.nn.train import train_and_evaluate

    x_train, y_train, x_val, y_val, task, num_classes = _prepare_data(
        benchmark_name, seed=seed,
    )
    rng = random.Random(seed)
    genome = Genome.create_seed(InnovationCounter(), rng, num_layers=4)
    batch_size = LM_BATCH_SIZE if task == "language_modeling" else TABULAR_BATCH_SIZE
    learning_rate = 0.002 if task == "language_modeling" else (0.001 if task == "regression" else 0.01)

    model = compile_genome(
        genome,
        input_dim=x_train.shape[1],
        num_classes=num_classes,
        task=task,
    )
    result = train_and_evaluate(
        model,
        x_train,
        y_train,
        x_val,
        y_val,
        epochs=EPOCHS,
        lr=learning_rate,
        batch_size=batch_size,
        task=task,
        lr_schedule="fixed",
        weight_decay=0.0,
        grad_clip_norm=1.0,
    )

    genome.model_bytes = estimate_model_bytes(genome)
    return {
        "task": task,
        "metric_name": result.metric_name,
        "metric_direction": result.metric_direction,
        "metric_value": (
            None if result.metric_value != result.metric_value else float(result.metric_value)
        ),
        "quality": None if result.quality != result.quality else float(result.quality),
        "native_fitness": (
            None if result.native_fitness != result.native_fitness else float(result.native_fitness)
        ),
        "train_seconds": float(result.train_seconds),
        "failure_reason": result.failure_reason,
        "topology": f"{len(genome.enabled_layers)}L/{len(genome.enabled_connections)}C",
        "model_bytes": genome.model_bytes,
        "subset": {
            "x_train": list(x_train.shape),
            "y_train": list(y_train.shape),
            "x_val": list(x_val.shape),
            "y_val": list(y_val.shape),
        },
    }


def render_markdown_report(payload: dict) -> str:
    results = payload["results"]
    ok = [row for row in results if row["status"] == "ok"]
    failed = [row for row in results if row["status"] != "ok"]
    lm_rows = [row for row in results if row.get("task") == "language_modeling"]
    slowest = sorted(results, key=lambda row: row.get("elapsed_seconds") or 0, reverse=True)[:10]

    lines = [
        "# Topograph Smoke 33+5 Report",
        "",
        "## Summary",
        "",
        f"- Benchmarks: {payload['benchmark_count']}",
        f"- Passed: {len(ok)}",
        f"- Failed: {len(failed)}",
        f"- Total Time: {payload['total_elapsed_seconds']}s",
        "",
        "## Task Breakdown",
        "",
    ]

    task_names = sorted({row.get("task", "unknown") for row in results})
    for task_name in task_names:
        count = sum(1 for row in results if row.get("task") == task_name)
        lines.append(f"- {task_name}: {count}")

    if lm_rows:
        lines.extend([
            "",
            "## Language Modeling",
            "",
            "| Benchmark | Metric | Value | Native Fitness | Elapsed |",
            "|-----------|--------|-------|----------------|---------|",
        ])
        for row in lm_rows:
            lines.append(
                f"| {row['benchmark_id']} | {row.get('metric_name')} | "
                f"{row.get('metric_value')} | {row.get('native_fitness')} | "
                f"{row.get('elapsed_seconds')}s |"
            )

    if failed:
        lines.extend([
            "",
            "## Failures",
            "",
            "| Benchmark | Native ID | Reason |",
            "|-----------|-----------|--------|",
        ])
        for row in failed:
            lines.append(
                f"| {row['benchmark_id']} | {row['native_id']} | {row.get('failure_reason')} |"
            )

    if slowest:
        lines.extend([
            "",
            "## Slowest Benchmarks",
            "",
            "| Benchmark | Native ID | Status | Elapsed |",
            "|-----------|-----------|--------|---------|",
        ])
        for row in slowest:
            lines.append(
                f"| {row['benchmark_id']} | {row['native_id']} | {row['status']} | "
                f"{row.get('elapsed_seconds')}s |"
            )

    cls_rows = [row for row in ok if row.get("task") == "classification" and row.get("metric_value") is not None]
    if cls_rows:
        cls_vals = [float(row["metric_value"]) for row in cls_rows]
        lines.extend([
            "",
            "## Aggregate Metrics",
            "",
            f"- Classification Mean Accuracy: {mean(cls_vals):.4f}",
        ])

    return "\n".join(lines) + "\n"


def main() -> int:
    args = build_parser().parse_args()
    selected = BENCHMARKS[: args.limit] if args.limit else BENCHMARKS
    report_path = args.report or args.output.with_suffix(".md")

    print(f"Smoke set: {len(selected)} benchmarks")
    if args.dry_run:
        for idx, (canonical, native) in enumerate(selected, start=1):
            print(f"[{idx:02d}] {canonical} ({native})")
        return 0

    results: list[dict] = []
    total_start = time.time()

    for idx, (canonical, native) in enumerate(selected, start=1):
        print(f"[{idx:02d}/{len(selected)}] {canonical} ({native}) ... ", end="", flush=True)
        t0 = time.time()
        try:
            outcome = smoke_eval(native, seed=SEED)
            elapsed = time.time() - t0
            status = "ok" if outcome["failure_reason"] is None else "failed"
            print(
                f"{outcome['metric_name']}={outcome['metric_value']} "
                f"native={outcome['native_fitness']} {elapsed:.1f}s "
                f"({outcome['topology']})"
            )
        except Exception as exc:
            elapsed = time.time() - t0
            status = "failed"
            outcome = {
                "task": "unknown",
                "metric_name": None,
                "metric_direction": None,
                "metric_value": None,
                "quality": None,
                "native_fitness": None,
                "train_seconds": None,
                "failure_reason": f"{type(exc).__name__}: {exc}",
                "topology": None,
                "model_bytes": None,
                "subset": None,
            }
            print(f"FAILED {elapsed:.1f}s")
            traceback.print_exc()

        results.append(
            {
                "benchmark_id": canonical,
                "native_id": native,
                "status": status,
                "elapsed_seconds": round(elapsed, 2),
                **outcome,
            }
        )

    total_elapsed = time.time() - total_start
    ok = [row for row in results if row["status"] == "ok"]
    failed = [row for row in results if row["status"] != "ok"]

    print("\n" + "=" * 80)
    print(
        f"DONE: {len(ok)}/{len(results)} passed, {len(failed)} failed, "
        f"{total_elapsed:.1f}s total"
    )

    args.output.parent.mkdir(parents=True, exist_ok=True)
    payload = {
        "system": "topograph",
        "smoke_type": "33plus5",
        "benchmark_count": len(selected),
        "seed": SEED,
        "epochs": EPOCHS,
        "results": results,
        "total_elapsed_seconds": round(total_elapsed, 2),
    }
    args.output.write_text(json.dumps(payload, indent=2), encoding="utf-8")
    report_path.write_text(render_markdown_report(payload), encoding="utf-8")
    print(f"Results saved: {args.output}")
    print(f"Report saved: {report_path}")
    return 0 if not failed else 1


if __name__ == "__main__":
    raise SystemExit(main())
