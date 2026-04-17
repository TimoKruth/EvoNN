#!/usr/bin/env python3
"""Smoke test for 33 proven Prism benchmarks plus 5 language-modeling benchmarks."""

from __future__ import annotations

import json
import random
import sys
import time
import traceback
from pathlib import Path

import numpy as np
import yaml

sys.path.insert(0, str(Path(__file__).parent.parent / "src"))

from prism.benchmarks.datasets import get_benchmark
from prism.benchmarks.parity import get_canonical_id
from prism.benchmarks.preprocess import Preprocessor
from prism.config import RunConfig
from prism.families import compile_genome, compatible_families
from prism.genome import apply_random_mutation, create_seed_genome, crossover
from prism.runtime.training import train_and_evaluate

PACK_PATH = Path(__file__).parent.parent / "configs" / "working_33_plus_5_lm_smoke.yaml"

POP_SIZE = 2
NUM_GENS = 1
EPOCHS = 1
BATCH_SIZE = 32
LM_BATCH_SIZE = 8
SEED = 42
LM_TRAIN_CAP = 512
LM_VAL_CAP = 128
LM_FAMILIES = ["attention"]


def load_pack(path: Path) -> list[str]:
    with open(path, encoding="utf-8") as f:
        payload = yaml.safe_load(f) or {}
    return list(payload.get("benchmarks", []))


def _detect_modality(spec) -> str:
    if hasattr(spec, "modality") and spec.modality:
        return spec.modality
    return "tabular"


def _cap_lm_splits(
    x_train: np.ndarray,
    y_train: np.ndarray,
    x_val: np.ndarray,
    y_val: np.ndarray,
) -> tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    return (
        x_train[:LM_TRAIN_CAP],
        y_train[:LM_TRAIN_CAP],
        x_val[:LM_VAL_CAP],
        y_val[:LM_VAL_CAP],
    )


def _input_shape_from_data(spec, x_train: np.ndarray) -> list[int]:
    if getattr(spec, "input_shape", None):
        return list(spec.input_shape)
    if x_train.ndim <= 1:
        return [1]
    return list(x_train.shape[1:])


def _family_pool(modality: str, task: str) -> list[str]:
    if task == "language_modeling":
        return list(LM_FAMILIES)
    families = compatible_families(modality)
    return families or ["mlp"]


def mini_evolve(benchmark_name: str, seed: int = SEED) -> dict:
    """Run minimal evolution on one benchmark."""
    spec = get_benchmark(benchmark_name)
    task = spec.task
    modality = _detect_modality(spec)
    x_train, y_train, x_val, y_val = spec.load_data(seed=seed)

    if task == "language_modeling":
        x_train, y_train, x_val, y_val = _cap_lm_splits(x_train, y_train, x_val, y_val)
    else:
        pp = Preprocessor()
        x_train = pp.fit_transform(x_train)
        x_val = pp.transform(x_val)

    input_shape = _input_shape_from_data(spec, x_train)
    output_dim = spec.output_dim or spec.num_classes or 1
    if task == "classification":
        output_dim = len(set(y_train.tolist() if hasattr(y_train, "tolist") else y_train))

    rng = random.Random(seed)
    config = RunConfig()
    families = _family_pool(modality, task)

    population = []
    for family in families[:POP_SIZE]:
        genome = create_seed_genome(family, config.evolution, rng)
        if task == "regression":
            genome = genome.model_copy(update={"learning_rate": 0.001})
        population.append(genome)
    while len(population) < POP_SIZE:
        genome = create_seed_genome(families[0], config.evolution, rng)
        if task == "regression":
            genome = genome.model_copy(update={"learning_rate": 0.001})
        population.append(genome)

    best_quality = -float("inf")
    best_genome = None
    best_result = None
    batch_size = LM_BATCH_SIZE if task == "language_modeling" else BATCH_SIZE

    for gen in range(NUM_GENS):
        generation_scores = []

        for genome in population:
            try:
                compiled = compile_genome(genome, input_shape, output_dim, modality, task=task)
                result = train_and_evaluate(
                    compiled.model,
                    x_train,
                    y_train,
                    x_val,
                    y_val,
                    task=task,
                    epochs=EPOCHS,
                    lr=genome.learning_rate,
                    batch_size=batch_size,
                    parameter_count=compiled.parameter_count,
                )
                quality = result.quality
                if not np.isfinite(quality):
                    quality = -float("inf")
                    result = None
            except Exception:
                quality = -float("inf")
                result = None

            generation_scores.append((genome, quality))
            if quality > best_quality:
                best_quality = quality
                best_genome = genome
                best_result = result

        if gen == NUM_GENS - 1:
            break

        scored = sorted(generation_scores, key=lambda item: (item[1], item[0].genome_id), reverse=True)
        elites = [genome for genome, score in scored[:2] if np.isfinite(score)]
        if not elites:
            elites = sorted(population, key=lambda genome: genome.genome_id)[:2]

        offspring = []
        for _ in range(POP_SIZE - len(elites)):
            if rng.random() < 0.5 and len(elites) >= 2:
                child = crossover(elites[0], elites[1], rng)
            else:
                parent = rng.choice(elites)
                child, _ = apply_random_mutation(parent, config.evolution, rng)
            offspring.append(child)
        population = elites + offspring

    if best_result is None:
        metric_name = "accuracy"
        metric_direction = "max"
        if task == "regression":
            metric_name = "mse"
            metric_direction = "min"
        elif task == "language_modeling":
            metric_name = "perplexity"
            metric_direction = "min"
        return {
            "metric_name": metric_name,
            "metric_direction": metric_direction,
            "metric_value": None,
            "quality": None,
            "native_fitness": None,
            "train_seconds": None,
            "failure_reason": "no_valid_result",
            "genome": best_genome,
        }

    metric_direction = "max"
    if best_result.metric_name in {"mse", "perplexity"}:
        metric_direction = "min"

    return {
        "metric_name": best_result.metric_name,
        "metric_direction": metric_direction,
        "metric_value": float(best_result.metric_value),
        "quality": float(best_result.quality),
        "native_fitness": float(best_result.quality),
        "train_seconds": float(best_result.train_seconds),
        "failure_reason": best_result.failure_reason,
        "genome": best_genome,
    }


def main() -> None:
    if not PACK_PATH.exists():
        print(f"Pack not found: {PACK_PATH}")
        sys.exit(1)

    benchmarks = load_pack(PACK_PATH)
    print(f"Prism Smoke Test 33+5 LM: {len(benchmarks)} benchmarks")
    print(
        f"Config: pop={POP_SIZE}, gen={NUM_GENS}, epochs={EPOCHS}, "
        f"batch={BATCH_SIZE}, lm_batch={LM_BATCH_SIZE}, seed={SEED}"
    )
    print(f"LM caps: train={LM_TRAIN_CAP}, val={LM_VAL_CAP}")
    print("=" * 80)

    results = []
    total_start = time.time()

    for idx, name in enumerate(benchmarks, 1):
        spec = get_benchmark(name)
        canonical = get_canonical_id(name)
        print(f"[{idx:2d}/{len(benchmarks)}] {canonical:<42s} ({name}) ... ", end="", flush=True)
        t0 = time.time()

        try:
            outcome = mini_evolve(name)
            elapsed = time.time() - t0
            genome = outcome["genome"]
            status = "ok" if outcome["failure_reason"] is None else "failed"
            family = genome.family if genome else "?"
            layers = len(genome.hidden_layers) if genome else 0
            if status == "ok":
                print(
                    f"{outcome['metric_name']}={outcome['metric_value']:.4f}  "
                    f"quality={outcome['quality']:.4f}  "
                    f"{elapsed:.1f}s  ({family}/{layers}L)"
                )
            else:
                print(f"FAILED ({elapsed:.1f}s): {outcome['failure_reason']}")
        except Exception as exc:
            elapsed = time.time() - t0
            status = "failed"
            family = "?"
            layers = 0
            outcome = {
                "metric_name": "perplexity" if spec.task == "language_modeling" else ("mse" if spec.task == "regression" else "accuracy"),
                "metric_direction": "min" if spec.task in {"regression", "language_modeling"} else "max",
                "metric_value": None,
                "quality": None,
                "native_fitness": None,
                "train_seconds": None,
                "failure_reason": str(exc),
            }
            print(f"FAILED ({elapsed:.1f}s): {exc}")
            traceback.print_exc()

        results.append(
            {
                "benchmark_id": canonical,
                "native_id": name,
                "task": spec.task,
                "metric_name": outcome["metric_name"],
                "metric_direction": outcome["metric_direction"],
                "metric_value": outcome["metric_value"],
                "quality": outcome["quality"],
                "native_fitness": outcome["native_fitness"],
                "status": status,
                "elapsed_seconds": round(elapsed, 2),
                "family": family,
                "layers": layers,
                "failure_reason": outcome.get("failure_reason"),
            }
        )

    total_elapsed = time.time() - total_start
    ok = [row for row in results if row["status"] == "ok"]
    failed = [row for row in results if row["status"] == "failed"]

    print("\n" + "=" * 80)
    print(f"DONE: {len(ok)}/{len(results)} passed, {len(failed)} failed, {total_elapsed:.1f}s total")

    if failed:
        print("\nFailed:")
        for row in failed:
            print(f"  - {row['benchmark_id']} ({row['native_id']})")

    cls_rows = [row for row in ok if row["task"] == "classification" and row["metric_value"] is not None]
    lm_rows = [row for row in ok if row["task"] == "language_modeling" and row["metric_value"] is not None]

    if cls_rows:
        accs = [row["metric_value"] for row in cls_rows]
        print(f"\nClassification ({len(accs)}): mean_acc={np.mean(accs):.4f}, min={min(accs):.4f}, max={max(accs):.4f}")
    if lm_rows:
        ppls = [row["metric_value"] for row in lm_rows]
        print(f"LM ({len(ppls)}): mean_ppl={np.mean(ppls):.4f}, min={min(ppls):.4f}, max={max(ppls):.4f}")

    family_counts: dict[str, int] = {}
    for row in ok:
        family = row["family"]
        family_counts[family] = family_counts.get(family, 0) + 1
    print(f"\nFamilies: {family_counts}")

    out_dir = Path(__file__).parent.parent / "runs" / "smoke_33_plus_5_lm"
    out_dir.mkdir(parents=True, exist_ok=True)
    out_path = out_dir / "results.json"
    with open(out_path, "w", encoding="utf-8") as f:
        json.dump(
            {
                "system": "prism",
                "version": "0.1.0",
                "pack": PACK_PATH.stem,
                "config": {
                    "population_size": POP_SIZE,
                    "num_generations": NUM_GENS,
                    "epochs": EPOCHS,
                    "batch_size": BATCH_SIZE,
                    "lm_batch_size": LM_BATCH_SIZE,
                    "lm_train_cap": LM_TRAIN_CAP,
                    "lm_val_cap": LM_VAL_CAP,
                    "seed": SEED,
                },
                "total_elapsed_seconds": round(total_elapsed, 2),
                "results": results,
            },
            f,
            indent=2,
        )
    print(f"\nResults: {out_path}")


if __name__ == "__main__":
    main()
