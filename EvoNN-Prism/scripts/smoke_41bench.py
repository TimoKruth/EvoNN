#!/usr/bin/env python3
"""Smoke test: run Prism on shared benchmark suite with minimal budget."""

import json
import sys
import time
import traceback
from pathlib import Path

import numpy as np
import yaml

sys.path.insert(0, str(Path(__file__).parent.parent / "src"))

import random
from prism.genome import create_seed_genome, apply_random_mutation, crossover
from prism.config import RunConfig
from prism.benchmarks.parity import get_canonical_id
from prism.benchmarks.datasets import get_benchmark
from prism.benchmarks.preprocess import Preprocessor

try:
    from prism.families import compile_genome, compatible_families
    from prism.runtime.training import train_and_evaluate
except ImportError as exc:  # pragma: no cover - host-dependent MLX runtime availability
    compile_genome = None
    compatible_families = None
    train_and_evaluate = None
    _MLX_RUNTIME_IMPORT_ERROR = exc
else:
    _MLX_RUNTIME_IMPORT_ERROR = None

PACK_PATH = Path(__file__).parent.parent.parent / "shared-benchmarks" / "suites" / "parity" / "shared_33plus5.yaml"

POP_SIZE = 4
NUM_GENS = 2
EPOCHS = 2
BATCH_SIZE = 32
LM_BATCH_SIZE = 8
SEED = 42
LM_TRAIN_CAP = 128
LM_VAL_CAP = 32
LM_FAMILIES = ["attention"]


def load_pack(path):
    with open(path, encoding="utf-8") as f:
        pack = yaml.safe_load(f) or {}
    benchmarks = []
    for native in pack.get("benchmarks", []):
        spec = get_benchmark(native)
        metric_name, metric_direction = _metric_contract(spec.task)
        benchmarks.append({
            "canonical_id": get_canonical_id(native),
            "native_id": native,
            "task": spec.task,
            "metric_name": metric_name,
            "metric_direction": metric_direction,
        })
    return benchmarks


def detect_modality(spec):
    """Detect modality from benchmark spec."""
    if hasattr(spec, 'modality') and spec.modality:
        return spec.modality
    name = spec.id if hasattr(spec, 'id') else spec.name
    if any(k in name for k in ['mnist', 'cifar', 'digits', 'image', 'fashion']):
        return "image"
    return "tabular"


def _cap_lm_splits(x_train, y_train, x_val, y_val):
    return (
        x_train[:LM_TRAIN_CAP],
        y_train[:LM_TRAIN_CAP],
        x_val[:LM_VAL_CAP],
        y_val[:LM_VAL_CAP],
    )


def _input_shape_from_data(spec, x_train):
    if getattr(spec, "input_shape", None):
        return list(spec.input_shape)
    if x_train.ndim <= 1:
        return [1]
    return list(x_train.shape[1:])


def _family_pool(modality, task):
    if task == "language_modeling":
        return list(LM_FAMILIES)
    families = compatible_families(modality)
    return families or ["mlp"]


def _metric_contract(task):
    if task == "language_modeling":
        return "perplexity", "min"
    if task == "regression":
        return "mse", "min"
    return "accuracy", "max"


def _require_runtime_dependencies():
    if compile_genome is None or compatible_families is None or train_and_evaluate is None:
        raise RuntimeError("MLX runtime unavailable for smoke_41bench Prism execution") from _MLX_RUNTIME_IMPORT_ERROR


def mini_evolve(benchmark_name, task, seed=SEED):
    """Run minimal family-based evolution on one benchmark."""
    _require_runtime_dependencies()
    spec = get_benchmark(benchmark_name)
    X_train, y_train, X_val, y_val = spec.load_data(seed=seed)
    modality = detect_modality(spec)

    if task == "language_modeling":
        X_train, y_train, X_val, y_val = _cap_lm_splits(X_train, y_train, X_val, y_val)
    else:
        pp = Preprocessor()
        X_train = pp.fit_transform(X_train)
        X_val = pp.transform(X_val)

    input_shape = _input_shape_from_data(spec, X_train)
    if task == "classification":
        num_classes = len(set(y_train.tolist() if hasattr(y_train, 'tolist') else y_train))
    else:
        num_classes = getattr(spec, "output_dim", None) or getattr(spec, "num_classes", None) or 1

    rng = random.Random(seed)
    config = RunConfig()
    batch_size = LM_BATCH_SIZE if task == "language_modeling" else BATCH_SIZE

    # Get compatible families
    families = _family_pool(modality, task)

    # Create seed population - one per family, fill rest with mlp
    population = []
    for fam in families[:POP_SIZE]:
        g = create_seed_genome(fam, config.evolution, rng)
        # Override LR for regression
        if task == "regression":
            g = g.model_copy(update={"learning_rate": 0.001})
        population.append(g)
    while len(population) < POP_SIZE:
        g = create_seed_genome(families[0], config.evolution, rng)
        if task == "regression":
            g = g.model_copy(update={"learning_rate": 0.001})
        population.append(g)

    best_quality = -float("inf")
    best_genome = None
    best_result = None

    for gen in range(NUM_GENS):
        generation_scores = []

        # Evaluate
        for g in population:
            try:
                cm = compile_genome(g, input_shape, num_classes, modality, task=task)
                lr = g.learning_rate
                result = train_and_evaluate(
                    cm.model, X_train, y_train, X_val, y_val,
                    task=task, epochs=EPOCHS, lr=lr, batch_size=batch_size,
                    parameter_count=cm.parameter_count,
                )
                quality = result.quality
                if not np.isfinite(quality):
                    quality = -float("inf")
                    result = None
            except Exception:
                quality = -float("inf")
                result = None

            generation_scores.append((g, quality))
            if quality > best_quality:
                best_quality = quality
                best_genome = g
                best_result = result

        # Last gen: skip reproduction
        if gen == NUM_GENS - 1:
            break

        # Simple reproduction
        scored = sorted(generation_scores, key=lambda item: (item[1], item[0].genome_id), reverse=True)
        elites = [g for g, score in scored[:2] if np.isfinite(score)]
        if not elites:
            elites = sorted(population, key=lambda g: g.genome_id)[:2]
        offspring = []
        for _ in range(POP_SIZE - 2):
            if rng.random() < 0.5 and len(elites) >= 2:
                child = crossover(elites[0], elites[1], rng)
            else:
                parent = rng.choice(elites)
                child, _ = apply_random_mutation(parent, config.evolution, rng)
            offspring.append(child)
        population = elites + offspring

    if best_result is None:
        metric_name, metric_direction = _metric_contract(task)
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

    return {
        "metric_name": best_result.metric_name,
        "metric_direction": "max" if best_result.metric_name == "accuracy" else "min",
        "metric_value": float(best_result.metric_value),
        "quality": float(best_result.quality),
        "native_fitness": float(best_result.quality),
        "train_seconds": float(best_result.train_seconds),
        "failure_reason": best_result.failure_reason,
        "genome": best_genome,
    }


def main() -> int:
    if not PACK_PATH.exists():
        print(f"Pack not found: {PACK_PATH}")
        return 1

    try:
        _require_runtime_dependencies()
    except RuntimeError as exc:
        print(f"Smoke runtime unavailable: {exc}")
        return 1

    benchmarks = load_pack(PACK_PATH)
    print(f"Prism Smoke Test: {len(benchmarks)} benchmarks")
    print(f"Config: pop={POP_SIZE}, gen={NUM_GENS}, epochs={EPOCHS}, seed={SEED}")
    print("=" * 80)

    results = []
    total_start = time.time()

    for i, bench in enumerate(benchmarks, 1):
        name = bench["native_id"]
        canonical = bench["canonical_id"]
        task = bench["task"]

        print(f"[{i:2d}/{len(benchmarks)}] {canonical:<42s} ({name}) ... ", end="", flush=True)
        t0 = time.time()

        try:
            outcome = mini_evolve(name, task)
            elapsed = time.time() - t0
            genome = outcome["genome"]
            status = "ok" if outcome["failure_reason"] is None else "failed"
            family = genome.family if genome else "?"
            layers = len(genome.hidden_layers) if genome else 0
            if status == "ok":
                print(
                    f"{outcome['metric_name']}={outcome['metric_value']:.4f}  "
                    f"quality={outcome['quality']:.4f}  "
                    f"native={outcome['native_fitness']:.4f}  "
                    f"{elapsed:.1f}s  ({family}/{layers}L)"
                )
            else:
                print(f"FAILED ({elapsed:.1f}s): {outcome['failure_reason']}")
        except Exception as e:
            elapsed = time.time() - t0
            status = "failed"
            family = "?"
            layers = 0
            metric_name, metric_direction = _metric_contract(task)
            outcome = {
                "metric_name": metric_name,
                "metric_direction": metric_direction,
                "metric_value": None,
                "quality": None,
                "native_fitness": None,
                "train_seconds": None,
            }
            print(f"FAILED ({elapsed:.1f}s): {e}")
            traceback.print_exc()

        results.append({
            "benchmark_id": canonical,
            "native_id": name,
            "task": task,
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
        })

    total_elapsed = time.time() - total_start

    ok = [r for r in results if r["status"] == "ok"]
    failed = [r for r in results if r["status"] == "failed"]

    print("\n" + "=" * 80)
    print(f"DONE: {len(ok)}/{len(results)} passed, {len(failed)} failed, {total_elapsed:.1f}s total")

    if failed:
        print("\nFailed:")
        for r in failed:
            print(f"  - {r['benchmark_id']} ({r['native_id']})")

    cls_results = [r for r in ok if r["task"] == "classification"]
    reg_results = [r for r in ok if r["task"] == "regression"]
    lm_results = [r for r in ok if r["task"] == "language_modeling"]

    if cls_results:
        accs = [r["metric_value"] for r in cls_results if r["metric_value"] is not None]
        if accs:
            print(f"\nClassification ({len(accs)}): mean_acc={np.mean(accs):.4f}, min={min(accs):.4f}, max={max(accs):.4f}")

    if reg_results:
        mses = [r["metric_value"] for r in reg_results if r["metric_value"] is not None]
        if mses:
            print(f"Regression ({len(mses)}): mean_mse={np.mean(mses):.4f}")

    if lm_results:
        ppls = [r["metric_value"] for r in lm_results if r["metric_value"] is not None]
        if ppls:
            print(f"LM ({len(ppls)}): mean_ppl={np.mean(ppls):.4f}, min={min(ppls):.4f}, max={max(ppls):.4f}")

    # Family distribution
    fam_counts = {}
    for r in ok:
        f = r["family"]
        fam_counts[f] = fam_counts.get(f, 0) + 1
    print(f"\nFamilies: {fam_counts}")

    # Save
    out_dir = Path(__file__).parent.parent / "runs" / "smoke_41bench"
    out_dir.mkdir(parents=True, exist_ok=True)
    out_path = out_dir / "results.json"
    with open(out_path, "w", encoding="utf-8") as f:
        json.dump({
            "system": "prism",
            "version": "0.1.0",
            "pack": PACK_PATH.stem,
            "config": {"population_size": POP_SIZE, "num_generations": NUM_GENS,
                       "epochs": EPOCHS, "batch_size": BATCH_SIZE,
                       "lm_batch_size": LM_BATCH_SIZE, "lm_train_cap": LM_TRAIN_CAP,
                       "lm_val_cap": LM_VAL_CAP, "seed": SEED},
            "total_elapsed_seconds": round(total_elapsed, 2),
            "results": results,
        }, f, indent=2)
    print(f"\nResults: {out_path}")
    return 0 if not failed else 1


if __name__ == "__main__":
    raise SystemExit(main())
