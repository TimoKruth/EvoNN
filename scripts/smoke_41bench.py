#!/usr/bin/env python3
"""Smoke test: run Topograph on all 41 shared benchmarks with minimal budget."""

import json
import os
import sys
import tempfile
import time
import traceback
from pathlib import Path

import yaml

# Add src to path
sys.path.insert(0, str(Path(__file__).parent.parent / "src"))

from topograph.benchmarks.parity import get_benchmark
from topograph.benchmarks.preprocess import Preprocessor
from topograph.config import RunConfig, EvolutionConfig, TrainingConfig, EarlyStoppingConfig
from topograph.genome import Genome, InnovationCounter
from topograph.nn.compiler import compile_genome, estimate_model_bytes
from topograph.nn.train import train_model
from topograph.operators.mutate import mutate_width, mutate_activation, mutate_add_layer
from topograph.operators.crossover import crossover
import random
import numpy as np


# ── Pack loader ──────────────────────────────────────────────────────────────

PACK_PATH = Path(__file__).parent.parent.parent / "EvoNN-Symbiosis" / "parity_packs" / "generated" / "all_shared.yaml"

def load_pack(path):
    with open(path) as f:
        pack = yaml.safe_load(f)
    benchmarks = []
    for b in pack["benchmarks"]:
        native = b["native_ids"].get("evonn2", b["benchmark_id"])
        benchmarks.append({
            "canonical_id": b["benchmark_id"],
            "native_id": native,
            "task": b["task_kind"],
            "metric_name": b["metric_name"],
            "metric_direction": b["metric_direction"],
        })
    return benchmarks


# ── Mini evolution ───────────────────────────────────────────────────────────

POP_SIZE = 4
NUM_GENS = 2
EPOCHS = 2
BATCH_SIZE = 32
SEED = 42


def mini_evolve(benchmark_name, task, seed=SEED):
    """Run minimal evolution on one benchmark. Returns (best_loss, best_genome, elapsed)."""
    spec = get_benchmark(benchmark_name)
    X_train, y_train, X_val, y_val = spec.load_data(seed=seed)

    pp = Preprocessor()
    X_train = pp.fit_transform(X_train)
    X_val = pp.transform(X_val)

    input_dim = X_train.shape[1]
    if task == "classification":
        num_classes = len(set(y_train.tolist() if hasattr(y_train, 'tolist') else y_train))
    else:
        num_classes = 1

    rng = random.Random(seed)
    ic = InnovationCounter()

    # Create seed population
    population = [Genome.create_seed(ic, rng, num_layers=rng.randint(3, 8)) for _ in range(POP_SIZE)]

    best_loss = float("inf")
    best_genome = None

    for gen in range(NUM_GENS):
        # Evaluate
        fitnesses = []
        for g in population:
            try:
                model = compile_genome(g, input_dim, num_classes, task)
                lr = 0.001 if task == "regression" else 0.01
                loss = train_model(
                    model, X_train, y_train, X_val, y_val,
                    epochs=EPOCHS, lr=lr, batch_size=BATCH_SIZE, task=task,
                )
                if not np.isfinite(loss):
                    loss = 999.0
                g.fitness = loss
                g.model_bytes = estimate_model_bytes(g)
                fitnesses.append(loss)
            except Exception:
                g.fitness = 999.0
                fitnesses.append(999.0)

        # Track best
        for g in population:
            if g.fitness < best_loss:
                best_loss = g.fitness
                best_genome = g

        # Last gen: skip reproduction
        if gen == NUM_GENS - 1:
            break

        # Simple reproduction: keep best 2, breed 2 offspring
        sorted_pop = sorted(population, key=lambda g: g.fitness)
        elites = sorted_pop[:2]
        offspring = []
        for _ in range(POP_SIZE - 2):
            if rng.random() < 0.5 and len(elites) >= 2:
                child = crossover(elites[0], elites[1], rng)
            else:
                parent = rng.choice(elites)
                child = Genome(
                    layers=list(parent.layers),
                    connections=list(parent.connections),
                )
                child.learning_rate = parent.learning_rate
                child.batch_size = parent.batch_size
            # Mutate
            if rng.random() < 0.5:
                child = mutate_width(child, rng)
            if rng.random() < 0.3:
                child = mutate_activation(child, rng)
            if rng.random() < 0.2:
                child = mutate_add_layer(child, ic, rng)
            offspring.append(child)
        population = elites + offspring

    return best_loss, best_genome


def loss_to_metric(loss, task, metric_direction):
    """Convert training loss to the metric expected by the pack."""
    if task == "classification" and metric_direction == "max":
        # loss is cross-entropy; approximate accuracy = 1 - min(loss, 1)
        # More precisely: for well-trained models, accuracy ≈ exp(-loss)
        return max(0.0, min(1.0, 1.0 - loss)) if loss < 5.0 else 0.0
    elif task == "regression" and metric_direction == "min":
        return loss  # MSE directly
    return loss


# ── Main ─────────────────────────────────────────────────────────────────────

def main():
    if not PACK_PATH.exists():
        print(f"Pack not found: {PACK_PATH}")
        sys.exit(1)

    benchmarks = load_pack(PACK_PATH)
    print(f"Topograph Smoke Test: {len(benchmarks)} benchmarks")
    print(f"Config: pop={POP_SIZE}, gen={NUM_GENS}, epochs={EPOCHS}, seed={SEED}")
    print("=" * 80)

    results = []
    total_start = time.time()

    for i, bench in enumerate(benchmarks, 1):
        name = bench["native_id"]
        canonical = bench["canonical_id"]
        task = bench["task"]
        metric = bench["metric_name"]
        direction = bench["metric_direction"]

        print(f"[{i:2d}/{len(benchmarks)}] {canonical:<42s} ({name}) ... ", end="", flush=True)
        t0 = time.time()

        try:
            loss, genome = mini_evolve(name, task)
            elapsed = time.time() - t0
            metric_value = loss_to_metric(loss, task, direction)
            status = "ok"
            layers = len(genome.enabled_layers) if genome else 0
            conns = len(genome.enabled_connections) if genome else 0
            params = genome.model_bytes if genome else 0
            print(f"{metric}={metric_value:.4f}  loss={loss:.4f}  {elapsed:.1f}s  ({layers}L/{conns}C)")
        except Exception as e:
            elapsed = time.time() - t0
            metric_value = None
            status = "failed"
            layers = conns = params = 0
            print(f"FAILED ({elapsed:.1f}s): {e}")
            traceback.print_exc()

        results.append({
            "benchmark_id": canonical,
            "native_id": name,
            "task": task,
            "metric_name": metric,
            "metric_direction": direction,
            "metric_value": metric_value,
            "loss": loss if status == "ok" else None,
            "status": status,
            "elapsed_seconds": round(elapsed, 2),
            "topology": f"{layers}L/{conns}C",
            "model_bytes": params,
        })

    total_elapsed = time.time() - total_start

    # Summary
    ok = [r for r in results if r["status"] == "ok"]
    failed = [r for r in results if r["status"] == "failed"]

    print("\n" + "=" * 80)
    print(f"DONE: {len(ok)}/{len(results)} passed, {len(failed)} failed, {total_elapsed:.1f}s total")

    if failed:
        print(f"\nFailed benchmarks:")
        for r in failed:
            print(f"  - {r['benchmark_id']} ({r['native_id']})")

    # Classification summary
    cls_results = [r for r in ok if r["task"] == "classification"]
    reg_results = [r for r in ok if r["task"] == "regression"]

    if cls_results:
        accs = [r["metric_value"] for r in cls_results]
        print(f"\nClassification ({len(cls_results)}): "
              f"mean_acc={np.mean(accs):.4f}, min={min(accs):.4f}, max={max(accs):.4f}")

    if reg_results:
        mses = [r["metric_value"] for r in reg_results]
        print(f"Regression ({len(reg_results)}): "
              f"mean_mse={np.mean(mses):.4f}, min={min(mses):.4f}, max={max(mses):.4f}")

    # Save results
    out_dir = Path(__file__).parent.parent / "runs" / "smoke_41bench"
    out_dir.mkdir(parents=True, exist_ok=True)
    out_path = out_dir / "results.json"
    with open(out_path, "w") as f:
        json.dump({
            "system": "topograph",
            "version": "0.1.0",
            "pack": "all_shared",
            "config": {
                "population_size": POP_SIZE,
                "num_generations": NUM_GENS,
                "epochs": EPOCHS,
                "batch_size": BATCH_SIZE,
                "seed": SEED,
            },
            "total_elapsed_seconds": round(total_elapsed, 2),
            "results": results,
        }, f, indent=2)
    print(f"\nResults saved: {out_path}")


if __name__ == "__main__":
    main()
