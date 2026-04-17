"""Minimal hybrid evolution engine for family-aware topology search."""

from __future__ import annotations

import json
import math
import random
import time
from dataclasses import dataclass
from pathlib import Path

import mlx.core as mx
import mlx.nn as nn
import numpy as np
from topograph.nn.train import train_and_evaluate

from evonn_compare.hybrid.compiler import compile_hybrid
from evonn_compare.hybrid.genome import HybridGenome
from evonn_compare.hybrid.mutation import mutate


class HybridConfig:
    """Configuration for a hybrid evolution run."""

    def __init__(
        self,
        seed: int = 42,
        population_size: int = 8,
        generations: int = 3,
        epochs: int = 20,
        elite_count: int = 2,
    ) -> None:
        self.seed = seed
        self.population_size = population_size
        self.generations = generations
        self.epochs = epochs
        self.elite_count = elite_count


@dataclass(frozen=True)
class HybridEvaluation:
    benchmark_id: str
    task: str
    genome_index: int
    loss: float
    metric_name: str
    metric_direction: str
    metric_value: float | None
    train_seconds: float
    architecture_summary: str
    genome_id: str
    status: str
    parameter_count: int | None = None
    failure_reason: str | None = None


class HybridEngine:
    """Evolution engine for hybrid genomes."""

    def __init__(self, config: HybridConfig, run_dir: Path | None = None) -> None:
        self.config = config
        self.run_dir = run_dir
        self.population: list[HybridGenome] = []
        self.innovation_counter: int = 0
        self.best_records: dict[str, HybridEvaluation] = {}
        self.generation_history: list[dict[str, float]] = []
        self.wall_clock_seconds: float = 0.0
        self.total_evaluations: int = 0
        self.rng = random.Random(config.seed)
        self._start_generation: int = 0
        self._current_generation: int = 0

        if run_dir is not None:
            self._init_run_dir()

    def _init_run_dir(self) -> None:
        assert self.run_dir is not None
        self.run_dir.mkdir(parents=True, exist_ok=True)
        (self.run_dir / "checkpoints").mkdir(parents=True, exist_ok=True)
        (self.run_dir / "config_snapshot.json").write_text(
            json.dumps(
                {
                    "seed": self.config.seed,
                    "population_size": self.config.population_size,
                    "generations": self.config.generations,
                    "epochs": self.config.epochs,
                    "elite_count": self.config.elite_count,
                },
                indent=2,
            ),
            encoding="utf-8",
        )

    def initialize(self) -> None:
        for _ in range(self.config.population_size):
            genome, self.innovation_counter = HybridGenome.create_seed(
                self.innovation_counter,
                width=self.rng.choice([32, 64, 128]),
            )
            self.population.append(genome)

    def evaluate_population(self, benchmarks: dict[str, tuple]) -> dict[str, list[HybridEvaluation]]:
        results: dict[str, list[HybridEvaluation]] = {}
        for benchmark_id, (X_train, y_train, X_val, y_val, task, num_classes) in benchmarks.items():
            evaluations: list[HybridEvaluation] = []
            for genome_index, genome in enumerate(self.population):
                start = time.perf_counter()
                try:
                    model = compile_hybrid(
                        genome,
                        input_dim=int(X_train.shape[1]),
                        num_classes=num_classes,
                        task=task,
                    )
                    parameter_count = _parameter_count(model)
                    if task == "language_modeling":
                        metric_name, metric_direction, metric_value, loss, failure_reason = _train_and_eval_lm(
                            model=model,
                            X_train=np.asarray(X_train),
                            y_train_last=np.asarray(y_train),
                            X_val=np.asarray(X_val),
                            y_val_full=np.asarray(y_val),
                            epochs=self.config.epochs,
                            lr=genome.learning_rate,
                            batch_size=genome.batch_size,
                        )
                        status = "ok" if failure_reason is None else "failed"
                    else:
                        evaluation = train_and_evaluate(
                            model,
                            np.asarray(X_train),
                            np.asarray(y_train),
                            np.asarray(X_val),
                            np.asarray(y_val),
                            task=task,
                            epochs=self.config.epochs,
                            lr=genome.learning_rate,
                            batch_size=genome.batch_size,
                            parameter_count=parameter_count,
                        )
                        metric_name = evaluation.metric_name
                        metric_direction = "min" if task == "regression" else "max"
                        loss = -evaluation.quality if task == "classification" else evaluation.metric_value
                        status = "ok" if evaluation.failure_reason is None else "failed"
                        failure_reason = evaluation.failure_reason
                        metric_value = evaluation.metric_value if status == "ok" else None
                except Exception as exc:
                    loss = float("inf")
                    metric_value = None
                    status = "failed"
                    failure_reason = f"{type(exc).__name__}:{exc}"
                    metric_name = "perplexity" if task == "language_modeling" else ("mse" if task == "regression" else "accuracy")
                    metric_direction = "min" if task in {"regression", "language_modeling"} else "max"
                    parameter_count = None

                evaluations.append(
                    HybridEvaluation(
                        benchmark_id=benchmark_id,
                        task=task,
                        genome_index=genome_index,
                        loss=float(loss),
                        metric_name=metric_name,
                        metric_direction=metric_direction,
                        metric_value=metric_value,
                        train_seconds=round(time.perf_counter() - start, 4),
                        architecture_summary=_architecture_summary(genome),
                        genome_id=f"g{genome_index}",
                        status=status,
                        parameter_count=parameter_count,
                        failure_reason=failure_reason,
                    )
                )
            results[benchmark_id] = evaluations
        self.total_evaluations += len(self.population) * len(benchmarks)
        return results

    def update_best_records(self, results: dict[str, list[HybridEvaluation]]) -> None:
        for benchmark_id, evaluations in results.items():
            benchmark_best = min(evaluations, key=lambda evaluation: evaluation.loss)
            current = self.best_records.get(benchmark_id)
            if current is None or _is_better(benchmark_best, current):
                self.best_records[benchmark_id] = benchmark_best

    def select_and_mutate(self, results: dict[str, list[HybridEvaluation]]) -> None:
        aggregate = self._rank_aggregate(results)
        for genome, fitness in zip(self.population, aggregate, strict=True):
            genome.fitness = fitness

        ranked_indices = sorted(range(len(self.population)), key=lambda index: aggregate[index])
        new_pop = [self.population[index].model_copy(deep=True) for index in ranked_indices[: self.config.elite_count]]

        while len(new_pop) < self.config.population_size:
            candidates = self.rng.sample(range(len(self.population)), min(3, len(self.population)))
            winner = min(candidates, key=lambda index: aggregate[index])
            child = self.population[winner].model_copy(deep=True)
            self.innovation_counter = mutate(child, self.rng, self.innovation_counter)
            new_pop.append(child)

        self.population = new_pop

    def run(self, benchmarks: dict[str, tuple]) -> dict[str, float]:
        t0 = time.perf_counter()
        if not self.population:
            self.initialize()
        for generation in range(self._start_generation, self.config.generations):
            self._current_generation = generation
            results = self.evaluate_population(benchmarks)
            self.update_best_records(results)
            self.generation_history.append(
                {
                    benchmark_id: _history_metric_value(record)
                    for benchmark_id, record in self.best_records.items()
                }
            )
            self._save_checkpoints(generation, results)
            self.select_and_mutate(results)
            self._save_state(generation)
            avg_loss = sum(record.loss for record in self.best_records.values()) / max(len(self.best_records), 1)
            print(f"[hybrid] gen {generation + 1}/{self.config.generations} best_avg_loss={avg_loss:.4f}", flush=True)
        self.wall_clock_seconds = time.perf_counter() - t0
        self._write_summary()
        return {
            benchmark_id: record.metric_value if record.metric_value is not None else float("inf")
            for benchmark_id, record in self.best_records.items()
        }

    def _save_checkpoints(self, generation: int, results: dict[str, list[HybridEvaluation]]) -> None:
        if self.run_dir is None:
            return
        gen_dir = self.run_dir / "checkpoints" / f"gen_{generation}"
        gen_dir.mkdir(parents=True, exist_ok=True)
        for benchmark_id, evaluations in results.items():
            best_eval = min(evaluations, key=lambda e: e.loss)
            if best_eval.status != "ok":
                continue
            genome = self.population[best_eval.genome_index]
            checkpoint = {
                "benchmark_id": benchmark_id,
                "generation": generation,
                "genome_index": best_eval.genome_index,
                "loss": best_eval.loss,
                "metric_name": best_eval.metric_name,
                "metric_value": best_eval.metric_value,
                "genome": genome.model_dump(mode="json"),
            }
            (gen_dir / f"{benchmark_id}.json").write_text(json.dumps(checkpoint, indent=2), encoding="utf-8")

    def _save_state(self, generation: int) -> None:
        if self.run_dir is None:
            return
        state = {
            "generation": generation,
            "next_generation": generation + 1,
            "innovation_counter": self.innovation_counter,
            "total_evaluations": self.total_evaluations,
            "wall_clock_seconds": self.wall_clock_seconds,
            "population": [genome.model_dump(mode="json") for genome in self.population],
            "generation_history": self.generation_history,
        }
        (self.run_dir / "state.json").write_text(json.dumps(state, indent=2), encoding="utf-8")

    def _write_summary(self) -> None:
        if self.run_dir is None:
            return
        summary = {
            "system": "hybrid",
            "status": "complete",
            "total_evaluations": self.total_evaluations,
            "wall_clock_seconds": round(self.wall_clock_seconds, 2),
            "generations_completed": self.config.generations,
            "epochs_per_candidate": self.config.epochs,
            "population_size": self.config.population_size,
            "best_fitness": {
                bid: rec.metric_value for bid, rec in self.best_records.items() if rec.metric_value is not None
            },
        }
        (self.run_dir / "summary.json").write_text(json.dumps(summary, indent=2), encoding="utf-8")

    @staticmethod
    def _rank_aggregate(results: dict[str, list[HybridEvaluation]]) -> list[float]:
        n = len(next(iter(results.values()))) if results else 0
        aggregate = [0.0] * n
        for evaluations in results.values():
            ordering = sorted(
                range(n),
                key=lambda index: (evaluations[index].status != "ok", evaluations[index].loss),
            )
            denominator = max(n - 1, 1)
            for rank, genome_index in enumerate(ordering):
                aggregate[genome_index] += rank / denominator
        benchmark_count = max(len(results), 1)
        return [value / benchmark_count for value in aggregate]


def _architecture_summary(genome: HybridGenome) -> str:
    node_parts = [f"{node.family.value}:{node.width}" for node in sorted(genome.enabled_nodes, key=lambda item: item.order)]
    return " -> ".join(node_parts) if node_parts else "empty"


def _parameter_count(model: nn.Module) -> int:
    leaves = nn.utils.tree_flatten(model.parameters())
    return int(sum(value.size for _, value in leaves))


def _is_better(left: HybridEvaluation, right: HybridEvaluation) -> bool:
    return (left.status == "ok", -left.loss) > (right.status == "ok", -right.loss)


def _history_metric_value(record: HybridEvaluation) -> float:
    if record.metric_value is None:
        return float("inf")
    return record.metric_value


class _LMTrainWrapper(nn.Module):
    """Reduce LM logits to last-token class probabilities for generic trainer."""

    def __init__(self, inner: nn.Module) -> None:
        super().__init__()
        self.inner = inner

    def __call__(self, x: mx.array) -> mx.array:
        out = self.inner(x)
        if out.ndim == 3:
            out = out[:, -1, :]
        return mx.softmax(out, axis=-1)


def _train_and_eval_lm(
    *,
    model: nn.Module,
    X_train: np.ndarray,
    y_train_last: np.ndarray,
    X_val: np.ndarray,
    y_val_full: np.ndarray,
    epochs: int,
    lr: float,
    batch_size: int,
) -> tuple[str, str, float | None, float, str | None]:
    wrapper = _LMTrainWrapper(model)
    train_result = train_and_evaluate(
        wrapper,
        X_train,
        y_train_last,
        X_val,
        y_val_full[:, -1],
        task="classification",
        epochs=epochs,
        lr=lr,
        batch_size=batch_size,
        parameter_count=_parameter_count(model),
    )
    if train_result.failure_reason is not None:
        return "perplexity", "min", None, float("inf"), train_result.failure_reason

    X = mx.array(X_val.astype(np.float32))
    y = mx.array(y_val_full.astype(np.int32))
    logits = model(X)
    mx.eval(logits)
    if logits.ndim != 3:
        return "perplexity", "min", None, float("inf"), "lm_logits_shape_mismatch"

    batch_size_actual, seq_len, vocab_size = logits.shape
    logits_flat = logits.reshape(batch_size_actual * seq_len, vocab_size)
    targets_flat = y.reshape(batch_size_actual * seq_len)
    probs_flat = mx.softmax(logits_flat, axis=-1)
    probs_flat = mx.clip(probs_flat, 1e-8, 1.0)
    loss = float(-mx.mean(mx.log(probs_flat[mx.arange(targets_flat.shape[0]), targets_flat])).item())
    if math.isnan(loss) or math.isinf(loss):
        return "perplexity", "min", None, float("inf"), "nan_loss"
    perplexity = math.exp(min(loss, 20.0))
    return "perplexity", "min", perplexity, loss, None
