"""Cache-compatible parallel evaluation.

Key design difference from EvoNN-2: compilation and cache lookup happen
BEFORE parallelization. This module only parallelizes the training step,
avoiding the need to serialize genome objects or re-import heavy modules
in worker processes.
"""

from __future__ import annotations

import os
from concurrent.futures import ProcessPoolExecutor, Future
from typing import Any, Callable


def _default_workers() -> int:
    """Auto-detect worker count: cpu_count - 1, minimum 1."""
    cpu = os.cpu_count() or 2
    return max(1, cpu - 1)


class ParallelEvaluator:
    """Parallel batch evaluator for training tasks.

    Usage::

        evaluator = ParallelEvaluator(max_workers=4)
        results = evaluator.evaluate_batch(tasks, train_fn)

    Where each task is a tuple of arguments passed to train_fn.
    Compilation and weight cache lookups should be done BEFORE
    creating the task list.
    """

    def __init__(self, max_workers: int = 0) -> None:
        self.max_workers = max_workers or _default_workers()

    def evaluate_batch(
        self,
        tasks: list[tuple],
        train_fn: Callable[..., Any],
    ) -> list[Any]:
        """Evaluate a batch of training tasks.

        Args:
            tasks: List of argument tuples. Each is unpacked as train_fn(*task).
            train_fn: A picklable function that takes task args and returns a result.

        Returns:
            List of results in the same order as tasks.
        """
        if not tasks:
            return []

        if self.max_workers <= 1 or len(tasks) == 1:
            return [train_fn(*t) for t in tasks]

        # Cap workers to task count
        n_workers = min(self.max_workers, len(tasks))

        with ProcessPoolExecutor(max_workers=n_workers) as pool:
            futures: list[Future] = [
                pool.submit(train_fn, *task) for task in tasks
            ]
            results: list[Any] = []
            for future in futures:
                try:
                    results.append(future.result())
                except Exception:
                    results.append(None)
            return results

    def map_batch(
        self,
        items: list[Any],
        fn: Callable[[Any], Any],
    ) -> list[Any]:
        """Simple parallel map over a list of items.

        Args:
            items: List of single arguments.
            fn: A picklable function applied to each item.

        Returns:
            List of results in order.
        """
        if not items:
            return []
        if self.max_workers <= 1 or len(items) == 1:
            return [fn(item) for item in items]

        n_workers = min(self.max_workers, len(items))
        with ProcessPoolExecutor(max_workers=n_workers) as pool:
            return list(pool.map(fn, items))
