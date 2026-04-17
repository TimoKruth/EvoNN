"""Wilcoxon signed-rank test and bootstrap CI for paired benchmark comparisons."""
from __future__ import annotations

from dataclasses import dataclass

import numpy as np
from scipy import stats


@dataclass(frozen=True)
class WilcoxonResult:
    p_value: float | None
    statistic: float | None
    n_pairs: int
    median_delta: float
    mean_delta: float

    @property
    def significant_at_005(self) -> bool:
        return self.p_value is not None and self.p_value < 0.05


@dataclass(frozen=True)
class BootstrapCI:
    lower: float
    upper: float
    mean: float
    confidence_level: float


MIN_PAIRS_FOR_WILCOXON = 6


def wilcoxon_signed_rank(
    left_values: list[float], right_values: list[float]
) -> WilcoxonResult:
    assert len(left_values) == len(right_values)
    deltas = [a - b for a, b in zip(left_values, right_values)]
    n = len(deltas)
    median_delta = float(np.median(deltas))
    mean_delta = float(np.mean(deltas))
    nonzero = [d for d in deltas if abs(d) > 1e-12]
    if not nonzero:
        return WilcoxonResult(
            p_value=1.0,
            statistic=0.0,
            n_pairs=n,
            median_delta=median_delta,
            mean_delta=mean_delta,
        )
    if len(nonzero) < MIN_PAIRS_FOR_WILCOXON:
        return WilcoxonResult(
            p_value=None,
            statistic=None,
            n_pairs=n,
            median_delta=median_delta,
            mean_delta=mean_delta,
        )
    result = stats.wilcoxon(nonzero, alternative="two-sided")
    return WilcoxonResult(
        p_value=float(result.pvalue),
        statistic=float(result.statistic),
        n_pairs=n,
        median_delta=median_delta,
        mean_delta=mean_delta,
    )


def bootstrap_confidence_interval(
    deltas: list[float],
    *,
    n_resamples: int = 10_000,
    confidence_level: float = 0.95,
    seed: int | None = None,
) -> BootstrapCI:
    rng = np.random.default_rng(seed)
    arr = np.array(deltas, dtype=np.float64)
    means = np.array(
        [
            float(rng.choice(arr, size=len(arr), replace=True).mean())
            for _ in range(n_resamples)
        ]
    )
    alpha = 1.0 - confidence_level
    lower = float(np.percentile(means, 100 * alpha / 2))
    upper = float(np.percentile(means, 100 * (1 - alpha / 2)))
    return BootstrapCI(
        lower=lower,
        upper=upper,
        mean=float(arr.mean()),
        confidence_level=confidence_level,
    )
