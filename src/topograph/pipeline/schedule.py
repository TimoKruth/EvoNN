"""Phase-based adaptive mutation scheduling."""

from __future__ import annotations

from dataclasses import dataclass, field
from enum import Enum

from topograph.config import MutationRates


class EvolutionPhase(Enum):
    EXPLORE = "explore"  # gen 0-33%
    REFINE = "refine"    # gen 33-66%
    POLISH = "polish"    # gen 66-100%


@dataclass
class OperatorStats:
    applications: int = 0
    improvements: int = 0
    ema_success: float = 0.5


# Phase profiles: operator_name -> base probability
_PHASE_PROFILES: dict[EvolutionPhase, dict[str, float]] = {
    EvolutionPhase.EXPLORE: {
        "width": 0.25,
        "activation": 0.15,
        "add_layer": 0.30,
        "remove_layer": 0.05,
        "add_connection": 0.25,
        "remove_connection": 0.05,
        "add_residual": 0.15,
        "weight_bits": 0.02,
        "activation_bits": 0.02,
        "sparsity": 0.05,
        "operator_type": 0.10,
    },
    EvolutionPhase.REFINE: {
        "width": 0.35,
        "activation": 0.15,
        "add_layer": 0.10,
        "remove_layer": 0.08,
        "add_connection": 0.12,
        "remove_connection": 0.08,
        "add_residual": 0.10,
        "weight_bits": 0.12,
        "activation_bits": 0.10,
        "sparsity": 0.12,
        "operator_type": 0.10,
    },
    EvolutionPhase.POLISH: {
        "width": 0.20,
        "activation": 0.10,
        "add_layer": 0.02,
        "remove_layer": 0.02,
        "add_connection": 0.05,
        "remove_connection": 0.05,
        "add_residual": 0.03,
        "weight_bits": 0.30,
        "activation_bits": 0.25,
        "sparsity": 0.25,
        "operator_type": 0.05,
    },
}

_ALL_OPERATORS = list(_PHASE_PROFILES[EvolutionPhase.EXPLORE].keys())


class MutationScheduler:
    """Adaptive mutation rate scheduler with phase-based profiles and EMA tracking."""

    def __init__(self, base_rates: MutationRates | None = None) -> None:
        self._base_rates = base_rates
        self._stats: dict[str, OperatorStats] = {
            name: OperatorStats() for name in _ALL_OPERATORS
        }

    def current_phase(self, generation: int, total_generations: int) -> EvolutionPhase:
        if total_generations <= 0:
            return EvolutionPhase.EXPLORE
        progress = generation / total_generations
        if progress < 1 / 3:
            return EvolutionPhase.EXPLORE
        if progress < 2 / 3:
            return EvolutionPhase.REFINE
        return EvolutionPhase.POLISH

    def get_rates(self, generation: int, total_generations: int) -> dict[str, float]:
        """Return operator probabilities scaled by phase profile and success EMA."""
        phase = self.current_phase(generation, total_generations)
        profile = _PHASE_PROFILES[phase]

        # If user provided base_rates, blend them (50/50) with phase profile
        if self._base_rates is not None:
            user = {
                name: getattr(self._base_rates, name, profile[name])
                for name in _ALL_OPERATORS
            }
            base = {name: (profile[name] + user[name]) / 2 for name in _ALL_OPERATORS}
        else:
            base = dict(profile)

        # Scale by operator success EMA and clamp within +-50% of base
        rates: dict[str, float] = {}
        for name in _ALL_OPERATORS:
            b = base[name]
            ema = self._stats[name].ema_success
            # Scale: ema=0.5 -> 1x, ema=1.0 -> 1.5x, ema=0.0 -> 0.5x
            scale = 0.5 + ema
            scaled = b * scale
            lo = b * 0.5
            hi = b * 1.5
            rates[name] = max(lo, min(hi, scaled))

        return rates

    def record_outcome(self, operator_name: str, improved: bool) -> None:
        """Update EMA and counters for an operator."""
        stats = self._stats.get(operator_name)
        if stats is None:
            self._stats[operator_name] = OperatorStats()
            stats = self._stats[operator_name]
        stats.applications += 1
        if improved:
            stats.improvements += 1
        stats.ema_success = 0.9 * stats.ema_success + 0.1 * (1.0 if improved else 0.0)

    def stats_summary(self) -> dict[str, dict[str, float | int]]:
        """Return per-operator stats for reporting."""
        return {
            name: {
                "applications": s.applications,
                "improvements": s.improvements,
                "ema_success": round(s.ema_success, 4),
                "success_rate": round(s.improvements / s.applications, 4)
                if s.applications > 0
                else 0.0,
            }
            for name, s in sorted(self._stats.items())
        }

    def to_dict(self) -> dict[str, object]:
        return {
            "stats": {
                name: {
                    "applications": stats.applications,
                    "improvements": stats.improvements,
                    "ema_success": stats.ema_success,
                }
                for name, stats in self._stats.items()
            }
        }

    def load_dict(self, data: dict[str, object] | None) -> None:
        stats = data.get("stats", {}) if data else {}
        for name, raw in stats.items():
            if name not in self._stats or not isinstance(raw, dict):
                continue
            self._stats[name] = OperatorStats(
                applications=int(raw.get("applications", 0)),
                improvements=int(raw.get("improvements", 0)),
                ema_success=float(raw.get("ema_success", 0.5)),
            )
