"""Named local-first compare lane presets."""

from __future__ import annotations

from dataclasses import dataclass


@dataclass(frozen=True)
class LanePreset:
    name: str
    pack: str
    seeds: tuple[int, ...]
    budgets: tuple[int, ...]
    description: str


LANE_PRESETS: dict[str, LanePreset] = {
    "smoke": LanePreset(
        name="smoke",
        pack="tier1_core_smoke",
        seeds=(42,),
        budgets=(16,),
        description="Lowest-cost repeatable compare lane for contract and artifact validation.",
    ),
    "local": LanePreset(
        name="local",
        pack="tier1_core",
        seeds=(42,),
        budgets=(64,),
        description="Default local-first compare lane for a fuller but still practical head-to-head.",
    ),
}


def resolve_lane_preset(name: str) -> LanePreset:
    try:
        return LANE_PRESETS[name]
    except KeyError as exc:
        available = ", ".join(sorted(LANE_PRESETS))
        raise ValueError(f"unknown lane preset '{name}'; available: {available}") from exc
