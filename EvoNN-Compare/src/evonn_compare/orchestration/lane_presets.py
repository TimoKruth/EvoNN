"""Named compare lane presets for recurring and exploratory workflows."""

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
        description="Lowest-cost validation lane for contract and artifact checks; not the trusted daily default.",
    ),
    "local": LanePreset(
        name="local",
        pack="tier1_core",
        seeds=(42,),
        budgets=(64,),
        description="Default trusted daily tier1_core lane for a fuller but still practical head-to-head.",
    ),
    "overnight": LanePreset(
        name="overnight",
        pack="tier1_core",
        seeds=(42,),
        budgets=(256,),
        description="Budget-richer daily lane for fairness convergence beyond the local 64-eval slice.",
    ),
    "weekend": LanePreset(
        name="weekend",
        pack="tier1_core",
        seeds=(42,),
        budgets=(1000,),
        description="High-budget trusted-lane preset for repeated weekend-scale comparison studies.",
    ),
    "tier_b_local": LanePreset(
        name="tier_b_local",
        pack="tier_b_core",
        seeds=(42,),
        budgets=(64,),
        description="Default bounded local Tier B research loop on the canonical ladder pack.",
    ),
    "tier_b_overnight": LanePreset(
        name="tier_b_overnight",
        pack="tier_b_core",
        seeds=(42,),
        budgets=(256,),
        description="Preferred deeper Tier B study preset on the canonical ladder pack.",
    ),
    "tier_b_weekend": LanePreset(
        name="tier_b_weekend",
        pack="tier_b_core",
        seeds=(42,),
        budgets=(1000,),
        description="High-budget Tier B preset for repeated longer local studies.",
    ),
    "tier_a_smoke": LanePreset(
        name="tier_a_smoke",
        pack="tier_a_contract",
        seeds=(42,),
        budgets=(16,),
        description="Lowest-budget Tier A contract lane with explicit contender-floor metadata.",
    ),
    "tier_a_contract": LanePreset(
        name="tier_a_contract",
        pack="tier_a_contract",
        seeds=(42,),
        budgets=(64,),
        description="Decision-grade Tier A contract lane with explicit contender-floor metadata.",
    ),
    "tier_b_local_v2": LanePreset(
        name="tier_b_local_v2",
        pack="tier_b_core_v2",
        seeds=(42,),
        budgets=(96,),
        description="Expanded Tier B local research lane with 12 benchmarks and required contender floors.",
    ),
    "tier_b_overnight_v2": LanePreset(
        name="tier_b_overnight_v2",
        pack="tier_b_core_v2",
        seeds=(42,),
        budgets=(384,),
        description="Expanded Tier B overnight lane for promotion evidence.",
    ),
    "tier_b_extended_v2": LanePreset(
        name="tier_b_extended_v2",
        pack="tier_b_core_v2",
        seeds=(42,),
        budgets=(768,),
        description="Expanded Tier B extended promotion-evidence lane.",
    ),
    "tier_b_weekend_v2": LanePreset(
        name="tier_b_weekend_v2",
        pack="tier_b_core_v2",
        seeds=(42,),
        budgets=(1536,),
        description="Expanded Tier B weekend lane.",
    ),
    "tier_c_local": LanePreset(
        name="tier_c_local",
        pack="tier_c_architecture_sensitive",
        seeds=(42,),
        budgets=(128,),
        description="Exploratory Tier C local architecture-sensitive lane.",
    ),
    "tier_c_overnight": LanePreset(
        name="tier_c_overnight",
        pack="tier_c_architecture_sensitive",
        seeds=(42,),
        budgets=(512,),
        description="Exploratory Tier C overnight architecture-sensitive lane.",
    ),
    "tier_c_extended": LanePreset(
        name="tier_c_extended",
        pack="tier_c_architecture_sensitive",
        seeds=(42,),
        budgets=(1024,),
        description="Exploratory Tier C extended architecture-sensitive lane for promotion evidence.",
    ),
    "tier_c_weekend": LanePreset(
        name="tier_c_weekend",
        pack="tier_c_architecture_sensitive",
        seeds=(42,),
        budgets=(2048,),
        description="Exploratory Tier C weekend architecture-sensitive lane.",
    ),
    "tier_d_local": LanePreset(
        name="tier_d_local",
        pack="tier_d_broad_shared",
        seeds=(42,),
        budgets=(208,),
        description="Special-only broad shared benchmark lane at the lowest admitted-pack Tier D budget.",
    ),
    "tier_d_broad": LanePreset(
        name="tier_d_broad",
        pack="tier_d_broad_shared",
        seeds=(42,),
        budgets=(416,),
        description="Special-only broad shared benchmark lane; keep separate until repeated clean runs exist.",
    ),
    "tier_d_overnight": LanePreset(
        name="tier_d_overnight",
        pack="tier_d_broad_shared",
        seeds=(42,),
        budgets=(832,),
        description="Special-only broad shared overnight lane; keep separate until repeated clean runs exist.",
    ),
    "tier_d_weekend": LanePreset(
        name="tier_d_weekend",
        pack="tier_d_broad_shared",
        seeds=(42,),
        budgets=(1664,),
        description="Special-only broad shared weekend lane; keep separate until repeated clean runs exist.",
    ),
}


def lane_preset_help(*, default_name: str | None = None) -> str:
    available = ", ".join(
        f"{preset.name} ({preset.description})"
        for preset in (LANE_PRESETS[name] for name in sorted(LANE_PRESETS))
    )
    if default_name:
        return (
            "Named lane preset "
            f"(defaults to {default_name} when neither --pack nor --preset is supplied; "
            f"available: {available})"
        )
    return f"Named lane preset (available: {available})"



def resolve_lane_preset(name: str) -> LanePreset:
    try:
        return LANE_PRESETS[name]
    except KeyError as exc:
        available = ", ".join(sorted(LANE_PRESETS))
        raise ValueError(f"unknown lane preset '{name}'; available: {available}") from exc
