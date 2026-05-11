# EvoNN

EvoNN is a research monorepo for multiple evolutionary/neural search systems,
their shared trust substrate, and the comparison workflows used to evaluate
them on a common benchmark surface.

The workspace is organized around one default operating lane and several
challenger systems:

- `EvoNN-Prism`: default day-to-day operating engine
- `EvoNN-Topograph`: first serious challenger
- `EvoNN-Compare`: shared compare and reporting layer
- `EvoNN-Shared`: shared contracts and trust substrate
- `EvoNN-Contenders`, `EvoNN-Primordia`, `EvoNN-Stratograph`: challenger lanes

## Active Execution Docs

Use `EVONN_CONSOLIDATED_PLAN.md` as the single active execution plan.
`VISION.md` remains product and research framing.

## Quick Start

Prerequisite:

- `uv` installed locally

Bootstrap the workspace from the repository root:

```bash
uv sync --all-packages --extra dev
```

Run the trusted recurring lane locally:

```bash
bash scripts/ci/shared-checks.sh all
bash scripts/ci/compare-checks.sh all
bash scripts/ci/contenders-checks.sh all
bash scripts/ci/primordia-checks.sh all
bash scripts/ci/stratograph-checks.sh all
```

For MLX-native engine work, keep using the macOS package checks for Prism and
Topograph separately from this Linux-safe recurring lane.

Run the default comparison surface:

```bash
uv run --package evonn-compare python -m evonn_compare fair-matrix \
  --workspace .tmp/fair-matrix-smoke
```

## Operating Model

- Prism is the default operating engine for routine work.
- Topograph is the primary challenger on the shared compare surface.
- `fair-matrix` and `campaign` default to the trusted daily `local` lane
  (`tier1_core` @ `64`) when no pack or preset is supplied.
- Expanded benchmark-ladder presets are available for staged research:
  `tier_a_smoke`, `tier_a_contract`, `tier_b_local_v2`,
  `tier_b_overnight_v2`, `tier_b_extended_v2`, `tier_b_weekend_v2`,
  `tier_c_local`, `tier_c_overnight`, `tier_c_extended`, `tier_c_weekend`,
  `tier_d_local`, `tier_d_broad`, `tier_d_overnight`, and `tier_d_weekend`.
- Use `evonn-compare benchmark-audit --pack <pack>` before promoting a new pack;
  promoted packs require explicit required contender-floor metadata.
- Tier D is promoted for the current 26-benchmark admitted broad pack, but it
  remains a separate broad-lane leaderboard from Tier A/B/C trend claims.
- Shared infrastructure should converge where it improves trust, parity, and
  maintenance, while search-core logic stays package-local.

## Git Workflow

- Do not push directly to `main`.
- Start each change on an issue-specific branch such as
  `agent/evo-24-short-description`.
- If you need isolation beyond a branch, use a dedicated git worktree per issue.
- Open a pull request and merge through review rather than publishing directly
  from the working branch.

See [CONTRIBUTING.md](./CONTRIBUTING.md) for the expected branch and PR flow.
Engine-advancement PRs must also follow
[RESEARCH_DECISION_GATE.md](./RESEARCH_DECISION_GATE.md).

## Where To Read Next

- [MONOREPO.md](./MONOREPO.md): workspace structure, commands, and validation matrix
- [ARCHITECTURE_OVERVIEW.md](./ARCHITECTURE_OVERVIEW.md): system-level technical framing
- [VISION.md](./VISION.md): product and research direction
- [EVONN_CONSOLIDATED_PLAN.md](./EVONN_CONSOLIDATED_PLAN.md): active execution plan
- [RESEARCH_DECISION_GATE.md](./RESEARCH_DECISION_GATE.md): decision categories,
  evidence bundle, and PR expectations for advancement claims

Package-specific usage and testing details live in each package README.
