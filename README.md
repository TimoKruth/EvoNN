# EvoNN

EvoNN is a research monorepo for multiple evolutionary/neural search systems,
their shared trust substrate, and the comparison workflows used to evaluate
them on a common benchmark surface.

The workspace is organized around one default operating engine, a shared compare
surface, and several challenger systems:

- `EvoNN-Prism`: default day-to-day operating engine
- `EvoNN-Topograph`: first serious challenger
- `EvoNN-Compare`: shared compare and reporting layer
- `EvoNN-Shared`: shared contracts and trust substrate
- `EvoNN-Contenders`, `EvoNN-Primordia`, `EvoNN-Stratograph`: challenger lanes

## Documentation Map

Start here for current work:

- [MONOREPO.md](./MONOREPO.md): workspace structure, commands, and validation matrix
- [EvoNN-Compare/README.md](./EvoNN-Compare/README.md): compare lanes,
  fair-matrix presets, trend artifacts, dashboards, and transfer workflows
- [PERFORMANCE_OPTIMIZATION_WORKFLOW.md](./PERFORMANCE_OPTIMIZATION_WORKFLOW.md):
  `EVO-46` performance epic closeout and optimization-branch review workflow
- [RESEARCH_DECISION_GATE.md](./RESEARCH_DECISION_GATE.md): evidence required
  before claiming a branch improved a system or should affect the default lane
- [EVONN_90_DAY_PLAN.md](./EVONN_90_DAY_PLAN.md): current quarter direction

Use these for strategy and backlog context:

- [ROADMAP.md](./ROADMAP.md): long-horizon umbrella sequencing
- [VISION.md](./VISION.md): product and research framing
- [.hermes/plans/README.md](./.hermes/plans/README.md): branch-sized
  package and subsystem advancement backlog

Historical/bootstrap plans are kept for context only. Do not use package-local
`IMPLEMENTATION_PLAN.md` files as the active execution source unless a newer doc
explicitly points back to them.

## Quick Start

Prerequisite:

- `uv` installed locally

Bootstrap the workspace from the repository root:

```bash
uv sync --all-packages --extra dev
```

Run the trusted recurring lane checks locally:

```bash
bash scripts/ci/shared-checks.sh all
bash scripts/ci/compare-checks.sh all
bash scripts/ci/contenders-checks.sh all
bash scripts/ci/primordia-checks.sh all
bash scripts/ci/stratograph-checks.sh all
```

For MLX-native engine work, keep using the macOS package checks for Prism and
Topograph separately from this Linux-safe recurring lane.

Run a low-cost all-project comparison smoke:

```bash
uv run --package evonn-compare python -m evonn_compare fair-matrix \
  --preset smoke \
  --workspace .tmp/fair-matrix-smoke
```

Run the default trusted daily comparison lane:

```bash
uv run --package evonn-compare python -m evonn_compare fair-matrix \
  --workspace .tmp/fair-matrix-local
```

## Operating Model

- Prism is the default operating engine for routine work.
- Topograph is the primary challenger on the shared compare surface.
- `fair-matrix` and `campaign` default to the trusted daily `local` lane
  (`tier1_core` @ `64`) when no pack or preset is supplied.
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

- [ARCHITECTURE_OVERVIEW.md](./ARCHITECTURE_OVERVIEW.md): system-level technical framing
- [BENCHMARK_LADDER.md](./BENCHMARK_LADDER.md): benchmark tiering and intended use
- [BUDGET_ACCOUNTING_POLICY.md](./BUDGET_ACCOUNTING_POLICY.md): evaluation-count
  and fairness accounting rules
- Package-specific usage and testing details live in each package README.
