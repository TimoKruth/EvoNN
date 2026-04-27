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

## Quick Start

Prerequisite:

- `uv` installed locally

Bootstrap the workspace from the repository root:

```bash
uv sync --all-packages --extra dev
```

Run the main trust-layer checks:

```bash
bash scripts/ci/shared-checks.sh all
bash scripts/ci/compare-checks.sh all
bash scripts/ci/contenders-checks.sh all
```

Run the default comparison surface:

```bash
uv run --package evonn-compare python -m evonn_compare fair-matrix \
  --workspace .tmp/fair-matrix-smoke
```

## Operating Model

- Prism is the default operating engine for routine work.
- Topograph is the primary challenger on the shared compare surface.
- `fair-matrix` and `campaign` default to the low-cost `smoke` lane when no
  pack or preset is supplied.
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

## Where To Read Next

- [MONOREPO.md](./MONOREPO.md): workspace structure, commands, and validation matrix
- [ARCHITECTURE_OVERVIEW.md](./ARCHITECTURE_OVERVIEW.md): system-level technical framing
- [VISION.md](./VISION.md): product and research direction
- [ROADMAP.md](./ROADMAP.md): execution sequencing
- [EVONN_90_DAY_PLAN.md](./EVONN_90_DAY_PLAN.md): current delivery window

Package-specific usage and testing details live in each package README.
