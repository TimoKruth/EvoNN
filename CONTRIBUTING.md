# Contributing To EvoNN

## Branch And PR Policy

- Do not work directly on `main`.
- Create an issue-specific branch before making changes.
- Use a descriptive branch name such as `agent/evo-24-pr-workflow` or
  `user/evo-57-compare-smoke-fix`.
- If multiple efforts need to stay isolated at once, create a separate git
  worktree per issue or feature branch.
- Push the branch, open a pull request, and merge through review.
- If the PR makes an engine-advancement claim, complete the decision-gate
  evidence summary from [RESEARCH_DECISION_GATE.md](./RESEARCH_DECISION_GATE.md)
  and link the exact trend, dashboard, case, and run artifacts used for the
  claim.

## Suggested Flow

```bash
git fetch origin
git switch main
git pull --ff-only origin main
git switch -c agent/evo-24-short-description
```

Optional worktree flow:

```bash
git fetch origin
git worktree add ../EvoNN-evo-24 -b agent/evo-24-short-description origin/main
```

Before opening a PR, run the relevant package checks from the repository root.
For shared-surface changes and updates to the trusted recurring lane docs or
workflow, run the full Linux-safe recurring lane:

```bash
bash scripts/ci/shared-checks.sh all
bash scripts/ci/compare-checks.sh all
bash scripts/ci/contenders-checks.sh all
bash scripts/ci/primordia-checks.sh all
bash scripts/ci/stratograph-checks.sh all
```

If the change also touches Prism or Topograph runtime behavior, run the
package-local macOS checks in their native lane as well.

## Evidence Expectations For Advancement PRs

When a PR argues that an engine improved, regressed, or should become the new
default, reviewers should not have to reconstruct the claim from chat or local
shell history.

The PR must include:

- one decision category from `RESEARCH_DECISION_GATE.md`
- the workspace trend report and dashboard paths
- exact case IDs and run IDs
- the dashboard slices used for the judgment
- the lane operating state, accounting state, and repeatability state

If the claim is relative to a prior branch or release, import that baseline into
the live workspace with `evonn-compare historical-baseline` and review both
cohorts from the same artifact surface.
