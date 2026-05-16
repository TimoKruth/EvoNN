# EvoNN Evidence Registry

This directory is the canonical place for promoted comparison evidence.

The registry should stay compact. It is not a dump for every local `.tmp` run.
Promote only runs that are useful for a decision, a PR, or a durable research
claim.

## Promote Evidence

```bash
uv run --package evonn-compare evonn-compare evidence promote <workspace-or-summary> \
  --registry evidence \
  --label <stable-cohort-label>
```

Promotion writes:

- `index.jsonl`
- `registry_manifest.json`
- `evidence_report.json`
- `evidence_report.md`
- copied compact summaries under `runs/` when `--copy-artifacts` is enabled

Large raw run directories should stay outside git unless intentionally curated.

## Validate Evidence

```bash
uv run --package evonn-compare evonn-compare evidence validate \
  --registry evidence \
  --require-artifacts
```

Use validation before citing registry evidence in a pull request. Missing or
stale artifacts must be fixed or called out explicitly.

## Review Evidence

```bash
uv run --package evonn-compare evonn-compare evidence report --registry evidence
uv run --package evonn-compare evonn-compare dashboard evidence
```

The report is the concise decision surface. The dashboard is the visual review
surface for promoted fair-matrix summaries.

## Retention Rules

- Do not hand-edit existing registry rows.
- Do not commit large raw workspaces by default.
- Add new promoted rows for new evidence.
- Keep labels stable enough to compare before/after cohorts.
- Treat single-seed evidence as exploratory unless the report explicitly says
  the repeated-seed gate is satisfied.
