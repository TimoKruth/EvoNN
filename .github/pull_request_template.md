## Summary

- What changed?
- Why was it needed?

## Performance Evidence

Complete this section for optimization branches. Use `N/A` only when the PR is
not making a performance claim.

- Optimization family:
  State the single optimization family changed on this branch.
- Baseline artifact path:
  Link the baseline `baseline_manifest.json` or `baseline_summary.md`.
- After-change artifact path:
  Link the rerun artifact root, summary, and `perf_rows.jsonl`.
- Matrix parity:
  Confirm packs, budgets, seeds, backend class, and trust-state target match
  the baseline.
- Historical comparison flow reviewed:
  Name the exact `historical-baseline` / `workspace-report` outputs reviewed.

## Guardrails

- Quality verdict:
  `same`, `better`, or `regressed`; include the declared tolerance when
  relevant.
- Trust-state verdict:
  Include operating state, accounting state, and repeatability state.
- Budget-accounting verdict:
  State whether accounting semantics changed. Silent semantic drift is not
  acceptable.
- Failure/completion delta:
  Call out failures, partial benchmarks, and completion-rate changes
  explicitly.
- Branch outcome:
  Choose exactly one: `accepted`, `rejected-for-revision`, `scrapped`
- Policy basis:
  Name the acceptance or scrap rule from `PERFORMANCE_OPTIMIZATION_WORKFLOW.md`.

## Decision Gate Summary

- Recommended category:
  Choose exactly one: `promote`, `regress`, `inconclusive`,
  `needs more seeds`, `Tier B-only gain`, `Tier 1 regression`
- Pack / budget / seeds:
- Comparison labels reviewed:
- Lane state:
  Include operating state, accounting state, and repeatability state.
- Exact case IDs:
- Exact run IDs:
- Dashboard slices reviewed:
- Evidence links:
  Link the workspace trend report, dashboard, and the most relevant summary
  JSON or markdown artifacts. For optimization branches, include both baseline
  and after-change artifact paths.
- Why:
  Explain the category in 2-4 bullets from the evidence.

## Validation

- Commands run:
- Relevant package or CI surfaces checked:

## Notes

- Follow-ups, caveats, or explicit blockers
- Canonical optimization workflow:
  `PERFORMANCE_OPTIMIZATION_WORKFLOW.md`
