You are running one bounded Ralph-style execution pass for the EvoNN monorepo.

Mission
Move the repository materially closer to implementing the root and package vision documents, without taking broad risky swings.

Primary source documents to read first
- VISION.md
- EvoNN-Primordia/VISION.md
- EvoNN-Prism/VISION.md
- EvoNN-Topograph/VISION.md
- EvoNN-Stratograph/VISION.md
- SEARCH_ENGINE_OUTPUT_PARITY_PLAN.md
- .hermes/plans/README.md

Secondary context only
- EVONN_90_DAY_PLAN.md (mostly outdated; use only when it still matches the vision docs and current code)
- Package-local plans/docs that are directly relevant to the next slice you choose

Definition of a good pass
- Pick exactly one high-leverage, bounded slice
- Prefer real code, validation, CI, contracts, or trust-lane work over cosmetic docs
- Keep the change focused enough to land as one reviewable commit
- Run the smallest sufficient test set that actually verifies the change
- Leave the tree clean

Hard constraints
- Work only in this repo
- At most one commit this pass
- Do not use destructive git commands
- Do not revert unrelated user changes
- If you cannot make a safe verified change, stop cleanly and explain why
- If you discover the highest-value next step belongs in a different subsystem, switch to that subsystem only for this pass

User-specific git discipline
- If you create a commit, keep it focused
- After the commit, the outer loop will pull latest remote branch changes, merge, and push; do not do extra unrelated git choreography

Priority heuristic
1. Trusted `tier1_core` daily-lane maturity
2. Budget/accounting truth and fairness semantics
3. Trend artifacts and compare decision surfaces
4. CI/trust-layer validation coverage
5. Contender floor hardening
6. Seeding/transfer loop execution readiness
7. Remaining package-local engine advancement that clearly supports the umbrella vision

Suggested workflow
1. Read the root vision/plan files first.
2. Inspect recent code and tests in the most relevant subsystem.
3. Choose one bounded slice that materially advances the roadmap.
4. Implement it.
5. Run relevant verification.
6. If verification passes, create one commit.
7. End with the exact template below.

Final response template
STOP_REASON: <committed_change|no_safe_change|no_remaining_slice|needs_human|verification_failed>
SUMMARY: <one sentence>
TESTS: <commands run, or none>
COMMIT: <sha or none>
NEXT: <one sentence>
