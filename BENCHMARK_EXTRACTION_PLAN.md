# Benchmark Extraction Plan

## Goal

Extract benchmark definitions into one independent root-level folder so all projects can use the same benchmark source at the same time, without copying YAMLs between repos.

Decision for this plan: **Plan A**.

Plan A means:

- one shared root-level benchmark folder becomes the normal default everywhere
- explicit per-project env overrides still work
- local repo copies remain transitional fallback only
- sibling-repo probing is migration-only behavior, not normal behavior

Recommended target:

```text
/Users/timokruth/Projekte/Evo Neural Nets/shared-benchmarks/
  catalog/
  suites/
    common/
    parity/
    topograph/
  README.md
```

Phase 1 should move benchmark data only, not force a new shared Python package yet.

## Source-based findings

These points come from current source and current repo layout, not from design docs.

### 1. Benchmark catalogs are already duplicated and have drifted

- `EvoNN-Topograph/benchmarks/catalog/` contains 152 YAML specs.
- `EvoNN-Contenders/benchmarks/catalog/` also contains 152 YAML specs.
- `EvoNN-Prism/benchmarks/catalog/` contains only 6 YAML specs.

Important drift already exists:

- `EvoNN-Prism/benchmarks/catalog/moons.yaml` and `EvoNN-Topograph/benchmarks/catalog/moons.yaml` differ.
- Prism LM YAMLs also differ from Topograph LM YAMLs:
  - `tiny_lm_synthetic.yaml`
  - `tinystories_lm.yaml`
  - `tinystories_lm_smoke.yaml`
  - `wikitext2_lm.yaml`
  - `wikitext2_lm_smoke.yaml`

So there is no safe way to call the current copied catalogs "equivalent".

### 2. Prism is already half-prepared for an external catalog

`EvoNN-Prism/src/prism/benchmarks/datasets.py` already resolves its catalog through `PRISM_CATALOG_DIR` and a default local path.

That means Prism can be migrated with the smallest change: point default resolution at the new shared folder, keep local fallback for transition.

### 3. Topograph still assumes repo-local benchmark ownership

`EvoNN-Topograph/src/topograph/benchmarks/registry.py` hardcodes its catalog directory under `EvoNN-Topograph/benchmarks/catalog/`.

`EvoNN-Topograph/src/topograph/benchmarks/parity.py` also hardcodes:

- catalog path under `EvoNN-Topograph/benchmarks/catalog/`
- suites path under `EvoNN-Topograph/benchmarks/suites/`

So Topograph is the current practical owner of the full benchmark catalog and suite set.

### 4. Stratograph already works around the duplication by probing sibling repos

`EvoNN-Stratograph/src/stratograph/benchmarks/registry.py` loads catalogs from:

- `EvoNN-Stratograph/benchmarks/catalog`
- `EvoNN-Topograph/benchmarks/catalog`
- `EvoNN-Prism/benchmarks/catalog`

in that order, with first-hit-wins behavior.

That is useful as a temporary bridge, but it is not a clean shared source:

- lookup order matters
- shadowing is implicit
- two different YAMLs with the same name can silently resolve differently depending on folder order

### 5. Compare is still coupled to Topograph for normal benchmark loading

`EvoNN-Compare/src/evonn_compare/hybrid/benchmarks.py` imports `topograph.benchmarks.parity.get_benchmark` for non-LM benchmark loading.

So even if the YAMLs are moved, Compare is not yet truly benchmark-independent.

### 6. Canonical benchmark ID maps are duplicated too

Separate `CANONICAL_BENCHMARK_IDS` maps exist in:

- `EvoNN-Prism/src/prism/benchmarks/parity.py`
- `EvoNN-Topograph/src/topograph/benchmarks/parity.py`
- `EvoNN-Stratograph/src/stratograph/benchmarks/parity.py`
- `EvoNN-Contenders/src/evonn_contenders/benchmarks/parity.py`

This is related, but it is not the best first cut for benchmark extraction. First centralize benchmark data, then reduce parity-map duplication.

## Recommendation

Best immediate approach:

1. create one shared root-level benchmark data folder
2. move catalog YAMLs there
3. move suite YAMLs there
4. make every project resolve shared paths first
5. keep local fallback paths during migration only

Do **not** start with a full shared runtime package.

Why:

- lowest blast radius
- no cross-project packaging/bootstrap work on day one
- all projects can read the same files immediately
- source duplication disappears first
- loader duplication can be cleaned up later

This gets the important part right first: one benchmark dataset definition per benchmark name.

Policy note:

- Prism's current root docs prefer explicit external catalogs.
- This extraction plan intentionally overrides that for the shared benchmark layer.
- After migration, normal runs should resolve shared benchmark data by default.
- Explicit env vars still win when a project needs custom override behavior.

## Proposed folder layout

```text
shared-benchmarks/
  README.md
  catalog/
    adult.yaml
    moons.yaml
    tiny_lm_synthetic.yaml
    ...
  suites/
    common/
      smoke.yaml
      broad.yaml
    parity/
      shared_33plus5.yaml
      tier1_core.yaml
      tier3_topology.yaml
    topograph/
      heavy_lm.yaml
  migration/
    prism_catalog_drift.md
```

Notes:

- `catalog/` is the main source of truth.
- `suites/` belongs here because Topograph currently treats suite definitions as benchmark-side config, not compare-contract config.
- `targets/` should stay in Topograph for now. Those are device/runtime targets, not benchmark definitions.
- `parity_packs/` should stay in `EvoNN-Compare` for phase 1. They are compare contracts, not benchmark specs.

## Source of truth choice

Use `EvoNN-Topograph/benchmarks/catalog/` as the initial baseline snapshot.

Why this is the safest current base:

- it is the complete 152-file set
- `EvoNN-Contenders/benchmarks/catalog/` currently matches it at the file-set level
- Topograph already owns the current suite files under `benchmarks/suites/`
- Prism’s local catalog is intentionally smaller and already diverged on shared names

Before cutover, normalize the few Prism-only drift cases manually and record the decisions.

## What should change in each project

### Prism

Change benchmark resolution to:

1. explicit env override
2. shared root folder
3. local repo fallback

Needed files:

- `EvoNN-Prism/src/prism/benchmarks/datasets.py`
- `EvoNN-Prism/src/prism/benchmarks/parity.py`

Prism already has the right shape for this change.

### Topograph

Change benchmark resolution to:

1. explicit env override
2. shared root folder
3. local repo fallback

Needed files:

- `EvoNN-Topograph/src/topograph/benchmarks/registry.py`
- `EvoNN-Topograph/src/topograph/benchmarks/parity.py`

Also update suite resolution to read from shared `suites/` first.

### Stratograph

Remove sibling-project probing as primary behavior.

Change benchmark resolution to:

1. explicit env override
2. shared root folder
3. optional legacy sibling fallback for a short migration window

Needed files:

- `EvoNN-Stratograph/src/stratograph/benchmarks/registry.py`
- `EvoNN-Stratograph/src/stratograph/benchmarks/parity.py`
- possibly `EvoNN-Stratograph/src/stratograph/benchmarks/datasets.py` if builtins should defer to shared catalog names

Main improvement here is removing implicit path-order shadowing.

### Contenders

Change benchmark resolution to:

1. explicit env override
2. shared root folder
3. local repo fallback

Needed files:

- `EvoNN-Contenders/src/evonn_contenders/benchmarks/registry.py`
- `EvoNN-Contenders/src/evonn_contenders/benchmarks/parity.py`
- possibly `EvoNN-Contenders/src/evonn_contenders/benchmarks/datasets.py`

### Compare

Compare must stop importing Topograph as its benchmark loader.

Needed file:

- `EvoNN-Compare/src/evonn_compare/hybrid/benchmarks.py`

Best change:

- add a small compare-local loader that reads the shared catalog directly
- or import from a later shared benchmark helper module

Do not leave Compare dependent on `topograph.benchmarks.parity.get_benchmark`, or the extraction is incomplete in practice.

## Migration plan

### Phase 0: Freeze and diff

1. Freeze current benchmark catalogs.
2. Treat Topograph catalog as baseline.
3. Diff Prism shared-name YAMLs against Topograph YAMLs.
4. Record every intentional keep/change decision in `shared-benchmarks/migration/`.

Exit criteria:

- every shared benchmark name has one agreed YAML
- no silent drift remains between the planned shared file and current project copies

### Phase 1: Create shared folder

1. Create `shared-benchmarks/` in repo root.
2. Copy baseline catalog from Topograph into `shared-benchmarks/catalog/`.
3. Copy benchmark suites from Topograph into `shared-benchmarks/suites/`.
4. Add a short `README.md` that says this folder is benchmark source of truth.

Exit criteria:

- root folder exists
- catalog and suites exist in one place

### Phase 2: Add path resolution to all consumers

Add one env var pattern everywhere:

- `EVONN_SHARED_BENCHMARKS_DIR=/abs/path/to/shared-benchmarks`

Recommended resolution logic:

1. explicit per-project override if it already exists
2. shared superproject override
3. shared default path from repo root
4. local legacy fallback

Examples:

- Prism keeps `PRISM_CATALOG_DIR`, but should also understand shared default
- Topograph should gain shared-root support
- Stratograph should stop assuming sibling folders are normal

Exit criteria:

- each project can load `moons`, `digits`, `tiny_lm_synthetic`, `tinystories_lm`, `wikitext2_lm` from shared path only

### Phase 3: Switch defaults to shared

1. Make shared folder the default path in all projects.
2. Keep local fallbacks temporarily.
3. Run smoke tests and one parity-pack run per project.

Exit criteria:

- shared path is default everywhere
- local copies are not needed for normal runs
- no project relies on sibling-repo probing in normal mode

### Phase 4: Decouple Compare

1. Replace Topograph import inside `EvoNN-Compare`.
2. Make Compare read shared benchmark definitions directly.
3. Keep LM bridge handling, but point it at shared benchmark metadata.

Exit criteria:

- Compare can load parity-pack benchmarks with Topograph absent

### Phase 5: Remove duplicated project-local benchmark data

Only after all consumers pass with shared defaults:

1. delete duplicated project-local catalog YAMLs
2. delete duplicated project-local suite YAMLs
3. leave temporary compatibility env vars in place

Exit criteria:

- one benchmark YAML location remains
- no project needs repo-local benchmark copies

## Normalization rules for the shared catalog

Use one superset YAML schema that all current loaders can tolerate.

Keep these fields where useful:

- `name`
- `task`
- `source`
- `dataset`
- `input_dim`
- `num_classes`
- `n_samples`
- `noise`
- `factor`
- `centers`
- `cluster_std`
- `n_informative`
- `n_redundant`
- `max_train_samples`
- `max_val_samples`
- `max_test_samples`
- optional metadata:
  - `modality`
  - `description`
  - `domain`
  - `tags`

Reason:

- Prism benefits from `modality`
- Topograph/Contenders already carry `description/domain/tags`
- Stratograph and others can ignore extra fields

So the shared schema should be a tolerant superset, not the smallest common denominator.

## Important non-goals for phase 1

Do not mix these into the first extraction:

- parity pack centralization
- canonical ID map centralization
- LM cache file relocation
- full shared benchmark Python package
- hardware target config extraction

Those can come later. If mixed in now, the migration gets much riskier than necessary.

## Risks

### Risk 1: Same benchmark name, different intended difficulty

Example already visible:

- `moons.yaml` differs between Prism and Topograph

Mitigation:

- explicitly choose canonical parameters
- record the decision in migration notes
- do not silently overwrite

### Risk 2: LM benchmark semantics drift

Example already visible:

- Prism and Topograph disagree on LM `num_classes` and sample caps

Mitigation:

- normalize LM YAMLs first
- test all LM consumers after switch

### Risk 3: Stratograph shadow-order bugs

Current sibling-folder probing can hide conflicts.

Mitigation:

- switch Stratograph to one shared folder first
- keep sibling probing only as short-lived fallback

### Risk 4: Compare still coupled after migration

If `EvoNN-Compare` still imports Topograph benchmark code, the extraction is only partial.

Mitigation:

- make Compare a required migration target, not optional cleanup

## Acceptance criteria

The extraction is successful when all of these are true:

1. one shared root folder holds benchmark YAML source of truth
2. Prism loads benchmarks from shared folder by default
3. Topograph loads benchmarks and suites from shared folder by default
4. Stratograph loads from shared folder without sibling probing in normal mode
5. Contenders loads from shared folder by default
6. Compare can load benchmark data without importing Topograph benchmark code
7. local project benchmark copies can be removed without breaking normal runs

## Recommended first implementation slice

Smallest safe first slice:

1. create `shared-benchmarks/`
2. copy Topograph catalog there
3. copy Topograph suites there
4. patch Prism, Topograph, Stratograph, Contenders path resolution to support shared-root-first loading
5. leave local copies in place
6. run smoke benchmark listing in all projects

That slice gives immediate value with low risk and creates the real source-of-truth boundary before deeper cleanup.
