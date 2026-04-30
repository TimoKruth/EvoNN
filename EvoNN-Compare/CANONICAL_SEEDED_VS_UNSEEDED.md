# Canonical Seeded vs Unseeded Compare

The canonical seeded control artifact path is `Primordia -> Topograph` on the
portable compare boundary.

Use the dedicated compare command from the repo root:

```bash
uv run --package evonn-compare evonn-compare seeded-compare \
  --workspace .tmp/evo38-seeded-vs-unseeded \
  --pack tier1_core_smoke \
  --seed 42 \
  --open
```

What this command does:

- generates a budget-stamped compare pack in the workspace
- runs Primordia once to materialize `seed_candidates.json`
- runs Topograph twice on the same pack/budget/seed:
  - `01-unseeded`
  - `02-seeded`, with `benchmark_pool.primordia_seed_candidates_path` pointed
    at the exported Primordia seed artifact
- writes a direct seeded-vs-unseeded markdown/JSON summary
- writes fair-matrix-compatible case summaries so the workspace trend report and
  dashboard can expose seed mode/source/artifact provenance
- marks the lane as portable seeding-contract evidence rather than native MLX
  transfer proof
- opens the canonical dashboard immediately when `--open` is supplied, so the
  seeded control and transfer lane can be reviewed from the same recurring
  evidence surface used by fair-matrix workspaces

Key outputs:

- `reports/seeded_vs_unseeded_summary.md`
- `reports/seeded_vs_unseeded_summary.json`
- `trends/fair_matrix_trends.md`
- `trends/fair_matrix_trends.json`
- `fair_matrix_dashboard.html`
- `fair_matrix_dashboard.json`

The command intentionally uses the portable Topograph exporter so the canonical
artifact set is reproducible on the supported host/runtime boundary without
depending on native MLX availability.

That also means this workspace proves portable seeding/export plumbing only. It
does not prove native MLX transfer behavior for Topograph.
