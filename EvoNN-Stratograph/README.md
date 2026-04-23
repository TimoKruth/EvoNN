# EvoNN-Stratograph

Hierarchy-first evolutionary neural architecture search.

Stratograph is a greenfield sibling to Prism and Topograph. Its core idea is
not family selection and not flat graph search. It evolves a macro graph of
cell instances plus a reusable library of micro-graphs that define the inner
structure of each cell.

Current state:
- Fresh project scaffold implemented
- Hierarchical genome + codec implemented
- Hierarchical mutation/crossover search loop implemented
- MLX-backed hierarchical compiler implemented
- MLX-backed evaluator implemented for tabular, image, and LM packs
- Lightweight runtime maturity added: stronger heads, LM bucket/trigram scorer, SGD inheritance
- Ablation runner implemented to test flat vs unshared vs shared hierarchy
- Motif mining implemented for repeated winning sub-cell structures
- Runtime maturity improved with resume, checkpoint, status, best-genome artifacts, and failure-pattern summaries in inspect/report
- Compare-compatible export and startup formats implemented
- Benchmark loading + parity/export boundary implemented
- Runtime backend/version metadata carried through compare exports
- Full long-horizon hierarchy-specialized trainer optimization still pending

Core docs:
- `VISION.md`
- `IMPLEMENTATION_PLAN.md`
- `ARCHITECTURE_RULES.md`
- `RESEARCH_NOTES.md`

CLI:
- `uv run stratograph benchmarks`
- `uv run stratograph warm-cache --config configs/ablation_matrix_smoke.yaml`
- `uv run stratograph list-lm-caches`
- `uv run stratograph evolve -c configs/working_33_plus_5_lm_smoke.yaml --run-dir runs/demo`
- `uv run stratograph report runs/demo`
- `uv run stratograph inspect runs/demo`
- `uv run stratograph symbiosis export runs/demo --pack /path/to/pack.yaml`
- `uv run stratograph ladder --workspace manual_compare_runs/execution_ladder_20260417`
- `uv run stratograph ablate --config configs/two_level_value_probe.yaml --workspace manual_compare_runs/two_level_value_probe_20260417`
- `uv run stratograph ablate-matrix --config configs/ablation_matrix_smoke.yaml --workspace manual_compare_runs/ablation_matrix_20260417_g2`
- `uv run stratograph ablate --config configs/ablation_image_hard.yaml --workspace manual_compare_runs/ablation_image_hard_20260417`
- `uv run stratograph ablate --config configs/ablation_openml_structured.yaml --workspace manual_compare_runs/ablation_openml_structured_20260417`
- `uv run stratograph ablate --config configs/ablation_lm_full.yaml --workspace manual_compare_runs/ablation_lm_full_20260417`
- `uv run stratograph motifs analyze --run-dir manual_compare_runs/two_level_value_probe_20260417/runs/two_level_value_probe__two_level_shared`
