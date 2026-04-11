# Topograph - Design Spec

## Overview

Topograph is a rewrite of EvoNN-2 as a new sibling project. Same essence — NEAT topology evolution with QD, mixed-precision, MLX — but 52% smaller codebase, cleaner pipeline architecture, and two new features: adaptive mutation scheduling and cache-compatible parallelism.

**Target:** ~4,030 lines across 27 files (vs EvoNN-2: 8,314 lines, 60+ files)

## Decisions

- **Name:** Topograph (topology + graph)
- **Scope:** Full feature parity with EvoNN-2 + new features
- **Architecture:** Dataclass pipeline — state flows as typed records through composable stages
- **Deployment:** New git submodule `EvoNN-Topograph/` in superproject
- **Symbiosis:** Built-in from day one
- **Mutation scheduling:** Phase-based + adaptive operator selection
- **Parallelism:** Split cache lookup (serial) from training (parallel) — fixes EvoNN-2's incompatibility

## Module Layout

```
src/topograph/
├── __init__.py
├── __main__.py
├── cli.py                 # Typer commands (~200)
├── config.py              # All Pydantic models (~180)
├── genome/
│   ├── genes.py           # Gene types (~60)
│   ├── genome.py          # Genome + seed creation (~80)
│   └── codec.py           # JSON serialize/deserialize (~50)
├── pipeline/
│   ├── coordinator.py     # Evolution loop (~200)
│   ├── evaluate.py        # Fitness eval stage (~200)
│   ├── select.py          # Rank + NSGA-II (~100)
│   ├── reproduce.py       # Crossover + mutation dispatch (~150)
│   ├── archive.py         # Novelty + MAP-Elites + benchmark elites unified (~150)
│   └── schedule.py        # Phase-based + adaptive mutation rates (~150)
├── operators/
│   ├── mutate.py          # All 12 mutation ops (~250)
│   └── crossover.py       # Gene alignment crossover (~80)
├── nn/
│   ├── compiler.py        # Genome → MLX model (~300)
│   ├── layers.py          # BitLinear + QuantizedLinear + Hadamard (~80)
│   ├── moe.py             # Mixture of Experts (~100)
│   └── train.py           # Training loop + loss (~200)
├── benchmarks/
│   ├── spec.py            # BenchmarkSpec (~150)
│   ├── registry.py        # Dataset catalog (~120)
│   ├── preprocess.py      # Preprocessing (~80)
│   └── parity.py          # Symbiosis parity maps (~100)
├── storage.py             # DuckDB single file (~200)
├── export/
│   ├── symbiosis.py       # Symbiosis contract (~300)
│   └── report.py          # Markdown + topology analysis (~250)
├── cache.py               # Weight inheritance cache (~100)
├── parallel.py            # concurrent.futures pool (~100)
└── monitor.py             # Rich terminal output (~100)
```

## Pipeline Data Flow

One `GenerationState` dataclass flows through all stages per generation:

```python
@dataclass
class GenerationState:
    generation: int
    population: list[Genome]
    fitnesses: list[float]
    model_bytes: list[int]
    behaviors: list[ndarray]          # 8D topology vectors
    phase: EvolutionPhase             # EXPLORE | REFINE | POLISH
    mutation_profile: MutationProfile # adaptive rates
```

Coordinator loop:

```python
for gen in range(num_generations):
    state = evaluate(state, config, cache)
    state = score(state, config)
    state = archive(state, archives)
    state = select(state, config)
    state = reproduce(state, config, schedule)
    checkpoint(state, store)
```

Each function: takes state, returns new state. No side effects except checkpoint.

## Genome & Genes

Frozen Pydantic models, same 6 operator types as EvoNN-2.

Shorter field names:
- `innovation` (was `innovation_number`)
- `source`/`target` (was `source_innovation`/`target_innovation`)
- `layers` (was `layer_genes`)
- `connections` (was `connection_genes`)
- `operator` (was `operator_type`)

Key change: **mutation rates removed from Genome**. Lives in `MutationScheduler` at population level.

### LayerGene fields
innovation, width, activation, weight_bits, activation_bits, sparsity, order, enabled, operator, num_heads

### ConnectionGene fields
innovation, source, target, enabled

### Activation enum
RELU, SIGMOID, TANH, GELU, SILU

### OperatorType enum
DENSE, SPARSE_DENSE, RESIDUAL, ATTENTION_LITE, SPATIAL, TRANSFORMER_LITE

### WeightBits enum
TERNARY(2), INT4(4), INT8(8), FP16(16)

### ActivationBits enum
INT4(4), INT8(8), FP16(16)

### ConvLayerGene, ExpertGene, GateConfig
Kept for image tasks and MoE. Same structure as EvoNN-2.

## New Feature: Adaptive Mutation Scheduling

### Phases

| Phase | Generation Range | Strategy |
|-------|-----------------|----------|
| EXPLORE | 0-33% | High topology mutation, low precision |
| REFINE | 33-66% | Balanced, width tuning dominant |
| POLISH | 66-100% | Low topology, high precision/sparsity |

### Default Phase Profiles

| Operator | EXPLORE | REFINE | POLISH |
|----------|---------|--------|--------|
| add_layer | 0.3 | 0.1 | 0.02 |
| remove_layer | 0.2 | 0.1 | 0.02 |
| add_connection | 0.25 | 0.15 | 0.05 |
| width | 0.15 | 0.35 | 0.15 |
| activation | 0.1 | 0.15 | 0.05 |
| weight_bits | 0.02 | 0.1 | 0.3 |
| sparsity | 0.02 | 0.1 | 0.25 |
| operator_type | 0.15 | 0.1 | 0.03 |

### Adaptive Mechanism

- Track per-operator success: `did_fitness_improve_after_this_mutation?`
- Exponential moving average (EMA), window ~10 generations
- Scale operator probability proportional to success EMA
- Phase sets min/max bounds, adaptation picks within

Replaces EvoNN-2's undirected Gaussian drift on per-genome mutation rates.

## New Feature: Cache-Compatible Parallelism

EvoNN-2 problem: weight cache and parallel evaluation are mutually exclusive.

Fix: separate cache lookup (fast, serial, needs shared state) from training (slow, parallel, independent).

```python
class ParallelEvaluator:
    def evaluate_batch(self, genomes, compile_fn, train_fn, cache=None):
        # Step 1: serial — cache lookup + model compilation (fast)
        models = [compile_fn(g, cache) for g in genomes]
        # Step 2: parallel — training (slow, independent)
        with ProcessPoolExecutor(max_workers) as pool:
            futures = [pool.submit(train_fn, m) for m in models]
            return [f.result() for f in futures]
```

## NN Compilation

Same approach as EvoNN-2. `compile_genome()` → `EvolvedModel(nn.Module)`.

- Reachability filtering at init (forward walk from INPUT)
- Pre-computed routing table
- 6 operator dispatch (DENSE, SPARSE_DENSE, RESIDUAL, ATTENTION_LITE, SPATIAL, TRANSFORMER_LITE)
- Precision-aware projections (BitLinear/QuantizedLinear/nn.Linear)
- Fake quantization for activations
- Optional LayerNorm, MoE routing

### layers.py
Three layer types in one file (EvoNN-2 spreads across 3):
- QuantizedLinear — STE fake quantization (INT4/INT8)
- BitLinear — Ternary weights
- hadamard_smooth — Hadamard transform for activation quantization

## Training

Same as EvoNN-2:
- AdamW + cosine LR schedule
- Gradient clipping (L2 norm)
- Loss: MSE (regression), cross-entropy (classification)
- Early stopping: divergence (>10.0) + plateau detection
- Multi-benchmark percentile aggregation

## Storage

Single-file `storage.py` with DuckDB. Per-run DB only.

5 tables: runs, genomes, innovation_counters, budget_metadata, benchmark_results.

Same schema as EvoNN-2, no global DB legacy.

## Benchmarks

Reuse EvoNN-2's 147 benchmark catalog YAMLs directly. Copy `benchmarks/` directory.

4 source files:
- `spec.py` — BenchmarkSpec with sklearn/csv/image/openml loading
- `registry.py` — Dataset catalog from YAML
- `preprocess.py` — StandardScaler pipeline
- `parity.py` — Canonical benchmark ID mapping for symbiosis

## Export

Two files instead of six (EvoNN-2: 2,018 lines → Topograph: ~550 lines).

### symbiosis.py (~300 lines)
- `export_symbiosis_contract()` — manifest.json + results.json
- Config snapshot, genome summary, model summary, dataset manifest
- Canonical benchmark mapping
- Budget and search telemetry manifests
- Designed-in, not retrofitted

### report.py (~250 lines)
- Topology analysis: dag_depth, width_profile, skip_connections, bottlenecks, precision_distribution
- Population diversity stats
- Speciation diagnostics
- Markdown report generation

## CLI

Typer app with commands:

```
topograph benchmarks          # list available
topograph evolve              # run evolution (--config, --run-dir, --resume)
topograph report <run>        # markdown report
topograph inspect <run>       # metrics summary
topograph export <run>        # JSON export
topograph symbiosis export    # symbiosis contract
topograph suite baselines     # run baselines
topograph suite list          # list datasets
```

Dropped from EvoNN-2: `analyze` (merged into `report`), `transfer`, `ensemble`. Can add back if needed.

## Monitor

Rich terminal only. No FastAPI dashboard server.

Subscribes to generation events, prints Rich table with: generation, best/avg/worst fitness, phase, active operators, archive fill %, elapsed time.

## Size Budget

| Module | Files | Lines |
|--------|-------|-------|
| cli | 1 | 200 |
| config | 1 | 180 |
| genome | 3 | 190 |
| pipeline | 6 | 950 |
| operators | 2 | 330 |
| nn | 4 | 680 |
| benchmarks | 4 | 450 |
| storage | 1 | 200 |
| export | 2 | 550 |
| cache | 1 | 100 |
| parallel | 1 | 100 |
| monitor | 1 | 100 |
| **Total** | **27** | **~4,030** |

## Dependencies

Same as EvoNN-2 minus FastAPI/uvicorn:
- mlx >= 0.31.0
- numpy >= 2.2.0
- pydantic >= 2.10.0
- pyyaml >= 6.0.2
- scikit-learn >= 1.6.1
- typer >= 0.15.1
- rich >= 13.9.0
- duckdb >= 1.1.3

Optional: pytest, ruff (dev), matplotlib (viz), openml/pandas (data), xgboost/lightgbm (baselines), scipy (stats)

## What's NOT Changing

- All 6 operator types
- All 5 activation functions
- All precision levels (TERNARY/INT4/INT8/FP16)
- NEAT innovation tracking
- Gene alignment crossover
- All 12 mutation operators
- MAP-Elites (1296 niches)
- Novelty search (KNN distance)
- NSGA-II multi-objective
- Weight inheritance / Lamarckian evolution
- Multi-fidelity training schedule
- Benchmark pool with undercovered bias
- Per-benchmark elite archive
- Quantization scheduling
- Self-adaptive hyperparameters (LR, batch size per genome)
