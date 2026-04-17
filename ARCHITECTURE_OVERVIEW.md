# Architecture Overview: Prism, Topograph, Stratograph

This document summarizes the architecture of the three newer EvoNN sibling systems in this superproject. Each system attacks neural architecture search from a different primary axis:

- Prism: choose and tune a model family
- Topograph: evolve a flat topology graph
- Stratograph: evolve a hierarchy of reusable cells plus a macro graph

The descriptions below are based on the current source code in this repository. If a document and the code disagree, the code should be treated as truth.

## Prism

### Core idea

Prism is a family-based evolutionary neural architecture search system. Its main question is not "what exact graph should exist?" but "which model family should solve this task, and how should that family be parameterized?" A Prism genome therefore selects a family first, then evolves widths, depth, activations, dropout, residual usage, normalization, optimizer-facing knobs, and a few family-specific options.

This makes Prism the most opinionated of the three systems: it searches inside a curated menu of architectural templates instead of inventing arbitrary graphs.

### Architectural unit of search

Prism evolves immutable `ModelGenome` records. A genome contains:

- `family`
- `hidden_layers`
- `activation`
- `dropout`
- `residual`
- `activation_sparsity`
- `learning_rate`
- `kernel_size`
- `embedding_dim`
- `num_heads`
- `norm_type`
- `weight_decay`
- `num_experts`
- `moe_top_k`

So the search space is "family choice + family parameters", not free-form connectivity.

### Model families

Prism compiles genomes into one of a fixed set of MLX model families:

- Tabular: `FlexMLP`, `SparseMLP`, `MoEMLP`
- Image: `ImageConvNet`, `LiteImageConvNet`
- Sequence: `SequenceConvNet`, `LiteSequenceConvNet`, `SequenceGRUNet`
- Text/sequence: `TextEmbeddingModel`, `AttentionEncoderNet`, `SparseAttentionNet`

The compiler enforces family/modality compatibility. For example, image convolution families are only valid for image tasks, while language modeling is restricted to embedding/attention families.

### Runtime architecture

Prism runtime has four main layers:

1. Genome layer
- immutable genome definition and mutation/crossover logic

2. Family layer
- concrete MLX `nn.Module` implementations for each family
- compiler that validates family/task/modality and instantiates the module

3. Pipeline layer
- generation state
- benchmark selection
- evaluation
- archive building
- reproduction

4. Run boundary
- DuckDB metrics store
- checkpoint files
- markdown/symbiosis export

### Evolution loop

Prism uses a straightforward generational pipeline:

1. Create seed population, ensuring family diversity.
2. Select benchmarks, with extra focus on undercovered benchmarks.
3. Compile each genome to an MLX model.
4. Optionally transfer inherited weights from parent genomes.
5. Train and evaluate on selected benchmarks.
6. Build three archive views:
- per-benchmark elites
- Pareto front on quality vs parameter count
- family-level niche archive
7. Reproduce through tournament selection, crossover, and single-step mutation.
8. Checkpoint state and persist metrics.

The archive design is important: Prism keeps a family-oriented notion of diversity even though it does not search arbitrary graphs.

### Training and evaluation

Prism uses real MLX training:

- AdamW
- cosine or constant LR schedule
- gradient clipping
- early stopping
- multi-fidelity training schedule across generations
- optional weight inheritance cache

Quality is task-dependent:

- classification: accuracy
- regression: negative MSE as search quality
- language modeling: negative perplexity as search quality

### What Prism is structurally best at

Prism is strongest when the search problem benefits from choosing among a small number of known architectural regimes. Its bias is:

- higher-level family selection over graph invention
- simpler genome representation
- direct use of mature MLX model code
- easier modality-aware constraints

The tradeoff is that it cannot express the same topology novelty as Topograph or the same reuse hierarchy as Stratograph.

## Topograph

### Core idea

Topograph is topology-first evolutionary neural architecture search. Its main question is: "what flat DAG of learned operators should solve this task?" It is intentionally closer to NEAT-style topology evolution than Prism, but with mixed precision, quantization-aware operators, benchmark pooling, and a cleaner staged pipeline.

Compared with Prism, Topograph moves the search object from family templates to graph structure. Compared with Stratograph, it stays flat: one graph, one level.

### Architectural unit of search

Topograph evolves a `Genome` made of gene collections:

- `LayerGene`
- `ConnectionGene`
- optional `ConvLayerGene`
- optional expert genes and `GateConfig`

Layer genes encode:

- width
- activation
- weight precision
- activation precision
- sparsity
- topological order
- operator type
- attention head count

Connection genes encode graph wiring through innovation numbers and source/target references. This gives Topograph a classic topology-evolution backbone with modern operator metadata attached to each node.

### Operator vocabulary

Topograph layer operators include:

- `dense`
- `sparse_dense`
- `residual`
- `attention_lite`
- `spatial`
- `transformer_lite`

This means the graph is not just a dense DAG. Each node can represent a different computation style, and precision/sparsity are part of the architecture itself.

### Runtime architecture

Topograph is built as a typed staged pipeline. The main state object for each generation carries:

- generation number
- population
- fitnesses
- model byte estimates
- behavior descriptors
- benchmark results
- raw losses
- current phase
- total evaluations

The overall module structure is:

1. Genome
- genes, genome, serialization

2. Operators
- mutation and crossover over graph structure

3. NN runtime
- genome compiler
- quantized/ternary layers
- MoE support
- MLX training loop

4. Pipeline
- evaluate
- score
- archive
- select
- reproduce
- mutation scheduling

5. Run boundary
- cache
- parallel evaluator
- DuckDB storage
- report/export

### Compilation model

Topograph compiles a genome into an MLX `EvolvedModel`. The compiler:

- sorts layers by topological order
- filters to reachable graph regions
- builds per-connection projection modules
- attaches operator-specific modules
- adds optional LayerNorm
- optionally attaches a mixture-of-experts head
- precomputes routing tables for execution

The forward pass is graph-driven rather than family-template-driven. Incoming projections are merged at each node, then node-specific operator logic is applied.

### Evolution loop

Topograph runs a fuller evolutionary loop than Prism:

1. Seed initial topology population.
2. Evaluate genomes on one benchmark or a sampled benchmark pool.
3. Score fitness, optionally blending task fitness with novelty.
4. Update archives:
- novelty archive
- MAP-Elites archive
- per-benchmark elite archive
5. Adapt mutation statistics from observed outcomes.
6. Reproduce into the next population.
7. Persist state, scheduler state, archives, and innovation counters.

This makes Topograph the most "evolution-system-heavy" of the three in the current codebase.

### Adaptive mutation scheduling

One of Topograph's defining architectural choices is phase-based adaptive mutation scheduling:

- `explore`
- `refine`
- `polish`

Each phase changes operator probabilities, and each operator's probability is then scaled by an EMA of observed success. So mutation policy is partly hand-shaped and partly learned during the run.

This is a notable difference from Prism, where mutation is simpler and family-centric.

### Parallel and cache architecture

Topograph explicitly separates:

- serial cache-sensitive work
- parallel expensive training work

That split allows weight inheritance and parallel execution to coexist. It also includes a memory-aware process-pool evaluator that estimates worker count conservatively from data size, weight snapshot size, CPU budget, and system memory.

### Training and evaluation

Topograph uses real MLX training and treats architecture details as first-class runtime concerns:

- mixed precision / quantized weights
- fake-quantized activations
- sparsity
- optional MoE
- optional benchmark-pool percentile aggregation
- model byte accounting
- reusable evaluation memoization

### What Topograph is structurally best at

Topograph is designed for open-ended flat-graph search with richer topology freedom than Prism. Its bias is:

- graph evolution over family selection
- operator-level topology diversity
- explicit quantization/sparsity in the genome
- stronger novelty/QD machinery
- more sophisticated mutation control

The tradeoff is complexity. It has a heavier runtime and more moving parts than Prism, and it does not express hierarchy as directly as Stratograph.

## Stratograph

### Core idea

Stratograph is hierarchy-first evolutionary architecture search. Its main question is: "what hierarchical structure of reusable cells should solve this task?" Instead of searching only a family or only a flat graph, it searches two coupled levels:

- a macro DAG of cell instances
- a reusable cell library of micro-graphs

This gives Stratograph a different hypothesis from both Prism and Topograph: good architectures may be modular systems built from repeated and gradually specialized motifs.

### Architectural unit of search

Stratograph evolves a `HierarchicalGenome` with:

- macro nodes
- macro edges
- a cell library

Each cell contains:

- `CellNodeGene`
- `CellEdgeGene`

Each macro node references a cell by `cell_id`, which means multiple macro nodes can reuse the same micro-graph. Evolution can then:

- keep cells shared
- clone a shared cell
- specialize a clone
- add macro structure around existing cells

This is the main architectural distinction of Stratograph.

### Genome invariants

Stratograph validates both levels aggressively:

- no duplicate macro node ids
- no duplicate cell node ids inside a cell
- every macro node must reference a real cell
- macro graph must be acyclic
- cell graphs must be acyclic
- macro graph must reach output

It also exposes hierarchy-specific structural metrics directly on the genome:

- macro depth
- average cell depth
- reuse ratio

Those metrics are later used for reporting and evaluation heuristics.

### Runtime architecture

Current Stratograph architecture is intentionally distinct from Topograph, but it is also still a prototype compared with Prism and Topograph.

The main runtime layers are:

1. Hierarchical genome layer
- macro graph + cell library models
- codec and digesting

2. Hierarchical compiler
- compile each cell independently
- compile macro graph over compiled cells
- deterministic NumPy execution path

3. Search layer
- hierarchy-aware mutation
- hierarchy-aware crossover
- novelty descriptor and niche key

4. Evaluation layer
- fast evaluator for tabular, image, and language-modeling tasks
- compare-compatible outputs

5. Run boundary
- DuckDB storage
- report/export
- ladder workflow for compare validation

### Compilation model

Stratograph compiles a genome into a `CompiledHierarchy` made from shared `CompiledCell` objects.

The compiler:

- topologically orders nodes inside each cell
- merges multiple inputs by deterministic projection and averaging
- applies primitive activations
- reuses compiled cells when several macro nodes point at the same cell
- runs the macro DAG over those compiled cell executors

The current compiler is deterministic NumPy-based, not a full MLX trainer/compiler stack. This is deliberate for the current phase: prove hierarchy semantics and compare-surface compatibility first.

### Search loop

Stratograph already has hierarchy-aware mutations and crossover. Current mutation modes include:

- width changes
- activation changes
- cloning a shared cell
- adding a macro node
- specializing a shared cell

Crossover takes macro segments from both parents, rebuilds a child hierarchy, then mutates it again.

This means Stratograph search is not just "Topograph with nested blocks". Its operators directly manipulate sharing and specialization.

### Evaluation approach

Current Stratograph evaluation is a fast proxy architecture:

- compile hierarchy
- encode inputs into learned-by-structure features
- for classification/image tasks, fit a lightweight classical head on encoded features
- for language modeling, use a structure-conditioned smoothed n-gram style evaluator

So Stratograph currently evaluates "how useful is this hierarchical representation?" rather than fully training a deep MLX model end to end.

That is why it can already run ladder comparisons across the shared benchmark lane while still not being feature-complete.

### Current maturity from source

Stratograph source already includes:

- a hierarchical genome and codec
- a deterministic compiler built on NumPy
- hierarchy-aware mutation and crossover
- a fast benchmark evaluator
- storage and export surfaces

What the source does not currently show in the same style as Prism/Topograph is a full MLX training runtime with weight-inheritance-based neural training. The architecture in code today is therefore best understood as a real search/evaluation prototype with a stable hierarchy model, not yet the same training stack as the other two systems.

### What Stratograph is structurally best at

Stratograph is designed for problems where reusable motifs, cell sharing, and gradual specialization may matter more than a single flat graph. Its bias is:

- explicit hierarchy
- reuse as a first-class search variable
- clone/specialize dynamics
- hierarchy-specific telemetry

The tradeoff is that the current runtime is still a prototype stack and does not yet offer the same end-to-end trained neural execution path as Prism and Topograph.

## Bottom line

The three systems are intentionally different reference architectures:

- Prism: family-first, simplest search object, strongest template bias
- Topograph: topology-first, richest flat-graph search, heaviest evolutionary runtime
- Stratograph: hierarchy-first, reusable-cell search, currently prototype runtime with real compare surface

If you want, I can next split this into three separate files as well:

- `PRISM_ARCHITECTURE.md`
- `TOPOGRAPH_ARCHITECTURE.md`
- `STRATOGRAPH_ARCHITECTURE.md`
