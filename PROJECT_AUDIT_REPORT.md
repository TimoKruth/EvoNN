# Evo Neural Nets — Project Status Report

Date: 2026-04-08

---

## Overview

Three evolutionary NAS projects connected through a protocol-level
comparison and hybrid experiment layer.

| Project | Role | Tests | Status |
|---------|------|:-:|---|
| **EvoNN** | Track A: family-based macro-NAS | 442 | Phase 1-4 complete, Phase 5 partial, Phase 6.1-6.6 + 7 in place |
| **EvoNN-2** | Track B: topology/speciation/QD search | 510 | Training Phases 1-3 + competitive features complete; B5 still open |
| **EvoNN-Symbiosis** | Comparison + hybrid layer | 111 | S1-S4 implemented, Observatory live, durability follow-up still partial |

---

## EvoNN (Track A)

17 model families, 102 benchmarks, 7 contender comparisons. The main
remaining Track A gap is still source-faithful NAS-Bench-360 data and a few
large-compute validation tasks.

Key deliverables:
- ✅ Symbiosis result export with canonical IDs and full artifact envelope
- ✅ 4 parity benchmark packs (tier1, tier2, tier3, full)
- ✅ Budget metadata with multi-fidelity and promotion screen flags
- ✅ summary.json for cross-system durability contract

## EvoNN-2 (Track B)

NEAT-style DAG evolution with speciation, mixed-precision, and QD search.

Key deliverables:
- ✅ Phase 1: Cosine LR, AdamW, LayerNorm, Kaiming init, gradient clipping
- ✅ Phase 2: Lamarckian weight inheritance with structural hashing
- ✅ Phase 3: Quality-Diversity (novelty search + MAP-Elites)
- ✅ Competitive features: multi-fidelity, residual mutation, undercovered bias
- ✅ S4-A: Per-benchmark elite archive
- ✅ Enhanced reporting: speciation diagnostics, topology diversity, DAG analysis
- ✅ SearchTelemetry export with multi-fidelity, residual count, bias config
- ✅ Full test suite green after MoE gradient-flow stabilization

## EvoNN-Symbiosis

Protocol-level comparison layer with 5 campaign modes and web dashboard.

Key deliverables:
- ✅ S1: Parity contracts, validation, ingest, comparison engine
- ✅ S2: Wilcoxon statistics, bootstrap CI, campaign orchestration, deep readers
- ✅ S3: A/B transfer framework, 65% friedman MSE improvement documented
- ✅ S4: Hybrid genome + compiler + engine, per-benchmark elite archive
- ✅ Observatory dashboard (FastAPI + Chart.js) at port 8417
- ✅ 5 campaign modes: solo, comparison, hybrid, symbiosis, exploration
- ✅ Hybrid durability: DuckDB store, state.json, resume, checkpoints
- ⚠️ Cross-system summary / savepoint normalization is still not fully complete

## Campaign Data

Many campaign runs completed across all tiers and budgets:
- Tier 1: 35+ comparison cases (64/128/256/512/1008 budgets)
- Tier 2: 6 cases (EvoNN-leaning image tasks)
- Tier 3: 10 cases (EvoNN-2-leaning topology tasks)
- Hybrid: 30+ runs at various budgets
- QD tuning: 4 parameter sweep campaigns
- Verification: 5 campaigns (solo×2, comparison, hybrid, symbiosis)

## Key Findings

- EvoNN wins overall but EvoNN-2 dominates digits_image (80-100%) and credit_g (60%)
- Multi-fidelity + residual mutations improved EvoNN-2's regression by 65%
- QD search (novelty + MAP-Elites) increases diversity but doesn't flip overall comparison
- Hybrid shows promise on classification but search space needs QD/speciation for longer runs
- Higher budgets narrow the gap — first 4-4 tie at budget 512

## Next Steps

- Complete EvoNN per-run DuckDB migration (Phase 3 of durability sync)
- Run QD-enabled campaigns with per-benchmark elite archive
- Explore hybrid with speciation to address search space convergence
- Tier 2 campaign expansion with image dataset caching
