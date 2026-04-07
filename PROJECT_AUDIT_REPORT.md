# Evo Neural Nets — Project Status Report

Date: 2026-04-07

---

## Overview

Three evolutionary NAS projects connected through a protocol-level
comparison and hybrid experiment layer.

| Project | Role | Tests | Status |
|---------|------|:-:|---|
| **EvoNN** | Track A: family-based macro-NAS | 417+ | Phase 7 complete |
| **EvoNN-2** | Track B: topology/speciation/QD search | 460+ | All training phases + competitive features complete |
| **EvoNN-Symbiosis** | Comparison + hybrid layer | 98 | S1-S4 complete, Observatory dashboard live |

---

## EvoNN (Track A)

17 model families, 102 benchmarks, 7 contender comparisons. Phases 1-7 complete.

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

## Campaign Data

~100 campaign runs completed across all tiers and budgets:
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
