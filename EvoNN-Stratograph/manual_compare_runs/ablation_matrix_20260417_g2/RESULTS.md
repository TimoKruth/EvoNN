# Stratograph Hierarchy Ablation Matrix

Study date: `2026-04-17`

Variants:

- `flat_macro`
- `two_level_unshared`
- `two_level_shared`
- `two_level_shared_no_clone`
- `two_level_shared_no_motif_bias`

Packs:

- `tabular_local` (`6`)
- `image_smoke` (`3`)
- `lm_smoke` (`3`)
- `mixed_38_smoke` (`38`, `34` ok because LM caches missing)

## Main Findings

`two_level_shared` is already better than `two_level_unshared`, but not yet better than `flat_macro`.

Global pairwise:

- shared vs flat: `8` wins, `29` losses, `7` ties
- shared vs unshared: `26` wins, `11` losses, `7` ties
- shared vs no-clone: `5` wins, `4` losses, `35` ties
- shared vs no-motif-bias: `21` wins, `13` losses, `10` ties

## Interpretation

- Reuse/shared hierarchy helps compared with deeper-but-unshared hierarchy.
- Current shared design still loses too often to flat macro baselines, especially on image and many broad smoke tasks.
- Motif bias appears useful overall:
  - shared beats no-motif-bias globally `21-13`
  - shared beats no-motif-bias on `image_smoke` and `mixed_38_smoke`
- Clone mutation currently matters only weakly:
  - shared vs no-clone globally `5-4-35`
  - signal exists, but much smaller than shared-vs-unshared or shared-vs-no-motif-bias

## Pack-Level Read

- `tabular_local`:
  - shared beats unshared
  - shared also beats flat
  - but no-clone and no-motif-bias variants sometimes match or exceed shared
- `image_smoke`:
  - shared beats unshared
  - flat still strongest
  - motif bias helps
- `lm_smoke`:
  - only `tiny_lm_synthetic` ran
  - flat strongest
  - cached LM benchmarks failed due missing local caches
- `mixed_38_smoke`:
  - shared beats unshared
  - shared still loses to flat overall
  - shared beats both no-clone and no-motif-bias

## Important Caveat

Missing LM caches:

- `tinystories_lm`
- `tinystories_lm_smoke`
- `wikitext2_lm`
- `wikitext2_lm_smoke`

So mixed pack results reflect `34` completed benchmarks, not full `38`.

## Artifacts

- `matrix_report.md`
- `tabular_local/ablation_report.md`
- `image_smoke/ablation_report.md`
- `lm_smoke/ablation_report.md`
- `mixed_38_smoke/ablation_report.md`
