# LM5 Direct Tiny

Tiny direct Prism run for the 5 language-modeling benchmarks:

- `tiny_lm_synthetic`
- `tinystories_lm`
- `wikitext2_lm`
- `tinystories_lm_smoke`
- `wikitext2_lm_smoke`

Run from project root:

```bash
uv run prism evolve -c configs/lm5_direct_tiny/config.yaml --run-dir runs/lm5_direct_tiny
```

Inspect/report:

```bash
uv run prism inspect runs/lm5_direct_tiny
uv run prism report runs/lm5_direct_tiny
```
