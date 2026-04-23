# Changelog

## Unreleased

- make Primordia MLX-first but installable on non-Darwin hosts by marking `mlx`
  as a platform-specific dependency
- carry runtime backend/version metadata through run summaries, trial records,
  reports, and compare exports instead of hardcoding `mlx`
- carry primitive usage, benchmark-group coverage, failure count, and wall-clock
  telemetry through Primordia reports and compare exports
- add `primitive_bank_summary.json` so primitive-family winners can be reused in
  later transfer/seeding analysis
- emit `primitive_bank_summary.json` during Primordia runs (not only compare
  export) and include primitive-bank winners in regenerated markdown reports
- add regression coverage for runtime metadata propagation across run artifacts
- improve Primordia CLI parity with richer `inspect` output for runtime, usage,
  primitive-bank wins, and best benchmark summaries
- suppress the default overview banner when invoking Primordia subcommands
- update the landing overview to reflect that primitive-bank export is already
  available
