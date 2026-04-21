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
- add regression coverage for runtime metadata propagation across run artifacts
