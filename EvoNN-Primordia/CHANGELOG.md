# Changelog

## Unreleased

- make Primordia MLX-first but installable on non-Darwin hosts by marking `mlx`
  as a platform-specific dependency
- carry runtime backend/version metadata through run summaries, trial records,
  reports, and compare exports instead of hardcoding `mlx`
- add regression coverage for runtime metadata propagation across run artifacts
