# Evo Neural Nets

Superproject for the three linked repositories:

- [EvoNN](/Users/timokruth/Projekte/Evo%20Neural%20Nets/EvoNN)
- [EvoNN-2](/Users/timokruth/Projekte/Evo%20Neural%20Nets/EvoNN-2)
- [EvoNN-Symbiosis](/Users/timokruth/Projekte/Evo%20Neural%20Nets/EvoNN-Symbiosis)

This root repository tracks those projects as Git submodules so one top-level
commit can point to a coherent multi-repo state.

## What Lives Here

- `EvoNN/` — Track A family-based evolutionary NAS
- `EvoNN-2/` — Track B topology-first evolutionary NAS
- `EvoNN-Symbiosis/` — comparison, campaign, transfer, and hybrid layer
- `OVERVIEW.md` — high-level cross-project summary
- `PROJECT_AUDIT_REPORT.md` — technical status snapshot

## Clone / Sync

Clone with submodules:

```bash
git clone --recurse-submodules <repo-url>
```

If already cloned:

```bash
git submodule update --init --recursive
```

To pull the latest superproject state plus the referenced submodule commits:

```bash
git pull
git submodule update --init --recursive
```

## Working With The Subrepos

Each project keeps its own history and can be worked on independently:

```bash
git -C EvoNN status
git -C EvoNN-2 status
git -C EvoNN-Symbiosis status
```

When a subrepo advances and you want the root repo to reference the new commit:

```bash
git add EvoNN EvoNN-2 EvoNN-Symbiosis
git commit -m "chore: update submodule refs"
```

## Current Verified Test Snapshot

- `EvoNN`: 442 passed
- `EvoNN-2`: 510 passed
- `EvoNN-Symbiosis`: 111 passed

See the subrepo readmes for full setup and project-specific workflows.
