#!/usr/bin/env python3
"""Delete old run and campaign directories across the Evo Neural Nets superproject.

Default behavior is a dry run. Pass ``--apply`` to actually delete candidates.
"""

from __future__ import annotations

import argparse
import shutil
import sys
from dataclasses import dataclass
from datetime import datetime, timedelta
from pathlib import Path


@dataclass(frozen=True)
class CleanupTarget:
    name: str
    root: Path


def _format_size(num_bytes: int) -> str:
    units = ("B", "KB", "MB", "GB", "TB")
    value = float(num_bytes)
    for unit in units:
        if value < 1024.0 or unit == units[-1]:
            return f"{value:.1f}{unit}"
        value /= 1024.0
    return f"{num_bytes}B"


def _directory_size(path: Path) -> int:
    total = 0
    for child in path.rglob("*"):
        try:
            if child.is_file():
                total += child.stat().st_size
        except FileNotFoundError:
            continue
    return total


def _iter_candidates(target: CleanupTarget, cutoff: datetime) -> list[tuple[Path, datetime, int]]:
    candidates: list[tuple[Path, datetime, int]] = []
    if not target.root.exists():
        return candidates
    for child in sorted(target.root.iterdir()):
        if child.name.startswith(".") or not child.is_dir() or child.is_symlink():
            continue
        try:
            mtime = datetime.fromtimestamp(child.stat().st_mtime)
        except FileNotFoundError:
            continue
        if mtime >= cutoff:
            continue
        candidates.append((child, mtime, _directory_size(child)))
    return candidates


def _build_targets(repo_root: Path) -> list[CleanupTarget]:
    return [
        CleanupTarget("EvoNN runs", repo_root / "EvoNN" / "runs"),
        CleanupTarget("EvoNN-2 runs", repo_root / "EvoNN-2" / "runs"),
        CleanupTarget("Symbiosis campaigns", repo_root / "EvoNN-Symbiosis" / "campaigns"),
    ]


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(
        description="Clean old run and campaign directories across all three projects."
    )
    parser.add_argument(
        "--days",
        type=int,
        default=5,
        help="Keep directories modified within the last N days. Default: 5.",
    )
    parser.add_argument(
        "--apply",
        action="store_true",
        help="Actually delete old directories. Default is dry-run.",
    )
    parser.add_argument(
        "--repo-root",
        type=Path,
        default=Path(__file__).resolve().parents[1],
        help="Path to the Evo Neural Nets superproject root.",
    )
    args = parser.parse_args(argv)

    if args.days < 0:
        parser.error("--days must be >= 0")

    repo_root = args.repo_root.resolve()
    cutoff = datetime.now() - timedelta(days=args.days)
    targets = _build_targets(repo_root)

    grand_total_size = 0
    grand_total_count = 0

    mode = "APPLY" if args.apply else "DRY-RUN"
    print(f"[cleanup] mode={mode} repo_root={repo_root}")
    print(f"[cleanup] keeping directories newer than {cutoff.isoformat(timespec='seconds')}")

    for target in targets:
        candidates = _iter_candidates(target, cutoff)
        target_size = sum(size for _, _, size in candidates)
        grand_total_size += target_size
        grand_total_count += len(candidates)

        print(f"\n[{target.name}] root={target.root}")
        if not target.root.exists():
            print("  missing; skipped")
            continue
        if not candidates:
            print("  nothing to delete")
            continue
        for path, mtime, size in candidates:
            print(
                f"  {'delete' if args.apply else 'would delete'} "
                f"{path}  mtime={mtime.isoformat(timespec='seconds')}  size={_format_size(size)}"
            )
            if args.apply:
                shutil.rmtree(path)
        print(f"  total={len(candidates)} dirs  reclaimed={_format_size(target_size)}")

    print(
        f"\n[cleanup] {'deleted' if args.apply else 'would delete'} "
        f"{grand_total_count} directories totalling {_format_size(grand_total_size)}"
    )
    return 0


if __name__ == "__main__":
    sys.exit(main())
