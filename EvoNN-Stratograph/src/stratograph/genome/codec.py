"""Genome serialization helpers."""

from __future__ import annotations

import hashlib
import json
from typing import Any

from stratograph.genome.models import HierarchicalGenome


def genome_to_dict(genome: HierarchicalGenome) -> dict[str, Any]:
    """Serialize genome into JSON-safe dict."""
    return genome.model_dump(mode="json")


def dict_to_genome(payload: dict[str, Any]) -> HierarchicalGenome:
    """Parse genome from serialized dict."""
    return HierarchicalGenome.model_validate(payload)


def genome_digest(genome: HierarchicalGenome) -> str:
    """Stable digest for deterministic compiler params."""
    encoded = json.dumps(genome_to_dict(genome), sort_keys=True, separators=(",", ":")).encode("utf-8")
    return hashlib.sha256(encoded).hexdigest()[:16]
