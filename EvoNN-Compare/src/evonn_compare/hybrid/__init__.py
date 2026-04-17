"""Family-aware topology hybrid primitives."""

from evonn_compare.hybrid.genome import (
    HybridConnectionGene,
    HybridFamily,
    HybridGenome,
    HybridNodeGene,
)
from evonn_compare.hybrid.mutation import mutate

__all__ = [
    "HybridConnectionGene",
    "HybridFamily",
    "HybridGenome",
    "HybridNodeGene",
    "mutate",
]


def compile_hybrid(*args, **kwargs):
    from evonn_compare.hybrid.compiler import compile_hybrid as _compile_hybrid

    return _compile_hybrid(*args, **kwargs)
