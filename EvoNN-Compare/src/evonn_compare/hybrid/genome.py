"""Hybrid genome combining Topograph DAG topology with Prism family blocks.

Each node in the DAG is a macro block with a family type (mlp, conv2d,
attention, sparse_mlp) instead of a plain linear layer. The topology
(connections, skip connections, layer ordering) evolves via Topograph-style
mutations. The family type and block hyperparameters evolve independently.
"""
from __future__ import annotations

from pydantic import BaseModel
from enum import Enum


class HybridFamily(str, Enum):
    MLP = "mlp"
    SPARSE_MLP = "sparse_mlp"
    CONV2D = "conv2d"
    ATTENTION = "attention"


class HybridNodeGene(BaseModel, frozen=True):
    """One macro block in the hybrid DAG."""
    innovation_number: int
    order: float
    family: HybridFamily = HybridFamily.MLP
    width: int = 64                    # Output width of this block
    internal_layers: int = 1           # Depth within the block (1-3)
    activation: str = "relu"           # relu | gelu | silu | tanh
    dropout: float = 0.0
    norm_type: str = "layer"           # none | layer | batch
    # Family-specific
    kernel_size: int = 3               # for conv2d
    num_heads: int = 2                 # for attention
    enabled: bool = True


class HybridConnectionGene(BaseModel, frozen=True):
    """Edge in hybrid DAG."""
    innovation_number: int
    source_innovation: int
    target_innovation: int
    enabled: bool = True


INPUT_INNOVATION = 0
OUTPUT_INNOVATION = -1


class HybridGenome(BaseModel):
    """A NEAT-style DAG where each node is a family-typed macro block."""
    nodes: list[HybridNodeGene]
    connections: list[HybridConnectionGene]
    fitness: float | None = None
    learning_rate: float = 0.01
    batch_size: int = 32

    @property
    def enabled_nodes(self) -> list[HybridNodeGene]:
        return [n for n in self.nodes if n.enabled]

    @property
    def enabled_connections(self) -> list[HybridConnectionGene]:
        return [c for c in self.connections if c.enabled]

    @classmethod
    def create_seed(cls, innovation_counter: int, width: int = 64) -> tuple["HybridGenome", int]:
        """Create a minimal seed genome: input -> one MLP block -> output."""
        node = HybridNodeGene(
            innovation_number=innovation_counter + 1,
            order=0.5,
            family=HybridFamily.MLP,
            width=width,
        )
        conn_in = HybridConnectionGene(
            innovation_number=innovation_counter + 2,
            source_innovation=INPUT_INNOVATION,
            target_innovation=node.innovation_number,
        )
        conn_out = HybridConnectionGene(
            innovation_number=innovation_counter + 3,
            source_innovation=node.innovation_number,
            target_innovation=OUTPUT_INNOVATION,
        )
        genome = cls(
            nodes=[node],
            connections=[conn_in, conn_out],
        )
        return genome, innovation_counter + 3
