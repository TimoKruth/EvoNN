"""JSON serialization for genomes."""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any

from topograph.genome.genes import (
    ConnectionGene,
    ConvLayerGene,
    ExpertConnectionGene,
    ExpertGene,
    GateConfig,
    LayerGene,
)
from topograph.genome.genome import Genome


def genome_to_dict(genome: Genome) -> dict[str, Any]:
    d: dict[str, Any] = {
        "layers": [g.model_dump(mode="json") for g in genome.layers],
        "connections": [g.model_dump(mode="json") for g in genome.connections],
    }
    if genome.conv_layers:
        d["conv_layers"] = [g.model_dump(mode="json") for g in genome.conv_layers]
    if genome.experts:
        d["experts"] = [g.model_dump(mode="json") for g in genome.experts]
    if genome.expert_connections:
        d["expert_connections"] = [g.model_dump(mode="json") for g in genome.expert_connections]
    if genome.gate_config is not None:
        d["gate_config"] = genome.gate_config.model_dump(mode="json")
    if genome.fitness is not None:
        d["fitness"] = genome.fitness
    if genome.param_count:
        d["param_count"] = genome.param_count
    if genome.model_bytes:
        d["model_bytes"] = genome.model_bytes
    if genome.learning_rate is not None:
        d["learning_rate"] = genome.learning_rate
    if genome.batch_size is not None:
        d["batch_size"] = genome.batch_size
    return d


def dict_to_genome(d: dict[str, Any]) -> Genome:
    genome = Genome(
        layers=[LayerGene.model_validate(x) for x in d["layers"]],
        connections=[ConnectionGene.model_validate(x) for x in d["connections"]],
    )
    genome.conv_layers = [ConvLayerGene.model_validate(x) for x in d.get("conv_layers", [])]
    genome.experts = [ExpertGene.model_validate(x) for x in d.get("experts", [])]
    genome.expert_connections = [
        ExpertConnectionGene.model_validate(x) for x in d.get("expert_connections", [])
    ]
    if gc := d.get("gate_config"):
        genome.gate_config = GateConfig.model_validate(gc)
    genome.fitness = d.get("fitness")
    genome.param_count = d.get("param_count", 0)
    genome.model_bytes = d.get("model_bytes", 0)
    genome.learning_rate = d.get("learning_rate")
    genome.batch_size = d.get("batch_size")
    return genome


def save_genomes(genomes: list[Genome], path: Path | str) -> None:
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    with open(path, "w") as f:
        json.dump([genome_to_dict(g) for g in genomes], f, indent=2)


def load_genomes(path: Path | str) -> list[Genome]:
    with open(path) as f:
        return [dict_to_genome(d) for d in json.load(f)]
