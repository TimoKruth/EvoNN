"""Genome: a variable-length collection of layer, connection, and expert genes."""

from __future__ import annotations

import random

from topograph.genome.genes import (
    Activation,
    ActivationBits,
    ConnectionGene,
    ConvLayerGene,
    ExpertConnectionGene,
    ExpertGene,
    GateConfig,
    LayerGene,
    WeightBits,
)

INPUT_INNOVATION = 0
OUTPUT_INNOVATION = -1


class InnovationCounter:
    """Thread-unsafe counter for assigning unique innovation numbers."""

    def __init__(self, start: int = 1) -> None:
        self._n = start

    @property
    def value(self) -> int:
        return self._n

    @value.setter
    def value(self, v: int) -> None:
        self._n = v

    def next(self) -> int:
        val = self._n
        self._n += 1
        return val


class Genome:
    def __init__(
        self,
        layers: list[LayerGene],
        connections: list[ConnectionGene],
    ) -> None:
        self.layers = list(layers)
        self.connections = list(connections)
        self.conv_layers: list[ConvLayerGene] = []
        self.experts: list[ExpertGene] = []
        self.expert_connections: list[ExpertConnectionGene] = []
        self.gate_config: GateConfig | None = None
        self.fitness: float | None = None
        self.param_count: int = 0
        self.model_bytes: int = 0
        self.learning_rate: float | None = None
        self.batch_size: int | None = None

    @property
    def enabled_layers(self) -> list[LayerGene]:
        return [g for g in self.layers if g.enabled]

    @property
    def enabled_connections(self) -> list[ConnectionGene]:
        return [g for g in self.connections if g.enabled]

    @staticmethod
    def create_seed(
        innovation_counter: InnovationCounter,
        rng: random.Random,
        num_layers: int | None = None,
        mixed_precision: bool = False,
    ) -> Genome:
        if num_layers is None:
            num_layers = rng.randint(5, 30)

        activations = list(Activation)
        rng.shuffle(activations)
        chosen = [activations[i % len(activations)] for i in range(num_layers)]

        layers: list[LayerGene] = []
        for i in range(num_layers):
            extra = (
                {
                    "weight_bits": rng.choice(list(WeightBits)),
                    "activation_bits": rng.choice(list(ActivationBits)),
                    "sparsity": round(rng.uniform(0, 0.3), 2),
                }
                if mixed_precision
                else {}
            )
            layers.append(
                LayerGene(
                    innovation=innovation_counter.next(),
                    width=rng.randint(16, 256),
                    activation=chosen[i],
                    order=float(i + 1),
                    **extra,
                )
            )

        connections: list[ConnectionGene] = []
        # input -> first layer
        connections.append(
            ConnectionGene(
                innovation=innovation_counter.next(),
                source=INPUT_INNOVATION,
                target=layers[0].innovation,
            )
        )
        # layer chain
        for i in range(num_layers - 1):
            connections.append(
                ConnectionGene(
                    innovation=innovation_counter.next(),
                    source=layers[i].innovation,
                    target=layers[i + 1].innovation,
                )
            )
        # last layer -> output
        connections.append(
            ConnectionGene(
                innovation=innovation_counter.next(),
                source=layers[-1].innovation,
                target=OUTPUT_INNOVATION,
            )
        )

        return Genome(layers=layers, connections=connections)
