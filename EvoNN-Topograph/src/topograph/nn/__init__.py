"""Neural network modules: compilation, layers, MoE, and training."""

from topograph.nn.compiler import EvolvedModel, compile_genome, estimate_model_bytes
from topograph.nn.layers import BitLinear, QuantizedLinear, hadamard_smooth
from topograph.nn.moe import ExpertNetwork, GateNetwork, MixtureOfExperts
from topograph.nn.train import (
    compute_percentile_fitness,
    cosine_lr,
    effective_model_bytes,
    extract_weights,
    load_weight_snapshot,
    train_model,
)

__all__ = [
    "BitLinear",
    "EvolvedModel",
    "ExpertNetwork",
    "GateNetwork",
    "MixtureOfExperts",
    "QuantizedLinear",
    "compile_genome",
    "compute_percentile_fitness",
    "cosine_lr",
    "effective_model_bytes",
    "estimate_model_bytes",
    "extract_weights",
    "hadamard_smooth",
    "load_weight_snapshot",
    "train_model",
]
