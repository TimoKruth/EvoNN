"""Frozen gene types for topology-based genome representation."""

from enum import Enum

from pydantic import BaseModel


class Activation(str, Enum):
    RELU = "relu"
    SIGMOID = "sigmoid"
    TANH = "tanh"
    GELU = "gelu"
    SILU = "silu"


class WeightBits(int, Enum):
    TERNARY = 2
    INT4 = 4
    INT8 = 8
    FP16 = 16


class ActivationBits(int, Enum):
    INT4 = 4
    INT8 = 8
    FP16 = 16


class OperatorType(str, Enum):
    DENSE = "dense"
    SPARSE_DENSE = "sparse_dense"
    RESIDUAL = "residual"
    ATTENTION_LITE = "attention_lite"
    SPATIAL = "spatial"
    TRANSFORMER_LITE = "transformer_lite"


class LayerGene(BaseModel, frozen=True):
    innovation: int
    width: int
    activation: Activation
    weight_bits: WeightBits = WeightBits.FP16
    activation_bits: ActivationBits = ActivationBits.FP16
    sparsity: float = 0.0
    order: float
    enabled: bool = True
    operator: OperatorType = OperatorType.DENSE
    num_heads: int = 1


class ConvLayerGene(BaseModel, frozen=True):
    innovation: int
    channels: int
    kernel_size: int
    stride: int = 1
    activation: Activation = Activation.RELU
    order: float = 0.0
    enabled: bool = True


class ConnectionGene(BaseModel, frozen=True):
    innovation: int
    source: int
    target: int
    enabled: bool = True


class ExpertGene(BaseModel, frozen=True):
    expert_id: int
    innovation: int
    width: int
    activation: Activation
    weight_bits: WeightBits = WeightBits.FP16
    activation_bits: ActivationBits = ActivationBits.FP16
    sparsity: float = 0.0
    order: float
    enabled: bool = True


class ExpertConnectionGene(BaseModel, frozen=True):
    expert_id: int
    innovation: int
    source: int
    target: int
    enabled: bool = True


class GateConfig(BaseModel, frozen=True):
    num_experts: int = 4
    top_k: int = 2
    gate_hidden_dim: int = 32
    load_balance_weight: float = 0.01
