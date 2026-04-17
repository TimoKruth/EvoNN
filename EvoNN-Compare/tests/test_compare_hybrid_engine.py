from __future__ import annotations

import importlib
import sys
import types


def _load_engine_module():
    mlx = types.ModuleType("mlx")
    mlx_core = types.ModuleType("mlx.core")
    mlx_core.array = object
    mlx_nn = types.ModuleType("mlx.nn")

    class Module:
        pass

    mlx_nn.Module = Module
    mlx.core = mlx_core
    mlx.nn = mlx_nn

    topograph = types.ModuleType("topograph")
    topograph_nn = types.ModuleType("topograph.nn")
    topograph_nn_train = types.ModuleType("topograph.nn.train")
    topograph_nn_train.train_and_evaluate = lambda *args, **kwargs: None

    compiler = types.ModuleType("evonn_compare.hybrid.compiler")
    compiler.compile_hybrid = lambda *args, **kwargs: None

    sys.modules.setdefault("mlx", mlx)
    sys.modules.setdefault("mlx.core", mlx_core)
    sys.modules.setdefault("mlx.nn", mlx_nn)
    sys.modules.setdefault("topograph", topograph)
    sys.modules.setdefault("topograph.nn", topograph_nn)
    sys.modules.setdefault("topograph.nn.train", topograph_nn_train)
    sys.modules.setdefault("evonn_compare.hybrid.compiler", compiler)

    return importlib.import_module("evonn_compare.hybrid.engine")


def test_history_metric_value_preserves_zero_metric() -> None:
    engine = _load_engine_module()
    record = engine.HybridEvaluation(
        benchmark_id="zero_loss_regression",
        task="regression",
        genome_index=0,
        loss=0.0,
        metric_name="mse",
        metric_direction="min",
        metric_value=0.0,
        train_seconds=0.1,
        architecture_summary="mlp:32",
        genome_id="g0",
        status="ok",
    )

    assert engine._history_metric_value(record) == 0.0
