import numpy as np

from stratograph.benchmarks import get_benchmark
from stratograph.config import load_config
from stratograph.genome import HierarchicalGenome, dict_to_genome, genome_to_dict
from stratograph.genome.models import PrimitiveKind
from stratograph.runtime import compile_genome


def test_config_loads(repo_root) -> None:
    config = load_config(repo_root / "configs" / "working_33_plus_5_lm_smoke.yaml")
    assert config.seed == 42
    assert len(config.benchmark_pool.benchmarks) == 38


def test_seed_genome_roundtrip() -> None:
    spec = get_benchmark("moons")
    genome = HierarchicalGenome.create_seed(
        benchmark_name=spec.name,
        task=spec.task,
        input_dim=spec.model_input_dim,
        output_dim=spec.model_output_dim,
        seed=42,
    )
    clone = dict_to_genome(genome_to_dict(genome))
    assert clone.genome_id == genome.genome_id
    assert clone.macro_depth >= 1
    assert clone.average_cell_depth >= 1.0
    assert len(clone.macro_edges) > len(clone.macro_nodes)


def test_compile_classification_shape() -> None:
    spec = get_benchmark("digits")
    genome = HierarchicalGenome.create_seed(
        benchmark_name=spec.name,
        task=spec.task,
        input_dim=spec.model_input_dim,
        output_dim=spec.model_output_dim,
        seed=42,
    )
    compiled = compile_genome(genome)
    output = compiled.forward(np.ones((3, spec.model_input_dim), dtype=np.float32))
    assert output.shape == (3, spec.model_output_dim)
    assert compiled.parameter_count() > 0
    assert "branch_factor=" in compiled.architecture_summary()


def test_compile_lm_shape() -> None:
    spec = get_benchmark("tiny_lm_synthetic")
    genome = HierarchicalGenome.create_seed(
        benchmark_name=spec.name,
        task=spec.task,
        input_dim=spec.model_input_dim,
        output_dim=spec.model_output_dim,
        seed=42,
    )
    compiled = compile_genome(genome)
    tokens = np.arange(2 * spec.model_input_dim, dtype=np.int32).reshape(2, spec.model_input_dim) % spec.model_output_dim
    output = compiled.forward(tokens)
    assert output.shape == (2, spec.model_input_dim, spec.model_output_dim)


def test_compile_branching_macro_graph_changes_encoding() -> None:
    spec = get_benchmark("moons")
    genome = HierarchicalGenome.create_seed(
        benchmark_name=spec.name,
        task=spec.task,
        input_dim=spec.model_input_dim,
        output_dim=spec.model_output_dim,
        seed=42,
    )
    chain_genome = genome.model_copy(
        update={
            "macro_edges": [
                edge
                for edge in genome.macro_edges
                if (edge.source, edge.target) not in {("macro_0", "macro_2"), ("macro_1", "output")}
            ]
        },
        deep=True,
    )
    sample = np.ones((4, spec.model_input_dim), dtype=np.float32)
    branched = compile_genome(genome).encode(sample)
    chain = compile_genome(chain_genome).encode(sample)
    assert not np.allclose(branched, chain)


def test_primitive_kind_changes_runtime_output() -> None:
    spec = get_benchmark("moons")
    genome = HierarchicalGenome.create_seed(
        benchmark_name=spec.name,
        task=spec.task,
        input_dim=spec.model_input_dim,
        output_dim=spec.model_output_dim,
        seed=42,
    )
    cell_id = genome.macro_nodes[0].cell_id
    cell = genome.cell_library[cell_id]
    linear_nodes = [node.model_copy(update={"kind": PrimitiveKind.LINEAR}) for node in cell.nodes]
    gate_nodes = [node.model_copy(update={"kind": PrimitiveKind.GATE}) for node in cell.nodes]
    linear_genome = genome.model_copy(
        update={"cell_library": {cell_id: cell.model_copy(update={"nodes": linear_nodes})}},
        deep=True,
    )
    gate_genome = genome.model_copy(
        update={"cell_library": {cell_id: cell.model_copy(update={"nodes": gate_nodes})}},
        deep=True,
    )
    sample = np.ones((4, spec.model_input_dim), dtype=np.float32)
    linear_output = compile_genome(linear_genome).forward(sample)
    gate_output = compile_genome(gate_genome).forward(sample)
    assert not np.allclose(linear_output, gate_output)
