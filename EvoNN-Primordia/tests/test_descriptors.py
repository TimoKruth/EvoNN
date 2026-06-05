from __future__ import annotations

from evonn_primordia.export.descriptors import build_descriptor_coverage, motif_descriptor


def test_motif_descriptor_keys_capture_primitive_shape() -> None:
    record = {
        "primitive_family": "sparse_mlp",
        "benchmark_group": "tabular",
        "parameter_count": 4096,
        "generation": 1,
        "mutation_operator": "sparsity",
        "genome_payload": {
            "family": "sparse_mlp",
            "hidden_layers": [96, 96, 96],
            "activation": "gelu",
            "norm_type": "layer",
            "residual": True,
            "activation_sparsity": 0.25,
        },
    }

    descriptor = motif_descriptor(record)

    assert descriptor["family"] == "sparse_mlp"
    assert descriptor["depth_bucket"] == "shallow"
    assert descriptor["width_bucket"] == "mid"
    assert descriptor["sparsity_bucket"] == "light"
    assert "family=sparse_mlp" in descriptor["descriptor_key"]


def test_descriptor_coverage_tracks_winning_cells_and_family_scope() -> None:
    summary = {"run_id": "descriptor-run", "run_name": "descriptor-run"}
    trials = [
        {
            "benchmark_name": "iris",
            "benchmark_group": "tabular",
            "primitive_family": "sparse_mlp",
            "status": "ok",
            "quality": 0.8,
            "search_score": 0.82,
            "parameter_count": 4096,
            "genome_id": "g-1",
            "architecture_summary": "sparse_mlp[96x96x96]",
            "genome_payload": {
                "family": "sparse_mlp",
                "hidden_layers": [96, 96, 96],
                "activation": "gelu",
                "norm_type": "layer",
                "activation_sparsity": 0.25,
            },
        },
        {
            "benchmark_name": "tiny_lm",
            "benchmark_group": "language_modeling",
            "primitive_family": "attention",
            "status": "ok",
            "quality": -3.2,
            "search_score": -3.1,
            "parameter_count": 12000,
            "genome_id": "g-2",
            "architecture_summary": "attention[128x128x128]",
            "genome_payload": {
                "family": "attention",
                "hidden_layers": [128, 128, 128],
                "activation": "gelu",
                "norm_type": "layer",
                "embedding_dim": 128,
                "num_heads": 4,
            },
        },
    ]
    best = [trials[0]]

    coverage = build_descriptor_coverage(summary=summary, trial_records=trials, best_results=best)

    assert coverage["descriptor_cell_count"] == 2
    assert coverage["winning_descriptor_cell_count"] == 1
    assert coverage["family_descriptor_coverage"]["sparse_mlp"]["winning_descriptor_cell_count"] == 1
    assert coverage["family_descriptor_coverage"]["attention"]["transfer_scope"] == "single_group"
    assert coverage["benchmark_group_coverage"]["tabular"]["families"] == ["sparse_mlp"]
