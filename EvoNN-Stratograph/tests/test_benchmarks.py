from stratograph.benchmarks import get_benchmark, list_benchmarks, load_parity_pack


def test_builtin_benchmark_count() -> None:
    names = [spec.name for spec in list_benchmarks()]
    assert len(names) == 38
    assert "tiny_lm_synthetic" in names
    assert "wikitext2_lm_smoke" in names


def test_synthetic_lm_loads() -> None:
    spec = get_benchmark("tiny_lm_synthetic")
    x_train, y_train, x_val, y_val = spec.load_data(seed=7)
    assert x_train.ndim == 2
    assert y_train.ndim == 2
    assert x_train.shape[1] == 128
    assert x_val.shape[1] == 128
    assert y_train.shape == x_train.shape
    assert y_val.shape == x_val.shape


def test_simple_pack_parses(repo_root) -> None:
    pack = load_parity_pack(repo_root / "configs" / "working_33_plus_5_lm_smoke.yaml")
    assert pack.name == "working_33_plus_5_lm_smoke"
    assert len(pack.benchmarks) == 38
    assert pack.benchmarks[0].metric_name == "accuracy"
